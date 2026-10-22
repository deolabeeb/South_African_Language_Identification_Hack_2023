"""Auditable prepare -> validation selection -> one-use holdout evaluation."""
import csv
import hashlib
import json
import platform
import sys
import time
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import scipy
import sklearn
import threadpoolctl
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from threadpoolctl import threadpool_limits

from .data import LABELS, load_dataset, make_split, normalize, sha256
from .models import CANDIDATES, build_model


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def source_hash():
    return digest({p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))})


def environment():
    return {"python": platform.python_version(), "platform": platform.platform(),
            "scikit_learn": sklearn.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
            "joblib": joblib.__version__, "threadpoolctl": threadpoolctl.__version__,
            "thread_limit": 1}


def prepare(data, run):
    run, data = Path(run), Path(data).resolve()
    if run.exists():
        raise ValueError("Run directory already exists; preserve it instead of overwriting evidence")
    rows, audit = load_dataset(data)
    split = make_split(rows)
    protocol = {"version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                "data_path": str(data), "data_sha256": sha256(data), "source_sha256": source_hash(),
                "environment": environment(), "seed": 42, "candidates": CANDIDATES,
                "selection_metric": "validation macro F1; lexical candidate-name tie-break",
                "split": split, "split_sha256": digest(split), "audit": audit,
                "normalized_ids_sha256": digest([r["id"] for r in rows]),
                "policy": "One selection; holdout consumed at most once after frozen selection; no refit"}
    run.mkdir(parents=True)
    write_json(run / "protocol.json", protocol)
    summary = dict(audit)
    summary["partition_counts"] = {key: dict(sorted(Counter(rows[i]["label"] for i in indices).items()))
                                   for key, indices in split.items()}
    summary["partition_sizes"] = {key: len(value) for key, value in split.items()}
    summary["normalized_text_overlap"] = 0
    summary["dataset_sha256"] = protocol["data_sha256"]
    write_json(run / "audit.json", summary)
    return summary


def load_run(run):
    run = Path(run)
    protocol = read_json(run / "protocol.json")
    if protocol["source_sha256"] != source_hash():
        raise ValueError("Source changed after protocol freeze")
    if protocol["environment"] != environment():
        raise ValueError("Environment changed after protocol freeze")
    if protocol["candidates"] != CANDIDATES or digest(protocol["split"]) != protocol["split_sha256"]:
        raise ValueError("Candidate configuration or split changed after protocol freeze")
    if sha256(protocol["data_path"]) != protocol["data_sha256"]:
        raise ValueError("Dataset changed after protocol freeze")
    rows, _ = load_dataset(protocol["data_path"])
    if digest([r["id"] for r in rows]) != protocol["normalized_ids_sha256"]:
        raise ValueError("Normalized dataset changed")
    # Reconstruct the predeclared split, rather than trusting edited indices.
    if make_split(rows, protocol["seed"]) != protocol["split"]:
        raise ValueError("Stored partition differs from deterministic split")
    return run, protocol, rows


def metrics(labels, predictions):
    report = classification_report(labels, predictions, labels=LABELS, output_dict=True, zero_division=0)
    per_language = {label: report[label] for label in LABELS}
    return {"rows": len(labels), "accuracy": accuracy_score(labels, predictions),
            "macro_f1": f1_score(labels, predictions, labels=LABELS, average="macro", zero_division=0),
            "weighted_f1": f1_score(labels, predictions, labels=LABELS, average="weighted", zero_division=0),
            "worst_language_f1": min(per_language[label]["f1-score"] for label in LABELS),
            "per_language": per_language, "confusion_labels": list(LABELS),
            "confusion_matrix": confusion_matrix(labels, predictions, labels=LABELS).tolist()}


def select(run):
    run, protocol, rows = load_run(run)
    # Exclusive markers retain failed attempts and prevent silent retries/retuning.
    with (run / "selection-started.json").open("x", encoding="utf-8") as marker:
        json.dump({"protocol_sha256": sha256(run / "protocol.json")}, marker)
    train, valid = (protocol["split"][key] for key in ["train", "validation"])
    x_train, y_train = [rows[i]["text"] for i in train], [rows[i]["label"] for i in train]
    x_valid, y_valid = [rows[i]["text"] for i in valid], [rows[i]["label"] for i in valid]
    results, best_model, best_name = {}, None, None
    with threadpool_limits(limits=1):
        for name in CANDIDATES:
            start = time.perf_counter()
            model = build_model(name)
            with warnings.catch_warnings(record=True) as recorded:
                warnings.simplefilter("always")
                model.fit(x_train, y_train)
            if any(issubclass(w.category, ConvergenceWarning) for w in recorded):
                raise RuntimeError("Candidate did not converge; preserve this failed selection receipt")
            fit_seconds = time.perf_counter() - start
            start = time.perf_counter()
            predictions = model.predict(x_valid)
            result = metrics(y_valid, predictions)
            result.update(fit_seconds=fit_seconds, predict_seconds=time.perf_counter()-start,
                          features=len(model.named_steps["tfidf"].vocabulary_),
                          warnings=[w.category.__name__ for w in recorded])
            results[name] = result
            write_json(run / "validation-progress.json", results)
            if (best_name is None or result["macro_f1"] > results[best_name]["macro_f1"] or
                (result["macro_f1"] == results[best_name]["macro_f1"] and name < best_name)):
                best_name, best_model = name, model
    model_path = run / "selected-model.joblib"
    joblib.dump(best_model, model_path, compress=3)
    selection = {"selected": best_name, "candidates": results,
                 "protocol_sha256": sha256(run / "protocol.json"),
                 "model_sha256": sha256(model_path), "source_sha256": source_hash(),
                 "fit_partition": "train only; no train+validation refit", "holdout_used_for_selection": False}
    write_json(run / "selection.json", selection)
    return selection


def load_selected(run):
    run = Path(run)
    protocol, selection = read_json(run / "protocol.json"), read_json(run / "selection.json")
    if source_hash() != protocol["source_sha256"] or selection["source_sha256"] != source_hash():
        raise ValueError("Source changed after model freeze")
    if protocol["environment"] != environment():
        raise ValueError("Model environment differs from frozen environment")
    if selection["protocol_sha256"] != sha256(run / "protocol.json"):
        raise ValueError("Protocol was edited after model selection")
    if sha256(run / "selected-model.joblib") != selection["model_sha256"]:
        raise ValueError("Selected model checksum mismatch")
    # Only load the artifact created by this local run; never open legacy/downloaded pickle files.
    return joblib.load(run / "selected-model.joblib"), selection


def evaluate(run):
    run, protocol, rows = load_run(run)
    model, selection = load_selected(run)
    with (run / "holdout-opened.json").open("x", encoding="utf-8") as marker:
        json.dump({"selection_sha256": sha256(run / "selection.json"),
                   "opened_utc": datetime.now(timezone.utc).isoformat()}, marker)
    ids = protocol["split"]["holdout"]
    texts, labels = [rows[i]["text"] for i in ids], [rows[i]["label"] for i in ids]
    start = time.perf_counter()
    with threadpool_limits(limits=1):
        predictions = model.predict(texts)
    predict_seconds = time.perf_counter() - start
    report = metrics(labels, predictions)
    report.update(selected=selection["selected"], predict_seconds=predict_seconds,
                  selection_sha256=sha256(run / "selection.json"),
                  model_sha256=selection["model_sha256"], dataset_sha256=protocol["data_sha256"],
                  split_sha256=protocol["split_sha256"], environment=protocol["environment"],
                  source_sha256=source_hash(), metric_scope="local deduplicated internal holdout; not Kaggle score")
    # Predeclared descriptive slices; no post-evaluation model selection is allowed.
    report["length_slices"] = {}
    for name, indices in [("at_most_80_chars", [i for i,t in enumerate(texts) if len(t)<=80]),
                           ("over_80_chars", [i for i,t in enumerate(texts) if len(t)>80])]:
        report["length_slices"][name] = (metrics([labels[i] for i in indices], [predictions[i] for i in indices])
                                            if indices else {"rows": 0, "status": "no examples"})
    write_json(run / "holdout.json", report)
    with (run / "per-language.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["language", "precision", "recall", "f1", "support"])
        for label, scores in report["per_language"].items():
            writer.writerow([label, scores["precision"], scores["recall"], scores["f1-score"], scores["support"]])
    return report


def predict(run, text):
    text = normalize(text)
    if not text or len(text) > 10000:
        raise ValueError("Provide between 1 and 10000 normalized characters")
    model, _ = load_selected(run)
    with threadpool_limits(limits=1):
        label = str(model.predict([text])[0])
    return {"language": label, "scope": "closed set of eleven benchmark labels; no calibrated confidence"}
