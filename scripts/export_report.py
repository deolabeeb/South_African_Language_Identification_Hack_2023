"""Export aggregate evidence only, without data text, paths or model binaries."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run = Path(args.run)
    protocol, audit = read(run / "protocol.json"), read(run / "audit.json")
    selected, holdout = read(run / "selection.json"), read(run / "holdout.json")
    assert hashlib.sha256((run / "selection.json").read_bytes()).hexdigest() == holdout["selection_sha256"]
    assert hashlib.sha256((run / "protocol.json").read_bytes()).hexdigest() == selected["protocol_sha256"]
    assert holdout["source_sha256"] == selected["source_sha256"] == protocol["source_sha256"]
    assert not selected["holdout_used_for_selection"]
    out = root / "reports"
    out.mkdir(exist_ok=True)
    evidence = {"audit": audit, "validation": selected["candidates"], "holdout": holdout,
                "selected": selected["selected"], "candidate_configurations": protocol["candidates"],
                "protocol_sha256": selected["protocol_sha256"], "source_sha256": protocol["source_sha256"],
                "fit_partition": selected["fit_partition"], "holdout_used_for_selection": False,
                "model_sha256": selected["model_sha256"], "seed": protocol["seed"]}
    (out / "metrics.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    # This CSV contains one aggregate row per language, not prediction rows.
    (out / "per-language.csv").write_bytes((run / "per-language.csv").read_bytes())
    lines = ["# Local experiment results", "", "Run date: 22 October 2026.", "",
             "These are freshly executed local results on a reserved internal holdout, not Kaggle scores. "
             "Old notebook scores use different preparation/splits and are not a comparable baseline.", "",
             "## Data audit", "",
             f"Original labelled rows: {audit['raw_rows']:,}. Retained distinct normalized texts: {audit['retained_unique_texts']:,}. "
             f"Duplicate rows collapsed: {audit['duplicate_rows_collapsed']:,}. "
             f"Conflicting-label groups excluded: {audit['conflicting_label_groups_excluded']:,} "
             f"({audit['conflicting_rows_excluded']:,} rows).", "",
             f"Seed 42; sizes: {audit['partition_sizes']}. Exact normalized-text overlap: zero. "
             "Near duplicates and original document sources were not independently identified.", "",
             "## Validation comparison", "", "Selection used validation macro F1 only; candidates were fixed beforehand.", "",
             "| Candidate | Macro F1 | Accuracy | Fit seconds | Features |",
             "|---|---:|---:|---:|---:|"]
    for name, scores in selected["candidates"].items():
        lines.append(f"| {name} | {scores['macro_f1']:.6f} | {scores['accuracy']:.6f} | {scores['fit_seconds']:.3f} | {scores['features']:,} |")
    lines += ["", f"Frozen choice: **{selected['selected']}**. Fitted on training rows only; no refit or post-holdout tuning.", "",
              "## One-use holdout", "",
              f"Rows: {holdout['rows']:,}; macro F1: **{holdout['macro_f1']:.6f}**; "
              f"weighted F1: {holdout['weighted_f1']:.6f}; accuracy: {holdout['accuracy']:.6f}; "
              f"lowest per-language F1: {holdout['worst_language_f1']:.6f}.", "",
              "| Language code | Precision | Recall | F1 | Support |", "|---|---:|---:|---:|---:|"]
    for language, scores in holdout["per_language"].items():
        lines.append(f"| {language} | {scores['precision']:.6f} | {scores['recall']:.6f} | {scores['f1-score']:.6f} | {int(scores['support'])} |")
    lines += ["", "Language-code differences are descriptive diagnostics, not proof of fairness. "
              "The confusion matrix and predeclared character-length slices are included in metrics.json.", "",
              "## Reproducibility", "", "See the README for the exact prepare, select, evaluate and predict commands.", "",
              f"- Dataset SHA-256: `{protocol['data_sha256']}`",
              f"- Package source SHA-256: `{protocol['source_sha256']}`",
              f"- Frozen protocol SHA-256: `{selected['protocol_sha256']}`",
              f"- Split SHA-256: `{protocol['split_sha256']}`",
              f"- Environment: `{json.dumps(protocol['environment'], sort_keys=True)}`", "",
              "Runtime is observed on this local host, not a latency guarantee. No fresh-machine reproduction, "
              "external corpus, production traffic, out-of-distribution or fairness validation was performed. "
              "All candidates used the same frozen training/validation partitions. The final holdout informed no model changes.", "",
              "## Interpretation", "",
              "The result measures this deduplicated text benchmark. Shared corpus/document sources, near duplicates "
              "and a historical corpus previously explored in coursework limit generalization claims. "
              "Short text, unseen domains, unknown languages and code-switching require separate future evaluation; "
              "the current model always chooses among the eleven known labels.", ""]
    (out / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"selected":selected["selected"], "holdout_macro_f1":holdout["macro_f1"],
                      "reports":"reports/RESULTS.md and aggregate JSON/CSV"}, indent=2))


if __name__ == "__main__":
    main()
