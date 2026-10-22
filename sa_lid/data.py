"""Language-neutral normalization and duplicate-controlled data partitions."""
import csv
import hashlib
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from sklearn.model_selection import train_test_split

LABELS = ("afr", "eng", "nbl", "nso", "sot", "ssw", "tsn", "tso", "ven", "xho", "zul")


def normalize(text):
    """Keep short words, punctuation and diacritics; never use a target label."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_dataset(path):
    groups = defaultdict(list)
    raw_counts = Counter()
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["lang_id", "text"]:
            raise ValueError("Expected CSV columns exactly: lang_id,text")
        for line, row in enumerate(reader, 2):
            if None in row or row["text"] is None:
                raise ValueError(f"Malformed CSV row at line {line}")
            label, text = row["lang_id"], normalize(row["text"])
            if label not in LABELS or not text:
                raise ValueError(f"Invalid label or empty text at line {line}")
            groups[text].append(label)
            raw_counts[label] += 1
    if set(raw_counts) != set(LABELS):
        raise ValueError("All eleven benchmark language labels are required")
    rows, conflicting_rows, conflicting_groups, duplicates = [], 0, 0, 0
    for text, labels in groups.items():
        if len(set(labels)) > 1:
            conflicting_rows += len(labels)
            conflicting_groups += 1
            continue
        duplicates += len(labels) - 1
        rows.append({"id": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                     "text": text, "label": labels[0]})
    rows.sort(key=lambda row: row["id"])
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Unexpected normalized-text hash collision")
    counts = Counter(r["label"] for r in rows)
    if any(counts[label] < 8 for label in LABELS):
        raise ValueError("At least eight distinct, unambiguous texts per language are required")
    audit = {"raw_rows": sum(raw_counts.values()), "raw_label_counts": dict(sorted(raw_counts.items())),
             "retained_unique_texts": len(rows), "retained_label_counts": dict(sorted(counts.items())),
             "duplicate_rows_collapsed": duplicates, "conflicting_label_groups_excluded": conflicting_groups,
             "conflicting_rows_excluded": conflicting_rows,
             "normalization": "NFKC, casefold, collapse Unicode whitespace; preserve punctuation and diacritics",
             "near_duplicates_checked": False}
    assert audit["raw_rows"] == len(rows) + duplicates + conflicting_rows
    return rows, audit


def make_split(rows, seed=42):
    """Each normalized text occurs once, so stratified splitting is group-disjoint."""
    indices = list(range(len(rows)))
    development, holdout = train_test_split(indices, test_size=0.2, random_state=seed,
                                           stratify=[r["label"] for r in rows])
    train, validation = train_test_split(development, test_size=0.25, random_state=seed,
                                         stratify=[rows[i]["label"] for i in development])
    split = {"train": sorted(train), "validation": sorted(validation), "holdout": sorted(holdout)}
    sets = [set(value) for value in split.values()]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
        raise ValueError("Split overlap")
    if set.union(*sets) != set(indices):
        raise ValueError("Split coverage mismatch")
    for value in split.values():
        if set(rows[i]["label"] for i in value) != set(LABELS):
            raise ValueError("Every partition must contain all eleven labels")
    return split
