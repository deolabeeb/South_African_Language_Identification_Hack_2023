import csv
import json
import tempfile
import unittest
from pathlib import Path

from sa_lid.data import LABELS, load_dataset, make_split, normalize
from sa_lid.models import build_model
from sa_lid.workflow import evaluate, load_selected, predict, prepare, select


def fixture(path, extra=()):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["lang_id", "text"])
        for label in LABELS:
            for index in range(15):
                writer.writerow([label, f"{label}language {label}words example{index}"])
        writer.writerows(extra)


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.data = Path(self.temp.name) / "train.csv"

    def test_normalization_retains_language_cues(self):
        self.assertEqual(normalize("  I  \t AM naïve! "), "i am naïve!")
        self.assertEqual(normalize("cafe\u0301"), normalize("café"))
        self.assertIn("ḓ", normalize("ḓivha"))

    def test_duplicates_collapsed_and_conflicts_quarantined(self):
        fixture(self.data, [("afr", "  AFRLANGUAGE AFRWORDS EXAMPLE0  "),
                            ("eng", "ambiguous"), ("zul", "ambiguous")])
        rows, audit = load_dataset(self.data)
        self.assertEqual(audit["duplicate_rows_collapsed"], 1)
        self.assertEqual(audit["conflicting_rows_excluded"], 2)
        self.assertEqual(len(rows), 165)
        self.assertFalse(any(r["text"] == "ambiguous" for r in rows))

    def test_partitions_disjoint_complete_and_order_stable(self):
        fixture(self.data)
        rows, _ = load_dataset(self.data)
        split = make_split(rows)
        self.assertEqual(split, make_split(rows))
        sets = [set(rows[i]["id"] for i in indices) for indices in split.values()]
        self.assertEqual(sum(map(len, sets)), len(rows))
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        original = self.data.read_text(encoding="utf-8").splitlines()
        self.data.write_text("\n".join([original[0], *reversed(original[1:])]) + "\n", encoding="utf-8")
        reordered, _ = load_dataset(self.data)
        self.assertEqual(rows, reordered)
        for indices in split.values():
            self.assertEqual({rows[i]["label"] for i in indices}, set(LABELS))

    def test_invalid_schema_rejected(self):
        self.data.write_text("text,lang_id\nhello,eng\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "columns"):
            load_dataset(self.data)

    def test_invalid_label_or_empty_text_rejected(self):
        for extra in [[("unknown", "hello")], [("eng", "  ")]]:
            fixture(self.data, extra)
            with self.assertRaisesRegex(ValueError, "Invalid"):
                load_dataset(self.data)

    def test_vectorizer_fitted_only_on_training_text(self):
        model = build_model("word_tfidf_nb")
        model.fit(["trainalpha familiar", "trainbeta other"], ["afr", "eng"])
        model.predict(["validationexclusive familiar"])
        self.assertNotIn("validationexclusive", model.named_steps["tfidf"].vocabulary_)


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data, self.run = self.root / "train.csv", self.root / "run"
        fixture(self.data)
        prepare(self.data, self.run)

    def test_data_change_invalidates_frozen_run(self):
        with self.data.open("a", encoding="utf-8") as stream:
            stream.write("eng,extra text\n")
        with self.assertRaisesRegex(ValueError, "Dataset changed"):
            select(self.run)

    def test_existing_run_not_overwritten(self):
        with self.assertRaisesRegex(ValueError, "already exists"):
            prepare(self.data, self.run)

    def test_selection_inference_and_single_holdout(self):
        selection = select(self.run)
        self.assertFalse(selection["holdout_used_for_selection"])
        self.assertFalse((self.run / "holdout-opened.json").exists())
        self.assertEqual(set(selection["candidates"]), {"word_tfidf_nb", "char_tfidf_svm"})
        model, _ = load_selected(self.run)
        text = "ENGLANGUAGE ENGWORDS NEWEXAMPLE"
        self.assertEqual(predict(self.run, text)["language"], model.predict([normalize(text)])[0])
        result = evaluate(self.run)
        self.assertEqual(sum(v["support"] for v in result["per_language"].values()), result["rows"])
        self.assertEqual(sum(map(sum, result["confusion_matrix"])), result["rows"])
        with self.assertRaises(FileExistsError):
            evaluate(self.run)
        with self.assertRaises(FileExistsError):
            select(self.run)

    def test_tampered_model_refused_before_deserialization(self):
        select(self.run)
        (self.run / "selected-model.joblib").write_bytes(b"not a trusted model")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            predict(self.run, "hello world")

    def test_edited_protocol_refused(self):
        select(self.run)
        path = self.run / "protocol.json"
        protocol = json.loads(path.read_text(encoding="utf-8"))
        protocol["seed"] = 7
        path.write_text(json.dumps(protocol), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Protocol was edited"):
            predict(self.run, "hello world")

    def test_invalid_inference_input_rejected(self):
        for text in [" ", "a" * 10001]:
            with self.assertRaisesRegex(ValueError, "characters"):
                predict(self.run, text)


if __name__ == "__main__":
    unittest.main()
