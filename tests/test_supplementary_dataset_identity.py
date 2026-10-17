"""Historical result compatibility must not admit altered experimental content."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from experiments.main_eval.supplementary.dataset_identity import dataset_provenance


class DatasetIdentityTests(unittest.TestCase):
    def setUp(self):
        scratch = Path(__file__).resolve().parents[1] / "tmp"
        scratch.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.dataset = Path(self.temp.name) / "main.jsonl"
        self.old = b'{"version": "1.0.1", "question": "original"}\n'
        self.new = self.old.replace(b'1.0.1', b'1.0.0')
        self.old_sha = hashlib.sha256(self.old).hexdigest()
        self.dataset.write_bytes(self.new)
        self.record = {"changed_dataset_fields": ["version"],
                       "original_dataset_version": "1.0.1", "dataset_version": "1.0.0",
                       "splits": {"main": {"records": 1,
                            "original_file_sha256": self.old_sha,
                            "normalized_file_sha256": hashlib.sha256(self.new).hexdigest()}}}
        self.record_path = self.dataset.parent / "version_normalization.json"

    def save_record(self):
        self.record_path.write_text(json.dumps(self.record), encoding="utf-8")

    def test_exact_match_needs_no_migration(self):
        result = dataset_provenance(self.dataset, hashlib.sha256(self.new).hexdigest())
        self.assertEqual(result["match"], "exact")

    def test_reversed_bytes_match_without_rewriting_input(self):
        self.save_record()
        result = dataset_provenance(self.dataset, self.old_sha)
        self.assertEqual(result["restored_sha256"], self.old_sha)
        self.assertEqual(self.dataset.read_bytes(), self.new)

    def test_missing_migration_rejected(self):
        with self.assertRaises(ValueError):
            dataset_provenance(self.dataset, self.old_sha)

    def test_changed_question_rejected_even_with_updated_record(self):
        changed = self.new.replace(b'original', b'altered')
        self.dataset.write_bytes(changed)
        self.record["splits"]["main"]["normalized_file_sha256"] = hashlib.sha256(changed).hexdigest()
        self.save_record()
        with self.assertRaisesRegex(ValueError, "does not reproduce"):
            dataset_provenance(self.dataset, self.old_sha)

    def test_other_migration_fields_rejected(self):
        self.record["changed_dataset_fields"].append("question")
        self.save_record()
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            dataset_provenance(self.dataset, self.old_sha)


if __name__ == "__main__":
    unittest.main()
