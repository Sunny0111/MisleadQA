"""Keep structural checks independent of optional local source archives."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from construction.screening.checks import validate_dataset
from misleadqa_utils import DEFAULT_DATASET, read_jsonl


class SourceArchiveChecksTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        directory = Path(temporary.name)
        self.archive = directory / "source.txt"
        self.dataset = directory / "dataset.jsonl"
        self.content = b"Archived source text.\n"
        self.item = read_jsonl(DEFAULT_DATASET)[0]
        self.item["grouping"]["analysis_group_size"] = 1
        self.item["provenance"] = {}
        for evidence in self.item["evidence"].values():
            evidence["source_refs"] = []
        for distractor in self.item["distractors"]:
            distractor["source_refs"] = []
        self.ref = {
            "archive_file": "data/raw/test/source.txt",
            "archive_file_sha256": hashlib.sha256(self.content).hexdigest(),
        }
        self.item["evidence"]["correct"]["source_refs"] = [self.ref]
        resolver = patch("construction.screening.checks.archive_path", return_value=self.archive)
        resolver.start()
        self.addCleanup(resolver.stop)
        self.save_dataset()

    def save_dataset(self):
        self.dataset.write_text(json.dumps(self.item) + "\n", encoding="utf-8")

    def test_default_validation_does_not_depend_on_or_report_local_archives(self):
        for present in (False, True):
            with self.subTest(present=present):
                if present:
                    self.archive.write_bytes(b"Different local content.")
                report = validate_dataset(self.dataset)
                self.assertEqual(report["status"], "pass", report["errors"])
                self.assertNotIn("missing_archives", report)
                self.assertEqual(report["referenced_archives"], 1)

    def test_raw_validation_rejects_missing_or_changed_archives(self):
        report = validate_dataset(self.dataset, require_raw=True)
        self.assertEqual(report["status"], "fail")
        self.assertEqual(report["missing_archives"], [self.ref["archive_file"]])
        self.archive.write_bytes(b"Different local content.")
        report = validate_dataset(self.dataset, require_raw=True)
        self.assertEqual(report["status"], "fail")
        self.assertTrue(any("Archive checksum mismatch" in e["error"] for e in report["errors"]))
        self.archive.write_bytes(self.content)
        report = validate_dataset(self.dataset, require_raw=True)
        self.assertEqual(report["status"], "pass", report["errors"])
        self.assertEqual(report["missing_archives"], [])

    def test_default_validation_still_rejects_missing_reference_hashes(self):
        self.ref["archive_file_sha256"] = ""
        self.save_dataset()
        report = validate_dataset(self.dataset)
        self.assertEqual(report["status"], "fail")
        self.assertTrue(any("Missing or conflicting archive hashes" in e["error"]
                            for e in report["errors"]))


if __name__ == "__main__":
    unittest.main()
