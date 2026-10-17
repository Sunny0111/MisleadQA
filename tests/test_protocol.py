"""Current release, independent prompts and portable server bundle."""
import copy
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest

from construction.screening.checks import check_item, validate_dataset
from experiments.main_eval.package import build_package
from experiments.main_eval.protocol import CONFIG_DIR, HERE, load_items, make_requests, file_hash, read_jsonl, validate_item
from misleadqa_utils import CURRENT_SCHEMA, CURRENT_VERSION, DEFAULT_DATASET, SUPPLEMENT_DATASET


class ProtocolTests(unittest.TestCase):
    def test_release_splits_and_unknown_review(self):
        for path, count in ((DEFAULT_DATASET, 959), (SUPPLEMENT_DATASET, 25)):
            with self.subTest(path=path.name):
                report = validate_dataset(path)
                self.assertEqual(report["status"], "pass", report["errors"])
                self.assertEqual(report["records"], count)
                self.assertIsNone(report["human_reviewed"])
                self.assertIsNone(report["inherited_evaluation_eligible"])
                rows, counts = load_items(path)
                self.assertEqual(CURRENT_VERSION, "1.0.0")
                self.assertTrue(all(row["version"] == CURRENT_VERSION and
                                    row["schema_version"] == CURRENT_SCHEMA for row in rows))
                self.assertEqual(counts["selected_items"], count)
                self.assertIsNone(counts["eligible_items"])
                with self.assertRaisesRegex(ValueError, "Eligibility metadata"):
                    load_items(path, selection="eligible")

    def test_eight_independent_conditions_exact_prompt_and_order(self):
        items, _ = load_items(DEFAULT_DATASET)
        requests = make_requests(items)
        self.assertEqual(len(requests), 7672)
        self.assertEqual(file_hash(CONFIG_DIR / "prompts/qa_v1.txt"),
                         "b6ecb8a70c7d19ef47eb44d2e04afe1c8be76bdb5b189e31b11937618bc766c2")
        for offset in range(0, len(requests), 8):
            block = requests[offset:offset + 8]
            self.assertEqual(len({r["condition_id"] for r in block}), 8)
            self.assertTrue(all(len(r["messages"]) == 1 for r in block))
            self.assertTrue(all(r["evaluation_eligible"] is None for r in block))
        item = copy.deepcopy(items[0])
        item["provenance"] = {"internal_canary": "DO_NOT_SEND_REVIEW_METADATA"}
        prompts = {r["condition_id"]: r["messages"][0]["content"] for r in make_requests([item])}
        options = "\n".join(f"{o['label']}. {o['text']}" for o in item["options"])
        base = ("Answer the following multiple-choice question. Return only A, B, C, D, or E.\n\n"
                f"Question:\n{item['question']}\n\nOptions:\n{options}")
        self.assertEqual(prompts["closed_book"], base + "\n")
        g, f, s = [item["evidence"][key]["text"] for key in ("correct", "false", "mis_scoped")]
        for condition, passages in (("g_only", [g]), ("f_only", [f]), ("s_only", [s]),
                                    ("g_then_f", [g, f]), ("f_then_g", [f, g]),
                                    ("g_then_s", [g, s]), ("s_then_g", [s, g])):
            self.assertEqual(prompts[condition], base + "\n\n" + "\n\n".join(passages) + "\n")
        self.assertTrue(all("DO_NOT_SEND_REVIEW_METADATA" not in p for p in prompts.values()))

    def test_current_validation_still_rejects_structural_errors(self):
        item = read_jsonl(DEFAULT_DATASET)[0]
        self.assertEqual(check_item(item), [])
        for mutate in (lambda r: r["evidence"]["false"].update(sha256="bad"),
                       lambda r: r["construction"]["false_edits"][0].update(before="bad"),
                       lambda r: r["grouping"].pop("cluster_id"),
                       lambda r: r.update(review={"human_reviewed": "unknown", "evaluation_eligible": True})):
            changed = copy.deepcopy(item)
            mutate(changed)
            self.assertTrue(check_item(changed))
        with self.assertRaisesRegex(ValueError, "Unexpected schema/dataset version"):
            validate_item(item | {"version": "0.37.0", "schema_version": "1.0.0"})

    def test_limits_are_reproducible(self):
        first, counts = load_items(DEFAULT_DATASET, limit=2, seed=2027)
        self.assertEqual(first, load_items(DEFAULT_DATASET, limit=2, seed=2027)[0])
        self.assertEqual(counts["selected_items"], 2)
        self.assertEqual(len(make_requests(first)), 16)

    def test_direct_entry_points_from_another_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ("run", "verify", "analyze", "preflight", "package"):
                result = subprocess.run([sys.executable, "-B", str((HERE.parent / "analysis" if name == "analyze" else HERE) / f"{name}.py"), "--help"],
                                        cwd=directory, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_bundle_snapshots_and_extracted_dry_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "bundle.tar.gz"
            result = build_package(destination)
            self.assertEqual(result["requests_six_models"], 47232)
            self.assertEqual(result["splits"]["main"]["requests_six_models"], 46032)
            extracted = root / "extracted"
            with tarfile.open(destination) as archive:
                names = archive.getnames()
                for member in archive.getmembers():
                    self.assertEqual((member.uid, member.gid, member.uname, member.gname),
                                     (0, 0, "", ""))
                    self.assertEqual(member.mtime, 0)
                    self.assertFalse(set(member.pax_headers) & {"atime", "ctime", "uname", "gname"})
                for name in ("experiments/analysis/metrics.py", "tests/test_metrics.py", "misleadqa_utils.py",
                             "construction/screening/checks.py", "construction/data/hashes.txt",
                             "construction/audit/interface/src/App.jsx",
                             "construction/audit/interface/src/styles.css",
                             "construction/audit/interface/server/index.js",
                             "construction/audit/interface/scripts/freeze-sample.js",
                             "construction/audit/interface/package-lock.json",
                             "construction/audit/interface/public/favicon.svg",
                             "construction/audit/analysis/analyze.mjs",
                             "construction/human audit/main150-v1/A1.submitted.json"):
                    self.assertIn(name, names)
                self.assertFalse(any("evidence_v037/" in n or "outputs/" in n or n == "dataset.jsonl" for n in names))
                self.assertFalse(any(set(Path(n).parts) & {"node_modules", "dist", "responses"}
                                     or n.endswith(".draft.json") for n in names))
                for path in (DEFAULT_DATASET, SUPPLEMENT_DATASET):
                    self.assertEqual(archive.extractfile("construction/data/" + path.name).read(), path.read_bytes())
                archive.extractall(extracted, filter="data")
            command = [sys.executable, "-B", str(extracted / "experiments/main_eval/run.py"), "--model", "qwen3-8b",
                       "--run-id", "portable", "--limit", "2", "--dry-run"]
            run = subprocess.run(command, cwd=root, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertTrue(destination.with_name(destination.name + ".sha256").exists())
            with self.assertRaisesRegex(ValueError, "already exists"):
                build_package(destination)


if __name__ == "__main__":
    unittest.main()
