"""Offline context-faithful regression tests; GPU inference is a separate smoke gate."""
import contextlib
import io
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from experiments.main_eval import protocol as base
from experiments.main_eval import run as baseline_run
from experiments.main_eval.refresh_manifest import check_manifest
from experiments.main_eval.supplementary import analyze, protocol, run, verify
from experiments.main_eval.supplementary.package import build_package, verify_bundle
from tests.test_runner import FakeBackend, fixture


class FakeContext(FakeBackend):
    def __init__(self, *args):
        super().__init__(*args)
        self.metadata["packages"] = {}


class SupplementaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.dataset = self.root / "data.jsonl"
        base.write_jsonl(self.dataset, [fixture()])
        FakeContext.calls, FakeContext.fail_at = [], None
        self.stack = contextlib.ExitStack()
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.stack.enter_context(patch.object(run, "package_versions", return_value={}))
        self.stack.enter_context(patch.object(run, "subprocess_freeze", return_value="test-runtime\n"))
        self.stack.enter_context(patch.object(run, "TransformersBackend", FakeContext))

    def tearDown(self):
        self.stack.close()
        self.temp.cleanup()

    def invoke(self, variant="context-faithful", seed=2027, run_id="test-cf-s2027", extra=()):
        return run.main(["--model", "qwen3-14b", "--variant", variant, "--seed", str(seed),
                         "--run-id", run_id, "--dataset", str(self.dataset), "--output-root", str(self.root / "outputs"),
                         *extra])

    def test_original_manifest_and_context_closed_book_are_unchanged(self):
        check_manifest()
        item = fixture()
        requests = protocol.make_requests([item], protocol.settings_for("context-faithful"))
        original = base.make_requests([item])
        self.assertEqual(requests[0]["messages"], original[0]["messages"])
        for a, b in zip(requests[1:], original[1:]):
            self.assertNotEqual(a["messages"], b["messages"])
            old_tail = b["messages"][0]["content"].split("Question:\n", 1)[1]
            self.assertEqual(a["messages"][0]["content"].split("Question:\n", 1)[1], old_tail)
        item["answer_key"]["abstain"] = "D"
        with self.assertRaisesRegex(ValueError, "abstain=E"):
            protocol.make_requests([item], protocol.settings_for("context-faithful"))


    def test_resume_retains_first_outputs_and_seeds_after_torn_append(self):
        FakeContext.fail_at = 3
        with self.assertRaisesRegex(RuntimeError, "interruption"):
            self.invoke()
        output = self.root / "outputs/qwen3-14b/test-cf-s2027"
        raw = output / "raw_responses.jsonl"
        preserved = raw.read_bytes()
        with raw.open("ab") as stream:
            stream.write(b'{"torn"')
        FakeContext.fail_at = None
        self.invoke()
        self.assertTrue(raw.read_bytes().startswith(preserved))
        self.assertEqual(len(FakeContext.calls), 8)
        _, _, rows = verify.verify_run(output, self.dataset)
        self.assertTrue(all(r["generation_seed"] == 2027 for r in rows))
        self.invoke()
        self.assertEqual(len(FakeContext.calls), 8)
        with self.assertRaisesRegex(ValueError, "seed 2027"):
            self.invoke(seed=2028)


    def test_completion_seal_detects_changed_outputs(self):
        output = self.invoke()
        path = output / "raw_responses.jsonl"
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "hashes"):
            verify.verify_run(output, self.dataset)


    def test_context_pair_with_original_runner_and_environment_mismatch(self):
        with patch.object(baseline_run, "TransformersBackend", FakeContext), \
                patch.object(baseline_run, "package_versions", return_value={}), \
                patch.object(baseline_run, "subprocess_freeze", return_value="test-runtime\n"):
            baseline = baseline_run.main(["--model", "qwen3-14b", "--run-id", "main-v100-full-v1",
                "--dataset", str(self.dataset), "--output-root", str(self.root / "baseline")])
        context = self.invoke("context-faithful", run_id="ctx-cf-s2027")
        summary, pairs = analyze.analyze(self.root / "outputs", self.root / "analysis", "ctx", self.dataset,
                                         self.root / "baseline", kind="context")
        self.assertEqual((len(summary), len(pairs)), (1, 1))
        reference = analyze.verify_baseline(baseline, self.dataset)
        treatment = verify.verify_run(context, self.dataset)
        treatment[1]["runtime"] = {"test": "changed"}
        with self.assertRaisesRegex(ValueError, "runtime mismatch"):
            analyze.check_pair(reference, treatment)


    def test_fixed_cohorts_keep_abstentions_and_invalids_in_denominators(self):
        def item(key, b, a, outcome):
            return {"item_id": key, "B_m": b, "A_m": a, "G_correct": a,
                    "outcomes": {c: {o: int(o == outcome) for o in analyze.OUTCOMES} for c in base.CONDITION_IDS}}
        left = [item("x", True, True, "lure"), item("y", True, True, "gold")]
        right = [item("x", True, True, "abstain"), item("y", False, False, "invalid")]
        result = analyze.paired_results(left, right)
        self.assertEqual(result["cohort_sizes"], {"all": 2, "reference_B": 2, "reference_A": 2, "common_B": 1, "common_A": 1})
        rows = [r for r in result["differences"] if r["cohort"] == "reference_A" and r["condition"] == "s_only"]
        self.assertEqual(sum(r["treatment_count"] for r in rows), 2)
        self.assertEqual(next(r["delta_pp"] for r in rows if r["outcome"] == "abstain"), 50)
        self.assertEqual(next(r["delta_pp"] for r in rows if r["outcome"] == "gold"), -50)

    def test_package_and_extracted_entry_points(self):
        target = self.root / "bundle.tar.gz"
        info = build_package(target)
        self.assertEqual(info["robustness"]["total_additional_responses"], 46032)
        extracted = self.root / "bundle"
        with tarfile.open(target) as archive:
            archive.extractall(extracted, filter="data")
        verify_bundle(extracted)
        for name in ("run", "verify", "analyze", "package"):
            result = subprocess.run([sys.executable, "-B", str(extracted / "experiments/main_eval/supplementary" / f"{name}.py"), "--help"],
                                    cwd=self.root, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
        result = subprocess.run([sys.executable, "-B", str(extracted / "experiments/main_eval/supplementary/run.py"), "--dry-run",
                                 "--variant", "context-faithful", "--model", "qwen3-14b", "--run-id", "dry",
                                 "--limit", "2"], cwd=self.root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
