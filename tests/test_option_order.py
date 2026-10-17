"""Offline checks for answer remapping, reference integrity and paired estimates."""
import contextlib
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.main_eval import protocol as qa
from experiments.main_eval.vllm_backend import load_engine
from experiments.option_order import analyze, protocol, run
from experiments.scope_diagnostic.protocol import SETTINGS, sample_items
from misleadqa_utils import CURRENT_VERSION, DEFAULT_DATASET
from tests.test_runner import FakeBackend, FakeVLLM


def response(request, letter):
    return request | {"response": letter, "generated_token_ids": [4 + "ABCDE".index(letter), 2],
                      "input_token_ids": [10], "input_tokens": 1, "output_tokens": 2,
                      "rendered_prompt": request["messages"][0]["content"], "finish_reason": "stop"}


def reference_fixture(alias="qwen3-8b"):
    items, sample = sample_items()
    spec = qa.load_models()[alias]
    backend = spec.get("required_backend", "transformers")
    engine = load_engine(spec, SETTINGS) if backend == "vllm" else None
    model = FakeVLLM(spec, SETTINGS, engine) if engine else FakeBackend(spec, SETTINGS)
    identity = {"model_alias": alias, "model": spec, "backend": backend, "engine": engine,
                "settings": SETTINGS.copy(), "protocol": qa.PROTOCOL_VERSION, "dry_run": False,
                "dataset_sha256": qa.file_hash(DEFAULT_DATASET), "selection": "all", "limit": None,
                "counts": {"selected_items": 959}, "conditions": list(qa.CONDITION_IDS)}
    return {"protocol": protocol.VERSION, "dataset_version": CURRENT_VERSION, "model_alias": alias,
            "sample_sha256": qa.digest(qa.canonical(sample)), "baseline_identity": identity,
            "model_metadata": model.metadata, "baseline_files_sha256": {"fixture": "0" * 64},
            "responses": [response(r, r["answer_key"]["gold"]) for r in qa.make_requests(items)
                          if r["condition_id"] in protocol.CONDITIONS]}


def complete_fixture(directory, alias, rerun_original=False):
    directory.mkdir(parents=True)
    reference = reference_fixture(alias)
    requests, identity = protocol.make_plan(reference, rerun_original=rerun_original)
    qa.save_json(directory / "baseline.json", reference)
    qa.write_jsonl(directory / "requests.jsonl", requests)
    qa.save_json(directory / "manifest.json", {"identity": identity | {"dry_run": False},
        "requests_file_sha256": qa.file_hash(directory / "requests.jsonl")})
    metadata = deepcopy(reference["model_metadata"])
    if rerun_original:
        metadata["runtime"] = {"gpu": "RTX A6000"}
    qa.save_json(directory / "model.json", metadata)
    rows = []
    for r in requests:
        correct = r["condition_id"] == "closed_book"
        if rerun_original and r["item_id"] == requests[0]["item_id"] and r["permutation_id"] == "original":
            correct = False
        rows.append(response(r, r["answer_key"]["gold" if correct else "lure"]))
    qa.write_jsonl(directory / "raw_responses.jsonl", rows)
    (directory / "requirements-frozen.txt").write_text("fake-backend==1\n", encoding="utf-8")
    qa.save_json(directory / "complete.json", {name: qa.file_hash(directory / name) for name in run.SEALED_FILES})


def complete_f_fixture(directory, baseline):
    manifest, reference, rows = run.verify_run(baseline)
    metadata = json.loads((baseline / "model.json").read_text(encoding="utf-8"))
    paired = {"identity": manifest["identity"], "model_metadata": metadata, "responses": rows,
              "source_files_sha256": {name: qa.file_hash(baseline / name)
                                      for name in (*run.SEALED_FILES, "complete.json")}}
    requests, identity = run.make_paired_plan(reference, paired)
    directory.mkdir(parents=True)
    qa.save_json(directory / "baseline.json", reference)
    qa.save_json(directory / "paired_baseline.json", paired)
    qa.save_json(directory / "model.json", metadata)
    qa.write_jsonl(directory / "requests.jsonl", requests)
    qa.save_json(directory / "manifest.json", {"identity": identity | {"dry_run": False},
        "requests_file_sha256": qa.file_hash(directory / "requests.jsonl")})
    qa.write_jsonl(directory / "raw_responses.jsonl", [response(r, r["answer_key"][
        "gold" if r["permutation_id"] == "original" else "lure"]) for r in requests])
    (directory / "requirements-frozen.txt").write_text("fake-backend==1\n", encoding="utf-8")
    qa.save_json(directory / "complete.json", {name: qa.file_hash(directory / name)
                                              for name in run.sealed_files(identity)})


class OptionOrderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.items, _ = sample_items()

    def test_rotations_move_answer_roles_and_keep_evidence_and_E(self):
        item = self.items[0]
        snapshot = deepcopy(item)
        for shift, original_labels in enumerate(("ABCDE", "BCDAE", "CDABE")):
            rotated, mapping = protocol.rotate_item(item, shift)
            self.assertEqual("".join(mapping.values()), original_labels)
            self.assertEqual(rotated["options"][-1], item["options"][-1])
            self.assertEqual(rotated["evidence"], item["evidence"])
            for role, old in item["answer_key"].items():
                new = rotated["answer_key"][role]
                self.assertEqual(mapping[new], old)
                self.assertEqual(rotated["option_roles"][new], role)
        self.assertEqual(item, snapshot)
        with self.assertRaises(ValueError):
            protocol.rotate_item(item, 3)
        requests = protocol.make_requests(self.items)
        self.assertEqual((len(requests), len({r["request_id"] for r in requests})), (800, 800))
        for original, repeated in zip([r for r in qa.make_requests(self.items) if r["condition_id"] in protocol.CONDITIONS],
                                      protocol.make_requests(self.items, shifts=(0,))):
            self.assertEqual(original["messages"], repeated["messages"])
            self.assertEqual(original["answer_key"], repeated["answer_key"])

    def test_reference_rejects_wrong_model_prompt_and_missing_first_responses(self):
        reference = reference_fixture()
        protocol.validate_reference(reference)
        for mutate in (lambda r: r["responses"].pop(),
                       lambda r: r["responses"].append(r["responses"][0]),
                       lambda r: r["responses"][0]["messages"][0].update(content="Changed prompt"),
                       lambda r: r["baseline_identity"]["settings"].update(seed=1),
                       lambda r: r["model_metadata"]["generation_config"].update(do_sample=True)):
            changed = deepcopy(reference)
            mutate(changed)
            with self.assertRaises(ValueError):
                protocol.validate_reference(changed)

    def test_resume_retains_first_answers_and_complete_seal_detects_tampering(self):
        FakeBackend.calls = []
        FakeBackend.fail_at = 3
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            qa.save_json(root / "qwen3-8b.json", reference_fixture())
            args = ["--model", "qwen3-8b", "--reference-root", str(root), "--smoke", "--output-root", str(root / "runs")]
            with patch.object(run, "TransformersBackend", FakeBackend), patch.object(run, "subprocess_freeze", return_value="fake==1\n"):
                try:
                    with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                        run.main(args)
                    directory = root / "runs/qwen3-8b/main-option-order-smoke-v1"
                    raw = directory / "raw_responses.jsonl"
                    prefix = raw.read_bytes()
                    with raw.open("ab") as stream:
                        stream.write(b'{"torn":')
                    FakeBackend.fail_at = None
                    run.main(args)
                    self.assertEqual(len(FakeBackend.calls), 20)
                    self.assertTrue(raw.read_bytes().startswith(prefix))
                    self.assertEqual(len(run.verify_run(directory)[2]), 20)
                    with patch.object(run, "TransformersBackend", side_effect=AssertionError("must not reload")):
                        run.main(args)
                    raw.write_bytes(raw.read_bytes() + b"\n")
                    with self.assertRaisesRegex(ValueError, "modified"):
                        run.verify_run(directory)
                finally:
                    FakeBackend.fail_at = None

    def test_runtime_mismatch_stops_before_new_answers(self):
        class ChangedBackend(FakeBackend):
            def __init__(self, *args):
                super().__init__(*args)
                self.metadata["runtime"] = {"different_gpu": True}

        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            qa.save_json(root / "qwen3-8b.json", reference_fixture())
            with patch.object(run, "TransformersBackend", ChangedBackend), patch.object(run, "subprocess_freeze", return_value="fake==1\n"):
                with self.assertRaisesRegex(ValueError, "runtime"):
                    run.main(["--model", "qwen3-8b", "--reference-root", str(root), "--output-root", str(root / "runs"), "--smoke"])
            self.assertFalse(list((root / "runs").rglob("raw_responses.jsonl")))

    def test_fresh_mode_allows_new_host_but_preserves_model_and_decoding(self):
        expected = reference_fixture()["model_metadata"]
        changed = deepcopy(expected)
        changed.update(runtime={"gpu": "RTX A6000"}, packages={"python": "3.12.10"}, device_map={"": "0"})
        with self.assertRaisesRegex(ValueError, "runtime"):
            run.check_metadata(changed, expected)
        run.check_metadata(changed, expected, rerun_original=True)
        for key, value in (("revision", "different"), ("chat_template_kwargs", {"enable_thinking": True}),
                           ("generation_config", {"do_sample": True}), ("tokenizer_sha256", "different")):
            with self.assertRaisesRegex(ValueError, key):
                run.check_metadata(changed | {key: value}, expected, rerun_original=True)
        requests, identity = protocol.make_plan(reference_fixture(), rerun_original=True)
        self.assertEqual(len(requests), 1200)
        self.assertEqual(len({r["request_id"] for r in requests}), 1200)
        self.assertEqual(identity["permutations"], list(protocol.ORDERS))
        self.assertTrue(identity["rerun_original"])

    def test_fresh_smoke_and_seal_include_all_three_orders(self):
        FakeBackend.calls = []
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            qa.save_json(root / "qwen3-8b.json", reference_fixture())
            with patch.object(run, "TransformersBackend", FakeBackend), patch.object(run, "subprocess_freeze", return_value="fake==1\n"):
                directory = run.main(["--model", "qwen3-8b", "--reference-root", str(root),
                    "--output-root", str(root / "runs"), "--smoke", "--rerun-original"])
            manifest, _, rows = run.verify_run(directory)
            self.assertEqual(directory.name, "main-option-order-smoke-fresh-v1")
            self.assertEqual(len(rows), 30)
            self.assertTrue(manifest["identity"]["rerun_original"])

    def test_stability_is_scored_by_role_and_selection_is_separate_from_order(self):
        items = self.items[:3]
        base = {"responses": [response(r, r["answer_key"]["gold" if r["condition_id"] == "closed_book"
                    or r["item_id"] == items[1]["id"] else "lure"])
                for r in qa.make_requests(items) if r["condition_id"] in protocol.CONDITIONS]}
        rows = []
        for r in protocol.make_requests(items):
            correct = (r["condition_id"] == "closed_book" and r["item_id"] != items[2]["id"])
            if r["condition_id"] == "s_only":
                correct = r["item_id"] == items[0]["id"]
            rows.append(response(r, r["answer_key"]["gold" if correct else "lure"]))
        joined = analyze.join_results(items, base, rows)
        summary = analyze.statistics(joined)
        self.assertEqual((summary["original_correct_items"], summary["stable_correct_items"]), (3, 2))
        rates = summary["metrics"]
        self.assertEqual(rates["original_cohort_S_lure_original"]["rate"], 2 / 3)
        self.assertEqual(rates["stable_cohort_S_lure_original"]["rate"], 1 / 2)
        self.assertEqual(rates["stable_cohort_S_lure_mean"], {"numerator": 3, "denominator": 6, "rate": 1 / 2})
        self.assertEqual(len({joined[0]["answers"][order]["closed_book"]["label"] for order in protocol.ORDERS}), 3)
        self.assertEqual(len({joined[0]["answers"][order]["closed_book"]["original_label"] for order in protocol.ORDERS}), 1)
        first = analyze.paired_bootstrap({"a": joined, "b": joined}, replicates=25)
        self.assertEqual(first, analyze.paired_bootstrap({"a": joined, "b": joined}, replicates=25))
        self.assertEqual(first["intervals"]["a"], first["intervals"]["equal_model_mean"])
        empty = deepcopy(joined)
        for item in empty:
            item["stable_correct"] = False
        result = analyze.paired_bootstrap({"a": joined, "empty": empty}, "dependency", replicates=25)
        interval = result["intervals"]["equal_model_mean"]["stable_cohort_S_lure_mean"]
        self.assertIsNone(interval["estimate"])
        self.assertIsNone(interval["lower"])
        self.assertEqual(interval["invalid_denominator_replicates"], 25)

    def test_full_six_model_analysis_and_baseline_reuse(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            for alias in qa.load_models():
                complete_fixture(root / alias / "full", alias)
            result = analyze.analyze_runs(root, "full", root / "analysis", replicates=3)
            self.assertEqual(len(result), 6)
            for summary in result.values():
                self.assertEqual(summary["stable_correct_items"], 200)
                self.assertEqual(summary["metrics"]["stable_cohort_S_lure_original"]["rate"], 0)
                self.assertEqual(summary["metrics"]["stable_cohort_S_lure_mean"]["rate"], 2 / 3)
            self.assertEqual(len(qa.read_jsonl(root / "analysis/joined_items.jsonl")), 1200)
            self.assertTrue((root / "analysis/metrics.csv").is_file())
            with self.assertRaisesRegex(ValueError, "exists"):
                analyze.analyze_runs(root, "full", root / "analysis", replicates=3)

    def test_fresh_analysis_uses_new_original_answers_and_rejects_mixed_modes(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            aliases = list(qa.load_models())
            for alias in aliases:
                complete_fixture(root / alias / "fresh", alias, rerun_original=True)
            result = analyze.analyze_runs(root, "fresh", root / "analysis", replicates=3)
            for summary in result.values():
                self.assertEqual(summary["original_correct_items"], 199)
                self.assertEqual(summary["stable_correct_items"], 199)
                self.assertEqual(summary["metrics"]["stable_cohort_S_lure_mean"]["rate"], 1)
            agreement = json.loads((root / "analysis/reference_agreement.json").read_text(encoding="utf-8"))
            self.assertEqual(len(agreement[aliases[0]]["closed_book"]["disagreements"]), 1)
            self.assertEqual(len(agreement[aliases[0]]["s_only"]["disagreements"]), 200)
            complete_fixture(root / aliases[-1] / "reused", aliases[-1])
            real_verify = run.verify_run
            def mixed_verify(directory):
                if directory.parent.name == aliases[-1]:
                    directory = directory.with_name("reused")
                return real_verify(directory)
            with patch.object(analyze, "verify_run", side_effect=mixed_verify):
                with self.assertRaisesRegex(ValueError, "Do not combine"):
                    analyze.analyze_runs(root, "fresh", root / "mixed-analysis", replicates=3)

    def test_f_only_plan_has_600_requests_and_preserves_rotated_lure_keys(self):
        requests = protocol.make_requests(self.items, (0, 1, 2), ("f_only",))
        self.assertEqual(len(requests), 600)
        self.assertEqual(len({r["request_id"] for r in requests}), 600)
        for shift in range(3):
            rotated, _ = protocol.rotate_item(self.items[0], shift)
            expected = next(r for r in qa.make_requests([rotated]) if r["condition_id"] == "f_only")
            actual = requests[shift]
            self.assertEqual(actual["messages"], expected["messages"])
            self.assertEqual(actual["answer_key"], expected["answer_key"])
        self.assertEqual(len({r["answer_key"]["lure"] for r in requests[:3]}), 3)

    def test_f_only_resume_keeps_baseline_and_rejects_runtime_changes(self):
        class MatchingBackend(FakeBackend):
            def __init__(self, *args):
                super().__init__(*args)
                self.metadata["runtime"] = {"gpu": "RTX A6000"}

        FakeBackend.calls = []
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            baseline = root / "baseline"
            complete_fixture(baseline, "qwen3-8b", rerun_original=True)
            original_hashes = {p.name: qa.file_hash(p) for p in baseline.iterdir()}
            args = ["--f-only-from", str(baseline), "--smoke", "--output-root", str(root / "runs")]
            with patch.object(run, "subprocess_freeze", return_value="fake==1\n"):
                with patch.object(run, "TransformersBackend", FakeBackend):
                    with self.assertRaisesRegex(ValueError, "runtime"):
                        run.main(args)
                with patch.object(run, "TransformersBackend", MatchingBackend):
                    FakeBackend.fail_at = 4
                    try:
                        with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                            run.main(args)
                        FakeBackend.fail_at = None
                        directory = run.main(args)
                    finally:
                        FakeBackend.fail_at = None
                self.assertEqual(len(FakeBackend.calls), 15)
                manifest, _, rows = run.verify_run(directory)
                self.assertEqual(manifest["identity"]["conditions"], ["f_only"])
                self.assertEqual(len(rows), 15)
                with patch.object(run, "TransformersBackend", side_effect=AssertionError("must not reload")):
                    run.main(args)
            self.assertEqual(original_hashes, {p.name: qa.file_hash(p) for p in baseline.iterdir()})
            paired_path = directory / "paired_baseline.json"
            paired = json.loads(paired_path.read_text(encoding="utf-8"))
            paired["responses"].pop()
            qa.save_json(paired_path, paired)
            with self.assertRaisesRegex(ValueError, "Incomplete paired baseline"):
                run.verify_run(directory)

    def test_f_results_never_select_cohorts_and_bootstrap_is_paired(self):
        items = self.items[:3]
        rows = []
        for request in protocol.make_requests(items, (0, 1, 2)):
            gold = request["condition_id"] == "closed_book"
            if request["item_id"] == items[2]["id"] and request["permutation_id"] != "original":
                gold = False
            rows.append(response(request, request["answer_key"]["gold" if gold else "lure"]))
        f_rows = [response(r, r["answer_key"]["lure" if r["item_id"] == items[2]["id"] else "gold"])
                  for r in protocol.make_requests(items, (0, 1, 2), ("f_only",))]
        joined = analyze.join_results(items, {}, rows, True, f_rows)
        summary = analyze.statistics(joined)
        self.assertEqual((summary["original_correct_items"], summary["stable_correct_items"]), (3, 2))
        self.assertEqual(summary["metrics"]["original_cohort_F_lure_original"]["rate"], 1 / 3)
        self.assertEqual(summary["metrics"]["stable_cohort_F_lure_mean"]["rate"], 0)
        self.assertEqual(summary["metrics"]["stable_cohort_F_minus_S_lure_mean"]["rate"], -1)
        self.assertEqual(summary["F_outcomes"]["stable_cohort.mean"]["gold"]["rate"], 1)
        for incomplete in (f_rows[:-1], f_rows + f_rows[:1]):
            with self.assertRaises(ValueError):
                analyze.join_results(items, {}, rows, True, incomplete)
        for item in joined:
            for order in protocol.ORDERS:
                item["answers"][order]["f_only"] = item["answers"][order]["s_only"].copy()
        result = analyze.paired_bootstrap({"a": joined, "b": deepcopy(joined)}, replicates=25)
        interval = result["intervals"]["equal_model_mean"]["stable_cohort_F_minus_S_lure_mean"]
        self.assertEqual((interval["estimate"], interval["lower"], interval["upper"]), (0, 0, 0))
        empty = deepcopy(joined)
        for item in empty:
            item["stable_correct"] = False
        result = analyze.paired_bootstrap({"a": joined, "empty": empty}, replicates=25)
        self.assertIsNone(result["intervals"]["equal_model_mean"]["stable_cohort_F_minus_S_lure_mean"]["estimate"])

    def test_complete_six_model_f_analysis_preserves_existing_s_results(self):
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            for alias in qa.load_models():
                baseline = root / alias / "fresh"
                complete_fixture(baseline, alias, rerun_original=True)
                complete_f_fixture(root / alias / "f-only", baseline)
            summaries = analyze.analyze_runs(root, "fresh", root / "analysis", replicates=3, f_run_id="f-only")
            for summary in summaries.values():
                self.assertEqual(summary["stable_correct_items"], 199)
                self.assertEqual(summary["metrics"]["stable_cohort_S_lure_mean"]["rate"], 1)
                self.assertEqual(summary["metrics"]["stable_cohort_F_lure_mean"]["rate"], 2 / 3)
                self.assertEqual(summary["metrics"]["stable_cohort_F_minus_S_lure_mean"]["rate"], -1 / 3)
            saved = json.loads((root / "analysis/statistics.json").read_text(encoding="utf-8"))
            self.assertIn("panel_F_outcomes", saved)
            report = (root / "analysis/report.md").read_text(encoding="utf-8")
            self.assertIn("Matched S/F comparisons", report)
            self.assertIn("-33.33 pp", report)


if __name__ == "__main__":
    unittest.main()
