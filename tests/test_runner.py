"""Offline integration tests for first-response persistence and analysis."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.analysis import analyze
from experiments.main_eval import protocol, run, verify, vllm_backend
from experiments.main_eval.backend import allowed_tokens, verify_generation
from misleadqa_utils import DEFAULT_DATASET


def fixture():
    item = copy.deepcopy(protocol.read_jsonl(DEFAULT_DATASET)[0])
    item["grouping"].update(analysis_group_size=1, analysis_group_size_evaluation=1)
    return item


class FakeBackend:
    calls = []
    fail_at = None

    def __init__(self, spec, settings, local_only=False):
        self.metadata = {"backend": "transformers", "model_id": spec["model_id"], "revision": spec["revision"],
                         "dtype": "bfloat16", "quantization": spec.get("weight_quantization"),
                         "choice_token_ids": dict(zip("ABCDE", range(4, 9))), "eos_token_ids": [2],
                         "chat_template_kwargs": spec["chat_template_kwargs"], "runtime": {"test": True},
                         "generation_config": {"do_sample": False, "max_new_tokens": 2,
                                               "num_beams": 1, "num_return_sequences": 1}}

    def preflight(self, requests):
        pass

    def generate(self, messages):
        if self.fail_at is not None and len(self.calls) == self.fail_at:
            raise RuntimeError("simulated interruption")
        assert len(messages) == 1 and set(messages[0]) == {"role", "content"}
        assert messages[0]["role"] == "user"
        self.calls.append(copy.deepcopy(messages))
        return {"response": "C", "generated_token_ids": [6, 2], "input_token_ids": [10, 11],
                "input_tokens": 2, "output_tokens": 2, "rendered_prompt": messages[0]["content"],
                "finish_reason": "stop"}


class FakeVLLM(FakeBackend):
    def __init__(self, spec, settings, engine, local_only=False):
        super().__init__(spec, settings, local_only)
        self.metadata.update(backend="vllm", engine=engine,
                             effective_sampling=vllm_backend.sampling_options(settings, [2]))


class RunnerTests(unittest.TestCase):
    def setUp(self):
        FakeBackend.calls = []
        FakeBackend.fail_at = None

    def test_interruption_resume_no_retest_and_input_change_rejected(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            data = root / "dataset.jsonl"
            protocol.write_jsonl(data, [fixture()])
            args = ["--dataset", str(data), "--model", "qwen3-8b", "--run-id", "resume",
                    "--output-root", str(root / "outputs")]
            output = root / "outputs/qwen3-8b/resume"
            FakeBackend.fail_at = 3
            with patch.object(run, "TransformersBackend", FakeBackend):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    run.main(args)
                self.assertFalse((output / ".lock").exists())
                self.assertEqual(len(protocol.read_jsonl(output / "raw_responses.jsonl")), 3)
                before = (output / "raw_responses.jsonl").read_bytes()
                with (output / "raw_responses.jsonl").open("ab") as stream:
                    stream.write(b'{"torn":')
                FakeBackend.fail_at = None
                run.main(args)
                self.assertEqual(len(FakeBackend.calls), 8)
                self.assertTrue((output / "raw_responses.jsonl").read_bytes().startswith(before))
                _, _, rows = verify.verify_run(output, data)
                self.assertEqual(len(rows), 8)
            with patch.object(run, "TransformersBackend", side_effect=AssertionError("must not reload")):
                run.main(args)
            with self.assertRaisesRegex(ValueError, "use a new run id"):
                run.main(args + ["--seed", "99"])
            rows.append(rows[0])
            protocol.write_jsonl(output / "raw_responses.jsonl", rows)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                verify.verify_run(output, data)

    def test_analysis_uses_current_data_and_explicit_mixed_runtimes(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            data = root / "dataset.jsonl"
            protocol.write_jsonl(data, [fixture()])
            common = ["--dataset", str(data), "--run-id", "mixed", "--output-root", str(root / "outputs")]
            with patch.object(run, "TransformersBackend", FakeBackend), patch.object(run, "VLLMBackend", FakeVLLM):
                tf_dir = run.main(common + ["--model", "qwen3-8b"])
                awq_dir = run.main(common + ["--model", "qwen3-32b-awq"])
            with self.assertRaisesRegex(ValueError, "Mixed backends"):
                analyze.analyze_runs([tf_dir, awq_dir], root / "blocked", dataset=data, replicates=5)
            summaries = analyze.analyze_runs([tf_dir, awq_dir], root / "analysis", dataset=data,
                                             allow_mixed_runtimes=True, replicates=5)
            self.assertEqual(len([s for s in summaries if s["cohort"] == "all_selected"]), 2)
            self.assertEqual(summaries[0]["A_m_size"], 1)
            self.assertTrue((root / "analysis/contrasts.csv").is_file())
            bootstrap = json.loads((root / "analysis/bootstrap.json").read_text())
            self.assertEqual(bootstrap["panel_models"], [])
            changed = fixture()
            changed["question"] += " Changed."
            protocol.write_jsonl(data, [changed])
            with self.assertRaisesRegex(ValueError, "Dataset hash"):
                verify.verify_run(tf_dir, data)

    def test_constrained_token_contract(self):
        fn = allowed_tokens(2, dict(zip("ABCDE", range(4, 9))), [2])
        self.assertEqual(fn(0, [10, 11]), [4, 5, 6, 7, 8])
        self.assertEqual(fn(0, [10, 11, 8]), [2])
        with self.assertRaisesRegex(ValueError, "Unexpected"):
            fn(0, [10, 11, 9])
        metadata = {"choice_token_ids": dict(zip("ABCDE", range(4, 9))), "eos_token_ids": [2]}
        result = {"response": "E", "generated_token_ids": [8, 2], "input_token_ids": [10],
                  "input_tokens": 1, "output_tokens": 2, "finish_reason": "stop"}
        verify_generation(result, metadata)
        for fields in ({"response": ""}, {"finish_reason": "length"}, {"generated_token_ids": [8]},
                       {"input_tokens": 2}, {"generated_token_ids": [8, 3]}):
            with self.assertRaises(ValueError):
                verify_generation(result | fields, metadata)

    def test_vllm_profile_cannot_change_decoding_or_drop_context(self):
        settings = {"seed": 2027, "max_input_tokens": 4096, "dtype": "bfloat16", "max_new_tokens": 2, "batch_size": 1}
        spec = protocol.load_models()["qwen3-32b-awq"]
        engine = vllm_backend.load_engine(spec, settings)
        for key, value in (("quantization", None), ("max_model_len", 4096), ("max_num_seqs", 2)):
            changed = copy.deepcopy(engine)
            changed["llm"][key] = value
            with self.assertRaises(ValueError):
                vllm_backend.validate_engine(changed, spec, settings)


if __name__ == "__main__":
    unittest.main()
