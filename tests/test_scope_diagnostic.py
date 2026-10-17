"""Offline protocol, denominator, pairing, resumption and portable-bundle checks."""
from collections import Counter
import contextlib
import copy
import io
import os
from pathlib import Path
import re
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from experiments.main_eval import protocol as base
from experiments.scope_diagnostic import analyze, backend, protocol, run
from experiments.scope_diagnostic.package import build_package, verify_bundle


def observation(item_id, labels, outcome="lure", correct=True, cluster=None):
    return {"item_id": item_id, "source": "FRED", "dimension": "time",
            "cluster_id": cluster or item_id, "judgments": dict(zip("GFS", labels)),
            "closed_book_correct": correct, "s_outcome": outcome}


class FakeScopeBackend:
    calls = []
    fail_at = None

    def __init__(self, spec, settings, engine=None, local_files_only=False):
        self.metadata = {"backend": "vllm" if engine else "transformers", "model_id": spec["model_id"],
            "revision": spec["revision"], "dtype": "bfloat16", "quantization": spec.get("weight_quantization"),
            "chat_template_kwargs": spec["chat_template_kwargs"],
            "label_token_ids": dict(zip(protocol.LABELS, ([4, 5], [6, 4, 5], [7, 8]))), "eos_token_ids": [2],
            "output_constraint": "greedy label-token trie then EOS",
            "generation_config": {"do_sample": False, "max_new_tokens": 4, "num_beams": 1, "num_return_sequences": 1},
            "tokenizer_sha256": "test-tokenizer", "chat_template_sha256": "test-template", "asset_sha256": {}}

    def preflight(self, requests):
        pass

    def generate(self, messages):
        if len(self.calls) == self.fail_at:
            raise RuntimeError("simulated interruption")
        assert len(messages) == 1 and set(messages[0]) == {"role", "content"}
        assert messages[0]["role"] == "user"
        self.calls.append(copy.deepcopy(messages))
        label = protocol.LABELS[(len(self.calls) - 1) % 3]
        tokens = self.metadata["label_token_ids"][label] + [2]
        return {"response": label, "generated_token_ids": tokens, "input_token_ids": [10],
                "input_tokens": 1, "output_tokens": len(tokens),
                "rendered_prompt": messages[0]["content"], "finish_reason": "stop"}


class ScopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.items, cls.sample = protocol.sample_items()

    def test_frozen_sample_and_independent_prompts(self):
        selected, sample = protocol.select_sample()
        self.assertEqual(sample, self.sample)
        self.assertEqual(len(selected), 200)
        self.assertEqual(set(Counter((i["source"], i["dimension"]) for i in selected).values()), {40})
        requests = protocol.make_requests(selected)
        self.assertEqual(len(requests), 600)
        self.assertEqual(len({r["request_id"] for r in requests}), 600)
        for item, rows in zip(selected, (requests[i:i + 3] for i in range(0, 600, 3))):
            self.assertEqual([r["condition_id"] for r in rows], list("GFS"))
            for row, key in zip(rows, protocol.EVIDENCE.values()):
                text = row["messages"][0]["content"]
                self.assertIn(item["question"], text)
                self.assertIn(item["evidence"][key]["text"], text)
                self.assertNotIn("answer_key", text)
                self.assertEqual(len(row["messages"]), 1)
        smoke, _ = protocol.sample_items(smoke=True)
        self.assertEqual(len(smoke), 5)
        self.assertTrue({i["id"] for i in smoke} <= {i["id"] for i in self.items})

    def test_english_prompt_labels_and_public_repository_text(self):
        self.assertEqual(protocol.LABELS, ("Applicable", "Not applicable", "Cannot determine"))
        prompt = (protocol.CONFIG_DIR / "prompts/scope_v1.txt").read_text(encoding="utf-8")
        self.assertIn("Ignore whether the answer value is factually correct", prompt)
        self.assertIn("Applicable, Not applicable, or Cannot determine", prompt)
        excluded = {".git", "__pycache__", ".venv", "dist", "tmp", "output", "outputs", "analysis",
                    "node_modules", ".vite", "downloads", "responses"}
        text_extensions = {".py", ".md", ".txt", ".sh", ".json", ".jsonl", ".csv", ".toml", ".yaml", ".yml"}
        han = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\U00020000-\U0002ffff]")
        for directory, folders, files in os.walk(base.ROOT):
            folders[:] = [name for name in folders if name not in excluded and not name.startswith(".venv-")]
            if Path(directory) == base.ROOT / "construction/sources":
                folders[:] = []  # Local raw archives are not repository documentation.
            for name in files:
                path = Path(directory) / name
                if name.endswith(".draft.json"):
                    continue
                if path.suffix in text_extensions:
                    self.assertIsNone(han.search(path.read_text(encoding="utf-8")), str(path.relative_to(base.ROOT)))

    def test_literal_multitoken_constraint_and_eos(self):
        sequences = dict(zip(protocol.LABELS, ([4, 5], [6, 4, 5], [7, 8])))
        trie = backend.LabelConstraint(sequences, [2, 3])
        self.assertEqual(trie.budget, 4)
        self.assertEqual(trie.allowed([]), [4, 6, 7])
        self.assertEqual(trie.allowed([6, 4]), [5])
        self.assertEqual(trie.allowed([6, 4, 5]), [2, 3])
        with self.assertRaises(ValueError):
            trie.allowed([6, 5])
        metadata = {"label_token_ids": sequences, "eos_token_ids": [2, 3]}
        row = {"response": "Not applicable", "generated_token_ids": [6, 4, 5, 2], "finish_reason": "stop",
               "input_token_ids": [10], "input_tokens": 1, "output_tokens": 4}
        backend.verify_generation(row, metadata)
        for altered in (row | {"response": "Applicable"}, row | {"finish_reason": "length"},
                        row | {"generated_token_ids": [6, 4, 5]}, row | {"output_tokens": 3}):
            with self.assertRaises(ValueError):
                backend.verify_generation(altered, metadata)

    def test_uncertain_denominators_full_tables_and_B_filter(self):
        items = [observation("a", ("Applicable", "Cannot determine", "Not applicable")),
                 observation("b", ("Not applicable", "Applicable", "Applicable"), "gold"),
                 observation("c", ("Cannot determine", "Applicable", "Cannot determine"), "invalid"),
                 observation("d", ("Applicable", "Applicable", "Not applicable"), correct=False)]
        summary = analyze.statistics(items)
        m = summary["metrics"]
        self.assertEqual(m["balanced_accuracy"]["rate"], (5 / 8 + 2 / 4) / 2)
        self.assertEqual(m["G.uncertain"], {"numerator": 1, "denominator": 4, "rate": .25})
        self.assertEqual(m["lure_given_inapplicable"], {"numerator": 1, "denominator": 1, "rate": 1})
        self.assertEqual(m["lure_given_applicable"]["rate"], 0)
        self.assertEqual(m["inapplicable_and_lure"], {"numerator": 1, "denominator": 3, "rate": 1 / 3})
        self.assertEqual(summary["confusion_matrix"]["Applicable"], {"Applicable": 5, "Not applicable": 1, "Cannot determine": 2})
        self.assertEqual(summary["cross_table_on_B_m"]["Cannot determine"]["invalid"], 1)
        self.assertEqual(sum(sum(r.values()) for r in summary["cross_table_on_B_m"].values()), 3)
        tables = analyze.equal_model_tables({"one": summary, "two": summary})
        self.assertEqual(tables["confusion_matrix"]["Applicable"]["Applicable"]["rate"], 5 / 8)
        self.assertEqual(tables["cross_table_on_B_m"]["Not applicable"]["lure"]["rate"], 1 / 3)
        self.assertEqual(tables["cross_table_on_B_m"]["Not applicable"]["lure"]["model_components"]["one"],
                         {"numerator": 1, "denominator": 3, "rate": 1 / 3})

    def test_zero_denominators_fixed_panel_and_paired_clusters(self):
        rows = [observation("a", ("Applicable", "Applicable", "Not applicable"), cluster="shared"),
                observation("b", ("Applicable", "Applicable", "Cannot determine"), "gold", cluster="shared"),
                observation("c", ("Not applicable", "Not applicable", "Applicable"), "gold", cluster="other")]
        zero = analyze.statistics([row | {"closed_book_correct": False} for row in rows])
        self.assertIsNone(zero["metrics"]["inapplicable_and_lure"]["rate"])
        mean = analyze.equal_model_mean({"a": analyze.statistics(rows), "b": zero})
        self.assertIsNone(mean["inapplicable_and_lure"]["rate"])
        models = {name: copy.deepcopy(rows) for name in ("m1", "m2")}
        result = analyze.paired_bootstrap(models, replicates=80, seed=7)
        self.assertEqual(result, analyze.paired_bootstrap(models, replicates=80, seed=7))
        self.assertEqual(result["clusters"], 2)
        intervals = result["intervals"]
        for key in intervals["m1"]:
            self.assertEqual(intervals["m1"][key], intervals["m2"][key])
            self.assertEqual(intervals["m1"][key], intervals["equal_model_mean"][key])
        self.assertGreater(intervals["m1"]["lure_given_inapplicable"]["invalid_denominator_replicates"], 0)
        models["m2"][0]["cluster_id"] = "changed"
        with self.assertRaisesRegex(ValueError, "memberships"):
            analyze.paired_bootstrap(models, 2)

    def test_resume_torn_append_complete_seal_and_no_retest(self):
        FakeScopeBackend.calls, FakeScopeBackend.fail_at = [], 4
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()), \
                patch.object(run, "ScopeBackend", FakeScopeBackend), \
                patch.object(run, "package_versions", return_value={}), \
                patch.object(run, "subprocess_freeze", return_value="test==1\n"):
            args = ["--model", "qwen3-8b", "--smoke", "--run-id", "test", "--output-root", temp]
            directory = Path(temp) / "qwen3-8b/test"
            with self.assertRaisesRegex(RuntimeError, "interruption"):
                run.main(args)
            path = directory / "raw_responses.jsonl"
            before = path.read_bytes()
            with path.open("ab") as stream:
                stream.write(b'{"torn":')
            FakeScopeBackend.fail_at = None
            run.main(args)
            self.assertTrue(path.read_bytes().startswith(before))
            self.assertEqual(len(FakeScopeBackend.calls), 15)
            run.main(args)
            self.assertEqual(len(FakeScopeBackend.calls), 15)
            self.assertEqual(len(run.verify_run(directory)[2]), 15)
            self.assertFalse((directory / ".lock").exists())
            rows = base.read_jsonl(path)
            rows[0]["response"] = "Cannot determine"
            base.write_jsonl(path, rows)
            with self.assertRaisesRegex(ValueError, "modified"):
                run.main(args)

    def test_six_model_analysis_and_portable_bundle(self):
        answers = [r | {"response": r["answer_key"]["gold" if r["condition_id"] == "closed_book" else "lure"],
                       "finish_reason": "stop"} for r in base.make_requests(base.load_items()[0])]
        judgments = [r | {"response": ("Not applicable" if index % 2 else "Cannot determine") if r["condition_id"] == "S" else "Applicable"}
                     for index, r in enumerate(protocol.make_requests(self.items))]
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            metadata = {}
            for name, spec in base.load_models().items():
                metadata[name] = FakeScopeBackend(spec, protocol.SETTINGS).metadata
                for kind in ("scope", "baseline"):
                    directory = root / kind / name / "full"
                    directory.mkdir(parents=True)
                    (directory / "raw_responses.jsonl").write_text("synthetic test data\n", encoding="utf-8")

            def loaded(directory, dataset):
                name = directory.parent.name
                original = "baseline" in directory.parts
                identity = {"model_alias": name, "model": base.load_models()[name], "smoke": False,
                            "dataset_sha256": base.file_hash(dataset), "settings": protocol.SETTINGS,
                            "counts": {"selected_items": 959}}
                return {"identity": identity}, metadata[name], answers if original else judgments

            with patch.object(analyze, "verify_run", side_effect=loaded), \
                    patch.object(analyze, "verify_baseline", side_effect=loaded):
                summaries, bootstrap = analyze.analyze(root / "scope", root / "analysis", "full",
                    base.DEFAULT_DATASET, root / "baseline", "full", replicates=20)
            self.assertEqual(len(summaries), 6)
            self.assertEqual(sum(s["items"] for s in summaries.values()), 1200)
            self.assertEqual(summaries["qwen3-8b"]["B_m_size"], 200)
            self.assertEqual(summaries["qwen3-8b"]["metrics"]["lure_given_inapplicable"]["rate"], 1)
            self.assertEqual(len(bootstrap["panel_models"]), 6)
            report = (root / "analysis/report.md").read_text(encoding="utf-8")
            self.assertIn("Not estimable", report)
            self.assertIn("Cannot determine", report)
            self.assertIn("causal", report)
            self.assertEqual(len(base.read_jsonl(root / "analysis/joined_items.jsonl")), 1200)
            info = build_package(root / "scope.tar.gz")
            self.assertEqual(info["scope_diagnostic"]["requests_six_models"], 3600)
            with tarfile.open(root / "scope.tar.gz") as archive:
                self.assertFalse(any("/outputs/" in p for p in archive.getnames()))
                self.assertIn("experiments/analysis/metrics.py", archive.getnames())
                archive.extractall(root / "unpacked", filter="data")
            verify_bundle(root / "unpacked")


if __name__ == "__main__":
    unittest.main()
