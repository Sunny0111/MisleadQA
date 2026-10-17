"""Regression for vLLM's startup package inventory change and strict version checks."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.main_eval.protocol import write_jsonl
from experiments.main_eval.supplementary import protocol, run, verify
from tests.test_runner import FakeVLLM, fixture


BEFORE = {"python": "3.12.14", "torch": "2.7.0+cu118", "transformers": "4.53.2",
          "installed_distributions": {"torch": "2.7.0+cu118", "transformers": "4.53.2", "vllm": "0.9.2+cu118"}}
AFTER = BEFORE | {"installed_distributions": BEFORE["installed_distributions"] | {"colorama": "0.4.6"}}


class DiscoveryVLLM(FakeVLLM):
    def __init__(self, *args):
        super().__init__(*args)
        self.metadata["packages"] = copy.deepcopy(AFTER)


class MetadataTests(unittest.TestCase):
    def test_only_known_vllm_discovery_is_allowed_without_mutating_snapshots(self):
        before, after = copy.deepcopy(BEFORE), copy.deepcopy(AFTER)
        with contextlib.redirect_stderr(io.StringIO()) as log:
            verify.validate_packages(before, after, "vllm")
        self.assertIn("colorama==0.4.6", log.getvalue())
        self.assertEqual((before, after), (BEFORE, AFTER))
        verify.validate_packages(BEFORE, BEFORE, "vllm")
        verify.validate_packages(AFTER, AFTER, "vllm")
        with self.assertRaisesRegex(ValueError, "packages.installed_distributions.colorama"):
            verify.validate_packages(before, after, "transformers")

    def test_core_versions_other_packages_removals_and_existing_colorama_are_strict(self):
        for group, key, value in ((None, "torch", "2.8.0"), (None, "python", "3.12.15"),
                                  ("installed_distributions", "vllm", "0.10.0"),
                                  ("installed_distributions", "numpy", "2.0"),
                                  ("installed_distributions", "colorama", "0.4.7")):
            changed = copy.deepcopy(AFTER)
            (changed[group] if group else changed)[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                verify.validate_packages(BEFORE, changed, "vllm")
        changed = copy.deepcopy(AFTER)
        del changed["installed_distributions"]["torch"]
        with self.assertRaisesRegex(ValueError, "installed_distributions.torch"):
            verify.validate_packages(BEFORE, changed, "vllm")
        with self.assertRaisesRegex(ValueError, "colorama"):
            verify.validate_packages(AFTER, BEFORE, "vllm")
        prior = BEFORE | {"installed_distributions": BEFORE["installed_distributions"] | {"colorama": "0.4.5"}}
        with self.assertRaisesRegex(ValueError, "colorama"):
            verify.validate_packages(prior, AFTER, "vllm")

    def test_diagnostics_name_model_and_decoding_mismatches(self):
        settings = protocol.settings_for("context-faithful")
        spec = protocol.model_for("qwen3-32b-awq", "context-faithful")
        identity = {"model": spec, "settings": settings, "backend": "vllm", "packages": BEFORE}
        metadata = DiscoveryVLLM(spec, settings, {}).metadata
        metadata["revision"] = "wrong"
        with self.assertRaisesRegex(ValueError, "revision.*wrong"):
            verify.validate_metadata(metadata, identity)
        metadata["revision"] = spec["revision"]
        metadata["generation_config"]["do_sample"] = True
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaisesRegex(ValueError, "Decoding configuration"):
            verify.validate_metadata(metadata, identity)

    def test_complete_vllm_run_remains_byte_identical_after_offline_verification(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()), \
                patch.object(run, "VLLMBackend", DiscoveryVLLM), \
                patch.object(run, "package_versions", return_value=BEFORE), \
                patch.object(run, "subprocess_freeze", return_value="frozen packages\n"):
            root = Path(temp)
            dataset = root / "dataset.jsonl"
            write_jsonl(dataset, [fixture()])
            output = run.main(["--variant", "context-faithful", "--model", "qwen3-32b-awq", "--run-id", "regression",
                               "--dataset", str(dataset), "--output-root", str(root / "outputs")])
            files = {p.name: p.read_bytes() for p in output.iterdir() if p.is_file()}
            manifest = json.loads(files["manifest.json"])
            metadata = json.loads(files["model.json"])
            self.assertNotEqual(manifest["identity"]["packages"], metadata["packages"])
            verify.main([str(output), "--dataset", str(dataset), "--smoke-check"])
            self.assertEqual(files, {p.name: p.read_bytes() for p in output.iterdir() if p.is_file()})


if __name__ == "__main__":
    unittest.main()
