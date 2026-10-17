"""Read-only verification of sealed context-faithful runs."""
import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    __package__ = "experiments.main_eval.supplementary"

from experiments.main_eval.protocol import (CONDITION_IDS, DEFAULT_DATASET, canonical, digest, file_hash,
                                 load_items, read_jsonl)
from experiments.main_eval.run import read_checkpoint, validate_checkpoint
from experiments.main_eval.vllm_backend import load_engine, sampling_options
from misleadqa_utils import SUPPLEMENT_DATASET
from .protocol import VERSION, make_requests, model_for, settings_for, validate_result
from .dataset_identity import dataset_provenance


def metadata_differences(expected, actual, prefix=""):
    """Explain mismatches without dumping the entire software inventory."""
    result = []
    for key in sorted(set(expected) | set(actual)):
        path = f"{prefix}.{key}" if prefix else key
        before, after = expected.get(key, "<missing>"), actual.get(key, "<missing>")
        if isinstance(before, dict) and isinstance(after, dict):
            result.extend(metadata_differences(before, after, path))
        elif before != after:
            result.append(f"{path}: manifest={before!r}, model={after!r}")
    return result


def validate_packages(before, after, backend):
    """Allow only the discovery-only addition observed in the frozen vLLM runs.

    importlib.metadata inventories depend on sys.path, which imports can extend.
    Both original snapshots remain intact. Existing versions, core packages and
    every other addition/removal are still checked strictly.
    """
    initial, loaded = before.get("installed_distributions"), after.get("installed_distributions")
    discovered = (backend == "vllm" and isinstance(initial, dict) and isinstance(loaded, dict)
                  and "colorama" not in initial and loaded.get("colorama") == "0.4.6")
    expected = before
    if discovered:
        expected = before | {"installed_distributions": initial | {"colorama": "0.4.6"}}
    differences = metadata_differences(expected, after, "packages")
    if differences:
        raise ValueError("Package snapshots differ: " + "; ".join(differences))
    if discovered:
        print("Metadata note: vLLM startup discovered colorama==0.4.6; original snapshots retained.", file=sys.stderr)


def validate_metadata(metadata, identity):
    spec, settings = identity["model"], identity["settings"]
    expected = {"model_id": spec["model_id"], "revision": spec["revision"],
                "backend": identity["backend"], "dtype": "bfloat16",
                "quantization": spec.get("weight_quantization"),
                "chat_template_kwargs": spec["chat_template_kwargs"]}
    differences = metadata_differences(expected, {k: metadata.get(k) for k in expected})
    if differences:
        raise ValueError("Model metadata differs from manifest: " + "; ".join(differences))
    validate_packages(identity["packages"], metadata["packages"], identity["backend"])
    choices, eos = metadata["choice_token_ids"], metadata["eos_token_ids"]
    if (set(choices) != set("ABCDE") or len(set(choices.values())) != 5 or not eos
            or any(type(i) is not int for i in [*choices.values(), *eos]) or set(choices.values()) & set(eos)):
        raise ValueError("Invalid choice/EOS metadata")
    expected_gen = {"max_new_tokens": settings["max_new_tokens"], "num_beams": 1, "num_return_sequences": 1}
    expected_gen["do_sample"] = False
    if any(metadata["generation_config"].get(k) != v for k, v in expected_gen.items()):
        raise ValueError("Decoding configuration differs from protocol")
    if identity["backend"] == "vllm":
        engine = load_engine(spec, settings)
        if (identity["engine"] != engine or metadata["engine"] != engine
                or metadata["effective_sampling"] != sampling_options(settings, eos)):
            raise ValueError("vLLM engine/sampling metadata mismatch")


def verify_run(directory, dataset=None):
    directory = Path(directory)
    hashes = json.loads((directory / "complete.json").read_text(encoding="utf-8"))
    expected_files = {"manifest.json", "model.json", "requests.jsonl", "raw_responses.jsonl", "requirements-frozen.txt"}
    if set(hashes) != expected_files or any(file_hash(directory / k) != v for k, v in hashes.items()):
        raise ValueError("Completed run file hashes differ from completion seal")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    identity = manifest["identity"]
    settings = identity["settings"]
    if identity["protocol"] != VERSION or identity["dry_run"]:
        raise ValueError("Expected real robustness inference outputs")
    expected_settings = settings_for(identity["variant"], settings["seed"], settings["max_new_tokens"],
                                     parser=settings["parser"])
    spec = model_for(identity["model_alias"], identity["variant"])
    if (settings != expected_settings or identity["model"] != spec or identity["selection"] != "all"
            or identity["backend"] != spec.get("required_backend", "transformers")
            or tuple(identity["conditions"]) != CONDITION_IDS):
        raise ValueError("Unexpected experiment identity")
    if dataset is None:
        candidates = [Path(manifest["dataset_path"]), DEFAULT_DATASET, SUPPLEMENT_DATASET]
        for candidate in candidates:
            if candidate.is_file():
                try:
                    dataset_provenance(candidate, identity["dataset_sha256"])
                except ValueError:
                    continue
                dataset = candidate
                break
    if dataset is None:
        raise ValueError("Matching dataset unavailable; supply --dataset")
    dataset_provenance(dataset, identity["dataset_sha256"])
    items, counts = load_items(dataset, "all", identity["limit"], settings["selection_seed"])
    requests = read_jsonl(directory / "requests.jsonl")
    if (file_hash(directory / "requests.jsonl") != manifest["requests_file_sha256"]
            or digest(canonical(requests)) != identity["requests_sha256"]
            or requests != make_requests(items, settings) or counts != identity["counts"]
            or identity["selected_item_ids"] != [i["id"] for i in items]
            or len(requests) != identity["expected_responses"]):
        raise ValueError("Saved requests differ from dataset/protocol/manifest")
    metadata = json.loads((directory / "model.json").read_text(encoding="utf-8"))
    validate_metadata(metadata, identity)
    rows = read_checkpoint(directory / "raw_responses.jsonl")
    if len(validate_checkpoint(rows, requests)) != len(requests):
        raise ValueError(f"Incomplete run: {len(rows)}/{len(requests)}")
    for row in rows:
        validate_result(row, metadata, settings)
    return manifest, metadata, rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--smoke-check", action="store_true", help="Verify every response follows the choice/EOS protocol")
    args = parser.parse_args(argv)
    _, _, rows = verify_run(args.run_directory, args.dataset)
    print(f"Verified {len(rows)} responses; {len(rows)//8} complete eight-condition items.")


if __name__ == "__main__":
    main()
