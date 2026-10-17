"""Refresh or check the adapted application manifest without downloading models."""
import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "experiments.main_eval"

from misleadqa_utils import CONFIG_DIR, DEFAULT_DATASET, SUPPLEMENT_DATASET, CURRENT_SCHEMA, CURRENT_VERSION
from .protocol import HERE, ROOT, PROTOCOL_VERSION, file_hash, load_items, load_models, save_json, source_hashes


def current_manifest():
    original = json.loads((HERE / "handoff_manifest.json").read_text(encoding="utf-8"))
    preserved = {
        "configs/models.json": CONFIG_DIR / "models.json",
        "configs/vllm_qwen32_awq_a40.json": CONFIG_DIR / "vllm_qwen32_awq_a40.json",
        "prompts/qa_v1.txt": CONFIG_DIR / "prompts/qa_v1.txt",
        "requirements/requirements-transformers-cu118.txt": CONFIG_DIR / "requirements/requirements-transformers-cu118.txt",
        "requirements/requirements-vllm-cu118.txt": CONFIG_DIR / "requirements/requirements-vllm-cu118.txt",
    }
    if any(file_hash(path) != original["files_sha256"][name] for name, path in preserved.items()):
        raise ValueError("Original model/prompt/dependency profile changed; version the protocol explicitly")
    splits = {}
    for name, path in (("main", DEFAULT_DATASET), ("supplement", SUPPLEMENT_DATASET)):
        items, _ = load_items(path)
        splits[name] = {"path": path.relative_to(ROOT).as_posix(), "sha256": file_hash(path),
                        "items": len(items), "requests_per_model": len(items) * 8}
    return {"package": "misleadqa-release-v100-a40-v1", "protocol": PROTOCOL_VERSION,
            "dataset_version": CURRENT_VERSION, "schema_version": CURRENT_SCHEMA,
            "selected_models": list(load_models()), "splits": splits,
            "preserved_handoff_files": [p.relative_to(ROOT).as_posix() for p in preserved.values()],
            "historical_handoff_paths": {name: p.relative_to(ROOT).as_posix() for name, p in preserved.items()},
            "prompt_preserved_byte_for_byte": True,
            "source_module": original["source_module"], "source_files_sha256": original["source_files_sha256"],
            "files_sha256": source_hashes()}


def check_manifest():
    expected = current_manifest()
    stored = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
    if stored != expected:
        raise ValueError("Application manifest is stale; run python -B experiments/main_eval/refresh_manifest.py")
    return expected


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        check_manifest()
        print("Application manifest verified")
    else:
        save_json(HERE / "manifest.json", current_manifest())
        print("Application manifest refreshed; original handoff manifest preserved")
