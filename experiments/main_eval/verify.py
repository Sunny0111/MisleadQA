"""Read-only verification of a complete eight-condition inference run."""
import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "experiments.main_eval"

from .backend import verify_generation
from .protocol import (CONDITION_IDS, LETTERS, PROTOCOL_VERSION, canonical, digest, file_hash,
                       read_jsonl, load_items, make_requests)
from misleadqa_utils import CURRENT_VERSION, DEFAULT_DATASET, SUPPLEMENT_DATASET
from .run import read_checkpoint, validate_checkpoint
from .vllm_backend import sampling_options, validate_engine


def verify_run(directory, dataset=None):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    identity = manifest["identity"]
    if identity.get("dataset_version", CURRENT_VERSION) != CURRENT_VERSION:
        raise ValueError("Dataset version differs from the current release")
    if identity["dry_run"] or identity["protocol"] != PROTOCOL_VERSION:
        raise ValueError("Expected a real inference run under the eight-condition protocol")
    if tuple(identity["conditions"]) != CONDITION_IDS:
        raise ValueError("Condition list differs from protocol")
    request_path = directory / "requests.jsonl"
    if file_hash(request_path) != manifest["requests_file_sha256"]:
        raise ValueError("Requests file hash mismatch")
    requests = read_jsonl(request_path)
    if digest(canonical(requests)) != identity["requests_sha256"]:
        raise ValueError("Request content hash mismatch")
    if dataset is None:
        candidates = [Path(manifest["dataset_path"]), DEFAULT_DATASET, SUPPLEMENT_DATASET]
        dataset = next((p for p in candidates if p.is_file() and file_hash(p) == identity["dataset_sha256"]), None)
        if dataset is None:
            raise ValueError("Matching dataset unavailable; supply --dataset")
    if file_hash(dataset) != identity["dataset_sha256"]:
        raise ValueError("Dataset hash differs from run manifest")
    items, counts = load_items(dataset, identity["selection"], identity["limit"], identity["settings"]["seed"])
    if canonical(counts) != canonical(identity["counts"]) or canonical(make_requests(items)) != canonical(requests):
        raise ValueError("Saved requests differ from the dataset and prompt protocol")
    selected = identity["selected_item_ids"]
    if len(selected) != len(set(selected)) or len(selected) != identity["counts"]["selected_items"]:
        raise ValueError("Invalid selected-item manifest")
    expected = {(item_id, condition) for item_id in selected for condition in CONDITION_IDS}
    if (len(requests) != len(expected) or len(requests) != identity["expected_responses"]
            or {(r["item_id"], r["condition_id"]) for r in requests} != expected
            or len({r["request_id"] for r in requests}) != len(requests)):
        raise ValueError("Expected exactly eight distinct conditions per selected item")
    for request in requests:
        if digest(canonical(request["messages"])) != request["prompt_sha256"]:
            raise ValueError("Message hash mismatch")
        if len(request["messages"]) != 1 or request["messages"][0]["role"] != "user":
            raise ValueError("Each condition must be an independent single-turn request")
        key = request["answer_key"]
        if set(key) != {"gold", "lure", "abstain"} or len(set(key.values())) != 3:
            raise ValueError("Invalid scoring labels")
        if any(label not in tuple(LETTERS) for label in key.values()):
            raise ValueError("Unexpected scoring label")
    metadata = json.loads((directory / "model.json").read_text(encoding="utf-8"))
    spec = identity["model"]
    if metadata["model_id"] != spec["model_id"] or metadata["revision"] != spec["revision"]:
        raise ValueError("Model revision differs from manifest")
    choices = metadata["choice_token_ids"]
    eos = metadata["eos_token_ids"]
    if (set(choices) != set(LETTERS) or len(set(choices.values())) != 5
            or any(type(i) is not int for i in choices.values()) or not eos
            or any(type(i) is not int for i in eos) or set(choices.values()) & set(eos)):
        raise ValueError("Invalid choice/EOS token metadata")
    generation = metadata["generation_config"]
    selected_backend = identity.get("backend", "transformers")
    if (metadata.get("backend", "transformers") != selected_backend
            or metadata["dtype"] != "bfloat16" or metadata["quantization"] != spec.get("weight_quantization")
            or metadata["chat_template_kwargs"] != spec["chat_template_kwargs"]
            or generation["do_sample"] is not False or generation["max_new_tokens"] != 2
            or generation["num_beams"] != 1 or generation["num_return_sequences"] != 1):
        raise ValueError("Model decoding protocol differs from manifest")
    if selected_backend == "vllm":
        validate_engine(identity["engine"], spec, identity["settings"])
        if (canonical(metadata["engine"]) != canonical(identity["engine"])
                or canonical(metadata["effective_sampling"]) != canonical(sampling_options(identity["settings"], eos))):
            raise ValueError("vLLM engine/sampling metadata differ from manifest")
    rows = read_checkpoint(directory / "raw_responses.jsonl", repair=False)
    if len(validate_checkpoint(rows, requests, metadata)) != len(requests):
        raise ValueError(f"Incomplete run: {len(rows)}/{len(requests)} responses")
    for row in rows:
        verify_generation(row, metadata)
        if not 0 < row["input_tokens"] <= identity["settings"]["max_input_tokens"]:
            raise ValueError("Empty or oversized model input")
    return manifest, metadata, rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--dataset", type=Path)
    args = parser.parse_args()
    _, _, rows = verify_run(args.run_directory, args.dataset)
    print(f"Verified {len(rows)} responses; {len(rows) // 8} complete eight-condition items.")
