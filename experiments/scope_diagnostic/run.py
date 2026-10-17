"""Run or verify scope judgments without rerunning closed-book or S-only answers."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "experiments.scope_diagnostic"

from experiments.main_eval.backend import package_versions
from experiments.main_eval.protocol import (DEFAULT_DATASET, canonical, digest, file_hash, load_models,
                                 read_jsonl, save_json, write_jsonl)
from experiments.main_eval.run import read_checkpoint, run_lock, subprocess_freeze, validate_checkpoint
from experiments.main_eval.vllm_backend import load_engine, sampling_options
from .backend import LabelConstraint, ScopeBackend, verify_generation
from .protocol import HERE, VERSION, SETTINGS, make_requests, sample_items, source_hashes
from misleadqa_utils import CURRENT_VERSION, OUTPUTS_DIR

SEALED_FILES = ("manifest.json", "model.json", "requests.jsonl", "raw_responses.jsonl", "requirements-frozen.txt")


def execute(requests, output, factory):
    path = output / "raw_responses.jsonl"
    rows = read_checkpoint(path, repair=True)
    model_path = output / "model.json"
    old = json.loads(model_path.read_text(encoding="utf-8")) if model_path.exists() else None
    if rows and old is None:
        raise ValueError("Responses exist without model metadata")
    done = validate_checkpoint(rows, requests)
    for row in rows:
        verify_generation(row, old)

    def progress():
        save_json(output / "progress.json", {"completed": len(done), "total": len(requests),
                                             "complete": len(done) == len(requests)})

    progress()
    if len(done) == len(requests):
        print(f"Already complete: {len(done)} judgments; no model loaded")
        return
    backend = factory()
    if old is not None and canonical(old) != canonical(backend.metadata):
        raise ValueError("Model/tokenizer/software/hardware changed; use a new run id")
    if old is None:
        save_json(model_path, backend.metadata)
    backend.preflight(requests)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        for request in requests:
            if request["request_id"] in done:
                continue
            start = time.monotonic()
            result = backend.generate(request["messages"])
            verify_generation(result, backend.metadata)
            record = request | result | {"elapsed_seconds": time.monotonic() - start}
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            done.add(request["request_id"])
            progress()
            print(f"{len(done)}/{len(requests)} {request['item_id']} {request['condition_id']} {result['response']}", flush=True)


def verify_run(directory, dataset=DEFAULT_DATASET):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    identity = manifest["identity"]
    if identity["dry_run"] or identity["protocol"] != VERSION or identity["settings"] != SETTINGS:
        raise ValueError("Expected a real scope diagnosis with the original greedy settings")
    items, sample = sample_items(dataset, identity["smoke"])
    requests = make_requests(items)
    spec = load_models()[identity["model_alias"]]
    backend = spec.get("required_backend", "transformers")
    engine = load_engine(spec, SETTINGS) if backend == "vllm" else None
    if (identity["model"] != spec or identity["backend"] != backend or identity["engine"] != engine
            or identity["dataset_sha256"] != file_hash(dataset)
            or identity["sample_sha256"] != digest(canonical(sample))
            or identity["selected_item_ids"] != [i["id"] for i in items]
            or identity["expected_responses"] != len(requests)
            or identity["requests_sha256"] != digest(canonical(requests))
            or file_hash(directory / "requests.jsonl") != manifest["requests_file_sha256"]
            or canonical(read_jsonl(directory / "requests.jsonl")) != canonical(requests)):
        raise ValueError("Model, sample, dataset or prompts differ from the frozen protocol")
    seal = json.loads((directory / "complete.json").read_text(encoding="utf-8"))
    if seal != {name: file_hash(directory / name) for name in SEALED_FILES}:
        raise ValueError("Completed run was modified")
    metadata = json.loads((directory / "model.json").read_text(encoding="utf-8"))
    constraint = LabelConstraint(metadata["label_token_ids"], metadata["eos_token_ids"])
    generation = metadata["generation_config"]
    if (metadata["model_id"] != spec["model_id"] or metadata["revision"] != spec["revision"]
            or metadata["backend"] != backend or metadata["dtype"] != "bfloat16"
            or metadata["quantization"] != spec.get("weight_quantization")
            or metadata["chat_template_kwargs"] != spec["chat_template_kwargs"]
            or metadata["output_constraint"] != "greedy label-token trie then EOS"
            or generation["do_sample"] is not False or generation["num_beams"] != 1
            or generation["num_return_sequences"] != 1 or generation["max_new_tokens"] != constraint.budget):
        raise ValueError("Scope model/decoding metadata differ from the protocol")
    if backend == "vllm" and (metadata["engine"] != engine or metadata["effective_sampling"] !=
            sampling_options(SETTINGS, metadata["eos_token_ids"]) | {"max_tokens": constraint.budget}):
        raise ValueError("vLLM engine/sampling metadata differ from the protocol")
    rows = read_checkpoint(directory / "raw_responses.jsonl")
    if len(validate_checkpoint(rows, requests)) != len(requests):
        raise ValueError(f"Incomplete judgments: {len(rows)}/{len(requests)}")
    for row in rows:
        verify_generation(row, metadata)
        if not 0 < row["input_tokens"] <= SETTINGS["max_input_tokens"]:
            raise ValueError("Empty or oversized scope input")
    return manifest, metadata, rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", type=Path, metavar="RUN_DIRECTORY")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model", choices=tuple(load_models()))
    parser.add_argument("--run-id", default="main-scope-full-v2")
    parser.add_argument("--output-root", type=Path, default=OUTPUTS_DIR / "scope_diagnostic")
    parser.add_argument("--smoke", action="store_true", help="One frozen sampled item per group; 15 calls/model")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args(argv)
    if args.verify:
        _, _, rows = verify_run(args.verify, args.dataset)
        print(f"Verified {len(rows)} independent scope judgments")
        return args.verify
    if not args.model or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}", args.run_id):
        parser.error("Provide --model and a simple --run-id")
    items, sample = sample_items(args.dataset, args.smoke)
    requests = make_requests(items)
    spec = load_models()[args.model]
    backend = spec.get("required_backend", "transformers")
    engine = load_engine(spec, SETTINGS) if backend == "vllm" else None
    identity = {"protocol": VERSION, "dataset_version": CURRENT_VERSION,
                "model_alias": args.model, "model": spec, "settings": SETTINGS,
                "dataset_sha256": file_hash(args.dataset), "sample_sha256": digest(canonical(sample)),
                "selected_item_ids": [i["id"] for i in items], "expected_responses": len(requests),
                "requests_sha256": digest(canonical(requests)), "backend": backend, "engine": engine,
                "smoke": args.smoke, "dry_run": args.dry_run, "code_sha256": source_hashes(),
                "packages": {} if args.dry_run else package_versions()}
    output = args.output_root.resolve() / args.model / args.run_id
    with run_lock(output):
        path = output / "manifest.json"
        if path.exists():
            manifest = json.loads(path.read_text(encoding="utf-8"))
            if canonical(manifest["identity"]) != canonical(identity):
                raise ValueError("Data/protocol/model/code/software changed; use a new run id")
            if file_hash(output / "requests.jsonl") != manifest["requests_file_sha256"]:
                raise ValueError("Saved requests changed")
        else:
            if {p.name for p in output.iterdir()} != {".lock"}:
                raise ValueError("Nonempty run directory without a manifest")
            write_jsonl(output / "requests.jsonl", requests)
            save_json(path, {"created_utc": datetime.now(timezone.utc).isoformat(), "identity": identity,
                            "requests_file_sha256": file_hash(output / "requests.jsonl")})
            if not args.dry_run:
                (output / "requirements-frozen.txt").write_text(subprocess_freeze(), encoding="utf-8")
        print(f"{len(items)} items x 3 = {len(requests)} scope calls; {output}")
        if not args.dry_run:
            seal_path = output / "complete.json"
            if seal_path.exists():
                verify_run(output, args.dataset)  # Never repair or reseal a modified complete run.
            else:
                execute(requests, output, lambda: ScopeBackend(spec, SETTINGS, engine, args.local_files_only))
                save_json(seal_path, {name: file_hash(output / name) for name in SEALED_FILES})
                verify_run(output, args.dataset)
    return output


if __name__ == "__main__":
    main()
