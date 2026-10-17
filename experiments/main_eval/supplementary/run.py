"""Run an context-faithful protocol with immutable manifests and resumable first responses."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    __package__ = "experiments.main_eval.supplementary"

from experiments.main_eval.backend import TransformersBackend, package_versions
from experiments.main_eval.protocol import (CONDITION_IDS, DEFAULT_DATASET, canonical, digest, file_hash,
                                 load_items, load_models, save_json, write_jsonl)
from experiments.main_eval.run import read_checkpoint, run_lock, subprocess_freeze, validate_checkpoint
from experiments.main_eval.vllm_backend import VLLMBackend, load_engine
from misleadqa_utils import CURRENT_VERSION, OUTPUTS_DIR
from .protocol import (VERSION, VARIANTS, SELECTION_SEED, make_requests, model_for,
                       settings_for, source_hashes, validate_result)


def seal_run(output):
    files = ("manifest.json", "model.json", "requests.jsonl", "raw_responses.jsonl", "requirements-frozen.txt")
    hashes = {name: file_hash(output / name) for name in files}
    path = output / "complete.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != hashes:
        raise ValueError("A previously completed run was modified")
    save_json(path, hashes)


def execute(requests, output, factory, settings):
    path = output / "raw_responses.jsonl"
    rows = read_checkpoint(path, repair=True)
    model_path = output / "model.json"
    old = json.loads(model_path.read_text(encoding="utf-8")) if model_path.exists() else None
    if rows and old is None:
        raise ValueError("Responses exist without model metadata")
    done = validate_checkpoint(rows, requests)
    for row in rows:
        validate_result(row, old, settings)

    def progress():
        save_json(output / "progress.json", {"completed": len(done), "total": len(requests),
                  "complete": len(done) == len(requests)})

    progress()
    if len(done) == len(requests):
        print(f"Already complete: {len(done)} responses. No model loaded.")
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
            # Scoring labels and provenance never enter the generation call.
            result = backend.generate(request["messages"])
            validate_result(request | result, backend.metadata, settings)
            record = request | result | {"elapsed_seconds": time.monotonic() - start}
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            done.add(request["request_id"])
            progress()
            print(f"{len(done)}/{len(requests)} {request['item_id']} {request['condition_id']} "
                  f"{result.get('parse_status', 'valid')} tokens={result['output_tokens']}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", default="context-faithful", choices=VARIANTS)
    parser.add_argument("--model", required=True, choices=tuple(load_models()))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-root", type=Path, default=OUTPUTS_DIR / "supplementary")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}", args.run_id):
        parser.error("run-id must be a simple directory name")
    settings = settings_for(args.variant, args.seed)
    spec = model_for(args.model, args.variant)
    items, counts = load_items(args.dataset, "all", args.limit, SELECTION_SEED)
    requests = make_requests(items, settings)
    backend = spec.get("required_backend", "transformers")
    engine = load_engine(spec, settings) if backend == "vllm" else None
    identity = {"protocol": VERSION, "dataset_version": CURRENT_VERSION,
                "variant": args.variant, "model_alias": args.model, "model": spec,
                "dataset_sha256": file_hash(args.dataset), "selection": "all", "limit": args.limit,
                "counts": counts, "selected_item_ids": [i["id"] for i in items],
                "conditions": CONDITION_IDS, "expected_responses": len(requests),
                "settings": settings, "requests_sha256": digest(canonical(requests)),
                "backend": backend, "engine": engine, "code_sha256": source_hashes(),
                "packages": {} if args.dry_run else package_versions(), "dry_run": args.dry_run}
    output = args.output_root.resolve() / args.model / args.run_id
    with run_lock(output):
        manifest_path = output / "manifest.json"
        request_path = output / "requests.jsonl"
        if manifest_path.exists():
            old = json.loads(manifest_path.read_text(encoding="utf-8"))
            if canonical(old["identity"]) != canonical(identity):
                raise ValueError("Protocol/data/model/seed/budget/code/software changed; use a new run id")
            if file_hash(request_path) != old["requests_file_sha256"]:
                raise ValueError("Saved requests file changed")
        else:
            if {p.name for p in output.iterdir()} != {".lock"}:
                raise ValueError("Nonempty run directory without a manifest")
            write_jsonl(request_path, requests)
            save_json(manifest_path, {"created_utc": datetime.now(timezone.utc).isoformat(),
                      "dataset_path": str(args.dataset.resolve()), "identity": identity,
                      "requests_file_sha256": file_hash(request_path)})
            if not args.dry_run:
                (output / "requirements-frozen.txt").write_text(subprocess_freeze(), encoding="utf-8")
        print(f"{args.variant}: {len(items)} items x 8 = {len(requests)} requests; {output}")
        if not args.dry_run:
            if backend == "vllm":
                factory = lambda: VLLMBackend(spec, settings, engine, args.local_files_only)
            else:
                factory = lambda: TransformersBackend(spec, settings, args.local_files_only)
            execute(requests, output, factory, settings)
            seal_run(output)
    return output


if __name__ == "__main__":
    main()
