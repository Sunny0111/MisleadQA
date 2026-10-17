"""Prepare or run the full eight-condition design, with strict resumption."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "experiments.main_eval"

from .backend import TransformersBackend, package_versions, verify_generation
from .protocol import (DEFAULT_DATASET, HERE, PROTOCOL_VERSION, CONDITION_IDS, canonical, digest, file_hash,
                      load_items, load_models, make_requests, save_json, source_hashes, write_jsonl)
from .vllm_backend import VLLMBackend, load_engine
from misleadqa_utils import CURRENT_VERSION, OUTPUTS_DIR


@contextmanager
def run_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ".lock"
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(f"Run is locked: {path}. Remove only after confirming its process has stopped") from None
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(f"pid={os.getpid()}\n")
        yield
    finally:
        path.unlink()


def read_checkpoint(path, repair=False):
    """Only the runner may repair a torn final append while holding the lock."""
    if not path.exists():
        return []
    data = path.read_bytes()
    rows, offset = [], 0
    lines = data.splitlines(keepends=True)
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except (ValueError, UnicodeError):
            if not repair or index != len(lines) - 1 or line.endswith(b"\n"):
                raise ValueError(f"Corrupt response line {index + 1}: {path}") from None
            with path.open("r+b") as stream:
                stream.truncate(offset)
            print("Recovered torn final response append", file=sys.stderr)
            break
        if not isinstance(row, dict):
            raise ValueError("Response JSONL record must be an object")
        rows.append(row)
        offset += len(line)
    else:
        if repair and data and not data.endswith(b"\n"):
            with path.open("ab") as stream:
                stream.write(b"\n")
    return rows


def validate_checkpoint(rows, requests, metadata=None):
    expected = {r["request_id"]: r for r in requests}
    done = set()
    for row in rows:
        key = row.get("request_id")
        if key not in expected or key in done or any(row.get(k) != v for k, v in expected[key].items()):
            raise ValueError("Duplicate, unknown, or altered checkpoint request")
        if metadata is not None:
            verify_generation(row, metadata)
        done.add(key)
    return done


def execute(requests, output, factory):
    path = output / "raw_responses.jsonl"
    rows = read_checkpoint(path, repair=True)
    metadata_path = output / "model.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else None
    if rows and metadata is None:
        raise ValueError("Responses exist without model metadata")
    done = validate_checkpoint(rows, requests, metadata)
    save_json(output / "progress.json", {"completed": len(done), "total": len(requests),
                                        "complete": len(done) == len(requests)})
    if len(done) == len(requests):
        print(f"Already complete: {len(done)} responses. No model loaded.")
        return
    backend = factory()
    if metadata is not None and canonical(metadata) != canonical(backend.metadata):
        raise ValueError("Model/tokenizer/software/hardware changed; use a new run id")
    if metadata is None:
        save_json(metadata_path, backend.metadata)
    backend.preflight(requests)
    # One standalone user turn per call. Only messages, never answer keys, reach the model.
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
            save_json(output / "progress.json", {"completed": len(done), "total": len(requests),
                                                "complete": len(done) == len(requests)})
            print(f"{len(done)}/{len(requests)} {request['item_id']} {request['condition_id']}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model", required=True, choices=tuple(load_models()))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--selection", choices=("eligible", "all"), default="all")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--output-root", type=Path, default=OUTPUTS_DIR / "main_eval")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--backend", choices=("auto", "transformers", "vllm"), default="auto")
    parser.add_argument("--engine-config", type=Path)
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}", args.run_id):
        parser.error("run-id must be a simple directory name")
    if args.max_input_tokens < 1:
        parser.error("max-input-tokens must be positive")
    items, counts = load_items(args.dataset, args.selection, args.limit, args.seed)
    requests = make_requests(items)
    spec = load_models()[args.model]
    settings = {"seed": args.seed, "max_input_tokens": args.max_input_tokens,
                "dtype": "bfloat16", "decoding": "greedy canonical ABCDE token then EOS",
                "max_new_tokens": 2, "batch_size": 1}
    selected_backend = spec.get("required_backend", "transformers") if args.backend == "auto" else args.backend
    if spec.get("required_backend") and selected_backend != spec["required_backend"]:
        parser.error(f"{args.model} requires {spec['required_backend']}")
    if selected_backend == "transformers" and args.engine_config is not None:
        parser.error("engine-config only applies to vLLM")
    engine = load_engine(spec, settings, args.engine_config) if selected_backend == "vllm" else None
    identity = {
        "protocol": PROTOCOL_VERSION, "dataset_version": CURRENT_VERSION,
        "dataset_sha256": file_hash(args.dataset),
        "model_alias": args.model, "model": spec, "selection": args.selection,
        "limit": args.limit, "counts": counts, "selected_item_ids": [r["id"] for r in items],
        "conditions": CONDITION_IDS, "expected_responses": len(requests),
        "requests_sha256": digest(canonical(requests)), "settings": settings,
        "backend": selected_backend, "engine": engine,
        "dry_run": args.dry_run, "code_sha256": source_hashes(),
        "packages": {} if args.dry_run else package_versions(),
    }
    output = args.output_root.resolve() / args.model / args.run_id
    with run_lock(output):
        manifest_path = output / "manifest.json"
        if manifest_path.exists():
            old = json.loads(manifest_path.read_text(encoding="utf-8"))
            if canonical(old["identity"]) != canonical(identity):
                raise ValueError("Dataset/protocol/model/code/software changed; use a new run id")
            request_path = output / "requests.jsonl"
            if file_hash(request_path) != old["requests_file_sha256"]:
                raise ValueError("Saved requests file changed")
        else:
            if any(output.iterdir()) and set(p.name for p in output.iterdir()) != {".lock"}:
                raise ValueError("Nonempty run directory without a manifest; use a new run id")
            write_jsonl(output / "requests.jsonl", requests)
            save_json(manifest_path, {"created_utc": datetime.now(timezone.utc).isoformat(),
                      "dataset_path": str(args.dataset.resolve()), "identity": identity,
                      "requests_file_sha256": file_hash(output / "requests.jsonl")})
            if not args.dry_run:
                freeze = subprocess_freeze()
                (output / "requirements-frozen.txt").write_text(freeze, encoding="utf-8")
        print(f"{counts['selected_items']} items x 8 conditions = {len(requests)} requests; {output}")
        if args.dry_run:
            print("Dry run: prompts and manifest saved; no weights loaded and no answers generated.")
        else:
            factory = (lambda: VLLMBackend(spec, settings, engine, args.local_files_only)) if selected_backend == "vllm" else (
                       lambda: TransformersBackend(spec, settings, args.local_files_only))
            execute(requests, output, factory)
    return output


def subprocess_freeze():
    import subprocess
    return subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)


if __name__ == "__main__":
    main()
