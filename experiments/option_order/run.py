"""Run option permutations or add F-only responses to a completed three-order run."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import signal
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "experiments.option_order"

from experiments.main_eval.backend import TransformersBackend, package_versions
from experiments.main_eval.protocol import (
    DEFAULT_DATASET, canonical, digest, file_hash, load_models, read_jsonl, save_json, write_jsonl,
)
from experiments.main_eval.run import execute, read_checkpoint, run_lock, subprocess_freeze, validate_checkpoint
from experiments.main_eval.vllm_backend import VLLMBackend
from misleadqa_utils import OUTPUTS_DIR
from .protocol import REFERENCE_ROOT, load_reference, make_plan, source_hashes

SEALED_FILES = ("manifest.json", "baseline.json", "model.json", "requests.jsonl",
                "raw_responses.jsonl", "requirements-frozen.txt")


def sealed_files(identity):
    return SEALED_FILES + (("paired_baseline.json",) if identity["conditions"] == ["f_only"] else ())


def make_paired_plan(reference, paired, dataset=DEFAULT_DATASET, smoke=False):
    """Keep the complete closed-book/S run as the immutable pairing reference."""
    requests, expected = make_plan(reference, dataset, rerun_original=True)
    identity, metadata = paired["identity"], paired["model_metadata"]
    if identity["dry_run"] or any(identity.get(k) != v for k, v in expected.items()):
        raise ValueError("F-only requires a complete fresh three-order closed-book/S-only run")
    check_metadata(metadata, reference["model_metadata"], rerun_original=True)
    if len(validate_checkpoint(paired["responses"], requests, metadata)) != len(requests):
        raise ValueError("Incomplete paired baseline")
    if any(not 0 < row["input_tokens"] <= identity["settings"]["max_input_tokens"]
           for row in paired["responses"]):
        raise ValueError("Invalid paired baseline input length")
    requests, identity = make_plan(reference, dataset, smoke, True, ("f_only",))
    identity["paired_baseline_sha256"] = digest(canonical(paired))
    return requests, identity


def check_metadata(actual, expected, rerun_original=False):
    # A new host requires regenerating all orders. Checkpoints, templates and decoding stay fixed.
    allowed = {"runtime", "packages", "device_map"} if rerun_original else set()
    different = sorted(key for key in (actual.keys() | expected.keys()) - allowed
                       if actual.get(key) != expected.get(key))
    if different:
        raise ValueError("Original model/runtime metadata differs: " + ", ".join(different)
                         + ". For another environment, first generate a complete closed-book/S-only "
                         "baseline with --rerun-original. F-only must match that baseline.")


def verify_run(directory, dataset=DEFAULT_DATASET):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    identity = manifest["identity"]
    if identity["dry_run"]:
        raise ValueError("Dry-run requests are not inference results")
    reference = load_reference(directory / "baseline.json", dataset)
    fresh = identity.get("rerun_original", False)
    if type(fresh) is not bool:
        raise ValueError("rerun_original must be a boolean")
    paired = None
    if identity["conditions"] == ["f_only"]:
        paired = json.loads((directory / "paired_baseline.json").read_text(encoding="utf-8"))
        requests, expected = make_paired_plan(reference, paired, dataset, identity["smoke"])
    else:
        requests, expected = make_plan(reference, dataset, identity["smoke"], fresh)
    if any(identity.get(key) != value for key, value in expected.items()):
        raise ValueError("Model, reference, sample or settings differ from the frozen protocol")
    if (file_hash(directory / "requests.jsonl") != manifest["requests_file_sha256"]
            or canonical(read_jsonl(directory / "requests.jsonl")) != canonical(requests)):
        raise ValueError("Saved requests differ from the option-order protocol")
    seal = json.loads((directory / "complete.json").read_text(encoding="utf-8"))
    if seal != {name: file_hash(directory / name) for name in sealed_files(identity)}:
        raise ValueError("Completed run was modified")
    metadata = json.loads((directory / "model.json").read_text(encoding="utf-8"))
    check_metadata(metadata, paired["model_metadata"] if paired else reference["model_metadata"],
                   fresh and paired is None)
    rows = read_checkpoint(directory / "raw_responses.jsonl")
    if len(validate_checkpoint(rows, requests, metadata)) != len(requests):
        raise ValueError(f"Incomplete run: {len(rows)}/{len(requests)} responses")
    if any(not 0 < row["input_tokens"] <= identity["settings"]["max_input_tokens"] for row in rows):
        raise ValueError("Empty or oversized model input")
    return manifest, reference, rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", type=Path, metavar="RUN_DIRECTORY")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model", choices=tuple(load_models()))
    parser.add_argument("--reference-root", type=Path, default=REFERENCE_ROOT)
    parser.add_argument("--f-only-from", type=Path, metavar="RUN_DIRECTORY",
                        help="Add F-only in all three orders, reusing this complete fresh closed-book/S run")
    parser.add_argument("--run-id", help="Defaults to a distinct full, smoke or preview run name")
    parser.add_argument("--output-root", type=Path, default=OUTPUTS_DIR / "option_order")
    parser.add_argument("--smoke", action="store_true", help="Use five fixed items")
    parser.add_argument("--rerun-original", action="store_true",
                        help="Generate all three orders on this host; never reuse old answers in the cohort")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args(argv)
    if args.verify:
        _, _, rows = verify_run(args.verify, args.dataset)
        print(f"Verified {len(rows)} option-order responses")
        return args.verify
    paired = None
    if args.f_only_from:
        if args.rerun_original:
            parser.error("F-only already generates all three orders; omit --rerun-original")
        manifest, reference, rows = verify_run(args.f_only_from, args.dataset)
        alias = manifest["identity"]["model_alias"]
        if args.model is not None and args.model != alias:
            parser.error("--model must match --f-only-from")
        args.model = alias
        paired = {"identity": manifest["identity"], "responses": rows,
                  "model_metadata": json.loads((args.f_only_from / "model.json").read_text(encoding="utf-8")),
                  "source_files_sha256": {name: file_hash(args.f_only_from / name)
                                          for name in (*SEALED_FILES, "complete.json")}}
        requests, identity = make_paired_plan(reference, paired, args.dataset, args.smoke)
    mode = "preview" if args.dry_run else "smoke" if args.smoke else "full"
    suffix = "f-only-v1" if paired else "fresh-v1" if args.rerun_original else "v1"
    args.run_id = args.run_id or f"main-option-order-{mode}-{suffix}"
    if not args.model or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}", args.run_id):
        parser.error("Provide --model and a simple --run-id")
    if paired is None:
        reference = load_reference(args.reference_root / f"{args.model}.json", args.dataset)
        if reference["model_alias"] != args.model:
            raise ValueError("Reference belongs to another model")
        requests, identity = make_plan(reference, args.dataset, args.smoke, args.rerun_original)
    identity.update(dry_run=args.dry_run, code_sha256=source_hashes(),
                    packages={} if args.dry_run else package_versions())
    output = args.output_root.resolve() / args.model / args.run_id
    with run_lock(output):
        path = output / "manifest.json"
        if path.exists():
            manifest = json.loads(path.read_text(encoding="utf-8"))
            if canonical(manifest["identity"]) != canonical(identity):
                raise ValueError("Reference/data/protocol/model/code/software changed; use a new run id")
            if (file_hash(output / "requests.jsonl") != manifest["requests_file_sha256"]
                    or canonical(load_reference(output / "baseline.json", args.dataset)) != canonical(reference)):
                raise ValueError("Saved requests or reference changed")
            if paired is not None and canonical(json.loads((output / "paired_baseline.json").read_text(
                    encoding="utf-8"))) != canonical(paired):
                raise ValueError("Saved paired baseline changed")
        else:
            if {p.name for p in output.iterdir()} != {".lock"}:
                raise ValueError("Nonempty run directory without a manifest")
            write_jsonl(output / "requests.jsonl", requests)
            save_json(output / "baseline.json", reference)
            if paired is not None:
                save_json(output / "paired_baseline.json", paired)
            save_json(path, {"created_utc": datetime.now(timezone.utc).isoformat(), "identity": identity,
                            "requests_file_sha256": file_hash(output / "requests.jsonl")})
        requirements = output / "requirements-frozen.txt"
        if not args.dry_run and not requirements.exists():
            requirements.write_text(subprocess_freeze(), encoding="utf-8")
        print(f"{len(identity['selected_item_ids'])} items x {len(identity['permutations'])} orders "
              f"x {len(identity['conditions'])} conditions = {len(requests)} new calls; {output}", flush=True)
        if not args.dry_run:
            def factory():
                spec, settings = identity["model"], identity["settings"]
                backend = (VLLMBackend(spec, settings, identity["engine"], args.local_files_only)
                           if identity["backend"] == "vllm"
                           else TransformersBackend(spec, settings, args.local_files_only))
                check_metadata(backend.metadata, paired["model_metadata"] if paired else reference["model_metadata"],
                               args.rerun_original)
                return backend

            seal_path = output / "complete.json"
            if not seal_path.exists():
                execute(requests, output, factory)
                save_json(seal_path, {name: file_hash(output / name) for name in sealed_files(identity)})
            verify_run(output, args.dataset)
    return output


if __name__ == "__main__":
    def stop_on_term(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop_on_term)
    main()
