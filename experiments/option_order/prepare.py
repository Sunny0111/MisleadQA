"""Export verified original answers and optionally package the remote experiment."""
import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "experiments.option_order"

from experiments.main_eval.package import build_package, verify_bundle
from experiments.main_eval.protocol import (
    DEFAULT_DATASET, canonical, digest, file_hash, load_models, save_json,
)
from experiments.main_eval.verify import verify_run
from experiments.scope_diagnostic.protocol import sample_items
from misleadqa_utils import CURRENT_VERSION, OUTPUTS_DIR
from .protocol import CONDITIONS, REFERENCE_ROOT, VERSION, load_reference, validate_reference


def export_reference(directory, destination, dataset=DEFAULT_DATASET):
    manifest, metadata, rows = verify_run(directory, dataset)
    items, sample = sample_items(dataset)
    selected = {item["id"] for item in items}
    reference = {
        "protocol": VERSION, "dataset_version": CURRENT_VERSION,
        "model_alias": manifest["identity"]["model_alias"],
        "sample_sha256": digest(canonical(sample)), "baseline_identity": manifest["identity"],
        "model_metadata": metadata,
        "baseline_files_sha256": {name: file_hash(Path(directory) / name) for name in
                                  ("manifest.json", "model.json", "requests.jsonl", "raw_responses.jsonl")},
        "responses": [row for row in rows if row["item_id"] in selected and row["condition_id"] in CONDITIONS],
    }
    validate_reference(reference, dataset)
    destination = Path(destination)
    if destination.exists():
        if canonical(load_reference(destination, dataset)) != canonical(reference):
            raise ValueError(f"Reference exists with different content: {destination}")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        save_json(destination, reference)
    print(f"Verified reference: {reference['model_alias']}, {len(reference['responses'])} reused answers")
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, default=OUTPUTS_DIR / "main_eval")
    parser.add_argument("--baseline-run-id", default="main-v100-full-v1")
    parser.add_argument("--reference-root", type=Path, default=REFERENCE_ROOT)
    parser.add_argument("--package", type=Path, help="Bundle source plus the six verified baseline subsets")
    parser.add_argument("--verify-bundle", type=Path)
    args = parser.parse_args(argv)
    if args.verify_bundle:
        return verify_bundle(args.verify_bundle)
    references = {}
    for alias in load_models():
        path = export_reference(args.baseline_root / alias / args.baseline_run_id,
                                args.reference_root / f"{alias}.json")
        references[f"experiments/outputs/option_order_references/{alias}.json"] = path
    if args.package:
        build_package(args.package, extra_files=references)


if __name__ == "__main__":
    main()
