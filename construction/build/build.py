"""Export a validated existing candidate; historical construction is pending."""

import argparse
from pathlib import Path
import shutil

from construction.screening.checks import validate_dataset
from misleadqa_utils import DEFAULT_DATASET, sha256_file

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, help="New candidate path; existing files are never overwritten")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    if args.status:
        print("Existing-candidate export: available\nConstruction stages: pending")
        return 0
    if not args.output:
        parser.error("--output is required for candidate export")
    if args.output.exists():
        parser.error("Output already exists; choose a new candidate path")
    report = validate_dataset(args.input)
    if report["status"] != "pass":
        parser.error(f"Input failed validation with {len(report['errors'])} errors")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open("rb") as source, args.output.open("xb") as target:
        shutil.copyfileobj(source, target)
    digest = sha256_file(args.output)
    print(f"Exported {report['records']} unchanged records to {args.output}")
    print(f"SHA-256: {digest}\nStatus: unchanged candidate export; source version retained")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

