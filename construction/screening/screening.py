"""Export LLM screening requests without claiming completed screening."""

import argparse
from pathlib import Path

from misleadqa_utils import DEFAULT_DATASET, read_jsonl, write_jsonl

SCREENING_INSTRUCTION = (
    "Review the question, target and alternate scopes, choices, and G/F/S evidence. "
    "Check answer uniqueness, scope applicability, factual support, false-edit "
    "locality, distractor leakage, and fidelity to the provided source material. "
    "Flag missing raw source material rather than asserting independent verification. "
    "Return an object with item_id, status (pass/flag), flags, and rationale."
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists; choose a new path")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    items = read_jsonl(args.dataset)
    if args.limit:
        items = items[:args.limit]
    requests = [{"item_id": item["id"], "status": "not_run",
                 "instruction": SCREENING_INSTRUCTION, "item": item,
                 "screening_model": None, "raw_response": None} for item in items]
    write_jsonl(args.output, requests)
    print(f"Exported {len(requests)} pending screening requests to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

