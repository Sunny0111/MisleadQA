"""Descriptive paired robustness analysis with explicit denominators and all five outcomes."""
import argparse
from collections import Counter
import csv
from pathlib import Path
import statistics as stats
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    __package__ = "experiments.main_eval.supplementary"

from experiments.analysis.metrics import OUTCOMES, item_results, statistics
from experiments.main_eval.protocol import CONDITION_IDS, DEFAULT_DATASET, ROOT, file_hash, save_json
from experiments.main_eval.verify import verify_run as verify_baseline
from .verify import verify_run
from .dataset_identity import dataset_provenance
from misleadqa_utils import OUTPUTS_DIR


def paired_results(reference, treatment):
    """Keep the denominator fixed within every comparison, including invalid outputs."""
    left = {r["item_id"]: r for r in reference}
    right = {r["item_id"]: r for r in treatment}
    if left.keys() != right.keys():
        raise ValueError("Paired runs must contain identical item IDs")
    cohorts = {"all": sorted(left),
               "reference_B": [k for k in sorted(left) if left[k]["B_m"]],
               "reference_A": [k for k in sorted(left) if left[k]["A_m"]],
               "common_B": [k for k in sorted(left) if left[k]["B_m"] and right[k]["B_m"]],
               "common_A": [k for k in sorted(left) if left[k]["A_m"] and right[k]["A_m"]]}
    differences, transitions = [], []
    for name, ids in cohorts.items():
        for condition in CONDITION_IDS:
            for outcome in OUTCOMES:
                a = sum(left[k]["outcomes"][condition][outcome] for k in ids)
                b = sum(right[k]["outcomes"][condition][outcome] for k in ids)
                differences.append({"cohort": name, "condition": condition, "outcome": outcome,
                                    "n": len(ids), "reference_count": a, "treatment_count": b,
                                    "reference_rate": a / len(ids) if ids else None,
                                    "treatment_rate": b / len(ids) if ids else None,
                                    "delta_pp": 100 * (b - a) / len(ids) if ids else None})
            counts = Counter((next(o for o in OUTCOMES if left[k]["outcomes"][condition][o]),
                              next(o for o in OUTCOMES if right[k]["outcomes"][condition][o])) for k in ids)
            transitions.extend({"cohort": name, "condition": condition, "from": a, "to": b, "count": n}
                               for (a, b), n in sorted(counts.items()))
    # LR uses common B; DR, resistance, and order use common A.
    common_b, common_a = set(cohorts["common_B"]), set(cohorts["common_A"])
    fixed = lambda rows: statistics([r | {"B_m": r["item_id"] in common_b,
                                         "A_m": r["item_id"] in common_a} for r in rows])
    return {"cohort_sizes": {k: len(v) for k, v in cohorts.items()},
            "differences": differences, "transitions": transitions,
            "reference_common_cohorts": fixed(reference), "treatment_common_cohorts": fixed(treatment)}


def check_pair(left, right, dataset=None):
    a, am, _ = left
    b, bm, _ = right
    ai, bi = a["identity"], b["identity"]
    for key in ("model_alias", "selected_item_ids", "backend", "engine"):
        if ai.get(key) != bi.get(key):
            raise ValueError(f"Paired run mismatch: {key}")
    if ai["dataset_sha256"] != bi["dataset_sha256"]:
        if dataset is None:
            raise ValueError("Paired run mismatch: dataset_sha256")
        dataset_provenance(dataset, ai["dataset_sha256"])
        dataset_provenance(dataset, bi["dataset_sha256"])
    for key in ("model_id", "revision", "dtype", "quantization", "tokenizer_sha256", "asset_sha256",
                "chat_template_sha256", "runtime", "packages", "choice_token_ids", "eos_token_ids"):
        if am.get(key) != bm.get(key):
            raise ValueError(f"Paired model/runtime mismatch: {key}; do not silently merge changed environments")
    for key in ("seed", "dtype", "batch_size", "max_input_tokens", "max_new_tokens"):
        if ai["settings"].get(key) != bi["settings"].get(key):
            raise ValueError(f"Context baseline setting changed: {key}")
    if am["generation_config"] != bm["generation_config"] or am["chat_template_kwargs"] != bm["chat_template_kwargs"]:
        raise ValueError("Context comparison requires identical decoding and chat template settings")
    if any(bi["code_sha256"].get(k) != v for k, v in ai["code_sha256"].items()):
        raise ValueError("Frozen baseline code changed")


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def diagnostics(rows):
    values = [r["output_tokens"] for r in rows]
    return {"responses": len(rows), "parse_status": {"valid": len(rows)},
            "mean_output_tokens": stats.mean(values), "max_output_tokens": max(values),
            "generation_seconds": sum(r["elapsed_seconds"] for r in rows)}


def analyze(root, output, prefix, dataset, baseline_root=None, baseline_run_id="main-v100-full-v1", kind="context"):
    if kind != "context":
        raise ValueError("Only context-faithful comparisons are supported")
    if baseline_root is None:
        raise ValueError("Context comparison requires --baseline-root pointing to first-round outputs")
    summaries, pairs, provenance, csv_outcomes, csv_differences = {}, {}, [], [], []
    for path in sorted(Path(root).glob(f"*/{prefix}-*/manifest.json")):
        loaded = verify_run(path.parent, dataset)
        identity = loaded[0]["identity"]
        model, seed = identity["model_alias"], identity["settings"]["seed"]
        name = f"{model}/context-faithful/seed-{seed}"
        if name in summaries:
            raise ValueError(f"Duplicate model/variant/seed: {name}")
        directory = Path(baseline_root) / model / baseline_run_id
        reference = verify_baseline(directory, dataset)
        check_pair(reference, loaded, dataset=dataset)
        for source, run, key in ((path.parent, loaded, "directory"), (directory, reference, "baseline_directory")):
            provenance.append({key: str(source.resolve()), "identity": run[0]["identity"],
                "dataset_verification": dataset_provenance(dataset, run[0]["identity"]["dataset_sha256"]),
                "manifest_sha256": file_hash(source / "manifest.json"),
                "raw_sha256": file_hash(source / "raw_responses.jsonl"),
                "model_sha256": file_hash(source / "model.json")})
        items = item_results(loaded[2])
        summary = statistics(items)
        summaries[name] = {"statistics": summary, "diagnostics": diagnostics(loaded[2])}
        csv_outcomes.extend({"run": name, "metric": key, **value} for key, value in summary["metrics"].items())
        paired = paired_results(item_results(reference[2]), items)
        comparison = "context-faithful-minus-original-greedy"
        pairs[f"{model}/{comparison}/seed-{seed}"] = paired
        csv_differences.extend({"model": model, "comparison": comparison, "seed": seed, **row}
                               for row in paired["differences"])
    if not summaries:
        raise ValueError("No matching complete runs; check root and run prefix")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for filename, value in (("statistics.json", summaries), ("comparisons.json", pairs), ("provenance.json", provenance)):
        save_json(output / filename, value)
    write_csv(output / "outcomes.csv", csv_outcomes)
    write_csv(output / "differences.csv", csv_differences)
    print(f"Verified and analyzed {len(summaries)} context-faithful comparisons: {output.resolve()}")
    return summaries, pairs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=OUTPUTS_DIR / "supplementary")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prefix", default="main-robust-full-v1")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--baseline-root", type=Path, default=ROOT / "experiments/outputs/main_eval")
    parser.add_argument("--baseline-run-id", default="main-v100-full-v1")
    parser.add_argument("--kind", choices=("context",), default="context")
    args = parser.parse_args(argv)
    analyze(args.root, args.output, args.prefix, args.dataset, args.baseline_root, args.baseline_run_id, args.kind)


if __name__ == "__main__":
    main()
