"""Verify release runs, join source clusters, and export paired analyses."""
import argparse
import copy
import csv
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "experiments.analysis"

from misleadqa_utils import CURRENT_VERSION, review_flag
from .metrics import item_results, paired_bootstrap, statistics
from experiments.main_eval.protocol import (DEFAULT_DATASET, canonical, digest, file_hash, load_items,
                       save_json, source_hashes, write_jsonl)
from experiments.main_eval.verify import verify_run


def attach_groups(items, dataset):
    by_id = {row["id"]: row for row in dataset}
    if len(by_id) != len(dataset):
        raise ValueError("Duplicate dataset IDs")
    results = copy.deepcopy(items)
    for item in results:
        row = by_id.get(item["item_id"])
        if row is None:
            raise ValueError("Response item is absent from dataset")
        grouping = row["grouping"]
        if not grouping.get("cluster_id"):
            raise ValueError("Missing source cluster")
        if (any(item[k] != row[k] for k in ("source", "dimension"))
                or item["analysis_group_id"] != grouping["analysis_group_id"]
                or item.get("evaluation_eligible") != review_flag(row, "evaluation_eligible")):
            raise ValueError("Response grouping metadata differs from dataset")
        item["cluster_id"] = grouping["cluster_id"]
    return results


def analyze_runs(directories, output, dataset=DEFAULT_DATASET, allow_mixed_runtimes=False,
                 replicates=10000, seed=2027, cluster_field="cluster_id", save_draws=False):
    output, dataset = Path(output), Path(dataset)
    if output.exists():
        raise ValueError("Analysis output already exists; choose a new directory")
    if not directories:
        raise ValueError("At least one run is required")
    dataset_rows, _ = load_items(dataset)
    loaded = [(Path(d), *verify_run(d, dataset=dataset)) for d in directories]
    reference = loaded[0][1]["identity"]
    fields = ("protocol", "dataset_sha256", "selection", "limit", "selected_item_ids",
              "requests_sha256", "settings", "code_sha256")
    if not allow_mixed_runtimes:
        fields += ("packages",)
        if len({m["identity"]["backend"] for _, m, _, _ in loaded}) != 1:
            raise ValueError("Mixed backends require --allow-mixed-runtimes")
    for directory, manifest, _, _ in loaded:
        if any(canonical(manifest["identity"][k]) != canonical(reference[k]) for k in fields):
            raise ValueError(f"Runs differ in dataset, prompts, settings, code or software: {directory}")
    aliases = [m["identity"]["model_alias"] for _, m, _, _ in loaded]
    if len(aliases) != len(set(aliases)):
        raise ValueError("Use one run per model in each analysis")
    models, summaries, csv_rows, contrast_rows, inputs = {}, [], [], [], []
    for directory, manifest, metadata, rows in sorted(loaded, key=lambda x: x[1]["identity"]["model_alias"]):
        alias = manifest["identity"]["model_alias"]
        items = attach_groups(item_results(rows), dataset_rows)
        models[alias] = items
        labels = {"model": alias, "backend": metadata["backend"],
                  "weight_precision": "AWQ 4-bit" if metadata["quantization"] == "awq_4bit" else "BF16"}
        cohorts = {"all_selected": items}
        for field in ("source", "dimension"):
            for value in sorted({r[field] for r in items}):
                cohorts[f"{field}:{value}"] = [r for r in items if r[field] == value]
        for cohort, subset in cohorts.items():
            summary = {**labels, "cohort": cohort, **statistics(subset)}
            summaries.append(summary)
            csv_rows.extend({**labels, "cohort": cohort, "metric": key, **value}
                            for key, value in summary["metrics"].items())
            contrast_rows.extend({**labels, "cohort": cohort, "contrast": key, **value}
                                 for key, value in summary["contrasts"].items())
        inputs.append({"run_directory": str(directory.resolve()), **labels,
                       "sha256": {name: file_hash(directory / name) for name in
                                  ("manifest.json", "model.json", "requests.jsonl", "raw_responses.jsonl")},
                       "runtime": metadata["runtime"]})
    bootstrap, draws = paired_bootstrap(models, replicates=replicates, seed=seed, cluster_field=cluster_field)
    output.mkdir(parents=True)
    write_jsonl(output / "item_metrics.jsonl", [{"model": name, **r} for name, items in models.items() for r in items])
    save_json(output / "statistics.json", summaries)
    save_json(output / "bootstrap.json", bootstrap)
    if save_draws:
        write_jsonl(output / "bootstrap_draws.jsonl", draws)
    for filename, records in (("statistics.csv", csv_rows), ("contrasts.csv", contrast_rows)):
        with (output / filename).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    save_json(output / "provenance.json", {
        "inputs": inputs, "dataset_version": CURRENT_VERSION,
        "dataset_sha256": file_hash(dataset), "dataset_path": str(dataset.resolve()),
        "allow_mixed_runtimes": allow_mixed_runtimes, "analysis_code_sha256": source_hashes(),
        "statistics_sha256": digest(canonical(summaries)), "bootstrap_sha256": digest(canonical(bootstrap)),
        "cluster_field": cluster_field, "replicates": replicates, "seed": seed,
        "sample_note": "limited smoke sample" if reference["limit"] else "full selected dataset",
    })
    lines = ["# MisleadQA release 1.0.0 results", "",
             "B_m: closed-book correct. A_m: both closed-book and G-only correct.",
             "LR: single-evidence lure rate on B_m. DR: mixed-order mean lure rate on A_m.",
             "Resistance: single-evidence minus mixed-order mean on the SAME A_m; positive means reduced lure selection.",
             "Order: (G then X) minus (X then G) on A_m; positive means more lure selection when X is last.",
             "All contrasts are percentage points. Empty denominators are null, never zero.", "",
             "| Model | Weights | N | B_m | A_m | LR_F | LR_S | DR_F | DR_S |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for summary in summaries:
        if summary["cohort"] != "all_selected":
            continue
        cells = [summary["model"], summary["weight_precision"], str(summary["items"]),
                 str(summary["B_m_size"]), str(summary["A_m_size"])]
        for key in ("LR_F", "LR_S", "DR_F", "DR_S"):
            value = summary["metrics"][key]
            cells.append("NA" if value["rate"] is None else
                         f"{value['rate']:.2%} ({value['numerator']}/{value['denominator']})")
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "The fixed equal-weight panel includes models with at least 20 A_m clusters.",
              "Panel Delta_LR and Delta_DR have two-sided 97.5% paired cluster percentile intervals (two primary comparisons).",
              "Resistance/order and individual-model intervals are exploratory 95% intervals without multiplicity adjustment.",
              "Excluded models retain descriptive estimates; their bootstrap intervals are null. Valid replicate counts are reported.",
              "Primary clusters are (source, cluster_id); analysis_group_id is a labeled dependency sensitivity analysis.",
              "Source/dimension counts and outcomes are in statistics.csv; order and resistance differences are in contrasts.csv.",
              "Main and condition-supplement datasets are analyzed separately. Review metadata absent from release is unknown."]
    if reference["limit"]:
        lines += ["", "This is a limited smoke sample; do not interpret it as the full experiment."]
    if allow_mixed_runtimes:
        lines += ["", "Mixed runtime comparison enabled. AWQ versus BF16 comparisons are confounded by precision and backend."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directories", nargs="+", type=Path)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--allow-mixed-runtimes", action="store_true")
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--cluster-field", choices=("cluster_id", "analysis_group_id"), default="cluster_id")
    parser.add_argument("--save-draws", action="store_true")
    args = parser.parse_args()
    analyze_runs(args.run_directories, args.output, args.dataset, args.allow_mixed_runtimes,
                 args.replicates, args.seed, args.cluster_field, args.save_draws)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
