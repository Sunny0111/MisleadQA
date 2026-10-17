"""Analyze option-stable cohorts without replacing the original main analysis."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import random
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "experiments.option_order"

from experiments.analysis.metrics import OUTCOMES, _percentile, fraction
from experiments.main_eval.protocol import DEFAULT_DATASET, file_hash, load_models, save_json, write_jsonl
from experiments.scope_diagnostic.protocol import sample_items
from misleadqa_utils import OUTPUTS_DIR
from .protocol import CONDITIONS, ORDERS, VERSION, source_hashes
from .run import SEALED_FILES, sealed_files, verify_run


def join_results(items, reference, rows, rerun_original=False, f_rows=None):
    conditions = CONDITIONS + (("f_only",) if f_rows is not None else ())
    answers = defaultdict(dict)
    reused = [] if rerun_original else [(ORDERS[0], r) for r in reference["responses"]]
    for order, row in reused + [
            (r["permutation_id"], r) for r in [*rows, *(f_rows or [])]]:
        key = (order, row["condition_id"])
        if order not in ORDERS or key[1] not in conditions or key in answers[row["item_id"]]:
            raise ValueError("Duplicate or unknown option-order response")
        response = row["response"]
        outcome = next((role for role, label in row["answer_key"].items() if label == response),
                       "other" if response in tuple("ABCDE") else "invalid")
        answers[row["item_id"]][key] = {"label": response, "outcome": outcome,
            "original_label": row.get("original_option_labels", dict(zip("ABCDE", "ABCDE"))).get(response)}
    if set(answers) != {item["id"] for item in items}:
        raise ValueError("Answers do not cover exactly the frozen sample")
    joined = []
    for item in items:
        cells = answers[item["id"]]
        if set(cells) != {(order, condition) for order in ORDERS for condition in conditions}:
            raise ValueError("Each item needs all three orders and all selected conditions")
        joined.append({"item_id": item["id"], "source": item["source"], "dimension": item["dimension"],
            "cluster_id": item["grouping"]["cluster_id"],
            "analysis_group_id": item["grouping"]["analysis_group_id"],
            "original_correct": cells[ORDERS[0], "closed_book"]["outcome"] == "gold",
            "stable_correct": all(cells[order, "closed_book"]["outcome"] == "gold" for order in ORDERS),
            "answers": {order: {condition: cells[order, condition] for condition in conditions} for order in ORDERS}})
    return joined


def reference_agreement(reference, rows):
    """Audit historical answers without using them to select the fresh cohort."""
    original = {(r["item_id"], r["condition_id"]): r["response"] for r in reference["responses"]}
    compared = [r for r in rows if r["permutation_id"] == "original"]
    return {condition: {"items": sum(r["condition_id"] == condition for r in compared),
        "disagreements": [{"item_id": r["item_id"], "historical": original[r["item_id"], condition],
                           "fresh": r["response"]} for r in compared
                          if r["condition_id"] == condition and r["response"] != original[r["item_id"], condition]]}
        for condition in CONDITIONS}


def contributions(item):
    original, stable = int(item["original_correct"]), int(item["stable_correct"])
    values = [1, original, stable]
    for condition in ("s_only", "f_only"):
        if condition in item["answers"][ORDERS[0]]:
            lure = [int(item["answers"][order][condition]["outcome"] == "lure") for order in ORDERS]
            values.extend([original * lure[0], *[stable * value for value in lure]])
    return values


def metrics(totals):
    if len(totals) not in (7, 11):
        raise ValueError("Expected S-only or paired S/F contributions")
    n, original, stable = totals[:3]
    result = {
        "original_closed_book_accuracy": fraction(original, n),
        "stable_closed_book_fraction": fraction(stable, n),
        "retention_of_original_correct": fraction(stable, original),
    }
    for index, offset in enumerate(range(3, len(totals), 4)):
        label = ("S", "F")[index]
        baseline, *orders = totals[offset:offset + 4]
        result[f"original_cohort_{label}_lure_original"] = fraction(baseline, original)
        result.update({f"stable_cohort_{label}_lure_{order}": fraction(value, stable)
                       for order, value in zip(ORDERS, orders)})
        result[f"stable_cohort_{label}_lure_mean"] = fraction(sum(orders), 3 * stable)
    if len(totals) == 11:
        for key in list(result):
            if "_S_lure_" in key:
                s, f = result[key], result[key.replace("_S_", "_F_")]
                result[key.replace("_S_lure_", "_F_minus_S_lure_")] = fraction(
                    f["numerator"] - s["numerator"], s["denominator"])
    return result


def statistics(items):
    totals = [sum(column) for column in zip(*(contributions(item) for item in items))] if items else [0] * 7
    original = [item for item in items if item["original_correct"]]
    stable = [item for item in items if item["stable_correct"]]
    outcomes = {}
    for evidence in (("S", "F") if len(totals) == 11 else ("S",)):
        outcomes[f"{evidence}_outcomes"] = {}
        for cohort, selected in (("original_cohort", original), ("stable_cohort", stable)):
            for label, orders in [(order, (order,)) for order in ORDERS] + [("mean", ORDERS)]:
                counts = Counter(item["answers"][order][f"{evidence.lower()}_only"]["outcome"]
                                 for item in selected for order in orders)
                outcomes[f"{evidence}_outcomes"][f"{cohort}.{label}"] = {
                    outcome: fraction(counts[outcome], len(selected) * len(orders)) for outcome in OUTCOMES}
    return {"sample_items": len(items), "original_correct_items": len(original), "stable_correct_items": len(stable),
            "stable_source_clusters": len({(i["source"], i["cluster_id"]) for i in stable}),
            "stable_dependency_groups": len({i["analysis_group_id"] for i in stable}),
            "closed_book_by_order": {order: fraction(sum(i["answers"][order]["closed_book"]["outcome"] == "gold"
                for i in items), len(items)) for order in ORDERS},
            "metrics": metrics(totals), **outcomes}


def equal_model_mean(values):
    result = {}
    for key in next(iter(values.values())):
        components = {name: entries[key] for name, entries in values.items()}
        rates = [entry["rate"] for entry in components.values()]
        result[key] = {"rate": sum(rates) / len(rates) if None not in rates else None,
                       "model_components": components}
    return result


def cluster_key(item, field):
    return (item["source"], item["cluster_id"]) if field == "source" else item["analysis_group_id"]


def paired_bootstrap(models, field="source", replicates=10000, seed=2027):
    """Resample whole clusters, sharing multiplicities across all models and orders."""
    if field not in ("source", "dependency") or not models or replicates < 1:
        raise ValueError("Provide models, a valid cluster field and positive replicates")
    memberships = [{item["item_id"]: cluster_key(item, field) for item in items} for items in models.values()]
    if (not memberships[0] or any(len(m) != len(items) for m, items in zip(memberships, models.values()))
            or any(m != memberships[0] for m in memberships[1:])):
        raise ValueError("Duplicate, missing or inconsistent item/cluster memberships")
    clusters = sorted(set(memberships[0].values()))
    index = {key: number for number, key in enumerate(clusters)}
    widths = {len(contributions(item)) for items in models.values() for item in items}
    if len(widths) != 1:
        raise ValueError("All models and items must have the same evidence conditions")
    columns = widths.pop()
    arrays = {}
    for name, items in models.items():
        array = [[0] * columns for _ in clusters]
        for item in items:
            target = array[index[cluster_key(item, field)]]
            for column, value in enumerate(contributions(item)):
                target[column] += value
        arrays[name] = array
    points = {name: statistics(items)["metrics"] for name, items in models.items()}
    points["equal_model_mean"] = equal_model_mean(points)
    draws = {name: {key: [] for key in values} for name, values in points.items()}
    rng = random.Random(seed)
    for _ in range(replicates):
        weights = Counter(rng.choices(range(len(clusters)), k=len(clusters)))
        sampled = {name: metrics([sum(weight * array[i][column] for i, weight in weights.items())
                                 for column in range(columns)]) for name, array in arrays.items()}
        sampled["equal_model_mean"] = equal_model_mean(sampled)
        for name, values in sampled.items():
            for key, value in values.items():
                if value["rate"] is not None:
                    draws[name][key].append(value["rate"])
    intervals = {}
    for name, values in points.items():
        intervals[name] = {}
        for key, point in values.items():
            valid = draws[name][key]
            defined = bool(valid) and point["rate"] is not None
            intervals[name][key] = {"estimate": point["rate"], "confidence": 0.95,
                "lower": _percentile(valid, 0.025) if defined else None,
                "upper": _percentile(valid, 0.975) if defined else None,
                "valid_replicates": len(valid), "invalid_denominator_replicates": replicates - len(valid)}
    return {"method": "paired pooled cluster percentile bootstrap", "cluster_field": field,
            "cluster_membership": clusters, "clusters": len(clusters), "replicates": replicates, "seed": seed,
            "panel_models": list(models), "panel_weighting": "fixed equal model weights",
            "undefined_policy": "null; never drop a model with a zero denominator",
            "inference": "exploratory 95%; no multiplicity adjustment",
            "pairing": "one cluster draw shared by all models, orders and conditions",
            "intervals": intervals}


def display(value, difference=False):
    return "Not estimable" if value is None else f"{100 * value:.2f}{' pp' if difference else '%'}"


def analyze_runs(run_root, run_id, output, replicates=10000, seed=2027, f_run_id=None, f_run_root=None):
    output = Path(output)
    if output.exists():
        raise ValueError("Analysis output exists; choose a new directory")
    items, _ = sample_items()
    models, provenance, agreements, f_provenance = {}, {}, {}, {}
    modes = set()
    for alias in load_models():
        directory = Path(run_root) / alias / run_id
        manifest, reference, rows = verify_run(directory)
        if (manifest["identity"]["smoke"] or manifest["identity"]["model_alias"] != alias
                or tuple(manifest["identity"]["conditions"]) != CONDITIONS):
            raise ValueError("Analysis requires one complete full run for each of the six models")
        fresh = manifest["identity"].get("rerun_original", False)
        modes.add(fresh)
        if len(modes) != 1:
            raise ValueError("Do not combine reused-original and fresh-original runs in one panel")
        provenance[alias] = {name: file_hash(directory / name) for name in (*SEALED_FILES, "complete.json")}
        f_rows = None
        if f_run_id:
            f_directory = Path(f_run_root or run_root) / alias / f_run_id
            f_manifest, _, f_rows = verify_run(f_directory)
            f_identity = f_manifest["identity"]
            if f_identity["smoke"] or f_identity["model_alias"] != alias or f_identity["conditions"] != ["f_only"]:
                raise ValueError("Expected a complete F-only extension for the same model")
            paired = json.loads((f_directory / "paired_baseline.json").read_text(encoding="utf-8"))
            if paired["source_files_sha256"] != provenance[alias]:
                raise ValueError("F-only responses belong to a different closed-book/S baseline")
            f_provenance[alias] = {name: file_hash(f_directory / name)
                                   for name in (*sealed_files(f_identity), "complete.json")}
        models[alias] = join_results(items, reference, rows, fresh, f_rows)
        if fresh:
            agreements[alias] = reference_agreement(reference, rows)
    summaries = {name: statistics(rows) for name, rows in models.items()}
    panel = equal_model_mean({name: value["metrics"] for name, value in summaries.items()})
    bootstraps = {field: paired_bootstrap(models, field, replicates, seed) for field in ("source", "dependency")}
    groups = {}
    for source, dimension in sorted({(item["source"], item["dimension"]) for item in items}):
        groups[f"{source}:{dimension}"] = {name: statistics([item for item in rows
            if (item["source"], item["dimension"]) == (source, dimension)]) for name, rows in models.items()}
    output.mkdir(parents=True)
    outcome_panels = {f"panel_{label}_outcomes": {key: equal_model_mean({name: summary[f"{label}_outcomes"][key]
        for name, summary in summaries.items()}) for key in next(iter(summaries.values()))[f"{label}_outcomes"]}
        for label in (("S", "F") if f_run_id else ("S",))}
    save_json(output / "statistics.json", {"protocol": VERSION, "rerun_original": fresh,
        "models": summaries, "equal_model_mean": panel, "f_run_id": f_run_id, **outcome_panels})
    save_json(output / "groups.json", groups)
    if fresh:
        save_json(output / "reference_agreement.json", agreements)
    save_json(output / "bootstrap.json", bootstraps)
    save_json(output / "provenance.json", {"runs": provenance, "f_runs": f_provenance, "code_sha256": source_hashes(),
                                           "dataset_sha256": file_hash(DEFAULT_DATASET)})
    write_jsonl(output / "joined_items.jsonl", [item | {"model": name} for name, rows in models.items() for item in rows])
    records = []
    for name, values in [(n, s["metrics"]) for n, s in summaries.items()] + [("equal_model_mean", panel)]:
        for key, value in values.items():
            record = {"model": name, "metric": key, "numerator": value.get("numerator"),
                      "denominator": value.get("denominator"), "rate": value["rate"],
                      "unit": "proportion_difference" if "_F_minus_S_" in key else "proportion"}
            for field, bootstrap in bootstraps.items():
                interval = bootstrap["intervals"][name][key]
                record.update({f"{field}_{k}": interval[k] for k in ("lower", "upper", "valid_replicates",
                                                                   "invalid_denominator_replicates")})
            records.append(record)
    with (output / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    design = ("All three orders are generated in the new environment. Historical original answers are audited in reference_agreement.json and do not define the cohort."
              if fresh else "Original answers are reused; only two new A-D rotations are generated.")
    lines = ["# Option-order robustness", "",
        "The same frozen 200-item diagnostic sample is used for all six models. " + design + " E stays fixed.",
        "Stable means correct closed-book on these three orders, not established knowledge. This subset check does not replace the main B_m analysis or directly compare with its full-set 29.3% rate.",
        "Rates are unweighted means over models. Undefined denominators remain null; models are never silently excluded. Small stable cohorts require cautious interpretation.", "",
        "| Model | Original correct | Stable correct | Retained | Original S lure on original cohort | Original S lure on stable cohort | Mean S lure on stable cohort |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    keys = ("retention_of_original_correct", "original_cohort_S_lure_original",
            "stable_cohort_S_lure_original", "stable_cohort_S_lure_mean")
    for name, summary in summaries.items():
        lines.append(f"| {name} | {summary['original_correct_items']} | {summary['stable_correct_items']} | "
                     + " | ".join(display(summary["metrics"][key]["rate"]) for key in keys) + " |")
    lines.append("| Unweighted mean | -- | -- | " + " | ".join(display(panel[key]["rate"]) for key in keys) + " |")
    if f_run_id:
        lines += ["", "## Matched S/F comparisons", "",
            "F-only responses reuse the verified closed-book/S-only baseline. Cohorts depend only on closed-book answers; F responses never select items. All three orders stay paired in each bootstrap draw.",
            "Rates are percentages; F minus S is in percentage points. A contrast numerator is the difference in lure counts, not a response count. Both grouping schemes use exploratory 95% intervals.", "",
            "| Model | Cohort and orders | S lure | F lure | F minus S |",
            "|---|---|---:|---:|---:|"]
        for name, values in [(n, s["metrics"]) for n, s in summaries.items()] + [("Unweighted mean", panel)]:
            for cohort, order in (("original_cohort", "original"),
                                  ("stable_cohort", "original"), ("stable_cohort", "mean")):
                estimates = [display(values[f"{cohort}_{label}_lure_{order}"]["rate"], label == "F_minus_S")
                             for label in ("S", "F", "F_minus_S")]
                lines.append(f"| {name} | {cohort}, {order} | " + " | ".join(estimates) + " |")
    lines += ["", "## Exploratory 95% intervals", "", "| Model | Metric | Estimate | Source clusters | Dependency groups |",
              "|---|---|---:|---|---|"]
    for record in records:
        difference = "_F_minus_S_" in record["metric"]
        intervals = ["Not estimable" if record[f"{field}_lower"] is None else
                     f"[{display(record[f'{field}_lower'], difference)}, {display(record[f'{field}_upper'], difference)}]"
                     for field in bootstraps]
        lines.append(f"| {record['model']} | {record['metric']} | {display(record['rate'], difference)} | " + " | ".join(intervals) + " |")
    lines += ["", "Bootstrap draws retain all three orders of each item and use shared cluster multiplicities across models. Intervals describe this fixed model panel; they do not treat repeated orders as independent observations.",
        "Full outcome counts (gold/lure/abstain/other/invalid), per-order closed-book accuracy, stable cluster counts, and source-scope cohort sizes are in statistics.json and groups.json. Valid/invalid replicate counts are in metrics.csv and bootstrap.json."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Analyzed six models and 200 items per model: {output.resolve()}")
    return summaries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=OUTPUTS_DIR / "option_order")
    parser.add_argument("--run-id", default="main-option-order-full-v1")
    parser.add_argument("--f-run-id", help="Add matched F-only results to the existing closed-book/S analysis")
    parser.add_argument("--f-run-root", type=Path, help="Defaults to --run-root")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args(argv)
    if args.replicates < 1:
        parser.error("replicates must be positive")
    if args.f_run_root and not args.f_run_id:
        parser.error("--f-run-root requires --f-run-id")
    output = args.output or OUTPUTS_DIR / "analysis/option_order" / (args.f_run_id or args.run_id)
    analyze_runs(args.run_root, args.run_id, output, args.replicates, args.seed, args.f_run_id, args.f_run_root)


if __name__ == "__main__":
    main()
