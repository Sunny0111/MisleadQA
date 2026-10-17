"""Descriptive scope/answer diagnostics and paired source-cluster 95% intervals."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import random
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "experiments.scope_diagnostic"

from experiments.analysis.metrics import OUTCOMES, _percentile, fraction, item_results
from experiments.main_eval.protocol import DEFAULT_DATASET, ROOT, file_hash, load_models, save_json, write_jsonl
from experiments.main_eval.verify import verify_run as verify_baseline
from .protocol import EVIDENCE, HERE, LABELS, SEED, SETTINGS, sample_items, source_hashes
from .run import verify_run
from misleadqa_utils import OUTPUTS_DIR


def contributions(item):
    """Nine G/F/S × label cells, then fifteen S-label × answer cells on B_m."""
    values = [0] * 24
    for index, condition in enumerate(EVIDENCE):
        values[3 * index + LABELS.index(item["judgments"][condition])] = 1
    if item["closed_book_correct"]:
        values[9 + 5 * LABELS.index(item["judgments"]["S"]) + OUTCOMES.index(item["s_outcome"])] = 1
    return values


def total_counts(items):
    totals = [0] * 24
    for item in items:
        for index, value in enumerate(contributions(item)):
            totals[index] += value
    return totals


def metrics(counts):
    n = sum(counts[:3])
    result = {"recall.applicable": fraction(counts[0] + counts[3], 2 * n),
              "recall.inapplicable": fraction(counts[7], n)}
    recalls = [result[key]["rate"] for key in result]
    result["balanced_accuracy"] = {"rate": sum(recalls) / 2 if None not in recalls else None,
        "components": {key: value.copy() for key, value in result.items()}}
    for index, condition in enumerate(EVIDENCE):
        result[f"{condition}.correct"] = fraction(counts[3 * index + (1 if condition == "S" else 0)], n)
        result[f"{condition}.uncertain"] = fraction(counts[3 * index + 2], n)
    result["lure_given_inapplicable"] = fraction(counts[15], sum(counts[14:19]))
    result["lure_given_applicable"] = fraction(counts[10], sum(counts[9:14]))
    result["inapplicable_and_lure"] = fraction(counts[15], sum(counts[9:]))
    return result


def statistics(items):
    counts = total_counts(items)
    confusion = {LABELS[0]: {label: counts[index] + counts[3 + index] for index, label in enumerate(LABELS)},
                 LABELS[1]: {label: counts[6 + index] for index, label in enumerate(LABELS)}}
    cross = {label: dict(zip(OUTCOMES, counts[9 + 5 * index:14 + 5 * index]))
             for index, label in enumerate(LABELS)}
    return {"items": len(items), "B_m_size": sum(counts[9:]), "metrics": metrics(counts),
            "confusion_matrix": confusion, "cross_table_on_B_m": cross}


def equal_model_mean(summaries):
    """Keep the fixed model panel; never pool denominators or drop undefined models."""
    result = {}
    for key in next(iter(summaries.values()))["metrics"]:
        components = {name: value["metrics"][key] for name, value in summaries.items()}
        rates = [value["rate"] for value in components.values()]
        result[key] = {"rate": sum(rates) / len(rates) if None not in rates else None,
                       "model_components": components}
    return result


def equal_model_tables(summaries):
    """Cell probabilities averaged over models; counts stay in model components."""
    result = {}
    for table in ("confusion_matrix", "cross_table_on_B_m"):
        result[table] = {}
        for row, cells in next(iter(summaries.values()))[table].items():
            result[table][row] = {}
            for column in cells:
                components = {name: fraction(summary[table][row][column],
                    sum(summary[table][row].values()) if table == "confusion_matrix" else summary["B_m_size"])
                    for name, summary in summaries.items()}
                rates = [value["rate"] for value in components.values()]
                result[table][row][column] = {"rate": sum(rates) / len(rates) if None not in rates else None,
                                             "model_components": components}
    return result


def paired_bootstrap(models, replicates=10000, seed=SEED):
    if not models or replicates < 1:
        raise ValueError("Models and positive bootstrap replicates are required")
    memberships = [{i["item_id"]: (i["source"], i["cluster_id"]) for i in items} for items in models.values()]
    if (any(len(m) != len(items) for m, items in zip(memberships, models.values()))
            or any(m != memberships[0] for m in memberships[1:]) or not memberships[0]):
        raise ValueError("Duplicate, missing or mismatched model item/cluster memberships")
    clusters = sorted(set(memberships[0].values()))
    cluster_index = {key: index for index, key in enumerate(clusters)}
    arrays = {}
    for name, items in models.items():
        sums = [[0] * 24 for _ in clusters]
        for item in items:
            target = sums[cluster_index[item["source"], item["cluster_id"]]]
            for index, value in enumerate(contributions(item)):
                target[index] += value
        arrays[name] = sums
    summaries = {name: statistics(items) for name, items in models.items()}
    points = {name: summary["metrics"] for name, summary in summaries.items()}
    points["equal_model_mean"] = equal_model_mean(summaries)
    draws = {name: {key: [] for key in values} for name, values in points.items()}
    rng = random.Random(seed)
    for _ in range(replicates):
        weights = Counter(rng.choices(range(len(clusters)), k=len(clusters)))
        sampled = {}
        for name, array in arrays.items():
            totals = [sum(weight * array[index][column] for index, weight in weights.items())
                      for column in range(24)]
            sampled[name] = metrics(totals)
        for key in points["equal_model_mean"]:
            rates = [sampled[name][key]["rate"] for name in models]
            for name, value in zip(models, rates):
                if value is not None:
                    draws[name][key].append(value)
            if None not in rates:
                draws["equal_model_mean"][key].append(sum(rates) / len(rates))
    intervals = {}
    for name, values in points.items():
        intervals[name] = {}
        for key, point in values.items():
            valid = draws[name][key]
            intervals[name][key] = {"estimate": point["rate"], "confidence": 0.95,
                "lower": _percentile(valid, 0.025) if valid and point["rate"] is not None else None,
                "upper": _percentile(valid, 0.975) if valid and point["rate"] is not None else None,
                "valid_replicates": len(valid), "invalid_denominator_replicates": replicates - len(valid)}
    return {"method": "paired pooled (source, cluster_id) percentile bootstrap",
            "replicates": replicates, "seed": seed, "clusters": len(clusters),
            "cluster_membership": [{"source": s, "cluster_id": c} for s, c in clusters],
            "pairing": "same cluster multiplicities for every model and G/F/S; retain whole clusters and item weighting",
            "panel_models": list(models), "panel_weighting": "equal weights; all six models fixed",
            "undefined_policy": "null; panel draw invalid if any model denominator is zero",
            "inference": "exploratory 95%, no multiplicity adjustment; groups and table counts descriptive",
            "intervals": intervals}


def join_results(items, judgments, answers):
    by_scope = defaultdict(dict)
    for row in judgments:
        condition = row["condition_id"]
        if condition not in EVIDENCE or condition in by_scope[row["item_id"]] or row["response"] not in LABELS:
            raise ValueError("Unknown or duplicate scope judgment")
        by_scope[row["item_id"]][condition] = row["response"]
    by_answer = {row["item_id"]: row for row in item_results(answers)}
    if set(by_scope) != {i["id"] for i in items}:
        raise ValueError("Scope run does not cover the complete frozen sample")
    joined = []
    for item in items:
        scope = by_scope[item["id"]]
        if set(scope) != set(EVIDENCE) or item["id"] not in by_answer:
            raise ValueError("Missing scope or original answer conditions")
        answer = by_answer[item["id"]]
        if (answer["source"], answer["dimension"]) != (item["source"], item["dimension"]):
            raise ValueError("Original answer source/group differs from dataset")
        outcome = next(o for o in OUTCOMES if answer["outcomes"]["s_only"][o])
        joined.append({"item_id": item["id"], "source": item["source"], "dimension": item["dimension"],
                       "cluster_id": item["grouping"]["cluster_id"], "judgments": scope,
                       "closed_book_correct": answer["B_m"], "s_outcome": outcome,
                       "closed_book_answer": answer["outcomes"]["closed_book"]["answer"],
                       "s_only_answer": answer["outcomes"]["s_only"]["answer"]})
    return joined


def check_pair(scope, baseline):
    si, bi = scope[0]["identity"], baseline[0]["identity"]
    if (si["smoke"] or bi["counts"]["selected_items"] != 959 or si["model_alias"] != bi["model_alias"]
            or si["model"] != bi["model"] or si["dataset_sha256"] != bi["dataset_sha256"]
            or any(bi["settings"].get(key) != value for key, value in SETTINGS.items())):
        raise ValueError("Pair requires full original greedy main answers from the same pinned model/data")
    for key in ("backend", "model_id", "revision", "dtype", "quantization", "chat_template_kwargs",
                "tokenizer_sha256", "chat_template_sha256", "asset_sha256"):
        if key not in scope[1] or key not in baseline[1] or scope[1][key] != baseline[1][key]:
            raise ValueError(f"Scope/answer model assets differ: {key}")


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def display_rate(value):
    return "Not estimable" if value is None else f"{100 * value:.2f}%"


def report(summaries, mean, bootstrap):
    lines = ["# Scope judgment and answering behavior consistency diagnostic", "",
        "Fixed stratified subset: 40 items from each of five groups, 200 items total. This does not represent the natural main-set distribution.",
        "Scope judgments and existing answers are independent calls. These results cannot show that a model knowingly used inapplicable material within the same answer.",
        "Differences between the two conditional probabilities have no causal interpretation. Cannot determine stays in the denominator and counts as incorrect classification.",
        "All intervals are exploratory 95% paired source-cluster percentile bootstrap intervals; source-scope groups and table counts are descriptive.",
        "Equal-model means use all six fixed models. A metric is not estimable if any model is undefined; model denominators are not pooled.", "",
        "| Model | Metric | Numerator/denominator | Estimate | 95% interval | Valid/requested bootstrap replicates |",
        "|---|---|---|---|---|---|"]
    for name, values in [(n, s["metrics"]) for n, s in summaries.items()] + [("equal_model_mean", mean)]:
        for key, value in values.items():
            interval = bootstrap["intervals"][name][key]
            if "numerator" in value:
                counts = f"{value['numerator']}/{value['denominator']}"
            elif "components" in value:
                counts = "; ".join(f"{k}: {v['numerator']}/{v['denominator']}" for k, v in value["components"].items())
            else:
                counts = "Model components in corresponding rows and statistics.json"
            ci = "Not estimable" if interval["lower"] is None else f"[{display_rate(interval['lower'])}, {display_rate(interval['upper'])}]"
            lines.append(f"| {name} | {key} | {counts} | {display_rate(value['rate'])} | {ci} | {interval['valid_replicates']}/{bootstrap['replicates']} |")
    for name, summary in summaries.items():
        lines += ["", f"## {name}", "", "True class by predicted label (counts; final column is the class denominator)", "",
                  "| True class | Applicable | Not applicable | Cannot determine | Denominator |", "|---|---:|---:|---:|---:|"]
        for actual, row in summary["confusion_matrix"].items():
            lines.append(f"| {actual} | " + " | ".join(str(row[label]) for label in LABELS) + f" | {sum(row.values())} |")
        lines += ["", f"S judgment by original S-only answer among originally correct closed-book items (B_m = {summary['B_m_size']})", "",
                  "| S judgment | gold | lure | abstain | other | invalid | Row denominator | B_m denominator |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for label, row in summary["cross_table_on_B_m"].items():
            lines.append(f"| {label} | " + " | ".join(str(row[o]) for o in OUTCOMES) +
                         f" | {sum(row.values())} | {summary['B_m_size']} |")
    tables = equal_model_tables(summaries)
    lines += ["", "## Tables with equal weights across six models", "", "Confusion matrix cells are output proportions within each true class, averaged with equal model weights.", "",
              "| True class | Applicable | Not applicable | Cannot determine |", "|---|---:|---:|---:|"]
    for actual, cells in tables["confusion_matrix"].items():
        lines.append(f"| {actual} | " + " | ".join(display_rate(cells[label]["rate"]) for label in LABELS) + " |")
    lines += ["", "Cross-table cells are joint proportions within each model's B_m, averaged with equal model weights. Model numerators/denominators are retained in JSON.", "",
              "| S judgment | gold | lure | abstain | other | invalid |", "|---|---:|---:|---:|---:|---:|"]
    for label, cells in tables["cross_table_on_B_m"].items():
        lines.append(f"| {label} | " + " | ".join(display_rate(cells[o]["rate"]) for o in OUTCOMES) + " |")
    lines += ["", "Group descriptions and metric numerators/denominators are saved in groups.json and metrics.csv; item-level joins are in joined_items.jsonl.",
              "Resampling uses only (source, cluster_id) clusters represented in the subset, draws the same number of clusters, and retains all items within each selected cluster.",
              "Bootstrap replicates with zero denominators are excluded from percentile calculations and counted for every metric. Interpret sparse conditional-probability intervals cautiously."]
    return "\n".join(lines) + "\n"


def analyze(root, output, run_id, dataset, baseline_root, baseline_run_id, replicates=10000, seed=SEED):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Analysis output is nonempty; choose a new --output directory")
    items, sample = sample_items(dataset)
    models, provenance = {}, {}
    for name in load_models():
        directory = Path(root) / name / run_id
        baseline_directory = Path(baseline_root) / name / baseline_run_id
        scope = verify_run(directory, dataset)
        baseline = verify_baseline(baseline_directory, dataset)
        check_pair(scope, baseline)
        models[name] = join_results(items, scope[2], baseline[2])
        provenance[name] = {kind: {"directory": str(path.resolve()), "identity": loaded[0]["identity"],
                           "model_metadata": loaded[1],
                           "raw_sha256": file_hash(path / "raw_responses.jsonl")}
                           for kind, path, loaded in (("scope", directory, scope),
                                                     ("original_answers", baseline_directory, baseline))}
    summaries = {name: statistics(rows) for name, rows in models.items()}
    mean = equal_model_mean(summaries)
    grouped = {}
    for name, rows in models.items():
        groups = defaultdict(list)
        for row in rows:
            groups[f"{row['source']}/{row['dimension']}"].append(row)
        grouped[name] = {group: statistics(subset) for group, subset in sorted(groups.items())}
    bootstrap = paired_bootstrap(models, replicates, seed)
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "statistics.json", {"models": summaries, "equal_model_mean": mean,
                                          "equal_model_tables": equal_model_tables(summaries)})
    save_json(output / "groups.json", grouped)
    save_json(output / "bootstrap.json", bootstrap)
    save_json(output / "provenance.json", {"sample": sample, "runs": provenance,
                                          "analysis_code_sha256": source_hashes()})
    write_jsonl(output / "joined_items.jsonl", [row | {"model": name} for name, rows in models.items() for row in rows])
    metric_rows = []
    for name, summary in summaries.items():
        for group, values in [("all", summary)] + list(grouped[name].items()):
            for key, value in values["metrics"].items():
                ci = bootstrap["intervals"][name][key] if group == "all" else {}
                metric_rows.append({"model": name, "group": group, "metric": key,
                    "numerator": value.get("numerator"), "denominator": value.get("denominator"),
                    "components": json.dumps(value.get("components", {}), ensure_ascii=False), "rate": value["rate"],
                    "lower_95": ci.get("lower"), "upper_95": ci.get("upper"),
                    "valid_replicates": ci.get("valid_replicates"),
                    "invalid_denominator_replicates": ci.get("invalid_denominator_replicates")})
    for key, value in mean.items():
        ci = bootstrap["intervals"]["equal_model_mean"][key]
        metric_rows.append({"model": "equal_model_mean", "group": "all", "metric": key,
            "numerator": None, "denominator": None,
            "components": json.dumps(value["model_components"], ensure_ascii=False), "rate": value["rate"],
            "lower_95": ci["lower"], "upper_95": ci["upper"], "valid_replicates": ci["valid_replicates"],
            "invalid_denominator_replicates": ci["invalid_denominator_replicates"]})
    write_csv(output / "metrics.csv", metric_rows)
    table_rows = []
    for name, summary in summaries.items():
        for table in ("confusion_matrix", "cross_table_on_B_m"):
            for row_label, cells in summary[table].items():
                for column, count in cells.items():
                    table_rows.append({"model": name, "table": table, "row": row_label, "column": column,
                        "numerator": count, "row_denominator": sum(cells.values()),
                        "table_denominator": 3 * summary["items"] if table == "confusion_matrix" else summary["B_m_size"]})
    write_csv(output / "tables.csv", table_rows)
    (output / "report.md").write_text(report(summaries, mean, bootstrap), encoding="utf-8")
    print(f"Verified six models; 3,600 scope judgments paired with original answers: {output.resolve()}")
    return summaries, bootstrap


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=OUTPUTS_DIR / "scope_diagnostic")
    parser.add_argument("--output", type=Path, default=OUTPUTS_DIR / "analysis/scope_diagnostic/main-scope-full-v2")
    parser.add_argument("--run-id", default="main-scope-full-v2")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--baseline-root", type=Path, default=ROOT / "experiments/outputs/main_eval")
    parser.add_argument("--baseline-run-id", default="main-v100-full-v1")
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    analyze(args.root, args.output, args.run_id, args.dataset, args.baseline_root,
            args.baseline_run_id, args.bootstrap_replicates, args.bootstrap_seed)


if __name__ == "__main__":
    main()
