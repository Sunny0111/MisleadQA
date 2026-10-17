"""Item-paired S/F answering analysis from saved main-evaluation outcomes.

No inference is run. Cohorts and the equal-model panel remain fixed to the
original main analysis. NumPy accelerates the existing pooled cluster draws.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from experiments.analysis.metrics import OUTCOMES, item_results, statistics
from experiments.main_eval.protocol import CONDITION_IDS, read_jsonl

DEFAULT_BASE = ROOT / "experiments/outputs/analysis/main-v100-full-v1"
DEFAULT_OUTPUT = ROOT / "experiments/outputs/analysis/paired-configurations-main-v1"
ORDERS = {
    "G_first": ("g_then_s", "g_then_f"),
    "misleading_first": ("s_then_g", "f_then_g"),
}
CELLS = ("both_gold", "S_only_gold", "F_only_gold", "neither_gold")
# Count vector: cohort size, two single-condition outcome vectors, then two
# 5 x 5 joint outcome matrices (S rows, F columns), each built before averaging.
WIDTH = 1 + 2 * len(OUTCOMES) + len(ORDERS) * len(OUTCOMES) ** 2


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_models(path):
    models = defaultdict(list)
    for row in read_jsonl(path):
        model = row.pop("model")
        models[model].append(row)
    reference = None
    for model, rows in sorted(models.items()):
        mapping = {}
        for row in rows:
            item = row["item_id"]
            if item in mapping:
                raise ValueError(f"Duplicate model/item: {model}/{item}")
            mapping[item] = tuple(row[k] for k in
                                  ("source", "dimension", "cluster_id", "analysis_group_id"))
            if any(not v for v in mapping[item]):
                raise ValueError("Missing grouping metadata")
            outcomes = row["outcomes"]
            if set(outcomes) != set(CONDITION_IDS):
                raise ValueError("Incomplete eight-condition item")
            for outcome in outcomes.values():
                values = [outcome[o] for o in OUTCOMES]
                if any(v not in (0, 1) for v in values) or sum(values) != 1:
                    raise ValueError("Outcomes must be mutually exclusive and exhaustive")
            b, g = bool(outcomes["closed_book"]["gold"]), bool(outcomes["g_only"]["gold"])
            if (row["B_m"], row["G_correct"], row["A_m"]) != (b, g, b and g):
                raise ValueError("Saved cohort flags disagree with baseline answers")
        if reference is not None and mapping != reference:
            raise ValueError("Item identities or grouping metadata differ across models")
        reference = mapping
    if not models:
        raise ValueError("No model outputs")
    return dict(models)


def contribution(row):
    values = np.zeros(WIDTH, dtype=np.int64)
    if not row["A_m"]:
        return values
    values[0] = 1
    for ei, condition in enumerate(("s_only", "f_only")):
        values[1 + 5 * ei:6 + 5 * ei] = [row["outcomes"][condition][o] for o in OUTCOMES]
    for oi, (s, f) in enumerate(ORDERS.values()):
        s_index = next(i for i, o in enumerate(OUTCOMES) if row["outcomes"][s][o])
        f_index = next(i for i, o in enumerate(OUTCOMES) if row["outcomes"][f][o])
        values[11 + 25 * oi + 5 * s_index + f_index] = 1
    return values


def metric_counts(totals):
    """Return exact numerators/denominators, also vectorized over bootstrap draws."""
    n = totals[..., 0]
    result = {}

    def add(key, numerator, denominator=n, unit="percent"):
        result[key] = (numerator, denominator, unit)

    singles = {e: totals[..., 1 + 5 * ei:6 + 5 * ei] for ei, e in enumerate(("S", "F"))}
    margins = {e: [] for e in ("S", "F")}
    for e, counts in singles.items():
        for i, o in enumerate(OUTCOMES):
            add(f"single.{e}.{o}", counts[..., i])
    for oi, order in enumerate(ORDERS):
        joint = totals[..., 11 + 25 * oi:36 + 25 * oi].reshape(totals.shape[:-1] + (5, 5))
        s_counts, f_counts = joint.sum(axis=-1), joint.sum(axis=-2)
        cells = (joint[..., 0, 0], joint[..., 0, 1:].sum(axis=-1),
                 joint[..., 1:, 0].sum(axis=-1), joint[..., 1:, 1:].sum(axis=(-2, -1)))
        for cell, count in zip(CELLS, cells):
            add(f"{order}.pair.{cell}", count)
        for e, counts in (("S", s_counts), ("F", f_counts)):
            margins[e].append(counts)
            for i, o in enumerate(OUTCOMES):
                add(f"{order}.{e}.{o}", counts[..., i])
            add(f"{order}.lure_reduction.{e}", singles[e][..., 1] - counts[..., 1],
                unit="percentage_points")
        add(f"{order}.gold_gap.S_minus_F", s_counts[..., 0] - f_counts[..., 0],
            unit="percentage_points")
        for i, o in enumerate(OUTCOMES):
            add(f"{order}.F_given_S_gold.{o}", joint[..., 0, i], s_counts[..., 0])
    # Average only rates/counts already formed from the two paired orders.
    # Do not define an order-averaged conditional success rate: it has multiple
    # possible estimands, and the two order-specific rates are the primary output.
    for e, counts in margins.items():
        combined = counts[0] + counts[1]
        for i, o in enumerate(OUTCOMES):
            add(f"order_mean.{e}.{o}", combined[..., i], 2 * n)
        add(f"order_mean.lure_reduction.{e}", 2 * singles[e][..., 1] - combined[..., 1],
            2 * n, "percentage_points")
    return result


def rate(numerator, denominator):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(denominator > 0, 100.0 * numerator / denominator, np.nan)


def interval(point, draws):
    valid = draws[np.isfinite(draws)]
    bounds = np.quantile(valid, [.025, .975], method="linear") if len(valid) else [None, None]
    return {"estimate": float(point) if np.isfinite(point) else None, "confidence": .95,
            "lower": None if bounds[0] is None else float(bounds[0]),
            "upper": None if bounds[1] is None else float(bounds[1]),
            "valid_replicates": len(valid), "invalid_replicates": len(draws) - len(valid)}


def cluster_analysis(models, panel, cluster_field, replicates, seed):
    names = sorted(models)
    reference = models[names[0]]
    clusters = sorted({(r["source"], r[cluster_field]) for r in reference})
    index = {c: i for i, c in enumerate(clusters)}
    array = np.zeros((len(clusters), len(names), WIDTH), dtype=np.int64)
    for mi, model in enumerate(names):
        for row in models[model]:
            array[index[(row["source"], row[cluster_field])], mi] += contribution(row)
    rng = random.Random(seed)
    weights = np.empty((replicates, len(clusters)), dtype=np.int64)
    for ri in range(replicates):
        weights[ri] = np.bincount(rng.choices(range(len(clusters)), k=len(clusters)),
                                  minlength=len(clusters))
    totals = array.sum(axis=0)
    draws = (weights @ array.reshape(len(clusters), -1)).reshape(replicates, len(names), WIDTH)
    point_metrics, draw_metrics = metric_counts(totals), metric_counts(draws)
    panel_indices = [names.index(name) for name in panel]
    result = {"cluster_field": cluster_field, "clusters": len(clusters),
              "replicates": replicates, "seed": seed, "panel_models": panel,
              "models": {name: {} for name in names}, "panel": {}}
    for key, (pn, pd, unit) in point_metrics.items():
        dn, dd, _ = draw_metrics[key]
        points, samples = rate(pn, pd), rate(dn, dd)
        for mi, name in enumerate(names):
            result["models"][name][key] = {"numerator": int(pn[mi]), "denominator": int(pd[mi]),
                "unit": unit, **interval(points[mi], samples[:, mi])}
        # Ordinary mean deliberately propagates undefined model rates. No model
        # is silently dropped from the fixed panel within a bootstrap replicate.
        result["panel"][key] = {"unit": unit,
            "model_numerators": {m: int(pn[names.index(m)]) for m in panel},
            "model_denominators": {m: int(pd[names.index(m)]) for m in panel},
            **interval(points[panel_indices].mean(), samples[:, panel_indices].mean(axis=1))}
    result["joint_outcomes"] = {
        name: {order: totals[mi, 11 + 25 * oi:36 + 25 * oi].reshape(5, 5).tolist()
               for oi, order in enumerate(ORDERS)} for mi, name in enumerate(names)}
    return result, point_metrics, draw_metrics


def check_saved_inputs(models, base):
    """Re-score raw first responses and reproduce every original point statistic."""
    provenance = json.loads((base / "provenance.json").read_text(encoding="utf-8"))
    old = {r["model"]: r for r in json.loads((base / "statistics.json").read_text(encoding="utf-8"))
           if r["cohort"] == "all_selected"}
    if set(models) != set(old) or set(models) != {r["model"] for r in provenance["inputs"]}:
        raise ValueError("Model panel differs from saved analysis")
    dataset_path = ROOT / "construction/data/main_split.jsonl"
    if sha(dataset_path) != provenance["dataset_sha256"]:
        raise ValueError("Dataset digest differs from original analysis")
    dataset = {r["id"]: r for r in read_jsonl(dataset_path)}
    checked = []
    for entry in provenance["inputs"]:
        model = entry["model"]
        raw = ROOT / entry["run_directory"] / "raw_responses.jsonl"
        if sha(raw) != entry["sha256"]["raw_responses.jsonl"]:
            raise ValueError(f"Raw-response digest mismatch: {model}")
        rescored = {r["item_id"]: r for r in item_results(read_jsonl(raw))}
        if set(rescored) != {r["item_id"] for r in models[model]} or set(rescored) != set(dataset):
            raise ValueError("Raw-response/dataset/analysis item identities differ")
        for row in models[model]:
            r = rescored[row["item_id"]]
            if any(row[k] != r[k] for k in ("outcomes", "B_m", "G_correct", "A_m")):
                raise ValueError("Saved item outcomes disagree with raw-response re-scoring")
            item = dataset[row["item_id"]]
            if (row["source"] != item["source"] or row["dimension"] != item["dimension"]
                    or any(row[k] != item["grouping"][k] for k in ("cluster_id", "analysis_group_id"))):
                raise ValueError("Grouping differs from frozen dataset")
        new = statistics(models[model])
        if any(new[k] != old[model][k] for k in new):
            raise ValueError(f"Original statistics do not reproduce: {model}")
        checked.append({"model": model, "raw_responses_sha256": sha(raw),
                        "items": len(rescored), "A_m": new["A_m_size"]})
    return checked


def check_new_margins(result, old_path):
    old = {r["model"]: r for r in json.loads(old_path.read_text(encoding="utf-8"))
           if r["cohort"] == "all_selected"}
    for model, metrics in result["models"].items():
        mapping = {}
        for e in ("S", "F"):
            for outcome in OUTCOMES:
                mapping[f"single.{e}.{outcome}"] = f"Single_{e}.on_A.{outcome}"
                mapping[f"order_mean.{e}.{outcome}"] = f"DR_{e}.{outcome}"
                for order, conditions in ORDERS.items():
                    condition = conditions[0 if e == "S" else 1]
                    mapping[f"{order}.{e}.{outcome}"] = f"DR_{e}.{condition}.{outcome}"
        for new_key, old_key in mapping.items():
            new, expected = metrics[new_key], old[model]["metrics"][old_key]
            if (new["numerator"], new["denominator"]) != (expected["numerator"], expected["denominator"]):
                raise ValueError(f"New joint margins disagree with saved metrics: {model}/{new_key}")


def check_original_draws(draw_metrics, base, panel, names):
    indices = [names.index(m) for m in panel]
    s = draw_metrics["order_mean.lure_reduction.S"]
    f = draw_metrics["order_mean.lure_reduction.F"]
    arrays = {"Resistance_S": rate(s[0], s[1])[:, indices].mean(axis=1),
              "Resistance_F": rate(f[0], f[1])[:, indices].mean(axis=1)}
    old = read_jsonl(base / "bootstrap_draws.jsonl")
    for key, values in arrays.items():
        np.testing.assert_allclose(values, [r["panel"][key] for r in old], rtol=0, atol=1e-10)


def fmt(metric, counts=False):
    if metric["estimate"] is None:
        return "NA"
    value = f"{metric['estimate']:.2f} [{metric['lower']:.2f}, {metric['upper']:.2f}]"
    if counts:
        value += f" ({metric['numerator']}/{metric['denominator']})"
    return value


def render_report(data):
    primary = data["source_cluster"]
    panel = primary["panel"]
    lines = ["# Matched S/F answering outcomes", "",
        "Reanalysis of saved main-evaluation responses; no model inference was rerun.", "",
        "## Estimands and uncertainty", "",
        "All results condition on each model's original A_m: gold answers both without evidence and with G alone. "
        "The same A_m is used for S, F, both orders, and the single-to-mixed comparisons. "
        "Gold means a correct selected answer, not evidence of internal reliance on G.", "",
        "G-first pairs G+S with G+F; misleading-first pairs S+G with F+G. "
        "Every joint table is formed within model, item, and order before aggregation. "
        "Not-gold includes lure, abstention, other wrong options, and invalid outputs.", "",
        "Panel estimates are arithmetic means of six model-specific rates, not pooled model-item rates. "
        "Models have different fixed cohorts. Each conditional rate uses that model and order's S-mixed gold count "
        "as its denominator. There is no single binomial denominator for a panel mean.", "",
        "Intervals are exploratory, unadjusted 95% percentile intervals from 10,000 paired pooled "
        "source-cluster draws (489 clusters; seed 2027). Each draw is shared across all models, conditions, "
        "and orders. Items remain weighted equally within each model; models remain weighted equally. "
        "The original six-model panel is held fixed. Undefined replicate rates are never replaced with zero "
        "or handled by dropping a model. Dependency-group sensitivity uses the same fixed panel and 132 groups.", "",
        "These intervals quantify variation over dataset clusters conditional on the saved responses and cohorts. "
        "They do not cover decoding randomness, uncertainty in baseline screening, or a population of models. "
        "At observed rates of zero or 100%, percentile bootstrap intervals can be degenerate; "
        "this does not establish population certainty.", "",
        "## Mixed-condition accuracy and conditional outcomes", "",
        "Values are percentages with source-cluster 95% intervals.", "",
        "| Order | S-mixed gold | F-mixed gold | F gold given S gold | F lure given S gold |",
        "| --- | --- | --- | --- | --- |"]
    for order in ORDERS:
        keys = (f"{order}.S.gold", f"{order}.F.gold", f"{order}.F_given_S_gold.gold",
                f"{order}.F_given_S_gold.lure")
        lines.append("| " + order + " | " + " | ".join(fmt(panel[k]) for k in keys) + " |")
    lines += ["", "## Paired success tables", "",
              "Cells below are model-equal mean percentages of A_m, with source-cluster intervals. "
              "They sum to 100% within each order (apart from rounding).", ""]
    for order in ORDERS:
        p = {c: panel[f"{order}.pair.{c}"] for c in CELLS}
        lines += [f"### {order}", "", "| S outcome | F gold | F not gold |",
                  "| --- | --- | --- |",
                  f"| S gold | {fmt(p['both_gold'])} | {fmt(p['S_only_gold'])} |",
                  f"| S not gold | {fmt(p['F_only_gold'])} | {fmt(p['neither_gold'])} |", ""]
        gap = panel[f"{order}.gold_gap.S_minus_F"]
        lines += [f"Paired accuracy difference (S minus F): {fmt(gap)} percentage points.", "",
                  "Per-model counts; columns are the four joint cells, not four separately filtered cohorts.", "",
                  "| Model | A_m | Both gold | S gold only | F gold only | Neither gold |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for model in primary["panel_models"]:
            m = primary["models"][model]
            cells = [str(m[f"{order}.pair.{c}"]["numerator"]) for c in CELLS]
            lines.append(f"| {model} | {m[f'{order}.S.gold']['denominator']} | " + " | ".join(cells) + " |")
        lines += ["", "| Model | S-mixed gold, % [CI] (n/N) | F-mixed gold, % [CI] (n/N) | "
                  "F gold given S gold, % [CI] (n/N) | F lure given S gold, % [CI] (n/N) |",
                  "| --- | --- | --- | --- | --- |"]
        for model in primary["panel_models"]:
            m = primary["models"][model]
            keys = (f"{order}.S.gold", f"{order}.F.gold", f"{order}.F_given_S_gold.gold",
                    f"{order}.F_given_S_gold.lure")
            lines.append("| " + model + " | " + " | ".join(fmt(m[k], True) for k in keys) + " |")
        lines += [""]
    lines += ["## Complete F outcomes conditional on S-mixed gold", "",
              "These five outcomes sum to 100%; gold and lure alone need not do so.", "",
              "| Order | Gold | Lure | Abstain | Other | Invalid |",
              "| --- | --- | --- | --- | --- | --- |"]
    for order in ORDERS:
        lines.append("| " + order + " | " + " | ".join(
            fmt(panel[f"{order}.F_given_S_gold.{o}"]) for o in OUTCOMES) + " |")
    lines += ["", "## Single-to-mixed changes on the same A_m", "",
              "All outcomes are percentages; reduction is single-condition lure minus mixed-condition lure, "
              "in percentage points. Order means equally weight the two already evaluated orders.", "",
              "| Evidence | Condition | Gold | Lure | Abstain | Other | Invalid | Lure reduction (pp) |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for e in ("S", "F"):
        for condition in ("single", *ORDERS, "order_mean"):
            cells = [fmt(panel[f"{condition}.{e}.{o}"]) for o in OUTCOMES]
            reduction = "N/A" if condition == "single" else fmt(panel[f"{condition}.lure_reduction.{e}"])
            lines.append(f"| {e} | {condition} | " + " | ".join(cells) + f" | {reduction} |")
    lines += ["", "## Dependency-group sensitivity", "",
              "The point estimates are unchanged. These intervals account for broader construction dependencies. "
              "All FRED items share one dependency group, so resampling that group can substantially widen intervals.", "",
              "| Order | Quantity | Source-cluster CI | Dependency-group CI | Valid dependency draws |",
              "| --- | --- | --- | --- | --- |"]
    for order in ORDERS:
        for suffix in ("S.gold", "F.gold", "F_given_S_gold.gold", "F_given_S_gold.lure", "gold_gap.S_minus_F"):
            key = f"{order}.{suffix}"
            dep = data["dependency_group"]["panel"][key]
            lines.append(f"| {order} | {suffix} | {fmt(panel[key])} | {fmt(dep)} | {dep['valid_replicates']} |")
    lines += ["", "## Validation and reusable files", "",
        "- All 46,032 saved responses were re-scored and matched the original item outcomes and cohort flags.",
        "- All original per-model point statistics reproduced exactly. New paired-table margins matched "
        "the original single and mixed outcome numerators and denominators.",
        "- All 10,000 source-cluster bootstrap draws for both original lure reductions reproduced "
        "the saved draws within 1e-10 percentage points.",
        "- `statistics.json` contains all counts, rates, intervals, full 5-by-5 joint matrices, and valid replicate counts.",
        "- `metrics.csv` provides the same estimates in long form; `paired_items.csv` permits item-level inspection.",
        "- `provenance.json` records input/code hashes, runtime versions, and validation details.", "",
        "The comparison measures observed answer correspondence between two evidence configurations. "
        "It does not isolate factual validity from scope applicability, establish internal use of G, "
        "or show learning/transfer between the independently prompted conditions."]
    return "\n".join(lines) + "\n"


def write_outputs(output, data, models, provenance):
    output.mkdir(parents=True, exist_ok=False)
    for filename, value in (("statistics.json", data), ("provenance.json", provenance)):
        (output / filename).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (output / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = ("clustering", "model", "metric", "numerator", "denominator", "unit", "estimate",
                  "confidence", "lower", "upper", "valid_replicates", "invalid_replicates")
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for clustering in ("source_cluster", "dependency_group"):
            result = data[clustering]
            for model, metrics in {**result["models"], "equal_model_panel": result["panel"]}.items():
                for key, value in metrics.items():
                    writer.writerow({"clustering": clustering, "model": model, "metric": key, **value})
    with (output / "paired_items.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = ("model", "item_id", "source", "dimension", "cluster_id", "analysis_group_id",
                  "order", "S_condition", "F_condition", "S_outcome", "F_outcome", "joint_cell")
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for model, rows in sorted(models.items()):
            for row in sorted(rows, key=lambda r: r["item_id"]):
                if not row["A_m"]:
                    continue
                for order, (s, f) in ORDERS.items():
                    so = next(o for o in OUTCOMES if row["outcomes"][s][o])
                    fo = next(o for o in OUTCOMES if row["outcomes"][f][o])
                    cell = CELLS[2 * (so != "gold") + (fo != "gold")]
                    writer.writerow({**{k: row[k] for k in fields[1:6]}, "model": model,
                                     "order": order, "S_condition": s, "F_condition": f,
                                     "S_outcome": so, "F_outcome": fo, "joint_cell": cell})
    (output / "report.md").write_text(render_report(data), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Output already exists; choose a new revision directory")
    models = load_models(args.base / "item_metrics.jsonl")
    validation = check_saved_inputs(models, args.base)
    original_bootstrap = json.loads((args.base / "bootstrap.json").read_text(encoding="utf-8"))
    panel = original_bootstrap["panel_models"]
    if set(panel) != set(models):
        raise ValueError("This reanalysis expects the original complete six-model panel")
    replicates, seed = original_bootstrap["replicates"], original_bootstrap["seed"]
    source, _, source_draws = cluster_analysis(models, panel, "cluster_id", replicates, seed)
    check_new_margins(source, args.base / "statistics.json")
    check_original_draws(source_draws, args.base, panel, sorted(models))
    dependency, _, _ = cluster_analysis(models, panel, "analysis_group_id", replicates, seed)
    data = {"cohort": "Original model-specific A_m, identical across conditions and orders",
            "panel_weighting": "Equal model weights; fixed original panel; no pooled conditional rate",
            "uncertainty": "Exploratory unadjusted 95% paired pooled cluster percentile bootstrap",
            "outcome_order": OUTCOMES, "order_pairs": ORDERS,
            "source_cluster": source, "dependency_group": dependency}
    inputs = [args.base / filename for filename in ("item_metrics.jsonl", "statistics.json", "bootstrap.json",
                                                   "bootstrap_draws.jsonl", "provenance.json")]
    provenance = {"input_sha256": {str(p.relative_to(ROOT)): sha(p) for p in inputs},
                  "code_sha256": {str(p.relative_to(ROOT)): sha(p) for p in
                                    (Path(__file__), ROOT / "experiments/analysis/metrics.py",
                                     ROOT / "experiments/main_eval/protocol.py")},
                  "python": sys.version, "numpy": np.__version__, "replicates": replicates, "seed": seed,
                  "raw_responses_verified": validation,
                  "validation": ["Saved raw response hashes verified; raw answers re-scored with existing scorer",
                                 "Original item outcomes, fixed cohort flags, dataset identities and all point statistics reproduced",
                                 "New joint margins reproduce old single/mixed outcome counts exactly",
                                 "10,000 source-cluster draws for both old lure reductions match within 1e-10 pp"],
                  "inference_rerun": False}
    write_outputs(args.output, data, models, provenance)
    print(args.output.resolve())
    for order in ORDERS:
        print(json.dumps({k: source["panel"][f"{order}.{k}"] for k in
                          ("S.gold", "F.gold", "F_given_S_gold.gold", "F_given_S_gold.lure")}))


if __name__ == "__main__":
    main()
