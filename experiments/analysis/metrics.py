"""Conditional lure rates and paired source-cluster bootstrap (standard library)."""
from collections import Counter, defaultdict
import random

from experiments.main_eval.protocol import CONDITION_IDS, LETTERS, canonical

OUTCOMES = ("gold", "lure", "abstain", "other", "invalid")
CONTRASTS = ("LR", "DR", "Resistance_F", "Resistance_S", "Order_F", "Order_S")


def item_results(rows):
    grouped = defaultdict(dict)
    for row in rows:
        item_id, condition = row["item_id"], row["condition_id"]
        if condition not in CONDITION_IDS or condition in grouped[item_id]:
            raise ValueError("Use first responses only: unknown or duplicate item/condition")
        grouped[item_id][condition] = row
    items = []
    fields = ("answer_key", "source", "dimension", "analysis_group_id", "evaluation_eligible")
    for item_id, conditions in sorted(grouped.items()):
        if set(conditions) != set(CONDITION_IDS):
            raise ValueError(f"Incomplete eight-condition item: {item_id}")
        if len({canonical({k: r.get(k) for k in fields}) for r in conditions.values()}) != 1:
            raise ValueError("Scoring labels or grouping metadata differ across conditions")
        first = conditions["closed_book"]
        key = first["answer_key"]
        if (set(key) != {"gold", "lure", "abstain"} or len(set(key.values())) != 3
                or any(label not in tuple(LETTERS) for label in key.values())):
            raise ValueError("Invalid answer key")
        result = {k: first.get(k) for k in ("item_id", *fields[1:])}
        answers = {}
        for condition, row in conditions.items():
            valid = row.get("finish_reason") == "stop" and row.get("response") in tuple(LETTERS)
            answer = row["response"] if valid else None
            outcome = next((role for role, label in key.items() if valid and answer == label),
                           "other" if valid else "invalid")
            answers[condition] = {"answer": answer, **{o: int(o == outcome) for o in OUTCOMES}}
        result["outcomes"] = answers
        result["B_m"] = bool(answers["closed_book"]["gold"])
        result["G_correct"] = bool(answers["g_only"]["gold"])
        result["A_m"] = result["B_m"] and result["G_correct"]
        items.append(result)
    return items


def fraction(numerator, denominator):
    return {"numerator": numerator, "denominator": denominator,
            "rate": numerator / denominator if denominator else None}


def difference(a, b):
    return None if a is None or b is None else 100 * (a - b)


def statistics(items):
    baseline = [r for r in items if r["B_m"]]
    cohort = [r for r in items if r["A_m"]]
    metrics, contrasts = {}, {}

    def add(prefix, selected, conditions):
        for outcome in OUTCOMES:
            metrics[f"{prefix}.{outcome}"] = fraction(
                sum(r["outcomes"][c][outcome] for r in selected for c in conditions),
                len(selected) * len(conditions))

    for condition in CONDITION_IDS:
        add(condition, items, (condition,))
    for evidence, single, mixed in (("F", "f_only", ("g_then_f", "f_then_g")),
                                    ("S", "s_only", ("g_then_s", "s_then_g"))):
        add(f"LR_{evidence}", baseline, (single,))
        add(f"Single_{evidence}.on_A", cohort, (single,))
        add(f"DR_{evidence}", cohort, mixed)
        metrics[f"LR_{evidence}"] = metrics[f"LR_{evidence}.lure"]
        metrics[f"DR_{evidence}"] = metrics[f"DR_{evidence}.lure"]
        for condition in mixed:
            add(f"DR_{evidence}.{condition}", cohort, (condition,))
            metrics[f"DR_{evidence}.{condition}"] = metrics[f"DR_{evidence}.{condition}.lure"]
            contrasts[f"Resistance_{evidence}.{condition}"] = {"percentage_points": difference(
                metrics[f"Single_{evidence}.on_A.lure"]["rate"],
                metrics[f"DR_{evidence}.{condition}"]["rate"])}
        contrasts[f"Resistance_{evidence}"] = {"percentage_points": difference(
            metrics[f"Single_{evidence}.on_A.lure"]["rate"], metrics[f"DR_{evidence}"]["rate"])}
        # Positive means more lure choices when misleading evidence is last.
        contrasts[f"Order_{evidence}"] = {"percentage_points": difference(
            metrics[f"DR_{evidence}.{mixed[0]}"]["rate"],
            metrics[f"DR_{evidence}.{mixed[1]}"]["rate"])}
    for name in ("LR", "DR"):
        contrasts[f"Delta_{name}"] = {"percentage_points": difference(
            metrics[f"{name}_F"]["rate"], metrics[f"{name}_S"]["rate"])}
    return {"items": len(items), "B_m_size": len(baseline),
            "G_correct_size": sum(r["G_correct"] for r in items), "A_m_size": len(cohort),
            "metrics": metrics, "contrasts": contrasts}


def _contribution(item):
    o, b, a = item["outcomes"], int(item["B_m"]), int(item["A_m"])
    return (b, a, b * (o["f_only"]["lure"] - o["s_only"]["lure"]),
            a * (o["g_then_f"]["lure"] + o["f_then_g"]["lure"]
                 - o["g_then_s"]["lure"] - o["s_then_g"]["lure"]),
            a * (2 * o["f_only"]["lure"] - o["g_then_f"]["lure"] - o["f_then_g"]["lure"]),
            a * (2 * o["s_only"]["lure"] - o["g_then_s"]["lure"] - o["s_then_g"]["lure"]),
            a * (o["g_then_f"]["lure"] - o["f_then_g"]["lure"]),
            a * (o["g_then_s"]["lure"] - o["s_then_g"]["lure"]))


def _contrasts(totals):
    b, a, lr, dr, rf, rs, of, os = totals
    values = (None if not b else 100 * lr / b,
              None if not a else 100 * dr / (2 * a),
              None if not a else 100 * rf / (2 * a),
              None if not a else 100 * rs / (2 * a),
              None if not a else 100 * of / a,
              None if not a else 100 * os / a)
    return dict(zip(CONTRASTS, values))


def _percentile(values, probability):
    values = sorted(values)
    position = (len(values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (position - lower) * (values[upper] - values[lower])


def _interval(estimate, values, confidence):
    alpha = (1 - confidence) / 2
    return {"estimate_pp": estimate, "confidence": confidence, "valid_replicates": len(values),
            "lower_pp": _percentile(values, alpha) if values else None,
            "upper_pp": _percentile(values, 1 - alpha) if values else None}


def paired_bootstrap(models, replicates=10000, seed=2027, cluster_field="cluster_id", min_clusters=20):
    """Same pooled cluster draw for all conditions/models; fixed eligible panel.

    Cluster sums avoid materializing duplicated items. Each sampled cluster adds
    all its item contributions, preserving item weighting and paired differences.
    """
    if not models or replicates < 1 or min_clusters < 1:
        raise ValueError("Models, positive replicates and positive min_clusters are required")
    if cluster_field not in ("cluster_id", "analysis_group_id"):
        raise ValueError("Unsupported cluster field")
    reference, membership = None, {}
    for name, items in sorted(models.items()):
        mapping = {}
        for item in items:
            if item["item_id"] in mapping or not item.get(cluster_field):
                raise ValueError("Duplicate item or missing cluster")
            mapping[item["item_id"]] = (item["source"], item[cluster_field])
        if reference is not None and mapping != reference:
            raise ValueError("Dataset/cluster memberships differ across models")
        reference = mapping
        membership[name] = mapping
    clusters = sorted(set(reference.values()))
    eligibility = {name: {"A_m_clusters": len({mapping[r["item_id"]] for r in models[name] if r["A_m"]})}
                   for name, mapping in membership.items()}
    panel = [name for name in sorted(models) if eligibility[name]["A_m_clusters"] >= min_clusters]
    for name in eligibility:
        eligibility[name]["eligible"] = name in panel
    arrays, points = {}, {}
    for name, items in sorted(models.items()):
        sums = {c: [0] * 8 for c in clusters}
        for item in items:
            target = sums[membership[name][item["item_id"]]]
            for i, value in enumerate(_contribution(item)):
                target[i] += value
        arrays[name] = [sums[c] for c in clusters]
        points[name] = _contrasts([sum(v[i] for v in arrays[name]) for i in range(8)])
    summary = {"method": "paired pooled cluster percentile bootstrap", "cluster_field": cluster_field,
               "clusters": len(clusters), "replicates": replicates, "seed": seed,
               "min_A_m_clusters": min_clusters, "panel_models": panel, "model_eligibility": eligibility,
               "panel_weighting": "equal model weights; fixed panel selected on original sample",
               "primary_family": ["LR", "DR"], "primary_confidence": 0.975,
               "secondary_confidence": 0.95, "secondary_inference": "exploratory; unadjusted",
               "panel_contrasts": {}, "model_contrasts": {}}
    draws, rng = [], random.Random(seed)
    if panel:
        for _ in range(replicates):
            weights = Counter(rng.choices(range(len(clusters)), k=len(clusters)))
            sampled = {}
            for name in panel:
                totals = [0] * 8
                for index, weight in weights.items():
                    for i, value in enumerate(arrays[name][index]):
                        totals[i] += weight * value
                sampled[name] = _contrasts(totals)
            aggregate = {}
            for key in CONTRASTS:
                values = [sampled[name][key] for name in panel]
                aggregate[key] = None if None in values else sum(values) / len(values)
            draws.append({"models": sampled, "panel": aggregate})
    for key in CONTRASTS:
        confidence = 0.975 if key in ("LR", "DR") else 0.95
        estimate = sum(points[name][key] for name in panel) / len(panel) if panel else None
        values = [d["panel"][key] for d in draws if d["panel"][key] is not None]
        summary["panel_contrasts"][key] = _interval(estimate, values, confidence)
    for name in sorted(models):
        summary["model_contrasts"][name] = {
            key: _interval(points[name][key],
                           [d["models"][name][key] for d in draws
                            if name in d["models"] and d["models"][name][key] is not None], 0.95)
            for key in CONTRASTS}
    return summary, draws
