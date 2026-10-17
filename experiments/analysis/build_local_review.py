"""Reproduce the local descriptive review without changing inference artifacts."""
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.main_eval.protocol import CONDITION_IDS, source_hashes, file_hash
from experiments.main_eval.verify import verify_run

BASE = ROOT / "experiments/outputs/analysis"
OUT = BASE / "local-review-v1"
OUT.mkdir(exist_ok=True)
ORDER = ["qwen3-8b", "qwen3-14b", "llama31-8b-instruct", "gemma3-12b-it", "olmo2-13b-instruct", "qwen3-32b-awq"]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def rate(rows, condition, outcome="lure"):
    return mean(r["outcomes"][condition][outcome] for r in rows) if rows else None


all_stats = {}
all_items = {}
panels = {}
audit = {"created_utc": datetime.now(timezone.utc).isoformat(), "runs": [], "total_responses": 0, "independent_metric_checks": 0}
current_hashes = source_hashes()
historical_hashes = None
raw_example = {}
example_id = "fred_time_d726bec83941_natural_v5"
for split, size in (("main", 959), ("supplement", 25)):
    run = f"{split}-v100-full-v1"
    analyses = BASE / run
    summaries = read(analyses / "statistics.json")
    all_stats[split] = {s["model"]: s for s in summaries if s["cohort"] == "all_selected"}
    panels[split] = read(analyses / "bootstrap.json")
    item_models = defaultdict(dict)
    for row in jsonl(analyses / "item_metrics.jsonl"):
        item_models[row["model"]][row["item_id"]] = row
    all_items[split] = item_models
    for model in ORDER:
        directory = ROOT / "experiments/outputs/main_eval" / model / run
        manifest, metadata, rows = verify_run(directory)
        grouped = defaultdict(dict)
        for row in rows:
            condition = row["condition_id"]
            assert condition not in grouped[row["item_id"]]
            grouped[row["item_id"]][condition] = row
        assert len(rows) == size * 8 and len(grouped) == size
        assert all(set(conditions) == set(CONDITION_IDS) for conditions in grouped.values())
        summary = all_stats[split][model]
        baseline = [x for x in grouped.values() if x["closed_book"]["response"] == x["closed_book"]["answer_key"]["gold"]]
        cohort = [x for x in baseline if x["g_only"]["response"] == x["g_only"]["answer_key"]["gold"]]
        assert len(baseline) == summary["B_m_size"] and len(cohort) == summary["A_m_size"]
        # Independently recompute all single/mixed lure numerators from raw answers.
        for kind, single, mixed in (("F", "f_only", ("g_then_f", "f_then_g")), ("S", "s_only", ("g_then_s", "s_then_g"))):
            comparisons = [(f"LR_{kind}", baseline, (single,)), (f"Single_{kind}.on_A.lure", cohort, (single,)), (f"DR_{kind}", cohort, mixed)]
            comparisons.extend((f"DR_{kind}.{condition}", cohort, (condition,)) for condition in mixed)
            for key, selected, conditions in comparisons:
                numerator = sum(x[c]["response"] == x[c]["answer_key"]["lure"] for x in selected for c in conditions)
                metric = summary["metrics"][key]
                assert (numerator, len(selected) * len(conditions)) == (metric["numerator"], metric["denominator"])
                audit["independent_metric_checks"] += 1
        if split == "main":
            raw_example[model] = {k: {"response": v["response"], "prompt": v["messages"][0]["content"]} for k, v in grouped[example_id].items()}
        same_code = manifest["identity"]["code_sha256"] == current_hashes
        if historical_hashes is None:
            historical_hashes = manifest["identity"]["code_sha256"]
        if manifest["identity"]["code_sha256"] != historical_hashes:
            raise ValueError(f"Saved inference source differs across runs: {split}/{model}")
        audit["total_responses"] += len(rows)
        audit["runs"].append({"split": split, "model": model, "items": size, "responses": len(rows), "exactly_eight_conditions": True, "current_code_matches_run": same_code, "backend": metadata["backend"], "weight_precision": summary["weight_precision"], "gpus": metadata["runtime"]["gpus"], "inference_seconds": sum(r["elapsed_seconds"] for r in rows), "raw_sha256": file_hash(directory / "raw_responses.jsonl")})
        del rows, grouped, baseline, cohort

main = all_stats["main"]
metric_keys = ["closed_book.gold", "g_only.gold", "LR_F", "LR_S", "Single_F.on_A.lure", "Single_S.on_A.lure", "DR_F", "DR_S", "DR_F.g_then_f", "DR_F.f_then_g", "DR_S.g_then_s", "DR_S.s_then_g", "Single_F.on_A.gold", "Single_S.on_A.gold", "DR_F.gold", "DR_S.gold", "DR_F.abstain", "DR_S.abstain", "DR_F.invalid", "DR_S.invalid"]
panel_rates = {split: {key: mean(s["metrics"][key]["rate"] for s in values.values()) for key in metric_keys} for split, values in all_stats.items()}
cohorts = defaultdict(list)
for s in read(BASE / "main-v100-full-v1/statistics.json"):
    cohorts[s["cohort"]].append(s)
cohort_rates = {key: {"items_per_model": values[0]["items"], "rates": {m: mean(s["metrics"][m]["rate"] for s in values) for m in metric_keys}} for key, values in cohorts.items()}
common = set.intersection(*(set(k for k, r in values.items() if r["A_m"]) for values in all_items["main"].values()))
common_rates = {model: {condition: rate([rows[k] for k in sorted(common)], condition) for condition in CONDITION_IDS} for model, rows in all_items["main"].items()}
transitions = {}
for model, items in all_items["main"].items():
    transitions[model] = dict(Counter(f"G_then_F_lure={r['outcomes']['g_then_f']['lure']},F_then_G_lure={r['outcomes']['f_then_g']['lure']}" for r in items.values() if r["A_m"]))

draws = jsonl(BASE / "main-v100-full-v1/bootstrap_draws.jsonl")
sensitivity = {}
for excluded in ("qwen3-32b-awq", "olmo2-13b-instruct"):
    members = [m for m in ORDER if m != excluded]
    result = {}
    for key in panels["main"]["panel_contrasts"]:
        values = sorted(mean(d["models"][m][key] for m in members) for d in draws)
        def quantile(prob):
            pos = (len(values) - 1) * prob
            lo = int(pos)
            return values[lo] + (pos - lo) * (values[min(lo + 1, len(values) - 1)] - values[lo])
        alpha = .0125 if key in ("LR", "DR") else .025
        result[key] = {"estimate_pp": mean(panels["main"]["model_contrasts"][m][key]["estimate_pp"] for m in members), "confidence": 1 - 2 * alpha, "lower_pp": quantile(alpha), "upper_pp": quantile(1 - alpha)}
    sensitivity["exclude_" + excluded] = {"label": "exploratory sensitivity; paired draws reused; no extra multiplicity adjustment", "members": members, "contrasts": result}
del draws

dataset = jsonl(ROOT / "construction/data/main_split.jsonl")
key_balance = {role: dict(Counter(r["answer_key"][role] for r in dataset)) for role in ("gold", "lure")}
audit["main_answer_key_balance"] = key_balance
audit["all_analysis_inputs_verified"] = True
audit["source_hashes"] = current_hashes
save("audit.json", audit)
dependency = read(BASE / "main-v100-full-v1-dependency/bootstrap.json")
review = {"panel_aggregation": "equal-weight average of six model-specific conditional rates", "panel_rates": panel_rates, "cohort_rates": cohort_rates, "main_panel_intervals": panels["main"]["panel_contrasts"], "dependency_group_sensitivity": dependency["panel_contrasts"], "panel_sensitivity": sensitivity, "common_A_cohort": {"items": len(common), "selection": "All six models correct both closed-book and G-only; descriptive robustness check", "model_rates": common_rates}, "F_order_transitions": transitions}
save("summary.json", review)
example = next(r for r in dataset if r["id"] == example_id)
assert all(v["closed_book"]["response"] == "A" and v["g_only"]["response"] == "A" and v["g_then_f"]["response"] == "A" and v["f_then_g"]["response"] == "B" for v in raw_example.values())
for values in raw_example.values():
    for condition, order in (("g_then_f", ("correct", "false")), ("f_then_g", ("false", "correct"))):
        prompt = values[condition]["prompt"]
        assert prompt.index(example["evidence"][order[0]]["text"]) < prompt.index(example["evidence"][order[1]]["text"])
save("order_example.json", {"selection": "post-hoc illustration, not a representative sample or independent truth verification", "item": example, "model_responses_and_prompts": raw_example})

csv_rows = []
for split, summaries in all_stats.items():
    for model in ORDER:
        s = summaries[model]
        csv_rows.append({"split": split, "model": model, "N": s["items"], "B_m": s["B_m_size"], "A_m": s["A_m_size"], **{k: s["metrics"][k]["rate"] for k in metric_keys}, **{k: v["percentage_points"] for k, v in s["contrasts"].items()}})
with (OUT / "model_summary.csv").open("w", encoding="utf-8-sig", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
    writer.writeheader()
    writer.writerows(csv_rows)

print(OUT.resolve())
