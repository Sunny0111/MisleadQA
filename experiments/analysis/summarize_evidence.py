"""Derive descriptive sensitivities and outcome summaries from saved results."""
import json
import sys
from pathlib import Path
from collections import defaultdict
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.analysis.metrics import statistics
from experiments.main_eval.protocol import file_hash
from misleadqa_utils import CURRENT_VERSION

OUT = ROOT / "experiments/outputs/paper"
OUT.mkdir(parents=True, exist_ok=True)
BASE = ROOT / "experiments/outputs/analysis/main-v100-full-v1"

def read(path):
    return json.loads(path.read_text(encoding="utf-8"))

def lines(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]

models = ["qwen3-8b", "qwen3-14b", "qwen3-32b-awq", "llama31-8b-instruct", "gemma3-12b-it", "olmo2-13b-instruct"]
main = {s["model"]: s for s in read(BASE / "statistics.json") if s["cohort"] == "all_selected"}
boot = read(BASE / "bootstrap.json")
draws = lines(BASE / "bootstrap_draws.jsonl")
families = {"Qwen3": models[:3], "Llama": [models[3]], "Gemma": [models[4]], "OLMo": [models[5]]}
family = {}
for key in boot["panel_contrasts"]:
    values = sorted(mean(mean(d["models"][m][key] for m in members) for members in families.values()) for d in draws)
    def quantile(p):
        x = p * (len(values) - 1)
        lo = int(x)
        return values[lo] + (x - lo) * (values[min(lo + 1, len(values) - 1)] - values[lo])
    family[key] = {"estimate_pp": mean(mean(boot["model_contrasts"][m][key]["estimate_pp"] for m in members) for members in families.values()), "confidence": .95, "lower_pp": quantile(.025), "upper_pp": quantile(.975)}
del draws
items = defaultdict(list)
for row in lines(BASE / "item_metrics.jsonl"):
    items[row["model"]].append(row)
strata_keys = [("FRED", "time"), ("FRED", "geography"), ("SituatedQA", "time"), ("SituatedQA", "geography"), ("AmbigDocs", "entity")]
strata = {}
for source, dimension in strata_keys:
    key = source + ":" + dimension
    per_model = {m: statistics([r for r in items[m] if r["source"] == source and r["dimension"] == dimension]) for m in models}
    strata[key] = {"items_per_model": next(iter(per_model.values()))["items"], "models": per_model, "panel_rates": {metric: mean(s["metrics"][metric]["rate"] for s in per_model.values()) for metric in ("LR_F", "LR_S", "DR_F", "DR_S")}, "panel_contrasts": {metric: mean(s["contrasts"][metric]["percentage_points"] for s in per_model.values()) for metric in ("Delta_LR", "Delta_DR", "Order_F", "Order_S")}}
outcomes = {prefix: {outcome: 100 * mean(s["metrics"][prefix + "." + outcome]["rate"] for s in main.values()) for outcome in ("gold", "lure", "abstain", "other", "invalid")} for prefix in ("LR_F", "LR_S", "Single_F.on_A", "Single_S.on_A", "DR_F", "DR_S", "DR_F.g_then_f", "DR_F.f_then_g", "DR_S.g_then_s", "DR_S.s_then_g")}
for distribution in outcomes.values():
    assert abs(sum(distribution.values()) - 100) < 1e-10
evidence = {"family_sensitivity": {"families": families, "weighting": "equal model weights within family; equal weights across the four represented families", "intervals": "exploratory 95%; reuse all 10,000 paired source-cluster draws", "contrasts": family}, "strata": strata, "outcomes_percent": outcomes, "input_sha256": {p.name: file_hash(p) for p in [BASE / "statistics.json", BASE / "bootstrap.json", BASE / "bootstrap_draws.jsonl", BASE / "item_metrics.jsonl"]}}
evidence["dataset_version"] = CURRENT_VERSION
(OUT / "experiments_evidence.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")


print(OUT.resolve())
