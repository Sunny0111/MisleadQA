"""Export descriptive response diagnostics as numerical JSON.

Uses saved responses only; does not run models or change frozen inputs.
"""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from misleadqa_utils import CURRENT_VERSION
INPUT = ROOT / "experiments/outputs/analysis/main-v100-full-v1/item_metrics.jsonl"
CASE_INPUT = ROOT / "experiments/outputs/analysis/local-review-v1/order_example.json"
OUTCOMES = ("gold", "lure", "abstain", "other", "invalid")
MODELS = [
    ("qwen3-8b", "Qwen3-8B"),
    ("qwen3-14b", "Qwen3-14B"),
    ("qwen3-32b-awq", "Qwen3-32B-AWQ"),
    ("llama31-8b-instruct", "Llama-3.1-8B"),
    ("gemma3-12b-it", "Gemma-3-12B"),
    ("olmo2-13b-instruct", "OLMo-2-13B"),
]
PATTERNS = {(1, 0): "F_only", (1, 1): "both", (0, 1): "S_only", (0, 0): "neither"}
CONDITIONS = {
    "F": ("f_only", ("g_then_f", "f_then_g")),
    "S": ("s_only", ("g_then_s", "s_then_g")),
}
grouped = defaultdict(list)
for line in INPUT.read_text(encoding="utf-8").splitlines():
    row = json.loads(line)
    grouped[row["model"]].append(row)
assert set(grouped) == {alias for alias, _ in MODELS}

by_model = {}
for alias, _ in MODELS:
    rows = grouped[alias]
    assert len(rows) == 959
    b = [r for r in rows if r["B_m"]]
    a = [r for r in rows if r["A_m"]]
    patterns = Counter((r["outcomes"]["f_only"]["lure"],
                        r["outcomes"]["s_only"]["lure"]) for r in b)
    info = {"B_items": len(b), "A_items": len(a),
            "pattern_counts": {name: patterns[key] for key, name in PATTERNS.items()},
            "pattern_percentages": {name: 100 * patterns[key] / len(b) for key, name in PATTERNS.items()},
            "transitions": {}}
    assert sum(info["pattern_counts"].values()) == len(b)
    for t, (single, orders) in CONDITIONS.items():
        selected = [r for r in a if r["outcomes"][single]["lure"]]
        assert selected
        info["transitions"][t] = {"initial_lure_items": len(selected), "orders": {}}
        for order in orders:
            counts = {out: sum(r["outcomes"][order][out] for r in selected) for out in OUTCOMES}
            assert sum(counts.values()) == len(selected)
            info["transitions"][t]["orders"][order] = {
                "counts": counts,
                "percentages": {out: 100 * counts[out] / len(selected) for out in OUTCOMES},
            }
    by_model[alias] = info

panel_patterns = {name: sum(v["pattern_percentages"][name] for v in by_model.values()) / 6
                  for name in PATTERNS.values()}
panel_orders = {}
panel_order_means = {}
for t, (_, orders) in CONDITIONS.items():
    for order in orders:
        panel_orders[order] = {
            out: sum(v["transitions"][t]["orders"][order]["percentages"][out]
                     for v in by_model.values()) / 6 for out in OUTCOMES}
    panel_order_means[t] = {out: sum(panel_orders[o][out] for o in orders) / 2 for out in OUTCOMES}
assert abs(sum(panel_patterns.values()) - 100) < 1e-10
for values in panel_orders.values():
    assert abs(sum(values.values()) - 100) < 1e-10

case = json.loads(CASE_INPUT.read_text(encoding="utf-8"))
item = case["item"]
case_id = item["id"]
release_items = [json.loads(line) for line in
                 (ROOT / "construction/data/main_split.jsonl").read_text(encoding="utf-8").splitlines()]
release_item = next(r for r in release_items if r["id"] == case_id)
assert item["question"] == release_item["question"]
assert item["answer_key"] == release_item["answer_key"]
assert all(item["evidence"][k]["text"] == release_item["evidence"][k]["text"] for k in item["evidence"])
case_rows = {r["model"]: r for rows in grouped.values() for r in rows if r["item_id"] == case_id}
for alias, _ in MODELS:
    assert case_rows[alias]["B_m"] and case_rows[alias]["A_m"]
    for condition, value in case["model_responses_and_prompts"][alias].items():
        assert case_rows[alias]["outcomes"][condition]["answer"] == value["response"]

paper_dir = ROOT / "experiments/outputs/paper"
paper_dir.mkdir(parents=True, exist_ok=True)

evidence = {
    "dataset_version": CURRENT_VERSION,
    "method": "descriptive matched response analysis; rates computed within model and averaged equally",
    "input": INPUT.relative_to(ROOT).as_posix(),
    "input_sha256": hashlib.sha256(INPUT.read_bytes()).hexdigest(),
    "case_input": CASE_INPUT.relative_to(ROOT).as_posix(),
    "case_input_sha256": hashlib.sha256(CASE_INPUT.read_bytes()).hexdigest(),
    "case_item_id": case_id,
    "patterns": {"cohort": "B_m", "panel_percentages": panel_patterns},
    "recovery": {"cohort": "U_m,t: A_m items with single-evidence lure selection",
                 "panel_order_percentages": panel_orders, "panel_order_mean_percentages": panel_order_means},
    "models": by_model,
    "case_outcomes": {alias: {c: case_rows[alias]["outcomes"][c] for c in
                       ("closed_book", "g_only", "f_only", "s_only", "g_then_f", "f_then_g", "g_then_s", "s_then_g")}
                      for alias, _ in MODELS},
    "inference": "descriptive; no added confidence intervals or mechanism identification",
}
(paper_dir / "analysis_evidence.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
print((paper_dir / "analysis_evidence.json").resolve())
