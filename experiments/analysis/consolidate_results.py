"""Consolidate saved experimental results without changing or rerunning inference."""
import hashlib
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments/outputs/01a1040f-0ef4-71f2-9563-55b4e111a9a1"
ANALYSIS = "experiments/outputs/analysis"
MODELS = ("qwen3-8b", "qwen3-14b", "llama31-8b-instruct", "gemma3-12b-it",
          "olmo2-13b-instruct", "qwen3-32b-awq")
LABELS = ("Qwen3-8B", "Qwen3-14B", "Llama-3.1-8B", "Gemma-3-12B", "OLMo-2-13B", "Qwen3-32B-AWQ")
PATHS = {
    "main": f"{ANALYSIS}/main-v100-full-v1/statistics.json",
    "main_ci": f"{ANALYSIS}/main-v100-full-v1/bootstrap.json",
    "dependency_ci": f"{ANALYSIS}/main-v100-full-v1-dependency/bootstrap.json",
    "supplement": f"{ANALYSIS}/supplement-v100-full-v1/statistics.json",
    "supplement_ci": f"{ANALYSIS}/supplement-v100-full-v1/bootstrap.json",
    "scope": f"{ANALYSIS}/scope_diagnostic/main-scope-full-v2/statistics.json",
    "scope_groups": f"{ANALYSIS}/scope_diagnostic/main-scope-full-v2/groups.json",
    "scope_ci": f"{ANALYSIS}/scope_diagnostic/main-scope-full-v2/bootstrap.json",
    "context": f"{ANALYSIS}/context-faithful-r3/statistics.json",
    "context_pairs": f"{ANALYSIS}/context-faithful-r3/comparisons.json",
    "context_summary": f"{ANALYSIS}/context-faithful-r3/context_summary.json",
    "review": f"{ANALYSIS}/local-review-v1/summary.json",
    "experiment_evidence": "experiments/outputs/paper/experiments_evidence.json",
    "behavior_evidence": "experiments/outputs/paper/analysis_evidence.json",
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build():
    source = {key: json.loads((ROOT / path).read_text(encoding="utf-8")) for key, path in PATHS.items()}
    for filename, expected in source["experiment_evidence"]["input_sha256"].items():
        if sha(ROOT / ANALYSIS / "main-v100-full-v1" / filename) != expected:
            raise ValueError(f"Stale descriptive evidence: {filename}")
    for field in ("input", "case_input"):
        e = source["behavior_evidence"]
        if sha(ROOT / e[field]) != e[field + "_sha256"]:
            raise ValueError(f"Stale behavioral evidence: {field}")

    metrics, pairs, contrasts = [], [], []

    def add_metrics(experiment, model, cohort, values, source_id, intervals=None):
        for key, value in values.items():
            ci = (intervals or {}).get(key, {})
            num, den, rate = value.get("numerator"), value.get("denominator"), value["rate"]
            if den is not None and ((den == 0 and rate is not None)
                    or (den > 0 and abs(num / den - rate) > 1e-12)):
                raise ValueError(f"Rate/denominator mismatch: {experiment}, {model}, {key}")
            metrics.append([experiment, model, cohort, key, num, den, rate, ci.get("confidence"),
                            ci.get("lower"), ci.get("upper"), ci.get("valid_replicates"), source_id])

    for experiment in ("main", "supplement"):
        for entry in source[experiment]:
            add_metrics(experiment, entry["model"], entry["cohort"], entry["metrics"], experiment)
    for name, entry in source["context"].items():
        add_metrics("context_own_cohort", name.split("/")[0], "own_B_A", entry["statistics"]["metrics"], "context")
    for model, entry in source["scope"]["models"].items():
        add_metrics("scope", model, "balanced_200", entry["metrics"], "scope",
                    source["scope_ci"]["intervals"][model])
    for model, groups in source["scope_groups"].items():
        for group, entry in groups.items():
            add_metrics("scope", model, group, entry["metrics"], "scope_groups")
    for group, data in source["experiment_evidence"]["strata"].items():
        for model, entry in data["models"].items():
            add_metrics("main", model, "source_scope:" + group, entry["metrics"], "experiment_evidence")
    for key, pair in source["context_pairs"].items():
        model = key.split("/")[0]
        for row in pair["differences"]:
            if row["n"] and abs(100 * (row["treatment_count"] - row["reference_count"]) / row["n"] - row["delta_pp"]) > 1e-12:
                raise ValueError("Paired difference mismatch")
            pairs.append([model, row["cohort"], row["condition"], row["outcome"], row["n"],
                          row["reference_count"], row["treatment_count"], row["reference_rate"],
                          row["treatment_rate"], row["delta_pp"], "context_pairs"])

    def add_contrasts(label, values, source_id, model="equal_model_mean", inference="exploratory"):
        for key, value in values.items():
            contrasts.append([label, model, key, value.get("estimate_pp"), value.get("confidence"),
                              value.get("lower_pp"), value.get("upper_pp"), value.get("valid_replicates"),
                              inference, source_id])

    for key, label in (("main_ci", "Source-cluster bootstrap"), ("dependency_ci", "Dependency-group bootstrap"),
                       ("supplement_ci", "Condition supplement")):
        add_contrasts(label, source[key]["panel_contrasts"], key, inference=(
            "Main primary LR/DR: 97.5%; secondary: exploratory 95%" if key == "main_ci" else
            "No eligible panel; descriptive only" if key == "supplement_ci" else "Dependency sensitivity"))
        for model, values in source[key]["model_contrasts"].items():
            add_contrasts(label, values, key, model, "Individual-model exploratory intervals")
    add_contrasts("Equal family weights", source["experiment_evidence"]["family_sensitivity"]["contrasts"], "experiment_evidence")
    for name, value in source["review"]["panel_sensitivity"].items():
        add_contrasts(name, value["contrasts"], "review")

    inventory = [
        ["Main answering", 959, 8, 6, 46032, "Complete", "Original prompt; main split"],
        ["Condition supplement", 25, 8, 6, 1200, "Complete", "Separate split; descriptive inference"],
        ["Scope judgment", 200, 3, 6, 3600, "Complete", "Balanced subset of the main split"],
        ["Context-faithful", 959, 8, 6, 46032, "Complete", "Instruction sensitivity; main split"],
    ]
    main_rows = {name: [next(r for r in source[name] if r["model"] == m and r["cohort"] == "all_selected")
                        for m in MODELS] for name in ("main", "supplement")}
    panel = {name: {metric: mean(row["metrics"][metric]["rate"] for row in rows)
                   for metric in ("LR_F", "LR_S", "DR_F", "DR_S")}
             for name, rows in main_rows.items()}
    package = {"title": "MisleadQA consolidated experiment results", "as_of": "2026-10-05",
               "models": list(MODELS), "model_labels": list(LABELS), "formal_response_count": 96864,
               "unique_dataset_items": 984, "inventory": inventory, "panel_rates": panel,
               "main_model_rows": main_rows, "metrics": metrics, "pairs": pairs, "contrasts": contrasts,
               "source_files": [{"id": key, "file": path, "sha256": sha(ROOT / path)} for key, path in PATHS.items()],
               "results": source,
               "inclusion": "Four completed experiments. Smoke and archive duplicates excluded.",
               "provenance": "All source outputs remain unchanged. Existing confidence intervals retained; no new hypothesis tests."}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "all_results.json").write_text(json.dumps(package, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT), "metric_rows": len(metrics), "paired_rows": len(pairs),
                      "contrast_rows": len(contrasts), "formal_responses": 96864}))


if __name__ == "__main__":
    build()
