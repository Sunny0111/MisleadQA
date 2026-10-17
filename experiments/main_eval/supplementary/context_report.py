"""Export numerical summaries from verified context-faithful comparisons."""
import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from experiments.main_eval.protocol import ROOT, file_hash, save_json

OUTCOMES = ("gold", "lure", "abstain", "other", "invalid")
MODELS = ("qwen3-8b", "qwen3-14b", "llama31-8b-instruct", "gemma3-12b-it",
          "olmo2-13b-instruct", "qwen3-32b-awq")


def outcomes(pair, cohort, conditions):
    selected = [r for r in pair["differences"]
                if r["cohort"] == cohort and r["condition"] in conditions]
    if len(selected) != len(conditions) * len(OUTCOMES):
        raise ValueError("Missing or duplicate condition/outcome comparison")
    return {outcome: {field: (1 if field == "delta_pp" else 100)
                     * mean(r[field] for r in selected if r["outcome"] == outcome)
                     for field in ("reference_rate", "treatment_rate", "delta_pp")}
            for outcome in OUTCOMES}


def metric_pair(pair, cohort, conditions):
    result = outcomes(pair, cohort, conditions)["lure"]
    return {"original_pct": result["reference_rate"], "context_pct": result["treatment_rate"],
            "delta_pp": result["delta_pp"]}


def build_report(directory, run_root=ROOT / "experiments/outputs/supplementary",
                 baseline_root=ROOT / "experiments/outputs/main_eval"):
    directory = Path(directory)
    read = lambda name: json.loads((directory / name).read_text(encoding="utf-8"))
    summaries, comparisons, provenance = (read(n) for n in
                                         ("statistics.json", "comparisons.json", "provenance.json"))
    pairs = {name.split("/")[0]: value for name, value in comparisons.items()}
    if set(pairs) != set(MODELS) or len(comparisons) != 6 or len(summaries) != 6:
        raise ValueError("Expected six complete context-only matched comparisons")
    normalization = json.loads((ROOT / "construction/data/version_normalization.json").read_text())
    audit = []
    for model in MODELS:
        treatment = next(p for p in provenance if p.get("directory")
                         and p["identity"]["model_alias"] == model)
        baseline = next(p for p in provenance if p.get("baseline_directory")
                        and p["identity"]["model_alias"] == model)
        baseline_id = Path(baseline["baseline_directory"].replace("\\", "/")).name
        treatment_id = Path(treatment["directory"].replace("\\", "/")).name
        source_paths = [Path(baseline_root) / model / baseline_id,
                        Path(run_root) / model / treatment_id]
        for path, entry in zip(source_paths, (baseline, treatment)):
            for filename, key in (("raw_responses.jsonl", "raw_sha256"), ("model.json", "model_sha256")):
                if file_hash(path / filename) != entry[key]:
                    raise ValueError(f"Saved analysis input changed: {path / filename}")
        closed = []
        for path in source_paths:
            with (path / "raw_responses.jsonl").open(encoding="utf-8") as stream:
                rows = (json.loads(line) for line in stream)
                closed.append({r["item_id"]: r["response"] for r in rows
                               if r["condition_id"] == "closed_book"})
        if closed[0].keys() != closed[1].keys():
            raise ValueError("Closed-book item sets differ")
        preserved = {}
        for filename in ("raw_responses.jsonl", "requests.jsonl", "model.json", "requirements-frozen.txt"):
            path = source_paths[0] / filename
            value = file_hash(path)
            if normalization["preserved_files_sha256"][path.relative_to(ROOT).as_posix()] != value:
                raise ValueError(f"Baseline preserved-file hash mismatch: {path}")
            preserved[filename] = value
        audit.append({"model": model, "closed_book_items": len(closed[0]),
                      "closed_book_answer_disagreements": sum(closed[0][k] != closed[1][k] for k in closed[0]),
                      "baseline_preserved_hashes": preserved,
                      "context_generation_seconds": summaries[f"{model}/context-faithful/seed-2027"]["diagnostics"]["generation_seconds"]})

    groups = {
        "F-only | original B": ("reference_B", ("f_only",)),
        "S-only | original B": ("reference_B", ("s_only",)),
        "G+F mean | original A": ("reference_A", ("g_then_f", "f_then_g")),
        "G+S mean | original A": ("reference_A", ("g_then_s", "s_then_g")),
        "G-only | all items": ("all", ("g_only",)),
        "G+F mean | common A": ("common_A", ("g_then_f", "f_then_g")),
        "G+S mean | common A": ("common_A", ("g_then_s", "s_then_g")),
    }
    panel = {}
    for label, (cohort, conditions) in groups.items():
        per_model = [outcomes(p, cohort, conditions) for p in pairs.values()]
        panel[label] = {outcome: {field: mean(p[outcome][field] for p in per_model)
                                  for field in ("reference_rate", "treatment_rate")}
                        for outcome in OUTCOMES}
    by_model = {}
    for model in MODELS:
        pair = pairs[model]
        by_model[model] = {"cohorts": pair["cohort_sizes"],
                          **{label: metric_pair(pair, cohort, conds) for label, cohort, conds in (
                              ("LR_F", "reference_B", ("f_only",)),
                              ("LR_S", "reference_B", ("s_only",)),
                              ("DR_F", "reference_A", ("g_then_f", "f_then_g")),
                              ("DR_S", "reference_A", ("g_then_s", "s_then_g")))}}
    order = {}
    for condition in ("g_then_f", "f_then_g", "g_then_s", "s_then_g"):
        values = [metric_pair(pair, "reference_A", (condition,)) for pair in pairs.values()]
        order[condition] = {field: mean(v[field] for v in values) for field in values[0]}
    status = Counter()
    for value in summaries.values():
        status.update(value["diagnostics"]["parse_status"])
    result = {"scope": "six context-faithful full main runs only",
              "aggregation": "equal model weights; descriptive, no confidence intervals or significance claims",
              "responses": sum(status.values()), "parse_status": dict(status),
              "panel_outcomes_pct": panel, "by_model": by_model, "order_on_reference_A": order,
              "integrity": audit,
              "analysis_sources_sha256": {p.relative_to(ROOT).as_posix(): file_hash(p) for p in (
                  Path(__file__), Path(__file__).with_name("analyze.py"),
                  Path(__file__).with_name("verify.py"), Path(__file__).with_name("dataset_identity.py"),
                  ROOT / "experiments/analysis/metrics.py", ROOT / "experiments/main_eval/protocol.py")}}
    save_json(directory / "context_summary.json", result)

    print(json.dumps({"responses": result["responses"], "status": dict(status),
                      "closed_book_disagreements": sum(r["closed_book_answer_disagreements"] for r in audit)}))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "experiments/outputs/analysis/context-faithful-r3")
    parser.add_argument("--run-root", type=Path, default=ROOT / "experiments/outputs/supplementary")
    parser.add_argument("--baseline-root", type=Path, default=ROOT / "experiments/outputs/main_eval")
    args = parser.parse_args()
    build_report(args.directory, args.run_root, args.baseline_root)
