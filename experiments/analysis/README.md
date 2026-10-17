# Analysis

`metrics.py` defines outcome metrics and paired bootstrap intervals. `analyze.py`
verifies completed main-evaluation runs, attaches source clusters, and exports
conditional rates, model/panel contrasts and provenance. Dependency-group
sensitivity uses `--cluster-field analysis_group_id`.

```powershell
py -3.12 -B experiments/analysis/analyze.py --help
```

The Linux launcher is `experiments/main_eval/scripts/analyze_six.sh`.
Generated analysis directories belong in `experiments/outputs/analysis/`, with
separate main and condition-supplement results. Existing directories are never
overwritten by the core analyzer.

`build_local_review.py` reproduces the descriptive review from the six saved
main and supplement runs. `summarize_evidence.py` derives descriptive
sensitivities and outcome summaries from those saved results.
These helpers require the corresponding full local outputs.

Numerical exports can be regenerated in dependency order:

```powershell
python -B experiments/analysis/build_local_review.py
python -B experiments/analysis/summarize_evidence.py
python -B experiments/analysis/response_diagnostics.py
python -B experiments/analysis/consolidate_results.py
```

These commands read existing full analyses and preserve raw inference outputs.
They export numerical results as JSON and CSV.
`consolidate_results.py` writes `all_results.json` under
`experiments/outputs/01a1040f-0ef4-71f2-9563-55b4e111a9a1/`, retaining the input
schema consumed by `plot_evidence_effects.py`. The summary covers the main,
condition-supplement, scope and CF analyses; option-order results have their own
analysis directory and are not included in this older consolidated schema.

Use the original B/A cohorts for the primary context-faithful comparison. The
scope subset overlaps the main split, and the condition supplement remains a
separate descriptive result. Missing intervals and zero-denominator rates are
not treated as zero.

All model measurements use v1.0.0. Version-label normalization updates release
references and hashes while preserving original requests, raw responses and
execution code fingerprints. Verify saved runs against the frozen data and
requests, and use fresh run IDs for new inference after code changes.

## Paired configuration reanalysis

`paired_configurations.py` adds within-model, within-item, within-order S/F
success tables and conditional F outcomes given S-mixed gold. It preserves the
original main-analysis A_m cohorts and fixed equal-model panel, re-scores the
saved raw responses, and checks the original point estimates and bootstrap draws.
It requires NumPy (validated with 2.3.5); no model inference is run.

```powershell
python -B experiments/analysis/paired_configurations.py
```

The default output is `experiments/outputs/analysis/paired-configurations-main-v1/`.
Choose a fresh `--output` directory for a new revision. It exports a report,
JSON/CSV statistics, paired item outcomes and provenance. Source-cluster and dependency-group intervals are exploratory,
unadjusted 95% intervals. The two evidence orders are paired separately; no
order-averaged binary outcome or pooled-model conditional rate is used.

## Metrics


- B_m: items answered correctly without evidence.
- G_correct: items answered correctly with G alone.
- A_m: the intersection of B_m and G_correct.
- LR_F / LR_S: single-evidence lure rates on B_m.
- DR_F / DR_S: lure rates pooled over the two mixed orders on A_m; denominator 2|A_m|.
- Delta_LR / Delta_DR: F minus S, in percentage points.
- Resistance_X: X-only lure rate on A_m minus mixed-order mean on the same A_m.
  Positive values mean adding G reduces lure selection.
- Order_X: (G then X) minus (X then G) lure rate on A_m, in percentage points.
  Positive values mean more lure selections when misleading evidence is last.

The order-specific rates, correct answers, lure answers, abstentions, other
choices and invalid responses are all retained. Empty denominators produce null.
Do not subtract LR on B_m from DR on A_m to estimate resistance.

Primary panel contrasts are equal-weight model means over the same fixed
eligible panel. Eligibility requires at least 20 distinct (source, cluster_id)
clusters represented in the original-sample A_m. The paired pooled-cluster
bootstrap uses the same cluster draw across all conditions and models, retaining
item weighting within clusters. The two primary panel comparisons use two-sided
97.5% percentile intervals (Bonferroni family alpha 0.05), 10,000 replicates and
seed 2027. Secondary and individual-model intervals are exploratory 95%.
Invalid-denominator draws and eligible panel membership are explicitly reported.
Excluded models keep descriptive estimates but have null bootstrap intervals.
Dependency-group resampling is available as a sensitivity analysis.
