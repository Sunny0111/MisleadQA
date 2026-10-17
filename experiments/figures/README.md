# Result figures

The scripts below read saved numerical analyses and export vector PDF figures.
Numerical exports are documented in `../analysis/README.md`.
All current figure scripts use Matplotlib and the shared `style.py` settings.

## Figure 2: S-only outcomes and the effect of correct evidence

`plot_evidence_effects.py` uses the supplied serif style and generates both panels
at a width of 3.03 inches. Run from the repository root:

```powershell
py -3.12 -m pip install -r experiments/figures/requirements.txt
py -3.12 -B experiments/figures/plot_evidence_effects.py
```

The default input is
`experiments/outputs/01a1040f-0ef4-71f2-9563-55b4e111a9a1/all_results.json`.
Use `--results PATH` or `--output-dir DIRECTORY` to override the defaults.
Default paths resolve from the script's repository location, so they do not
depend on the working directory.

- `fig2a_s_only_outcomes_by_group.pdf`: S-only Lure, Gold, Abstain and
  Other rates on each model's group-specific B_m. Read
  `results.experiment_evidence.strata[group].models[model].metrics`, average the
  six saved rates equally, then sort groups by descending Lure. Counts are not
  pooled across models. The script verifies B_m denominators and checks that
  the four outcomes sum to 100%; it does not renormalize incomplete data.
- `fig2b_lure_rates_with_gold.pdf`: S and F lure rates on A_m under
  Alone, G first and S/F first. Read
  `results.review.cohort_rates.all_selected.rates`, whose values are already
  equal-model means, and multiply by 100. The y-axis spans 0-100%; markers use
  matching edge colors and are not clipped at the axis. F labels sit 5 percentage
  points above each point. The S Alone label sits 5 percentage points below its
  point; the two near-zero S labels sit slightly above and to the right.

Outputs go to `experiments/outputs/figures/`. PDF files retain vector artwork;
Colors match the supplied code: vermilion for Lure/S,
blue for Gold/F, yellow for Abstain and green for Other. Console output reports
the plotted percentages for numerical review. B_m contains closed-book-correct
items; A_m additionally requires a correct answer with G alone. The panels use
different cohorts and should not be subtracted to estimate resistance.

## Additional scope-recognition and context-faithful figures

Both scripts read local saved analysis results, share `style.py` with Figure 2,
and use the existing Matplotlib dependency. Run from the repository root:

```powershell
py -3.12 -B experiments/figures/plot_scope_recognition.py
py -3.12 -B experiments/figures/plot_context_faithful.py
```

Both accept `--results PATH` and `--output-dir DIRECTORY`. Default paths resolve
from the repository location, independent of the working directory. Output goes
to `experiments/outputs/figures/`: vector PDF with embedded TrueType fonts, each 3.03 inches wide. Panel (a) is 1.45 inches high and panel (b)
is 1.70 inches high. Panel labels and headings are not embedded in the images.

- `fig3a_s_scope_recognition_by_model_group.pdf`: six models by five
  groups, with a fixed 0-100% blue color scale beside the matrix and one-decimal
  cell labels. The vertical colorbar marks 0, 25, 50, 75 and 100; source headers
  and scope labels occupy separate rows.
  Read `experiments/outputs/analysis/scope_diagnostic/main-scope-full-v2/groups.json`
  at `[model][group].metrics.S.correct`. Each denominator is all 40 items in the
  group; this is not a B_m-filtered measure. `Cannot determine` stays in the
  denominator and counts as an incorrect classification. Group headers distinguish
  AmbigDocs entity, FRED time/location and SituatedQA time/location. The script
  checks the model panel, groups, denominators and saved rates against counts.
- `fig3b_original_vs_context_faithful_outcomes.pdf`: paired Original/CF
  100% stacked bars for S-only, G-only and G+S, separated by thin rules. Read
  `experiments/outputs/analysis/context-faithful-r3/context_summary.json`
  at `panel_outcomes_pct`, using `reference_rate` and `treatment_rate` for each
  outcome. These values are already percentages and equal-model means: do not
  multiply by 100 or average them again. S-only uses `S-only | original B`,
  G-only uses `G-only | all items` (959 items), and G+S uses
  `G+S mean | original A` (an equal average over the two evidence orders).
  Within each pair, both prompts use the same original items. Compare prompts
  within each pair; the three conditions use different cohorts. The script
  rejects nonzero invalid rates and distributions that do not sum to 100%,
  without renormalization. Segments of at least 6% have one-decimal labels;
  console output reports every segment to two decimals.

## Figure 3: Conditional lure selection after correct scope judgments

`plot_scope_conditional_lure.py` reads the saved scope diagnostic and draws six
model rates plus their unweighted mean, with exploratory 95% source-cluster
bootstrap intervals. It uses the shared typography and model labels with an
orange (`#EB6834`) and gray palette.

```powershell
py -3.12 -B experiments/figures/plot_scope_conditional_lure.py
```

The default input directory is
`experiments/outputs/analysis/scope_diagnostic/main-scope-full-v2/`.
The script reads `lure_given_inapplicable` from `statistics.json` and
`bootstrap.json`, checks conditional counts and interval consistency, and converts
saved proportions to percentages. Interval endpoints are plotted at full precision;
only text labels are rounded. The dashed reference is the equal-model mean S-only
lure rate on all diagnostic B_m items, computed from `cross_table_on_B_m` (including
all scope judgments). No inference or bootstrap is rerun.
Use `--results-dir DIRECTORY` or `--output-dir DIRECTORY` to override the defaults.

Output is `experiments/outputs/figures/fig3_scope_conditional_lure.pdf`,
a vector PDF at 3.03 by 1.95 inches.
Orange points represent models; the black diamond is their unweighted mean.
Counts beside each interval give lure selections divided by eligible items.
The mean has no pooled count. No title or panel letter is embedded in the figure.

## Paired conditional answering outcomes

`plot_paired_outcomes.py` plots F-mixed outcome distributions conditional on
S-mixed correctness separately for G-first and misleading-first presentation.
It reuses `style.py` and the existing Matplotlib dependency:

```powershell
py -3.12 -m pip install -r experiments/figures/requirements.txt
py -3.12 -B experiments/figures/plot_paired_outcomes.py
```

The input is `experiments/outputs/analysis/paired-configurations-main-v1/statistics.json`.
Read `source_cluster.panel["<order>.F_given_S_gold.<outcome>"].estimate`.
These estimates are percentages and unweighted means over models, not pooled
response counts. The script verifies them against the saved per-model numerators
and denominators and checks that all five outcome counts exhaust each denominator.
It does not renormalize distributions or rerun model inference or the bootstrap.

Output is `experiments/outputs/figures/fig2_paired_conditional_outcomes.pdf`
(vector PDF). `--results`, `--output-dir`, `--width`,
and `--height` can override the defaults; the default size is 3.03 by 1.85 inches.
Gold and lure labels remain visible, with a callout for the narrow G-first lure
segment. Abstain and Other segments retain their full precision; all rates are
printed to the console. Invalid outcomes are retained in the denominator and
drawn if nonzero. No confidence intervals are drawn on the stacked segments.

