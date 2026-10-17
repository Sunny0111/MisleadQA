# Scope judgment and answering behavior consistency diagnostic

This experiment is independent of `experiments/main_eval/` and `experiments/main_eval/supplementary/`. It reuses
the original six checkpoints, chat templates, greedy settings, BF16/AWQ profiles,
and inference environments. Original prompts and results retain their bytes;
inference imports and paths follow the reorganized repository. The input is the
frozen release v1.0.0 main split of 959 items; the
25-item supplement and superseded dataset versions are not accepted.

## Fixed protocol

- Seed 2027; sample 40 items without replacement from each source-scope group,
  using fixed group order and items sorted by ID. `sample.json` freezes 200 IDs,
  their groups, source clusters, and the dataset hash. Sampling never reads model
  answers. Groups are FRED time/geography, SituatedQA time/geography, and
  AmbigDocs entity.
- Present G, F, and S in separate independent user turns. Preserve the original
  question, A-E options, and evidence text. Only the content rendered from
  `experiments/configs/prompts/scope_v1.txt` reaches the model; evidence types, answer keys, and expected scope
  labels are hidden. G/F are `Applicable`; S is `Not applicable`.
- Encode the three literal English labels with each pinned tokenizer and apply
  a shared token-prefix constraint during greedy generation, followed by EOS.
  Outputs are exactly `Applicable`, `Not applicable`, or `Cannot determine`.
  Labels can span multiple tokens, so the completion cap is the longest label
  token sequence plus one EOS token. Save label token IDs, the resolved budget,
  and effective generation settings in `model.json`. This adapts the original
  single-letter constraint while retaining the other greedy/model settings.
- Full inference adds 600 calls per model, 3,600 across six models. Join existing
  closed-book and S-only answers during analysis; do not generate them again.
  Smoke uses the first frozen sampled item in each group: 15 calls per model,
  another 90 calls across six models.
- Manifests freeze requests, sampling, code, software, and model identity. Flush
  and fsync every response. Resume only unfinished requests; never retry a saved
  first response. Identity/environment changes reject continuation under the
  same run ID. `.lock` prevents concurrent writers; completed runs seal file hashes.

Protocol v2 uses an English prompt and English output labels. It preserves the
same 200 sampled IDs. Use the v2 run names below; earlier protocol outputs cannot
be resumed or merged into v2 runs.

## Local validation and preparation

Preparation, analysis, and tests require only Python 3.12 and the standard
library. GPU libraries are imported only for actual inference.

```powershell
py -3.12 -B -m unittest tests.test_scope_diagnostic -v
py -3.12 -B experiments/scope_diagnostic/run.py --model qwen3-8b --run-id main-scope-preview-v2 --dry-run
py -3.12 -B experiments/scope_diagnostic/package.py
```

The package and SHA-256 file are written to
`dist/misleadqa-v1.0.0-scope-v2.tar.gz` and `.sha256`. They contain frozen data,
baseline code, diagnostic code, the sample, prompt, tests, and documentation;
weights, caches, and results are excluded. Existing packages cannot be
overwritten. After further changes, choose a new filename with `--output`.

## Sequential A40 inference

Upload the package and checksum to the original server. Extract into a new
directory, enter tmux, and use the two existing inference environments. Set
`TF_ENV` and `VLLM_ENV` to their Conda names if they differ from these defaults:

```bash
cd ~
sha256sum -c misleadqa-v1.0.0-scope-v2.tar.gz.sha256
mkdir misleadqa-v1.0.0-scope-v2
tar -xzf misleadqa-v1.0.0-scope-v2.tar.gz -C misleadqa-v1.0.0-scope-v2
cd misleadqa-v1.0.0-scope-v2
tmux new -s misleadqa-scope

export TF_PYTHON="$(conda run -n "${TF_ENV:-misleadqa-v100-tf}" python -c 'import sys; print(sys.executable)')"
export VLLM_PYTHON="$(conda run -n "${VLLM_ENV:-misleadqa-v100-vllm}" python -c 'import sys; print(sys.executable)')"
export CUDA_VISIBLE_DEVICES=0
"$TF_PYTHON" -B experiments/scope_diagnostic/package.py --verify .
set -o pipefail
bash experiments/scope_diagnostic/run.sh smoke 2>&1 | tee -a scope-smoke.log
# Start full inference after SCOPE smoke COMPLETE:
bash experiments/scope_diagnostic/run.sh full 2>&1 | tee -a scope-full.log
```

Full run ID: `main-scope-full-v2`; smoke run ID: `main-scope-smoke-v2`.
Repeat the same command to resume. `RUN_ID` overrides the name; pass the same
`--run-id` to analysis. To run one model, use
`experiments/scope_diagnostic/run.py --model <alias>`. Offline verification uses
`experiments/scope_diagnostic/run.py --verify experiments/outputs/scope_diagnostic/<alias>/<run-id>`.

## Join original answers and analyze

Original results must have the layout
`<baseline-root>/<model>/main-v100-full-v1/`, including manifest, model.json,
requests.jsonl, and raw_responses.jsonl. Analysis verifies all six complete
original and diagnostic runs and checks dataset, checkpoint, tokenizer, and
template identity. Analyze on the server, or download `experiments/outputs/scope_diagnostic/`
for local analysis with the original results also available.

```bash
"$TF_PYTHON" -B experiments/scope_diagnostic/analyze.py \
  --baseline-root experiments/outputs/main_eval
```

Default output: `experiments/outputs/analysis/scope_diagnostic/main-scope-full-v2/`. Use `--output`
for a new directory; existing analyses are not overwritten.
`--baseline-run-id` selects a different original run name.

- `statistics.json`: per-model statistics and fixed six-model equal-weight
  means. Rates retain numerators/denominators; balanced accuracy retains both
  recall components, and means retain every model's components. Also includes
  equal-model confusion proportions within each true class and joint cross-table
  proportions within each model's B_m.
- `report.md`, `metrics.csv`: class recalls, balanced accuracy, G/F/S correctness
  and inability-to-determine rates, two conditional lure probabilities on the
  originally correct closed-book cohort B_m, and the joint probability of
  `Not applicable` and lure selection.
- `tables.csv`: the 2-by-3 confusion matrix and 3-by-5 cross table
  (gold/lure/abstain/other/invalid), including cell numerators, row denominators,
  and table denominators. The full cross table retains `Cannot determine`.
- `groups.json`: the same descriptive statistics for the five source-scope groups.
- `bootstrap.json`: exploratory 95% intervals for all scalar rates above,
  valid replicate counts, and zero-denominator replicate counts. Defaults:
  10,000 replicates, seed 2027. Resample the represented `(source, cluster_id)`
  clusters with replacement, drawing the same number of clusters. Use the same
  cluster multiplicities for G/F/S and all six models, retaining whole clusters
  and item weights within clusters. Do not stratify bootstrap draws by the five
  source-scope groups. Compute each model's rate before the equal-weight mean;
  any undefined model invalidates that mean replicate. Zero-denominator point
  estimates are JSON null, blank CSV cells, and `Not estimable` in the report;
  models are never silently dropped.
- `joined_items.jsonl`, `provenance.json`: item-level audit joins and original
  response/model/code/dataset fingerprints.

Balanced accuracy = 0.5 * (correct G/F judgments / 400 + correct S judgments / 200).
`Cannot determine` counts as incorrect classification and stays in the denominator.
The three behavior-metric denominators are, respectively, the B_m items judged
`Not applicable`, the B_m items judged `Applicable`, and all B_m items.
All stored rates range from 0 to 1; the report displays percentages.

The balanced diagnostic subset does not represent the natural main-set
distribution. The experiment distinguishes difficulty judging scope from
correct independent judgments accompanied by lure selection during answering.
It cannot establish that the model knowingly used inapplicable evidence within
the same answer. Differences between the two conditional probabilities have no
causal interpretation. Intervals are exploratory, without multiplicity correction.
