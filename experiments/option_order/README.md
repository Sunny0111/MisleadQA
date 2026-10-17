# Option-order robustness

Test whether scope-inapplicable evidence still elicits the lure on questions
answered correctly across three option orders. This exploratory check preserves
the main experiment and its original B_m cohorts.

The F-only extension below adds a matched comparison without changing the
completed closed-book/S-only experiment.

The [Slurm launcher](submit-slurm.sh)
regenerates all three orders on the new host (7,200 calls) and audits agreement
with historical answers separately. The reuse workflow below requires the
original environment; do not bypass its metadata check on a different host.

The Slurm launcher defaults to `.venv-tf/bin/python` and
`.venv-vllm/bin/python`, relative to the submitted project directory. Set
`TF_PYTHON` and `VLLM_PYTHON` before submission to select other installations.
The shell integration tests discover Git Bash on Windows through `git` on
`PATH`; `BASH_EXECUTABLE` can select a different Bash installation.

## Frozen design

- Reuse the existing scope diagnostic's 200 items, 40 per source-scope group,
  selected independently of model responses. Do not draw another sample.
- Keep the original order and left-rotate the A-D option texts by one and two
  places: `A B C D E`, `B C D A E`, `C D A B E`. Letters on the displayed options
  remain A-E. E never moves. Gold and lure each occupy three distinct positions.
- Evaluate closed-book and S-only in independent requests under each order.
  Question, evidence and original QA prompt remain unchanged. Score by remapped
  answer roles, not by comparing output letters across orders.
- Reuse the original 400 answers per model from verified complete main runs.
  Only the two rotations require inference: 800 calls/model, 4,800 for six models.
  An optional five-item smoke run adds 20 calls/model, 120 in total.
- Preserve the original pinned checkpoints, backends, tokenizer, chat template,
  greedy choice-token-plus-EOS constraint, seed, software and GPU metadata.
  Qwen thinking remains disabled. Metadata differences stop inference; do not
  upgrade the existing environments for this experiment.
- Define the option-stable cohort separately for each model: the intersection of
  its correct closed-book answers across all three orders. This demonstrates
  stability over these orders, not stable knowledge or absence of guessing.

The default full run ID is `main-option-order-full-v1`; smoke and preview use
separate IDs. Runs retain their baseline subset, prompts, remapped keys, raw
responses, token IDs, model/runtime metadata and provenance hashes. The shared
runner flushes each first response, resumes only missing requests, and repairs
only a torn final append. A lock excludes concurrent writers; completion seals
detect later changes. Never remove a lock before confirming its process stopped.

## Prepare locally

Use Python 3.12. Preparation, verification, analysis and tests use the standard
library; GPU packages are imported only for actual inference.

```powershell
py -3.12 -B -m unittest tests.test_option_order -v
py -3.12 -B experiments/main_eval/refresh_manifest.py
py -3.12 -B experiments/option_order/prepare.py --package dist/misleadqa-v1.0.0-option-order-v1.tar.gz
```

Preparation verifies the full saved runs at
`experiments/outputs/main_eval/<model>/main-v100-full-v1/`, then exports only the
frozen sample's original closed-book/S-only records. The references include
original manifest identity, model metadata and source-file hashes. They are
written to `experiments/outputs/option_order_references/` and included in the
portable bundle. No model is loaded and no original response is regenerated.
Use `--baseline-root` or `--baseline-run-id` if the original runs are elsewhere.

Identical references can be exported again; changed references and existing
archives cannot be overwritten. After source changes, refresh the application
manifest and choose a new archive filename. Source bundles contain no weights,
caches or credentials. This bundle intentionally includes 2,400 existing sample
answers, allowing the remote experiment to run without locating old server runs.

```powershell
scp .\dist\misleadqa-v1.0.0-option-order-v1.tar.gz .\dist\misleadqa-v1.0.0-option-order-v1.tar.gz.sha256 USER@SERVER:~/
```

Replace `USER@SERVER` with your SSH destination.

## Run remotely on the original A40

Extract once into a new directory:

```bash
cd ~
sha256sum -c misleadqa-v1.0.0-option-order-v1.tar.gz.sha256
mkdir misleadqa-v1.0.0-option-order-v1
tar -xzf misleadqa-v1.0.0-option-order-v1.tar.gz -C misleadqa-v1.0.0-option-order-v1
cd misleadqa-v1.0.0-option-order-v1
tmux new -s misleadqa-option-order
```

Inside tmux, from the extracted project directory, run the following. Set
`TF_ENV` and `VLLM_ENV` to your original environments' names if they differ
from the defaults:

```bash
export TF_PYTHON="$(conda run -n "${TF_ENV:-misleadqa-v100-tf}" python -c 'import sys; print(sys.executable)')"
export VLLM_PYTHON="$(conda run -n "${VLLM_ENV:-misleadqa-v100-vllm}" python -c 'import sys; print(sys.executable)')"
export CUDA_VISIBLE_DEVICES=0
"$TF_PYTHON" -B experiments/option_order/prepare.py --verify-bundle .
set -o pipefail
(bash experiments/option_order/run.sh smoke && bash experiments/option_order/run.sh full) 2>&1 | tee -a option-order.log
```

Detach with Ctrl-b, then d. Reattach with `tmux attach -t misleadqa-option-order`.
For a progress snapshot, read `option-order.log` or the per-run `progress.json`.
The final marker is `OPTION ORDER full COMPLETE`. Repeating the same run command
resumes missing requests; completed runs are verified without loading weights.
Keep source, reference files and environments unchanged during resumption.

Individual entry points:

```bash
"$TF_PYTHON" -B experiments/option_order/run.py --model qwen3-8b --dry-run
"$TF_PYTHON" -B experiments/option_order/run.py --model qwen3-8b --smoke
"$TF_PYTHON" -B experiments/option_order/run.py --model qwen3-8b
"$TF_PYTHON" -B experiments/option_order/run.py --verify experiments/outputs/option_order/qwen3-8b/main-option-order-full-v1
```

Use the vLLM environment for `qwen3-32b-awq`. `RUN_ID` overrides the shell
script's run name; pass the same full name to analysis if changed.

## Analyze and retrieve results

```bash
"$TF_PYTHON" -B experiments/option_order/analyze.py
```

Alternatively, download the runs and analyze locally:

```powershell
scp -r USER@SERVER:~/misleadqa-v1.0.0-option-order-v1/experiments/outputs/option_order ./experiments/outputs/
py -3.12 -B experiments/option_order/analyze.py
```

Analysis requires all six full runs; smoke or incomplete runs are rejected.
The downloaded runs contain their own original-answer references. Default output:
`experiments/outputs/analysis/option_order/main-option-order-full-v1/`.
Use `--output` for a new analysis directory; existing results are not overwritten.

The main report separates selection from option-order sensitivity:

1. Original-order closed-book-correct count within these 200 items.
2. Three-order stable count, fraction of 200, and retention of original correct items.
3. Original-order S lure rate on the original-correct cohort.
4. Original-order S lure rate on the stable cohort.
5. Per-order and three-order mean S lure rates on that same stable cohort.

`statistics.json` also retains complete S outcome distributions, per-order
closed-book accuracy and stable source/dependency cluster counts. `groups.json`
reports the same descriptive statistics within each source-scope group.
`joined_items.jsonl` retains each selected label, its original-label mapping,
scored outcome and cohort membership. `metrics.csv` and `report.md` summarize
counts, rates and intervals; `provenance.json` records verified input hashes.

Both source-cluster and dependency-group bootstraps use 10,000 paired resamples,
seed 2027, and exploratory 95% percentile intervals. Draw the represented
clusters with replacement without stratification, retain whole clusters and all
three orders, and use identical multiplicities across models. Compute each
model's rate before taking the fixed six-model equal-weight mean. Never treat
the three orders as independent observations. `bootstrap.json` records cluster
membership and valid/invalid denominator draws. Empty denominators produce null,
blank CSV cells and `Not estimable`; undefined models are never dropped from the
panel. Very small stable cohorts or few independent clusters limit interpretation.

The 200-item balanced subset differs from the full main-set distribution.
Compare rates within this sample, not directly against the main 29.3% result.
E is fixed, so this experiment does not evaluate abstention-position sensitivity.

## Add F-only to a completed three-order run

Use the same frozen 200 items and original/rotate1/rotate2 arrangements. The
extension generates only F-only responses: 600 calls per model, or 3,600 across
six models. A separate five-item smoke run uses 15 calls per model. The supplied
baseline must be a complete fresh three-order closed-book/S-only run, not a
smoke run or a run that reused historical original-order responses.

Preview one model without loading weights:

```bash
"$TF_PYTHON" -B experiments/option_order/run.py \
  --f-only-from experiments/outputs/option_order/qwen3-8b/main-option-order-slurm-full-fresh-v1 \
  --dry-run
```

For inference, replace `--dry-run` with `--smoke`, then omit both flags for the
full run. Use the original vLLM environment for Qwen3-32B-AWQ. The model is
inferred from the baseline; an explicit `--model` must match. Do not pass
`--rerun-original`: F-only always generates all three orders. Defaults use new
run IDs ending in `smoke-f-only-v1` or `full-f-only-v1`. The existing Slurm
workflow also accepts `f-only`.

The runner snapshots the verified baseline in `paired_baseline.json`, records
its file hashes, and requires matching checkpoint, tokenizer, prompt, decoding,
software and hardware metadata before generating F responses. There is no
override for a mismatch. Regenerate a complete closed-book/S baseline in the
chosen environment if necessary. Existing baseline files are never edited.
Interrupted F runs resume without replacing completed answers.

Analyze the six extensions with their exact baselines (these IDs are the
Slurm defaults):

```bash
"$TF_PYTHON" -B experiments/option_order/analyze.py \
  --run-id main-option-order-slurm-full-fresh-v1 \
  --f-run-id main-option-order-slurm-full-f-only-v1
```

`--run-root` locates the baseline runs; `--f-run-root` optionally locates F runs
elsewhere. The default new analysis directory uses the F run ID, preserving the
existing S-only analysis. Baseline hashes must match for every model. Missing,
duplicate, partial or mismatched F responses stop analysis.

The report adds S lure, F lure and paired F-minus-S differences for:

1. Original-order-correct items, evaluated in the original order.
2. Three-order-correct items, evaluated in the original order.
3. The same three-order-correct items, evaluated separately in each order and
   averaged across the three orders.

F responses never affect eligibility. `statistics.json` and `groups.json` retain
all five outcome categories for S and F. Numerators and denominators accompany
each model's rates; contrasts store the difference in lure counts over the same
denominator. Panel estimates are unweighted means over models, never pooled
counts. CSV rates and differences are in proportion units; the report displays
percentages and percentage points. Both bootstrap schemes use shared draws
across S, F, orders and models, with exploratory 95% intervals. This extension
does not evaluate G-only or mixed-evidence conditions.
