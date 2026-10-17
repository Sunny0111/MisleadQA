# Running release 1.0.0 on a single A40

This directory contains a runnable inference and analysis application for the
959-item main split and 25-item condition supplement. The original handoff
manifest is preserved as `handoff_manifest.json`; `manifest.json` describes the
adapted application. Use fresh run IDs; historical v0.37 runs are incompatible.

All model measurements use dataset v1.0.0: 984 frozen items in total, with 959
main items and 25 condition-supplement items run separately. The normalized
release changes only the dataset version label. Saved requests and raw responses
remain unchanged. Their opaque request-ID namespace is retained for verification;
it does not identify a different dataset. Original execution references are
recorded in each saved manifest's `version_normalization` note.

## Fixed model panel and prompt

| Alias | Checkpoint | Backend / weights |
| --- | --- | --- |
| qwen3-8b | Qwen/Qwen3-8B | Transformers / BF16 |
| qwen3-14b | Qwen/Qwen3-14B | Transformers / BF16 |
| llama31-8b-instruct | meta-llama/Llama-3.1-8B-Instruct | Transformers / BF16 |
| gemma3-12b-it | google/gemma-3-12b-it | Transformers / BF16, text only |
| olmo2-13b-instruct | allenai/OLMo-2-1124-13B-Instruct | Transformers / BF16 |
| qwen3-32b-awq | Qwen/Qwen3-32B-AWQ | vLLM V0 / official AWQ 4-bit |

Revisions and official chat-template kwargs are pinned in `../configs/models.json`.
The prompt is unchanged; SHA-256:
`b6ecb8a70c7d19ef47eb44d2e04afe1c8be76bdb5b189e31b11937618bc766c2`.

The prompt uses Python string.Template placeholders `${question}`,
`${options}`, `${evidence_block}`. Each option is formatted as `A. text`.
Evidence follows the options after two newlines; two passages are separated by
two newlines. No G/F/S headings, scoring keys or review metadata enter messages.

Each condition is an independent user turn. The official checkpoint chat
template supplies model-specific delimiters. Qwen uses enable_thinking=False;
Llama uses date_string="01 Oct 2026". Gemma receives text only.
Canonical A/B/C/D/E token IDs are resolved independently for each tokenizer.
The first generation step is masked to those five tokens; the second is masked
to model-specific EOS IDs. A two-token budget alone is not used as a constraint.

`../configs/inference_policy.json` documents the policy implemented by the code.
It is a reference, not an arbitrary runtime override file. CLI seed and maximum
input length overrides are recorded in the run identity. The default seed is
2027, input limit 4096, one request per call; overlong inputs fail without
truncation. Transformers uses strict deterministic settings and SDPA math.
The AWQ backend uses the supplied single-A40 vLLM 0.9.2 V0/FLASH_ATTN profile.
Cross-backend bitwise equality is not claimed.

## Create and upload the portable bundle

On Windows in the repository root:

```powershell
py -3.12 -B experiments/main_eval/refresh_manifest.py
py -3.12 -B -m unittest discover -s tests -v
py -3.12 -B experiments/main_eval/package.py
scp .\dist\misleadqa-v1.0.0-a40.tar.gz .\dist\misleadqa-v1.0.0-a40.tar.gz.sha256 USER@SERVER:~/
```

Replace USER@SERVER with your SSH destination. An existing package is never
overwritten; use `--output dist/misleadqa-v1.0.0-a40-r2.tar.gz` after changes.
The package has been byte-verified against the workspace files.
After editing application source, refresh the manifest before packaging (and
before running bundle tests); use a new run ID for any subsequent inference.

On the Linux x86_64 server, from the upload directory:

```bash
sha256sum -c misleadqa-v1.0.0-a40.tar.gz.sha256
mkdir misleadqa-v1.0.0-a40
tar -xzf misleadqa-v1.0.0-a40.tar.gz -C misleadqa-v1.0.0-a40
cd misleadqa-v1.0.0-a40
(cd construction/data && sha256sum -c hashes.txt)
nvidia-smi
free -h
df -h .
```

Use a new extraction directory. The profile expects one visible BF16-capable
A40 and at least 16 GiB available host RAM for AWQ. Model downloads require
additional disk space for all six pinned checkpoints.

## Install two isolated environments

Conda and network access must already be available. Keep Transformers and vLLM
in different environments because their pinned Transformers/Hub versions differ.
The CUDA 11.8 runtime is provided by the wheels; the system CUDA toolkit is not
replaced. The historical profile targeted an NVIDIA 535 driver.
[vLLM 0.9.2 installation reference](https://docs.vllm.ai/en/v0.9.2/getting_started/installation/gpu.html).

```bash
conda create -n misleadqa-v100-tf python=3.12 pip -y
conda run --no-capture-output -n misleadqa-v100-tf python -m pip install -r experiments/configs/requirements/requirements-transformers-cu118.txt
conda run -n misleadqa-v100-tf python -m pip check

conda create -n misleadqa-v100-vllm python=3.12 pip -y
conda run --no-capture-output -n misleadqa-v100-vllm python -m pip install -r experiments/configs/requirements/requirements-vllm-cu118.txt
conda run -n misleadqa-v100-vllm python -m pip check
```

Authenticate on the server with a Hugging Face account that has access to
[Llama](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) and
[Gemma](https://huggingface.co/google/gemma-3-12b-it). If already authenticated,
the shared default Hub cache can be reused by both environments.

```bash
conda run --no-capture-output -n misleadqa-v100-tf python -c "from huggingface_hub import login; login()"
export CUDA_VISIBLE_DEVICES=0
conda run --no-capture-output -n misleadqa-v100-tf python -B experiments/main_eval/preflight.py --backend transformers --check-model-access
conda run --no-capture-output -n misleadqa-v100-vllm python -B experiments/main_eval/preflight.py --backend vllm --check-model-access
conda run -n misleadqa-v100-tf python -B -m unittest discover -s tests -v
```

Preflight validates package versions, GPU access and a real BF16 matrix
multiplication. The access check downloads only config.json at the pinned
revisions. It does not test loaded model kernels; real smoke inference does that.

## Run the six models sequentially

### Reuse existing inference environments

Existing environments can be reused after both preflight checks pass. Select
the Python executables from your own Transformers and vLLM environments. The
examples below discover them through the Conda environment names configured
above; override `TF_ENV` and `VLLM_ENV` if your environments have other names.

Upload the updated bundle from **local Windows PowerShell**, in the repository
root. Replace `USER@SERVER` with your SSH destination:

```powershell
scp .\dist\misleadqa-v1.0.0-a40-r2.tar.gz .\dist\misleadqa-v1.0.0-a40-r2.tar.gz.sha256 USER@SERVER:~/
```

Then run on **the remote Linux server**. The extraction directory must be new:

```bash
cd ~
sha256sum -c misleadqa-v1.0.0-a40-r2.tar.gz.sha256 && \
mkdir misleadqa-v1.0.0-a40-r2 && \
tar -xzf misleadqa-v1.0.0-a40-r2.tar.gz -C misleadqa-v1.0.0-a40-r2 && \
cd misleadqa-v1.0.0-a40-r2
```

Continue only after successful extraction. Set these variables in each new
shell or tmux session, and run from the extracted project directory:

```bash
export CUDA_VISIBLE_DEVICES=0
export TF_PYTHON="$(conda run -n "${TF_ENV:-misleadqa-v100-tf}" python -c 'import sys; print(sys.executable)')"
export VLLM_PYTHON="$(conda run -n "${VLLM_ENV:-misleadqa-v100-vllm}" python -c 'import sys; print(sys.executable)')"
"$TF_PYTHON" -B experiments/main_eval/preflight.py --backend transformers --check-model-access
"$VLLM_PYTHON" -B experiments/main_eval/preflight.py --backend vllm --check-model-access
```

If a model access check fails, use `"$TF_PYTHON" -c
"from huggingface_hub import login; login()"` to authenticate with an account
that has accepted the corresponding model terms, then rerun preflight.
After both preflights pass, the launch commands below use these executables.
`TF_PYTHON` and `VLLM_PYTHON` take precedence over the Conda environment names;
`TF_PYTHON` also applies to `analyze_six.sh`.

### Launch smoke and full runs

The scripts run one model at a time and stop on the first error. Each completed
model is verified. Keep the server session alive, for example inside an existing
tmux session.

```bash
# Prepare all main prompts without loading weights.
bash experiments/main_eval/scripts/run_six.sh dry-run main

# Real inference: two deterministically sampled items per split, 16 calls/model.
bash experiments/main_eval/scripts/run_six.sh smoke main
bash experiments/main_eval/scripts/run_six.sh smoke supplement

# Full main: 7,672 calls/model, 46,032 for all six.
bash experiments/main_eval/scripts/run_six.sh full main

# Full supplement, separately: 200 calls/model, 1,200 for all six.
bash experiments/main_eval/scripts/run_six.sh full supplement
```

The default run IDs are main-v100-dry-run-v1, main-v100-smoke-v1,
main-v100-full-v1 and the corresponding supplement IDs. Smoke and full runs
must have different IDs. `TF_ENV` and `VLLM_ENV` override Conda environment names.
`RUN_ID` overrides the run ID; use a fresh ID if code, data, software, hardware,
seed or input limit changes. To resume after a normal interruption, repeat the
same command with the same ID; completed first responses are retained.

An abrupt kill may leave `.lock` in that model/run directory. Confirm the old
process has stopped before removing only that stale lock. The runner recovers
an incomplete final JSONL append; other corruption is rejected. Do not edit
existing manifests or responses to bypass a mismatch.

To run just one model:

```bash
conda run --no-capture-output -n misleadqa-v100-tf python -B experiments/main_eval/run.py \
  --model qwen3-8b --run-id main-v100-smoke-v1 --limit 2
```

## Analyze completed runs

```bash
bash experiments/main_eval/scripts/analyze_six.sh main
bash experiments/main_eval/scripts/analyze_six.sh supplement
```

Default output directories are `experiments/outputs/analysis/main-v100-full-v1` and
`experiments/outputs/analysis/supplement-v100-full-v1`. Existing analysis directories are
never overwritten; set `ANALYSIS_OUTPUT` to a new path to reanalyze.
If you used a custom inference RUN_ID, supply the same RUN_ID here.
The launcher explicitly enables mixed runtimes for the intended six-model panel.

- raw_responses.jsonl: first answer, exact messages/rendered input, input/output
  token IDs, condition, local scoring metadata and elapsed time.
- statistics.json / statistics.csv: unconditional and conditional outcome counts
  and rates, including each mixed order.
- contrasts.csv: F-minus-S, same-cohort resistance and order differences.
- item_metrics.jsonl: per-item outcomes and cohort membership.
- bootstrap.json: paired intervals, fixed panel eligibility and valid draws.
- provenance.json: exact input hashes, analysis hashes, runtime and precision.
- report.md: human-readable result summary.

B_m is closed-book correct. A_m is closed-book AND G-only correct. LR_X uses B_m.
DR_X averages the two mixed orders on A_m. Resistance_X compares single X to
mixed evidence on the SAME A_m. Order_X is (G then X) minus (X then G).
Positive resistance means G reduces lure selection; positive order effect means
more lure selection when X is last. All contrasts use percentage points.
Abstention, other choices and invalid answers are not counted as lure.

Primary panel Delta_LR and Delta_DR use two-sided 97.5% paired pooled
source-cluster percentile intervals, fixed equal model weights, at least 20
A_m source clusters per eligible model, 10,000 draws and seed 2027. Secondary
and individual-model intervals are exploratory 95%, without multiplicity
adjustment. Models below the threshold retain point estimates but no intervals.
The small supplement may have no eligible panel; null intervals are expected.
No review metadata is inferred from release inclusion.

To save draws or run dependency-group sensitivity analysis, use
`experiments/analysis/analyze.py --save-draws` or
`--cluster-field analysis_group_id` with an explicit new output directory.
Always analyze main and supplement separately. Cross-model differences can
reflect different conditional cohorts; the 32B result also differs in precision
and backend.

## Configuration files

`experiments/configs/models.json` pins checkpoints and chat-template settings;
`inference_policy.json` documents decoding and preparation. The AWQ engine
profile is `vllm_qwen32_awq_a40.json`. `prompts/qa_v1.txt` is the original
eight-condition prompt; `scope_v1.txt` and `context_faithful_v1.txt` define the
scope and robustness prompts. `requirements/` holds separate pinned Transformers
and vLLM GPU profiles. Original model, QA prompt and dependency bytes are checked
against `handoff_manifest.json` by `refresh_manifest.py`.
