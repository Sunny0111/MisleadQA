#!/usr/bin/env bash
# Sequential jobs for a single A40. Invoke with bash; executable bit is optional.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.."
mode="${1:-smoke}"
split="${2:-main}"
case "$mode" in smoke|full|dry-run) ;; *) echo 'Mode must be smoke, full or dry-run' >&2; exit 2;; esac
case "$split" in
  main) dataset='construction/data/main_split.jsonl';;
  supplement) dataset='construction/data/condition_supplement.jsonl';;
  *) echo 'Split must be main or supplement' >&2; exit 2;;
esac
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
tf_env="${TF_ENV:-misleadqa-v100-tf}"
vllm_env="${VLLM_ENV:-misleadqa-v100-vllm}"
tf_command=(conda run --no-capture-output -n "$tf_env" python)
vllm_command=(conda run --no-capture-output -n "$vllm_env" python)
if [[ -n "${TF_PYTHON:-}" ]]; then tf_command=("$TF_PYTHON"); fi
if [[ -n "${VLLM_PYTHON:-}" ]]; then vllm_command=("$VLLM_PYTHON"); fi
run_id="${RUN_ID:-${split}-v100-${mode}-v1}"
extra=()
if [[ "$mode" == smoke ]]; then extra+=(--limit 2); fi
if [[ "$mode" == dry-run ]]; then extra+=(--dry-run); fi
models=(qwen3-8b qwen3-14b llama31-8b-instruct gemma3-12b-it olmo2-13b-instruct qwen3-32b-awq)
for model in "${models[@]}"; do
  python_command=("${tf_command[@]}")
  if [[ "$model" == qwen3-32b-awq ]]; then python_command=("${vllm_command[@]}"); fi
  "${python_command[@]}" -B experiments/main_eval/run.py \
    --dataset "$dataset" --model "$model" --run-id "$run_id" "${extra[@]}"
  if [[ "$mode" != dry-run ]]; then
    "${python_command[@]}" -B experiments/main_eval/verify.py \
      "experiments/outputs/main_eval/$model/$run_id" --dataset "$dataset"
  fi
done
