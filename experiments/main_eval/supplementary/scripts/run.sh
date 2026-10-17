#!/usr/bin/env bash
# Sequential context-faithful evaluation; invoke via bash.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.."
mode="${1:-smoke}"
split="${2:-main}"
case "$mode" in smoke|full|dry-run) ;; *) echo 'Use smoke, full, or dry-run' >&2; exit 2;; esac
case "$split" in
  main) dataset='construction/data/main_split.jsonl';;
  supplement) dataset='construction/data/condition_supplement.jsonl';;
  *) echo 'Use main or supplement' >&2; exit 2;;
esac
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
if [[ "$mode" != dry-run ]]; then
  : "${TF_PYTHON:?Set TF_PYTHON to the Transformers environment python}"
  : "${VLLM_PYTHON:?Set VLLM_PYTHON to the vLLM environment python}"
  "$TF_PYTHON" -B experiments/main_eval/preflight.py --backend transformers
  "$VLLM_PYTHON" -B experiments/main_eval/preflight.py --backend vllm
fi
tf_python="${TF_PYTHON:-python3}"
vllm_python="${VLLM_PYTHON:-$tf_python}"
prefix="${RUN_PREFIX:-${split}-cf-${mode}-v1}"
extra=()
verify_extra=()
if [[ "$mode" == smoke ]]; then extra+=(--limit 2); verify_extra+=(--smoke-check); fi
if [[ "$mode" == dry-run ]]; then extra+=(--dry-run); fi
for model in qwen3-8b qwen3-14b llama31-8b-instruct gemma3-12b-it olmo2-13b-instruct qwen3-32b-awq; do
  python_command="$tf_python"
  if [[ "$model" == qwen3-32b-awq ]]; then python_command="$vllm_python"; fi
  run_id="${prefix}-cf-s2027"
  "$python_command" -B experiments/main_eval/supplementary/run.py --dataset "$dataset" --model "$model" \
    --run-id "$run_id" "${extra[@]}"
  if [[ "$mode" != dry-run ]]; then
    "$python_command" -B experiments/main_eval/supplementary/verify.py \
      "experiments/outputs/supplementary/$model/$run_id" --dataset "$dataset" "${verify_extra[@]}"
  fi
done
echo "CONTEXT FAITHFUL ${mode} ${split} COMPLETE"
