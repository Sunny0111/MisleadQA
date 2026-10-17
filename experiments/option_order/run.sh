#!/usr/bin/env bash
# Run all six models sequentially on one A40 using the original environments.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
mode="${1:-smoke}"
case "$mode" in smoke|full|dry-run) ;; *) echo 'Use smoke, full, or dry-run' >&2; exit 2;; esac
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
tf_python="${TF_PYTHON:-python3}"
vllm_python="${VLLM_PYTHON:-$tf_python}"
if [[ "$mode" != dry-run ]]; then
  : "${TF_PYTHON:?Set TF_PYTHON to the original Transformers environment python}"
  : "${VLLM_PYTHON:?Set VLLM_PYTHON to the original vLLM environment python}"
  "$tf_python" -B experiments/main_eval/preflight.py --backend transformers
  "$vllm_python" -B experiments/main_eval/preflight.py --backend vllm
fi
run_id="${RUN_ID:-main-option-order-${mode}-v1}"
extra=()
if [[ "$mode" == smoke ]]; then extra+=(--smoke); fi
if [[ "$mode" == dry-run ]]; then extra+=(--dry-run); fi
for model in qwen3-8b qwen3-14b llama31-8b-instruct gemma3-12b-it olmo2-13b-instruct qwen3-32b-awq; do
  python_command="$tf_python"
  if [[ "$model" == qwen3-32b-awq ]]; then python_command="$vllm_python"; fi
  "$python_command" -B experiments/option_order/run.py --model "$model" --run-id "$run_id" "${extra[@]}"
done
echo "OPTION ORDER ${mode} COMPLETE"
