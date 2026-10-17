#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.."
split="${1:-main}"
case "$split" in
  main) dataset='construction/data/main_split.jsonl';;
  supplement) dataset='construction/data/condition_supplement.jsonl';;
  *) echo 'Split must be main or supplement' >&2; exit 2;;
esac
run_id="${RUN_ID:-${split}-v100-full-v1}"
output="${ANALYSIS_OUTPUT:-experiments/outputs/analysis/$run_id}"
runs=()
for model in qwen3-8b qwen3-14b llama31-8b-instruct gemma3-12b-it olmo2-13b-instruct qwen3-32b-awq; do
  runs+=("experiments/outputs/main_eval/$model/$run_id")
done
python_command=(conda run --no-capture-output -n "${TF_ENV:-misleadqa-v100-tf}" python)
if [[ -n "${TF_PYTHON:-}" ]]; then python_command=("$TF_PYTHON"); fi
"${python_command[@]}" -B experiments/analysis/analyze.py \
  "${runs[@]}" --dataset "$dataset" --output "$output" --allow-mixed-runtimes
