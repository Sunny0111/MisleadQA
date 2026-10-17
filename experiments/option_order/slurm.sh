#!/usr/bin/env bash
# One model per task. Slurm copies this file, so use the submitted working directory.
#SBATCH --job-name=misleadqa-order
#SBATCH --partition=gpu_nodes
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-5%1
#SBATCH --signal=TERM@120

set -euo pipefail
: "${SLURM_JOB_ID:?Submit through submit-slurm.sh}"
cd -- "${SLURM_SUBMIT_DIR:?Missing Slurm working directory}"
mode="${1:-check}"
case "$mode" in check|smoke|full|f-only) ;; *) echo 'Use check, smoke, full, or f-only' >&2; exit 2;; esac
models=(qwen3-8b qwen3-14b llama31-8b-instruct gemma3-12b-it olmo2-13b-instruct qwen3-32b-awq)
index="${SLURM_ARRAY_TASK_ID:?Submit as a job array}"
[[ "$index" =~ ^[0-5]$ ]] || { echo 'Array index must be 0-5' >&2; exit 2; }
model="${models[$index]}"
backend=transformers
python_command="${TF_PYTHON:-.venv-tf/bin/python}"
if [[ "$model" == qwen3-32b-awq ]]; then
  backend=vllm
  python_command="${VLLM_PYTHON:-.venv-vllm/bin/python}"
fi
[[ -x "$python_command" ]] || { echo "Missing Python executable: $python_command" >&2; exit 2; }
# Retain Slurm's GPU assignment; never replace CUDA_VISIBLE_DEVICES with 0.
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
echo "Job=$SLURM_JOB_ID task=$index host=$(hostname) model=$model mode=$mode"
echo "Python=$python_command CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
"$python_command" -B experiments/option_order/prepare.py --verify-bundle .
preflight=(--backend "$backend")
if [[ "$mode" == check ]]; then preflight+=(--check-model-access); fi
srun --unbuffered "$python_command" -u -B experiments/main_eval/preflight.py "${preflight[@]}"
if [[ "$mode" == check ]]; then
  echo "OPTION ORDER CHECK COMPLETE: $backend"
  exit 0
fi

prefix="${RUN_PREFIX:-main-option-order-slurm}"
pairing=(--rerun-original)
suffix=fresh-v1
if [[ "$mode" == f-only ]]; then
  baseline_root="${BASE_RUN_ROOT:-experiments/outputs/option_order}"
  baseline_id="${BASE_RUN_ID:-main-option-order-slurm-full-fresh-v1}"
  pairing=(--f-only-from "$baseline_root/$model/$baseline_id")
  suffix=f-only-v1
fi
run_model() {
  local stage="$1"
  shift
  srun --unbuffered "$python_command" -u -B experiments/option_order/run.py \
    --model "$model" "${pairing[@]}" --run-id "$prefix-$stage-$suffix" "$@"
}
run_model smoke --smoke
if [[ "$mode" == full || "$mode" == f-only ]]; then run_model full; fi
echo "OPTION ORDER $mode COMPLETE: $model"
