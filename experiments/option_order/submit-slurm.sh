#!/usr/bin/env bash
# Print only the parsable job ID to stdout; pass site overrides through to sbatch.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
mode="${1:-check}"
if [[ $# -gt 0 ]]; then shift; fi
case "$mode" in check|smoke|full|f-only) ;; *) echo 'Use check, smoke, full, or f-only' >&2; exit 2;; esac
mkdir -p logs
array=0-5%1
if [[ "$mode" == check ]]; then array=0,5%1; fi
sbatch --parsable --array="$array" --chdir="$PWD" \
  --output="$PWD/logs/option-order-%A_%a.out" \
  --error="$PWD/logs/option-order-%A_%a.err" \
  "$@" experiments/option_order/slurm.sh "$mode"
