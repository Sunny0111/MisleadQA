# Context-faithful instruction comparison

Re-run all eight conditions with the same pinned checkpoints, greedy
choice-token/EOS decoding and seed 2027. The seven contextual conditions use
`experiments/configs/prompts/context_faithful_v1.txt`; closed-book retains the
original prompt. Compare both prompts on the original B_m/A_m cohorts, with
common-cohort sensitivities reported separately. The 25-item condition supplement
is a separate dataset, not part of the main comparison.

## Preparation and execution

Use the Transformers and vLLM environments documented in `../README.md`.
Refresh the source manifest and choose a new archive after source changes:

```bash
python -B experiments/main_eval/refresh_manifest.py
python -B experiments/main_eval/supplementary/package.py
python -B experiments/main_eval/supplementary/run.py --model qwen3-8b --run-id cf-preview-v1 --limit 2 --dry-run
```

On the GPU host, set `TF_PYTHON` and `VLLM_PYTHON` to the two environment
executables. Run smoke before full inference:

```bash
bash experiments/main_eval/supplementary/scripts/run.sh smoke main
bash experiments/main_eval/supplementary/scripts/run.sh full main
```

The launcher uses `main-cf-smoke-v1-cf-s2027` and
`main-cf-full-v1-cf-s2027`. Set `RUN_PREFIX` for another revision. Use fresh run
IDs after any source or environment change. Repeat an unchanged command to
resume missing requests. Each completed run has a checksum seal; verification
does not rewrite requests, answers or runtime metadata.

## Saved results and analysis

Run directories belong under
`experiments/outputs/supplementary/<model>/<run-id>/`. The six historical full
runs use `main-robust-full-v1-cf-s2027`; their files retain their original bytes.
Historical paths and code fingerprints in their manifests remain provenance,
not requirements to restore the old directory layout.

Analyze saved runs into a new directory:

```bash
python -B experiments/main_eval/supplementary/analyze.py --root experiments/outputs/supplementary --prefix main-robust-full-v1 --baseline-root experiments/outputs/main_eval --output experiments/outputs/analysis/context-faithful-new
python -B experiments/main_eval/supplementary/context_report.py --directory experiments/outputs/analysis/context-faithful-new
```

For new launcher runs, use `--prefix main-cf-full-v1` and a matching
`--baseline-run-id`. Both runs must share the frozen code and runtime profile.
Analysis verifies paired
datasets, model/runtime metadata and decoding, then exports numerical JSON/CSV.
`context_report.py` verifies input hashes and exports `context_summary.json`
for the CF figure. Its `--run-root` and `--baseline-root` options locate raw
runs without changing historical provenance. Original cohort rates, all five
answer outcomes, paired differences and transitions remain available.

CF changes the instructed task. Report lure, gold and abstention rates together;
a lower lure rate alone does not imply improved gold accuracy. These comparisons
are descriptive and do not generate significance claims.
