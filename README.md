# MisleadQA

MisleadQA measures whether external evidence shifts a model from a correct
closed-book answer to a designated incorrect lure. G is true and applicable;
F is false at the target scope; S is true at another scope. F and S support the
same lure. Each item has five choices, including "Insufficient information".

## Dataset and evaluation

The frozen v1.0.0 release (schema 1.1.0) contains **984 items**. Main and
condition-supplement results are evaluated and reported separately.

| Split | Items | Sources | Requests per model |
| --- | ---: | --- | ---: |
| main | 959 | FRED 400; SituatedQA 367; AmbigDocs 192 | 7,672 |
| condition_supplement | 25 | CondAmbigQA | 200 |

Each item receives eight independent one-turn requests: `closed_book`, `g_only`,
`f_only`, `s_only`, `g_then_f`, `f_then_g`, `g_then_s`, and `s_then_g`.
Only questions, labeled options and ordered evidence texts are sent to models.
Scoring labels remain in local records. Five models use BF16 Transformers;
Qwen3-32B-AWQ uses vLLM with official AWQ weights.

See the [dataset card](construction/data/DATASET_CARD.md),
[checksums](construction/data/hashes.txt), and
[metric definitions](experiments/analysis/README.md#metrics).
The [version normalization record](construction/data/version_normalization.json)
documents the label-only change to v1.0.0; questions, evidence, options, answer
keys, and saved model responses were preserved.

## Quick start

Use Python 3.12 (reference version: `.python-version`). Preparation, structural
checks and evaluation orchestration use the standard library. The full offline
test suite and paired reanalysis also require NumPy; inference and plotting
have separate requirements.
Run commands from the repository root. On Windows, `py -3.12` can replace
`python` when that interpreter is installed.

```bash
python -m pip install -r requirements.txt
python -B experiments/main_eval/refresh_manifest.py --check
python -B -m unittest discover -s tests -v
python -B -m construction.screening.checks --report tmp/checks-main.json
python -B -m construction.screening.checks --dataset construction/data/condition_supplement.jsonl --report tmp/checks-supplement.json
python -B -m construction.build.clusters --check
python -B experiments/main_eval/run.py --model qwen3-8b --run-id main-v100-preview-v1 --limit 2 --dry-run
python -B experiments/main_eval/package.py
```

The package command creates a checksummed source archive under `dist/`. Choose
`--output PATH` for a new archive if the default already exists. After changing
source or protocol documentation, run `refresh_manifest.py` without `--check`
before testing or packaging. Use fresh run IDs after code changes.

## Repository guide

| Path | Contents and documentation |
| --- | --- |
| [construction/](construction/README.md) | Source inventory, construction and screening tools, adjudication templates |
| [construction/data/](construction/data/DATASET_CARD.md) | Frozen splits, cluster index, checksums and errata |
| [construction/audit/](construction/audit/README.md) | Frozen sample, assignments, reviewer guide and local review app |
| `construction/human audit/main150-v1/` | Six collected formal audit submissions |
| `experiments/configs/` | Pinned checkpoints, decoding policy, prompts and GPU requirements |
| [experiments/main_eval/](experiments/main_eval/README.md) | Main evaluation, GPU setup, verification and packaging |
| [experiments/main_eval/supplementary/](experiments/main_eval/supplementary/README.md) | Context-faithful instruction comparison |
| [experiments/scope_diagnostic/](experiments/scope_diagnostic/README.md) | Balanced 200-item scope diagnostic |
| [experiments/option_order/](experiments/option_order/README.md) | Option-order robustness; [Slurm workflow](experiments/option_order/SLURM.md) |
| [experiments/analysis/](experiments/analysis/README.md) | Metrics, bootstrap intervals and saved-result analysis |
| [experiments/figures/](experiments/figures/README.md) | Figure and table generation |
| `tests/` | Offline regression and integration tests |

The completed human audit covers 150 main items, with two reviewers per item:
149 received two all-Yes reviews and one awaits adjudication. It assesses textual
construction, not independent source factuality. Reproduce its checks with
`node construction/audit/analysis/analyze.mjs` (Node.js 24.14+).

## Files included in a submission

Source code, tests, frozen data, configuration, audit assignments and formal
submissions are included. Downloads, raw source caches, editable audit drafts,
local responses, generated reports, inference outputs, dependencies and archives
are ignored by Git and excluded from portable source bundles. Local artifacts
can be regenerated using the relevant module guide; full analyses require saved
inference outputs. Historical manuscript sources are not included.

The original construction generator, completed screening ledger and organizer
adjudications are not supplied. Release records omit review/eligibility metadata;
missing status remains unknown. Structural validation does not establish the
factual correctness of G or S. Historical archive paths are resolved by
`archive_path()` in `misleadqa_utils.py`; see the [source inventory](construction/sources/SOURCES.md).
Data terms and authorship remain as recorded in [LICENSE-DATA](LICENSE-DATA) and
[CITATION.cff](CITATION.cff); code licensing is in [LICENSE](LICENSE).
