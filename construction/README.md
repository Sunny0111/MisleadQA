# Dataset construction

| Directory | Contents |
| --- | --- |
| [sources/](sources/SOURCES.md) | Source inventory, archive manifest and retrieval tools |
| [build/](build/construction.md) | Scope and G/S/F guidelines, candidate export and cluster index generation |
| [screening/](screening/screening.md) | Structural checks, LLM screening instructions and pending-request export |
| `adjudication/` | Whole-item verification and revision record templates |
| [audit/](audit/README.md) | Frozen main150 sample, assignments, reviewer guide and local review app |
| `human audit/main150-v1/` | Six collected formal submissions; local drafts are ignored |
| [data/](data/DATASET_CARD.md) | Frozen main and condition splits, clusters, hashes and errata |

## Preparation and structural validation

Run from the repository root with Python 3.12:

```bash
python -B -m construction.build.build --status
python -B -m construction.build.clusters --check
python -B -m construction.screening.checks --report tmp/checks-main.json
python -B -m construction.screening.checks --dataset construction/data/condition_supplement.jsonl --report tmp/checks-supplement.json
python -B -m construction.screening.screening --limit 2 --output construction/screening/outputs/requests-preview.jsonl
```

`build/build.py` exports an existing structurally validated candidate unchanged;
it does not reconstruct the original dataset. The original generator and
balancing pipeline are not supplied. `build/clusters.py` derives source clusters
and dependency groups from the frozen splits. If intentionally regenerating
`clusters.json`, review it and update its checksum in `data/hashes.txt`.

`screening/checks.py` validates structure, evidence hashes, false edits, option
roles, group sizes and release checksums. Add `--require-raw` to check local
source archives and their checksums; only this mode reports missing archives.
`screening/screening.py` exports pending
requests without claiming completed reviews. Store generated requests, raw
responses and reports in the ignored `screening/outputs/` directory, retaining
model identity, prompt, item ID and revision. Structural success does not
independently establish evidence factuality.

## Audit and adjudication

The [human audit](audit/README.md) contains 300 reviews of 150 main items:
149 passed all three checks for both reviewers; one awaits organizer
adjudication. Analyze only the six `*.submitted.json` files. Each includes the
displayed inputs, judgments, notes, hashes and timestamps. Editable drafts are
local working files and are not additional observations.

`adjudication/template.json` and `adjudication/verification/template.json` are
schema examples, not completed decisions. Preserve both original judgments and
record contested items, Q1-Q3 findings, rationales, assignment versions and
timestamps separately. A No response or disagreement is not automatically a
confirmed construction error. The templates do not encode every audit field;
retain missing fields in the review ledger.

For this release, `revise` and `reject` are error labels only. Record confirmed
issues in [data/errata.md](data/errata.md). Exclusion lists belong to separately
reported sensitivity analyses using the original responses. Keep v1.0.0 and
primary results frozen; actual corrections require a future release with
before/after content, hashes and rechecks. Do not infer review or eligibility
flags from an export. Completed screening ledgers and organizer adjudications
are not supplied; release records omit review metadata.
