# Release errata

Current dataset version: 1.0.0. Schema version: 1.1.0.

No additional item corrections are recorded in this file. The repository
reorganization renamed the two frozen JSONL files to `main_split.jsonl` and
`condition_supplement.jsonl` without changing their bytes. `clusters.json` is a
derived membership index; `hashes.txt` records the frozen input checksums.

## Dataset version label normalization

On 2026-10-04, all 984 records were assigned dataset version 1.0.0 at the
repository owner's request. The previous label was 1.0.1. Only the top-level
`version` field changed; question text, evidence, options, answer keys, item IDs,
source references and grouping are identical. Schema remains 1.1.0.

All recorded model measurements refer to this same 984-item v1.0.0 release.
Dataset checksums, run-manifest references, the frozen scope sample, source
inventory and analysis provenance were synchronized. Original requests and raw
responses were not regenerated or modified. The old labels and file hashes are
retained only as migration provenance in `version_normalization.json` and the
saved run manifests. Result directories now use `v100` names.

Historical source archive paths inside the JSONL records are retained as
provenance and resolved under `construction/sources/` by `archive_path()` in
`misleadqa_utils.py`.

For any future confirmed correction, record the item ID, affected fields,
reason, supporting evidence, old/new hashes and release version. Do not silently
rewrite a frozen split.
