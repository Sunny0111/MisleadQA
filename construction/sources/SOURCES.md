# Current release source inventory

source_manifest.json is extracted from both release 1.0.0 JSONL files. It records
357 distinct raw, donor and historical construction archives: 352 referenced by
the main split and 6 by the supplement, with one shared archive. None of these
referenced archives is supplied here. The split paths and SHA-256 values are
recorded in the manifest.

| Source | Current records |
| --- | ---: |
| FRED | 400 |
| SituatedQA | 367 |
| AmbigDocs | 192 |
| CondAmbigQA supplement | 25 |

URLs, revisions and hashes are inherited provenance metadata. License status
and original download dates remain unknown unless explicitly verified.
A reference URL need not serve the exact historical archived bytes.

Regenerate the inventory from the current release with:

```bash
python -B -m construction.sources.inventory
python -B -m construction.sources.download --list
```

The inventory preserves existing license annotations for matching archive paths.
It does not download content. The download utility accepts one unambiguous HTTPS
archive URL only when its bytes match the expected hash and its license metadata
has been supplied. `archive_path()` maps historical `data/` references into
`construction/sources/`; for example, `data/raw/ambigdocs/test.json` resolves to
`construction/sources/ambigdocs/test.json`. Restore archives at those locations.
Local archive subdirectories and retrieval logs are ignored by Git. New
retrieval timestamps do not establish original construction dates.

Source archives are optional for normal structural validation. To require all
referenced archives and verify their checksums, run:

```bash
python -B -m construction.screening.checks --require-raw
python -B -m construction.screening.checks --dataset construction/data/condition_supplement.jsonl --require-raw
```
