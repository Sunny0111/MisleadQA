"""Rebuild the offline source inventory from both current release splits."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from misleadqa_utils import (ROOT, DEFAULT_DATASET, SUPPLEMENT_DATASET, archive_path,
                            read_jsonl, sha256_file, source_refs, write_json)


def build_inventory(output):
    output = Path(output)
    previous = json.loads(output.read_text(encoding="utf-8")) if output.exists() else {}
    old = {r["archive_file"]: r for r in previous.get("archives", [])}
    collected = defaultdict(lambda: defaultdict(set))
    datasets = []
    for path in (DEFAULT_DATASET, SUPPLEMENT_DATASET):
        rows = read_jsonl(path)
        datasets.append({"path": path.relative_to(ROOT).as_posix(), "sha256": sha256_file(path), "records": len(rows)})
        for row in rows:
            refs = list(source_refs(row)) + [row.get("provenance", {}) | {"kind": "construction_provenance"}]
            for ref in refs:
                for key in ("archive_file", "donor_file"):
                    relative = ref.get(key)
                    if not relative:
                        continue
                    entry = collected[relative]
                    entry["expected_sha256"].add(ref.get(key + "_sha256", ""))
                    entry["source_families"].add(row["source"])
                    entry["reference_kinds"].add(ref.get("kind", "unspecified"))
                    if ref.get("source_url"):
                        entry["source_urls"].add(ref["source_url"])
                    if ref.get("source_revision"):
                        entry["source_revisions"].add(ref["source_revision"])
    archives = []
    for relative, values in sorted(collected.items()):
        prior = old.get(relative, {})
        entry = {"archive_file": relative, **{key: sorted(values[key]) for key in
                 ("expected_sha256", "source_urls", "source_revisions", "source_families", "reference_kinds")}}
        entry.update({key: prior.get(key, default) for key, default in
                     (("license", None), ("license_verified", False), ("original_download_date", None))})
        entry["archive_status"] = "present" if archive_path(relative).is_file() else "not_supplied"
        archives.append(entry)
    manifest = {"status": "metadata_extracted_from_release; source_terms_unverified",
                "datasets": datasets, "archives": archives}
    write_json(output, manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "construction/sources/source_manifest.json")
    args = parser.parse_args()
    result = build_inventory(args.output)
    print(f"Wrote {len(result['archives'])} archive references from both current release splits")
