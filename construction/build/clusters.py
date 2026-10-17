"""Export or verify source-cluster and dependency-group membership for the release."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

from misleadqa_utils import (CURRENT_SCHEMA, CURRENT_VERSION, DEFAULT_DATASET,
                            RELEASE_DIR, ROOT, SUPPLEMENT_DATASET, read_jsonl,
                            sha256_file, write_json)


def release_clusters():
    result = {"dataset_version": CURRENT_VERSION, "schema_version": CURRENT_SCHEMA,
              "source_cluster_key": ["source", "grouping.cluster_id"],
              "dependency_group_key": "grouping.analysis_group_id", "splits": {}}
    for name, path in (("main", DEFAULT_DATASET), ("condition_supplement", SUPPLEMENT_DATASET)):
        rows = read_jsonl(path)
        clusters, groups = defaultdict(list), defaultdict(list)
        seen = set()
        for row in rows:
            if row["id"] in seen:
                raise ValueError(f"Duplicate item ID in {name}: {row['id']}")
            seen.add(row["id"])
            clusters[(row["source"], row["grouping"]["cluster_id"])].append(row["id"])
            groups[row["grouping"]["analysis_group_id"]].append(row["id"])
        result["splits"][name] = {
            "dataset": path.relative_to(ROOT).as_posix(), "dataset_sha256": sha256_file(path),
            "items": len(rows), "source_cluster_count": len(clusters),
            "dependency_group_count": len(groups),
            "source_clusters": [{"source": source, "cluster_id": cluster, "item_ids": sorted(ids)}
                                for (source, cluster), ids in sorted(clusters.items())],
            "dependency_groups": [{"analysis_group_id": group, "item_ids": sorted(ids)}
                                  for group, ids in sorted(groups.items())],
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RELEASE_DIR / "clusters.json")
    parser.add_argument("--check", action="store_true", help="Verify the existing index without writing")
    args = parser.parse_args()
    expected = release_clusters()
    if args.check:
        if json.loads(args.output.read_text(encoding="utf-8")) != expected:
            raise ValueError("Cluster index differs from the frozen release")
        entries = [line.split() for line in (RELEASE_DIR / "hashes.txt").read_text().splitlines()]
        recorded = [entry[0] for entry in entries if len(entry) == 2 and entry[1] == args.output.name]
        if recorded != [sha256_file(args.output)]:
            raise ValueError("Cluster index checksum mismatch")
        print("Cluster membership and checksum verified")
    else:
        write_json(args.output, expected)
        print(f"Wrote cluster index: {args.output}")


if __name__ == "__main__":
    main()
