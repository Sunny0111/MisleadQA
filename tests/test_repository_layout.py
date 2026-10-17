"""Verify frozen grouping metadata and archive resolution after reorganization."""
from collections import Counter
import hashlib
import json
import unittest

from construction.build.clusters import release_clusters
from misleadqa_utils import (DEFAULT_DATASET, RELEASE_DIR, ROOT, SOURCE_DIR,
                            SUPPLEMENT_DATASET, archive_path, read_jsonl, sha256_file)


class RepositoryLayoutTests(unittest.TestCase):
    def test_version_normalization_preserves_frozen_item_content(self):
        record = json.loads((RELEASE_DIR / "version_normalization.json").read_text(encoding="utf-8"))
        self.assertEqual((record["dataset_version"], record["records"]), ("1.0.0", 984))
        self.assertEqual(record["changed_dataset_fields"], ["version"])
        self.assertFalse(record["model_inference_repeated"])
        for info in record["splits"].values():
            path = ROOT / info["dataset"]
            rows = read_jsonl(path)
            content = [{key: value for key, value in row.items() if key != "version"} for row in rows]
            encoded = json.dumps(content, ensure_ascii=False, sort_keys=True,
                                 separators=(",", ":")).encode("utf-8")
            self.assertEqual(hashlib.sha256(encoded).hexdigest(), info["content_without_version_sha256"])
            self.assertEqual(sha256_file(path), info["normalized_file_sha256"])
            self.assertTrue(all(row["version"] == "1.0.0" for row in rows))

    def test_available_saved_requests_and_responses_keep_their_original_bytes(self):
        record = json.loads((RELEASE_DIR / "version_normalization.json").read_text(encoding="utf-8"))
        checked = 0
        for relative, expected_hash in record["preserved_files_sha256"].items():
            path = ROOT / relative
            # Portable source bundles exclude local inference outputs.
            if not path.exists() and "experiments/outputs/" in relative:
                continue
            self.assertEqual(sha256_file(path), expected_hash, relative)
            checked += 1
        self.assertGreater(checked, 0)

    def test_cluster_index_partitions_both_frozen_splits(self):
        stored = json.loads((RELEASE_DIR / "clusters.json").read_text(encoding="utf-8"))
        self.assertEqual(stored, release_clusters())
        for split, path, expected_counts in (
                ("main", DEFAULT_DATASET, (959, 489, 132)),
                ("condition_supplement", SUPPLEMENT_DATASET, (25, 20, 20))):
            index = stored["splits"][split]
            self.assertEqual((index["items"], index["source_cluster_count"],
                              index["dependency_group_count"]), expected_counts)
            self.assertEqual(index["dataset_sha256"], sha256_file(path))
            expected_ids = Counter(row["id"] for row in read_jsonl(path))
            for field in ("source_clusters", "dependency_groups"):
                members = Counter(item_id for group in index[field] for item_id in group["item_ids"])
                self.assertEqual(members, expected_ids)
        checksums = {name: digest for digest, name in
                     (line.split() for line in (RELEASE_DIR / "hashes.txt").read_text().splitlines())}
        self.assertEqual(checksums["clusters.json"], sha256_file(RELEASE_DIR / "clusters.json"))

    def test_historical_archive_references_resolve_under_sources(self):
        for reference, expected in (
                ("data/raw/fred/example.json", "fred/example.json"),
                ("data/supplied_v1/sources/proof.json", "supplied_v1/sources/proof.json"),
                ("construction/sources/situatedqa/example.json", "situatedqa/example.json")):
            self.assertEqual(archive_path(reference), (SOURCE_DIR / expected).resolve())
        for reference in ("../outside.json", "construction/data/main_split.jsonl",
                          "construction/sources/../../outside.json", "data/../../outside.json",
                          str(ROOT.parent / "outside.json")):
            with self.subTest(reference=reference), self.assertRaises(ValueError):
                archive_path(reference)


if __name__ == "__main__":
    unittest.main()
