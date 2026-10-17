"""Read-only bridge for the audited dataset version-label normalization."""
import hashlib
import json
from pathlib import Path


def dataset_provenance(dataset, recorded_sha256):
    """Accept exact bytes, or prove the old bytes by reversing only version labels.

    No saved dataset, manifest, response, or completion seal is rewritten.
    The normalization record alone is insufficient: restored bytes must hash to
    the SHA-256 frozen in the inference manifest.
    """
    dataset = Path(dataset)
    content = dataset.read_bytes()
    actual = hashlib.sha256(content).hexdigest()
    result = {"path": str(dataset.resolve()), "file_sha256": actual,
              "recorded_sha256": recorded_sha256, "match": "exact"}
    if actual == recorded_sha256:
        return result
    record_path = dataset.parent / "version_normalization.json"
    if not record_path.is_file():
        raise ValueError("Dataset hash mismatch without a normalization record")
    record_bytes = record_path.read_bytes()
    record = json.loads(record_bytes)
    if (record.get("changed_dataset_fields") != ["version"]
            or record.get("dataset_version") != "1.0.0"
            or record.get("original_dataset_version") != "1.0.1"):
        raise ValueError("Unsupported dataset normalization")
    matches = [s for s in record["splits"].values()
               if s["normalized_file_sha256"] == actual
               and s["original_file_sha256"] == recorded_sha256]
    old, new = b'"version": "1.0.1"', b'"version": "1.0.0"'
    if len(matches) != 1 or content.count(new) != matches[0]["records"]:
        raise ValueError("Dataset does not match the recorded version-only migration")
    restored = content.replace(new, old)
    if hashlib.sha256(restored).hexdigest() != recorded_sha256:
        raise ValueError("Reversing the version labels does not reproduce the inference dataset")
    return result | {"match": "verified-version-label-normalization",
                     "restored_sha256": recorded_sha256,
                     "records": matches[0]["records"],
                     "normalization_record": str(record_path.resolve()),
                     "normalization_record_sha256": hashlib.sha256(record_bytes).hexdigest()}
