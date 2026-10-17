"""Small shared helpers for the configurable MisleadQA scaffold."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RELEASE_DIR = ROOT / "construction/data"
SOURCE_DIR = ROOT / "construction/sources"
CONFIG_DIR = ROOT / "experiments/configs"
OUTPUTS_DIR = ROOT / "experiments/outputs"
DEFAULT_DATASET = RELEASE_DIR / "main_split.jsonl"
SUPPLEMENT_DATASET = RELEASE_DIR / "condition_supplement.jsonl"
CURRENT_VERSION = "1.0.0"
CURRENT_SCHEMA = "1.1.0"


def review_flag(item, field):
    """Unknown review metadata stays unknown; release inclusion is not approval."""
    review = item.get("review")
    return review.get(field) if isinstance(review, dict) else None


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"{path}:{number}: blank JSONL record")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{number}: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number}: expected an object")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no records")
    return rows


def load_config(path):
    # Repository configurations use JSON syntax.
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def archive_path(relative):
    """Resolve current or frozen historical source references under sources/."""
    path = (ROOT / relative).resolve()
    legacy_root = ROOT / "data"
    if path.is_relative_to(legacy_root):
        suffix = path.relative_to(legacy_root)
        if suffix.parts and suffix.parts[0] == "raw":
            suffix = Path(*suffix.parts[1:])
        path = (SOURCE_DIR / suffix).resolve()
    if not path.is_relative_to(SOURCE_DIR.resolve()):
        raise ValueError(f"Archive path must stay inside construction/sources/: {relative}")
    return path


def source_refs(item):
    for evidence in item.get("evidence", {}).values():
        yield from evidence.get("source_refs", [])
    for distractor in item.get("distractors", []):
        yield from distractor.get("source_refs", [])


def archive_refs(item):
    """Include raw, donor, and historical construction archive dependencies."""
    for ref in source_refs(item):
        for prefix in ("archive_file", "donor_file"):
            if ref.get(prefix):
                yield ref[prefix], ref.get(prefix + "_sha256", "")
    provenance = item.get("provenance", {})
    if provenance.get("archive_file"):
        yield provenance["archive_file"], provenance.get("archive_file_sha256", "")

