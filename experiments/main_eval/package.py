"""Create and byte-verify a portable bundle of both frozen release splits."""
import argparse
import io
import json
import os
from pathlib import Path, PurePosixPath
import sys
import tarfile

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "experiments.main_eval"

from construction.screening.checks import validate_dataset
from misleadqa_utils import CURRENT_VERSION, DEFAULT_DATASET, SUPPLEMENT_DATASET
from .protocol import ROOT, PROTOCOL_VERSION, file_hash, load_items, load_models
from .refresh_manifest import check_manifest


def source_files(directory):
    """Collect portable source files without entering local runtime directories."""
    excluded = {"outputs", "__pycache__", "node_modules", "dist", ".vite", "responses"}
    extensions = {".py", ".json", ".txt", ".sh", ".md", ".html", ".js", ".mjs", ".jsx", ".css", ".svg"}
    for current, directories, files in os.walk(directory):
        directories[:] = sorted(name for name in directories if name not in excluded)
        for name in sorted(files):
            path = Path(current) / name
            if (path.suffix in extensions or name == ".gitignore") and not name.endswith(".draft.json"):
                # The audit analyzer regenerates these local reports.
                if path.parent == ROOT / "construction/audit/analysis" and path.suffix in {".json", ".html"}:
                    continue
                yield path


def public_tar_metadata(member):
    """Omit local account identities and filesystem timestamps from releases."""
    member.uid = member.gid = 0
    member.uname = member.gname = ""
    member.mtime = 0
    member.pax_headers = {}
    return member


def build_package(destination, extra_files=None):
    destination = Path(destination)
    checksum_path = destination.with_name(destination.name + ".sha256")
    if destination.exists() or checksum_path.exists():
        raise ValueError("Package/checksum already exists; choose a new destination")
    check_manifest()
    splits = {}
    for name, path in (("main", DEFAULT_DATASET), ("supplement", SUPPLEMENT_DATASET)):
        report = validate_dataset(path)
        if report["status"] != "pass":
            raise ValueError(f"Release validation failed: {path}: {report['errors'][:3]}")
        items, counts = load_items(path)
        splits[name] = {"path": path.relative_to(ROOT).as_posix(), "sha256": file_hash(path),
                        **counts, "requests_per_model": len(items) * 8, "requests_six_models": len(items) * 48}
    paths = list(source_files(ROOT / "experiments"))
    paths += list((ROOT / "tests").glob("*.py"))
    for directory in ("build", "screening", "adjudication", "audit"):
        paths += list(source_files(ROOT / "construction" / directory))
    paths += list((ROOT / "construction/human audit/main150-v1").glob("*.submitted.json"))
    paths += list((ROOT / "construction").glob("*.py"))
    paths += list((ROOT / "construction").glob("*.md"))
    paths += list((ROOT / "construction/sources").glob("*.py"))
    paths += list((ROOT / "construction/sources").glob("*.md"))
    paths += list((ROOT / "construction/data").glob("*"))
    paths += [ROOT / name for name in ("misleadqa_utils.py", "README.md", "requirements.txt",
               ".python-version", ".gitignore", ".gitattributes", "LICENSE", "LICENSE-DATA",
               "construction/sources/source_manifest.json", "construction/sources/SOURCES.md")]
    entries = {p.relative_to(ROOT).as_posix(): p for p in paths if p.is_file()}
    for name, path in (extra_files or {}).items():
        relative = PurePosixPath(name)
        if (relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name
                or name in entries or name == "bundle_manifest.json" or str(relative) != name):
            raise ValueError(f"Unsafe or duplicate extra bundle path: {name}")
        entries[name] = Path(path)
    info = {"protocol": PROTOCOL_VERSION, "dataset_version": CURRENT_VERSION,
            "splits": splits, "selected_models": list(load_models()),
            "requests_six_models": sum(s["requests_six_models"] for s in splits.values()),
            "files_sha256": {name: file_hash(path) for name, path in sorted(entries.items())}}
    info_bytes = (json.dumps(info, indent=2) + "\n").encode("utf-8")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(destination, "x:gz", format=tarfile.PAX_FORMAT) as archive:
        for name, path in sorted(entries.items()):
            if path.is_symlink():
                raise ValueError(f"Refusing symlink in bundle: {name}")
            archive.add(path, arcname=name, recursive=False, filter=public_tar_metadata)
        member = tarfile.TarInfo("bundle_manifest.json")
        member.size = len(info_bytes)
        archive.addfile(member, io.BytesIO(info_bytes))
    with tarfile.open(destination, "r:gz") as archive:
        names = archive.getnames()
        if set(names) != set(entries) | {"bundle_manifest.json"} or len(names) != len(set(names)):
            raise ValueError("Archive member mismatch")
        for name, path in entries.items():
            if archive.extractfile(name).read() != path.read_bytes():
                raise ValueError(f"Archive content mismatch: {name}")
        if archive.extractfile("bundle_manifest.json").read() != info_bytes:
            raise ValueError("Bundle manifest mismatch")
    checksum_path.write_text(f"{file_hash(destination)}  {destination.name}\n", encoding="ascii", newline="\n")
    print(f"Verified {len(entries) + 1} files; {destination.resolve()}")
    return info


def verify_bundle(directory):
    directory = Path(directory).resolve()
    info = json.loads((directory / "bundle_manifest.json").read_text(encoding="utf-8"))
    for name, expected in info["files_sha256"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or not path.is_file() or file_hash(path) != expected:
            raise ValueError(f"Bundle file missing or modified: {name}")
    print(f"Bundle verified: {len(info['files_sha256'])} file hashes")
    return info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist/misleadqa-v1.0.0-a40.tar.gz")
    args = parser.parse_args()
    build_package(args.output)
