"""Add this diagnostic to the verified baseline bundle, without outputs/weights."""
import argparse
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "experiments.scope_diagnostic"

from experiments.main_eval.package import build_package as build_base
from experiments.main_eval.protocol import ROOT, canonical, digest, file_hash
from .protocol import HERE, LABELS, SEED, VERSION, sample_items


def build_package(destination):
    destination = Path(destination)
    checksum = destination.with_name(destination.name + ".sha256")
    if destination.exists() or checksum.exists():
        raise ValueError("Package/checksum exists; choose a new destination")
    _, sample = sample_items()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temp:
        base = Path(temp) / "base.tar.gz"
        build_base(base)
        with tarfile.open(base, "r:gz") as archive:
            contents = {m.name: archive.extractfile(m).read() for m in archive.getmembers()}
        info = json.loads(contents.pop("bundle_manifest.json"))
        for path in HERE.rglob("*"):
            if (path.is_file() and path.suffix in (".py", ".txt", ".json", ".sh", ".md")
                    and not {"outputs", "analysis", "__pycache__"}.intersection(path.relative_to(HERE).parts)):
                if path.is_symlink():
                    raise ValueError(f"Refusing symlink: {path}")
                contents[path.relative_to(ROOT).as_posix()] = path.read_bytes()
        info["scope_diagnostic"] = {"protocol": VERSION, "sample_sha256": digest(canonical(sample)),
            "seed": SEED, "items": 200, "items_per_group": 40, "labels": LABELS,
            "requests_per_model": 600, "requests_six_models": 3600,
            "reuses": ["original closed_book", "original s_only"]}
        info["files_sha256"] = {name: digest(data) for name, data in sorted(contents.items())}
        contents["bundle_manifest.json"] = (json.dumps(info, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        with tarfile.open(destination, "x:gz", format=tarfile.PAX_FORMAT) as archive:
            for name, data in sorted(contents.items()):
                member = tarfile.TarInfo(name)
                member.size, member.mode = len(data), 0o644
                archive.addfile(member, io.BytesIO(data))
        with tarfile.open(destination, "r:gz") as archive:
            if archive.getnames() != sorted(contents) or any(
                    archive.extractfile(name).read() != data for name, data in contents.items()):
                raise ValueError("Archive content mismatch")
    checksum.write_text(f"{file_hash(destination)}  {destination.name}\n", encoding="ascii", newline="\n")
    print(f"Verified scope diagnostic bundle: {destination.resolve()}")
    return info


def verify_bundle(directory):
    directory = Path(directory).resolve()
    info = json.loads((directory / "bundle_manifest.json").read_text(encoding="utf-8"))
    if info.get("scope_diagnostic", {}).get("protocol") != VERSION:
        raise ValueError("Not a scope diagnostic bundle")
    for name, expected in info["files_sha256"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or not path.is_file() or file_hash(path) != expected:
            raise ValueError(f"Bundle file missing or modified: {name}")
    print(f"Bundle verified: {len(info['files_sha256'])} file hashes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist/misleadqa-v1.0.0-scope-v2.tar.gz")
    parser.add_argument("--verify", type=Path, metavar="EXTRACTED_DIRECTORY")
    args = parser.parse_args()
    verify_bundle(args.verify) if args.verify else build_package(args.output)
