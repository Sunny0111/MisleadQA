"""Bundle frozen baseline code plus robustness extensions, excluding all results and weights."""
import argparse
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    __package__ = "experiments.main_eval.supplementary"

from experiments.main_eval.package import build_package as build_base
from experiments.main_eval.protocol import ROOT, digest, file_hash
from .protocol import HERE, VERSION


def build_package(destination):
    destination = Path(destination)
    checksum = destination.with_name(destination.name + ".sha256")
    if destination.exists() or checksum.exists():
        raise ValueError("Package/checksum already exists; choose a new destination")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temp:
        base_path = Path(temp) / "base.tar.gz"
        build_base(base_path)  # Validates the unchanged first-round manifest and release data.
        with tarfile.open(base_path, "r:gz") as archive:
            contents = {m.name: archive.extractfile(m).read() for m in archive.getmembers()}
        info = json.loads(contents.pop("bundle_manifest.json"))
        for path in sorted(HERE.rglob("*")):
            if (path.is_file() and path.suffix in (".py", ".txt", ".sh", ".md")
                    and not {"outputs", "analysis", "__pycache__"}.intersection(path.relative_to(HERE).parts)):
                if path.is_symlink():
                    raise ValueError(f"Refusing symlink: {path}")
                contents[path.relative_to(ROOT).as_posix()] = path.read_bytes()
        count = info["splits"]["main"]["selected_items"] * 8
        info["robustness"] = {"protocol": VERSION, "default_split": "main", "variant": "context-faithful",
                              "context_responses": 6 * count, "total_additional_responses": 6 * count}
        info["files_sha256"] = {name: digest(data) for name, data in sorted(contents.items())}
        contents["bundle_manifest.json"] = (json.dumps(info, indent=2) + "\n").encode()
        with tarfile.open(destination, "x:gz", format=tarfile.PAX_FORMAT) as archive:
            for name, data in sorted(contents.items()):
                member = tarfile.TarInfo(name)
                member.size, member.mode = len(data), 0o644
                archive.addfile(member, io.BytesIO(data))
        with tarfile.open(destination, "r:gz") as archive:
            if archive.getnames() != sorted(contents):
                raise ValueError("Archive member mismatch")
            for name, data in contents.items():
                if archive.extractfile(name).read() != data:
                    raise ValueError(f"Archive content mismatch: {name}")
    checksum.write_text(f"{file_hash(destination)}  {destination.name}\n", encoding="ascii", newline="\n")
    print(f"Verified robustness bundle: {len(contents)} files; {destination.resolve()}")
    return info


def verify_bundle(directory):
    directory = Path(directory).resolve()
    info = json.loads((directory / "bundle_manifest.json").read_text(encoding="utf-8"))
    if info.get("robustness", {}).get("protocol") != VERSION:
        raise ValueError("Not a robustness bundle")
    for name, expected in info["files_sha256"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or not path.is_file() or file_hash(path) != expected:
            raise ValueError(f"Bundle file missing or modified: {name}")
    print(f"Bundle verified: {len(info['files_sha256'])} file hashes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist/misleadqa-v1.0.0-context-faithful.tar.gz")
    parser.add_argument("--verify", type=Path, metavar="EXTRACTED_DIRECTORY")
    args = parser.parse_args()
    verify_bundle(args.verify) if args.verify is not None else build_package(args.output)
