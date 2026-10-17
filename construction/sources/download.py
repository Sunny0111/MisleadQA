"""Inspect the source manifest or fetch one archive with checksum verification."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import urllib.request

from misleadqa_utils import ROOT, archive_path, load_config


def fetch(entry):
    urls = entry.get("source_urls", [])
    hashes = entry.get("expected_sha256", [])
    if len(urls) != 1 or len(hashes) != 1 or not re.fullmatch(r"[0-9a-f]{64}", hashes[0]):
        raise ValueError("A single archive URL and expected SHA-256 are required")
    if not urls[0].startswith("https://"):
        raise ValueError("Archive URL must use HTTPS")
    if not entry.get("license_verified") or not entry.get("license"):
        raise ValueError("Complete license and license_verified in the manifest before fetching")
    destination = archive_path(entry["archive_file"])
    if destination.exists():
        raise ValueError("Archive already exists; refusing to overwrite it")
    temporary = destination.with_name(destination.name + ".part")
    if temporary.exists():
        raise ValueError("Partial file already exists; inspect it before retrying")
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    created = False
    try:
        request = urllib.request.Request(urls[0], headers={"User-Agent": "MisleadQA-source-fetcher"})
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("xb") as stream:
            created = True
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                digest.update(chunk)
                stream.write(chunk)
        if digest.hexdigest() != hashes[0]:
            raise ValueError("Downloaded bytes differ from the expected historical archive")
        # An exclusive destination write also protects against concurrent creation.
        with temporary.open("rb") as source, destination.open("xb") as target:
            import shutil
            shutil.copyfileobj(source, target)
    finally:
        if created and temporary.exists():
            temporary.unlink()
    log = {"archive_file": entry["archive_file"], "source_url": urls[0],
           "sha256": digest.hexdigest(), "retrieved_at_utc": datetime.now(timezone.utc).isoformat()}
    with (ROOT / "construction/sources/retrieval_log.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(log) + "\n")
    return log


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "construction/sources/source_manifest.json")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--list", action="store_true", help="Show an offline inventory")
    mode.add_argument("--fetch", metavar="ARCHIVE_PATH", help="Fetch one manifest entry")
    args = parser.parse_args()
    manifest = load_config(args.manifest)
    entries = manifest["archives"]
    if args.list:
        print(json.dumps({"archives": len(entries), "source_families": dict(Counter(
            source for entry in entries for source in entry["source_families"])),
            "present": sum(archive_path(entry["archive_file"]).is_file() for entry in entries),
            "license_verified": sum(bool(entry.get("license_verified")) for entry in entries),
            "manifest": str(args.manifest)}, indent=2))
        return 0
    matches = [entry for entry in entries if entry["archive_file"] == args.fetch]
    if len(matches) != 1:
        parser.error("Archive path must identify exactly one manifest entry")
    try:
        print(json.dumps(fetch(matches[0]), indent=2))
    except (OSError, ValueError) as error:
        parser.exit(1, f"Download failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

