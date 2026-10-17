"""Check structural consistency; factual and human review are separate tasks."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import unicodedata

from misleadqa_utils import (CURRENT_SCHEMA, CURRENT_VERSION, DEFAULT_DATASET, RELEASE_DIR,
                            archive_path, archive_refs, read_jsonl, review_flag,
                            sha256_file, write_json)

EVIDENCE_FLAGS = {
    "correct": (True, True),
    "false": (False, True),
    "mis_scoped": (True, False),
}
ROLES = {"gold", "lure", "distractor_1", "distractor_2", "abstain"}


def normalized_answer(text):
    """Match stored aliases and unambiguous same-century abbreviated year ranges."""
    text = unicodedata.normalize("NFKC", text).strip().casefold()
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    match = re.fullmatch(r"(\d{4})-(\d{2})", text)
    if match and int(match[2]) >= int(match[1][-2:]):
        text = f"{match[1]}-{match[1][:2]}{match[2]}"
    return text


def matches_alias(text, item, role):
    return normalized_answer(text) in {
        normalized_answer(alias) for alias in item[f"{role}_aliases"]
    }


def check_item(item):
    errors = []

    def require(condition, message):
        if not condition:
            errors.append(message)

    try:
        current = (item.get("schema_version"), item.get("version")) == (CURRENT_SCHEMA, CURRENT_VERSION)
        legacy = (item.get("schema_version"), item.get("version")) == ("1.0.0", "0.37.0")
        require(current or legacy, "unsupported schema/dataset version pair")
        for field in ("schema_version", "version", "id", "source", "dimension",
                      "question", "base_question", "gold_answer", "lure_answer"):
            require(isinstance(item[field], str) and bool(item[field].strip()),
                    f"{field} must be a nonempty string")
        require(item["gold_answer"] != item["lure_answer"], "gold equals lure")
        require(set(item["answer_key"]) == {"gold", "lure", "abstain"}, "unexpected answer key fields")
        require(item["target_scope"] != item["alternate_scope"], "scopes are identical")
        labels = [option["label"] for option in item["options"]]
        require(labels == list("ABCDE"), "options must be ordered A through E")
        texts = {option["label"]: option["text"] for option in item["options"]}
        require(all(isinstance(text, str) and text.strip() for text in texts.values()),
                "option text must be nonempty")
        require(len(set(texts.values())) == 5, "option texts are not distinct")
        roles = item["option_roles"]
        require(set(roles) == set(labels), "role labels differ from option labels")
        require(set(roles.values()) == ROLES, "expected five distinct option roles")
        for role in ("gold", "lure", "abstain"):
            label = item["answer_key"][role]
            require(roles.get(label) == role, f"answer key mismatch: {role}")
            if role != "abstain":
                require(matches_alias(texts[label], item, role),
                        f"option text is absent from normalized {role} aliases")
        for role in ("gold", "lure"):
            aliases = item[f"{role}_aliases"]
            require(isinstance(aliases, list) and bool(aliases), f"missing {role} aliases")
            require(item[f"{role}_answer"] in aliases, f"display answer absent from {role} aliases")
        require(len(item["distractors"]) == 2, "expected two distractors")
        for index, distractor in enumerate(item["distractors"], 1):
            label = distractor["option_label"]
            require(roles.get(label) == f"distractor_{index}", "distractor role mismatch")
            require(texts.get(label) == distractor["display_value"], "distractor text mismatch")
        for key, flags in EVIDENCE_FLAGS.items():
            evidence = item["evidence"][key]
            text = evidence["text"]
            require(isinstance(text, str) and bool(text.strip()), f"empty {key} evidence")
            require(text == text.strip("\r\n"), f"boundary newlines in {key} evidence")
            require(hashlib.sha256(text.encode("utf-8")).hexdigest() == evidence["sha256"],
                    f"evidence hash mismatch: {key}")
            require(evidence["intended_factuality"] is flags[0], f"factuality flag mismatch: {key}")
            require(evidence["intended_applicability"] is flags[1], f"applicability flag mismatch: {key}")
            expected_scope = item["alternate_scope" if key == "mis_scoped" else "target_scope"]
            require(evidence["scope"] == expected_scope, f"scope mismatch: {key}")
            role = "gold" if key == "correct" else "lure"
            require(matches_alias(evidence["answer"], item, role), f"evidence answer mismatch: {key}")
        original = item["evidence"]["correct"]["text"]
        result = original
        edits = sorted(item["construction"]["false_edits"], key=lambda edit: edit["start"], reverse=True)
        boundary = len(original)
        for edit in edits:
            start, end = edit["start"], edit["end"]
            require(type(start) is int and type(end) is int, "false-edit offsets must be integers")
            require(0 <= start < end <= boundary, "invalid or overlapping false-edit offsets")
            require(original[start:end] == edit["before"], "false-edit before text mismatch")
            result = result[:start] + edit["after"] + result[end:]
            boundary = start
        require(result == item["evidence"]["false"]["text"], "false edits do not reproduce F")
        if "review" in item or legacy:
            for field in ("human_reviewed", "evaluation_eligible"):
                require(type(item["review"][field]) is bool, f"review.{field} must be boolean")
        for field in ("analysis_group_id", "cluster_id"):
            require(isinstance(item["grouping"][field], str) and bool(item["grouping"][field]),
                    f"missing grouping.{field}")
    except (KeyError, TypeError, ValueError, AttributeError, IndexError) as error:
        errors.append(f"Malformed record: {error}")
    return errors


def validate_dataset(path, require_raw=False):
    """Validate release records; inspect local source files only when requested."""
    rows = read_jsonl(path)
    errors, groups, expected_archives = [], defaultdict(list), defaultdict(set)
    ids = Counter(item.get("id", "") for item in rows)
    for item_id, count in ids.items():
        if count > 1:
            errors.append({"id": item_id, "error": f"Duplicate ID ({count} records)"})
    for number, item in enumerate(rows, 1):
        errors.extend({"id": item.get("id"), "line": number, "error": error}
                      for error in check_item(item))
        group = item.get("grouping", {}).get("analysis_group_id")
        if group:
            groups[group].append(item)
        try:
            for relative, digest in archive_refs(item):
                expected_archives[relative].add(digest)
        except (TypeError, AttributeError) as error:
            errors.append({"id": item.get("id"), "error": f"Malformed source references: {error}"})
    for group, members in groups.items():
        known_eligibility = all(type(review_flag(item, "evaluation_eligible")) is bool for item in members)
        eligible = sum(review_flag(item, "evaluation_eligible") is True for item in members)
        for item in members:
            metadata = item["grouping"]
            if (metadata.get("analysis_group_size") != len(members) or
                    (known_eligibility and metadata.get("analysis_group_size_evaluation") != eligible)):
                errors.append({"id": item.get("id"), "error": f"Group size mismatch: {group}"})
    missing = []
    for relative, hashes in sorted(expected_archives.items()):
        try:
            archive = archive_path(relative)
            if len(hashes) != 1 or "" in hashes:
                errors.append({"error": f"Missing or conflicting archive hashes: {relative}"})
            if not require_raw:
                continue
            if not archive.is_file():
                missing.append(relative)
            elif sha256_file(archive) not in hashes:
                errors.append({"error": f"Archive checksum mismatch: {relative}"})
        except ValueError as error:
            errors.append({"error": str(error)})
    if require_raw and missing:
        errors.append({"error": f"{len(missing)} referenced archives are missing"})
    if Path(path).resolve().parent == RELEASE_DIR.resolve():
        checksum_path = RELEASE_DIR / "hashes.txt"
        if not checksum_path.is_file():
            errors.append({"error": "Release checksum file is missing"})
        else:
            entries = [line.split() for line in checksum_path.read_text().splitlines() if line.strip()]
            expected = [entry[0] for entry in entries if len(entry) == 2 and entry[1] == Path(path).name]
            if expected != [sha256_file(path)]:
                errors.append({"error": "Release checksum mismatch"})
    report = {
        "status": "pass" if not errors else "fail",
        "dataset_sha256": sha256_file(path),
        "records": len(rows),
        "composition": dict(Counter(f"{item.get('source')} / {item.get('dimension')}" for item in rows)),
        "versions": dict(Counter(item.get("version") for item in rows)),
        "schema_versions": dict(Counter(item.get("schema_version") for item in rows)),
        "human_reviewed": (sum(review_flag(item, "human_reviewed") is True for item in rows)
                           if all(type(review_flag(item, "human_reviewed")) is bool for item in rows) else None),
        "inherited_evaluation_eligible": (sum(review_flag(item, "evaluation_eligible") is True for item in rows)
                           if all(type(review_flag(item, "evaluation_eligible")) is bool for item in rows) else None),
        "review_metadata_unknown": sum("review" not in item for item in rows),
        "evaluation_group_sizes_checked": all(type(review_flag(item, "evaluation_eligible")) is bool for item in rows),
        "analysis_groups": len(groups),
        "position_counts": {role: dict(Counter(item.get("answer_key", {}).get(role) for item in rows))
                            for role in ("gold", "lure", "abstain")},
        "referenced_archives": len(expected_archives),
        "errors": errors,
        "limitations": ["Structural checks do not establish factuality, applicability, or human approval."],
    }
    if require_raw:
        report["missing_archives"] = missing
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--require-raw", action="store_true",
                        help="Require every referenced source archive and verify its checksum")
    args = parser.parse_args()
    try:
        report = validate_dataset(args.dataset, args.require_raw)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Validation failed: {error}\n")
    if args.report:
        write_json(args.report, report)
    summary = {key: value for key, value in report.items() if key not in ("missing_archives", "errors")}
    summary["error_count"] = len(report["errors"])
    summary["first_errors"] = report["errors"][:10]
    if args.require_raw:
        summary["missing_archive_count"] = len(report["missing_archives"])
    print(json.dumps(summary, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

