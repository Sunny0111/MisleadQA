"""Eight independent conditions for release 1.0.0; stdlib-only preparation."""
import hashlib
import json
from pathlib import Path
import random
from string import Template

HERE = Path(__file__).resolve().parent
from misleadqa_utils import (CONFIG_DIR, OUTPUTS_DIR, ROOT, CURRENT_SCHEMA,
                            CURRENT_VERSION, DEFAULT_DATASET, review_flag)
from construction.screening.checks import check_item, validate_dataset
CONDITIONS = (
    ("closed_book", ()),
    ("g_only", ("correct",)),
    ("f_only", ("false",)),
    ("s_only", ("mis_scoped",)),
    ("g_then_f", ("correct", "false")),
    ("f_then_g", ("false", "correct")),
    ("g_then_s", ("correct", "mis_scoped")),
    ("s_then_g", ("mis_scoped", "correct")),
)
CONDITION_IDS = tuple(c[0] for c in CONDITIONS)
LETTERS = "ABCDE"
PROTOCOL_VERSION = "release_v100_eight_abcde_v1"
# Preserve saved request IDs when normalizing the dataset's administrative label.
# This is an opaque historical identity namespace, not the dataset release version.
REQUEST_ID_NAMESPACE = "release_v101_eight_abcde_v1"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"Blank JSONL line {number}: {path}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL line {number} is not an object")
            rows.append(value)
    return rows


def save_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def validate_item(row):
    item_id = row.get("id")
    if not isinstance(item_id, str) or not item_id:
        raise ValueError("Missing item id")
    if row.get("version") != CURRENT_VERSION or row.get("schema_version") != CURRENT_SCHEMA:
        raise ValueError(f"Unexpected schema/dataset version: {item_id}")
    errors = check_item(row)
    if errors:
        raise ValueError(f"Invalid item {item_id}: {'; '.join(errors)}")
    if [o["label"] for o in row["options"]] != list(LETTERS):
        raise ValueError(f"Expected ordered A/B/C/D/E options: {item_id}")
    fields = [row["question"]] + [o["text"] for o in row["options"]]
    if any(not isinstance(v, str) or not v.strip() for v in fields):
        raise ValueError(f"Empty question or option: {item_id}")
    keys = row["answer_key"]
    if set(keys) != {"gold", "lure", "abstain"} or len(set(keys.values())) != 3:
        raise ValueError(f"Invalid answer key: {item_id}")
    for role, label in keys.items():
        if label not in LETTERS or row["option_roles"].get(label) != role:
            raise ValueError(f"Inconsistent option role: {item_id}")
    # Display options can use a different date/range spelling from canonical answers.
    # This is letter selection: answer_key + option_roles are the scoring authority.
    expected = {"correct": (True, True, row["gold_answer"]),
                "false": (False, True, row["lure_answer"]),
                "mis_scoped": (True, False, row["lure_answer"])}
    for key, (factuality, applicability, answer) in expected.items():
        evidence = row["evidence"][key]
        text = evidence["text"]
        if not isinstance(text, str) or not text.strip() or text != text.strip("\r\n"):
            raise ValueError(f"Empty evidence or boundary newlines: {item_id}/{key}")
        if evidence.get("sha256") != digest(text.encode("utf-8")):
            raise ValueError(f"Evidence hash mismatch: {item_id}/{key}")
        role = "gold" if key == "correct" else "lure"
        acceptable_answers = [answer, *row.get(role + "_aliases", [])]
        if (evidence["intended_factuality"] is not factuality
                or evidence["intended_applicability"] is not applicability
                or evidence["answer"] not in acceptable_answers):
            raise ValueError(f"Inconsistent evidence metadata: {item_id}/{key}")


def load_items(path=DEFAULT_DATASET, selection="all", limit=None, seed=2027):
    rows = read_jsonl(path)
    if not rows:
        raise ValueError("Empty dataset")
    if selection not in ("eligible", "all"):
        raise ValueError("Selection must be eligible or all")
    seen = set()
    for row in rows:
        validate_item(row)
        if row["id"] in seen:
            raise ValueError(f"Duplicate item id: {row['id']}")
        seen.add(row["id"])
    report = validate_dataset(path)
    if report["status"] != "pass":
        raise ValueError(f"Dataset validation failed: {report['errors'][:3]}")
    eligibility = [review_flag(r, "evaluation_eligible") for r in rows]
    unknown = sum(value is None for value in eligibility)
    if selection == "eligible" and unknown:
        raise ValueError("Eligibility metadata is unavailable in this release; use --selection all")
    items = sorted((r for r in rows if selection == "all" or review_flag(r, "evaluation_eligible") is True), key=lambda r: r["id"])
    eligible_count = None if unknown else sum(eligibility)
    if limit is not None:
        if not 0 < limit <= len(items):
            raise ValueError(f"Limit must be between 1 and {len(items)}")
        items = sorted(random.Random(seed).sample(items, limit), key=lambda r: r["id"])
    if not items:
        raise ValueError("Selection contains no items")
    return items, {"dataset_items": len(rows), "eligible_items": eligible_count,
                   "ineligible_items": None if unknown else len(rows) - eligible_count,
                   "review_unknown_items": unknown, "selected_items": len(items)}


def make_requests(items, template=None):
    if template is None:
        template = (CONFIG_DIR / "prompts/qa_v1.txt").read_text(encoding="utf-8")
    requests = []
    for item in items:
        for number, (condition, keys) in enumerate(CONDITIONS, 1):
            evidence = "\n\n".join(item["evidence"][key]["text"] for key in keys)
            prompt = Template(template).substitute(
                question=item["question"],
                options="\n".join(f"{o['label']}. {o['text']}" for o in item["options"]),
                evidence_block="\n\n" + evidence if keys else "")
            grouping = item.get("grouping", {})
            request = {
                "request_id": digest(canonical([REQUEST_ID_NAMESPACE, item["id"], condition])),
                "item_id": item["id"], "condition_id": condition, "condition_number": number,
                "source": item["source"], "dimension": item["dimension"],
                "evaluation_eligible": review_flag(item, "evaluation_eligible"),
                "analysis_group_id": grouping["analysis_group_id"],
                "answer_key": item["answer_key"],
                "messages": [{"role": "user", "content": prompt}],
            }
            request["prompt_sha256"] = digest(canonical(request["messages"]))
            requests.append(request)
    return requests


def source_hashes():
    directories = (HERE, CONFIG_DIR, ROOT / "experiments/analysis")
    paths = sorted(p for directory in directories for p in directory.rglob("*") if p.is_file()
                   and p.suffix in (".py", ".json", ".txt", ".sh", ".md")
                   and p.name != "manifest.json"
                   and not ({"outputs", "__pycache__"} & set(p.relative_to(ROOT).parts)))
    paths += [ROOT / "misleadqa_utils.py", ROOT / "construction/screening/checks.py"]
    return {p.relative_to(ROOT).as_posix(): file_hash(p) for p in sorted(paths)}


def load_models():
    models = json.loads((CONFIG_DIR / "models.json").read_text(encoding="utf-8"))
    import re
    for spec in models.values():
        if not re.fullmatch(r"[0-9a-f]{40}", spec["revision"]):
            raise ValueError("Model revision must be a full commit SHA")
    return models
