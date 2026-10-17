"""Two cyclic A-D rotations, fixed E, and verified original-answer references."""
from copy import deepcopy
import json
from pathlib import Path

from experiments.main_eval.protocol import (
    CONDITION_IDS, DEFAULT_DATASET, PROTOCOL_VERSION, ROOT, canonical, digest,
    file_hash, load_models, make_requests as qa_requests, source_hashes as qa_hashes,
)
from experiments.main_eval.run import validate_checkpoint
from experiments.scope_diagnostic.protocol import SETTINGS, sample_items
from misleadqa_utils import CURRENT_VERSION, OUTPUTS_DIR

HERE = Path(__file__).resolve().parent
VERSION = "misleadqa_option_order_v1"
REFERENCE_ROOT = OUTPUTS_DIR / "option_order_references"
CONDITIONS = ("closed_book", "s_only")
ORDERS = ("original", "rotate1", "rotate2")


def rotate_item(item, shift):
    """Left-rotate A-D option texts; keep E and all evidence unchanged."""
    if shift not in (0, 1, 2):
        raise ValueError("Only the original order and rotations 1/2 are defined")
    rotated = deepcopy(item)
    old_labels = "ABCD"[shift:] + "ABCD"[:shift] + "E"
    mapping = dict(zip("ABCDE", old_labels))
    options = {option["label"]: option["text"] for option in item["options"]}
    rotated["options"] = [{"label": label, "text": options[old]} for label, old in mapping.items()]
    rotated["option_roles"] = {label: item["option_roles"][old] for label, old in mapping.items()}
    inverse = {old: label for label, old in mapping.items()}
    rotated["answer_key"] = {role: inverse[label] for role, label in item["answer_key"].items()}
    return rotated, mapping


def make_requests(items, shifts=(1, 2), conditions=CONDITIONS):
    if tuple(conditions) not in (CONDITIONS, ("f_only",)):
        raise ValueError("Use the original conditions or the F-only extension")
    requests = []
    for item in items:
        for shift in shifts:
            rotated, mapping = rotate_item(item, shift)
            for request in qa_requests([rotated]):
                if request["condition_id"] not in conditions:
                    continue
                request.update(
                    permutation_id=ORDERS[shift], original_option_labels=mapping,
                    request_id=digest(canonical([VERSION, item["id"], shift, request["condition_id"]])),
                )
                requests.append(request)
    return requests


def validate_reference(reference, dataset=DEFAULT_DATASET):
    """Validate an exported subset against the original QA prompts and model policy."""
    items, sample = sample_items(dataset)
    identity = reference["baseline_identity"]
    spec = load_models()[reference["model_alias"]]
    if (reference["protocol"] != VERSION or reference["dataset_version"] != CURRENT_VERSION
            or reference["sample_sha256"] != digest(canonical(sample))
            or identity["protocol"] != PROTOCOL_VERSION or identity["dry_run"]
            or identity["model_alias"] != reference["model_alias"] or identity["model"] != spec
            or identity["dataset_sha256"] != file_hash(dataset)
            or identity["selection"] != "all" or identity["limit"] is not None
            or identity["counts"]["selected_items"] != 959
            or tuple(identity["conditions"]) != CONDITION_IDS
            or identity["backend"] != spec.get("required_backend", "transformers")
            or any(identity["settings"].get(k) != v for k, v in SETTINGS.items())):
        raise ValueError("Reference must reuse the original full main run and frozen sample")
    expected = [r for r in qa_requests(items) if r["condition_id"] in CONDITIONS]
    metadata = reference["model_metadata"]
    generation = metadata["generation_config"]
    if (metadata["model_id"] != spec["model_id"] or metadata["revision"] != spec["revision"]
            or metadata["chat_template_kwargs"] != spec["chat_template_kwargs"]
            or metadata.get("backend", "transformers") != identity["backend"]
            or metadata["dtype"] != "bfloat16" or metadata["quantization"] != spec.get("weight_quantization")
            or generation["do_sample"] is not False or generation["max_new_tokens"] != 2
            or generation["num_beams"] != 1 or generation["num_return_sequences"] != 1):
        raise ValueError("Reference model metadata differs from its pinned checkpoint")
    if len(validate_checkpoint(reference["responses"], expected, metadata)) != len(expected):
        raise ValueError("Reference does not contain all 400 original sample answers")
    return items, sample


def load_reference(path, dataset=DEFAULT_DATASET):
    reference = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_reference(reference, dataset)
    return reference


def make_plan(reference, dataset=DEFAULT_DATASET, smoke=False, rerun_original=False, conditions=CONDITIONS):
    if tuple(conditions) == ("f_only",) and not rerun_original:
        raise ValueError("F-only requires all three option orders")
    items, sample = sample_items(dataset, smoke)
    shifts = (0, 1, 2) if rerun_original else (1, 2)
    requests = make_requests(items, shifts, conditions)
    baseline = reference["baseline_identity"]
    identity = {
        "protocol": VERSION, "dataset_version": CURRENT_VERSION,
        "dataset_sha256": file_hash(dataset), "sample_sha256": digest(canonical(sample)),
        "reference_sha256": digest(canonical(reference)), "model_alias": reference["model_alias"],
        "model": baseline["model"], "settings": baseline["settings"],
        "backend": baseline["backend"], "engine": baseline.get("engine"),
        "smoke": smoke, "selected_item_ids": [item["id"] for item in items],
        "permutations": [ORDERS[shift] for shift in shifts], "conditions": list(conditions),
        "expected_responses": len(requests), "requests_sha256": digest(canonical(requests)),
    }
    if rerun_original:
        identity["rerun_original"] = True
    return requests, identity


def source_hashes():
    paths = [p for p in HERE.rglob("*") if p.is_file() and p.suffix in (".py", ".md", ".sh")]
    paths += [ROOT / "experiments/scope_diagnostic" / name for name in ("protocol.py", "sample.json")]
    return qa_hashes() | {p.relative_to(ROOT).as_posix(): file_hash(p) for p in sorted(paths)}
