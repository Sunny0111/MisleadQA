"""Context-faithful prompts with the original closed-book baseline preserved."""
import copy
from pathlib import Path

from experiments.main_eval import protocol as base
from experiments.main_eval.backend import verify_generation
from misleadqa_utils import CONFIG_DIR

HERE = Path(__file__).resolve().parent
VERSION = "misleadqa_robustness_v1"
VARIANTS = ("context-faithful",)
SELECTION_SEED = 2027


def settings_for(variant="context-faithful", seed=2027, budget=2, *, parser=None):
    if variant not in VARIANTS or seed != 2027 or budget != 2:
        raise ValueError("Context-faithful requires seed 2027 and a two-token answer budget")
    parser = parser or "canonical-choice-then-eos"
    if parser != "canonical-choice-then-eos":
        raise ValueError("Unsupported answer parser")
    # Retain the saved CF identity fields so existing sealed runs remain verifiable.
    return {"variant": variant, "seed": seed, "selection_seed": SELECTION_SEED,
            "dtype": "bfloat16", "batch_size": 1, "max_input_tokens": 4096,
            "max_new_tokens": 2, "sampling": None, "parser": parser,
            "request_seed_scheme": "sha256-v1-31bit-item-condition"}


def model_for(alias, variant="context-faithful"):
    if variant not in VARIANTS:
        raise ValueError("Unsupported experiment variant")
    return copy.deepcopy(base.load_models()[alias])


def make_requests(items, settings):
    if settings != settings_for():
        raise ValueError("Unexpected context-faithful settings")
    if any(item["answer_key"]["abstain"] != "E" for item in items):
        raise ValueError("The context-faithful prompt requires abstain=E")
    template = (CONFIG_DIR / "prompts/context_faithful_v1.txt").read_text(encoding="utf-8")
    original = base.make_requests(items)
    contextual = base.make_requests(items, template)
    requests = [a if a["condition_id"] == "closed_book" else b
                for a, b in zip(original, contextual)]
    for row in requests:
        row["request_id"] = base.digest(base.canonical(
            [VERSION, settings["variant"], settings["seed"], row["item_id"], row["condition_id"]]))
        row["generation_seed"] = settings["seed"]
    return requests


def source_hashes():
    paths = [p for p in HERE.rglob("*") if p.is_file() and p.suffix in (".py", ".txt", ".sh", ".md")
             and not {"outputs", "analysis", "__pycache__"}.intersection(p.relative_to(HERE).parts)]
    return base.source_hashes() | {p.relative_to(base.ROOT).as_posix(): base.file_hash(p) for p in sorted(paths)}


def validate_result(row, metadata, settings):
    tokens, inputs = row["generated_token_ids"], row["input_token_ids"]
    if (any(type(t) is not int for t in tokens + inputs)
            or not 0 < len(inputs) <= settings["max_input_tokens"]):
        raise ValueError("Invalid input tokens or context length")
    verify_generation(row, metadata)
