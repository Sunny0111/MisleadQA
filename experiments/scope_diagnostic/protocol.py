"""Frozen balanced sampling and three independent, label-only scope requests."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
from string import Template

from experiments.main_eval.protocol import (DEFAULT_DATASET, ROOT, canonical, digest, file_hash,
                                 load_items, source_hashes as baseline_hashes)

HERE = Path(__file__).resolve().parent
from misleadqa_utils import CONFIG_DIR, CURRENT_VERSION
VERSION = "misleadqa_scope_diagnostic_v2"
SEED = 2027
LABELS = ("Applicable", "Not applicable", "Cannot determine")
EVIDENCE = {"G": "correct", "F": "false", "S": "mis_scoped"}
GROUPS = (("AmbigDocs", "entity", 192), ("FRED", "geography", 200),
          ("FRED", "time", 200), ("SituatedQA", "geography", 184),
          ("SituatedQA", "time", 183))
SETTINGS = {"seed": SEED, "max_input_tokens": 4096, "dtype": "bfloat16",
            "batch_size": 1, "max_new_tokens": 2}
# max_new_tokens=2 is the original loader policy. Actual label budget is resolved
# from each tokenizer, including EOS, and retained in model.json.


def select_sample(dataset=DEFAULT_DATASET):
    items, _ = load_items(dataset)
    populations = defaultdict(list)
    for item in items:
        populations[item["source"], item["dimension"]].append(item)
    if Counter({key: len(value) for key, value in populations.items()}) != Counter(
            {(source, dimension): size for source, dimension, size in GROUPS}):
        raise ValueError("Scope diagnosis requires the 959-item main split and its five groups")
    rng = random.Random(SEED)
    selected = sorted((item for source, dimension, _ in GROUPS
                       for item in rng.sample(populations[source, dimension], 40)), key=lambda i: i["id"])
    sample = {"protocol": VERSION, "dataset_version": CURRENT_VERSION,
              "dataset_sha256": file_hash(dataset), "seed": SEED,
              "sampling": "sorted group and item order; Python random.Random.sample; without replacement",
              "items_per_group": 40, "selection_uses_model_responses": False,
              "selected": [{"item_id": i["id"], "source": i["source"], "dimension": i["dimension"],
                            "cluster_id": i["grouping"]["cluster_id"]} for i in selected]}
    return selected, sample


def sample_items(dataset=DEFAULT_DATASET, smoke=False):
    selected, expected = select_sample(dataset)
    frozen = json.loads((HERE / "sample.json").read_text(encoding="utf-8"))
    if canonical(frozen) != canonical(expected):
        raise ValueError("Frozen sample differs from the dataset or selection protocol")
    if smoke:
        selected = [next(i for i in selected if (i["source"], i["dimension"]) == (s, d))
                    for s, d, _ in GROUPS]
        selected.sort(key=lambda i: i["id"])
    return selected, frozen


def make_requests(items):
    template = Template((CONFIG_DIR / "prompts/scope_v1.txt").read_text(encoding="utf-8"))
    requests = []
    for item in items:
        for condition, key in EVIDENCE.items():
            prompt = template.substitute(question=item["question"],
                options="\n".join(f"{o['label']}. {o['text']}" for o in item["options"]),
                evidence=item["evidence"][key]["text"])
            messages = [{"role": "user", "content": prompt}]
            requests.append({"request_id": digest(canonical([VERSION, item["id"], condition])),
                "item_id": item["id"], "condition_id": condition,
                "messages": messages, "prompt_sha256": digest(canonical(messages))})
    return requests


def source_hashes():
    paths = [p for p in HERE.rglob("*") if p.is_file() and p.suffix in (".py", ".txt", ".json", ".sh", ".md")
             and not {"outputs", "analysis", "__pycache__"}.intersection(p.relative_to(HERE).parts)]
    return baseline_hashes() | {p.relative_to(ROOT).as_posix(): file_hash(p) for p in sorted(paths)}
