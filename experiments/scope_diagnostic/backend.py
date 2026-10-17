"""Reuse frozen loaders; greedily generate one literal English label then EOS."""
from experiments.main_eval.backend import TransformersBackend
from experiments.main_eval.vllm_backend import VLLMBackend, sampling_options
from .protocol import LABELS


def label_token_ids(tokenizer):
    sequences = {label: tokenizer.encode(label, add_special_tokens=False) for label in LABELS}
    for label, ids in sequences.items():
        if (not ids or set(ids) & set(tokenizer.all_special_ids)
                or tokenizer.decode(ids, skip_special_tokens=False,
                                    clean_up_tokenization_spaces=False) != label):
            raise ValueError(f"Label must round-trip through the pinned tokenizer: {label}")
    return sequences


class LabelConstraint:
    """A token trie, including terminal EOS, shared by both inference backends."""
    def __init__(self, sequences, eos):
        if (set(sequences) != set(LABELS) or not eos or any(type(i) is not int for i in eos)
                or any(not ids or any(type(i) is not int for i in ids) or set(ids) & set(eos)
                       for ids in sequences.values())
                or len({tuple(ids) for ids in sequences.values()}) != len(LABELS)):
            raise ValueError("Invalid label/EOS token metadata")
        self.next = {}
        for ids in sequences.values():
            for end, token in enumerate(ids):
                self.next.setdefault(tuple(ids[:end]), set()).add(token)
            self.next.setdefault(tuple(ids), set()).update(eos)
        self.budget = max(map(len, sequences.values())) + 1

    def allowed(self, prefix):
        try:
            return sorted(self.next[tuple(prefix)])
        except KeyError:
            raise ValueError("Unexpected scope-label token prefix") from None

    def __call__(self, generated_ids, logits):
        import torch
        masked = torch.full_like(logits, -float("inf"))
        permitted = self.allowed(generated_ids)
        masked[permitted] = logits[permitted]
        return masked


def verify_generation(row, metadata):
    tokens = row["generated_token_ids"]
    sequences, eos = metadata["label_token_ids"], metadata["eos_token_ids"]
    LabelConstraint(sequences, eos)
    if (row["response"] not in LABELS or not tokens or tokens[-1] not in eos
            or tokens[:-1] != sequences[row["response"]] or row["finish_reason"] != "stop"
            or row["input_tokens"] != len(row["input_token_ids"])
            or row["output_tokens"] != len(tokens)):
        raise ValueError("Expected exactly one scope label followed by EOS")


class ScopeBackend:
    """Adapter keeps the original checkpoint, chat template and greedy settings."""
    def __init__(self, spec, settings, engine=None, local_files_only=False):
        self.base = (VLLMBackend(spec, settings, engine, local_files_only) if engine else
                     TransformersBackend(spec, settings, local_files_only))
        self.sequences = label_token_ids(self.base.tokenizer)
        self.constraint = LabelConstraint(self.sequences, self.base.eos)
        self.metadata = self.base.metadata.copy()
        self.metadata.pop("choice_token_ids")
        self.metadata.update(label_token_ids=self.sequences, output_constraint="greedy label-token trie then EOS",
                             generation_config=self.base.metadata["generation_config"] |
                             {"max_new_tokens": self.constraint.budget})
        if engine:
            from vllm import SamplingParams
            sampling = sampling_options(settings, self.base.eos) | {"max_tokens": self.constraint.budget}
            self.sampling = SamplingParams(**sampling, logits_processors=[self.constraint])
            self.metadata["effective_sampling"] = sampling
        else:
            self.base.generation.max_new_tokens = self.constraint.budget

    def preflight(self, requests):
        maximum = max(len(self.base.render(r["messages"])[1]) for r in requests)
        if self.metadata["backend"] == "vllm":
            capacity = self.base.engine["llm"]["max_model_len"]
        else:
            config = getattr(self.base.model.config, "text_config", self.base.model.config)
            capacity = getattr(config, "max_position_embeddings", None)
        if capacity is not None and maximum + self.constraint.budget > capacity:
            raise ValueError("Prompt plus scope label exceeds the context window")
        print(f"Scope preflight: maximum {maximum} input tokens; label/EOS cap {self.constraint.budget}", flush=True)

    def generate(self, messages):
        rendered, ids = self.base.render(messages)
        if self.metadata["backend"] == "vllm":
            outputs = self.base.llm.generate([{"prompt_token_ids": ids}], self.sampling, use_tqdm=False)
            if len(outputs) != 1 or len(outputs[0].outputs) != 1 or not outputs[0].finished:
                raise ValueError("Expected one finished scope response")
            completion = outputs[0].outputs[0]
            tokens, finish = list(completion.token_ids), completion.finish_reason
        else:
            from transformers import set_seed
            from torch.nn.attention import sdpa_kernel, SDPBackend
            set_seed(self.base.settings["seed"])
            inputs = self.base.tokenizer(rendered, add_special_tokens=False, return_tensors="pt")
            inputs = inputs.to(self.base.model.get_input_embeddings().weight.device)
            if inputs["input_ids"][0].tolist() != ids:
                raise ValueError("Tokenizer encode/call mismatch")
            with self.base.torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
                output = self.base.model.generate(**inputs, generation_config=self.base.generation,
                    use_model_defaults=False,
                    prefix_allowed_tokens_fn=lambda batch, tokens: self.constraint.allowed(tokens[len(ids):].tolist()))
            tokens = output[0, len(ids):].tolist()
            finish = "stop" if tokens and tokens[-1] in self.base.eos else "length"
        response = self.base.tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        result = {"response": response, "generated_token_ids": tokens, "input_token_ids": ids,
                  "input_tokens": len(ids), "output_tokens": len(tokens),
                  "rendered_prompt": rendered, "finish_reason": finish}
        verify_generation(result, self.metadata)
        return result
