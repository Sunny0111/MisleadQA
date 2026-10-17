"""Transformers BF16 backend; one canonical choice token followed by EOS."""
import importlib.metadata
import os
from pathlib import Path
import platform
import subprocess

from .protocol import LETTERS, digest, file_hash


def package_versions():
    names = ("torch", "transformers", "accelerate", "huggingface-hub", "tokenizers",
             "safetensors", "sentencepiece", "protobuf", "numpy")
    result = {"python": platform.python_version()}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    result["installed_distributions"] = dict(sorted(
        (dist.metadata["Name"].lower(), dist.version) for dist in importlib.metadata.distributions()
        if dist.metadata.get("Name")))
    return result


def setup(seed):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    import torch
    from transformers import set_seed
    set_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    return torch


def runtime(torch):
    try:
        command = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=10)
        driver = command.stdout.strip() if command.returncode == 0 else None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        driver = None
    return {
        "platform": platform.platform(), "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(), "driver": driver,
        "gpus": [{"name": torch.cuda.get_device_name(i),
                  "capability": list(torch.cuda.get_device_capability(i)),
                  "total_memory": torch.cuda.get_device_properties(i).total_memory}
                 for i in range(torch.cuda.device_count())],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "attention_backend": "sdpa_math", "tf32": torch.backends.cuda.matmul.allow_tf32,
        "torch_build": torch.__config__.show(),
    }


def choice_token_ids(tokenizer):
    choices = {}
    for letter in LETTERS:
        ids = tokenizer.encode(letter, add_special_tokens=False)
        if (len(ids) != 1 or ids[0] in tokenizer.all_special_ids
                or tokenizer.decode(ids, skip_special_tokens=False,
                                    clean_up_tokenization_spaces=False) != letter):
            raise ValueError(f"{letter} must encode to one canonical non-special token")
        choices[letter] = ids[0]
    if len(set(choices.values())) != len(LETTERS):
        raise ValueError("Choice tokens must be distinct")
    return choices


def allowed_tokens(prompt_length, choices, eos_ids):
    if (not eos_ids or any(type(i) is not int for i in eos_ids)
            or set(choices.values()) & set(eos_ids)):
        raise ValueError("EOS and choice tokens must be valid and disjoint")

    def allowed(batch_id, input_ids):
        count = len(input_ids) - prompt_length
        if count == 0:
            return list(choices.values())
        if count == 1 and int(input_ids[-1]) in choices.values():
            return eos_ids
        raise ValueError("Unexpected constrained generation prefix")
    return allowed


def verify_generation(row, metadata):
    choices, eos = metadata["choice_token_ids"], metadata["eos_token_ids"]
    tokens = row["generated_token_ids"]
    if (row["response"] not in tuple(LETTERS) or len(tokens) != 2
            or tokens[0] != choices.get(row["response"]) or tokens[1] not in eos
            or row["finish_reason"] != "stop"):
        raise ValueError("Expected exactly one A/B/C/D/E token followed by EOS")
    if row["input_tokens"] != len(row["input_token_ids"]) or row["output_tokens"] != len(tokens):
        raise ValueError("Token counts differ from retained token IDs")


class TransformersBackend:
    def __init__(self, spec, settings, local_files_only=False):
        self.torch = setup(settings["seed"])
        torch = self.torch
        if not torch.cuda.is_available():
            raise RuntimeError("Real inference requires a CUDA GPU; use --dry-run for local preparation")
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("The selected CUDA GPU does not support BF16")
        from huggingface_hub import snapshot_download
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig
        from transformers import Gemma3ForConditionalGeneration
        snapshot = snapshot_download(spec["model_id"], revision=spec["revision"],
                                     local_files_only=local_files_only,
                                     allow_patterns=["*.safetensors", "*.json", "*.model", "*.txt", "*.tiktoken", "*.jinja"])
        config = AutoConfig.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
        if config.model_type != spec["model_type"] or getattr(config, "quantization_config", None):
            raise ValueError("Checkpoint has an unexpected model type or quantization")
        self.tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
        self.choice_ids = choice_token_ids(self.tokenizer)
        loader = Gemma3ForConditionalGeneration if spec["loader"] == "gemma3_conditional" else AutoModelForCausalLM
        self.model = loader.from_pretrained(snapshot, dtype=torch.bfloat16, device_map="auto",
                                           attn_implementation="sdpa", local_files_only=True,
                                           trust_remote_code=False).eval()
        device_map = {k: str(v) for k, v in self.model.hf_device_map.items()}
        if any(v in ("cpu", "disk") for v in device_map.values()):
            raise RuntimeError("Insufficient GPU memory: weights were offloaded. Select more/larger GPUs")
        self.settings = settings
        self.template_kwargs = spec["chat_template_kwargs"]
        defaults = self.model.generation_config
        eos = defaults.eos_token_id
        if eos is None:
            eos = self.tokenizer.eos_token_id
        self.eos = eos if isinstance(eos, list) else [eos]
        allowed_tokens(0, self.choice_ids, self.eos)
        pad = defaults.pad_token_id if defaults.pad_token_id is not None else self.eos[0]
        self.generation = GenerationConfig(
            do_sample=False, num_beams=1, num_return_sequences=1, max_new_tokens=2,
            min_new_tokens=0, repetition_penalty=1.0, length_penalty=1.0,
            no_repeat_ngram_size=0, use_cache=True, disable_compile=True,
            eos_token_id=self.eos, pad_token_id=pad, bos_token_id=defaults.bos_token_id)
        # Only small configuration/tokenizer assets are hashed. Weights come from a pinned Hub snapshot.
        assets = {p.relative_to(snapshot).as_posix(): file_hash(p) for p in sorted(Path(snapshot).rglob("*"))
                  if p.is_file() and p.suffix in (".json", ".model", ".txt", ".tiktoken", ".jinja")}
        self.metadata = {
            "backend": "transformers", "model_id": spec["model_id"], "revision": spec["revision"],
            "loader": spec["loader"], "dtype": "bfloat16", "quantization": None,
            "choice_token_ids": self.choice_ids, "eos_token_ids": self.eos,
            "chat_template_kwargs": self.template_kwargs,
            "chat_template_sha256": digest(self.tokenizer.get_chat_template().encode("utf-8")),
            "tokenizer_sha256": digest(self.tokenizer.backend_tokenizer.to_str().encode("utf-8")),
            "asset_sha256": assets, "generation_config": self.generation.to_dict(),
            "device_map": device_map, "packages": package_versions(), "runtime": runtime(torch),
        }

    def render(self, messages):
        rendered = self.tokenizer.apply_chat_template(messages, tokenize=False,
                    add_generation_prompt=True, **self.template_kwargs)
        # Chat templates already insert BOS and other required special tokens.
        ids = self.tokenizer.encode(rendered, add_special_tokens=False)
        if len(ids) > self.settings["max_input_tokens"]:
            raise ValueError(f"Prompt has {len(ids)} tokens; refusing to truncate")
        return rendered, ids

    def preflight(self, requests):
        maximum = max(len(self.render(r["messages"])[1]) for r in requests)
        text_config = getattr(self.model.config, "text_config", self.model.config)
        capacity = getattr(text_config, "max_position_embeddings", None)
        if capacity is not None and maximum + 2 > capacity:
            raise ValueError("Prompt plus answer exceeds the model context window")
        print(f"Tokenizer preflight: maximum {maximum} input tokens", flush=True)

    def generate(self, messages):
        from transformers import set_seed
        from torch.nn.attention import sdpa_kernel, SDPBackend
        set_seed(self.settings["seed"])
        rendered, ids = self.render(messages)
        inputs = self.tokenizer(rendered, add_special_tokens=False, return_tensors="pt")
        inputs = inputs.to(self.model.get_input_embeddings().weight.device)
        size = len(ids)
        if inputs["input_ids"][0].tolist() != ids:
            raise ValueError("Tokenizer encode/call mismatch")
        with self.torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
            output = self.model.generate(**inputs, generation_config=self.generation,
                use_model_defaults=False,
                prefix_allowed_tokens_fn=allowed_tokens(size, self.choice_ids, self.eos))
        tokens = output[0, size:].tolist()
        response = self.tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        result = {"response": response, "generated_token_ids": tokens, "input_token_ids": ids,
                  "input_tokens": size, "output_tokens": len(tokens), "rendered_prompt": rendered,
                  "finish_reason": "stop" if tokens and tokens[-1] in self.eos else "length"}
        verify_generation(result, self.metadata)
        return result
