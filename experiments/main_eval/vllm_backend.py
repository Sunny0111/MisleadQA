"""Pinned offline vLLM V0 backend with the same ABCDE-then-EOS protocol."""
import importlib.metadata
import json
import os
from pathlib import Path
import platform

from .backend import choice_token_ids, package_versions, runtime, verify_generation
from .protocol import CONFIG_DIR, digest, file_hash


def default_engine_path(spec):
    if spec["model_id"] == "Qwen/Qwen3-32B":
        return CONFIG_DIR / "vllm_qwen32_bf16_a40.json"
    if spec["model_id"] == "Qwen/Qwen3-32B-AWQ":
        return CONFIG_DIR / "vllm_qwen32_awq_a40.json"
    raise ValueError("This vLLM profile supports Qwen3-32B checkpoints only")


def validate_engine(engine, spec, settings):
    if (engine["model_id"] != spec["model_id"] or spec["model_type"] != "qwen3"
            or spec["model_id"] not in ("Qwen/Qwen3-32B", "Qwen/Qwen3-32B-AWQ")):
        raise ValueError("vLLM engine profile does not match the selected Qwen3-32B checkpoint")
    if (engine["vllm_version"] != "0.9.2" or engine["engine_version"] != "V0"
            or engine["attention_backend"] != "FLASH_ATTN"):
        raise ValueError("This profile requires vLLM 0.9.2, V0, and FLASH_ATTN")
    llm = engine["llm"]
    quantization = "awq_marlin" if spec.get("weight_quantization") == "awq_4bit" else None
    expected = {"quantization": quantization, "dtype": "bfloat16", "tensor_parallel_size": 1,
                "max_num_seqs": 1, "enforce_eager": True, "enable_prefix_caching": False,
                "enable_chunked_prefill": False, "swap_space": 0, "generation_config": "vllm",
                "trust_remote_code": False}
    if any(llm.get(k) != v for k, v in expected.items()):
        raise ValueError("Unexpected single-A40 engine policy")
    if not 0 < llm["gpu_memory_utilization"] < 1 or llm["cpu_offload_gb"] < 0:
        raise ValueError("Invalid memory budget")
    if (llm["max_model_len"] < settings["max_input_tokens"] + 2
            or llm["max_num_batched_tokens"] < llm["max_model_len"]):
        raise ValueError("Engine must fit complete prompts plus the answer without truncation")
    if (settings["dtype"] != "bfloat16" or settings["max_new_tokens"] != 2
            or settings["batch_size"] != 1):
        raise ValueError("vLLM requires the common BF16, one-request, letter/EOS protocol")


def load_engine(spec, settings, path=None):
    path = Path(path) if path is not None else default_engine_path(spec)
    engine = json.loads(path.read_text(encoding="utf-8"))
    validate_engine(engine, spec, settings)
    return engine


def host_memory():
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, value = line.split(":", 1)
        values[key] = int(value.split()[0]) * 1024
    return {"total_gib": values["MemTotal"] / 2**30,
            "available_gib": values["MemAvailable"] / 2**30}


def sampling_options(settings, eos):
    return {"n": 1, "temperature": 0.0, "top_p": 1.0, "top_k": -1, "min_p": 0.0,
            "seed": settings["seed"], "max_tokens": 2, "min_tokens": 0,
            "repetition_penalty": 1.0, "presence_penalty": 0.0, "frequency_penalty": 0.0,
            "ignore_eos": False, "stop_token_ids": eos, "skip_special_tokens": True,
            "spaces_between_special_tokens": False}


class ChoiceThenEOS:
    """The V0 processor receives generated token IDs, then the current logits."""
    def __init__(self, choices, eos):
        self.choices, self.eos = tuple(choices), tuple(eos)
        if not self.eos or set(self.choices) & set(self.eos):
            raise ValueError("Choice and EOS tokens must be distinct")

    def __call__(self, generated_ids, logits):
        import torch
        if len(generated_ids) == 0:
            permitted = self.choices
        elif len(generated_ids) == 1 and generated_ids[0] in self.choices:
            permitted = self.eos
        else:
            raise ValueError("Unexpected constrained vLLM generation prefix")
        masked = torch.full_like(logits, -float("inf"))
        masked[list(permitted)] = logits[list(permitted)]
        return masked


class VLLMBackend:
    def __init__(self, spec, settings, engine, local_files_only=False):
        validate_engine(engine, spec, settings)
        if platform.system() != "Linux" or platform.machine() != "x86_64":
            raise RuntimeError("Real vLLM inference requires Linux x86_64")
        os.environ["VLLM_USE_V1"] = "0"
        os.environ["VLLM_ATTENTION_BACKEND"] = engine["attention_backend"]
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        actual = importlib.metadata.version("vllm")
        if actual.split("+")[0] != engine["vllm_version"]:
            raise ValueError(f"Expected vLLM 0.9.2, found {actual}")
        import torch
        if (torch.version.cuda != "11.8" or not torch.cuda.is_available()
                or torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported()):
            raise RuntimeError("Select one BF16-capable GPU and the pinned CUDA 11.8 vLLM environment")
        memory = host_memory()
        if memory["available_gib"] < engine["minimum_available_ram_gib"]:
            raise RuntimeError(f"Profile needs at least {engine['minimum_available_ram_gib']} GiB available RAM; "
                               f"found {memory['available_gib']:.1f} GiB. Do not substitute weights silently")
        from huggingface_hub import snapshot_download
        from transformers import AutoConfig, AutoTokenizer, GenerationConfig
        from vllm import LLM, SamplingParams
        snapshot = snapshot_download(spec["model_id"], revision=spec["revision"],
                    local_files_only=local_files_only,
                    allow_patterns=["*.safetensors", "*.json", "*.model", "*.txt", "*.tiktoken", "*.jinja"])
        config = AutoConfig.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
        checkpoint_quantization = getattr(config, "quantization_config", None)
        if spec.get("weight_quantization") == "awq_4bit":
            if not checkpoint_quantization or checkpoint_quantization.get("quant_method") != "awq" or checkpoint_quantization.get("bits") != 4:
                raise ValueError("Expected the actual official 4-bit AWQ checkpoint")
        elif checkpoint_quantization:
            raise ValueError("The BF16 alias cannot load quantized weights")
        self.model_config = config
        self.tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
        self.choice_ids = choice_token_ids(self.tokenizer)
        defaults = GenerationConfig.from_pretrained(snapshot, local_files_only=True)
        eos = defaults.eos_token_id if defaults.eos_token_id is not None else self.tokenizer.eos_token_id
        self.eos = eos if isinstance(eos, list) else [eos]
        if not self.eos or any(type(i) is not int for i in self.eos) or set(self.choice_ids.values()) & set(self.eos):
            raise ValueError("Invalid choice/EOS metadata")
        self.settings, self.engine = settings, engine
        self.template_kwargs = spec["chat_template_kwargs"]
        sampling = sampling_options(settings, self.eos)
        self.sampling = SamplingParams(**sampling, logits_processors=[ChoiceThenEOS(self.choice_ids.values(), self.eos)])
        self.llm = LLM(model=snapshot, tokenizer=snapshot, seed=settings["seed"], **engine["llm"])
        environment = runtime(torch)
        environment["attention_backend"] = engine["attention_backend"]
        environment["deterministic_algorithms"] = None
        environment["host_total_memory_gib"] = memory["total_gib"]
        environment["environment"] = {k: os.environ.get(k) for k in
                                      ("VLLM_USE_V1", "VLLM_ATTENTION_BACKEND", "VLLM_WORKER_MULTIPROC_METHOD")}
        assets = {p.relative_to(snapshot).as_posix(): file_hash(p) for p in sorted(Path(snapshot).rglob("*"))
                  if p.is_file() and p.suffix in (".json", ".model", ".txt", ".tiktoken", ".jinja")}
        self.metadata = {
            "backend": "vllm", "vllm_version": actual, "model_id": spec["model_id"], "revision": spec["revision"],
            "loader": "vllm_offline_V0", "dtype": "bfloat16", "quantization": spec.get("weight_quantization"),
            "choice_token_ids": self.choice_ids, "eos_token_ids": self.eos,
            "chat_template_kwargs": self.template_kwargs,
            "chat_template_sha256": digest(self.tokenizer.get_chat_template().encode("utf-8")),
            "tokenizer_sha256": digest(self.tokenizer.backend_tokenizer.to_str().encode("utf-8")),
            "asset_sha256": assets, "effective_sampling": sampling, "engine": engine,
            "generation_config": {"do_sample": False, "max_new_tokens": 2, "num_beams": 1,
                                  "num_return_sequences": 1, "eos_token_id": self.eos},
            "packages": package_versions(), "runtime": environment,
            "determinism": "Greedy, fixed seed, eager sequential execution; no cross-backend bitwise guarantee",
        }

    def render(self, messages):
        rendered = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                      **self.template_kwargs)
        ids = self.tokenizer.encode(rendered, add_special_tokens=False)
        if len(ids) > self.settings["max_input_tokens"]:
            raise ValueError(f"Prompt has {len(ids)} tokens; refusing to truncate")
        return rendered, ids

    def preflight(self, requests):
        maximum = max(len(self.render(r["messages"])[1]) for r in requests)
        if maximum + 2 > self.engine["llm"]["max_model_len"]:
            raise ValueError("Prompt plus answer exceeds the vLLM engine window")
        print(f"vLLM tokenizer preflight: maximum {maximum} input tokens", flush=True)

    def generate(self, messages):
        rendered, ids = self.render(messages)
        outputs = self.llm.generate([{"prompt_token_ids": ids}], self.sampling, use_tqdm=False)
        if len(outputs) != 1 or len(outputs[0].outputs) != 1 or not outputs[0].finished:
            raise ValueError("Expected one finished vLLM response")
        completion = outputs[0].outputs[0]
        tokens = list(completion.token_ids)
        response = self.tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        result = {"response": response, "generated_token_ids": tokens, "input_token_ids": ids,
                  "input_tokens": len(ids), "output_tokens": len(tokens), "rendered_prompt": rendered,
                  "finish_reason": completion.finish_reason, "stop_reason": completion.stop_reason}
        verify_generation(result, self.metadata)
        return result
