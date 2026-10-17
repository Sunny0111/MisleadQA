"""Check the existing single-A40 environment before downloading any weights."""
import argparse
import importlib.metadata
from pathlib import Path
import platform
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "experiments.main_eval"

from .vllm_backend import host_memory
from .protocol import load_models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True, choices=("transformers", "vllm"))
    parser.add_argument("--check-model-access", action="store_true",
                        help="Download only config.json at each pinned revision to check access")
    args = parser.parse_args()
    print("Python:", sys.version.split()[0], "Executable:", sys.executable, flush=True)
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise SystemExit("This A40 installation profile requires Linux x86_64")
    if sys.version_info[:2] != (3, 12):
        raise SystemExit("Use an isolated Python 3.12 environment for this installation profile")
    expected = ({"torch": "2.7.0+cu118", "transformers": "4.57.6", "accelerate": "1.10.1",
                 "huggingface-hub": "0.36.2"} if args.backend == "transformers" else
                {"torch": "2.7.0+cu118", "transformers": "4.53.2", "huggingface-hub": "0.33.2",
                 "vllm": "0.9.2"})
    differences = []
    for name, version in expected.items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual = "not installed"
        print(f"{name}: {actual}; expected {version}", flush=True)
        comparable = actual.split("+")[0] if name == "vllm" else actual
        if comparable != version:
            differences.append(name)
    if differences:
        requirements = "requirements-transformers-cu118.txt" if args.backend == "transformers" else "requirements-vllm-cu118.txt"
        raise SystemExit(f"Environment does not match: {', '.join(differences)}. "
                         f"Use a fresh Python 3.12 environment with experiments/configs/requirements/{requirements}")
    import torch
    print("CUDA runtime:", torch.version.cuda, flush=True)
    if (torch.version.cuda != "11.8" or not torch.cuda.is_available()
            or torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported()):
        raise SystemExit("Set CUDA_VISIBLE_DEVICES=0 and use the pinned CUDA 11.8 environment on the A40")
    props = torch.cuda.get_device_properties(0)
    print("GPU:", props.name, "VRAM GiB:", round(props.total_memory / 2**30, 2), flush=True)
    a = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
    if not torch.isfinite(a @ a).all().item():
        raise SystemExit("CUDA BF16 matrix multiplication failed")
    if args.backend == "vllm":
        import vllm
        memory = host_memory()
        print("Available RAM GiB:", round(memory["available_gib"], 1), flush=True)
        if memory["available_gib"] < 16:
            raise SystemExit("AWQ profile needs at least 16 GiB available RAM")
    if args.check_model_access:
        from huggingface_hub import hf_hub_download
        for alias, spec in load_models().items():
            if spec.get("required_backend", "transformers") != args.backend:
                continue
            print(f"Checking pinned config access: {alias}", flush=True)
            hf_hub_download(repo_id=spec["model_id"], filename="config.json", revision=spec["revision"])
    print(f"Environment preflight passed ({args.backend}). Run the actual model smoke test next.", flush=True)


if __name__ == "__main__":
    main()
