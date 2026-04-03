import argparse
import os
from pathlib import Path

from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None


def load_env() -> None:
    if load_dotenv is None:
        return
    repo_root = Path(__file__).resolve().parents[1]
    env_path = repo_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)


def apply_runtime_env() -> None:
    mapping = {
        "HTTP_PROXY": "http_proxy",
        "HTTPS_PROXY": "https_proxy",
        "TRANSFORMERS_CACHE": "TRANSFORMERS_CACHE",
        "HF_DATASETS_CACHE": "HF_DATASETS_CACHE",
        "HF_HOME": "HF_HOME",
        "HF_ENDPOINT": "HF_ENDPOINT",
    }
    for src_key, dst_key in mapping.items():
        value = os.getenv(src_key)
        if value:
            os.environ[dst_key] = value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download HuggingFace model/tokenizer into local cache.")
    parser.add_argument("--model-id", default="Qwen/Qwen3-32B", help="HuggingFace model id")
    parser.add_argument(
        "--model-type",
        choices=["causal-lm", "sequence-classification", "encoder", "tokenizer-only"],
        default="tokenizer-only",
        help="Type of model to download",
    )
    parser.add_argument(
        "--download-dir",
        default=os.getenv("HF_MODEL_CACHE", str(Path(__file__).resolve().parent / "hf_model_cache")),
        help="Local directory used as cache_dir",
    )
    parser.add_argument("--trust-remote-code", action="store_true", help="Enable trust_remote_code")
    parser.add_argument("--torch-dtype", default="auto", help="torch_dtype argument")
    parser.add_argument("--skip-tokenizer", action="store_true", help="Skip tokenizer download")
    return parser.parse_args()


def download_assets(args: argparse.Namespace) -> None:
    token = os.getenv("HF_TOKEN")
    download_dir = Path(args.download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_tokenizer:
        AutoTokenizer.from_pretrained(
            args.model_id,
            cache_dir=str(download_dir),
            token=token,
            trust_remote_code=args.trust_remote_code,
        )

    if args.model_type == "tokenizer-only":
        return

    common_kwargs = {
        "cache_dir": str(download_dir),
        "token": token,
        "trust_remote_code": args.trust_remote_code,
    }

    if args.model_type == "causal-lm":
        AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=args.torch_dtype, **common_kwargs)
    elif args.model_type == "sequence-classification":
        AutoModelForSequenceClassification.from_pretrained(args.model_id, **common_kwargs)
    elif args.model_type == "encoder":
        AutoModel.from_pretrained(args.model_id, **common_kwargs)


def main() -> None:
    load_env()
    apply_runtime_env()
    args = parse_args()
    download_assets(args)
    print(f"Download completed: {args.model_id} -> {args.download_dir}")


if __name__ == "__main__":
    main()
