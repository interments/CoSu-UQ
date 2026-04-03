import os
from pathlib import Path


_DEFAULT_ROOT = str(Path(__file__).resolve().parents[1])

run_dir = os.getenv("COSU_UQ_RUN_DIR", _DEFAULT_ROOT)
data_dir = os.getenv("COSU_UQ_DATA_DIR", "./datasets")
output_dir = os.getenv("COSU_UQ_OUTPUT_DIR", "./results")

hf_datasets_cache = os.getenv("HF_DATASETS_CACHE", os.path.join(run_dir, "hf_datasets_cache"))
transformers_cache = os.getenv("TRANSFORMERS_CACHE", os.path.join(run_dir, "transformers_cache"))
hf_home_cache = os.getenv("HF_HOME", os.path.join(run_dir, "hf_home_cache"))
hf_model_cache = os.getenv("COSU_UQ_HF_MODEL_CACHE", os.path.join(run_dir, "models", "hf_model_cache"))

# Tokens are intentionally read from env vars only.
hf_access_token_llama_3 = os.getenv("HF_ACCESS_TOKEN_LLAMA_3", "")
hf_llama3_8B_instruct = "meta-llama/Llama-3.1-8B-Instruct"
