# CoSu-UQ: Uncertainty Quantification for Reasoning via Confidence and Semantic Support

Official implementation of the IJCNN 2026 paper **CoSu-UQ**, a multi-sample uncertainty quantification framework for long-form reasoning in large language models.

- Paper: [IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/11730028)
- DOI: [10.1109/IJCNN69854.2026.11730028](https://doi.org/10.1109/IJCNN69854.2026.11730028)
- Code: [github.com/interments/CoSu-UQ](https://github.com/interments/CoSu-UQ)

Authors: Yinglun Feng, Qinhong Lin, Yuhao Zhang, Zhongliang Yang, Linna Zhou.

Affiliations: School of Cyberspace Security, Beijing University of Posts and Telecommunications, Beijing, China; QuanCheng Laboratory, Jinan, China. Zhongliang Yang and Linna Zhou are corresponding authors.

## Overview

CoSu-UQ combines two complementary uncertainty signals:

- **Confidence signal**: token-level generation probabilities are aggregated to keyword, sentence, and response-level confidence.
- **Semantic support signal**: sentence-level entailment probabilities are used to measure cross-sample support over reasoning paths.

The two signals are fused at the response level, and uncertainty is aggregated over semantic clusters formed by bidirectional entailment on cleaned final answers. This keeps the uncertainty estimate anchored at the final decision rather than at noisy intermediate reasoning steps.

The paper evaluates CoSu-UQ on GSM8K, MATH, HotpotQA, 2WikiMultiHopQA, and MedQA with Qwen3-4B, Qwen3-14B, Qwen3-32B, and Llama-3.1-8B.

## Highlights

- End-to-end pipeline from dataset construction to AUROC comparison.
- Unified shell orchestration for single-stage, full-pipeline, and batch execution.
- Baselines supported by this code release: PE, LN-PE, SE, Sentence-SAR, Token-SAR, SC, LUQ, and CoT-UQ.
- Reproducible CoSu-UQ signal decomposition:
  - `confidence_level`
  - `support_level`
  - `combined_scores`
- Paper-level results and the exact experimental configuration are summarized in [Main Results](#main-results).

> Self-Consistency (SC) is implemented in stage 6b as `1 - (largest semantic cluster mass / number of samples)`.

## Repository Structure

```text
CoSu-UQ/
├── cache/                        # runtime cache
├── datasets/                     # processed dataset files
├── logs/                         # logs
├── metrics/                      # evaluation artifacts
├── models/
│   ├── Download_models.py        # HuggingFace model download helper
│   └── hf_model_cache/           # local model cache (created on demand)
├── results/                      # per-run outputs and final comparison CSV
├── src/
│   ├── pipeline/                 # stage 0 - stage 4 core pipeline
│   ├── baselines/                # baseline cache/scoring, LUQ and CoT-UQ
│   ├── eval/                     # judge labels, AUROC labels, final comparison
│   ├── scripts/                  # orchestration shell scripts
│   ├── docs/                     # IO contracts and stage documentation
│   ├── utils/                    # shared utilities
│   ├── config.py
│   └── requirements.txt
├── LICENSE
└── README.md
```

## Environment

- Linux with an NVIDIA GPU.
- CUDA-enabled PyTorch. Stages 1-4 use CUDA directly; CPU-only and macOS/MPS execution is not supported out of the box.
- Python 3.10+.
- The paper experiments used an Intel Xeon Platinum 8336C CPU and two NVIDIA A100-SXM4-80GB GPUs.

Model weights and generated results are not committed to this repository. Plan for sufficient disk space for the HuggingFace cache, especially for the 14B and 32B checkpoints.

## Installation

```bash
git clone https://github.com/interments/CoSu-UQ.git
cd CoSu-UQ

python3 -m venv .venv
source .venv/bin/activate
pip install -r src/requirements.txt

# Needed only when invoking Python modules manually.
export PYTHONPATH="$PWD/src"
```

The shell scripts set `PYTHONPATH` automatically. The manual export is for commands such as `python -m pipeline.stage1_generate_cot`.

## Model Preparation

Set a HuggingFace token if you use gated checkpoints such as Llama:

```bash
export HF_TOKEN="hf_..."
```

Download a model into the local cache:

```bash
python models/Download_models.py \
  --model-id Qwen/Qwen3-4B \
  --model-type causal-lm \
  --download-dir models/hf_model_cache
```

The code is configured for the following checkpoints. The paper refers to the Llama model as Llama-3.1-8B; `src/config.py` and the final comparison ordering use the Instruct checkpoint.

| Model | HuggingFace ID |
| :--- | :--- |
| Llama-3.1-8B | `meta-llama/Llama-3.1-8B-Instruct` |
| Qwen3-4B | `Qwen/Qwen3-4B` |
| Qwen3-14B | `Qwen/Qwen3-14B` |
| Qwen3-32B | `Qwen/Qwen3-32B` |

Set `MODEL_DIR` to the local snapshot directory containing `config.json`, tokenizer files, and model weights. When using the HuggingFace cache layout, this is usually a path like:

```text
models/hf_model_cache/models--Qwen--Qwen3-4B/snapshots/<revision>/
```

`TOKENIZER_MODEL_DIR` defaults to `MODEL_DIR`. Set it explicitly when the generation model and the tokenizer used for confidence extraction are different, for example when using the API generation path.

## Dataset Preparation

Build the processed datasets from HuggingFace:

```bash
bash src/scripts/run_stage.sh 0
```

With the defaults in `src/scripts/common.env` (`NUM_SAMPLES=500`), this writes:

| Dataset | Output file |
| :--- | :--- |
| GSM8K | `datasets/gsm8k_cot_uq_validation_0-500_src.jsonl` |
| MATH | `datasets/math_cot_uq_validation_0-500.jsonl` |
| HotpotQA | `datasets/hotpotqa_cot_uq_validation_0-500_src.jsonl` |
| 2WikiMultiHopQA | `datasets/2WikimultihopQA_cot_uq_validation_0-500_src.jsonl` |
| MedQA | `datasets/MedQA_cot_uq_validation_0-500_src.jsonl` |

Dataset processing follows the paper:

- MATH answers are normalized to parsable numeric values or simple expressions.
- For multi-hop QA, only the question text is used; supporting documents are not provided.
- All tasks use one-shot Chain-of-Thought prompting.
- MedQA enforces the `Final Answer: [Option Letter] ([Option Text])` format during parsing.

`FRACTION=0.6` means that stage 1 uses 60% of the 500 processed samples, i.e. 300 samples per dataset, matching the paper's experimental setting.

`DATA_FILE` points to one dataset at a time. The default in `common.env` is GSM8K; update it manually when switching datasets. `DATA_FILE` is not derived automatically from `NUM_SAMPLES`.

## Quick Start

1. Edit the main configuration file:

```bash
${EDITOR:-vi} src/scripts/common.env
```

At minimum, set:

```bash
MODEL_NAME="Qwen3-4B"
MODEL_DIR="/absolute/path/to/Qwen3-4B"
DATA_FILE="datasets/gsm8k_cot_uq_validation_0-500_src.jsonl"
DEVICE="0"
```

2. Build the datasets:

```bash
bash src/scripts/run_stage.sh 0
```

3. Run the full pipeline (stages 1-7) for the configured dataset:

```bash
bash src/scripts/run_pipeline.sh
```

`run_pipeline.sh` enables stages 1-7 by default and disables stage 0. To include dataset construction in a full run:

```bash
DO_STAGE0=true bash src/scripts/run_pipeline.sh
```

To run a single stage, use `bash src/scripts/run_stage.sh <stage>`. If generation has already completed, continue from stage 2 with:

```bash
DO_STAGE1=false bash src/scripts/run_pipeline.sh
```

Stage 5j requires judge API credentials. For a pipeline smoke test without judge labeling, disable the label-dependent stages:

```bash
DO_STAGE5J=false DO_STAGE5L=false DO_STAGE7=false bash src/scripts/run_pipeline.sh
```

To run all five datasets, override `DATA_FILE` per invocation. Each invocation derives its own `run_setting` from the file name and generation config:

```bash
DATA_FILE=datasets/gsm8k_cot_uq_validation_0-500_src.jsonl bash src/scripts/run_pipeline.sh
DATA_FILE=datasets/math_cot_uq_validation_0-500.jsonl bash src/scripts/run_pipeline.sh
DATA_FILE=datasets/hotpotqa_cot_uq_validation_0-500_src.jsonl bash src/scripts/run_pipeline.sh
DATA_FILE=datasets/2WikimultihopQA_cot_uq_validation_0-500_src.jsonl bash src/scripts/run_pipeline.sh
DATA_FILE=datasets/MedQA_cot_uq_validation_0-500_src.jsonl bash src/scripts/run_pipeline.sh
```

Batch multiple precomputed `run_setting` values and reuse existing generations:

```bash
DO_STAGE1=false bash src/scripts/run_batch.sh <run_settings_file>
```

The batch file contains one `run_setting` per line. Empty lines and lines starting with `#` are ignored.

## Configuration

The main configuration file is [`src/scripts/common.env`](src/scripts/common.env). It is sourced by the shell scripts, so values use shell syntax.

Defaults in `common.env` are applied only when a variable is unset, so inline overrides such as `DATA_FILE=... bash src/scripts/run_pipeline.sh` or `NLI_MODEL_CLUSTER=... bash src/scripts/run_stage.sh 2` take effect.

Key runtime variables:

| Variable | Default | Meaning |
| :--- | :--- | :--- |
| `PYTHON_BIN` | `python` | Python executable used by the shell scripts. |
| `MODEL_NAME` | `Llama-3.1-8B` | Model name used in `run_setting` and output paths. |
| `MODEL_DIR` | `/path/to/local/model` | Local model snapshot path for generation. |
| `DATA_FILE` | GSM8K file | Input dataset JSONL for stage 1. |
| `DEVICE` | `0` | Value assigned to `CUDA_VISIBLE_DEVICES`. |
| `FRACTION` | `0.6` | Fraction of samples used by stage 1; `0.6` of 500 gives 300 samples. |
| `NUM_GENERATIONS` | `5` | Number of sampled CoT responses per question. |
| `TEMPERATURE` | `1.0` | Sampling temperature. |
| `TOP_K` | `5` | Top-k sampling parameter. |
| `TOP_P` | `0.95` | Nucleus sampling parameter. |
| `MAX_LENGTH` | `512` | Maximum generated length. |
| `DECODE_METHOD` | `greedy` | Decoding method recorded in `run_setting`. |
| `SEED` | `42` | Random seed. |
| `USE_API` | `false` | Use the API generation path instead of a local model. |
| `API_KEY` / `API_BASE` / `API_MODEL_NAME` | empty | DashScope-compatible API generation settings. |
| `NLI_MODEL_SUPPORT` | `potsawee/deberta-v3-large-mnli` | NLI checkpoint used by stage 3 for support scoring. |
| `NLI_MODEL_CLUSTER` | `microsoft/deberta-large-mnli` | NLI checkpoint used by stage 2 for answer clustering. |
| `SPLIT_METHOD` | `step_answer` | Reasoning-unit split method for support files. |
| `TOKENIZER_MODEL_DIR` | `MODEL_DIR` | Tokenizer used by confidence extraction and baseline token importance. |
| `JUDGERS` | empty | Judge specification list for stage 5j. |
| `MAX_RETRIES` / `RETRY_INTERVAL` | `3` / `5.0` | Judge retry policy. |
| `SENT_SIM_MODEL` | `cross-encoder/stsb-roberta-large` | Sentence-similarity model for Sentence-SAR. |
| `TOKEN_IMPT_MODEL` | `cross-encoder/stsb-roberta-large` | Token-importance model for Token-SAR and CoT-UQ. |
| `NUM_GENERATION_FOR_BASELINE` | `5` | Number of generations used by baseline scoring. |
| `SAR_TEMPERATURE` | `0.001` | Temperature used by SAR scoring. |
| `COTUQ_PROMPT_TYPE` | `src` | CoT-UQ prompt type. |
| `COTUQ_AGG_METHOD` | `probas_min` | CoT-UQ aggregation method. |
| `COTUQ_USE_GREEDY` | `false` | Use sampled generations for CoT-UQ. |
| `RUN_SETTING` | empty | Explicit run name; derived from stage-1 args when empty. |
| `RUN_SETTINGS_JSON` | empty | Optional JSON list of run settings for stage 7. When empty, `run_stage.sh 7` evaluates the current `RUN_SETTING`. |
| `MAX_WORKERS` | `8` | Parallel workers for final comparison. |
| `OUTPUT_CSV` | `results/final_uq_baseline_compare.csv` | Final comparison table path. |

### `run_setting` naming

When `RUN_SETTING` is empty, the shell scripts derive it from the model and dataset configuration. `final_compare.py` parses the name as:

```python
model = run_setting.split("_")[0]
dataset = run_setting.split("_")[1]
```

Therefore:

- `MODEL_NAME` must not contain underscores.
- The dataset identifier must be the second underscore-separated field.
- Keep `RUN_SETTING` identical across stages.
- The model ordering in `final_compare.py` expects `Llama-3.1-8B-Instruct`, `Qwen3-4B`, `Qwen3-14B`, and `Qwen3-32B`. If you use a different `MODEL_NAME`, update the ordering list in `src/eval/final_compare.py` or accept an unranked model row.

## Pipeline Stages

| Stage | Module | Purpose | Main output |
| :--- | :--- | :--- | :--- |
| `0` | `pipeline.stage0_build_datasets` | Build processed JSONL datasets from HuggingFace. | `datasets/*.jsonl` |
| `1` | `pipeline.stage1_generate_cot` | Multi-sample CoT generation. | `generations.pkl` |
| `2` | `pipeline.stage2_semantic_cluster` | Bidirectional-entailment semantic clustering. | `semantic_clusters.pkl` |
| `3` | `pipeline.stage3_compute_support` | Compute LUQPair and LUQ support uncertainty. | `support_uncertainty_luqpair_{split_method}.json`, `support_uncertainty_luq_{split_method}.json` |
| `4` | `pipeline.stage4_extract_confidence` | Extract keyword token probabilities. | `confidence_keywords_probs.json` |
| `5j` | `eval.judge_responses` | Judge response correctness and backfill labels. | Updated `generations.pkl` |
| `5l` | `eval.build_auroc_labels` | Build AUROC labels. | `AUROC_labels.json` |
| `6s` | `baselines.cache_sentence_similarity` | Cache sentence similarities. | `sentence_similarities_{model_key}.pkl` |
| `6t` | `baselines.cache_token_importance` | Cache token importance. | `tokenwise_importance_{model_key}_from_generation.pkl` |
| `6b` | `baselines.build_baseline_scores` | Build baseline score files. | `*_scores.pkl` |
| `6c` | `baselines.cotuq_keyword_extraction` + `baselines.cotuq_aggregate_scores` | Build CoT-UQ baseline. | `cotuq_scores.pkl` |
| `7` | `eval.final_compare` | Compute AUROC for CoSu-UQ and baselines. | `results/final_uq_baseline_compare.csv` |

Detailed I/O schemas:

- [`src/docs/pipeline_io_contract.md`](src/docs/pipeline_io_contract.md)
- Chinese version: [`src/docs/pipeline_io_contract_zh.md`](src/docs/pipeline_io_contract_zh.md)
- Stage overview: [`src/docs/pipeline_stages.md`](src/docs/pipeline_stages.md)
- Data fields: [`src/docs/data_fields.md`](src/docs/data_fields.md)

Stage 7 automatically evaluates the current `RUN_SETTING`. For a manual multi-run comparison, pass one or more `--run-setting` arguments or a `--run-settings-json` file to `src/eval/final_compare.py`. The `DEFAULT_RUN_SETTINGS` list in the module is only a fallback for manual invocations without either option.

## Method-to-File Contract

To avoid method leakage between CoSu-UQ and the LUQ baseline, stage 3 writes two distinct support files:

- `support_uncertainty_luqpair_{split_method}.json`
  Used by the CoSu-UQ main branch (stages 4 and 7).
- `support_uncertainty_luq_{split_method}.json`
  Used only for LUQ baseline scoring (`luq_scores.pkl` in stage 6b).

Confidence extraction output is standardized as:

- `confidence_keywords_probs.json`

This separation is required for faithful reproduction of the paper setup.

### NLI checkpoints used in the paper

The two branches use separate NLI checkpoints, matching the defaults in `common.env`:

- `NLI_MODEL_SUPPORT=potsawee/deberta-v3-large-mnli` for stage 3 support scoring.
- `NLI_MODEL_CLUSTER=microsoft/deberta-large-mnli` for stage 2 answer clustering.

Override either value inline when needed:

```bash
NLI_MODEL_CLUSTER=microsoft/deberta-large-mnli bash src/scripts/run_stage.sh 2
NLI_MODEL_SUPPORT=potsawee/deberta-v3-large-mnli bash src/scripts/run_stage.sh 3
```

## Output Artifacts

Each run writes to `results/{run_setting}/`.

Core pipeline:

- `generations.pkl`
- `semantic_clusters.pkl`
- `support_uncertainty_luqpair_{split_method}.json`
- `support_uncertainty_luq_{split_method}.json`
- `confidence_keywords_probs.json`
- `AUROC_labels.json`

Baselines:

- `predictive_entropy_scores.pkl`
- `len_normed_predictive_entropy_scores.pkl`
- `semantic_entropy_scores.pkl`
- `self_consistency_scores.pkl`
- `sentence_sar_scores.pkl`
- `token_sar_scores.pkl`
- `luq_scores.pkl`
- `cotuq_scores.pkl` (after stage 6c)

Final table:

- `results/final_uq_baseline_compare.csv`

Final table columns:

| Column | Meaning |
| :--- | :--- |
| `model` | Model name parsed from `run_setting`. |
| `dataset` | Dataset name parsed from `run_setting`. |
| `predictive-entropy-from-generations` | PE baseline. |
| `len-normed-predictive-entropy-from-generations` | LN-PE baseline. |
| `sentence-sar-from-generations` | Sentence-SAR baseline. |
| `token-sar-from-generations` | Token-SAR baseline. |
| `semantic-entropy-from-generations` | Semantic Entropy baseline. |
| `SC` | Self-Consistency baseline. |
| `LUQ` | LUQ baseline. |
| `cotuq` | CoT-UQ baseline, available after stage 6c. |
| `combined_scores` | CoSu-UQ fused uncertainty. |
| `confidence_level` | CoSu-UQ confidence-only uncertainty. |
| `support_level` | CoSu-UQ support-only uncertainty. |

## Evaluation Protocol

The primary metric is AUROC for separating correct from incorrect final answers.

In the paper, correctness labels are obtained by majority voting among three independent LLM judges:

- GPT-4.1-mini
- DeepSeek-V3.2-Fast
- Grok-4-1-Fast-Reasoning

The paper reports 98%-99% agreement between these automatic labels and human annotations. `JUDGERS` must be configured explicitly to match this setup; the Python fallback in `judge_responses.py` uses a different five-judge list.

By default, `final_compare.py` uses `most_cluster_label` from `AUROC_labels.json`:

- `most_cluster_label` is the label of the highest-probability response in the dominant semantic answer cluster.
- The current code selects the dominant cluster by summed response probability. The paper describes selecting the largest semantic cluster; these criteria can differ when cluster sizes and probability masses disagree.
- `calc_auroc` calls `roc_auc_score(1 - labels, scores)`, so labels are expected to be binary (`1` = correct, `0` = incorrect) and higher uncertainty should rank incorrect answers higher.
- If `final_judge_result_labels` contains `-1` for an item, the AUROC calculation for that run may return `nan`. Make sure stage 5j produced valid binary labels.

## Main Results

The following table reproduces Table I of the paper (AUROC). Best and second-best markers are omitted here for readability; see the paper for the exact bold/dagger annotations.

S-SAR and T-SAR denote SentenceSAR and TokenSAR. SC denotes Self-Consistency, computed from the mass of the largest entailment cluster.

| Model | Dataset | PE | LN-PE | S-SAR | T-SAR | SE | SC | LUQ | CoT-UQ | CoSu-UQ |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Llama-3.1-8B | GSM8K | 0.8314 | 0.8200 | 0.5698 | 0.8199 | 0.8201 | 0.7815 | 0.7953 | 0.7536 | 0.8261 |
| Llama-3.1-8B | MATH | 0.8499 | 0.7805 | 0.7843 | 0.7805 | 0.7723 | 0.8867 | 0.8881 | 0.7967 | 0.9066 |
| Llama-3.1-8B | HotpotQA | 0.7609 | 0.7776 | 0.7741 | 0.7776 | 0.7492 | 0.8784 | 0.8251 | 0.7522 | 0.8408 |
| Llama-3.1-8B | 2WikiMultiHopQA | 0.6811 | 0.7296 | 0.7471 | 0.7296 | 0.6120 | 0.8589 | 0.6710 | 0.7479 | 0.8004 |
| Llama-3.1-8B | MedQA | 0.7022 | 0.6741 | 0.6370 | 0.6741 | 0.6719 | 0.8198 | 0.7732 | 0.6498 | 0.8209 |
| Qwen3-4B | GSM8K | 0.8344 | 0.8087 | 0.5664 | 0.8088 | 0.8091 | 0.7338 | 0.8174 | 0.7939 | 0.8612 |
| Qwen3-4B | MATH | 0.9174 | 0.8578 | 0.6241 | 0.8579 | 0.8653 | 0.8733 | 0.8944 | 0.8469 | 0.9580 |
| Qwen3-4B | HotpotQA | 0.6852 | 0.6798 | 0.5612 | 0.6799 | 0.6947 | 0.8632 | 0.8315 | 0.6343 | 0.8653 |
| Qwen3-4B | 2WikiMultiHopQA | 0.6662 | 0.6826 | 0.4887 | 0.6823 | 0.6157 | 0.8146 | 0.7726 | 0.7337 | 0.8335 |
| Qwen3-4B | MedQA | 0.6648 | 0.6233 | 0.5838 | 0.6234 | 0.6446 | 0.6576 | 0.6740 | 0.5881 | 0.7093 |
| Qwen3-14B | GSM8K | 0.8534 | 0.8116 | 0.6791 | 0.8113 | 0.8113 | 0.6804 | 0.8717 | 0.7474 | 0.8921 |
| Qwen3-14B | MATH | 0.9262 | 0.8300 | 0.5877 | 0.8302 | 0.8296 | 0.9114 | 0.9232 | 0.8801 | 0.9865 |
| Qwen3-14B | HotpotQA | 0.6489 | 0.6748 | 0.6494 | 0.6747 | 0.6802 | 0.8391 | 0.8133 | 0.6316 | 0.8177 |
| Qwen3-14B | 2WikiMultiHopQA | 0.6941 | 0.7839 | 0.6584 | 0.7838 | 0.7656 | 0.8189 | 0.7224 | 0.7881 | 0.8247 |
| Qwen3-14B | MedQA | 0.6875 | 0.7009 | 0.6747 | 0.7009 | 0.6992 | 0.7093 | 0.6704 | 0.6614 | 0.7407 |
| Qwen3-32B | GSM8K | 0.8839 | 0.8858 | 0.7984 | 0.8864 | 0.8864 | 0.7620 | 0.8603 | 0.8569 | 0.9465 |
| Qwen3-32B | MATH | 0.9210 | 0.8253 | 0.7193 | 0.8249 | 0.8211 | 0.8188 | 0.8593 | 0.8318 | 0.9732 |
| Qwen3-32B | HotpotQA | 0.6702 | 0.6874 | 0.6775 | 0.6875 | 0.6667 | 0.8606 | 0.8423 | 0.6623 | 0.8736 |
| Qwen3-32B | 2WikiMultiHopQA | 0.5826 | 0.7219 | 0.6443 | 0.7220 | 0.7505 | 0.8431 | 0.8260 | 0.7323 | 0.8452 |
| Qwen3-32B | MedQA | 0.7223 | 0.7177 | 0.6829 | 0.7177 | 0.7189 | 0.8276 | 0.7858 | 0.6777 | 0.8587 |

### Runtime

On the paper hardware (two A100-SXM4-80GB GPUs, Intel Xeon Platinum 8336C), the paper reports the following costs under its profiling protocol, described as 300 sampled responses per dataset:

- Support branch: approximately 4 h 32 min.
- Answer clustering: approximately 3.54 s.
- Confidence branch with keyword filtering: approximately 4 min 50 s.
- Confidence branch with direct aggregation: approximately 10 s.

The support branch dominates end-to-end runtime.

## Paper vs. This Repository

The code follows the paper's main pipeline, but a few implementation details differ. These are intentionally documented rather than hidden:

| Item | Paper | This repository |
| :--- | :--- | :--- |
| Self-Consistency baseline | Reported in Table I. | Implemented in stage 6b as `1 - largest entailment-cluster mass`; stage 7 reports it as `SC`. |
| Judge models | GPT-4.1-mini, DeepSeek-V3.2-Fast, Grok-4-1-Fast-Reasoning (majority vote). | `JUDGERS` must be configured manually. If empty, `judge_responses.py` falls back to five different judges. |
| Clustering NLI model | `microsoft/deberta-large-mnli`; support uses `potsawee/deberta-v3-large-mnli`. | Matches via separate `NLI_MODEL_CLUSTER` and `NLI_MODEL_SUPPORT` settings. |
| Representative selection | Highest-probability response in the largest semantic cluster. | Highest-probability response in the cluster with the largest summed response probability. |
| Top-k / top-p | Not specified in the main paper text. | `TOP_K=5`, `TOP_P=0.95` by default. Verify these match your intended reproduction setting. |

## API-Based Components

### Stage 1 generation API

Stage 1 can use a DashScope-compatible generation API instead of a local model:

```bash
USE_API=true \
API_KEY="..." \
API_BASE="..." \
API_MODEL_NAME="..." \
bash src/scripts/run_stage.sh 1
```

The API path uses `dashscope.Generation.call` and requires logprobs. The local tokenizer specified by `TOKENIZER_MODEL_DIR` is still used to align generated text with token probabilities.

### Stage 5j judge API

Stage 5j uses OpenAI-compatible chat-completion endpoints. `JUDGERS` is a space-separated list; each item contains five `[split]`-separated fields:

```text
model[split]api_base[split]api_key[split]temperature[split]top_p
```

Example:

```bash
export OPENAI_API_KEY="..."
export JUDGERS="gpt-4.1-mini[split]https://api.openai.com/v1[split]${OPENAI_API_KEY}[split]0.5[split]0.8"
```

For a paper-style three-judge setup, add one item per endpoint, separated by spaces. Do not hardcode keys in code or commit them to the repository.

If `JUDGERS` is empty, `run_stage.sh` does not pass `--judgers`, and `judge_responses.py` uses its Python fallback list. That fallback requires `OPENAI_API_BASE`, `OPENAI_API_KEY`, `QWEN_API_BASE`, `QWEN_API_KEY`, `YI_API_BASE`, `YI_API_KEY`, `DOUBAO_API_BASE`, and `DOUBAO_API_KEY` to be set for the corresponding entries.

## Reproducibility Checklist

- Build datasets with `NUM_SAMPLES=500`.
- Keep `FRACTION=0.6` to use 300 samples per dataset.
- Use `NUM_GENERATIONS=5` and `TEMPERATURE=1.0`.
- Keep `NLI_MODEL_SUPPORT=potsawee/deberta-v3-large-mnli` and `NLI_MODEL_CLUSTER=microsoft/deberta-large-mnli`.
- Configure `JUDGERS` with the three paper judges and majority voting.
- Run stages 6b and 6c before stage 7.
- Keep one consistent `run_setting` across all stages.
- Compare the resulting CSV against [Main Results](#main-results).

## Troubleshooting

**Stage 1 cannot find `DATA_FILE`**

Run stage 0 first, or point `DATA_FILE` to an existing JSONL file. `DATA_FILE` is not updated automatically when `NUM_SAMPLES` changes.

**CUDA out of memory**

Use a smaller model, reduce `MAX_LENGTH`, or set `DEVICE` to an available GPU. The paper setting uses 80GB A100 GPUs; smaller GPUs may require model-specific memory optimizations.

**Stage 5j reports missing API configuration**

Set `JUDGERS` explicitly with non-empty `api_base` and `api_key` fields, or export the environment variables expected by the fallback judge list.

**Final AUROC is `nan`**

Check that `AUROC_labels.json` and the judge labels are binary. Labels of `-1` or missing label files cause the AUROC computation to fail or return `nan`.

**Model row is missing or unranked in the final CSV**

`final_compare.py` parses `model` from the first underscore-separated field of `run_setting`. Avoid underscores in `MODEL_NAME`, and update the model ordering list if you use a name not covered by the defaults.

## Citation

```bibtex
@INPROCEEDINGS{11730028,
  author={Feng, Yinglun and Lin, Qinhong and Zhang, Yuhao and Yang, Zhongliang and Zhou, Linna},
  booktitle={2026 International Joint Conference on Neural Networks (IJCNN)},
  title={CoSu-UQ: Uncertainty Quantification for Reasoning via Confidence and Semantic Support},
  year={2026},
  pages={1-8},
  doi={10.1109/IJCNN69854.2026.11730028}
}
```

IEEE Xplore: [https://ieeexplore.ieee.org/abstract/document/11730028](https://ieeexplore.ieee.org/abstract/document/11730028)

## Acknowledgements

This work was supported in part by the Research Project of Quan Cheng Laboratory, China (Grant No. QCL20250203), and in part by the National Natural Science Foundation of China under Grant 62172053 and Grant 62302059.

## License

This project is released under the MIT License. Copyright (c) 2026 Yinglun Feng. See [`LICENSE`](LICENSE) for the full text.

## Contact

- Yinglun Feng: 1353910719@qq.com
- GitHub issues: [github.com/interments/CoSu-UQ/issues](https://github.com/interments/CoSu-UQ/issues)
