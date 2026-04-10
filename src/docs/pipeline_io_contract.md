# Pipeline Execution & IO Contract

This document describes the end-to-end execution flow of the current CoSu-UQ project, including:
- which script runs at each stage,
- required inputs,
- produced outputs,
- output data formats,
- and how each output is consumed by downstream stages.

## 0. Orchestration Entry

Primary orchestrator:
- `src/scripts/run_stage.sh`

Supported stage ids:
- `0 1 2 3 4 5j 5l 6s 6t 6b 6c 7`

Full pipeline runner:
- `src/scripts/run_pipeline.sh`

Batch runner (multiple `run_setting` values):
- `src/scripts/run_batch.sh`

`run_setting` naming is derived in `run_stage.sh` from stage-1 generation config if `RUN_SETTING` is not explicitly set.

---

## 1) Stage 0: Dataset Build

Script:
- `python -m pipeline.stage0_build_datasets`

Inputs:
- HuggingFace datasets (`gsm8k`, `hotpotqa`, `math`, `2wiki`, `medqa`) and options from `common.env`.

Outputs:
- Cleaned dataset jsonl files under `datasets/`, e.g.:
  - `gsm8k_cot_uq_validation_0-500_src.jsonl`

Main format (per jsonl line):
- A single QA sample adapted into CoT prompting format.

Consumed by:
- Stage 1 (`--data_file`).

---

## 2) Stage 1: CoT Multi-Sampling Generation

Script:
- `python -m pipeline.stage1_generate_cot`

Inputs:
- Dataset jsonl from Stage 0.
- Generation config (`MODEL_DIR`, `NUM_GENERATIONS`, `TEMPERATURE`, etc.).

Outputs:
- `results/{run_setting}/generations.pkl`

Main format (list of dict):
- Each item is one sample with keys such as:
  - `id`, `question`, `prompt_text`, `prompt`
  - `generated_texts`: list[str] (sampled responses)
  - `generated_probs`: list[list[float]] (token probabilities per sampled response)
  - `generated_ids`: token id tensor for sampled responses
  - `generated_success_flag`: list[bool]
  - `cleaned_generated_texts`, `cleaned_generated_ids`
  - `most_likely_generation`, `most_likely_generation_probs`, `most_likely_generation_ids`

Consumed by:
- Stage 2, 3, 4, 5j, 5l, 6s, 6t.

---

## 3) Stage 2: Semantic Clustering

Script:
- `python -m pipeline.stage2_semantic_cluster`

Inputs:
- `results/{run_setting}/generations.pkl`

Outputs:
- `results/{run_setting}/semantic_clusters.pkl`

Main format (dict keyed by sample id):
- `semantic_set_ids`: cluster ids for cleaned responses
- `semantic_set_ids_raw`: cluster ids for raw responses
- `semantic_set_ids_entailment`: bidirectional-entailment cluster ids

Consumed by:
- Stage 4 (`semantic_set_ids`)
- Stage 5l (`semantic_set_ids_entailment`)
- Stage 7 (cluster-based uncertainty aggregation)
- Stage 6b (semantic entropy baseline)

---

## 4) Stage 3: Support Signal (LUQ family)

Script:
- `python -m pipeline.stage3_compute_support`

Typical config:
- Manual single-method run: `--luq_method LUQPair --split_method step_answer` (or `LUQ`)
- `run_stage.sh` stage `3` runs both methods (`LUQPair` + `LUQ`) by default.

Inputs:
- `results/{run_setting}/generations.pkl`

Outputs:
- Method-specific files:
  - `results/{run_setting}/support_uncertainty_luqpair_{split_method}.json` (CoSu-UQ support branch)
  - `results/{run_setting}/support_uncertainty_luq_{split_method}.json` (LUQ baseline)

Main format (list of dict):
- `id`, `prompt_text`, `answer`
- `generated_texts`, `responses`
- `splited_responses`: segmented reasoning units
- `uncertainty_scores`: per-response support uncertainty
- `score`: sample-level support uncertainty
- optional `nli_probability_matrix` if `--save_matrix`

Consumed by:
- Stage 4 (uses sentence segmentation metadata)
- Stage 6b (`luq_scores.pkl` builder reads `support_uncertainty_luq_{split_method}.json`)
- Stage 7 (SU branch reads `uncertainty_scores` and converts to support information)

---

## 5) Stage 4: Confidence Signal (Keyword Probabilities)

Script:
- `python -m pipeline.stage4_extract_confidence`

Inputs:
- `results/{run_setting}/support_uncertainty_luqpair_{split_method}.json`
- `results/{run_setting}/generations.pkl`
- `results/{run_setting}/semantic_clusters.pkl`

Outputs:
- `results/{run_setting}/confidence_keywords_probs.json`

Main format (list of dict):
- `id`, `question`, `semantic_set_ids`
- `responses_data`: list over responses, each containing per-sentence entries:
  - `sentence`
  - `keywords_probs`: `{keyword: [prob, ...]}`
  - `keywords_start_end_string_from_sentence`
  - `keywords_list`
  - `sentence_string_position_in_response`
  - `sentence_token_range`
- `responses`, `prompt_text`, `answer`, `splited_responses`, `generated_success_flag`

Consumed by:
- Stage 7 (GU/combined confidence branch).

---

## 6) Stage 5j: Judge Labels Backfill

Script:
- `python -m eval.judge_responses`

Inputs:
- `results/{run_setting}/generations.pkl`
- `--judgers` config (supports per-model API base/key, including `${ENV_VAR}` placeholders)

Outputs:
- Writes back into same file:
  - `results/{run_setting}/generations.pkl`

New/updated fields in each sample:
- `<model>_judge_result_labels`
- `final_judge_result_labels`

Consumed by:
- Stage 5l (AUROC label construction).

---

## 7) Stage 5l: AUROC Label Build

Script:
- `python -m eval.build_auroc_labels`

Inputs:
- `results/{run_setting}/generations.pkl` (must include `final_judge_result_labels`)
- `results/{run_setting}/semantic_clusters.pkl`

Outputs:
- `results/{run_setting}/AUROC_labels.json`

Main format (list of dict):
- `id`
- `greedy_label`
- `most_sampled_label`
- `most_cluster_label`  (used as default target label in Stage 7)

Consumed by:
- Stage 7 final AUROC evaluation.

---

## 8) Stage 6s/6t/6b: Baseline Cache + Score Build

### Stage 6s
Script:
- `python -m baselines.cache_sentence_similarity`

Output:
- `results/{run_setting}/sentence_similarities_{model_key}.pkl`

### Stage 6t
Script:
- `python -m baselines.cache_token_importance`

Output:
- `results/{run_setting}/tokenwise_importance_{model_key}_from_generation.pkl`

### Stage 6b
Script:
- `python -m baselines.build_baseline_scores`

Inputs:
- `generations.pkl`, `semantic_clusters.pkl`, cached sentence similarity + token importance
- Support uncertainty file via (`--luq_split_method`) -> `support_uncertainty_luq_{split_method}.json`

Outputs:
- `predictive_entropy_scores.pkl`
- `len_normed_predictive_entropy_scores.pkl`
- `semantic_entropy_scores.pkl`
- `sentence_sar_scores.pkl`
- `token_sar_scores.pkl`
- `luq_scores.pkl`

Format for all `*_scores.pkl` files:
- list of sample-level scores (`torch.Tensor` scalar per sample)

Consumed by:
- Stage 7 baseline comparison.

---

## 9) Stage 6c: CoT-UQ Baseline (migrated to baselines)

Stage 6c runs two scripts in sequence.

### 6c-1 Keyword extraction
Script:
- `python -m baselines.cotuq_keyword_extraction`

Input:
- `results/{run_setting}/generations.pkl`

Output:
- `results/{run_setting}/keywords_probabilities_{prompt_type}_{sampled|greedy}.json`

Main format (list of dict):
- `id`, `question`
- `responses`: per sampled path:
  - `response_idx`
  - `keywords_probabilities`
  - `keywords_contributions`
  - `cot_uq_success_flag`

### 6c-2 Score aggregation
Script:
- `python -m baselines.cotuq_aggregate_scores`

Input:
- `keywords_probabilities_{prompt_type}_{sampled|greedy}.json`

Output:
- `results/{run_setting}/Cot_uq_{aggregated_method}_scores_{prompt_type}_{sampled|greedy}.pkl`
- `results/{run_setting}/cotuq_scores.pkl` (canonical CoT-UQ baseline score file)

Format:
- list of sample-level CoT-UQ uncertainty scores (`torch.Tensor` scalar per sample)

Consumed by:
- Stage 7 (`final_compare.py` auto-detects known `Cot_uq_*` score filenames).

---

## 10) Stage 7: Final CoSu-UQ + Baselines AUROC

Script:
- `python -m eval.final_compare`

Inputs:
- CoSu-UQ core files:
  - `confidence_keywords_probs.json`
  - `support_uncertainty_luqpair_{split_method}.json`
  - `semantic_clusters.pkl`
  - `generations.pkl`
  - `AUROC_labels.json`
- Baseline score PKLs from Stage 6b
- Optional CoT-UQ score PKL from Stage 6c

Outputs:
- CSV summary, default:
  - `results/final_uq_baseline_compare.csv`

Main output columns:
- `model`, `dataset`
- Baseline AUROC columns: PE, LN-PE, sentence-sar, token-sar, SE, LUQ, optional CoT-UQ
- CoSu-UQ AUROC columns: `confidence_level`, `support_level`, `combined_scores`

---

## Dependency Graph (Practical)

- Stage 0 -> Stage 1
- Stage 1 -> Stage 2/3/4/5j/6s/6t
- Stage 2 + Stage 5j -> Stage 5l
- Stage 3 + Stage 1 + Stage 2 -> Stage 4
- Stage 1 + Stage 2 + Stage 6s + Stage 6t + Stage 3 -> Stage 6b
- Stage 1 -> Stage 6c-1 -> Stage 6c-2
- Stage 4 + Stage 3 + Stage 2 + Stage 1 + Stage 5l + Stage 6b (+ optional Stage 6c) -> Stage 7

---

## Notes on Contract Stability

- `run_setting` must stay consistent across all stages for file discovery.
- Stage 3 writes method-specific support outputs for CoSu-UQ (`luqpair`) and LUQ baseline (`luq`).
- Stage 4/7 consume `support_uncertainty_luqpair_{split_method}.json`.
- Stage 6b consumes `support_uncertainty_luq_{split_method}.json`.
