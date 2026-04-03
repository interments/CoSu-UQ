# Reorganization Mapping (Old -> New)

## Core pipeline
- `src/pipeline/build_cot_uq_datasets.py` -> `src/pipeline/stage0_build_datasets.py`
  - Logic: build unified cleaned datasets for GSM8K/MATH/HotpotQA/2Wiki/MedQA.
- `src/pipeline/Generation_Cot_UQ.py` -> `src/pipeline/stage1_generate_cot.py`
  - Logic: CoT multi-sampling generation and save `generations.pkl`.
- `src/pipeline/Get_semantic_clusters_llama_merge.py` -> `src/pipeline/stage2_semantic_cluster.py`
  - Logic: semantic clustering with NLI; save `semantic_clusters.pkl`.
- `src/pipeline/LUQ.py` -> `src/pipeline/stage3_compute_support.py`
  - Logic: LUQ/LUQPair support uncertainty scoring; save `LUQ*_splited_results.json`.
- `src/pipeline/Get_LUQPair_keywords_probs.py` -> `src/pipeline/stage4_extract_confidence.py`
  - Logic: extract keyword token probabilities for confidence signal; save `LUQPair_keywords_probs.json`.

## Evaluation
- `src/pipeline/Judge_results_all_Cot_UQ.py` -> `src/eval/judge_responses.py`
  - Logic: judge sampled/greedy outputs and write `final_judge_result_labels` into `generations.pkl`.
- `src/pipeline/Get_AUROC_labels.py` -> `src/eval/build_auroc_labels.py`
  - Logic: build `AUROC_labels.json` (`greedy_label`, `most_cluster_label`, `most_sampled_label`).
- `src/final_uq_scoring/final_score_compare.py` -> `src/eval/final_compare.py`
  - Logic: compute CoSu-UQ GU/SU/Combined + baseline AUROC comparison table.

## Baselines
- `src/baseline_cache_minimal.py` -> `src/baselines/build_baseline_scores.py`
  - Logic: PE/LN-PE/SE/Sentence-SAR/Token-SAR/LUQ score pkl build.
- `src/pipeline/Get_sentence_similarities.py` -> `src/baselines/cache_sentence_similarity.py`
  - Logic: cache sentence-level pairwise similarity for Sentence-SAR.
- `src/pipeline/Get_tokenwise_importance_from_generation.py` -> `src/baselines/cache_token_importance.py`
  - Logic: cache token-wise importance for Token-SAR.

## Shared modules
- `src/final_uq_scoring/config.py` -> `src/config.py`
  - Logic: project paths and runtime config constants.
- `src/final_uq_scoring/API_chat.py` -> `src/utils/api_chat.py`
  - Logic: common chat/embedding API client wrappers.
- `src/pipeline/Cot_uq_utils.py` -> `src/utils/cot_uq_utils.py`
  - Logic: utility functions for CoT parsing/token aggregation used by multiple pipeline scripts.
- `src/pipeline/Get_step_exact_tokens.py` -> `src/utils/get_step_exact_tokens.py`
  - Logic: prompt/template helper for step-level keyword extraction flow.

## Documentation
- `data_fields_description.md` -> `src/docs/data_fields.md`
- `src/pipeline/PIPELINE_STAGE_NOTES.md` -> `src/docs/pipeline_stages.md`
- `src/final_uq_scoring/METHODOLOGY_FINAL_SCORING.md` -> `src/docs/final_scoring_methodology.md`

## New shell scripts
- `src/scripts/common.env`: central configurable variables.
- `src/scripts/run_stage.sh`: run a single pipeline stage.
- `src/scripts/run_pipeline.sh`: orchestrate full pipeline with stage toggles.
- `src/scripts/run_batch.sh`: run multiple `run_setting` values from a file.
