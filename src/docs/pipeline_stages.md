# CoSu-UQ Pipeline Stages (Reorganized)

## Stage 0: Unified Dataset Build
- Script: `src/pipeline/stage0_build_datasets.py`
- Output examples:
  - `datasets/gsm8k_cot_uq_validation_0-500_src.jsonl`
  - `datasets/hotpotqa_cot_uq_validation_0-500_src.jsonl`
  - `datasets/math_cot_uq_validation_0-500.jsonl`
  - `datasets/2WikimultihopQA_cot_uq_validation_0-500_src.jsonl`
  - `datasets/MedQA_cot_uq_validation_0-500_src.jsonl`

## Stage 1: CoT Multi-Sampling
- Script: `src/pipeline/stage1_generate_cot.py`
- Input: cleaned dataset jsonl
- Main output: `results/{run_setting}/generations.pkl`

## Stage 2: Semantic Clustering
- Script: `src/pipeline/stage2_semantic_cluster.py`
- Input: `results/{run_setting}/generations.pkl`
- Main output: `results/{run_setting}/semantic_clusters.pkl`

## Stage 3: Support Signal (LUQPair)
- Script: `src/pipeline/stage3_compute_support.py`
- Main outputs:
  - `results/{run_setting}/support_uncertainty_luqpair_{split_method}.json`
  - `results/{run_setting}/support_uncertainty_luq_{split_method}.json`

## Stage 4: Confidence Signal (Keyword Token Probs)
- Script: `src/pipeline/stage4_extract_confidence.py`
- Inputs:
  - `results/{run_setting}/support_uncertainty_luqpair_{split_method}.json`
  - `results/{run_setting}/generations.pkl`
  - `results/{run_setting}/semantic_clusters.pkl`
- Main output: `results/{run_setting}/confidence_keywords_probs.json`

## Stage 5a: Judge Labels Backfill
- Script: `src/eval/judge_responses.py`
- Input: `results/{run_setting}/generations.pkl`
- Output: updates `final_judge_result_labels` in `generations.pkl`

## Stage 5b: AUROC Label Build
- Script: `src/eval/build_auroc_labels.py`
- Inputs:
  - `results/{run_setting}/generations.pkl`
  - `results/{run_setting}/semantic_clusters.pkl`
- Main output: `results/{run_setting}/AUROC_labels.json`

## Stage 6: Baseline Cache + Score Build
- Scripts:
  - `src/baselines/cache_sentence_similarity.py`
  - `src/baselines/cache_token_importance.py`
  - `src/baselines/build_baseline_scores.py`
- Main outputs:
  - `predictive_entropy_scores.pkl`
  - `len_normed_predictive_entropy_scores.pkl`
  - `semantic_entropy_scores.pkl`
  - `sentence_sar_scores.pkl`
  - `token_sar_scores.pkl`
  - `luq_scores.pkl`
  - `cotuq_scores.pkl` (produced by stage `6c`)

## Stage 7: Final CoSu-UQ + Baseline AUROC Compare
- Script: `src/eval/final_compare.py`
- Consumes all stage outputs above
- Main output: final AUROC comparison CSV

## Shell Orchestration
- `src/scripts/common.env`: all configurable variables
- `src/scripts/run_stage.sh`: run a single stage
- `src/scripts/run_pipeline.sh`: run full or partial pipeline
- `src/scripts/run_batch.sh`: run multiple run settings from a file
