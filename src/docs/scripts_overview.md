# Scripts Directory Overview

This document explains what each shell script in `src/scripts` does and when to use it.

## `common.env`
Purpose:
- Central runtime config for the shell orchestration layer.
- Defines project paths, stage parameters, model settings, judge settings, and output settings.

What it controls:
- Stage 0: dataset build options (`DATASETS`, `NUM_SAMPLES`, ...)
- Stage 1: generation options (`MODEL_DIR`, `NUM_GENERATIONS`, `USE_API`, ...)
- Stage 2/3/4: NLI/splitting/confidence extraction options
- Stage 5: judge and AUROC label options
- Stage 6: baseline cache/scoring options
- Stage 7: final compare options

Typical usage:
- Edit values in this file before running scripts.
- Or override any variable inline at runtime, for example:

```bash
RUN_SETTING="..." DEVICE="1" ./src/scripts/run_stage.sh 3
```

---

## `run_stage.sh`
Purpose:
- Run exactly one pipeline stage by stage ID.

Usage:

```bash
./src/scripts/run_stage.sh <stage>
```

Supported stage IDs:
- `0`: Build cleaned datasets (`pipeline.stage0_build_datasets`)
- `1`: Generate CoT samples (`pipeline.stage1_generate_cot`)
- `2`: Semantic clustering (`pipeline.stage2_semantic_cluster`)
- `3`: Compute support signal/LUQ variants (`pipeline.stage3_compute_support`)
- `4`: Extract confidence signal (`pipeline.stage4_extract_confidence`)
- `5j`: Run multi-judge labeling (`eval.judge_responses`)
- `5l`: Build AUROC labels (`eval.build_auroc_labels`)
- `6s`: Cache sentence similarity for SAR (`baselines.cache_sentence_similarity`)
- `6t`: Cache token importance for Token-SAR (`baselines.cache_token_importance`)
- `6b`: Build baseline score PKLs (`baselines.build_baseline_scores`)
- `6c`: Build CoT-UQ baseline scores (`pipeline.Keywords_extraction_and_scoring` + `pipeline.Aggregated_probs`)
- `7`: Final AUROC comparison (`eval.final_compare`)

Implementation notes:
- Auto-derives `RUN_SETTING` from generation config when `RUN_SETTING` is empty.
- Converts boolean-like env values using `str_to_bool`.
- For stage `1`, if `USE_API=true`, it forwards API args to generation.
- For stage `5j`, if `JUDGERS` is non-empty, it forwards custom judger configs.

---

## `run_pipeline.sh`
Purpose:
- Run the full multi-stage pipeline in order (or a selected subset via toggles).

Usage:

```bash
./src/scripts/run_pipeline.sh
```

How it works:
- Reads stage toggles `DO_STAGE0 ... DO_STAGE7` (with defaults in the script).
- Calls `run_stage.sh` sequentially for enabled stages.

Default behavior in current script:
- Stage 0 is off by default.
- Stage 1 to Stage 7 are on by default.

Typical pattern:

```bash
DO_STAGE0=true DO_STAGE5J=false ./src/scripts/run_pipeline.sh
```

---

## `run_batch.sh`
Purpose:
- Batch-run the pipeline for multiple precomputed `run_setting` values.

Usage:

```bash
./src/scripts/run_batch.sh <run_settings_file>
```

Input file format:
- One `run_setting` per line.
- Empty lines are ignored.
- Lines starting with `#` are treated as comments.

How it works:
- Iterates each run setting.
- Launches `run_pipeline.sh` with `RUN_SETTING` overridden per line.

Example `run_settings.txt`:

```text
# qwen runs
Qwen3-4B_gsm8k_cot_uq_validation_0-500_src_fraction_0.6_max_length_512_num_generations_5_temperature_1.0_top_k_5_top_p_0.95_decode_method_greedy_seed_42
Qwen3-14B_gsm8k_cot_uq_validation_0-500_src_fraction_0.6_max_length_512_num_generations_5_temperature_1.0_top_k_5_top_p_0.95_decode_method_greedy_seed_42
```

---

## Quick Selection Guide
- Want to run just one step while debugging: use `run_stage.sh`.
- Want end-to-end execution for one config: use `run_pipeline.sh`.
- Want end-to-end execution for many run settings: use `run_batch.sh`.
- Want to change defaults globally: edit `common.env`.
