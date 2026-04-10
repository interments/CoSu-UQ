# CoSu-UQ: Uncertainty Quantification for Reasoning via Confidence and Semantic Support

This repository contains the official implementation of **CoSu-UQ**, a multi-sample uncertainty quantification framework for long-form reasoning in large language models.

CoSu-UQ combines:
- **Confidence signal**: token-level generation probabilities aggregated to response/sentence-level confidence.
- **Semantic support signal**: cross-sample sentence-level entailment structure over reasoning paths.

These signals are fused and aggregated at the final-answer semantic cluster level for robust uncertainty estimation.


## Highlights
- End-to-end pipeline from dataset build to AUROC comparison.
- Unified shell orchestration for single-stage, full pipeline, and batch execution.
- Baseline suite: PE, LN-PE, SE, Sentence-SAR, Token-SAR, LUQ, CoT-UQ.
- Reproducible CoSu-UQ signal decomposition:
  - `confidence_level`
  - `support_level`
  - `combined_scores`

## Repository Structure
```text
CoSu_UQ_src/
├── datasets/                     # processed dataset files
├── models/                       # model download helpers and local cache root
├── results/                      # per-run outputs
├── src/
│   ├── pipeline/                 # stage0-stage4 core pipeline
│   ├── baselines/                # baseline cache/scoring + CoT-UQ + LUQ calculators
│   ├── eval/                     # judge labels, AUROC labels, final comparison
│   ├── scripts/                  # orchestration shell scripts
│   ├── docs/                     # IO contracts and stage docs
│   ├── utils/                    # shared utilities
│   └── requirements.txt
└── README.md
```

## Method-to-File Contract (Important)

To avoid method leakage between CoSu-UQ and LUQ baseline, stage 3 writes **two distinct support files**:

- `support_uncertainty_luqpair_{split_method}.json`  
  Used by CoSu-UQ main branch (stage 4 + stage 7).
- `support_uncertainty_luq_{split_method}.json`  
  Used only for LUQ baseline scoring (`luq_scores.pkl` in stage 6b).

Confidence extraction output is standardized as:

- `confidence_keywords_probs.json`

This separation is required for faithful reproduction of the paper setup.

## Installation

Recommended Python: **3.10+**

```bash
cd CoSu_UQ_src
python -m venv .venv
source .venv/bin/activate
pip install -r src/requirements.txt
```

## Runtime Configuration

Main config file:
- [`src/scripts/common.env`](src/scripts/common.env)

You should set at least:
- `MODEL_DIR` (local generation model path)
- `MODEL_NAME`
- `DATA_FILE` (for stage 1)
- `DEVICE`

Optional environment overrides (see [`src/config.py`](src/config.py)):
- `COSU_UQ_RUN_DIR`
- `COSU_UQ_DATA_DIR`
- `COSU_UQ_OUTPUT_DIR`
- `COSU_UQ_HF_MODEL_CACHE`
- `HF_DATASETS_CACHE`, `TRANSFORMERS_CACHE`, `HF_HOME`

## Quick Start

Run the full pipeline (default: stages 1-7 enabled, stage 0 disabled):

```bash
cd CoSu_UQ_src
bash src/scripts/run_pipeline.sh
```

Run a single stage:

```bash
bash src/scripts/run_stage.sh 3
```

Batch multiple `run_setting` values:

```bash
bash src/scripts/run_batch.sh <run_settings_file>
```

## Pipeline Stages

Supported stage IDs:
- `0`: dataset build
- `1`: CoT multi-sampling generation
- `2`: semantic clustering
- `3`: support uncertainty (runs `LUQPair` then `LUQ`)
- `4`: confidence keyword probability extraction
- `5j`: judge labels backfill
- `5l`: AUROC label build
- `6s`: sentence similarity cache
- `6t`: token importance cache
- `6b`: baseline score build
- `6c`: CoT-UQ baseline build
- `7`: final AUROC comparison

For detailed I/O schemas:
- [`src/docs/pipeline_io_contract.md`](src/docs/pipeline_io_contract.md)
- Chinese version: [`src/docs/pipeline_io_contract_zh.md`](src/docs/pipeline_io_contract_zh.md)

## Main Output Artifacts (per `results/{run_setting}/`)

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
- `sentence_sar_scores.pkl`
- `token_sar_scores.pkl`
- `luq_scores.pkl`
- `cotuq_scores.pkl` (after stage 6c)

Final table:
- `results/final_uq_baseline_compare.csv`

## Evaluation Protocol

`src/eval/final_compare.py` reports AUROC for:
- Baselines: PE, LN-PE, Sentence-SAR, Token-SAR, SE, LUQ, CoT-UQ
- CoSu-UQ: `confidence_level`, `support_level`, `combined_scores`

By default, labels use `most_cluster_label` from `AUROC_labels.json`.

## API-Based Components

Two parts may require API configuration:
- Stage 1 (`--use_api`) generation path (optional)
- Stage `5j` judge labeling (`eval.judge_responses`)

`JUDGERS` format is documented in [`src/scripts/common.env`](src/scripts/common.env).  
Do not hardcode keys in code; use environment variables.

## Reproducibility Notes

- Keep a consistent `run_setting` across stages.
- Use identical decoding and sampling settings when comparing methods.
- Stage 3 must produce both LUQPair and LUQ support files before stages 6b/7.
- For CoT-UQ baseline consistency, run stage `6c` with sampled generations (`COTUQ_USE_GREEDY=false`) unless intentionally evaluating greedy.

## Citation

If you use this codebase, please cite the CoSu-UQ paper.

```bibtex
@article{cosu_uq_2026,
  title   = {CoSu-UQ: Uncertainty Quantification for Reasoning via Confidence and Semantic Support},
  journal = {IJCNN (submission/manuscript)},
  year    = {2026}
}
```

Replace the entry above with the final published bibliographic metadata when available.

## License

This repository currently does not include a final `LICENSE` file.  
Please add your intended open-source license before public release.
