# Pipeline 执行与 IO 契约（中文）

本文档描述当前 CoSu-UQ 项目的端到端执行流程，包括：
- 每个阶段对应运行的脚本，
- 必需输入，
- 产出输出，
- 输出数据格式，
- 以及输出如何被下游阶段消费。

## 0. 编排入口

主编排脚本：
- `src/scripts/run_stage.sh`

支持的阶段 id：
- `0 1 2 3 4 5j 5l 6s 6t 6b 6c 7`

全流程脚本：
- `src/scripts/run_pipeline.sh`

批量运行脚本（多个 `run_setting`）：
- `src/scripts/run_batch.sh`

如果未显式设置 `RUN_SETTING`，`run_stage.sh` 会基于 stage-1 的生成配置自动推导 `run_setting` 名称。

---

## 1) Stage 0：数据集构建

脚本：
- `python -m pipeline.stage0_build_datasets`

输入：
- 来自 HuggingFace 的数据集（`gsm8k`, `hotpotqa`, `math`, `2wiki`, `medqa`）以及 `common.env` 中配置的参数。

输出：
- `datasets/` 目录下的清洗后 jsonl 文件，例如：
  - `gsm8k_cot_uq_validation_0-500_src.jsonl`

主要格式（每行 jsonl）：
- 一条转换为 CoT 提示风格的 QA 样本。

下游消费：
- Stage 1（通过 `--data_file` 读取）。

---

## 2) Stage 1：CoT 多采样生成

脚本：
- `python -m pipeline.stage1_generate_cot`

输入：
- Stage 0 产出的数据集 jsonl。
- 生成配置（`MODEL_DIR`, `NUM_GENERATIONS`, `TEMPERATURE` 等）。

输出：
- `results/{run_setting}/generations.pkl`

主要格式（`list[dict]`）：
- 每个元素对应一个样本，包含字段（示例）：
  - `id`, `question`, `prompt_text`, `prompt`
  - `generated_texts`: `list[str]`（采样回答）
  - `generated_probs`: `list[list[float]]`（每条采样回答的 token 概率）
  - `generated_ids`: 采样回答的 token id 张量
  - `generated_success_flag`: `list[bool]`
  - `cleaned_generated_texts`, `cleaned_generated_ids`
  - `most_likely_generation`, `most_likely_generation_probs`, `most_likely_generation_ids`

下游消费：
- Stage 2、3、4、5j、5l、6s、6t。

---

## 3) Stage 2：语义聚类

脚本：
- `python -m pipeline.stage2_semantic_cluster`

输入：
- `results/{run_setting}/generations.pkl`

输出：
- `results/{run_setting}/semantic_clusters.pkl`

主要格式（按样本 id 索引的 `dict`）：
- `semantic_set_ids`：cleaned 响应的聚类 id
- `semantic_set_ids_raw`：raw 响应的聚类 id
- `semantic_set_ids_entailment`：双向蕴含聚类 id

下游消费：
- Stage 4（使用 `semantic_set_ids`）
- Stage 5l（使用 `semantic_set_ids_entailment`）
- Stage 7（基于聚类进行不确定性聚合）
- Stage 6b（语义熵 baseline）

---

## 4) Stage 3：支持度信号（LUQ 系列）

脚本：
- `python -m pipeline.stage3_compute_support`

常用配置：
- 手动单方法运行：`--luq_method LUQPair --split_method step_answer`（或 `LUQ`）
- `run_stage.sh` 的 stage `3` 默认会连续运行 `LUQPair` + `LUQ`。

输入：
- `results/{run_setting}/generations.pkl`

输出：
- 按方法分别输出：
  - `results/{run_setting}/support_uncertainty_luqpair_{split_method}.json`（CoSu-UQ 主分支）
  - `results/{run_setting}/support_uncertainty_luq_{split_method}.json`（LUQ baseline）

主要格式（`list[dict]`）：
- `id`, `prompt_text`, `answer`
- `generated_texts`, `responses`
- `splited_responses`：切分后的推理单元
- `uncertainty_scores`：每条 response 的支持度不确定性
- `score`：样本级支持度不确定性
- 可选 `nli_probability_matrix`（当 `--save_matrix` 时）

下游消费：
- Stage 4（使用句子切分信息）
- Stage 6b（通过 `--luq_split_method` 读取 `support_uncertainty_luq_{split_method}.json` 构造 `luq_scores.pkl`）
- Stage 7（SU 分支读取 `uncertainty_scores` 并转为支持度信息）

---

## 5) Stage 4：置信度信号（关键词概率）

脚本：
- `python -m pipeline.stage4_extract_confidence`

输入：
- `results/{run_setting}/support_uncertainty_luqpair_{split_method}.json`
- `results/{run_setting}/generations.pkl`
- `results/{run_setting}/semantic_clusters.pkl`

输出：
- `results/{run_setting}/confidence_keywords_probs.json`

主要格式（`list[dict]`）：
- `id`, `question`, `semantic_set_ids`
- `responses_data`：对每条 response 的句级信息列表，每个句子包含：
  - `sentence`
  - `keywords_probs`: `{keyword: [prob, ...]}`
  - `keywords_start_end_string_from_sentence`
  - `keywords_list`
  - `sentence_string_position_in_response`
  - `sentence_token_range`
- `responses`, `prompt_text`, `answer`, `splited_responses`, `generated_success_flag`

下游消费：
- Stage 7（GU/组合置信度分支）。

---

## 6) Stage 5j：Judge 标签回填

脚本：
- `python -m eval.judge_responses`

输入：
- `results/{run_setting}/generations.pkl`
- `--judgers` 配置（支持每个模型单独设置 API base/key，支持 `${ENV_VAR}` 占位）

输出：
- 回写同一个文件：
  - `results/{run_setting}/generations.pkl`

每个样本新增/更新字段：
- `<model>_judge_result_labels`
- `final_judge_result_labels`

下游消费：
- Stage 5l（构建 AUROC 标签）。

---

## 7) Stage 5l：AUROC 标签构建

脚本：
- `python -m eval.build_auroc_labels`

输入：
- `results/{run_setting}/generations.pkl`（必须已有 `final_judge_result_labels`）
- `results/{run_setting}/semantic_clusters.pkl`

输出：
- `results/{run_setting}/AUROC_labels.json`

主要格式（`list[dict]`）：
- `id`
- `greedy_label`
- `most_sampled_label`
- `most_cluster_label`（Stage 7 默认目标标签）

下游消费：
- Stage 7 最终 AUROC 评估。

---

## 8) Stage 6s/6t/6b：Baseline 缓存与分数构建

### Stage 6s
脚本：
- `python -m baselines.cache_sentence_similarity`

输出：
- `results/{run_setting}/sentence_similarities_{model_key}.pkl`

### Stage 6t
脚本：
- `python -m baselines.cache_token_importance`

输出：
- `results/{run_setting}/tokenwise_importance_{model_key}_from_generation.pkl`

### Stage 6b
脚本：
- `python -m baselines.build_baseline_scores`

输入：
- `generations.pkl`、`semantic_clusters.pkl`、句相似度缓存、token 重要性缓存
- 通过（`--luq_split_method`）定位支持度文件 `support_uncertainty_luq_{split_method}.json`

输出：
- `predictive_entropy_scores.pkl`
- `len_normed_predictive_entropy_scores.pkl`
- `semantic_entropy_scores.pkl`
- `sentence_sar_scores.pkl`
- `token_sar_scores.pkl`
- `luq_scores.pkl`

所有 `*_scores.pkl` 格式：
- 样本级分数列表（每个样本是一个 `torch.Tensor` 标量）

下游消费：
- Stage 7 baseline 对比。

---

## 9) Stage 6c：CoT-UQ Baseline（已迁移到 baselines）

Stage 6c 顺序运行两个脚本。

### 6c-1 关键词抽取
脚本：
- `python -m baselines.cotuq_keyword_extraction`

输入：
- `results/{run_setting}/generations.pkl`

输出：
- `results/{run_setting}/keywords_probabilities_{prompt_type}_{sampled|greedy}.json`

主要格式（`list[dict]`）：
- `id`, `question`
- `responses`：每条采样路径对应一个元素：
  - `response_idx`
  - `keywords_probabilities`
  - `keywords_contributions`
  - `cot_uq_success_flag`

### 6c-2 分数聚合
脚本：
- `python -m baselines.cotuq_aggregate_scores`

输入：
- `keywords_probabilities_{prompt_type}_{sampled|greedy}.json`

输出：
- `results/{run_setting}/Cot_uq_{aggregated_method}_scores_{prompt_type}_{sampled|greedy}.pkl`
- `results/{run_setting}/cotuq_scores.pkl`（统一的 CoT-UQ baseline 分数文件）

格式：
- 样本级 CoT-UQ 不确定性分数列表（每个样本是 `torch.Tensor` 标量）

下游消费：
- Stage 7（`final_compare.py` 自动识别已知 `Cot_uq_*` 分数文件）。

---

## 10) Stage 7：最终 CoSu-UQ + Baselines AUROC

脚本：
- `python -m eval.final_compare`

输入：
- CoSu-UQ 核心文件：
  - `confidence_keywords_probs.json`
  - `support_uncertainty_luqpair_{split_method}.json`
  - `semantic_clusters.pkl`
  - `generations.pkl`
  - `AUROC_labels.json`
- Stage 6b 的 baseline 分数 PKL
- 可选 Stage 6c 的 CoT-UQ 分数 PKL

输出：
- 汇总 CSV（默认）：
  - `results/final_uq_baseline_compare.csv`

主要输出列：
- `model`, `dataset`
- baseline 的 AUROC 列：PE、LN-PE、sentence-sar、token-sar、SE、LUQ、可选 CoT-UQ
- CoSu-UQ AUROC 列：`GU_scores`, `SU_scores`, `combined_scores`

---

## 依赖图（实用）

- Stage 0 -> Stage 1
- Stage 1 -> Stage 2/3/4/5j/6s/6t
- Stage 2 + Stage 5j -> Stage 5l
- Stage 3 + Stage 1 + Stage 2 -> Stage 4
- Stage 1 + Stage 2 + Stage 6s + Stage 6t + Stage 3 -> Stage 6b
- Stage 1 -> Stage 6c-1 -> Stage 6c-2
- Stage 4 + Stage 3 + Stage 2 + Stage 1 + Stage 5l + Stage 6b (+ optional Stage 6c) -> Stage 7

---

## 契约稳定性说明

- 所有阶段必须使用一致的 `run_setting`，否则文件发现会失败。
- Stage 3 按方法分别输出 CoSu-UQ 主分支（`luqpair`）与 LUQ baseline（`luq`）文件。
- Stage 4/7 读取 `support_uncertainty_luqpair_{split_method}.json`。
- Stage 6b 读取 `support_uncertainty_luq_{split_method}.json`。
