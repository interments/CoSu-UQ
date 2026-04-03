# CoSu-UQ 最终分数合成说明

本目录中的 final_score_compare.py 是你给的 notebook 代码的脚本化整理版，功能是：

1. 读取已有中间结果（不是重跑前序 pipeline）。
2. 计算 CoSu-UQ 的三种最终分数：
   - GU_scores_fyl：基于关键词概率的置信度不确定性。
   - SU_scores_fyl：基于语义支持度的支持不确定性。
   - combined_scores_fyl：先融合置信度与支持度信息，再做语义簇级不确定性聚合。
3. 与 baseline 不确定性分数进行 AUROC 对比。

## 输入文件（每个 run_setting 目录）

- LUQPair_keywords_probs.json
- LUQPair_step_answer_splited_results.json
- semantic_clusters.pkl
- generations.pkl
- AUROC_labels.json
- luq_scores.pkl
- predictive_entropy_from_generations_scores.pkl
- len_normed_predictive_entropy_from_generations_scores.pkl
- semantic_entropy_from_generations_scores.pkl
- sentence_sar_from_generations_scores.pkl
- token_sar_from_generations_scores.pkl

## CoSu-UQ 在最终阶段具体做了什么

1. 构建 Confidence Information
   - 关键词概率聚合：min
   - 句内聚合：mean
   - 响应级聚合：prod
   - 得到每个样本多条生成的置信信息矩阵。

2. 构建 Support Information
   - 从 LUQPair_step_answer_splited_results.json 读取每步不确定性，转换为支持度：1 - uncertainty_scores。

3. 语义簇聚合得到不确定性
   - 使用 semantic_set_ids_entailment 作为聚类结果。
   - 在簇内做对数和聚合并归一，得到样本级不确定性。

4. 最终三类 CoSu-UQ 分数
   - GU_scores_fyl：仅由 Confidence Information 聚合得到。
   - SU_scores_fyl：仅由 Support Information 聚合得到。
   - combined_scores_fyl：先按 mean 融合两类信息，再做同样的语义簇聚合。

5. 与 baseline 对比
   - baseline: PE, LN-PE, sentence-SAR, token-SAR, SE, LUQ
   - 指标: AUROC，标签使用 AUROC_labels.json 里的 most_cluster_label。

## 运行方式

```bash
python src/eval/final_compare.py \
  --results-root ./results \
  --output-csv ./results/final_uq_baseline_compare.csv
```

可选参数：

- --run-settings-json：传入一个 JSON 文件（list[str]）覆盖默认 run_settings。
- --max-workers：并行处理线程数。
