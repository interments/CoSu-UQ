# 数据字段含义说明文档

本文档详细解释了 `generations.pkl` 数据文件中各条目字段的含义。这些字段来源于模型生成、后处理重写以及事实性评分等多个处理阶段。

## 1. 基础信息与配置
这些字段通常在 `Generation_general.py` 中初始化或记录。

| 字段名 | 类型 | 说明 |
| :--- | :--- | :--- |
| **id** | `int` | 样本的唯一标识符。 |
| **prompt_text** | `str` | 输入给模型的原始提示文本（Prompt）。 |
| **question** | `str` | 对应的问题文本。 |
| **topic** | `str` | 当前样本所属的主题或关键词（例如 "Focus..."）。 |
| **answer** | `tuple` | 问题的标准参考答案（Ground Truth），通常为元组形式。 |
| **decode_method** | `str` | 使用的解码策略（例如 `'greedy'`）。 |
| **prompt** | `tensor` | 输入提示文本对应的 Token ID 序列。 |
| **temperature**, **top_k**, **top_p** | `float/int` | (隐含) 生成时的采样参数配置。 |

## 2. 生成结果 (Generations)
记录模型根据输入生成的原始输出及相关概率信息。

| 字段名 | 类型 | 说明 |
| :--- | :--- | :--- |
| **generated_ids** | `tensor` | 多次生成的文本对应的 Token ID 矩阵（每一行代表一次采样生成）。 |
| **generated_texts** | `list[str]` | 多次生成的原始文本列表。 |
| **generated_probs** | `list[list[float]]` | 对应 `generated_texts` 中每个 Token 的生成概率序列。 |
| **cleaned_generated_texts** | `list[str]` | 经过清洗（如去除 Prompt 部分、特殊 token 或截断）后的生成文本列表。 |
| **cleaned_generated_ids** | `tensor` | 对应 `cleaned_generated_texts` 的 Token ID 序列。 |

## 3. 最优采样结果 (Most Likely Sampled)
由 `Rewrite_generations_general.py` 脚本根据 `likelihoods.pkl` 计算并添加到数据中，代表在多次采样中概率最高的那个结果。

| 字段名 | 类型 | 说明 |
| :--- | :--- | :--- |
| **most_likely_sampled_generation** | `str` | 所有采样结果中，模型似然度（Likelihood）最高的生成文本。 |
| **most_likely_sampled_generation_ids** | `tensor` | 对应的 Token IDs。 |
| **most_likely_sampled_generation_probs** | `list[float]` | 对应的 Token 概率列表。 |
| **cleaned_most_likely_sampled_generation** | `str` | 清洗后的最优采样文本。 |
| **cleaned_most_likely_sampled_generation_ids**| `tensor` | 清洗后版本对应的 Token IDs。 |

## 4. 贪婪解码结果 (Greedy / Most Likely)
对应 `greedy` 解码策略的生成结果。

| 字段名 | 类型 | 说明 |
| :--- | :--- | :--- |
| **most_likely_generation** | `str` | 贪婪解码生成的文本。 |
| **most_likely_generation_ids** | `tensor` | 对应的 Token IDs。 |
| **most_likely_generation_probs** | `list[float]` | 对应的 Token 概率。 |
| **cleaned_most_likely_generation** | `str` | 清洗后的贪婪生成文本。 |
| **cleaned_most_likely_generation_ids** | `tensor` | 对应的 Token IDs。 |

## 5. 评估指标 (Evaluation Metrics)
由 `Rewrite_generations_general.py` 等脚本计算，衡量生成文本与参考答案 (`answer`) 的相似度。

### 通用指标
| 字段名 | 类型 | 说明 |
| :--- | :--- | :--- |
| **exact_match** | `float` | 精确匹配得分。 |
| **bertscore_precision/recall/f1** | `float` | BERTScore 相关指标。 |
| **rouge1/rouge2/rougeL_to_target** | `float` | ROUGE 相似度指标。 |

### 最优采样特定指标 (Suffix: `_most_sampled`)
针对 `most_likely_sampled_generation` 专门计算的指标：
- **exact_match_most_sampled**
- **bertscore_precision_most_sampled**, **bertscore_recall_most_sampled**, **bertscore_f1_most_sampled**
- **rouge1_to_target_most_sampled**, **rouge2_to_target_most_sampled**, **rougeL_to_target_most_sampled**

## 6. 事实性评分 (Atomic Fact Score)
由 `Get_atomic_fact_score.py` 脚本计算。该脚本将生成文本拆解为“原子事实”（Atomic Facts），并利用外部知识库（如 Wikipedia、Google Search）或大模型进行验证。

根据参数 `greedy_as_score_flag` 的不同，结果会保存在 `_greedy` 或 `_most_sampled` 后缀的字段中。

| 字段名 | 类型 | 说明 |
| :--- | :--- | :--- |
| **atomic_facts_greedy** | `list[str]` | 从贪婪生成结果中拆解出的原子事实句子列表（使用 spaCy 拆分）。 |
| **gpt_labels_greedy** | `list[str]` | 对每个原子事实的验证标签列表。<br>- **'S'**: Supported (事实正确/支持)<br>- **'NS'**: Not Supported (不支持/不正确)<br>- **Refusal**: 拒绝回答通常也被视为 'S' (Supported/Correct behavior)。 |
| **fact_score_greedy** | `float` | 贪婪生成结果的事实性得分，计算公式为：`count('S') / total_facts`。 |
| **fact_score_error_greedy** | `str` | (可选) 如果处理出错，记录错误信息，此时得分为 -1。 |
| **atomic_facts_most_sampled** | `list[str]` | (同上) 针对最优采样结果的原子事实列表。 |
| **gpt_labels_most_sampled** | `list[str]` | (同上) 针对最优采样结果的验证标签。 |
| **fact_score_most_sampled** | `float` | (同上) 针对最优采样结果的事实性得分。 |

---
**注意**：部分字段的存在依赖于具体的实验运行设置（如是否运行了 rewrite 脚本或 fact score 脚本）。
