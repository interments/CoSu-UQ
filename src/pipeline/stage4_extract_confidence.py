import json
import pickle
import re
import spacy
from typing import List, Optional, Tuple
import logging
from transformers import AutoTokenizer
import config
import argparse
from utils.cot_uq_utils import *
import os
import numpy as np
from collections import Counter

os.chdir(config.run_dir)

# ==================== 参数解析 ====================
parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="0", help="CUDA device number")
parser.add_argument('--run_setting', type=str, required=True, help='Run setting name')
parser.add_argument('--model_dir', type=str, required=True, help='Model directory path')
parser.add_argument('--split_method', type=str, default="step_answer", choices=["step_answer", "spacy"], help='Method to split responses')
args = parser.parse_args()

# ==================== 日志配置 ====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s"
)
logger = logging.getLogger(__name__)

# ==================== 全局变量 ====================
_default_nlp = None
_include_pos = ['NOUN', 'ADJ', 'VERB', 'PROPN', 'NUM']

# ==================== 工具函数 ====================
def extract_content_words(
    sentence: str,
    nlp: Optional[spacy.language.Language] = None,
    include_pos: Optional[List[str]] = None,
    unique: bool = True,
    return_lemmas: bool = False,
    lowercase: bool = False
) -> List[str]:
    """从句子中提取实词（NOUN, ADJ, VERB, PROPN, NUM）"""
    global _default_nlp, _include_pos
    
    if include_pos is None:
        include_pos = _include_pos

    if nlp is None:
        if _default_nlp is None:
            _default_nlp = spacy.load("en_core_web_sm")
        nlp = _default_nlp

    doc = nlp(sentence)
    results = []
    seen = set()
    for token in doc:
        if token.pos_ in include_pos:
            word = token.text
            if lowercase:
                word = word.lower()
            if unique:
                if word in seen:
                    continue
                seen.add(word)
            results.append(word)
    return results


def clean_step_text(step_text: str) -> str:
    """清理步骤文本，去掉 'Step X:' 前缀"""
    cleaned = re.sub(r'^Step\s*\d+\s*:\s*', '', step_text, flags=re.IGNORECASE)
    return cleaned.strip()


def get_item_from_generations(generations, id):
    """根据ID从generations中获取对应项"""
    for gen in generations:
        if gen["id"] == id:
            return gen
    return None


def find_sentence_token_range(
    sentence: str,
    output_ids: list,
    tokenizer: AutoTokenizer,
    current_token_idx: int = 0
) -> Tuple[int, int]:
    """
    使用逐步解码的方式查找句子在token序列中的位置
    
    Args:
        sentence: 要查找的句子文本
        output_ids: 输出的token ID序列
        tokenizer: tokenizer对象
        current_token_idx: 开始搜索的token索引
    
    Returns:
        (start_idx, end_idx): 句子对应的token范围，如果未找到返回(-1, -1)
    """
    # 使用正则删除所有空白字符（包括空格、换行符 \n 等）
    sentence_clean = re.sub(r'\s+', '', sentence).lower()
    step_start_idx = current_token_idx
    accumulated_text = ""
    
    # 逐token解码，寻找句子
    while current_token_idx < len(output_ids):
        # 解码当前累积的tokens
        current_tokens = output_ids[step_start_idx:current_token_idx + 1]
        accumulated_text = tokenizer.decode(current_tokens, skip_special_tokens=True).strip()
        
        # 同样使用正则删除所有空白字符
        accumulated_text_clean = re.sub(r'\s+', '', accumulated_text).lower()
        
        # 检查是否包含目标句子
        if sentence_clean in accumulated_text_clean:
            # 找到匹配，返回token范围
            return step_start_idx, current_token_idx
        
        current_token_idx += 1
    
    # 未找到匹配
    return -1, -1

def extract_keyword_probs_from_sentence(
    sentence: str,
    sentence_token_start: int,
    sentence_token_end: int,
    keyword: str,
    output_ids: list,
    output_probs: list,
    tokenizer: AutoTokenizer
) -> Tuple[list, list]:
    from collections import Counter

    # 1. 检查关键词是否在句子中
    if not is_word_in_sentence(sentence, keyword):
        return [], []

    # 2. 提取关键词在句子中的字符串位置
    string_positions = []
    pattern = r'\b' + re.escape(keyword) + r'\b'
    for match in re.finditer(pattern, sentence, flags=re.IGNORECASE):
        string_positions.append([match.start(), match.end() - 1])

    # 3. 解码句子的token序列
    sentence_tokens = output_ids[sentence_token_start:sentence_token_end + 1]

    # 4. 滑动窗口查找所有关键词token区间
    keyword_lower = keyword.lower()
    keyword_token_ranges = []
    used_token_ranges = set()

    for i in range(len(sentence_tokens)):
        for j in range(i + 1, min(i + 10, len(sentence_tokens) + 1)):
            window_tokens = sentence_tokens[i:j]
            window_text = tokenizer.decode(window_tokens, skip_special_tokens=True)
            if window_text.strip().lower() == keyword_lower:
                abs_start = sentence_token_start + i
                abs_end = sentence_token_start + j - 1
                if (abs_start, abs_end) not in used_token_ranges:
                    keyword_token_ranges.append((abs_start, abs_end))
                    used_token_ranges.add((abs_start, abs_end))
                break

    # 5. 收集所有窗口的概率
    all_probs = []
    for abs_start, abs_end in keyword_token_ranges:
        probs = [float(output_probs[idx]) for idx in range(abs_start, abs_end + 1) if idx < len(output_probs)]
        all_probs.extend(probs)

    # 6. 按照string_positions个数切分
    n = len(string_positions)
    if n == 0 or not all_probs:
        return all_probs, string_positions

    avg_len = len(all_probs) // n if n > 0 else 0
    result_probs = []
    for i in range(n):
        start = i * avg_len
        end = (i + 1) * avg_len if i < n - 1 else len(all_probs)
        segment = all_probs[start:end]
        if segment:
            most_common = Counter(segment).most_common(1)
            result_probs.append(most_common[0][0] if most_common else None)
        else:
            # 分配失败，直接返回原始概率列表
            return all_probs, string_positions

    return result_probs, string_positions



# ==================== 主处理函数 ====================
def process_luqpair_keywords_extraction(
    run_setting: str,
    model_dir: str,
    output_dir: str
):
    """
    处理LUQPair结果，提取关键词并计算概率
    
    Args:
        run_setting: 运行配置名称
        model_dir: 模型目录路径
        output_dir: 输出目录路径
    """
    
    # 1. 加载数据
    logger.info("正在加载数据...")
    input_file = f"{output_dir}/{run_setting}/LUQPair_{args.split_method}_splited_results.json"
    with open(input_file, "r") as infile:
        LUQPair_result = json.load(infile)
    
    gen_file = f"{output_dir}/{run_setting}/generations.pkl"
    with open(gen_file, "rb") as infile:
        generations = pickle.load(infile)
    
    cluster_file = f"{output_dir}/{run_setting}/semantic_clusters.pkl"
    with open(cluster_file, "rb") as infile:
        semantic_clusters = pickle.load(infile)
    
    # 2. 加载tokenizer
    logger.info("正在加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
    
    # 3. 初始化spaCy
    logger.info("正在加载spaCy模型...")
    global _default_nlp
    if _default_nlp is None:
        _default_nlp = spacy.load("en_core_web_sm")
    
    # 4. 处理每个样本
    result = []
    total = len(LUQPair_result)
    not_found_count = 0
    total_sentences = 0
    
    for idx, item in enumerate(LUQPair_result):
        if (idx + 1) % 50 == 0:
            logger.info(f"处理进度: {idx + 1}/{total}, 未找到句子数: {not_found_count}/{total_sentences}")
        
        generation = get_item_from_generations(generations, item["id"])
        
        if generation is None:
            logger.warning(f"ID {item['id']} 在generations中找不到,跳过")
            continue
        
        # 获取必要信息
        generated_probs = generation["generated_probs"]
        question = generation["question"]
        id = generation["id"]
        generated_ids = generation["generated_ids"]
        prompt = generation["prompt"]
        generated_texts = generation["generated_texts"]
        generated_success_flag = generation.get("generated_success_flag", [True] * len(generated_texts))
        # generated_success_flag = generation["generated_success_flag"]
        semantic_set_ids = semantic_clusters[id]["semantic_set_ids"]
        splited_responses = item.get("splited_responses", None)
        
        # 检查splited_responses
        if splited_responses is None:
            logger.warning(f"ID {id} 缺少 splited_responses 字段,跳过")
            continue
        elif not isinstance(splited_responses, list) or len(splited_responses) == 0:
            logger.warning(f"ID {id} 的 splited_responses 字段格式不正确或为空,跳过")
            continue
        
        # 处理每个response
        responses_data = []
        
        # 获取generated_ids列表
        generated_ids_list = [
            seq[prompt.shape[1]:][(seq[prompt.shape[1]:] != tokenizer.eos_token_id) & (seq[prompt.shape[1]:] != tokenizer.pad_token_id)]
            for seq in generated_ids
        ]
        
        for response_idx, response in enumerate(splited_responses):
            # response_result = {}
            response_result = []
            current_generated_ids = generated_ids_list[response_idx].tolist()
            current_probs = generated_probs[response_idx]
            
            # 获取原始response_text
            response_text = item.get("responses", [])[response_idx] if response_idx < len(item.get("responses", [])) else ""
            
            # 用于追踪当前搜索位置
            current_search_idx = 0
            
            # 处理每个句子(step)
            for sentence_idx, sentence in enumerate(response):
                total_sentences += 1
                
                # 清理句子
                cleaned_sentence = clean_step_text(sentence)
                
                # 提取实词
                content_words = extract_content_words(
                    cleaned_sentence,
                    unique=True,
                    return_lemmas=False,
                    lowercase=False
                )
                
                if not content_words:
                    logger.debug(f"ID {id} 句子 '{sentence[:50]}...' 未提取到实词")
                    response_result.append({
                        "sentence": sentence,
                        "keywords_probs": None,
                        "keywords_start_end_string_from_sentence": None,
                        "keywords_list": None,
                        "sentence_string_position_in_response": None,
                        "sentence_token_range": None
                    })
                    continue
                
                # 🎯 使用新的方法查找句子在token序列中的位置
                sentence_token_start, sentence_token_end = find_sentence_token_range(
                    sentence=sentence,
                    output_ids=current_generated_ids,
                    tokenizer=tokenizer,
                    current_token_idx=current_search_idx
                )
                
                if sentence_token_start == -1:
                    not_found_count += 1
                    logger.warning(f"ID {id} response {response_idx} 句子 {sentence_idx} '{sentence[:50]}...' 在generated_ids中找不到")
                    response_result.append({
                        "sentence": sentence,
                        "keywords_probs": None,
                        "keywords_start_end_string_from_sentence": None,
                        "keywords_list": None,
                        "sentence_string_position_in_response": None,
                        "sentence_token_range": None
                    })
                    continue
                
                # 更新搜索位置
                current_search_idx = sentence_token_end + 1
                
                # 查找sentence在response_text中的字符串位置
                sentence_string_start = response_text.find(sentence)
                sentence_string_end = sentence_string_start + len(sentence) - 1 if sentence_string_start != -1 else -1
                
                # 提取每个关键词的概率和位置
                keywords_probs = {}
                keywords_start_end_string_from_sentence = {}
                
                for keyword in content_words:
                    # 🎯 使用新方法提取关键词概率
                    keyword_probs, string_positions = extract_keyword_probs_from_sentence(
                        sentence=sentence,
                        sentence_token_start=sentence_token_start,
                        sentence_token_end=sentence_token_end,
                        keyword=keyword,
                        output_ids=current_generated_ids,
                        output_probs=current_probs,
                        tokenizer=tokenizer
                    )
                    
                    if keyword_probs is None:
                        logger.debug(f"ID {id} 关键词 '{keyword}' 提取失败")
                        continue
                    
                    # 存储结果
                    keywords_probs[keyword] = keyword_probs
                    keywords_start_end_string_from_sentence[keyword] = string_positions
                
                # 存储句子级别的结果
                # response_result[sentence] = {
                #     "keywords_probs": keywords_probs,
                #     "keywords_start_end_string_from_sentence": keywords_start_end_string_from_sentence,
                #     "keywords_list": content_words,
                #     "sentence_string_position_in_response": [sentence_string_start, sentence_string_end],
                #     "sentence_token_range": [sentence_token_start, sentence_token_end]  # 新增：token范围
                # }
                response_result.append({
                    "sentence": sentence,
                    "keywords_probs": keywords_probs,
                    "keywords_start_end_string_from_sentence": keywords_start_end_string_from_sentence,
                    "keywords_list": content_words,
                    "sentence_string_position_in_response": [sentence_string_start, sentence_string_end],
                    "sentence_token_range": [sentence_token_start, sentence_token_end]
                })
                
                logger.debug(f"ID {id} response {response_idx} 句子 {sentence_idx} 成功处理，token范围: [{sentence_token_start}, {sentence_token_end}]")
            
            responses_data.append(response_result)
        
        # 构建最终结果
        result.append({
            "id": id,
            "question": question,
            "semantic_set_ids": semantic_set_ids,
            "responses_data": responses_data,
            "responses": item.get("responses", None),
            "prompt_text": item.get("prompt_text", None),
            "answer": item.get("answer", None),
            "splited_responses": splited_responses,
            "generated_success_flag": generated_success_flag
        })
        
        logger.debug(f"已处理 ID {id}")
    
    # 5. 保存结果
    output_file = f"{output_dir}/{run_setting}/LUQPair_keywords_probs.json"
    with open(output_file, "w", encoding="utf-8") as outfile:
        json.dump(result, outfile, indent=4, ensure_ascii=False)
    
    logger.info(f"✅ 结果已保存到: {output_file}")
    # logger.info(f"总样本数: {len(result)}")
    logger.info(f"原始样本数: {total}")
    logger.info(f"最终保存样本数: {len(result)}")
    logger.info(f"丢失样本数: {total - len(result)}")
    
    # 6. 打印统计信息
    total_responses = sum(len(item["responses_data"]) for item in result)
    successful_responses = sum(
        sum(1 for r in item["responses_data"] if r is not None)
        for item in result
    )
    logger.info(f"总response数: {total_responses}")
    logger.info(f"成功处理的response数: {successful_responses}")
    logger.info(f"总句子数: {total_sentences}")
    logger.info(f"未找到的句子数: {not_found_count} ({not_found_count/total_sentences*100:.2f}%)")


# ==================== 主程序入口 ====================
if __name__ == "__main__":
    process_luqpair_keywords_extraction(
        run_setting=args.run_setting,
        model_dir=args.model_dir,
        output_dir=config.output_dir
    )