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

# ==================== Argument Parsing ====================
parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="0", help="CUDA device number")
parser.add_argument('--run_setting', type=str, required=True, help='Run setting name')
parser.add_argument('--model_dir', type=str, required=True, help='Model directory path')
parser.add_argument('--split_method', type=str, default="step_answer", choices=["step_answer", "spacy"], help='Method to split responses')
args = parser.parse_args()

# ==================== Logging Configuration ====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s"
)
logger = logging.getLogger(__name__)

# ==================== Global Variables ====================
_default_nlp = None
_include_pos = ['NOUN', 'ADJ', 'VERB', 'PROPN', 'NUM']

# ==================== Utility Functions ====================
def extract_content_words(
    sentence: str,
    nlp: Optional[spacy.language.Language] = None,
    include_pos: Optional[List[str]] = None,
    unique: bool = True,
    return_lemmas: bool = False,
    lowercase: bool = False
) -> List[str]:
    """Extract content words from a sentence (NOUN, ADJ, VERB, PROPN, NUM)."""
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
    """Clean step text by removing the 'Step X:' prefix."""
    cleaned = re.sub(r'^Step\s*\d+\s*:\s*', '', step_text, flags=re.IGNORECASE)
    return cleaned.strip()


def get_item_from_generations(generations, id):
    """Get the corresponding item from generations by ID."""
    for gen in generations:
        if gen["id"] == id:
            return gen
    return None


def resolve_support_input_file(output_dir: str, run_setting: str, split_method: str) -> str:
    path = f"{output_dir}/{run_setting}/support_uncertainty_luqpair_{split_method}.json"
    if os.path.exists(path):
        logger.info("[stage4] using support input: %s", path)
        return path
    raise FileNotFoundError(f"Required support result file not found: {path}")


def find_sentence_token_range(
    sentence: str,
    output_ids: list,
    tokenizer: AutoTokenizer,
    current_token_idx: int = 0
) -> Tuple[int, int]:
    """
    Find the sentence position in the token sequence via incremental decoding.
    
    Args:
        sentence: Sentence text to locate.
        output_ids: Output token ID sequence.
        tokenizer: Tokenizer object.
        current_token_idx: Token index where the search starts.
    
    Returns:
        (start_idx, end_idx): Token span for the sentence; returns (-1, -1) if not found.
    """
    # Remove all whitespace characters (including spaces, newlines, etc.) with regex.
    sentence_clean = re.sub(r'\s+', '', sentence).lower()
    step_start_idx = current_token_idx
    accumulated_text = ""
    
    # Decode token-by-token to locate the sentence.
    while current_token_idx < len(output_ids):
        # Decode currently accumulated tokens.
        current_tokens = output_ids[step_start_idx:current_token_idx + 1]
        accumulated_text = tokenizer.decode(current_tokens, skip_special_tokens=True).strip()
        
        # Apply the same whitespace normalization.
        accumulated_text_clean = re.sub(r'\s+', '', accumulated_text).lower()
        
        # Check whether the target sentence appears in the accumulated text.
        if sentence_clean in accumulated_text_clean:
            # Match found: return the token span.
            return step_start_idx, current_token_idx
        
        current_token_idx += 1
    
    # No match found.
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

    # 1. Check whether the keyword appears in the sentence.
    if not is_word_in_sentence(sentence, keyword):
        return [], []

    # 2. Extract keyword character spans in the sentence.
    string_positions = []
    pattern = r'\b' + re.escape(keyword) + r'\b'
    for match in re.finditer(pattern, sentence, flags=re.IGNORECASE):
        string_positions.append([match.start(), match.end() - 1])

    # 3. Decode the sentence token sequence.
    sentence_tokens = output_ids[sentence_token_start:sentence_token_end + 1]

    # 4. Use a sliding window to find all keyword token spans.
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

    # 5. Collect probabilities from all matched windows.
    all_probs = []
    for abs_start, abs_end in keyword_token_ranges:
        probs = [float(output_probs[idx]) for idx in range(abs_start, abs_end + 1) if idx < len(output_probs)]
        all_probs.extend(probs)

    # 6. Split by the number of string positions.
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
            # If allocation fails, return the original probability list directly.
            return all_probs, string_positions

    return result_probs, string_positions



# ==================== Main Processing Function ====================
def process_luqpair_keywords_extraction(
    run_setting: str,
    model_dir: str,
    output_dir: str
):
    """
    Process support-uncertainty output and extract keyword probabilities.
    
    Args:
        run_setting: Run setting name.
        model_dir: Model directory path.
        output_dir: Output directory path.
    """
    
    # 1. Load input data.
    logger.info("[stage4] loading inputs")
    input_file = resolve_support_input_file(output_dir, run_setting, args.split_method)
    with open(input_file, "r") as infile:
        support_result = json.load(infile)
    
    gen_file = f"{output_dir}/{run_setting}/generations.pkl"
    with open(gen_file, "rb") as infile:
        generations = pickle.load(infile)
    
    cluster_file = f"{output_dir}/{run_setting}/semantic_clusters.pkl"
    with open(cluster_file, "rb") as infile:
        semantic_clusters = pickle.load(infile)
    
    # 2. Load tokenizer.
    logger.info("[stage4] loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
    
    # 3. Initialize spaCy.
    logger.info("[stage4] loading spaCy model")
    global _default_nlp
    if _default_nlp is None:
        _default_nlp = spacy.load("en_core_web_sm")
    
    # 4. Process each sample.
    result = []
    total = len(support_result)
    not_found_count = 0
    total_sentences = 0
    
    for idx, item in enumerate(support_result):
        if (idx + 1) % 50 == 0:
            logger.info(f"[stage4] progress={idx + 1}/{total}, sentence_not_found={not_found_count}/{total_sentences}")
        
        generation = get_item_from_generations(generations, item["id"])
        
        if generation is None:
            logger.warning(f"[stage4] skip id={item['id']}: missing in generations")
            continue
        
        # Retrieve required fields.
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
        
        # Validate splited_responses.
        if splited_responses is None:
            logger.warning(f"[stage4] skip id={id}: missing splited_responses")
            continue
        elif not isinstance(splited_responses, list) or len(splited_responses) == 0:
            logger.warning(f"[stage4] skip id={id}: invalid or empty splited_responses")
            continue
        
        # Process each response.
        responses_data = []
        
        # Build the list of generated IDs.
        generated_ids_list = [
            seq[prompt.shape[1]:][(seq[prompt.shape[1]:] != tokenizer.eos_token_id) & (seq[prompt.shape[1]:] != tokenizer.pad_token_id)]
            for seq in generated_ids
        ]
        
        for response_idx, response in enumerate(splited_responses):
            # response_result = {}
            response_result = []
            current_generated_ids = generated_ids_list[response_idx].tolist()
            current_probs = generated_probs[response_idx]
            
            # Get the original response text.
            response_text = item.get("responses", [])[response_idx] if response_idx < len(item.get("responses", [])) else ""
            
            # Track current search position.
            current_search_idx = 0
            
            # Process each sentence (step).
            for sentence_idx, sentence in enumerate(response):
                total_sentences += 1
                
                # Clean sentence text.
                cleaned_sentence = clean_step_text(sentence)
                
                # Extract content words.
                content_words = extract_content_words(
                    cleaned_sentence,
                    unique=True,
                    return_lemmas=False,
                    lowercase=False
                )
                
                if not content_words:
                    logger.debug(f"ID {id} sentence '{sentence[:50]}...' has no extracted content words")
                    response_result.append({
                        "sentence": sentence,
                        "keywords_probs": None,
                        "keywords_start_end_string_from_sentence": None,
                        "keywords_list": None,
                        "sentence_string_position_in_response": None,
                        "sentence_token_range": None
                    })
                    continue
                
                # Use the new method to locate sentence token span.
                sentence_token_start, sentence_token_end = find_sentence_token_range(
                    sentence=sentence,
                    output_ids=current_generated_ids,
                    tokenizer=tokenizer,
                    current_token_idx=current_search_idx
                )
                
                if sentence_token_start == -1:
                    not_found_count += 1
                    logger.warning(f"[stage4] id={id} response={response_idx} sentence={sentence_idx}: token span not found")
                    response_result.append({
                        "sentence": sentence,
                        "keywords_probs": None,
                        "keywords_start_end_string_from_sentence": None,
                        "keywords_list": None,
                        "sentence_string_position_in_response": None,
                        "sentence_token_range": None
                    })
                    continue
                
                # Update search position.
                current_search_idx = sentence_token_end + 1
                
                # Find sentence character span in response_text.
                sentence_string_start = response_text.find(sentence)
                sentence_string_end = sentence_string_start + len(sentence) - 1 if sentence_string_start != -1 else -1
                
                # Extract probability and position for each keyword.
                keywords_probs = {}
                keywords_start_end_string_from_sentence = {}
                
                for keyword in content_words:
                    # Use the new method to extract keyword probabilities.
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
                        logger.debug(f"ID {id} keyword '{keyword}' extraction failed")
                        continue
                    
                    # Store results.
                    keywords_probs[keyword] = keyword_probs
                    keywords_start_end_string_from_sentence[keyword] = string_positions
                
                # Store sentence-level results.
                # response_result[sentence] = {
                #     "keywords_probs": keywords_probs,
                #     "keywords_start_end_string_from_sentence": keywords_start_end_string_from_sentence,
                #     "keywords_list": content_words,
                #     "sentence_string_position_in_response": [sentence_string_start, sentence_string_end],
                #     "sentence_token_range": [sentence_token_start, sentence_token_end]  # Added: token span
                # }
                response_result.append({
                    "sentence": sentence,
                    "keywords_probs": keywords_probs,
                    "keywords_start_end_string_from_sentence": keywords_start_end_string_from_sentence,
                    "keywords_list": content_words,
                    "sentence_string_position_in_response": [sentence_string_start, sentence_string_end],
                    "sentence_token_range": [sentence_token_start, sentence_token_end]
                })
                
                logger.debug(f"ID {id} response {response_idx} sentence {sentence_idx} processed successfully, token span: [{sentence_token_start}, {sentence_token_end}]")
            
            responses_data.append(response_result)
        
        # Build final output record.
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
        
        logger.debug(f"Processed ID {id}")
    
    # 5. Save results.
    output_file = f"{output_dir}/{run_setting}/confidence_keywords_probs.json"
    with open(output_file, "w", encoding="utf-8") as outfile:
        json.dump(result, outfile, indent=4, ensure_ascii=False)
    
    logger.info(f"[stage4] saved: {output_file}")
    logger.info(f"[stage4] total_samples={total}")
    logger.info(f"[stage4] saved_samples={len(result)}")
    logger.info(f"[stage4] dropped_samples={total - len(result)}")
    
    # 6. Print statistics.
    total_responses = sum(len(item["responses_data"]) for item in result)
    successful_responses = sum(
        sum(1 for r in item["responses_data"] if r is not None)
        for item in result
    )
    logger.info(f"[stage4] total_responses={total_responses}")
    logger.info(f"[stage4] successful_responses={successful_responses}")
    logger.info(f"[stage4] total_sentences={total_sentences}")
    logger.info(f"[stage4] sentence_not_found={not_found_count} ({not_found_count/total_sentences*100:.2f}%)")


# ==================== Program Entry ====================
if __name__ == "__main__":
    process_luqpair_keywords_extraction(
        run_setting=args.run_setting,
        model_dir=args.model_dir,
        output_dir=config.output_dir
    )
