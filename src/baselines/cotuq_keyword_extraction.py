import argparse
import json
import logging
import os
import pickle
import re
from typing import Dict, List

import config
import spacy
import tqdm
from transformers import AutoTokenizer

from utils.cot_uq_utils import parse_response_to_dict


def clean_step_text(step_text: str) -> str:
    return re.sub(r"^Step\s*\d+\s*:\s*", "", step_text, flags=re.IGNORECASE).strip()


def extract_content_words(nlp, sentence: str) -> List[str]:
    include_pos = {"NOUN", "ADJ", "VERB", "PROPN", "NUM"}
    doc = nlp(sentence)
    seen = set()
    words = []
    for token in doc:
        if token.pos_ in include_pos:
            word = token.text.strip()
            if not word:
                continue
            if word in seen:
                continue
            seen.add(word)
            words.append(word)
    return words


def build_step_keywords(step_text: str, step_probs: List[float], nlp) -> Dict[str, List[float]]:
    text = clean_step_text(step_text)
    keywords = extract_content_words(nlp, text)
    if not keywords:
        return {}, {}

    if step_probs:
        rep_prob = float(sum(step_probs) / len(step_probs))
    else:
        rep_prob = float("nan")

    probs = {k: [rep_prob] for k in keywords}
    contrib = {k: 1 for k in keywords}
    return probs, contrib


def build_response_keyword_payload(tokenizer, nlp, response_text: str, response_probs: List[float]):
    llm_answer, steps_dict, _ = parse_response_to_dict(response_text)
    if llm_answer is None or not steps_dict:
        return None

    cursor = 0
    keywords_probabilities = {}
    keywords_contributions = {}

    for step_name, step_text in steps_dict.items():
        step_ids = tokenizer.encode(step_text, add_special_tokens=False)
        step_len = len(step_ids)
        step_slice = response_probs[cursor : cursor + step_len] if step_len > 0 else []
        cursor += step_len

        step_probs, step_contrib = build_step_keywords(step_text, step_slice, nlp)
        if step_probs:
            keywords_probabilities[step_name] = step_probs
            keywords_contributions[step_name] = step_contrib

    return {
        "keywords_probabilities": keywords_probabilities,
        "keywords_contributions": keywords_contributions,
        "cot_uq_success_flag": len(keywords_probabilities) > 0,
    }


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s",
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="0", help="reserved")
    parser.add_argument("--model_dir", type=str, required=True, help="Tokenizer model directory")
    parser.add_argument("--use_greedy", action="store_true", help="Use greedy generation only")
    parser.add_argument("--prompt_type", type=str, default="src", choices=["0-100", "standard", "src"])
    parser.add_argument("--run_setting", type=str, default="", help="run setting")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    os.environ["HF_DATASETS_CACHE"] = config.hf_datasets_cache
    os.chdir(config.run_dir)

    if args.run_setting:
        run_setting = args.run_setting
    else:
        with open(f"{config.output_dir}/run_setting.txt", "r") as f:
            run_setting = f.read().strip()

    with open(f"{config.output_dir}/{run_setting}/generations.pkl", "rb") as f:
        generations = pickle.load(f)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    nlp = spacy.load("en_core_web_sm")

    result = []
    for sample in tqdm.tqdm(generations, desc="Extracting CoT-UQ keywords from generations"):
        sample_payload = {
            "id": sample["id"],
            "question": sample["question"],
            "responses": [],
        }

        if args.use_greedy:
            texts = [sample.get("most_likely_generation", "")]
            probs_list = [sample.get("most_likely_generation_probs", [])]
            flags = [bool(sample.get("most_likely_generation_success_flag", True))]
        else:
            texts = sample.get("generated_texts", [])
            probs_list = sample.get("generated_probs", [])
            flags = sample.get("generated_success_flag", [True] * len(texts))

        for idx, (txt, probs, ok) in enumerate(zip(texts, probs_list, flags)):
            item = {
                "response_idx": idx,
                "keywords_probabilities": None,
                "keywords_contributions": None,
                "cot_uq_success_flag": False,
            }
            if ok and txt and probs:
                payload = build_response_keyword_payload(tokenizer, nlp, txt, probs)
                if payload is not None:
                    item.update(payload)
            sample_payload["responses"].append(item)

        result.append(sample_payload)

    suffix = f"{args.prompt_type}_greedy" if args.use_greedy else f"{args.prompt_type}_sampled"
    save_path = f"{config.output_dir}/{run_setting}/keywords_probabilities_{suffix}.json"

    with open(save_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info("Saved keyword probabilities to %s", save_path)


if __name__ == "__main__":
    main()

