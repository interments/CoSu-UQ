import argparse
import json
import logging
import os
import pickle

import torch

import config
from utils.cot_uq_utils import extract_p, extract_p_t_importance, weighted_sum


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s",
)
logger = logging.getLogger(__name__)


def compute_single_response_score(question, keyword_token_probability, contribution_scores, aggregated_method, model_dir, measure_model_name):
    if aggregated_method == "probas_min":
        probabilities, contribution_dict = extract_p(keyword_token_probability, contribution_scores, use_min=True)
    elif aggregated_method == "probas_mean":
        probabilities, contribution_dict = extract_p(keyword_token_probability, contribution_scores, use_min=False)
    elif aggregated_method == "token_sar":
        from sentence_transformers.cross_encoder import CrossEncoder
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_dir, cache_dir=config.hf_model_cache)
        measure_model = CrossEncoder(model_name=measure_model_name, cache_dir=config.hf_model_cache, num_labels=1)
        probabilities, contribution_dict = extract_p_t_importance(
            question,
            keyword_token_probability,
            tokenizer,
            measure_model,
            contribution_scores,
        )
    else:
        raise ValueError(f"Unknown aggregated method: {aggregated_method}")

    if not probabilities:
        return float("nan")

    probabilities = {key: weighted_sum(value) for key, value in probabilities.items()}
    contributions = {key: sum(value) / len(value) for key, value in contribution_dict.items()}

    total_sum = sum(probabilities[key] * contributions[key] for key in probabilities)
    total_weight = sum(contributions[key] for key in contributions)

    if total_weight == 0:
        p_list = [v for v in probabilities.values()]
        confidence = sum(p_list) / len(p_list)
    else:
        confidence = total_sum / total_weight

    return float(1 - confidence)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--measure_model", type=str, default="cross-encoder/stsb-roberta-large")
    parser.add_argument(
        "--aggregated_method",
        type=str,
        default="token_sar",
        choices=["probas_mean", "probas_min", "token_sar"],
    )
    parser.add_argument("--prompt_type", type=str, default="src", choices=["0-100", "standard", "src"])
    parser.add_argument("--use_greedy", action="store_true")
    parser.add_argument("--run_setting", type=str, default="")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    os.environ["HF_DATASETS_CACHE"] = config.hf_datasets_cache
    os.chdir(config.run_dir)

    if args.run_setting:
        run_setting = args.run_setting
    else:
        with open(f"{config.output_dir}/run_setting.txt", "r") as f:
            run_setting = f.read().strip()

    suffix = f"{args.prompt_type}_greedy" if args.use_greedy else f"{args.prompt_type}_sampled"
    input_path = f"{config.output_dir}/{run_setting}/keywords_probabilities_{suffix}.json"

    with open(input_path, "r") as infile:
        keyword_samples = json.load(infile)

    sample_scores = []
    for sample in keyword_samples:
        question = sample["question"]
        response_scores = []

        for resp in sample.get("responses", []):
            if not resp.get("cot_uq_success_flag", False):
                continue
            keyword_token_probability = resp.get("keywords_probabilities")
            contribution_scores = resp.get("keywords_contributions")
            if not keyword_token_probability or not contribution_scores:
                continue
            try:
                score = compute_single_response_score(
                    question=question,
                    keyword_token_probability=keyword_token_probability,
                    contribution_scores=contribution_scores,
                    aggregated_method=args.aggregated_method,
                    model_dir=args.model_dir,
                    measure_model_name=args.measure_model,
                )
                if not torch.isnan(torch.tensor(score)):
                    response_scores.append(score)
            except Exception as e:
                logger.info("Failed response score on sample id=%s, response_idx=%s: %s", sample.get("id"), resp.get("response_idx"), e)

        if response_scores:
            sample_scores.append(torch.tensor(sum(response_scores) / len(response_scores)))
        else:
            sample_scores.append(torch.tensor(float("nan")))

    run_dir = f"{config.output_dir}/{run_setting}"
    save_path = f"{run_dir}/Cot_uq_{args.aggregated_method}_scores_{suffix}.pkl"
    with open(save_path, "wb") as outfile:
        pickle.dump(sample_scores, outfile)

    canonical_path = f"{run_dir}/cotuq_scores.pkl"
    with open(canonical_path, "wb") as outfile:
        pickle.dump(sample_scores, outfile)

    logger.info("Saved CoT-UQ scores to %s", save_path)
    logger.info("Saved canonical CoT-UQ scores to %s", canonical_path)


if __name__ == "__main__":
    main()
