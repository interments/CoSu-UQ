import argparse
import json
import logging
import os
import pickle

import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import config
from baselines.luq_support_calculators import LUQCalculator, LUQPairCalculator
from utils.sentence_splitters import get_sentence_splitter


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s",
)
logger = logging.getLogger(__name__)


def _to_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _resolve_run_setting(run_setting_arg: str) -> str:
    if run_setting_arg:
        return run_setting_arg
    with open(f"{config.output_dir}/run_setting.txt", "r") as f:
        return f.read().strip()


def _build_responses(result_item: dict, use_greedy: bool):
    if use_greedy:
        return [result_item.get("most_likely_generation", "")] + [text for text in result_item["generated_texts"]]
    return [text for text in result_item["generated_texts"]]


def _support_output_file(
    output_dir: str,
    run_setting: str,
    split_method: str,
    luq_method: str,
    use_greedy: bool,
) -> str:
    suffix = "_use_greedy" if use_greedy else ""
    method_tag = luq_method.lower()
    return f"{output_dir}/{run_setting}/support_uncertainty_{method_tag}_{split_method}{suffix}.json"


def _build_calculator(luq_method: str, nli_model, nli_tokenizer, sentence_splitter):
    if luq_method == "LUQ":
        return LUQCalculator(
            nli_model=nli_model,
            nli_tokenizer=nli_tokenizer,
            sentence_splitter=sentence_splitter,
            device=nli_model.device.type,
        )
    if luq_method == "LUQPair":
        return LUQPairCalculator(
            nli_model=nli_model,
            nli_tokenizer=nli_tokenizer,
            sentence_splitter=sentence_splitter,
            device=nli_model.device.type,
        )
    raise ValueError(f"Unsupported luq_method: {luq_method}")


def main():
    parser = argparse.ArgumentParser(description="Compute support uncertainty scores")
    parser.add_argument("--device", type=str, default="0", help="CUDA device id")
    parser.add_argument(
        "--model_name",
        type=str,
        default="potsawee/deberta-v3-large-mnli",
        help="NLI model name",
    )
    parser.add_argument(
        "--luq_method",
        type=str,
        default="LUQPair",
        choices=["LUQ", "LUQPair"],
        help="Support uncertainty method",
    )
    parser.add_argument("--run_setting", type=str, default="", help="Run setting")
    parser.add_argument(
        "--split_method",
        type=str,
        default="step_answer",
        choices=["nltk", "spacy", "step_answer"],
        help="Sentence split method",
    )
    parser.add_argument("--use_greedy", type=str, default="False", help="Whether to include greedy output")
    parser.add_argument("--save_matrix", action="store_true", help="Whether to save nli_probability_matrix")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    os.environ["HF_DATASETS_CACHE"] = config.hf_datasets_cache
    os.chdir(config.run_dir)

    sentence_splitter = get_sentence_splitter(args.split_method, logger=logger)
    run_setting = _resolve_run_setting(args.run_setting)
    use_greedy = _to_bool(args.use_greedy)

    model_device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "" else "cpu"
    if model_device == "cuda":
        try:
            import torch

            if not torch.cuda.is_available():
                model_device = "cpu"
        except Exception:
            model_device = "cpu"

    nli_model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        cache_dir=config.hf_model_cache,
    ).to(model_device)
    nli_tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=config.hf_model_cache)
    calculator = _build_calculator(args.luq_method, nli_model, nli_tokenizer, sentence_splitter)

    input_file = f"{config.output_dir}/{run_setting}/generations.pkl"
    with open(input_file, "rb") as infile:
        generated_results = pickle.load(infile)

    support_results = []
    for result in tqdm.tqdm(generated_results, desc="Computing support uncertainty"):
        responses = _build_responses(result, use_greedy=use_greedy)
        uncertainty_score = -1
        uncertainty_scores = []
        splited_response = None
        nli_probability_matrix = None
        error_msg = ""

        try:
            uncertainty_score, uncertainty_scores = calculator.compute_uncertainty_score(responses)
            splited_response = calculator.last_splited_response
            nli_probability_matrix = calculator.last_nli_probability_matrix
        except Exception as exc:
            error_msg = str(exc)
            logger.error("Error processing sample %s: %s", result["id"], exc)

        result_item = {
            "id": result["id"],
            "support_method": args.luq_method,
            "prompt_text": result["prompt_text"],
            "generated_texts_greedy": result.get("most_likely_generation", ""),
            "generated_texts": result["generated_texts"],
            "answer": result["answer"],
            "responses": responses,
            "splited_responses": splited_response,
            "uncertainty_scores": uncertainty_scores,
            "score": uncertainty_score,
        }
        if args.save_matrix and nli_probability_matrix is not None:
            result_item["nli_probability_matrix"] = nli_probability_matrix
        if error_msg:
            result_item["error"] = error_msg
        support_results.append(result_item)

    output_file = _support_output_file(
        output_dir=config.output_dir,
        run_setting=run_setting,
        split_method=args.split_method,
        luq_method=args.luq_method,
        use_greedy=use_greedy,
    )
    with open(output_file, "w", encoding="utf-8") as outfile:
        json.dump(support_results, outfile, indent=4, ensure_ascii=False)
    logger.info("Support uncertainty results saved to %s", output_file)


if __name__ == "__main__":
    main()
