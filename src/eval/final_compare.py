import argparse
import json
import os
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score


DEFAULT_RUN_SETTINGS = [
    "Llama-3.1-8B-Instruct_triviaqa_cot_uq_validation_0-500_src_fraction_0.6_max_length_512_num_generations_5_temperature_1.0_top_k_5_top_p_0.95_decode_method_greedy_seed_42",
    "Qwen3-4B_triviaqa_cot_uq_validation_0-500_src_fraction_0.6_max_length_512_num_generations_5_temperature_1.0_top_k_5_top_p_0.95_decode_method_greedy_seed_42",
    "Qwen3-14B_triviaqa_cot_uq_validation_0-500_src_fraction_0.6_max_length_512_num_generations_5_temperature_1.0_top_k_5_top_p_0.95_decode_method_greedy_seed_42",
]


def _maybe_load_pickle(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_pickle_with_fallback(base_path, file_candidates, required=True):
    for file_name in file_candidates:
        data = _maybe_load_pickle(os.path.join(base_path, file_name))
        if data is not None:
            return data, file_name
    if required:
        raise FileNotFoundError(
            f"None of the required files exist under {base_path}: {file_candidates}"
        )
    return None, None


def _load_cotuq_scores(base_path):
    candidates = [
        "cotuq_scores.pkl",
        "Cot_uq_token_sar_scores_src_sampled.pkl",
        "Cot_uq_probas_mean_scores_src_sampled.pkl",
        "Cot_uq_probas_min_scores_src_sampled.pkl",
        "Cot_uq_token_sar_scores_src_greedy.pkl",
        "Cot_uq_probas_mean_scores_src_greedy.pkl",
        "Cot_uq_probas_min_scores_src_greedy.pkl",
    ]
    for name in candidates:
        path = os.path.join(base_path, name)
        data = _maybe_load_pickle(path)
        if data is not None:
            return data, name
    return None, None


def _resolve_json_input(base_path, candidates, required=True):
    for file_name in candidates:
        path = os.path.join(base_path, file_name)
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f), file_name
    if required:
        raise FileNotFoundError(f"None of the required files exist under {base_path}: {candidates}")
    return None, None


def aggregate_probs(token_list, agg_type: str) -> float:
    def geometric_mean(arr):
        arr = np.array(arr)
        arr = arr[arr > 0]
        if len(arr) == 0:
            return float("nan")
        return np.exp(np.mean(np.log(arr)))

    arr = [p if p > 0 else 1e-10 for p in token_list]
    if not arr:
        return float("nan")
    if agg_type == "geometric":
        return float(geometric_mean(arr))
    if agg_type == "mean":
        return float(np.mean(arr))
    if agg_type == "min":
        return float(np.min(arr))
    if agg_type == "max":
        return float(np.max(arr))
    if agg_type == "prod":
        return float(np.prod(arr))
    raise ValueError(f"Unsupported aggregation type: {agg_type}")


def calc_auroc(scores, labels):
    scores = np.array(scores)
    labels = np.array(labels)
    mask = ~np.isnan(scores)
    valid_num = int(np.sum(mask))
    total_num = len(scores)
    if valid_num == 0:
        return np.nan, f"0/{total_num}"
    try:
        auc = roc_auc_score(1 - labels[mask], scores[mask])
        return auc, f"{valid_num}/{total_num}"
    except Exception:
        return np.nan, f"{valid_num}/{total_num}"


def get_probs_matrix_from_result(result):
    all_probs_matrix = []
    for sample in result:
        sample_probs_matrix = []
        for response in sample["response_picked_tokens_probs"]:
            sentence_probs = []
            for sentence_info in response:
                sentence_probs.append(sentence_info["sentence_prob"])
            sample_probs_matrix.append(sentence_probs)
        all_probs_matrix.append(sample_probs_matrix)
    return all_probs_matrix


def aggregate_uncertainties(
    confidence_information: torch.Tensor,
    ids_list: torch.Tensor,
    semantic_clusters,
    agg_type="semantic_cluster_entailment",
):
    result = []
    for confidence, sample_id in zip(confidence_information, ids_list):
        if agg_type in {
            "semantic_cluster",
            "semantic_cluster_raw",
            "semantic_cluster_entailment",
        }:
            cluster_info = semantic_clusters[int(sample_id)]
            if agg_type == "semantic_cluster":
                semantic_set_ids = torch.tensor(cluster_info["semantic_set_ids"])
            elif agg_type == "semantic_cluster_raw":
                semantic_set_ids = torch.tensor(cluster_info["semantic_set_ids_raw"])
            else:
                semantic_set_ids = torch.tensor(cluster_info["semantic_set_ids_entailment"])

            semantic_cluster_entropy = []
            for semantic_id in torch.unique(semantic_set_ids):
                semantic_cluster_entropy.append(
                    torch.log(torch.sum(confidence[semantic_set_ids == semantic_id], dim=0))
                )

            semantic_cluster_entropy = torch.tensor(semantic_cluster_entropy)
            semantic_cluster_entropy = -torch.sum(semantic_cluster_entropy, dim=0) / torch.tensor(
                semantic_cluster_entropy.shape[0]
            )
            confidence_uncertainty_score = 1 - torch.exp(-semantic_cluster_entropy) / confidence.shape[0]
            result.append(confidence_uncertainty_score)
        elif agg_type == "mean":
            result.append(1 - torch.mean(confidence))
        elif agg_type == "geometric":
            result.append(1 - torch.exp(torch.mean(torch.log(confidence + 1e-10))))
        elif agg_type == "weighted_by_sorted_s":
            denom = torch.sum(torch.arange(len(confidence), 0, -1, dtype=torch.float32))
            result.append(1 - torch.sum(confidence) / denom)
        else:
            raise ValueError(f"Unsupported uncertainty aggregation type: {agg_type}")
    return result


def process_run_setting(run_setting, results_root, split_method):
    base_path = os.path.join(results_root, run_setting)
    confidence_keywords_probs, keyword_file = _resolve_json_input(
        base_path,
        ["confidence_keywords_probs.json"],
    )
    with open(os.path.join(base_path, "generations.pkl"), "rb") as f:
        generations = pickle.load(f)
    support_results, support_file = _resolve_json_input(
        base_path,
        [f"support_uncertainty_luqpair_{split_method}.json"],
    )
    with open(os.path.join(base_path, "semantic_clusters.pkl"), "rb") as f:
        semantic_clusters = pickle.load(f)
    with open(os.path.join(base_path, "AUROC_labels.json"), "r") as infile:
        auroc_labels = json.load(infile)
    print(f"[info] {run_setting} support file={support_file}, keyword file={keyword_file}")

    luq_scores, _ = _load_pickle_with_fallback(base_path, ["luq_scores.pkl"])
    pe_scores, pe_file = _load_pickle_with_fallback(
        base_path,
        ["predictive_entropy_scores.pkl", "predictive_entropy_from_generations_scores.pkl"],
    )
    lnpe_scores, lnpe_file = _load_pickle_with_fallback(
        base_path,
        [
            "len_normed_predictive_entropy_scores.pkl",
            "len_normed_predictive_entropy_from_generations_scores.pkl",
        ],
    )
    se_scores, se_file = _load_pickle_with_fallback(
        base_path,
        ["semantic_entropy_scores.pkl", "semantic_entropy_from_generations_scores.pkl"],
    )
    sentence_sar_scores, sent_file = _load_pickle_with_fallback(
        base_path,
        ["sentence_sar_scores.pkl", "sentence_sar_from_generations_scores.pkl"],
    )
    token_sar_scores, tok_file = _load_pickle_with_fallback(
        base_path,
        ["token_sar_scores.pkl", "token_sar_from_generations_scores.pkl"],
    )
    print(
        f"[info] {run_setting} baseline score files: "
        f"PE={pe_file}, LNPE={lnpe_file}, SE={se_file}, "
        f"sentence-SAR={sent_file}, token-SAR={tok_file}"
    )
    cotuq_scores, cotuq_file = _load_cotuq_scores(base_path)
    if cotuq_file:
        print(f"[info] {run_setting} loaded cotuq from {cotuq_file}")

    keyword_agg_type = "min"
    sentence_agg_type = "mean"
    response_agg_type = "prod"
    uncertainty_agg_type = "semantic_cluster_entailment"
    information_agg_type = "mean"
    uncertainty_agg_type2 = "semantic_cluster_entailment"

    result = []
    for item in confidence_keywords_probs:
        response_picked_tokens_list = []
        for response_data in item["responses_data"]:
            picked_tokens_probs = []
            for sentence_data in response_data:
                sentence_agg_token_probs = {}
                if sentence_data["keywords_probs"] is None:
                    picked_tokens_probs.append(
                        {
                            "sentence": sentence_data["sentence"],
                            "agg_token_probs": -1,
                            "token_aggregate_method": None,
                        }
                    )
                    continue

                for keyword_k, keyword_v in sentence_data["keywords_probs"].items():
                    sentence_agg_token_probs[keyword_k] = aggregate_probs(keyword_v, agg_type=keyword_agg_type)

                picked_tokens_probs.append(
                    {
                        "sentence": sentence_data["sentence"],
                        "agg_token_probs": sentence_agg_token_probs,
                        "token_aggregate_method": keyword_agg_type,
                    }
                )
            response_picked_tokens_list.append(picked_tokens_probs)

        result.append(
            {
                "id": item["id"],
                "question": item["question"],
                "semantic_set_ids": item["semantic_set_ids"],
                "responses": item["responses"],
                "prompt_text": item["prompt_text"],
                "answer": item["answer"],
                "splited_responses": item["splited_responses"],
                "response_picked_tokens_probs": response_picked_tokens_list,
            }
        )

    for sample in result:
        for response in sample["response_picked_tokens_probs"]:
            for sentence_info in response:
                if sentence_info["agg_token_probs"] == -1:
                    sentence_info["sentence_prob"] = -1
                    sentence_info["sentence_aggregated_method"] = None
                    continue
                sentence_prob = aggregate_probs(
                    sentence_info["agg_token_probs"].values(),
                    agg_type=sentence_agg_type,
                )
                sentence_info["sentence_prob"] = sentence_prob
                sentence_info["sentence_aggregated_method"] = sentence_agg_type

    probs_matrices = get_probs_matrix_from_result(result)
    probs_matrices_agg_probs = []
    for probs_matrix in probs_matrices:
        probs_matrix_agg_probs = []
        for probs_sentence in probs_matrix:
            probs_matrix_agg_probs.append(aggregate_probs(probs_sentence, response_agg_type))
        probs_matrices_agg_probs.append(probs_matrix_agg_probs)

    confidence_information = torch.tensor(probs_matrices_agg_probs)
    support_information = torch.tensor(
        [[1 - score for score in item["uncertainty_scores"]] for item in support_results]
    )
    ids = torch.tensor([gen["id"] for gen in generations])
    labels = np.array([item["most_cluster_label"] for item in auroc_labels])

    aggregation_strategies = {
        "mean": lambda scores: torch.mean(
            torch.stack([torch.tensor(s) if not isinstance(s, torch.Tensor) else s for s in scores]),
            dim=0,
        )
    }

    confidence_level_scores = aggregate_uncertainties(
        confidence_information, ids, semantic_clusters, uncertainty_agg_type
    )
    support_level_scores = aggregate_uncertainties(
        support_information, ids, semantic_clusters, uncertainty_agg_type
    )

    combined_information = [
        aggregation_strategies[information_agg_type]([c, s])
        for c, s in zip(confidence_information, support_information)
    ]
    combined_scores = aggregate_uncertainties(
        torch.stack([torch.tensor(x) for x in combined_information]),
        ids,
        semantic_clusters,
        uncertainty_agg_type2,
    )

    baseline_scores = {
        "predictive-entropy-from-generations": pe_scores,
        "len-normed-predictive-entropy-from-generations": lnpe_scores,
        "sentence-sar-from-generations": sentence_sar_scores,
        "token-sar-from-generations": token_sar_scores,
        "semantic-entropy-from-generations": se_scores,
        "LUQ": luq_scores,
    }
    if cotuq_scores is not None:
        baseline_scores["cotuq"] = cotuq_scores
    co_su_uq_scores = {
        "combined_scores": combined_scores,
        "confidence_level": confidence_level_scores,
        "support_level": support_level_scores,
    }

    metrics = {}
    for name, sc in {**baseline_scores, **co_su_uq_scores}.items():
        if isinstance(sc, torch.Tensor):
            sc = sc.detach().cpu().numpy()
        else:
            try:
                if len(sc) > 0 and isinstance(sc[0], torch.Tensor):
                    sc = torch.stack(sc).detach().cpu().numpy()
            except Exception:
                pass

        auc, _ = calc_auroc(sc, labels)
        if np.isnan(auc):
            metrics[name] = "nan"
        else:
            metrics[name] = f"{auc:.6f}"

    model = run_setting.split("_")[0]
    dataset = run_setting.split("_")[1]
    new_row = {"model": model, "dataset": dataset}
    new_row.update(metrics)
    return new_row


def run_all(run_settings, results_root, split_method="step_answer", max_workers=8):
    rows = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_run = {
            executor.submit(process_run_setting, run_setting, results_root, split_method): run_setting
            for run_setting in run_settings
        }
        for future in as_completed(future_to_run):
            run_setting = future_to_run[future]
            try:
                rows.append(future.result())
            except Exception as exc:
                print(f"{run_setting} failed: {exc}")

    new_df = pd.DataFrame(rows)
    if new_df.empty:
        return new_df

    cols = ["model", "dataset"] + [col for col in new_df.columns if col not in ["model", "dataset"]]
    new_df = new_df[cols]

    method_col_map = {
        "PE": "predictive-entropy-from-generations",
        "LN-PE": "len-normed-predictive-entropy-from-generations",
        "sentence-sar": "sentence-sar-from-generations",
        "token-sar": "token-sar-from-generations",
        "SE": "semantic-entropy-from-generations",
        "LUQ": "LUQ",
        "CoT-UQ": "cotuq",
    }
    base_cols = ["model", "dataset"]
    main_methods = [
        method_col_map[k]
        for k in ["PE", "LN-PE", "sentence-sar", "token-sar", "SE", "LUQ", "CoT-UQ"]
        if method_col_map[k] in new_df.columns
    ]
    extra_methods = ["combined_scores", "confidence_level", "support_level"]
    ordered_cols = base_cols + main_methods + [col for col in extra_methods if col in new_df.columns]

    model_order = [
        "Llama-3.1-8B-Instruct",
        "Qwen3-4B",
        "Qwen3-14B",
        "Qwen3-32B",
    ]
    dataset_order = ["gsm8k", "math", "hotpotqa", "2wikimultihopqa", "medqa", "triviaqa"]

    new_df["dataset"] = new_df["dataset"].str.upper()
    new_df["model"] = pd.Categorical(new_df["model"], categories=model_order, ordered=True)
    new_df["dataset"] = pd.Categorical(
        new_df["dataset"],
        categories=[d.upper() for d in dataset_order],
        ordered=True,
    )

    new_df = new_df.sort_values(by=["model", "dataset"]).reset_index(drop=True)
    return new_df[ordered_cols]


def parse_args():
    parser = argparse.ArgumentParser(description="Final CoSu-UQ scoring and baseline AUROC comparison")
    parser.add_argument(
        "--results-root",
        type=str,
        default="./results",
        help="Root directory containing per-run result folders.",
    )
    parser.add_argument(
        "--run-settings-json",
        type=str,
        default="",
        help="Optional JSON file path containing a list[str] of run_setting values.",
    )
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument(
        "--split_method",
        type=str,
        default="step_answer",
        choices=["step_answer", "spacy", "nltk"],
        help="Split method used in stage3 support output filename.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="./results/final_uq_baseline_compare.csv",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_settings = DEFAULT_RUN_SETTINGS
    if args.run_settings_json:
        with open(args.run_settings_json, "r") as f:
            run_settings = json.load(f)

    df = run_all(
        run_settings,
        args.results_root,
        split_method=args.split_method,
        max_workers=args.max_workers,
    )
    if df.empty:
        print("No run_setting was processed successfully.")
        return

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    print("Full table:")
    print(df)

    medqa_df = df[df["dataset"].astype(str).str.contains("MEDQA", na=False)].reset_index(drop=True)
    print("\nMedQA subset:")
    print(medqa_df)
    print(f"\nSaved to: {args.output_csv}")


if __name__ == "__main__":
    main()
