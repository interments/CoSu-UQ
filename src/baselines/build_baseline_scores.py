import argparse
import json
import os
import pickle

import torch


def _safe_load_pickle(path, required=True):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    if required:
        raise FileNotFoundError(f"Required file not found: {path}")
    return None


def _safe_load_json(path, required=True):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    if required:
        raise FileNotFoundError(f"Required file not found: {path}")
    return None


def load_cached_minimal(output_dir, run_setting, token_impt_meas_model, senten_sim_meas_model):
    run_dir = os.path.join(output_dir, run_setting)

    token_model_key = token_impt_meas_model.replace("/", "-")
    sent_model_key = senten_sim_meas_model.replace("/", "-")

    generations = _safe_load_pickle(os.path.join(run_dir, "generations.pkl"), required=True)
    semantic_clusters = _safe_load_pickle(os.path.join(run_dir, "semantic_clusters.pkl"), required=True)

    sentence_similarities = _safe_load_pickle(
        os.path.join(run_dir, f"sentence_similarities_{sent_model_key}.pkl"),
        required=True,
    )

    token_importance_from_generation = _safe_load_pickle(
        os.path.join(run_dir, f"tokenwise_importance_{token_model_key}_from_generation.pkl"),
        required=True,
    )

    return {
        "run_dir": run_dir,
        "generations": generations,
        "semantic_clusters": semantic_clusters,
        "sentence_similarities": sentence_similarities,
        "most_likely_sampled_token_importance_from_generation": token_importance_from_generation,
    }


def sentence_sar_from_generations(cached_data, t=0.001, num_generation=None):
    generations = cached_data["generations"]
    sentence_similarities = cached_data["sentence_similarities"]
    scores = []

    def semantic_weighted_log(similarities, entropies, temperature):
        entropies = entropies.to(torch.float64)
        probs = torch.exp(-entropies)
        weighted_entropy = []
        for idx, prob in enumerate(probs):
            sim = torch.tensor(similarities[idx], dtype=torch.float64)
            others = torch.cat([probs[:idx], probs[idx + 1 :]])
            w_ent = -torch.log(prob + ((sim / temperature) * others).sum())
            weighted_entropy.append(w_ent)
        return torch.tensor(weighted_entropy)

    for sample_idx, sample in enumerate(generations):
        gen_probs_list = sample["generated_probs"]
        similarity = sentence_similarities[sample_idx]
        if num_generation is not None:
            gen_probs_list = gen_probs_list[:num_generation]
            similarity = similarity[:num_generation]

        gen_scores = []
        for probs in gen_probs_list:
            probs_tensor = torch.tensor(probs, dtype=torch.float32)
            token_entropy = -torch.log(probs_tensor + 1e-10)
            gen_scores.append(token_entropy.mean())

        if not gen_scores:
            scores.append(torch.tensor(float("nan")))
            continue

        gen_scores = torch.tensor(gen_scores)
        gen_scores = semantic_weighted_log(similarity, gen_scores, temperature=t)
        scores.append(gen_scores.mean())
    return scores


def token_sar_from_generations(cached_data, num_generation=None):
    generations = cached_data["generations"]
    token_importance = cached_data["most_likely_sampled_token_importance_from_generation"]
    scores = []

    for sample_idx, sample in enumerate(generations):
        gen_probs_list = sample["generated_probs"]
        if num_generation is not None:
            gen_probs_list = gen_probs_list[:num_generation]

        gen_scores = []
        for k, probs in enumerate(gen_probs_list):
            probs_tensor = torch.tensor(probs, dtype=torch.float32)
            token_entropy = -torch.log(probs_tensor + 1e-10)
            importance = token_importance[sample_idx * len(gen_probs_list) + k]
            if len(importance) == len(token_entropy):
                weighted_score = (importance / importance.sum()) * token_entropy
                gen_scores.append(torch.tensor(weighted_score).sum())
            elif len(importance) == len(token_entropy) + 1:
                importance = importance[: len(token_entropy)]
                weighted_score = (importance / importance.sum()) * token_entropy
                gen_scores.append(torch.tensor(weighted_score).sum())
            else:
                gen_scores.append(torch.tensor(float("nan")))

        if gen_scores:
            scores.append(torch.tensor(gen_scores).mean())
        else:
            scores.append(torch.tensor(float("nan")))
    return scores


def semantic_entropy_from_generations(cached_data, num_generation=None):
    generations = cached_data["generations"]
    semantic_clusters = cached_data["semantic_clusters"]
    llh_shift = torch.tensor(5.0)
    scores = []

    for sample in generations:
        gen_probs_list = sample["generated_probs"]
        sample_id = sample["id"]
        semantic_set_ids = semantic_clusters[sample_id]["semantic_set_ids_raw"]

        if num_generation is not None:
            gen_probs_list = gen_probs_list[:num_generation]
            semantic_set_ids = semantic_set_ids[:num_generation]

        gen_entropy = []
        for probs in gen_probs_list:
            probs_tensor = torch.tensor(probs, dtype=torch.float32)
            token_entropy = -torch.log(probs_tensor + 1e-10)
            gen_entropy.append(token_entropy.mean())

        if not gen_entropy:
            scores.append(torch.tensor(float("nan")))
            continue

        gen_entropy = torch.tensor(gen_entropy)
        semantic_set_ids = torch.tensor(semantic_set_ids)
        semantic_cluster_entropy = []
        for semantic_id in torch.unique(semantic_set_ids):
            mask = semantic_set_ids == semantic_id
            semantic_cluster_entropy.append(torch.logsumexp(-gen_entropy[mask], dim=0))

        semantic_cluster_entropy = torch.tensor(semantic_cluster_entropy) - llh_shift
        semantic_cluster_entropy = -torch.sum(semantic_cluster_entropy, dim=0) / torch.tensor(
            semantic_cluster_entropy.shape[0]
        )
        scores.append(torch.mean(semantic_cluster_entropy))
    return scores


def len_normed_predictive_entropy_from_generations(cached_data, num_generation=None):
    generations = cached_data["generations"]
    scores = []
    for sample in generations:
        gen_probs_list = sample["generated_probs"]
        if num_generation is not None:
            gen_probs_list = gen_probs_list[:num_generation]
        gen_entropies = []
        for probs in gen_probs_list:
            probs_tensor = torch.tensor(probs, dtype=torch.float32)
            token_entropy = -torch.log(probs_tensor + 1e-10)
            gen_entropies.append(token_entropy.mean())
        if gen_entropies:
            scores.append(torch.mean(torch.stack(gen_entropies)))
        else:
            scores.append(torch.tensor(float("nan")))
    return scores


def predictive_entropy_from_generations(cached_data, num_generation=None):
    generations = cached_data["generations"]
    scores = []
    for sample in generations:
        gen_probs_list = sample["generated_probs"]
        if num_generation is not None:
            gen_probs_list = gen_probs_list[:num_generation]
        gen_entropies = []
        for probs in gen_probs_list:
            probs_tensor = torch.tensor(probs, dtype=torch.float32)
            token_entropy = -torch.log(probs_tensor + 1e-10)
            gen_entropies.append(token_entropy.sum())
        if gen_entropies:
            scores.append(torch.mean(torch.stack(gen_entropies)))
        else:
            scores.append(torch.tensor(float("nan")))
    return scores


def luq_scores(output_dir, run_setting, split_method="step_answer"):
    luq_path = os.path.join(output_dir, run_setting, f"support_uncertainty_luq_{split_method}.json")
    if not os.path.exists(luq_path):
        raise FileNotFoundError(f"Required support score file not found: {luq_path}")
    total_scores = _safe_load_json(luq_path, required=True)
    scores = []
    for item in total_scores:
        score = item.get("score", -1)
        if score == -1:
            scores.append(torch.tensor(float("nan")))
        else:
            scores.append(torch.tensor(score))
    return scores


def _dump_scores(path, scores):
    with open(path, "wb") as f:
        pickle.dump(scores, f)


def _dump_scores_multi(run_dir, file_names, scores):
    for file_name in file_names:
        _dump_scores(os.path.join(run_dir, file_name), scores)


def main():
    parser = argparse.ArgumentParser(description="Minimal baseline cache + score pipeline for CoSu-UQ.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--run_setting", type=str, required=True)
    parser.add_argument("--senten_sim_meas_model", type=str, default="cross-encoder/stsb-roberta-large")
    parser.add_argument("--token_impt_meas_model", type=str, default="cross-encoder/stsb-roberta-large")
    parser.add_argument("--num_generation", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.001)
    parser.add_argument("--luq_split_method", type=str, default="step_answer")
    args = parser.parse_args()

    cached = load_cached_minimal(
        output_dir=args.output_dir,
        run_setting=args.run_setting,
        token_impt_meas_model=args.token_impt_meas_model,
        senten_sim_meas_model=args.senten_sim_meas_model,
    )

    run_dir = cached["run_dir"]

    pe = predictive_entropy_from_generations(cached, num_generation=args.num_generation)
    lnpe = len_normed_predictive_entropy_from_generations(cached, num_generation=args.num_generation)
    se = semantic_entropy_from_generations(cached, num_generation=args.num_generation)
    sent_sar = sentence_sar_from_generations(cached, t=args.temperature, num_generation=args.num_generation)
    tok_sar = token_sar_from_generations(cached, num_generation=args.num_generation)
    luq = luq_scores(
        output_dir=args.output_dir,
        run_setting=args.run_setting,
        split_method=args.luq_split_method,
    )

    _dump_scores_multi(
        run_dir,
        ["predictive_entropy_scores.pkl", "predictive_entropy_from_generations_scores.pkl"],
        pe,
    )
    _dump_scores_multi(
        run_dir,
        [
            "len_normed_predictive_entropy_scores.pkl",
            "len_normed_predictive_entropy_from_generations_scores.pkl",
        ],
        lnpe,
    )
    _dump_scores_multi(
        run_dir,
        ["semantic_entropy_scores.pkl", "semantic_entropy_from_generations_scores.pkl"],
        se,
    )
    _dump_scores_multi(
        run_dir,
        ["sentence_sar_scores.pkl", "sentence_sar_from_generations_scores.pkl"],
        sent_sar,
    )
    _dump_scores_multi(
        run_dir,
        ["token_sar_scores.pkl", "token_sar_from_generations_scores.pkl"],
        tok_sar,
    )
    _dump_scores(os.path.join(run_dir, "luq_scores.pkl"), luq)

    print("Saved baseline score pkls to:", run_dir)


if __name__ == "__main__":
    main()
