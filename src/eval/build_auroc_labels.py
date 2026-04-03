import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build AUROC_labels.json from generations.pkl and semantic_clusters.pkl"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "results"),
        help="Directory containing per-run_setting result folders",
    )
    parser.add_argument(
        "--run_setting",
        action="append",
        default=[],
        help="A run_setting folder name. Repeat this argument for multiple settings.",
    )
    parser.add_argument(
        "--run_settings_file",
        type=str,
        default="",
        help="Optional text file with one run_setting per line",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing AUROC_labels.json if present",
    )
    return parser.parse_args()


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def load_run_settings(args: argparse.Namespace) -> List[str]:
    settings = list(args.run_setting)
    if args.run_settings_file:
        run_settings_path = Path(args.run_settings_file)
        with run_settings_path.open("r", encoding="utf-8") as f:
            for line in f:
                item = line.strip()
                if item and not item.startswith("#"):
                    settings.append(item)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for item in settings:
        if item not in seen:
            unique.append(item)
            seen.add(item)
    return unique


def lookup_semantic_cluster(
    semantic_clusters: Any,
    gen_id: Any,
) -> Dict[str, Any]:
    if isinstance(semantic_clusters, dict):
        if gen_id in semantic_clusters:
            return semantic_clusters[gen_id]
        if str(gen_id) in semantic_clusters:
            return semantic_clusters[str(gen_id)]
        if isinstance(gen_id, str) and gen_id.isdigit() and int(gen_id) in semantic_clusters:
            return semantic_clusters[int(gen_id)]
        if isinstance(gen_id, int) and str(gen_id) in semantic_clusters:
            return semantic_clusters[str(gen_id)]
        raise KeyError(f"gen_id={gen_id} not found in semantic_clusters")

    if isinstance(semantic_clusters, list):
        return semantic_clusters[int(gen_id)]

    raise TypeError("semantic_clusters must be dict or list")


def collect_labels(generations: Iterable[Dict[str, Any]], semantic_clusters: Any) -> List[Dict[str, Any]]:
    results = []

    for gen in generations:
        gen_id = gen["id"]
        final_labels = gen.get("final_judge_result_labels", [])
        greedy_label = final_labels[0] if len(final_labels) > 0 else None
        sampled_labels = final_labels[1:] if len(final_labels) > 1 else []
        sampled_probs = gen.get("generated_probs", [])

        cluster_info = lookup_semantic_cluster(semantic_clusters, gen_id)
        semantic_ids = cluster_info.get("semantic_set_ids_entailment", [])

        # sentence_probs[i] = product of token probabilities for sampled sentence i
        sentence_probs = [float(np.prod(p)) for p in sampled_probs]

        # most_sampled_label: sampled label from the highest-probability sampled sentence
        if sentence_probs and sampled_labels:
            max_sampled_idx = int(np.argmax(sentence_probs))
            most_sampled_label = sampled_labels[max_sampled_idx]
        else:
            most_sampled_label = None

        # most_cluster_label: sampled label from max-probability sentence inside the max-mass semantic cluster
        cluster_prob_sum = defaultdict(float)
        cluster_sent_indices = defaultdict(list)
        for idx, (cid, prob) in enumerate(zip(semantic_ids, sentence_probs)):
            cluster_prob_sum[cid] += prob
            cluster_sent_indices[cid].append(idx)

        if cluster_prob_sum and sampled_labels:
            max_cluster = max(cluster_prob_sum, key=cluster_prob_sum.get)
            indices_in_cluster = cluster_sent_indices[max_cluster]
            probs_in_cluster = [sentence_probs[i] for i in indices_in_cluster]
            max_idx_in_cluster = indices_in_cluster[int(np.argmax(probs_in_cluster))]
            most_cluster_label = sampled_labels[max_idx_in_cluster]
        else:
            most_cluster_label = None

        results.append(
            {
                "id": gen_id,
                "greedy_label": greedy_label,
                "most_cluster_label": most_cluster_label,
                "most_sampled_label": most_sampled_label,
            }
        )

    return results


def process_one_run(results_dir: Path, run_setting: str, overwrite: bool) -> Optional[Path]:
    run_dir = results_dir / run_setting
    generations_path = run_dir / "generations.pkl"
    semantic_clusters_path = run_dir / "semantic_clusters.pkl"
    output_path = run_dir / "AUROC_labels.json"

    if output_path.exists() and not overwrite:
        print(f"[skip] {run_setting}: AUROC_labels.json already exists")
        return None

    if not generations_path.exists() or not semantic_clusters_path.exists():
        missing = []
        if not generations_path.exists():
            missing.append(str(generations_path))
        if not semantic_clusters_path.exists():
            missing.append(str(semantic_clusters_path))
        print(f"[skip] {run_setting}: missing files -> {', '.join(missing)}")
        return None

    generations = load_pickle(generations_path)
    semantic_clusters = load_pickle(semantic_clusters_path)

    labels = collect_labels(generations, semantic_clusters)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=4)

    print(f"[ok] {run_setting}: wrote {output_path}")
    return output_path


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    run_settings = load_run_settings(args)

    if not run_settings:
        raise ValueError("No run settings provided. Use --run_setting or --run_settings_file.")

    success = 0
    for run_setting in run_settings:
        out = process_one_run(results_dir, run_setting, overwrite=args.overwrite)
        if out is not None:
            success += 1

    print(f"Done. Generated {success}/{len(run_settings)} AUROC_labels.json files.")


if __name__ == "__main__":
    main()
