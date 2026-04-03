
import tqdm
import os
import config
import random
import argparse
import logging

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="2", help="CUDA device id")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--run_setting", type=str, default="", help="Run setting")
    parser.add_argument("--nli_model", type=str, default="potsawee/deberta-v3-large-mnli", help="NLI model name or path")
    args = parser.parse_args()

    os.chdir(config.run_dir)

    seed_value = args.seed
    model_dir = args.nli_model

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    os.environ["TRANSFORMERS_CACHE"] = config.transformers_cache
    os.environ["HF_DATASETS_CACHE"] = config.hf_datasets_cache
    os.environ["HF_HOME"] = config.hf_home_cache

    import pickle
    import torch
    import numpy as np
    import evaluate
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s"
    )

    logger = logging.getLogger(__name__)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).cuda()
    rouge = evaluate.load('rouge')
    if args.run_setting == "":
        with open(f'{config.output_dir}/run_setting.txt', 'r') as f:
            run_setting = f.read()
    else:
        run_setting = args.run_setting
    with open(f"{config.output_dir}/{run_setting}/generations.pkl", "rb") as infile:
        generations = pickle.load(infile)
        
    result_dict = {}
    
    for sample in tqdm.tqdm(generations, desc="Getting semantic clusters"):
        question = sample['question']
        id_ = sample['id']

        # 1) Cluster cleaned generations.
        cleaned_texts = sample['cleaned_generated_texts']
        unique_cleaned_texts = list(set(cleaned_texts))
        semantic_set_ids = {answer: idx for idx, answer in enumerate(unique_cleaned_texts)}
        # NLI clustering logic.
        if len(unique_cleaned_texts) > 1:
            for i, ref in enumerate(unique_cleaned_texts):
                with torch.no_grad():
                    for j in range(i + 1, len(unique_cleaned_texts)):
                        # qa_1 = question + ' ' + unique_cleaned_texts[i]
                        # qa_2 = question + ' ' + unique_cleaned_texts[j]
                        qa_1 = unique_cleaned_texts[i]
                        qa_2 = unique_cleaned_texts[j]
                        input_ = qa_1 + ' [SEP] ' + qa_2
                        encoded_input = tokenizer.encode(input_, padding=True)
                        prediction = model(torch.tensor([encoded_input], device='cuda'))['logits']
                        predicted_label = torch.argmax(prediction, dim=1).item()
                        reverse_input = qa_2 + ' [SEP] ' + qa_1
                        encoded_reverse_input = tokenizer.encode(reverse_input, padding=True)
                        reverse_prediction = model(torch.tensor([encoded_reverse_input], device='cuda'))['logits']
                        reverse_predicted_label = torch.argmax(reverse_prediction, dim=1).item()
                        # 0 means contradiction; do not merge.
                        if predicted_label == 0 or reverse_predicted_label == 0:
                            continue
                        else:
                            semantic_set_ids[unique_cleaned_texts[j]] = semantic_set_ids[unique_cleaned_texts[i]]
        list_of_semantic_set_ids = [semantic_set_ids[x] for x in cleaned_texts]

        # 2) Cluster raw generated_texts.
        raw_texts = sample['generated_texts']
        unique_raw_texts = list(set(raw_texts))
        semantic_set_ids_raw = {answer: idx for idx, answer in enumerate(unique_raw_texts)}
        if len(unique_raw_texts) > 1:
            for i, ref in enumerate(unique_raw_texts):
                with torch.no_grad():
                    for j in range(i + 1, len(unique_raw_texts)):
                        qa_1 = question + ' ' + unique_raw_texts[i]
                        qa_2 = question + ' ' + unique_raw_texts[j]
                        input_ = qa_1 + ' [SEP] ' + qa_2
                        encoded_input = tokenizer.encode(input_, padding=True)
                        prediction = model(torch.tensor([encoded_input], device='cuda'))['logits']
                        predicted_label = torch.argmax(prediction, dim=1).item()
                        reverse_input = qa_2 + ' [SEP] ' + qa_1
                        encoded_reverse_input = tokenizer.encode(reverse_input, padding=True)
                        reverse_prediction = model(torch.tensor([encoded_reverse_input], device='cuda'))['logits']
                        reverse_predicted_label = torch.argmax(reverse_prediction, dim=1).item()
                        # 0 means contradiction; do not merge.
                        if predicted_label == 0 or reverse_predicted_label == 0:
                            continue
                        else:
                            semantic_set_ids_raw[unique_raw_texts[j]] = semantic_set_ids_raw[unique_raw_texts[i]]
        list_of_semantic_set_ids_raw = [semantic_set_ids_raw[x] for x in raw_texts]

        # 3) Cluster cleaned generations with bidirectional entailment.
        cleaned_texts = sample['cleaned_generated_texts']
        unique_cleaned_texts = list(set(cleaned_texts))
        semantic_set_ids = {answer: idx for idx, answer in enumerate(unique_cleaned_texts)}
        # NLI clustering logic.
        if len(unique_cleaned_texts) > 1:
            for i, ref in enumerate(unique_cleaned_texts):
                with torch.no_grad():
                    for j in range(i + 1, len(unique_cleaned_texts)):
                        # qa_1 = question + ' ' + unique_cleaned_texts[i]
                        # qa_2 = question + ' ' + unique_cleaned_texts[j]
                        qa_1 = unique_cleaned_texts[i]
                        qa_2 = unique_cleaned_texts[j]
                        input_ = qa_1 + ' [SEP] ' + qa_2
                        encoded_input = tokenizer.encode(input_, padding=True)
                        prediction = model(torch.tensor([encoded_input], device='cuda'))['logits']
                        predicted_label = torch.argmax(prediction, dim=1).item()
                        reverse_input = qa_2 + ' [SEP] ' + qa_1
                        encoded_reverse_input = tokenizer.encode(reverse_input, padding=True)
                        reverse_prediction = model(torch.tensor([encoded_reverse_input], device='cuda'))['logits']
                        reverse_predicted_label = torch.argmax(reverse_prediction, dim=1).item()
                        # Merge when entailment condition is satisfied.
                        if predicted_label == 2 and reverse_predicted_label == 2:
                            semantic_set_ids[unique_cleaned_texts[j]] = semantic_set_ids[unique_cleaned_texts[i]]
                        else:
                            continue
        list_of_semantic_set_ids_entailment = [semantic_set_ids[x] for x in cleaned_texts]
        # Keep other fields aligned with previous behavior.
        result_dict[id_] = {
            'semantic_set_ids': list_of_semantic_set_ids,
            'semantic_set_ids_raw': list_of_semantic_set_ids_raw,
            'semantic_set_ids_entailment': list_of_semantic_set_ids_entailment,
            # Additional fields can be added here if needed.
        }

    with open(f"{config.output_dir}/{run_setting}/semantic_clusters.pkl", "wb") as outfile:
        pickle.dump(result_dict, outfile)
    save_path = os.path.join(config.run_dir, config.output_dir.lstrip("./"), f"{run_setting}" "semantic_clusters.pkl")
    logger.info(f"Semantic clusters saved to {save_path}")
