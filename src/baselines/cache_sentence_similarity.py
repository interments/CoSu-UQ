import pickle
import argparse
import os
import logging
import tqdm
import config
import torch
import pandas as pd
import sklearn
import sklearn.metrics
from sentence_transformers.cross_encoder import CrossEncoder
from transformers import AutoTokenizer
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s"
)

logger = logging.getLogger(__name__)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--measurement_model", type=str, default="cross-encoder/stsb-roberta-large",
                        help="Measurement model for token-wise importance")
    parser.add_argument("--use_cleaned", type=str, default="True", help="Whether to use cleaned generations")
    parser.add_argument("--run_setting", type=str, default="", help="Run setting")
    args = parser.parse_args()
    os.chdir(config.run_dir)
    os.environ["HF_DATASETS_CACHE"] = config.hf_datasets_cache
    
    measure_model = CrossEncoder(model_name_or_path=args.measurement_model, num_labels=1)
    if args.run_setting == "":
        with open(f'{config.output_dir}/run_setting.txt', 'r') as f:
            run_setting = f.read()
    else:
        run_setting = args.run_setting
    
    with open(f'{config.output_dir}/{run_setting}/generations.pkl', 'rb') as infile:
        generations = pickle.load(infile)
    
    similarity_list = []
    
    for sample_idx, sample in enumerate(tqdm.tqdm(generations, desc="Calculating sentence similarities")):
        if args.use_cleaned.lower() == "true":
            generated_texts = sample['cleaned_generated_texts']
        else:
            generated_texts = sample['generated_texts']
        similarities = {}
        for i in range(len(generated_texts)):
            similarities[i] = []
        question = sample['question']

        for i in range(len(generated_texts)):
            for j in range(i+1, len(generated_texts)):
                gen_i = question + generated_texts[i]
                gen_j = question + generated_texts[j]
                similarity_i_j = measure_model.predict([gen_i, gen_j])
                similarities[i].append(similarity_i_j)
                similarities[j].append(similarity_i_j)

        similarity_list.append(similarities)

    measure_model_name = "-".join(args.measurement_model.split("/")[-2:])
    measure_model_name = args.measurement_model.replace('/', '-')
    sentence_similarity_path = f'{config.output_dir}/{run_setting}/sentence_similarities_{measure_model_name}.pkl'
    with open(sentence_similarity_path, 'wb') as f:
        pickle.dump(similarity_list, f)
    save_path = os.path.join(config.run_dir, config.output_dir.lstrip("./"), f"{run_setting}",f"sentence_similarities_{measure_model_name}.pkl")
    logger.info(f"Sentence similarities saved to {save_path}")
    
    
