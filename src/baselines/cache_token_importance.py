import pickle
import argparse
import os
import tqdm
import config
import torch
import pandas as pd
from sentence_transformers.cross_encoder import CrossEncoder
from transformers import AutoTokenizer
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s"
)

logger = logging.getLogger(__name__)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--measurement_model", type=str, default="cross-encoder/stsb-roberta-large",
                        help="Measurement model for token-wise importance")
    parser.add_argument("--tokenizer_model", type=str, default=os.getenv("COSU_UQ_TOKENIZER_MODEL", "meta-llama/Llama-2-7b-chat-hf"),
                        help="Tokenizer model for token-wise importance")
    parser.add_argument("--run_setting", type=str, default="", help="Run setting")
    args = parser.parse_args()
    os.chdir(config.run_dir)
    os.environ["HF_DATASETS_CACHE"] = config.hf_datasets_cache
    
    measure_model = CrossEncoder(model_name_or_path=args.measurement_model, num_labels=1)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if args.run_setting == "":
        with open(f'{config.output_dir}/run_setting.txt', 'r') as f:
            run_setting = f.read()
    else:
        run_setting = args.run_setting
        
    with open(f'{config.output_dir}/{run_setting}/generations.pkl', 'rb') as infile:
        generations = pickle.load(infile)
    scores = []
    token_importance_list = []
    for sample_idx, sample in enumerate(
        tqdm.tqdm(
            generations,
            desc="Calculating token-wise importance",
            leave=True,
            dynamic_ncols=True,
            ncols=100
        )
    ):
        gen_scores=[]
        prompt = sample['prompt']
        question = sample['question']
        generated_ids = [sample['generated_ids'][i][sample['prompt'].shape[-1]:] for i in range(sample['generated_ids'].shape[0])]
        generated_ids = [item[(item != tokenizer.pad_token_id) & (item != tokenizer.eos_token_id)] for item in generated_ids]
        for k in range(len(generated_ids)):
            gen_id = generated_ids[k]
            gen_text = tokenizer.decode(gen_id)
            token_importance = []
            for idx, token in enumerate(gen_id):
                gen_id_replace = torch.cat([gen_id[:idx], gen_id[idx+1:]], dim=0)
                replace_text = tokenizer.decode(gen_id_replace)
                similarity_to_original = measure_model.predict([question + gen_text,
                                                                question + replace_text],show_progress_bar=False)
                token_importance.append(1 - torch.tensor(similarity_to_original))

            token_importance = torch.tensor(token_importance).reshape(-1)
            token_importance_list.append(token_importance)
    measure_model_name = "-".join(args.measurement_model.split("/")[-2:])
    measure_model_name = args.measurement_model.replace('/', '-')
    token_wise_importance_path = f'{config.output_dir}/{run_setting}/tokenwise_importance_{measure_model_name}_from_generation.pkl'
    with open(token_wise_importance_path, 'wb') as f:
        pickle.dump(token_importance_list, f)
    save_path = os.path.join(config.run_dir, config.output_dir.lstrip("./"), f"{run_setting}",f"tokenwise_importance_{measure_model_name}_from_generation.pkl")
    logger.info(f"Token-wise importance saved to {save_path}")
