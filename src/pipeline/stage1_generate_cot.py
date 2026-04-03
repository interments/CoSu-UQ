import os
import argparse
import config
import logging
parser = argparse.ArgumentParser(description="LLAMA Generation Script")
parser.add_argument("--model_dir", type=str, default=os.getenv("COSU_UQ_MODEL_DIR", ""), help="Model directory")
parser.add_argument("--model_name",type=str, default="Llama-3.1-8B", help="Model name")
parser.add_argument("--data_file", type=str, default=None, help="Dataset file path (jsonl)")
parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
parser.add_argument("--fraction", type=float, default=0.005, help="Dataset fraction to use")
parser.add_argument("--max_length", type=int, default=512, help="Maximum generated length")
parser.add_argument("--num_generations_per_prompt", type=int, default=5, help="Number of sampled generations per prompt")
parser.add_argument("--temperature", type=float, default=1, help="Sampling temperature")
parser.add_argument("--top_k", type=int, default=5, help="top_k")
parser.add_argument("--top_p", type=float, default=0.95, help="top_p")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--device", type=str, default="2", help="CUDA device id")
parser.add_argument("--decode_method", type=str, default="greedy", choices=["greedy", "beam_search"], help="Decoding method")
parser.add_argument("--num_beams", type=int, default=5, help="Beam width (effective only for beam_search)")
parser.add_argument("--num_return_sequences", type=int, default=1, help="Returned sequence count (effective only for beam_search)")
parser.add_argument("--use_api", action="store_true", help="Use API generation instead of local model")
parser.add_argument("--api_key", type=str, default=os.getenv("COSU_UQ_API_KEY", ""), help="API Key")
parser.add_argument("--api_base", type=str, default=os.getenv("COSU_UQ_API_BASE", ""), help="API Base URL")
parser.add_argument("--api_model_name", type=str, default=None, help="API model name (if different from model_name)")
args = parser.parse_args()


# Resolve runtime parameters from CLI/config.
seed_value = args.seed
device = args.device
fraction_of_data_to_use = args.fraction
max_length_of_generated_sequence = args.max_length
temperature = args.temperature
top_k = args.top_k
top_p = args.top_p
model_dir = args.model_dir
data_file = args.data_file 
output_dir = args.output_dir or config.output_dir
decode_method = args.decode_method
num_generations_per_prompt = args.num_generations_per_prompt
num_beams = args.num_beams if args.decode_method == "beam_search" else 1
num_return_sequences = args.num_return_sequences if args.decode_method == "beam_search" else 1
use_api = args.use_api
api_key = args.api_key
api_base = args.api_base
api_model_name = args.api_model_name or args.model_name

if use_api and (not api_key or not api_base):
    raise ValueError("When --use_api is enabled, both --api_key and --api_base must be provided (or set via env vars).")
if not use_api and not model_dir:
    raise ValueError("When using local generation, --model_dir is required (or set COSU_UQ_MODEL_DIR).")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s"
)

logger = logging.getLogger(__name__)

os.environ["CUDA_VISIBLE_DEVICES"] = device
logger.info("CUDA device: {}".format(os.getenv("CUDA_VISIBLE_DEVICES")))
os.chdir(config.run_dir)
os.environ["HF_DATASETS_CACHE"] = config.hf_datasets_cache
os.environ["PYTHONHASHSEED"] = str(seed_value)

from transformers import AutoTokenizer, AutoModelForCausalLM,StoppingCriteria, StoppingCriteriaList
from utils.cot_uq_utils import parse_response_to_dict
import torch
import random
import numpy as np
import datasets
import tqdm
import evaluate
import pickle
import re

random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
if use_api:
    from dashscope import Generation
    import dashscope
    import numpy as np
    
    # Configure DashScope.
    dashscope.api_key = api_key
    dashscope.base_http_api_url = api_base
    # Load tokenizer only in API mode.
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info(f"DashScope client initialized with model: {api_model_name}")
    logger.info("API mode currently supports Qwen-family models only.")
    model = None  # Local model is not needed in API mode.
else:
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map={"": 0} if torch.cuda.is_available() else "cpu",
        attn_implementation="flash_attention_2",
    )
    # model = model.to_bettertransformer()
    logger.info("Model loaded.")
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token

logger.info("Loading dataset...")
dataset = datasets.load_dataset("json", data_files=data_file, split="train")
train_dataset = dataset.train_test_split(test_size=(1 - fraction_of_data_to_use), seed=seed_value)['train']
logger.info(f"Dataset loaded. Number of samples: {len(train_dataset)}")
model_name = args.model_name
dataset_name = data_file.split("/")[-1].replace(".jsonl", "").replace(".json", "")
if use_api:
    run_setting = f"{model_name}_API_{dataset_name}_fraction_{args.fraction}_max_length_{args.max_length}_num_generations_{args.num_generations_per_prompt}_temperature_{args.temperature}_top_p_{args.top_p}_seed_{args.seed}"
    logger.info(f"Run setting: {run_setting}")
else:
    run_setting = f"{model_name}_{dataset_name}_fraction_{args.fraction}_max_length_{args.max_length}_num_generations_{args.num_generations_per_prompt}_temperature_{args.temperature}_top_k_{args.top_k}_top_p_{args.top_p}_decode_method_{args.decode_method}_seed_{args.seed}"
    logger.info(f"Run setting: {run_setting}")

def max_length(questions:list):
    max_len_ids=0
    max_len=0
    for idx,que in enumerate(questions):
        current_len = len(que)
        if current_len > max_len:
            max_len = current_len
            max_len_ids = idx
    return max_len_ids

def extract_question(text):
    # Match Question: / question: case-insensitively.
    match = re.split(r'Question:', text, maxsplit=1)
    if len(match) > 1:
        return match[-1].strip()
    else:
        return text.strip()
    
class KeywordStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, keywords, input_length):
        self.tokenizer = tokenizer
        self.keywords = keywords
        self.input_length = input_length
    
    def __call__(self, input_ids, scores, **kwargs):
        if input_ids.shape[1] <= self.input_length:
            return False
            
        generated_ids = input_ids[0][self.input_length:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        for keyword in self.keywords:
            if keyword in generated_text:
                return True
        return False

def encode(examples):
    if "instruct" in tokenizer.name_or_path.lower():
        messages = [
            {"role": "system", "content": examples['request']},
            {"role": "user", "content": examples['one_shot_user']},
            {"role": "assistant", "content": examples['one_shot_assistant']},
            {"role": "user", "content": examples['your_turn']}
        ]
        return tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                truncation=False,
                padding=False,
            )
    elif "qwen3" in tokenizer.name_or_path.lower():
        messages = [
            {"role": "system", "content": examples['request']},
            {"role": "user", "content": examples['one_shot_user']},
            {"role": "assistant", "content": examples['one_shot_assistant']},
            {"role": "user", "content": examples['your_turn']}
        ]
        return tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                truncation=False,
                padding=False,
                enable_thinking=False
            )
    elif "chat" in tokenizer.name_or_path.lower() and "llama-2" in tokenizer.name_or_path.lower():
        system_prompt = examples['request']
        one_shot_user = examples['one_shot_user']
        one_shot_assistant = examples['one_shot_assistant']
        your_turn = examples['your_turn']
        prompt = f"""<s>[INST] <<SYS>>
{system_prompt}
<</SYS>>

{one_shot_user} [/INST] {one_shot_assistant} </s><s>[INST] {your_turn} [/INST]"""
        return tokenizer(
            prompt,
            add_special_tokens=False,   # Special tokens are manually assembled here.
            return_tensors="pt",
            truncation=False,
            padding=False,
        )
    else:
        return tokenizer(examples['input'], truncation=False, padding=False)

def encode_and_format_dataset(dataset):
    dataset = dataset.map(encode, batched=False, load_from_cache_file=False)
    dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'], output_all_columns=True)
    return dataset

questions = encode_and_format_dataset(train_dataset)
dataloader = torch.utils.data.DataLoader(questions, batch_size=1)
logger.info("Data preparation complete.")
rouge = evaluate.load('rouge',cache_dir="./metrics")
exact_match_metric = evaluate.load("exact_match",cache_dir="./metrics")
bertscore_metric = evaluate.load("bertscore",cache_dir="./metrics")

# from metrics.rouge import RougeOffline
# from metrics.exactmatch import ExactMatchOffline
# from metrics.bertscore import BertScoreOffline
# rouge = RougeOffline(cache_dir="./metrics")
# exact_match_metric = ExactMatchOffline(cache_dir="./metrics")
# bertscore_metric = BertScoreOffline(cache_dir="./metrics")

def clean_generated_sequence(sequence: str) -> str:
    sequence = sequence.strip()
    pattern = r'final\s+answer\s*[：:]\s*(.+?)(?:\n|$)'
    match = re.search(pattern, sequence, re.IGNORECASE | re.DOTALL)
    
    if match:
        answer = match.group(1).strip()
        answer = re.sub(r'\n+', ' ', answer)
        return answer
    
    sequences = sequence.split("\n")
    for seq in reversed(sequences):
        seq = seq.strip()
        if seq:
            return seq
    
    return sequence
    
def get_most_likely_generation(model, input_ids, stop_criteria,max_length_of_generated_sequence, num_beams=5, num_return_sequences=2):
    success_flag = True
    try_num = 5
    if decode_method == "greedy":
        while try_num > 0:
            most_likely_generation = model.generate(input_ids=input_ids, 
                                max_length=input_ids.shape[-1] + max_length_of_generated_sequence, 
                                num_beams=1, 
                                num_return_sequences=1, 
                                do_sample=False,
                                eos_token_id=tokenizer.eos_token_id,
                                pad_token_id=tokenizer.pad_token_id,
                                temperature = temperature,
                                stopping_criteria=stop_criteria,
                                return_dict_in_generate=True, 
                                output_scores=True,
                                )
            generated_ids = most_likely_generation.sequences[0][input_ids.shape[1]:-1]
            generation_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            llm_answer, _, _ = parse_response_to_dict(generation_text)
            token_probabilities = [{i: p for i, p in enumerate(prob[0]) if p > 0} for prob in [torch.softmax(score, dim=1).tolist() for score in most_likely_generation.scores[:-1]]]
            print(len(token_probabilities))
            print(generated_ids.shape)
            probabilities = [prob[id.item()] for id, prob in zip(generated_ids.detach().to("cpu"), token_probabilities)]
            if "step" in clean_generated_sequence(generation_text).lower():
                try_num -= 1
                logger.warning(f"Generation contains 'step' in the most likely text, retrying... ({try_num} attempts left)")
                continue
            elif generated_ids.shape[-1] == tokenizer.eos_token_id:
                try_num -= 1
                logger.warning(f"Generation ended with EOS token, retrying... ({try_num} attempts left)")
                continue
            elif generated_ids.size(0) == 0:
                try_num -= 1
                logger.warning(f"Generation returned empty sequence, retrying... ({try_num} attempts left)")
                continue
            elif llm_answer is None or llm_answer in ['', ' ']:
                logger.warning(f'New Reasoning Tokens Are None, Current try is {try_num + 1}')
                try_num -= 1
            else:
                break
    else:
        raise NotImplementedError
    if try_num <= 0:
        success_flag = False
    return most_likely_generation.sequences, probabilities ,success_flag

def get_generations(model, tokenizer, dataloader, number_of_generations):
    with torch.no_grad():
        sequences = []
        for batch_idx, batch in enumerate(tqdm.tqdm(dataloader, desc="Generating text")):
            if torch.cuda.is_available():
                logger.info(f"GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB / {torch.cuda.memory_reserved()/1024**3:.2f}GB")
                
            if "instruct" in model.name_or_path.lower() or ("chat" in model.name_or_path.lower() and "llama-2" in model.name_or_path.lower()) or ("qwen3" in model.name_or_path.lower()):
                input_ids = batch['input_ids'].reshape(-1).unsqueeze(0).to("cuda:0")
            else:
                input_ids = batch['input_ids'].to('cuda:0')
                
            input_length = input_ids.shape[-1]
            stop_criteria = StoppingCriteriaList([
                KeywordStoppingCriteria(tokenizer, ["\n\n\n\n"], input_length)
            ])
            likely_generations, probabilities, greedy_success_flag = get_most_likely_generation(
                model, input_ids, stop_criteria, max_length_of_generated_sequence, num_beams=num_beams, num_return_sequences=2
            )
            generations = torch.full((number_of_generations, input_length + max_length_of_generated_sequence),tokenizer.pad_token_id,
                                    dtype=torch.long,
                                    device="cuda:0")
            probabilities_sampled = []
            sampled_success_flags=[]
            for i in tqdm.tqdm(range(number_of_generations), desc="Sampling generations"):
                sampled_success_flag = True
                try_num = 5
                while try_num > 0:
                    generation = model.generate(input_ids,
                                            do_sample=True,
                                            num_return_sequences=1,
                                            num_beams=args.num_beams,
                                            max_length=input_length + max_length_of_generated_sequence,
                                            # eos_token_id=period_token_id,
                                            eos_token_id=tokenizer.eos_token_id,
                                            pad_token_id=tokenizer.pad_token_id,
                                            # stopping_criteria=stop_criteria,
                                            temperature=temperature,
                                            top_p=args.top_p,
                                            return_dict_in_generate=True, 
                                            output_scores=True,
                                            )
                    generated_ids = generation.sequences[0][input_length:-1]
                    generation_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                    logger.info(generation_text)
                    llm_answer, _, _ = parse_response_to_dict(generation_text)
                    if "step" in clean_generated_sequence(generation_text).lower():
                        try_num -= 1
                        logger.warning(f"Generation contains 'step' in the most likely text, retrying... ({try_num} attempts left)")
                        continue
                    elif generated_ids.shape[-1] == tokenizer.eos_token_id:
                        try_num -= 1
                        logger.warning(f"Generation ended with EOS token, retrying... ({try_num} attempts left)")
                        continue
                    elif generated_ids.size(0) == 0:
                        try_num -= 1
                        logger.warning(f"Generation returned empty sequence, retrying... ({try_num} attempts left)")
                        continue
                    elif llm_answer is None or llm_answer in ['', ' ']:
                        logger.warning(f'New Reasoning Tokens Are None, Current try is {try_num + 1}')
                        try_num -= 1
                    else:
                        break
                if try_num <= 0:
                    sampled_success_flag = False
                if args.num_beams == 1:
                    sampled_success_flags.append(sampled_success_flag)
                    generations[i, :generation.sequences.shape[1]] = generation.sequences
                    token_probabilities_sampled = [{i: p for i, p in enumerate(prob[0]) if p > 0} for prob in [torch.softmax(score, dim=1).tolist() for score in generation.scores[:-1]]]
                    probabilities_sampled.append([prob[id.item()] for id, prob in zip(generated_ids.detach().to("cpu"), token_probabilities_sampled)])
                elif args.num_beams > 1:
                    sampled_success_flags.append(sampled_success_flag)
                    generations[i, :generation.sequences.shape[1]] = generation.sequences[0]
                    token_probabilities_list = [[{i: p for i, p in enumerate(prob[j]) if p > 0} for prob in [torch.softmax(score, dim=1).tolist() for score in generation.scores[:-1]]] for j in range(args.num_beams)]
                    beam_indices = generation.beam_indices[0].tolist()[:-1]
                    print("Beam indices length:", len(beam_indices))
                    print("Generated IDs length:", generated_ids.shape)
                    token_probabilities_sampled = [token_probabilities_list[beam_idx][step] for step, beam_idx in enumerate(beam_indices)]
                    probabilities_sampled.append([prob[id.item()] for id, prob in zip(generated_ids.detach().to("cpu"), token_probabilities_sampled)])
                    
            # generations = [model.generate(input_ids = input_ids,
            #                                     max_length = input_ids.shape[1] + max_length_of_generated_sequence,
            #                                     do_sample = True,
            #                                     temperature = temperature,
            #                                     top_k = top_k,
            #                                     top_p = top_p,
            #                                     eos_token_id=tokenizer.eos_token_id,
            #                                     pad_token_id=tokenizer.pad_token_id
            #                                     )[:,input_ids.shape[1]:].detach().to("cpu")
            #                                     for _ in tqdm.tqdm(range(number_of_generations), desc="Do sample")]
            # generations = torch.reshape(generations, (-1, number_of_generations, generations.shape[-1]))
            generations = torch.reshape(generations, (-1, number_of_generations, generations.shape[-1]))
            for i in range(generations.shape[0]):
                generated_texts = []
                for generation in generations[i]:
                    generated_texts.append(
                        tokenizer.decode(generation[input_length:], skip_special_tokens=True))
                question = batch["question"][0]
                sequence_dict = {
                    "id": batch["index"][0].item(),
                    "decode_method": decode_method,
                    "prompt" : input_ids.detach().to("cpu"),
                    "prompt_text": batch["input"][0],
                    "question": question,
                    "generated_success_flag": sampled_success_flags,
                    "generated_ids": generations[i].to('cpu'),
                    "generated_texts": generated_texts,
                    "generated_probs": probabilities_sampled,
                    "cleaned_generated_texts": [clean_generated_sequence(text) for text in generated_texts],
                    "most_likely_generation_success_flag": greedy_success_flag,
                    "most_likely_generation_ids": likely_generations[0][input_ids.shape[1]:-1].detach().to("cpu"),
                    "most_likely_generation_probs": probabilities,
                    "most_likely_generation": tokenizer.decode(likely_generations[0][input_ids.shape[1]:].detach().to("cpu"), skip_special_tokens=True),
                    # "second_most_likely_generation_ids": likely_generations[1][input_ids.shape[1]:].detach().to("cpu") if len(likely_generations) > 1 else None,
                    # "second_most_likely_generation": tokenizer.decode(likely_generations[1][input_ids.shape[1]:].detach().to("cpu"), skip_special_tokens=True) if len(likely_generations) > 1 else None,
                    "second_most_likely_generation_ids": None,
                    "second_most_likely_generation": None
                }
                try:
                    cleaned_generations = torch.full((number_of_generations, input_length + max_length_of_generated_sequence),
                                                    tokenizer.pad_token_id,
                                                    dtype = torch.long,
                                                    device = "cuda:0")
                    for i in range(number_of_generations):
                        cleaned_generation = tokenizer.encode(sequence_dict["cleaned_generated_texts"][i],return_tensors="pt", add_special_tokens=False).squeeze(0).to("cuda:0")
                        full_generation = torch.cat([sequence_dict["prompt"].squeeze(0).to("cuda:0"),cleaned_generation], dim=0)
                        cleaned_generations[i ,:full_generation.shape[0]] = full_generation
                    sequence_dict["cleaned_generated_ids"] = cleaned_generations.detach().to("cpu")
                    

                except Exception as e:
                    logger.error(f"Error in cleaning generated sequences for sample ID {sequence_dict['id']}: {e}")
                    sequence_dict["cleaned_generated_ids"] = None
                
                cleaned_sequence = clean_generated_sequence(sequence_dict["most_likely_generation"])
                sequence_dict["cleaned_most_likely_generation"] = cleaned_sequence
                # sequence_dict["cleaned_most_likely_generation_ids"] = torch.cat([sequence_dict["prompt"].squeeze(0), tokenizer(cleaned_sequence, return_tensors="pt").input_ids.to("cpu").squeeze(0)], dim=0)
                
                sequence_dict["cleaned_most_likely_generation_ids"] = tokenizer(cleaned_sequence, return_tensors="pt").input_ids.to("cpu").squeeze(0)[1:]
                if sequence_dict["second_most_likely_generation"] is not None:
                    cleaned_second_most_likely_generation = clean_generated_sequence(sequence_dict["second_most_likely_generation"])
                    sequence_dict["cleaned_second_most_likely_generation"] = cleaned_second_most_likely_generation
                    sequence_dict["cleaned_second_most_likely_generation_ids"] = torch.cat([sequence_dict["prompt"].squeeze(0), tokenizer(cleaned_second_most_likely_generation, return_tensors="pt").input_ids.to("cpu").squeeze(0)], dim=0)
                else:
                    sequence_dict["cleaned_second_most_likely_generation"] = None
                    sequence_dict["cleaned_second_most_likely_generation_ids"] = None
                
                rouge_types = ['rouge1', 'rouge2', 'rougeL']
                for rouge_type in rouge_types:
                    if rouge_type in batch:
                        sequence_dict[rouge_type + '_reference_answers'] = batch[rouge_type]
                    else:
                        sequence_dict[rouge_type + '_reference_answers'] = None
                    
                
                if "hotpotqa" in data_file.lower():
                    sequence_dict["answer"] = batch["outputs"][0]
                    reference_answers = batch["outputs"]
                elif "gsm8k" in data_file.lower():
                    sequence_dict["answer"] = batch["outputs"][0]
                    reference_answers = batch["outputs"]
                elif "2wikimultihopqa" in data_file.lower():
                    sequence_dict["answer"] = batch["outputs"][0]
                    reference_answers = batch["outputs"]
                elif "math" in data_file.lower():
                    sequence_dict["answer"] = batch["outputs"][0]
                    reference_answers = batch["outputs"]
                elif "medqa" in data_file.lower():
                    sequence_dict["answer"] = batch["outputs"][0]
                    reference_answers = batch["outputs"]
                elif "triviaqa" in data_file.lower():
                    sequence_dict["answer"] = batch["outputs"][0]
                    reference_answers = batch["outputs"]
                else:
                    raise NotImplementedError("Only hotpotqa ,2WikimultihopQA and gsm8k dataset is supported for now")
                
                sequence_dict['exact_match'] = 0.0
                sequence_dict['cleaned_exact_match'] = 0.0
                sequence_dict['bertscore_precision'] = 0.0
                sequence_dict['cleaned_bertscore_precision'] = 0.0
                sequence_dict['bertscore_recall'] = 0.0
                sequence_dict['cleaned_bertscore_recall'] = 0.0
                sequence_dict['bertscore_f1'] = 0.0
                sequence_dict['cleaned_bertscore_f1'] = 0.0
                
                for rouge_type in rouge_types:
                    sequence_dict[rouge_type + '_to_target'] = 0.0
                    
                for rouge_type in rouge_types:
                    sequence_dict["cleaned_" + rouge_type + '_to_target'] = 0.0
                    
                for answer in reference_answers:
                    if isinstance(answer, (tuple, list)):
                        reference_str = str(answer[0])
                    else:
                        reference_str = str(answer)
                    if "llama" in model_dir.lower() or "qwen" in model_dir.lower():    
                        predictions = [sequence_dict["most_likely_generation"].lstrip()]
                    references = [reference_str]
                    exact_match_results = exact_match_metric.compute(predictions=predictions,
                                                            references=references,
                                                            ignore_case=True,
                                                            ignore_punctuation=True)
                    bertscore_results = bertscore_metric.compute(predictions=predictions, references=references,
                                                                lang="en")
                    rouge_results = rouge.compute(predictions=predictions, references=references)
                    sequence_dict["exact_match"] = max(sequence_dict["exact_match"], exact_match_results["exact_match"])
                    sequence_dict['bertscore_precision'] = max(sequence_dict['bertscore_precision'], bertscore_results['precision'][0])
                    sequence_dict['bertscore_recall'] = max(sequence_dict['bertscore_recall'], bertscore_results['recall'][0])
                    sequence_dict['bertscore_f1'] = max(sequence_dict['bertscore_f1'], bertscore_results['f1'][0])
                    for rouge_type in rouge_types:
                        sequence_dict[rouge_type + '_to_target'] = max(sequence_dict[rouge_type + '_to_target'], rouge_results[rouge_type])
                        
                for answer in reference_answers:
                    if isinstance(answer, (tuple, list)):
                        reference_str = str(answer[0])
                    else:
                        reference_str = str(answer)
                    if "llama" in model_dir.lower() or "qwen" in model_dir.lower():    
                        predictions = [sequence_dict["cleaned_most_likely_generation"].lstrip()]
                    references = [reference_str]
                    exact_match_results = exact_match_metric.compute(predictions=predictions,
                                                            references=references,
                                                            ignore_case=True,
                                                            ignore_punctuation=True)
                    bertscore_results = bertscore_metric.compute(predictions=predictions, references=references,
                                                                lang="en")
                    rouge_results = rouge.compute(predictions=predictions, references=references)
                    sequence_dict["cleaned_exact_match"] = max(sequence_dict["exact_match"], exact_match_results["exact_match"])
                    sequence_dict['cleaned_bertscore_precision'] = max(sequence_dict['bertscore_precision'], bertscore_results['precision'][0])
                    sequence_dict['cleaned_bertscore_recall'] = max(sequence_dict['bertscore_recall'], bertscore_results['recall'][0])
                    sequence_dict['cleaned_bertscore_f1'] = max(sequence_dict['bertscore_f1'], bertscore_results['f1'][0])
                    for rouge_type in rouge_types:
                        sequence_dict["cleaned_" + rouge_type + '_to_target'] = max(sequence_dict["cleaned_" + rouge_type + '_to_target'], rouge_results[rouge_type])
                sequences.append(sequence_dict)
    return sequences

def get_most_likely_generation_api(messages, api_model_name, tokenizer, max_length_of_generated_sequence, temperature=0.0):
    """Get the most likely generation via DashScope API (greedy decoding)."""
    success_flag = True
    try_num = 10
    
    while try_num > 0:
        try:
            response = Generation.call(
                model=api_model_name,
                messages=messages,
                result_format="message",
                temperature=temperature,  # greedy
                max_tokens=max_length_of_generated_sequence,
                logprobs=True,
                top_logprobs=5,
                enable_thinking=False,
                stream=False
            )
            print(response)
            # Extract generated text.
            generation_text = response['output']['choices'][0]['message']['content']
            llm_answer, _, _ = parse_response_to_dict(generation_text)
            # Extract logprobs.
            logprobs_content = response['output']['choices'][0]['logprobs']['content']
            token_logprobs = [token_data['logprob'] for token_data in logprobs_content]
            probabilities = [np.exp(lp) for lp in token_logprobs]
            
            # Encode generated text to token ids.
            generated_ids = tokenizer.encode(generation_text, add_special_tokens=False, return_tensors="pt").squeeze(0)
            
            # Keep debug output aligned with local generation mode.
            print(len(probabilities))
            print(generated_ids.shape)
            
            # Validate tensor length consistency.
            if len(probabilities) != generated_ids.shape[0]:
                logger.warning(f"Dimension mismatch: probabilities={len(probabilities)}, generated_ids={generated_ids.shape[0]}")
                try_num -= 1
                continue
            
            # Validate generation quality.
            if "step" in clean_generated_sequence(generation_text).lower():
                try_num -= 1
                logger.warning(f"Generation contains 'step' in the most likely text, retrying... ({try_num} attempts left)")
                continue
            elif llm_answer is None or llm_answer in ['', ' ']:
                logger.warning(f'New Reasoning Tokens Are None, Current try is {try_num + 1}')
                try_num -= 1
                continue
            else:
                break
                
        except Exception as e:
            logger.error(f"Dashscope API call failed: {e}, retrying... ({try_num} attempts left)")
            try_num -= 1
            continue
    
    if try_num <= 0:
        success_flag = False
        generation_text = ""
        generated_ids = torch.tensor([])
        probabilities = []
    
    return generation_text, generated_ids, probabilities, success_flag


def get_generations_api(tokenizer, dataloader, number_of_generations, api_model_name, max_length_of_generated_sequence, temperature, top_p):
    """Generate text via DashScope API with output format aligned to local generation."""
    sequences = []
    
    for batch_idx, batch in enumerate(tqdm.tqdm(dataloader, desc="Generating text with Dashscope API")):
        # Read pre-tokenized input_ids from batch.
        input_ids = batch['input_ids'].reshape(-1).unsqueeze(0)  # API mode does not need GPU tensors.
        input_length = input_ids.shape[-1]
        
        # Build chat messages from dataset fields.
        messages = [
            {"role": "system", "content": batch['request'][0]},
            {"role": "user", "content": batch['one_shot_user'][0]},
            {"role": "assistant", "content": batch['one_shot_assistant'][0]},
            {"role": "user", "content": batch['your_turn'][0]}
        ]
        
        # 1) Get most likely generation (greedy, temperature=0).
        logger.info("Getting most likely generation (greedy)...")
        likely_generation_text, likely_generation_ids, likely_probabilities, greedy_success_flag = \
            get_most_likely_generation_api(messages, api_model_name, tokenizer, max_length_of_generated_sequence, temperature=0.0)
        
        # 2) Generate multiple sampled responses.
        logger.info(f"Sampling {number_of_generations} generations...")
        generations = []
        generated_texts = []
        probabilities_sampled = []
        sampled_success_flags = []
        
        for i in tqdm.tqdm(range(number_of_generations), desc="Sampling generations"):
            sampled_success_flag = True
            try_num = 10
            
            while try_num > 0:
                try:
                    response = Generation.call(
                        model=api_model_name,
                        messages=messages,
                        result_format="message",
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_length_of_generated_sequence,
                        logprobs=True,
                        top_logprobs=5,
                        enable_thinking=False,
                        stream=False
                    )
                    
                    # Extract generated text.
                    generation_text = response['output']['choices'][0]['message']['content']
                    logger.info(generation_text)
                    llm_answer, _, _ = parse_response_to_dict(generation_text)
                    # print(generation_text)
                    # Extract logprobs.
                    logprobs_content = response['output']['choices'][0]['logprobs']['content']
                    token_logprobs = [token_data['logprob'] for token_data in logprobs_content]
                    probabilities = [np.exp(lp) for lp in token_logprobs]
                    # print(probabilities)
                    
                    # Encode generated text.
                    generated_ids_temp = tokenizer.encode(generation_text, add_special_tokens=False, return_tensors="pt").squeeze(0)
                    
                    # Keep debug output aligned with local generation mode.
                    print(len(probabilities))
                    print(generated_ids_temp.shape)
                    
                    # Validate tensor length consistency.
                    if len(probabilities) != generated_ids_temp.shape[0]:
                        logger.warning(f"Dimension mismatch: probabilities={len(probabilities)}, generated_ids={generated_ids_temp.shape[0]}")
                        try_num -= 1
                        continue
                    
                    # Validate generation quality.
                    if "step" in clean_generated_sequence(generation_text).lower():
                        try_num -= 1
                        logger.warning(f"Generation contains 'step', retrying... ({try_num} attempts left)")
                        continue
                    elif llm_answer is None or llm_answer in ['', ' ']:
                        logger.warning(f'New Reasoning Tokens Are None, Current try is {try_num + 1}')
                        try_num -= 1
                        continue
                    else:
                        break
                        
                except Exception as e:
                    logger.error(f"Dashscope API call failed: {e}, retrying... ({try_num} attempts left)")
                    try_num -= 1
                    continue
            
            if try_num <= 0:
                sampled_success_flag = False
                generation_text = ""
                probabilities = []
                generated_ids_temp = torch.tensor([])
            
            # Pad to a consistent length (aligned with local generation format).
            padded_ids = torch.full((input_length + max_length_of_generated_sequence,), tokenizer.pad_token_id, dtype=torch.long)
            if generated_ids_temp.shape[0] > 0:
                full_sequence = torch.cat([input_ids.squeeze(0), generated_ids_temp], dim=0)
                padded_ids[:full_sequence.shape[0]] = full_sequence
            else:
                padded_ids[:input_ids.shape[1]] = input_ids.squeeze(0)
            
            generations.append(padded_ids)
            generated_texts.append(generation_text)
            probabilities_sampled.append(probabilities)
            sampled_success_flags.append(sampled_success_flag)
        
        # Convert to tensor (aligned with local generation format).
        generations = torch.stack(generations)
        
        question = batch["question"][0]
        sequence_dict = {
            "id": batch["index"][0].item(),
            "decode_method": "greedy",  # API path uses greedy for the anchor generation.
            "prompt": input_ids.detach().to("cpu"),
            "prompt_text": batch["input"][0],
            "question": question,
            "generated_success_flag": sampled_success_flags,
            "generated_ids": generations.to('cpu'),
            "generated_texts": generated_texts,
            "generated_probs": probabilities_sampled,
            "cleaned_generated_texts": [clean_generated_sequence(text) for text in generated_texts],
            "most_likely_generation_success_flag": greedy_success_flag,
            "most_likely_generation_ids": likely_generation_ids.detach().to("cpu") if isinstance(likely_generation_ids, torch.Tensor) and likely_generation_ids.numel() > 0 else torch.tensor([]),
            "most_likely_generation_probs": likely_probabilities,
            "most_likely_generation": likely_generation_text,
            "second_most_likely_generation_ids": None,
            "second_most_likely_generation": None
        }
        
        # Build cleaned generation tensors (aligned with local generation format).
        cleaned_generations = torch.full((number_of_generations, input_length + max_length_of_generated_sequence),
                                        tokenizer.pad_token_id,
                                        dtype=torch.long)
        for i in range(number_of_generations):
            cleaned_generation = tokenizer.encode(sequence_dict["cleaned_generated_texts"][i], return_tensors="pt", add_special_tokens=False).squeeze(0)
            full_generation = torch.cat([sequence_dict["prompt"].squeeze(0), cleaned_generation], dim=0)
            cleaned_generations[i, :full_generation.shape[0]] = full_generation
        
        sequence_dict["cleaned_generated_ids"] = cleaned_generations.detach().to("cpu")
        
        cleaned_sequence = clean_generated_sequence(sequence_dict["most_likely_generation"])
        sequence_dict["cleaned_most_likely_generation"] = cleaned_sequence
        sequence_dict["cleaned_most_likely_generation_ids"] = tokenizer(cleaned_sequence, return_tensors="pt").input_ids.to("cpu").squeeze(0)[1:]
        
        sequence_dict["cleaned_second_most_likely_generation"] = None
        sequence_dict["cleaned_second_most_likely_generation_ids"] = None
        
        # Attach ROUGE references.
        rouge_types = ['rouge1', 'rouge2', 'rougeL']
        for rouge_type in rouge_types:
            if rouge_type in batch:
                sequence_dict[rouge_type + '_reference_answers'] = batch[rouge_type]
            else:
                sequence_dict[rouge_type + '_reference_answers'] = None
        
        # Set answer and reference answers (aligned with local generation logic).
        if "hotpotqa" in data_file.lower():
            sequence_dict["answer"] = batch["outputs"][0]
            reference_answers = batch["outputs"]
        elif "gsm8k" in data_file.lower():
            sequence_dict["answer"] = batch["outputs"][0]
            reference_answers = batch["outputs"]
        elif "2wikimultihopqa" in data_file.lower():
            sequence_dict["answer"] = batch["outputs"][0]
            reference_answers = batch["outputs"]
        elif "math" in data_file.lower():
            sequence_dict["answer"] = batch["outputs"][0]
            reference_answers = batch["outputs"]
        elif "medqa" in data_file.lower():
            sequence_dict["answer"] = batch["outputs"][0]
            reference_answers = batch["outputs"]
        elif "triviaqa" in data_file.lower():
            sequence_dict["answer"] = batch["outputs"][0]
            reference_answers = batch["outputs"]
        else:
            raise NotImplementedError("Only hotpotqa, 2WikimultihopQA, gsm8k and math datasets are supported")
        
        # Initialize metrics.
        sequence_dict['exact_match'] = 0.0
        sequence_dict['cleaned_exact_match'] = 0.0
        sequence_dict['bertscore_precision'] = 0.0
        sequence_dict['cleaned_bertscore_precision'] = 0.0
        sequence_dict['bertscore_recall'] = 0.0
        sequence_dict['cleaned_bertscore_recall'] = 0.0
        sequence_dict['bertscore_f1'] = 0.0
        sequence_dict['cleaned_bertscore_f1'] = 0.0
        
        for rouge_type in rouge_types:
            sequence_dict[rouge_type + '_to_target'] = 0.0
            sequence_dict["cleaned_" + rouge_type + '_to_target'] = 0.0
        
        # Compute metrics.
        for answer in reference_answers:
            if isinstance(answer, (tuple, list)):
                reference_str = str(answer[0])
            else:
                reference_str = str(answer)
            
            predictions = [sequence_dict["most_likely_generation"].lstrip()]
            references = [reference_str]
            
            exact_match_results = exact_match_metric.compute(predictions=predictions, references=references,
                                                            ignore_case=True, ignore_punctuation=True)
            bertscore_results = bertscore_metric.compute(predictions=predictions, references=references, lang="en")
            rouge_results = rouge.compute(predictions=predictions, references=references)
            
            sequence_dict["exact_match"] = max(sequence_dict["exact_match"], exact_match_results["exact_match"])
            sequence_dict['bertscore_precision'] = max(sequence_dict['bertscore_precision'], bertscore_results['precision'][0])
            sequence_dict['bertscore_recall'] = max(sequence_dict['bertscore_recall'], bertscore_results['recall'][0])
            sequence_dict['bertscore_f1'] = max(sequence_dict['bertscore_f1'], bertscore_results['f1'][0])
            
            for rouge_type in rouge_types:
                sequence_dict[rouge_type + '_to_target'] = max(sequence_dict[rouge_type + '_to_target'], rouge_results[rouge_type])
        
        # cleaned metrics
        for answer in reference_answers:
            if isinstance(answer, (tuple, list)):
                reference_str = str(answer[0])
            else:
                reference_str = str(answer)
            
            predictions = [sequence_dict["cleaned_most_likely_generation"].lstrip()]
            references = [reference_str]
            
            exact_match_results = exact_match_metric.compute(predictions=predictions, references=references,
                                                            ignore_case=True, ignore_punctuation=True)
            bertscore_results = bertscore_metric.compute(predictions=predictions, references=references, lang="en")
            rouge_results = rouge.compute(predictions=predictions, references=references)
            
            sequence_dict["cleaned_exact_match"] = max(sequence_dict["cleaned_exact_match"], exact_match_results["exact_match"])
            sequence_dict['cleaned_bertscore_precision'] = max(sequence_dict['cleaned_bertscore_precision'], bertscore_results['precision'][0])
            sequence_dict['cleaned_bertscore_recall'] = max(sequence_dict['cleaned_bertscore_recall'], bertscore_results['recall'][0])
            sequence_dict['cleaned_bertscore_f1'] = max(sequence_dict['cleaned_bertscore_f1'], bertscore_results['f1'][0])
            
            for rouge_type in rouge_types:
                sequence_dict["cleaned_" + rouge_type + '_to_target'] = max(sequence_dict["cleaned_" + rouge_type + '_to_target'], rouge_results[rouge_type])
        
        sequences.append(sequence_dict)
    
    return sequences
# model_name = model_dir.split("/")[-1]

if use_api:
    # Validate Qwen-family model requirement in API mode.
    if "qwen" not in api_model_name.lower():
        raise ValueError(f"API mode supports Qwen-family models only, got: {api_model_name}")
    
    # Run API generation.
    generations = get_generations_api(
        tokenizer, dataloader, 
        number_of_generations=num_generations_per_prompt,
        api_model_name=api_model_name,
        max_length_of_generated_sequence=max_length_of_generated_sequence,
        temperature=temperature,
        top_p=top_p
    )
    run_setting = f"{model_name}_API_{dataset_name}_fraction_{args.fraction}_max_length_{args.max_length}_num_generations_{args.num_generations_per_prompt}_temperature_{args.temperature}_top_p_{args.top_p}_seed_{args.seed}"
    os.makedirs(f"{output_dir}/{run_setting}", exist_ok=True)
    with open(f"{output_dir}/{run_setting}/generations.pkl", 'wb') as outfile:
        pickle.dump(generations, outfile)
else:
    # Local generation path.
    generations = get_generations(model, tokenizer, dataloader, number_of_generations=num_generations_per_prompt)
    run_setting = f"{model_name}_{dataset_name}_fraction_{args.fraction}_max_length_{args.max_length}_num_generations_{args.num_generations_per_prompt}_temperature_{args.temperature}_top_k_{args.top_k}_top_p_{args.top_p}_decode_method_{args.decode_method}_seed_{args.seed}"
    os.makedirs(f"{output_dir}/{run_setting}", exist_ok=True)
    with open(f"{output_dir}/{run_setting}/generations.pkl", 'wb') as outfile:
        pickle.dump(generations, outfile)
            
