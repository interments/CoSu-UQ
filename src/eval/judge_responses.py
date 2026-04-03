import config
import os
import pickle
import re
import tqdm
import time
import threading
import argparse
from utils.api_chat import Chat
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s"
)
logger = logging.getLogger(__name__)
os.chdir(config.run_dir)


def _resolve_env_placeholder(value: str) -> str:
    if value.startswith("${") and value.endswith("}"):
        env_name = value[2:-1].strip()
        return os.getenv(env_name, "")
    return value

class Judge:
    def __init__(self, chat: Chat, max_retries: int = 3, retry_interval: float = 10):
        self.chat = chat
        self.max_retries = max_retries
        self.retry_interval = retry_interval

    def judge(self, generations: list) -> None:
        for generation in tqdm.tqdm(generations, desc=f"Judging {self.chat.model}"):
            question = generation['prompt_text'].split("\n\n\n\n")[-1] if "medqa" in generation.get("question", "").lower() else generation["question"]
            cleaned_generated_texts = [generation["cleaned_most_likely_generation"]] + generation["cleaned_generated_texts"]
            answer = generation["answer"][0]
            labels = []
            for response in cleaned_generated_texts:
                prompt_text = f"""
You are a precise and objective evaluator. Your task is to determine whether the given Response correctly answers the Question, using the provided ground truth Answer as the reference.

Focus specifically on the semantic alignment between the Response and the Answer. It is acceptable for the Response to differ in wording or include additional context, as long as it is factually correct, consistent with the Answer, and fully addresses the Question.

Please proceed step-by-step:
Step 1: Carefully read and understand the Question.
Step 2: Understand the key facts and meaning conveyed in the Answer.
Step 3: Compare the Response with the Answer. Determine whether the Response expresses the same factual content and correctly responds to the Question.
Step 4: If the Response is semantically equivalent to the Answer, return 1.
Otherwise, return 0.

Now begin your reasoning step by step.

Question: {question}
Response: {response}
Answer: {answer}

In the final line, output only a single digit: 1 if the Response is correct, or 0 if it is not. Do not include any other text.
"""
                label = -1
                for attempt in range(self.max_retries):
                    try:
                        result_str = self.chat.ask(prompt_text)
                        numbers = re.findall(r'-?\d+\.?\d*(?:[eE][+-]?\d+)?', result_str)
                        label = int(float(numbers[-1])) if numbers and int(float(numbers[-1])) in [0,1] else -1
                        if label not in [0, 1]:
                            if attempt < self.max_retries - 1:
                                time.sleep(self.retry_interval)
                            continue
                        break
                    except Exception as e:
                        if attempt < self.max_retries - 1:
                            time.sleep(self.retry_interval)
                labels.append(label)
            # Save all labels into generation.
            generation[f"{self.chat.model}_judge_result_labels"] = labels

def parse_args():
    parser = argparse.ArgumentParser(description="Judge script with multi-thread and retry support")
    parser.add_argument('--max_retries', type=int, default=3, help='Maximum retry count')
    parser.add_argument('--retry_interval', type=float, default=5.0, help='Retry interval in seconds')
    parser.add_argument("--run_setting", type=str, default="", help="Run setting")
    parser.add_argument('--judgers', type=str, 
                        default=[
                            "gpt-4.1-mini[split]${OPENAI_API_BASE}[split]${OPENAI_API_KEY}[split]0.5[split]0.8",
                            "qwen-max[split]${QWEN_API_BASE}[split]${QWEN_API_KEY}[split]0.5[split]0.8",
                            "yi-lightning[split]${YI_API_BASE}[split]${YI_API_KEY}[split]0.5[split]0.8",
                            "yi-medium[split]${YI_API_BASE}[split]${YI_API_KEY}[split]0.5[split]0.8",
                            "Doubao-pro-256k[split]${DOUBAO_API_BASE}[split]${DOUBAO_API_KEY}[split]0.5[split]0.8",
                        ], 
                        nargs="+",help='Judger specs separated by space. Supports [split] format with plain values or env placeholders, e.g. "gpt-4.1-mini[split]${OPENAI_API_BASE}[split]${OPENAI_API_KEY}[split]0.5[split]0.8"')
    return parser.parse_args()

def multi_model_judge(generations, judgers, max_retries, retry_interval):
    threads = []
    for judger in judgers:
        chat = Chat(
            api_key=judger["api_key"],
            api_base=judger["api_base"],
            model=judger["model"],
            temperature=judger["temperature"],
            top_p=judger["top_p"]
        )
        judge_instance = Judge(chat, max_retries=max_retries, retry_interval=retry_interval)
        t = threading.Thread(target=judge_instance.judge, args=(generations,))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

def llm_vote(generations, judgers):
    for generation in generations:
        # Use judge result length to keep voting dimension consistent.
        num_sentences = len(generation[f"{judgers[0]['model']}_judge_result_labels"])
        final_labels = []
        for i in range(num_sentences):
            labels = [generation.get(f"{judger['model']}_judge_result_labels", [])[i] for judger in judgers]
            valid_labels = [label for label in labels if label in [0, 1]]
            if valid_labels:
                zeros = valid_labels.count(0)
                ones = valid_labels.count(1)
                if zeros > ones:
                    final_labels.append(0)
                elif ones > zeros:
                    final_labels.append(1)
                else:
                    final_labels.append(1)
            else:
                final_labels.append(-1)
        generation["final_judge_result_labels"] = final_labels

if __name__ == "__main__":
    args = parse_args()
    if args.run_setting == "":
        with open(f'{config.output_dir}/run_setting.txt', 'r') as f:
            run_setting = f.read()
    else:
        run_setting = args.run_setting
    with open(f"{config.output_dir}/{run_setting}/generations.pkl","rb") as infile:
        generations = pickle.load(infile)
    judgers = []
    for judge in args.judgers:
        parts = judge.split("[split]")
        if len(parts) != 5:
            raise ValueError(f"Invalid judger format: {judge}")
        model = parts[0].strip()
        api_base = _resolve_env_placeholder(parts[1].strip())
        api_key = _resolve_env_placeholder(parts[2].strip())
        if not api_base or not api_key:
            raise ValueError(
                f"Missing api_base/api_key for judger '{model}'. "
                "Please pass plain values or ${ENV_VAR} placeholders with exported env vars."
            )
        judgers.append(
            {
                "model": model,
                "api_base": api_base,
                "api_key": api_key,
                "temperature": float(parts[3]),
                "top_p": float(parts[4]),
            }
        )
    multi_model_judge(generations, judgers, args.max_retries, args.retry_interval)
    llm_vote(generations, judgers)
    with open(f"{config.output_dir}/{run_setting}/generations.pkl", "wb") as outfile:
        pickle.dump(generations, outfile)
    logger.info(f"Final judge results saved to {config.output_dir}/{run_setting}/generations.pkl")
