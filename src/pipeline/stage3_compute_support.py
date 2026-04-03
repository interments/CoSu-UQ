import json
import torch
import tqdm
from transformers import AutoTokenizer,AutoModelForSequenceClassification
import os
import config
import numpy as np
import re
import argparse
from utils.api_chat import Chat
import logging
from concurrent.futures import ThreadPoolExecutor
import time
import pickle


# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s"
)
logger = logging.getLogger(__name__)

def extract_facts_from_response(answer: str) -> list:
    """Extract content wrapped by <begin>...<end> and split by <split>."""
    if not answer:
        return []
    
    # Match content between <begin> and <end>.
    match = re.search(r'<begin>(.*?)<end>', answer, re.DOTALL)
    if match:
        content = match.group(1)
        facts = [fact.strip() for fact in content.split('<split>') if fact.strip()]
        return facts
    
    return []

def sentence_split_step_answer(response: str) -> list:
    """
    Deterministically split a response while preserving original text fragments.
    Split points:
    - before **Step X: or Step X:
    - before **Final Answer: or Final Answer:
    """
    import re
    
    # Keep ** inside lookahead so markdown-emphasized and plain forms are handled.
    pattern = r'(?=\*\*Step\s*\d+\s*:)|(?=(?<!\*)Step\s*\d+\s*:)|(?=\*\*Final\s*Answer\s*:)|(?=(?<!\*)Final\s*Answer\s*:)'
    
    parts = re.split(pattern, response, flags=re.IGNORECASE)
    
    # Filter empty fragments.
    result = []
    for part in parts:
        stripped = part.strip()
        if stripped:
            result.append(stripped)
    
    return result

def sentence_split_spacy(text):
    """
    Split text into sentences with spaCy, supporting paragraph breaks via \\n\\n.
    """
    if not text:
        return []
    # Split by paragraph first, then sentence-split each segment.
    try:
        segments = [seg.strip() for seg in text.split('\n\n') if seg.strip()]
        sentences = []
        for seg in segments:
            doc = nlp(seg)
            sentences.extend([sent.text for sent in doc.sents])
        return sentences
    except Exception as e:
        logger.error(f"Sentence splitting failed: {e}")
        return [text]

def sentence_split_nltk(text):
    """
    Split text into sentences with NLTK.
    """
    if not text:
        return []
    try:
        sentences = nltk.sent_tokenize(text)
        return sentences
    except Exception as e:
        logger.error(f"Sentence splitting failed: {e}")
        return [text]

class LUQCalculator:
    def __init__(self, nli_model, nli_tokenizer):
        self.nli_model = nli_model
        self.nli_tokenizer = nli_tokenizer
        self.last_splited_response = None
        self.last_nli_probability_matrix = None  # Store NLI probability matrix for debugging/inspection.

    def compute_entail_prob(self, sentence_a, sentence_b) -> float:
        inputs = self.nli_tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=[(sentence_a, sentence_b)],
            add_special_tokens=True, return_tensors="pt"
        ).to("cuda")
        logits = self.nli_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0].detach().cpu()
        return probs[0].item()

    def compute_similarity_score(self, response_a, response_b) -> float:
        sentences = sentence_split(response_a.strip())
        probabilities = [self.compute_entail_prob(sentence_a, response_b) for sentence_a in sentences]
        return sum(probabilities) / len(probabilities) if probabilities else 0.0 

    def compute_uncertainty_score(self, responses: list) -> tuple[float, list]:
        if len(responses) < 2:
            return 0.0, []
        
        uncertainty_scores = []
        splited_responses = [sentence_split(response.strip()) for response in responses]
        self.last_splited_response = splited_responses
        
        # Initialize NLI probability matrix.
        nli_matrix = []
        
        for i in range(len(splited_responses)):
            similarity_scores = []
            nli_matrix_i = []  # Probabilities between response i and other responses.
            
            for j in range(len(splited_responses)):
                if i != j:
                    responses_i = splited_responses[i]
                    responses_j = responses[j].strip()
                    
                    # Compute and store NLI probabilities for each sentence.
                    sentence_probabilities = []
                    for sentence in responses_i:
                        prob = self.compute_entail_prob(sentence, responses_j)
                        sentence_probabilities.append({
                            "sentence_i": sentence,
                            "response_j": responses_j,
                            "probability": prob
                        })
                    
                    nli_matrix_i.append({
                        "response_j_index": j,
                        "sentence_probabilities": sentence_probabilities
                    })
                    
                    # Compute similarity score.
                    probabilities = [sp["probability"] for sp in sentence_probabilities]
                    similarity_scores.append(sum(probabilities) / len(probabilities) if probabilities else 0.0)
            
            nli_matrix.append({
                "response_i_index": i,
                "comparisons": nli_matrix_i
            })
            uncertainty_scores.append(1 - (sum(similarity_scores) / len(similarity_scores)))
        
        # Persist NLI probability matrix for this sample.
        self.last_nli_probability_matrix = nli_matrix
        
        return sum(uncertainty_scores) / len(uncertainty_scores), uncertainty_scores
        

class LUQAtomicCalculator(LUQCalculator):
    def __init__(
        self,
        nli_model,
        nli_tokenizer,
        api_key="",
        api_base="",
        model="Doubao-pro-32k",
        temperature=1,
        num_threads=4
    ):
        super().__init__(nli_model, nli_tokenizer)
        self.api_key = api_key
        self.api_base = api_base
        self.model = model
        self.temperature = temperature
        self.chat = Chat(
            api_key=self.api_key,
            api_base=self.api_base,
            model=self.model,
            temperature=self.temperature
        )
        self.num_threads = num_threads
        self.last_nli_probability_matrix = None  # Store NLI probability matrix for debugging/inspection.

    def compute_similarity_score(self, response_a, response_b) -> float:
        facts_a = self.get_atomic_facts(response_a)
        if not facts_a:
            return 0.0
        probabilities = [self.compute_entail_prob(sentence_a, response_b) for sentence_a in facts_a]
        return sum(probabilities) / len(probabilities) if probabilities else 0.0
    
    def compute_uncertainty_score(self, responses: list) -> tuple[float, list]:
        if len(responses) < 2:
            return 0.0, []
        
        uncertainty_scores = []
        
        # Fetch atomic facts with multithreading.
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = [executor.submit(self.get_atomic_facts, response.strip()) for response in responses]
            splited_responses = [future.result() for future in futures]
        
        self.last_splited_response = splited_responses
        
        # Initialize NLI probability matrix.
        nli_matrix = []
        
        for i in range(len(splited_responses)):
            similarity_scores = []
            nli_matrix_i = []  # Probabilities between response i and other responses.
            
            for j in range(len(responses)):
                if i != j:
                    responses_i = splited_responses[i]
                    responses_j = responses[j].strip()
                    
                    # Compute and store NLI probabilities for each atomic fact.
                    fact_probabilities = []
                    for fact in responses_i:
                        prob = self.compute_entail_prob(fact, responses_j)
                        fact_probabilities.append({
                            "atomic_fact": fact,
                            "response_j": responses_j,
                            "probability": prob
                        })
                    
                    nli_matrix_i.append({
                        "response_j_index": j,
                        "fact_probabilities": fact_probabilities
                    })
                    
                    # Compute similarity score.
                    probabilities = [fp["probability"] for fp in fact_probabilities]
                    similarity_scores.append(sum(probabilities) / len(probabilities) if probabilities else 0.0)
            
            nli_matrix.append({
                "response_i_index": i,
                "comparisons": nli_matrix_i
            })
            uncertainty_scores.append(1 - (sum(similarity_scores) / len(similarity_scores)))
        
        # Persist NLI probability matrix for this sample.
        self.last_nli_probability_matrix = nli_matrix
        
        return sum(uncertainty_scores) / len(uncertainty_scores), uncertainty_scores
    
    def get_atomic_facts(self, sent, try_num=5) -> list:
        template="""
        Request:
        1 - You are given a sentence. Your task is to break the sentence down into a list of atomic facts.
        2 - An atomic fact is a sentence containing a singular piece of information.
        3 - Each atomic fact in the outputted list should check a different piece of information.
        4 - Use the following examples to learn how to do this.
        5 - Your task is to do this for the last sentence that is given!!!
        6 - The output please follow the format of ***OUTPUT***!!!

        Please breakdown the following sentence into independent facts:
        Step 1: The Eiffel Tower is located in Paris.\nStep 2: Paris is the capital city of France.\nStep 3: Therefore, the Eiffel Tower is located in the capital of France.\nFinal Answer: Paris
        Facts:<begin>The Eiffel Tower is located in Paris.<split>Paris is the capital city of France.<split>The Eiffel Tower is located in the capital of France.<split>The final answer is Paris.<end>

        Please breakdown the following sentence into independent facts:
        Sentence:Step 1: Billy's first 3 customers buy one DVD each, so we multiply 3 by 1: 3 * 1 = 3.\nStep 2: His next 2 customers buy 2 DVDs each, so we multiply 2 by 2: 2 * 2 = 4.\nStep 3: To find the total number of DVDs sold, we add the DVDs sold to the first 3 customers and the next 2 customers: 3 + 4 = 7.\nStep 4: Since the last 3 customers didn't buy any DVDs, it doesn't affect the total number of DVDs sold.\nFinal Answer: 7
        Facts:<begin>Billy’s first 3 customers each buy 1 DVD.<split>These 3 customers buy a total of 3 DVDs.<split>Billy’s next 2 customers each buy 2 DVDs.<split>These 2 customers buy a total of 4 DVDs.<split>The last 3 customers do not buy any DVDs.<split>Billy sells a total of 7 DVDs.<end>

        Please breakdown the following sentence into independent facts:
        Sentence:Step 1: Consider two cases: $x \ge 0$ and $x < 0$.\nStep 2: Case 1 ($x \ge 0$): The equation becomes $x = 2x - 3$.\nStep 3: Solve: $x = 3$, which satisfies $x \ge 0$, so it is valid.\nStep 4: Case 2 ($x < 0$): The equation becomes $-x = 2x - 3$.\nStep 5: Solve: $3 = 3x$, so $x = 1$, but $1$ does not satisfy $x < 0$, so this solution is invalid.\nFinal Answer: $3$
        Facts:<begin>Two cases are considered: $x \ge 0$ and $x < 0$.<split>When $x \ge 0$, the equation becomes $x = 2x - 3$.<split>Solving this gives $x = 3$, which satisfies $x \ge 0$.<split>When $x < 0$, the equation becomes $-x = 2x - 3$.<split>Solving this gives $x = 1$, which does not satisfy $x < 0$.<split>The valid solution is $3$.<end>


        Please breakdown the following sentence into independent facts:
        Sentence: \"{SENTENCE_PLACEHOLDER}\"
        ***OUTPUT***:Facts:
        """
        try_num = try_num
        input_temp = template.replace("{SENTENCE_PLACEHOLDER}", sent)
        num = 0
        
        while(num < try_num):
            facts = []
            answer = self.chat.ask(input_temp).replace("\n", "")
            logger.info(f"Atomic facts generation attempt {num+1}, answer: {answer}")
            
            try:
                # Extract using outermost <begin>/<end> and <split> markers.
                facts = extract_facts_from_response(answer)
                
                if facts:
                    logger.info(f"Successfully extracted {len(facts)} atomic facts")
                    return facts
                
            except Exception as e:
                logger.warning(f"Parsing failed (attempt {num+1}): {e}")
                
            # Strengthen prompt on retry.
            if num < try_num - 1:
                input_temp = template.replace("{SENTENCE_PLACEHOLDER}", sent) + \
                            f"\n\nIMPORTANT: Output must be in format <split>fact1<split>fact2<split>fact3<split>"
                time.sleep(10)
            num += 1
        
        logger.warning(f"Try {num} times but can not properly generate Facts! Sentence: {sent}")
        return []

class LUQPairCalculator(LUQCalculator):
    def __init__(self, nli_model, nli_tokenizer):
        super().__init__(nli_model, nli_tokenizer)
        self.last_nli_probability_matrix = None  # Store NLI probability matrix for debugging/inspection.

    def compute_similarity_score(self, response_a, response_b) -> float:
        sentences_a = sentence_split(response_a.strip())
        sentences_b = sentence_split(response_b.strip())
        if not sentences_a or not sentences_b:
            return 0.0
        probabilities = []
        for sa in sentences_a:
            max_prob = max([self.compute_entail_prob(sa, sb) for sb in sentences_b])
            probabilities.append(max_prob)
        return sum(probabilities) / len(probabilities) if probabilities else 0.0
    
    def compute_uncertainty_score(self, responses: list) -> tuple[float, list]:
        if len(responses) < 2:
            return 0.0, []
        
        uncertainty_scores = []
        splited_responses = [sentence_split(response.strip()) for response in responses]
        self.last_splited_response = splited_responses
        
        # Initialize NLI probability matrix.
        # Shape: nli_matrix[i][j] = { "sentence_i": str, "sentence_j": str, "probability": float }
        nli_matrix = []
        
        for i in range(len(splited_responses)):
            similarity_scores = []
            nli_matrix_i = []  # Probabilities between response i and other responses.
            
            for j in range(len(splited_responses)):
                if i != j:
                    # Compute and store NLI probabilities for sentence pairs.
                    sentence_pairs = []
                    for response_i in splited_responses[i]:
                        max_prob = -1
                        best_match = None
                        probs_for_this_sentence = []
                        
                        for response_j in splited_responses[j]:
                            prob = self.compute_entail_prob(response_i, response_j)
                            probs_for_this_sentence.append({
                                "sentence_i": response_i,
                                "sentence_j": response_j,
                                "probability": prob
                            })
                            if prob > max_prob:
                                max_prob = prob
                                best_match = response_j
                        
                        sentence_pairs.append({
                            "sentence_i": response_i,
                            "best_match_sentence_j": best_match,
                            "max_probability": max_prob,
                            "all_probabilities": probs_for_this_sentence
                        })
                    
                    nli_matrix_i.append({
                        "response_j_index": j,
                        "sentence_pairs": sentence_pairs
                    })
                    
                    # Compute similarity score.
                    probabilities = [pair["max_probability"] for pair in sentence_pairs]
                    similarity_scores.append(sum(probabilities) / len(probabilities) if probabilities else 0.0)
            
            nli_matrix.append({
                "response_i_index": i,
                "comparisons": nli_matrix_i
            })
            uncertainty_scores.append(1 - (sum(similarity_scores) / len(similarity_scores)))
        
        # Persist NLI probability matrix for this sample.
        self.last_nli_probability_matrix = nli_matrix
        
        return sum(uncertainty_scores) / len(uncertainty_scores), uncertainty_scores

class LUQAtomicPairCalculator(LUQCalculator):
    def __init__(
        self,
        nli_model,
        nli_tokenizer,
        api_key="",
        api_base="",
        model="Doubao-pro-32k",
        temperature=1,
        num_threads=4
    ):
        super().__init__(nli_model, nli_tokenizer)
        self.api_key = api_key
        self.api_base = api_base
        self.model = model
        self.temperature = temperature
        self.chat = Chat(
            api_key=self.api_key,
            api_base=self.api_base,
            model=self.model,
            temperature=self.temperature
        )
        self.num_threads = num_threads
        self.last_nli_probability_matrix = None  # Store NLI probability matrix for debugging/inspection.

    def compute_similarity_score(self, response_a, response_b) -> float:
        facts_a = self.get_atomic_facts(response_a)
        facts_b = self.get_atomic_facts(response_b)
        if not facts_a or not facts_b:
            return 0.0
        probabilities = []
        for fact_a in facts_a:
            max_prob = max([self.compute_entail_prob(fact_a, fact_b) for fact_b in facts_b])
            probabilities.append(max_prob)
        return sum(probabilities) / len(probabilities) if probabilities else 0.0
    
    def compute_uncertainty_score(self, responses: list) -> tuple[float, list]:
        if len(responses) < 2:
            return 0.0, []
        
        uncertainty_scores = []
        
        # Fetch atomic facts with multithreading.
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = [executor.submit(self.get_atomic_facts, response.strip()) for response in responses]
            splited_responses = [future.result() for future in futures]
        
        self.last_splited_response = splited_responses
        
        # Initialize NLI probability matrix.
        nli_matrix = []
        
        for i in range(len(splited_responses)):
            similarity_scores = []
            nli_matrix_i = []  # Probabilities between response i and other responses.
            
            for j in range(len(splited_responses)):
                if i != j:
                    facts_i = splited_responses[i]
                    facts_j = splited_responses[j]
                    
                    # Compute and store NLI probabilities for atomic-fact pairs.
                    fact_pairs = []
                    for fact_i in facts_i:
                        max_prob = -1
                        best_match = None
                        probs_for_this_fact = []
                        
                        for fact_j in facts_j:
                            prob = self.compute_entail_prob(fact_i, fact_j)
                            probs_for_this_fact.append({
                                "fact_i": fact_i,
                                "fact_j": fact_j,
                                "probability": prob
                            })
                            if prob > max_prob:
                                max_prob = prob
                                best_match = fact_j
                        
                        fact_pairs.append({
                            "fact_i": fact_i,
                            "best_match_fact_j": best_match,
                            "max_probability": max_prob,
                            "all_probabilities": probs_for_this_fact
                        })
                    
                    nli_matrix_i.append({
                        "response_j_index": j,
                        "fact_pairs": fact_pairs
                    })
                    
                    # Compute similarity score.
                    probabilities = [pair["max_probability"] for pair in fact_pairs]
                    similarity_scores.append(sum(probabilities) / len(probabilities) if probabilities else 0.0)
            
            nli_matrix.append({
                "response_i_index": i,
                "comparisons": nli_matrix_i
            })
            uncertainty_scores.append(1 - (sum(similarity_scores) / len(similarity_scores)))
        
        # Persist NLI probability matrix for this sample.
        self.last_nli_probability_matrix = nli_matrix
        
        return sum(uncertainty_scores) / len(uncertainty_scores), uncertainty_scores
    
    def get_atomic_facts(self, sent, try_num=5) -> list:
        template="""
        Request:
        1 - You are given a sentence. Your task is to break the sentence down into a list of atomic facts.
        2 - An atomic fact is a sentence containing a singular piece of information.
        3 - Each atomic fact in the outputted list should check a different piece of information.
        4 - Use the following examples to learn how to do this.
        5 - Your task is to do this for the last sentence that is given!!!
        6 - The output please follow the format of ***OUTPUT***!!!

        Please breakdown the following sentence into independent facts:
        Step 1: The Eiffel Tower is located in Paris.\nStep 2: Paris is the capital city of France.\nStep 3: Therefore, the Eiffel Tower is located in the capital of France.\nFinal Answer: Paris
        Facts:<begin>The Eiffel Tower is located in Paris.<split>Paris is the capital city of France.<split>The Eiffel Tower is located in the capital of France.<split>The final answer is Paris.<end>

        Please breakdown the following sentence into independent facts:
        Sentence:Step 1: Billy's first 3 customers buy one DVD each, so we multiply 3 by 1: 3 * 1 = 3.\nStep 2: His next 2 customers buy 2 DVDs each, so we multiply 2 by 2: 2 * 2 = 4.\nStep 3: To find the total number of DVDs sold, we add the DVDs sold to the first 3 customers and the next 2 customers: 3 + 4 = 7.\nStep 4: Since the last 3 customers didn't buy any DVDs, it doesn't affect the total number of DVDs sold.\nFinal Answer: 7
        Facts:<begin>Billy’s first 3 customers each buy 1 DVD.<split>These 3 customers buy a total of 3 DVDs.<split>Billy’s next 2 customers each buy 2 DVDs.<split>These 2 customers buy a total of 4 DVDs.<split>The last 3 customers do not buy any DVDs.<split>Billy sells a total of 7 DVDs.<end>

        Please breakdown the following sentence into independent facts:
        Sentence:Step 1: Consider two cases: $x \ge 0$ and $x < 0$.\nStep 2: Case 1 ($x \ge 0$): The equation becomes $x = 2x - 3$.\nStep 3: Solve: $x = 3$, which satisfies $x \ge 0$, so it is valid.\nStep 4: Case 2 ($x < 0$): The equation becomes $-x = 2x - 3$.\nStep 5: Solve: $3 = 3x$, so $x = 1$, but $1$ does not satisfy $x < 0$, so this solution is invalid.\nFinal Answer: $3$
        Facts:<begin>Two cases are considered: $x \ge 0$ and $x < 0$.<split>When $x \ge 0$, the equation becomes $x = 2x - 3$.<split>Solving this gives $x = 3$, which satisfies $x \ge 0$.<split>When $x < 0$, the equation becomes $-x = 2x - 3$.<split>Solving this gives $x = 1$, which does not satisfy $x < 0$.<split>The valid solution is $3$.<end>


        Please breakdown the following sentence into independent facts:
        Sentence: \"{SENTENCE_PLACEHOLDER}\"
        ***OUTPUT***:Facts:
        """
        try_num = try_num
        input_temp = template.replace("{SENTENCE_PLACEHOLDER}", sent)
        num = 0
        
        while(num < try_num):
            facts = []
            answer = self.chat.ask(input_temp).replace("\n", "")
            logger.info(f"Atomic facts generation attempt {num+1}, answer: {answer}")
            
            try:
                # Extract using outermost <begin>/<end> and <split> markers.
                facts = extract_facts_from_response(answer)
                
                if facts:
                    logger.info(f"Successfully extracted {len(facts)} atomic facts")
                    return facts
                
            except Exception as e:
                logger.warning(f"Parsing failed (attempt {num+1}): {e}")
                
            # Strengthen prompt on retry.
            if num < try_num - 1:
                input_temp = template.replace("{SENTENCE_PLACEHOLDER}", sent) + \
                            f"\n\nIMPORTANT: Output must be in format <split>fact1<split>fact2<split>fact3<split>"
                time.sleep(10)
            num += 1
        
        logger.warning(f"Try {num} times but can not properly generate Facts! Sentence: {sent}")
        return []
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LUQ Score Calculation")
    parser.add_argument("--device", type=str, default="0", help="CUDA device id")
    parser.add_argument("--model_name", type=str, default="potsawee/deberta-v3-large-mnli", help="NLI model name")
    parser.add_argument("--api_key", type=str, default=os.getenv("COSU_UQ_API_KEY", ""), help="API KEY")
    parser.add_argument("--api_base", type=str, default=os.getenv("COSU_UQ_API_BASE", ""), help="API BASE")
    parser.add_argument("--api_model", type=str, default="Doubao-pro-32k", help="API model name")
    parser.add_argument("--api_temperature", type=float, default=1, help="API temperature")
    parser.add_argument("--num_threads", type=int, default=5, help="API thread count")
    parser.add_argument("--luq_method", type=str, default="LUQPair", 
                        choices=["LUQ","LUQAtomic","LUQPair","LUQAtomicPair"],
                        help="LUQ uncertainty method")
    parser.add_argument("--run_setting", type=str, default="", help="Run setting")
    parser.add_argument("--split_method", type=str, default="step_answer", 
                        choices=["nltk", "spacy","step_answer"], help="Sentence split method")
    parser.add_argument("--use_greedy", type=str, default="False", help="Whether to include greedy output")
    parser.add_argument("--save_matrix", action="store_true", help="Whether to save nli_probability_matrix")
    
    args = parser.parse_args()
    if args.luq_method in {"LUQAtomic", "LUQAtomicPair"} and (not args.api_key or not args.api_base):
        raise ValueError("LUQAtomic/LUQAtomicPair require --api_key and --api_base (or COSU_UQ_API_KEY/COSU_UQ_API_BASE).")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    os.chdir(config.run_dir)
    os.environ["HF_DATASETS_CACHE"] = config.hf_datasets_cache
    seed_value = 42
    
    if args.split_method == "nltk":
        import nltk
        nltk.download('punkt', download_dir=f'{config.run_dir}/nltk_data')
        sentence_split = sentence_split_nltk
    elif args.split_method == "spacy":
        import spacy
        nlp = spacy.load("en_core_web_sm")
        sentence_split = sentence_split_spacy
    elif args.split_method == "step_answer":
        sentence_split = sentence_split_step_answer
    else:
        raise ValueError("Unsupported split method. Choose 'nltk' or 'spacy' or 'step_answer'.")

    model_name = args.model_name
    nli_model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir="../models").to("cuda")
    nli_tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="../models")
    if args.run_setting == "":
        with open(f'{config.output_dir}/run_setting.txt', 'r') as f:
            run_setting = f.read()
    else:
        run_setting = args.run_setting
    input_file = f"{config.output_dir}/{run_setting}/generations.pkl"
    luq_calculator = LUQCalculator(nli_model, nli_tokenizer)
    luqAtomic_calculator = LUQAtomicCalculator(
        nli_model, nli_tokenizer,
        api_key=args.api_key,
        api_base=args.api_base,
        model=args.api_model,
        temperature=args.api_temperature,
        num_threads=args.num_threads
    )
    luqPair_calculator = LUQPairCalculator(nli_model, nli_tokenizer)
    luqAtomicPair_calculator = LUQAtomicPairCalculator(
        nli_model, nli_tokenizer,
        api_key=args.api_key,
        api_base=args.api_base,
        model=args.api_model,
        temperature=args.api_temperature,
        num_threads=args.num_threads
    )
    with open(input_file, "rb") as infile:
        generated_results = pickle.load(infile)

    LUQ_results = []
    luq_method = args.luq_method
    for result in tqdm.tqdm(generated_results, desc="Processing responses"):
        if args.use_greedy.lower() == "true":
            responses = [result["most_likely_generation"]] + [text for text in result["generated_texts"]]
        else:
            responses = [text for text in result["generated_texts"]]
        uncertainty_score = -1
        uncertainty_scores = []
        splited_response = None
        nli_probability_matrix = None
        error_msg = ""
        try:
            if luq_method == "LUQ":
                uncertainty_score, uncertainty_scores = luq_calculator.compute_uncertainty_score(responses)
                splited_response = luq_calculator.last_splited_response
                nli_probability_matrix = luq_calculator.last_nli_probability_matrix
            elif luq_method == "LUQAtomic":
                uncertainty_score, uncertainty_scores = luqAtomic_calculator.compute_uncertainty_score(responses)
                splited_response = luqAtomic_calculator.last_splited_response
                nli_probability_matrix = luqAtomic_calculator.last_nli_probability_matrix
            elif luq_method == "LUQPair":
                uncertainty_score, uncertainty_scores = luqPair_calculator.compute_uncertainty_score(responses)
                splited_response = luqPair_calculator.last_splited_response
                nli_probability_matrix = luqPair_calculator.last_nli_probability_matrix
            elif luq_method == "LUQAtomicPair":  
                uncertainty_score, uncertainty_scores = luqAtomicPair_calculator.compute_uncertainty_score(responses)
                splited_response = luqAtomicPair_calculator.last_splited_response
                nli_probability_matrix = luqAtomicPair_calculator.last_nli_probability_matrix
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error processing sample {result['id']}: {e}")
        
        result_item = {
            "id": result["id"],
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
        LUQ_results.append(result_item)
    if args.use_greedy.lower() == "true":
        output_file = f"{config.output_dir}/{run_setting}/{luq_method}_{args.split_method}_use_greedy_splited_results.json"
    else:
        output_file = f"{config.output_dir}/{run_setting}/{luq_method}_{args.split_method}_splited_results.json"
    with open(output_file, "w", encoding="utf-8") as outfile:
        json.dump(LUQ_results, outfile, indent=4, ensure_ascii=False)
    logger.info(f"LUQ results saved to {output_file}")
