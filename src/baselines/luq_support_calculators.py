from typing import Callable, List, Optional, Tuple

import torch


class LUQCalculator:
    def __init__(
        self,
        nli_model,
        nli_tokenizer,
        sentence_splitter: Callable[[str], List[str]],
        device: Optional[str] = None,
    ):
        self.nli_model = nli_model
        self.nli_tokenizer = nli_tokenizer
        self.sentence_splitter = sentence_splitter
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.last_splited_response = None
        self.last_nli_probability_matrix = None

    def compute_entail_prob(self, sentence_a: str, sentence_b: str) -> float:
        inputs = self.nli_tokenizer.batch_encode_plus(
            batch_text_or_text_pairs=[(sentence_a, sentence_b)],
            add_special_tokens=True,
            return_tensors="pt",
        ).to(self.device)
        logits = self.nli_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0].detach().cpu()
        return probs[0].item()

    def compute_uncertainty_score(self, responses: List[str]) -> Tuple[float, List[float]]:
        if len(responses) < 2:
            return 0.0, []

        uncertainty_scores = []
        splited_responses = [self.sentence_splitter(response.strip()) for response in responses]
        self.last_splited_response = splited_responses

        nli_matrix = []
        for i in range(len(splited_responses)):
            similarity_scores = []
            nli_matrix_i = []
            for j in range(len(splited_responses)):
                if i == j:
                    continue
                responses_i = splited_responses[i]
                response_j = responses[j].strip()
                sentence_probabilities = []
                for sentence in responses_i:
                    prob = self.compute_entail_prob(sentence, response_j)
                    sentence_probabilities.append(
                        {
                            "sentence_i": sentence,
                            "response_j": response_j,
                            "probability": prob,
                        }
                    )

                nli_matrix_i.append(
                    {
                        "response_j_index": j,
                        "sentence_probabilities": sentence_probabilities,
                    }
                )
                probabilities = [sp["probability"] for sp in sentence_probabilities]
                similarity_scores.append(sum(probabilities) / len(probabilities) if probabilities else 0.0)

            nli_matrix.append({"response_i_index": i, "comparisons": nli_matrix_i})
            uncertainty_scores.append(1 - (sum(similarity_scores) / len(similarity_scores)))

        self.last_nli_probability_matrix = nli_matrix
        return sum(uncertainty_scores) / len(uncertainty_scores), uncertainty_scores


class LUQPairCalculator(LUQCalculator):
    def compute_uncertainty_score(self, responses: List[str]) -> Tuple[float, List[float]]:
        if len(responses) < 2:
            return 0.0, []

        uncertainty_scores = []
        splited_responses = [self.sentence_splitter(response.strip()) for response in responses]
        self.last_splited_response = splited_responses

        nli_matrix = []
        for i in range(len(splited_responses)):
            similarity_scores = []
            nli_matrix_i = []
            for j in range(len(splited_responses)):
                if i == j:
                    continue
                sentence_pairs = []
                for response_i_sentence in splited_responses[i]:
                    max_prob = -1.0
                    best_match = None
                    probs_for_sentence = []
                    for response_j_sentence in splited_responses[j]:
                        prob = self.compute_entail_prob(response_i_sentence, response_j_sentence)
                        probs_for_sentence.append(
                            {
                                "sentence_i": response_i_sentence,
                                "sentence_j": response_j_sentence,
                                "probability": prob,
                            }
                        )
                        if prob > max_prob:
                            max_prob = prob
                            best_match = response_j_sentence

                    sentence_pairs.append(
                        {
                            "sentence_i": response_i_sentence,
                            "best_match_sentence_j": best_match,
                            "max_probability": max_prob,
                            "all_probabilities": probs_for_sentence,
                        }
                    )

                nli_matrix_i.append({"response_j_index": j, "sentence_pairs": sentence_pairs})
                probabilities = [pair["max_probability"] for pair in sentence_pairs]
                similarity_scores.append(sum(probabilities) / len(probabilities) if probabilities else 0.0)

            nli_matrix.append({"response_i_index": i, "comparisons": nli_matrix_i})
            uncertainty_scores.append(1 - (sum(similarity_scores) / len(similarity_scores)))

        self.last_nli_probability_matrix = nli_matrix
        return sum(uncertainty_scores) / len(uncertainty_scores), uncertainty_scores
