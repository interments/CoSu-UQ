import re
import torch
import math
# def step_exacts_2_list(response):
#     # Split response into lines and filter out empty lines
#     lines = response.splitlines()
#     lines = [line for line in lines if line.strip()]

#     keywords_by_step = []
#     contributions_by_step = []
#     valid_response_text = []

#     for line in lines:
#         # Match lines starting with "Step X:"
#         match = re.search(r"Step \d+: (.+)", line)
#         if match:
#             if "(/" not in line or "/)" not in line:
#                 continue  # Skip invalid lines

#             # Extract keywords with contributions
#             keywords_w_contribution = match.group(1).split("; ")

#             # Check for valid format and skip invalid lines
#             if any("(/" not in key_w_c or "/)" not in key_w_c for key_w_c in keywords_w_contribution):
#                 continue

#             try:
#                 # Extract keywords and contributions
#                 keywords = [key_w_c.split("(/")[0].strip() for key_w_c in keywords_w_contribution]
#                 contributions = [int(key_w_c.split("(/")[1].split("/)")[0].strip()) for key_w_c in keywords_w_contribution]
#             except ValueError:
#                 return False  # Return False if contributions cannot be converted to int

#             for i in contributions:
#                 if i > 10:
#                     return False

#             keywords_by_step.append(keywords)
#             contributions_by_step.append(contributions)
#             valid_response_text.append(line)  # Add valid lines from the original response

#     # If no valid lines are found, return False
#     if not valid_response_text:
#         return False

#     return "\n".join(valid_response_text), keywords_by_step, contributions_by_step

def step_exacts_2_list(response, prompt_type):
    # Split response into lines and filter out empty lines
    lines = response.splitlines()
    lines = [line for line in lines if line.strip()]

    keywords_by_step = []
    contributions_by_step = []
    valid_response_text = []

    for line in lines:
        # Match lines starting with "Step X:"
        match = re.search(r"Step \d+: (.+)", line)
        if match:
            step_content = match.group(1).strip()
            
            # Handle NO ANSWER lines.
            if step_content.upper() == "NO ANSWER":
                keywords_by_step.append(["NO ANSWER"])
                contributions_by_step.append([0])  # Default contribution.
                valid_response_text.append(line)
                continue
            
            # Require contribution score format.
            if "(/" not in line or "/)" not in line:
                continue  # Skip invalid lines

            # Extract keywords with contributions
            keywords_w_contribution = step_content.split("; ")

            # Check for valid format and skip invalid lines
            if any("(/" not in key_w_c or "/)" not in key_w_c for key_w_c in keywords_w_contribution):
                continue

            try:
                # Extract keywords and contributions
                keywords = [key_w_c.split("(/")[0].strip() for key_w_c in keywords_w_contribution]
                contributions = [int(key_w_c.split("(/")[1].split("/)")[0].strip()) for key_w_c in keywords_w_contribution]
            except ValueError:
                return False  # Return False if contributions cannot be converted to int

            for i in contributions:
                # TODO: make threshold fully configurable.
                if prompt_type == "0-100":
                    if i > 100:
                        return False
                else:
                    if i > 10:
                        return False

            keywords_by_step.append(keywords)
            contributions_by_step.append(contributions)
            valid_response_text.append(line)

    # If no valid lines are found, return False
    if not valid_response_text:
        return False

    return "\n".join(valid_response_text), keywords_by_step, contributions_by_step

def parse_response_to_dict(response):
    steps = {}  
    final_answer = None

    # Match Final Answer
    match = re.search(r"Final Answer:\s*(.+?)\s*(?=(\n|$))", response, re.DOTALL)
    if match:
        final_answer = match.group(1).strip()
        response_before_final_answer = response[:match.start()].strip()
    else:
        return None, None, None

    # Match Steps
    matches = list(re.finditer(r'(Step \d+):', response_before_final_answer))
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(response_before_final_answer)
        segment = response[start:end].strip()
        steps[match.group(1)] = segment

    return_response = response_before_final_answer
    return final_answer, steps, return_response


def find_subsequence_position(sub_sequence, long_sequence):
    len_long = long_sequence.size(0)
    len_sub = len(sub_sequence) 

    sub_sequence_tensor = torch.tensor(sub_sequence, device=long_sequence.device)
    
    for i in range(len_long - len_sub + 1):
        if torch.equal(long_sequence[i:i + len_sub], sub_sequence_tensor):
            return i 
    return -1  
def is_word_in_sentence(sentence, word):
    pattern = re.escape(word)
    match = re.search(pattern, sentence, re.IGNORECASE)
    return True if match else False

def clean_words(word):
  return word.replace(" ", "").replace(".", "").replace("\"", "").replace("\n", "").replace("_", "").replace("Ġ", "").lower()

def find_token_indices(tokens, word):
    word_len = len(word.replace(" ", ""))
    
    for start_index in range(len(tokens)):
        combined_text = ""
        end_index = start_index       
        while end_index < len(tokens) and len(combined_text) < word_len:
            combined_text += tokens[end_index]
            if clean_words(combined_text) == clean_words(word):
                return start_index, end_index
            end_index += 1
    
    return -1, -1 

def extract_p(keyword_token_probability, contribution_scores = None ,use_min = True):
    if contribution_scores == None:
        return_dict = {}
        for step, inner_dict in keyword_token_probability.items():
            for key, values in inner_dict.items():
                if len(values) == 0:
                    continue
                # if key.isdigit(): 
                #     value_to_add = values[0] 
                # else:
                #     value_to_add = values[0] 
                # value_to_add = sum(values)/len(values)
                if use_min:
                    value_to_add = min(values)
                else:
                    value_to_add = sum(values)/len(values)
                # value_to_add = max(values)
                if key in return_dict:
                    return_dict[key].append(value_to_add)
                else:
                    return_dict[key] = [value_to_add]
        return return_dict
    else:
        return_keyword_dict = {}
        return_contribution_dict = {}
        for step, inner_dict in keyword_token_probability.items():
            for key, values in inner_dict.items():
                if len(values) == 0:
                    continue
                # if key.isdigit(): 
                #     value_to_add = values[-1] 
                # else:
                #     value_to_add = values[0] 
                # value_to_add = sum(values)/len(values)
                if use_min:
                    value_to_add = min(values)
                else:
                    value_to_add = sum(values)/len(values)
                # value_to_add = max(values)
                if key in return_keyword_dict:
                    return_keyword_dict[key].append(value_to_add)
                    return_contribution_dict[key].append(contribution_scores[step][key])
                else:
                    return_keyword_dict[key] = [value_to_add]
                    return_contribution_dict[key] = [contribution_scores[step][key]]
        return return_keyword_dict, return_contribution_dict
    
def weighted_sum(values):
    if len(values) == 1:
        return values[0] 
    weights = [math.exp(-c) for c in values]  
    sum_weights = sum(weights)  
    normalized_weights = [w / sum_weights for w in weights] 
    result = sum(w * c for w, c in zip(normalized_weights, values)) 
    return result

def extract_p_t_importance(question, keyword_token_probability, tokenizer, measure_model, contribution_scores = None):
    if contribution_scores == None:
        return_dict = {}
        for step, inner_dict in keyword_token_probability.items():
            for key, values in inner_dict.items():
                if len(values) == 0:
                    continue
                token_importance = get_tokenwise_importance(question, key, tokenizer, measure_model).data.cpu().numpy()
                if len(token_importance) == len(values):
                    weighted_score = ((token_importance / sum(token_importance)) * values)
                    value_to_add = sum(weighted_score)
                elif len(token_importance) - len(values) > 0:
                    start = len(token_importance) - len(values)
                    # end = len(values) - len(token_importance)
                    weighted_score = ((token_importance[start:] / sum(token_importance[start:])) * values)
                    value_to_add = sum(weighted_score)
                elif len(token_importance) - len(values) < 0:
                    start = len(values) - len(token_importance)
                    end = len(token_importance) - len(values)
                    weighted_score = ((token_importance[:] / sum(token_importance[:])) * values[start:])
                    value_to_add = sum(weighted_score)
                else:
                    value_to_add = sum(values) / len(values)
                if key in return_dict:
                    return_dict[key].append(value_to_add)
                else:
                    return_dict[key] = [value_to_add]
        return return_dict
    else:
        return_keyword_dict = {}
        return_contribution_dict = {}
        for step, inner_dict in keyword_token_probability.items():
            for key, values in inner_dict.items():
                if len(values) == 0:
                    continue
                token_importance = get_tokenwise_importance(question, key, tokenizer, measure_model).data.cpu().numpy()
                if len(token_importance) == len(values):
                    weighted_score = ((token_importance / sum(token_importance)) * values)
                    value_to_add = sum(weighted_score)
                elif len(token_importance) - len(values) > 0:
                    start = len(token_importance) - len(values)
                    end = len(values) - len(token_importance)
                    weighted_score = ((token_importance[start:] / sum(token_importance[start:])) * values)
                    value_to_add = sum(weighted_score)
                elif len(token_importance) - len(values) < 0:
                    start = len(values) - len(token_importance)
                    end = len(token_importance) - len(values)
                    weighted_score = ((token_importance[:] / sum(token_importance[:])) * values[start:])
                    value_to_add = sum(weighted_score)
                else:
                    value_to_add = sum(values) / len(values)
                if key in return_keyword_dict:
                    return_keyword_dict[key].append(value_to_add)
                    return_contribution_dict[key].append(contribution_scores[step][key])
                else:
                    return_keyword_dict[key] = [value_to_add]
                    return_contribution_dict[key] = [contribution_scores[step][key]]
        return return_keyword_dict, return_contribution_dict

def get_tokenwise_importance(question, generated_text, tokenizer, measure_model):
    token_importance_list = []
    tokenized = torch.tensor(tokenizer.encode(generated_text, add_special_tokens=False))

    token_importance = []
    # measure cosine similarity by removing each token and compare the similarity
    for token in tokenized:
        similarity_to_original = measure_model.predict([question + generated_text,
                                                        question + generated_text.replace(
                                                            tokenizer.decode(token, skip_special_tokens=True),
                                                            '')])
        token_importance.append(1 - torch.tensor(similarity_to_original))

    token_importance = torch.tensor(token_importance).reshape(-1)
    return token_importance
