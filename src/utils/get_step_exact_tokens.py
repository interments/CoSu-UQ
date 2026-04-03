instruction = ''' 
You will be provided with a question and a multi-step response containing reasoning steps. 
For each long reasoning step labeled "Step i:", extract the keywords, only the relevant tokens for that specific reasoning step.
You also need to evaluate the importance of each keyword to the final answer. Please evaluate the importance score following with the keyword by (/<importance score>/) on a scale of 1 to 10, where 1 is the least critical and 10 is the most critical.
If you find more than one keyword in a specific step, separate them with “;”.
If a specific step does not contribute meaningfully to deriving the final answer (e.g., repeating information already provided in the question, introducing irrelevant assumptions or speculations), return "Step i: NO ANSWER" for that step. For example:




Question:
<QUESTION>
Multi-Step Response:
<RESPONSE>
Keywords for Each Reasoning Step:

'''


instruction_hotpotqa_w_contribution_score = ''' 
You will be provided with a question and a multi-step response containing reasoning steps. 
For each long reasoning step labeled "Step i:", extract the keywords, only the relevant tokens for that specific reasoning step.
You also need to evaluate the importance of each keyword to the final answer. Please evaluate the importance score following with the keyword by (/<importance score>/) on a scale of 1 to 10, where 1 is the least critical and 10 is the most critical.
If you find more than one keyword in a specific step, separate them with “;”.
If a specific step does not contribute meaningfully to deriving the final answer (e.g., repeating information already provided in the question, introducing irrelevant assumptions or speculations), return "Step i: NO ANSWER" for that step. For example:
Q: Which band has more members, "We Are the Ocean" or "The Dream Academy"?
A: Step 1: The question is asking which band has more members.
Step 2: "We Are the Ocean" has 5 members.
Step 3: "The Dream Academy" has 3 members.
Step 4: 5 is greater than 3.
Step 5: Therefore, "We Are the Ocean" has more members.
Final Answer: We Are the Ocean
Keywords for Each Reasoning Step:
Step 1: NO ANSWER
Step 2: We Are the Ocean(/5/); 5(/10/)
Step 3: The Dream Academy(/5/); 3(/10/)
Step 4: greater(/7/)
Step 5: We Are the Ocean(/5/)




The following is your task:
Q: <QUESTION>
A: <RESPONSE>
Keywords for Each Reasoning Step:

'''

instruction_hotpotqa_w_contribution_score_standard = '''
You will be provided with a question and a multi-step response containing reasoning steps. 
For each reasoning step labeled "Step i:", extract the keywords — only the tokens that are relevant for that specific reasoning step. 
You also need to evaluate the importance of each keyword to the final answer. 
Please assign an importance score in the format (/score/), where the score ranges from 1 to 10:
- If the keyword has a larger impact on deriving the final result, give it a higher score.
- If the keyword carries higher semantic importance within the reasoning step, give it a higher score.

Scoring Guidelines (1–10 scale):
1–2: Very minor detail, barely affects reasoning or could be omitted without changing result.
3–4: Low importance, helps context but not directly critical.
5–6: Moderate importance, useful for reasoning but not the decisive factor.
7–8: High importance, strongly contributes to progressing toward the final answer.
9: Nearly essential, without it the reasoning step would be incomplete or incorrect.
10: Absolutely essential keyword, directly determines the final result.

If a specific step does not contribute meaningfully to deriving the final answer (e.g., repeats information already provided in the question, introduces irrelevant assumptions), return "Step i: NO ANSWER" for that step.For example:
Q: Which band has more members, "We Are the Ocean" or "The Dream Academy"?
A: Step 1: The question is asking which band has more members.
Step 2: "We Are the Ocean" has 5 members.
Step 3: "The Dream Academy" has 3 members.
Step 4: 5 is greater than 3.
Step 5: Therefore, "We Are the Ocean" has more members.
Final Answer: We Are the Ocean
Keywords for Each Reasoning Step:
Step 1: NO ANSWER
Step 2: We Are the Ocean(/5/); 5(/10/)
Step 3: The Dream Academy(/5/); 3(/10/)
Step 4: greater(/7/)
Step 5: We Are the Ocean(/5/)




The following is your task:
Q: <QUESTION>
A: <RESPONSE>
Keywords for Each Reasoning Step:

'''

instruction_hotpotqa_w_contribution_score_0_100 = '''
You will be provided with a question and a multi-step response containing reasoning steps. 
For each long reasoning step labeled "Step i:", extract the keywords, only the relevant tokens for that specific reasoning step.
You also need to evaluate the importance of each keyword to the final answer. Please evaluate the importance score following with the keyword by (/<importance score>/) on a scale of 1 to 100, where 1 is the least critical and 100 is the most critical.
If you find more than one keyword in a specific step, separate them with “;”.
If a specific step does not contribute meaningfully to deriving the final answer (e.g., repeating information already provided in the question, introducing irrelevant assumptions or speculations), return "Step i: NO ANSWER" for that step. For example:
Q: Which band has more members, "We Are the Ocean" or "The Dream Academy"?
A: Step 1: The question is asking which band has more members.
Step 2: "We Are the Ocean" has 5 members.
Step 3: "The Dream Academy" has 3 members.
Step 4: 5 is greater than 3.
Step 5: Therefore, "We Are the Ocean" has more members.
Final Answer: We Are the Ocean
Keywords for Each Reasoning Step:
Step 1: NO ANSWER
Step 2: We Are the Ocean(/54/); 5(/99/)
Step 3: The Dream Academy(/48/); 3(/99/)
Step 4: greater(/73/)
Step 5: We Are the Ocean(/59/)




The following is your task:
Q: <QUESTION>
A: <RESPONSE>
Keywords for Each Reasoning Step:

'''



##TODO:在前面给出一个打分标准，然后观察结果
instruction_math_w_contribution_score_hard = ''' 
You will be provided with a question and a multi-step response containing reasoning steps. 
For each long reasoning step labeled "Step i:", extract keywords (***ensure that the extracted keyword tokens are exactly the same as in the original text, including case***), only the relevant tokens for that specific reasoning step.
You also need to evaluate the importance of each keyword to the final answer. Please evaluate the importance score following with the keyword by (/<importance score>/) on a scale of 1 to 10, where 1 is the least critical and 10 is the most critical.
If you find more than one keyword in a specific step, separate them with “;”.
If a specific step does not contribute meaningfully to deriving the final answer (e.g., repeating information already provided in the question, introducing irrelevant assumptions or speculations), return "Step i: NO ANSWER" for that step. For example:
Q: A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?
A:Step 1: We consider two cases for $x$: $x\ge 0$ and $x<0$.
Step 2: If $x\ge 0$, then $x|x| = x^2$, so $x^2 = 2x+1$ which gives $x^2 - 2x - 1 = 0$.
Step 3: Solve the quadratic: $x = \frac{2 \pm \sqrt{4 + 4}}{2} = 1 \pm \sqrt{2}$.
Step 4: From $x\ge 0$ we take $x = 1 + \sqrt{2}$.
Step 5: If $x<0$, then $x|x| = -x^2$, so $-x^2 = 2x+1$ which gives $x^2 + 2x + 1 = 0$.
Step 6: Factor: $(x+1)^2 = 0$.
Step 7: So $x = -1$ for the negative case.
Step 8: The smallest solution is $-1$.
Final Answer: $-1$
Keywords for Each Reasoning Step:
Step 1: NO ANSWER
Step 2: $x^2 = 2x+1$ (/9/); $x^2 - 2x - 1 = 0$ (/9/)
Step 3: $1 \pm \sqrt{2}$ (/10/)
Step 4: $x = 1 + \sqrt{2}$ (/8/)
Step 5: $-x^2 = 2x+1$ (/9/); $x^2 + 2x + 1 = 0$ (/9/)
Step 6: $(x+1)^2 = 0$ (/8/)
Step 7: $x = -1$ (/10/)
Step 8: $-1$ (/10/)




The following is your task:
Q: <QUESTION>
A: <RESPONSE>
Keywords for Each Reasoning Step:

'''
instruction_math_w_contribution_score = ''' 
You will be provided with a question and a multi-step response containing reasoning steps. 
For each long reasoning step labeled "Step i:", extract keywords (***ensure that the extracted keyword tokens are exactly the same as in the original text, including case***), only the relevant tokens for that specific reasoning step.
You also need to evaluate the importance of each keyword to the final answer. Please evaluate the importance score following with the keyword by (/<importance score>/) on a scale of 1 to 10, where 1 is the least critical and 10 is the most critical.
If you find more than one keyword in a specific step, separate them with “;”.
If a specific step does not contribute meaningfully to deriving the final answer (e.g., repeating information already provided in the question, introducing irrelevant assumptions or speculations), return "Step i: NO ANSWER" for that step. For example:
Q: A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?
A: Step 1: Let me understand what the question is asking about the robe.
Step 2: Identify the amount of blue fiber needed. The robe requires 2 bolts of blue fiber.  
Step 3: Determine the amount of white fiber needed. It is half the amount of blue fiber, which is 2 ÷ 2 = 1 bolt.  
Step 4: Compute the total number of bolts. Add the bolts of blue fiber (2) and white fiber (1) to get 3 bolts.
Final Answer: 3
Keywords for Each Reasoning Step:
Step 1: NO ANSWER
Step 2: 2 bolts (/3/)
Step 3: half the amount of blue fiber (/7/); 1 bolt (/10/)
Step 4: 3 bolts (/7/)




The following is your task:
Q: <QUESTION>
A: <RESPONSE>
Keywords for Each Reasoning Step:

'''

instruction_math_w_contribution_score_standard = ''' 
You will be provided with a question and a multi-step response containing reasoning steps. 
For each reasoning step labeled "Step i:", extract the keywords — only the tokens that are relevant for that specific reasoning step. 
You also need to evaluate the importance of each keyword to the final answer. 
Please assign an importance score in the format (/score/), where the score ranges from 1 to 10:
- If the keyword has a larger impact on deriving the final result, give it a higher score.
- If the keyword carries higher semantic importance within the reasoning step, give it a higher score.

Scoring Guidelines (1–10 scale):
1–2: Very minor detail, barely affects reasoning or could be omitted without changing result.
3–4: Low importance, helps context but not directly critical.
5–6: Moderate importance, useful for reasoning but not the decisive factor.
7–8: High importance, strongly contributes to progressing toward the final answer.
9: Nearly essential, without it the reasoning step would be incomplete or incorrect.
10: Absolutely essential keyword, directly determines the final result.

If a specific step does not contribute meaningfully to deriving the final answer (e.g., repeats information already provided in the question, introduces irrelevant assumptions), return "Step i: NO ANSWER" for that step.For example:
Q: A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?
A: Step 1: Let me understand what the question is asking about the robe.
Step 2: Identify the amount of blue fiber needed. The robe requires 2 bolts of blue fiber.  
Step 3: Determine the amount of white fiber needed. It is half the amount of blue fiber, which is 2 ÷ 2 = 1 bolt.  
Step 4: Compute the total number of bolts. Add the bolts of blue fiber (2) and white fiber (1) to get 3 bolts.
Final Answer: 3
Keywords for Each Reasoning Step:
Step 1: NO ANSWER
Step 2: 2 bolts (/3/)
Step 3: half the amount of blue fiber (/7/); 1 bolt (/10/)
Step 4: 3 bolts (/7/)




The following is your task:
Q: <QUESTION>
A: <RESPONSE>
Keywords for Each Reasoning Step:

'''

instruction_medqa_w_contribution_score='''
You will be provided with a question and a multi-step response containing reasoning steps. 
For each long reasoning step labeled "Step i:", extract the keywords, only the relevant tokens for that specific reasoning step.
You also need to evaluate the importance of each keyword to the final answer. Please evaluate the importance score following with the keyword by (/<importance score>/) on a scale of 1 to 10, where 1 is the least critical and 10 is the most critical.
If you find more than one keyword in a specific step, separate them with “;”.
If a specific step does not contribute meaningfully to deriving the final answer (e.g., repeating information already provided in the question, introducing irrelevant assumptions or speculations), return "Step i: NO ANSWER" for that step. For example:
Q: A 45-year-old man presents with a 2-week history of cough and low-grade fever. He has a history of hypertension and takes lisinopril. On examination, his temperature is 37.8°C, blood pressure is 130/85 mmHg, pulse is 78/min, and oxygen saturation is 98% on room air. Chest auscultation is clear. A chest X-ray is unremarkable. Which of the following is the most likely diagnosis?
A: Step 1: Review the patient's symptoms of cough and low-grade fever.
Step 2: Note the absence of abnormal findings on chest auscultation and chest X-ray.
Step 3: Consider that the patient has no signs of pneumonia or other acute illness.
Step 4: Recognize that a viral upper respiratory infection is the most likely cause.
Final Answer: Viral upper respiratory infection
Keywords for Each Reasoning Step:
Step 1: NO ANSWER
Step 2: chest auscultation (/7/); chest X-ray (/8/); absence of abnormal findings (/9/)
Step 3: no signs of pneumonia (/9/); no acute illness (/8/)
Step 4: viral upper respiratory infection (/10/)



The following is your task:
Q: <QUESTION>
A: <RESPONSE>
Keywords for Each Reasoning Step:
'''

instruction_math_w_contribution_score_0_100 = ''' 
You will be provided with a question and a multi-step response containing reasoning steps. 
For each long reasoning step labeled "Step i:", extract the keywords, only the relevant tokens for that specific reasoning step.
You also need to evaluate the importance of each keyword to the final answer. Please evaluate the importance score following with the keyword by (/<importance score>/) on a scale of 1 to 100, where 1 is the least critical and 100 is the most critical.
If you find more than one keyword in a specific step, separate them with “;”.
If a specific step does not contribute meaningfully to deriving the final answer (e.g., repeating information already provided in the question, introducing irrelevant assumptions or speculations), return "Step i: NO ANSWER" for that step. For example:
Q: A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?
A: Step 1: Let me understand what the question is asking about the robe.
Step 2: Identify the amount of blue fiber needed. The robe requires 2 bolts of blue fiber.  
Step 3: Determine the amount of white fiber needed. It is half the amount of blue fiber, which is 2 ÷ 2 = 1 bolt.  
Step 4: Compute the total number of bolts. Add the bolts of blue fiber (2) and white fiber (1) to get 3 bolts.
Final Answer: 3
Keywords for Each Reasoning Step:
Step 1: NO ANSWER
Step 2: 2 bolts (/42/)
Step 3: half the amount of blue fiber (/78/); 1 bolt (/91/)
Step 4: 3 bolts (/79/)




The following is your task:
Q: <QUESTION>
A: <RESPONSE>
Keywords for Each Reasoning Step:

'''



instruction_2wiki_w_contribution_score = ''' 
You will be provided with a question and a multi-step response containing reasoning steps. 
For each long reasoning step labeled "Step i:", extract the keywords, only the relevant tokens for that specific reasoning step.
You also need to evaluate the importance of each keyword to the final answer. Please evaluate the importance score following with the keyword by (/<importance score>/) on a scale of 1 to 10, where 1 is the least critical and 10 is the most critical.
If you find more than one keyword in a specific step, separate them with “;”.
If a specific step does not contribute meaningfully to deriving the final answer (e.g., repeating information already provided in the question, introducing irrelevant assumptions or speculations), return "Step i: NO ANSWER" for that step. For example:
Q: Who is the paternal grandmother of Joseph Ferdinand Of Bavaria?
A: Step 1: Joseph Ferdinand's father was Maximilian II Emanuel, Elector of Bavaria.
Step 2: Maximilian II Emanuel was the son of Ferdinand Maria, Elector of Bavaria, and his wife, Henriette Adelaide of Savoy.
Step 3: As the mother of Maximilian II Emanuel, Henriette Adelaide of Savoy is the paternal grandmother of Joseph Ferdinand of Bavaria.
Final Answer: Henriette Adelaide of Savoy
Keywords for Each Reasoning Step:
Step 1: father (/8/); Maximilian II Emanuel, Elector of Bavaria (/8/)
Step 2: son (/8/); Ferdinand Maria, Elector of Bavaria (/5/); Henriette Adelaide of Savoy (/9/)
Step 3: mother (/10/)




The following is your task:
Q: <QUESTION>
A: <RESPONSE>
Keywords for Each Reasoning Step:

'''


def get_step_exact_tokens(dataset, question, cot_response, prompt_type = None):
    if dataset in ["gsm8k", "svamp", "ASDiv"]:
        if prompt_type == "0-100":
            prompt = instruction_math_w_contribution_score_0_100.replace('<QUESTION>', question).replace('<RESPONSE>', cot_response)
        elif prompt_type == "standard":
            prompt = instruction_math_w_contribution_score_standard.replace('<QUESTION>', question).replace('<RESPONSE>', cot_response)
        else:
            prompt = instruction_math_w_contribution_score.replace('<QUESTION>', question).replace('<RESPONSE>', cot_response)
    elif dataset in ["math"]:
        prompt = instruction_math_w_contribution_score_hard.replace('<QUESTION>', question).replace('<RESPONSE>', cot_response)
    elif dataset in ["hotpotQA"]:
        if prompt_type == "0-100":
            prompt = instruction_hotpotqa_w_contribution_score_0_100.replace('<QUESTION>', question).replace('<RESPONSE>', cot_response)
        elif prompt_type == "standard":
            prompt = instruction_hotpotqa_w_contribution_score_standard.replace('<QUESTION>', question).replace('<RESPONSE>', cot_response)
        else:
            prompt = instruction_hotpotqa_w_contribution_score.replace('<QUESTION>', question).replace('<RESPONSE>', cot_response)
    elif dataset == "medqa":
        prompt = instruction_medqa_w_contribution_score.replace('<QUESTION>', question).replace('<RESPONSE>', cot_response)
    elif dataset == "2WikimhQA":
        prompt = instruction_2wiki_w_contribution_score.replace('<QUESTION>', question).replace('<RESPONSE>', cot_response)
    else:
        prompt = instruction.replace('<QUESTION>', question).replace('<RESPONSE>', cot_response)
    return prompt