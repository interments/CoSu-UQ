import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from datasets import Dataset, load_dataset


COMMON_REQUEST = (
    'Please reason the following question step by step. Label each reasoning step as "Step i:", '
    'where "i" is the step number. You need to ensure that each step builds on the previous one '
    'and contributes meaningfully toward reaching the final answer. Once you finish all steps, put '
    'your final answer on a separate line after the reasoning steps, starting with "Final Answer:" '
    '(do not label it as a step).\n\n'
)

COMMON_EXAMPLE_QUESTION = (
    "Emily picked 36 apples. She gave 12 apples to her friend and then picked 9 more apples. "
    "How many apples does Emily have now?"
)
COMMON_EXAMPLE_ASSISTANT = (
    "Step 1: Emily started with 36 apples.\n"
    "Step 2: She gave away 12 apples, so we subtract 12 from 36: 36 - 12 = 24.\n"
    "Step 3: Then she picked 9 more apples, so we add 9 to the remaining 24: 24 + 9 = 33.\n"
    "Final Answer: 33"
)

HOTPOT_EXAMPLE_USER = (
    "Question: Which band has more members, We Are the Ocean or The Dream Academy?\n"
    "Response: Let's think step by step.\n"
)
HOTPOT_EXAMPLE_ASSISTANT = (
    "Step 1: We Are the Ocean has 5 members.\n"
    "Step 2: The Dream Academy has 3 members.\n"
    "Step 3: 5 is greater than 3.\n"
    "Step 4: Therefore, We Are the Ocean has more members.\n"
    "Final Answer: We Are the Ocean"
)

MATH_EXAMPLE_QUESTION = (
    "A right triangle with integer leg lengths is called \"cool\" if the number of square units "
    "in its area is equal to twice the number of units in the sum of the lengths of its legs. What is the "
    "sum of all the different possible areas of cool right triangles?"
)
MATH_EXAMPLE_USER = (
    f"Question: {MATH_EXAMPLE_QUESTION}\n\n"
    "Response: Let's think step by step.\n"
)
MATH_EXAMPLE_ASSISTANT = (
    "Step 1: Let legs be $a$ and $b$. The \"cool\" condition: $\\frac{1}{2}ab = 2(a + b)$.\n"
    "Step 2: Simplify to $ab - 4a - 4b = 0$.\n"
    "Step 3: Factor: $(a - 4)(b - 4) = 16$.\n"
    "Step 4: Integer factor pairs of 16: $(1,16), (2,8), (4,4), (8,2), (16,1)$.\n"
    "Step 5: Corresponding $(a,b)$: $(5,20), (6,12), (8,8), (12,6), (20,5)$.\n"
    "Step 6: Areas (removing duplicates): $50, 36, 32$.\n"
    "Step 7: Sum: $32 + 36 + 50 = 118$.\n"
    "Final Answer: $118$"
)

Row = Dict[str, Any]


def _build_common_input(question: str) -> str:
    return (
        COMMON_REQUEST
        + "Example:\n"
        + f"Question: {COMMON_EXAMPLE_QUESTION}\n\n"
        + "Response: Let's think step by step.\n"
        + COMMON_EXAMPLE_ASSISTANT
        + " \n\n\n\n"
        + f"Question: {question}\n\n"
        + "Response: Let's think step by step.\n"
    )


def _iter_head(ds: Dataset, num_samples: int):
    return ds.select(range(min(num_samples, len(ds))))


def _dump_jsonl(path: Path, rows: Iterable[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _build_row(
    *,
    index: int,
    question: str,
    request: str,
    one_shot_user: str,
    one_shot_assistant: str,
    input_text: str,
    outputs: List[str],
    extra_fields: Optional[Dict[str, Any]] = None,
) -> Row:
    row: Row = {
        "index": index,
        "question": question,
        "request": request,
        "one_shot_user": one_shot_user,
        "one_shot_assistant": one_shot_assistant,
        "your_turn": f"Question: {question}\n\nResponse: Let's think step by step.\n",
        "input": input_text,
        "outputs": outputs,
        "length": len(question),
    }
    if extra_fields:
        row.update(extra_fields)
    return row


def _parse_gsm8k_answer(answer_text: str) -> str:
    text = answer_text
    if "####" in text:
        text = text.split("####")[-1].strip()
    text = text.replace(",", "").strip()
    try:
        return str(float(text))
    except ValueError:
        return text


def build_gsm8k(num_samples: int):
    ds = load_dataset("gsm8k", "main", split="test")
    rows = []
    for idx, item in enumerate(_iter_head(ds, num_samples)):
        question = str(item["question"]).strip()
        rows.append(
            _build_row(
                index=idx,
                question=question,
                request=COMMON_REQUEST,
                one_shot_user=(
                    f"Question: {COMMON_EXAMPLE_QUESTION}\n\n"
                    "Response: Let's think step by step.\n"
                ),
                one_shot_assistant=COMMON_EXAMPLE_ASSISTANT,
                input_text=_build_common_input(question),
                outputs=[_parse_gsm8k_answer(str(item["answer"]))],
            )
        )
    return rows


def build_hotpotqa(num_samples: int):
    ds = load_dataset("hotpotqa", "distractor", split="validation")
    rows = []
    for idx, item in enumerate(_iter_head(ds, num_samples)):
        question = str(item["question"]).strip()
        rows.append(
            _build_row(
                index=idx,
                question=question,
                request=COMMON_REQUEST,
                one_shot_user=HOTPOT_EXAMPLE_USER,
                one_shot_assistant=HOTPOT_EXAMPLE_ASSISTANT,
                input_text=_build_common_input(question),
                outputs=[str(item["answer"])],
            )
        )
    return rows


def _get_first_available(d: Dict[str, Any], keys: Iterable[str]) -> Optional[Any]:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        if not value:
            return ""
        first = value[0]
        if isinstance(first, dict):
            for k in ("text", "answer", "value"):
                if k in first:
                    return str(first[k]).strip()
        return str(first).strip()
    if isinstance(value, dict):
        for k in ("text", "answer", "value"):
            if k in value:
                return str(value[k]).strip()
    return str(value).strip()


def _load_2wiki_hf(repo: str, split: str) -> Dataset:
    ds = load_dataset(repo, split=split)
    if not isinstance(ds, Dataset):
        raise ValueError(f"Expected a Dataset split, got: {type(ds)}")
    return ds


def build_2wiki(num_samples: int, repo: str, split: str):
    ds = _load_2wiki_hf(repo=repo, split=split)
    rows = []
    for item in _iter_head(ds, num_samples):
        question_raw = _get_first_available(item, ["question", "query"])
        answer_raw = _get_first_available(item, ["answer", "answers", "final_answer"])
        question = _to_text(question_raw)
        answer = _to_text(answer_raw)
        if not question:
            continue

        rows.append(
            _build_row(
                index=len(rows),
                question=question,
                request=COMMON_REQUEST,
                one_shot_user=HOTPOT_EXAMPLE_USER,
                one_shot_assistant=HOTPOT_EXAMPLE_ASSISTANT,
                input_text=_build_common_input(question),
                outputs=[answer],
            )
        )
    return rows


def _extract_boxed_answer(solution_text: str) -> str:
    matches = re.findall(r"\\\\boxed\{([^}]+)\}", solution_text)
    if matches:
        return matches[-1].strip()
    lines = solution_text.strip().split("\n")
    for line in reversed(lines):
        if line.strip() and not line.strip().startswith("##"):
            return line.strip()
    return ""


def _is_numerical_answer(answer_text: str) -> bool:
    if not answer_text:
        return False
    text = answer_text.strip()
    bad_patterns = [
        r"[a-zA-Z](?!sqrt|frac|pi|infty|cdot|times|div)",
        r"\\text",
        r"\\\{[^}]*,[^}]*\\\}",
        r"\[[^\]]*,[^\]]*\]",
        r"\([^)]*,[^)]*\)",
        r"or|and|if|then",
        r"=.*[a-zA-Z]",
    ]
    for pattern in bad_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return False

    cleaned = re.sub(r"\\[a-zA-Z]+", "", text)
    cleaned = re.sub(r"[{}\s]", "", cleaned)
    if cleaned and re.match(r"^[-+]?[\d.,/^()*eE\-+]+$", cleaned):
        return True

    if any(x in text for x in [r"\frac", r"\sqrt", r"\pi"]):
        temp = text
        for cmd in [r"\frac", r"\sqrt", r"\pi", r"\infty", r"\cdot", r"\times"]:
            temp = temp.replace(cmd, "")
        temp = re.sub(r"\d", "", temp)
        temp = re.sub(r"[-+*/^().,{}%\s]", "", temp)
        return re.search(r"[a-zA-Z]", temp) is None

    return False


def build_math(num_samples: int):
    ds = load_dataset("hendrycks/competition_math", split="train")
    rows = []
    for original_index, item in enumerate(ds):
        if len(rows) >= num_samples:
            break

        question = str(item["problem"]).strip()
        solution = str(item["solution"]) if item.get("solution") is not None else ""
        answer = _extract_boxed_answer(solution)
        if not _is_numerical_answer(answer):
            continue

        new_index = len(rows)
        rows.append(
            _build_row(
                index=new_index,
                question=question,
                request=COMMON_REQUEST,
                one_shot_user=MATH_EXAMPLE_USER,
                one_shot_assistant=MATH_EXAMPLE_ASSISTANT,
                input_text=(
                    COMMON_REQUEST
                    + "Example:\n"
                    + f"{MATH_EXAMPLE_USER}"
                    + f"{MATH_EXAMPLE_ASSISTANT} \n\n\n\n"
                    + f"Question: {question}\n\n"
                    + "Response: Let's think step by step.\n"
                ),
                outputs=[answer],
                extra_fields={
                    "original_index": original_index,
                    "type": str(item.get("type", "")),
                    "level": str(item.get("level", "")),
                },
            )
        )
    return rows


def _format_options(options):
    return "\n".join([f"{opt['key']}: {opt['value']}" for opt in options])


def build_medqa(num_samples: int):
    ds = load_dataset("bigbio/med_qa", trust_remote_code=True)
    train_split = ds["train"]
    test_shot = ds["test"][1]

    request_text = (
        'Please reason the following question step by step. Label each reasoning step as "Step i:", where "i" is '
        'the step number.\nYou need to ensure that each step builds on the previous one and contributes meaningfully '
        'toward reaching the final answer.\nOnce you finish all steps, put your final answer on a separate line after '
        'the reasoning steps, starting with "Final Answer:" (do not label it as a step).\nExtremely Important: The '
        'Final Answer must follow the format "Final Answer: [Option Letter]（[Option Text]）". For example, '
        '"Final Answer: C（Option value text ）".\n\n'
    )

    shot_question = str(test_shot["question"]).strip()
    shot_options = _format_options(test_shot["options"])
    shot_answer_idx = str(test_shot["answer_idx"]).strip()

    shot_answer_text = ""
    for opt in test_shot["options"]:
        if str(opt["key"]) == shot_answer_idx:
            shot_answer_text = str(opt["value"])
            break
    shot_answer = f"{shot_answer_idx}（{shot_answer_text}）"

    rows = []
    for idx, item in enumerate(_iter_head(train_split, num_samples)):
        question = str(item["question"]).strip()
        option_text = _format_options(item["options"])
        answer_idx = str(item["answer_idx"]).strip()

        one_shot_user = (
            f"Question: {shot_question}\nOptions:\n{shot_options}\n"
            "Response: Let's think step by step.\n"
        )
        one_shot_assistant = (
            "Step 1: Identify the clinical pattern and key findings in the stem.\n"
            "Step 2: Match each option with mechanism/indication and eliminate inconsistent options.\n"
            "Step 3: Select the option that best explains all findings.\n"
            f"Final Answer: {shot_answer}"
        )

        rows.append(
            _build_row(
                index=idx,
                question=question,
                request=request_text,
                one_shot_user=one_shot_user,
                one_shot_assistant=one_shot_assistant,
                input_text=(
                    request_text
                    + f"Example:\nQuestion: {shot_question}\nOptions:\n{shot_options}\n"
                    + "Response: Let's think step by step.\n"
                    + "Step 1: Identify the clinical pattern and key findings in the stem.\n"
                    + "Step 2: Match each option with mechanism/indication and eliminate inconsistent options.\n"
                    + "Step 3: Select the option that best explains all findings.\n"
                    + f"Final Answer: {shot_answer}\n\n"
                    + f"Question: {question}\nOptions:\n{option_text}\n"
                    + "Response: Let's think step by step.\n"
                ),
                outputs=[answer_idx],
                extra_fields={"original_index": idx},
            )
        )

    return rows


def main():
    parser = argparse.ArgumentParser(description="Build unified CoT-UQ cleaned datasets from HuggingFace")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["gsm8k", "hotpotqa", "math", "2wiki", "medqa"],
        choices=["gsm8k", "hotpotqa", "math", "2wiki", "medqa"],
        help="Datasets to generate",
    )
    parser.add_argument("--num_samples", type=int, default=500, help="Number of samples per dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./datasets",
        help="Output directory for cleaned jsonl files",
    )
    parser.add_argument(
        "--wiki2_hf_repo",
        type=str,
        default="scholarly-shadows-syndicate/2wikimultihopqa_with_qid",
        help="2WikiMultiHopQA HuggingFace dataset repo",
    )
    parser.add_argument(
        "--wiki2_hf_split",
        type=str,
        default="train",
        help="2WikiMultiHopQA split name",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_handlers = {
        "gsm8k": (
            lambda: build_gsm8k(args.num_samples),
            f"gsm8k_cot_uq_validation_0-{args.num_samples}_src.jsonl",
        ),
        "hotpotqa": (
            lambda: build_hotpotqa(args.num_samples),
            f"hotpotqa_cot_uq_validation_0-{args.num_samples}_src.jsonl",
        ),
        "math": (
            lambda: build_math(args.num_samples),
            f"math_cot_uq_validation_0-{args.num_samples}.jsonl",
        ),
        "2wiki": (
            lambda: build_2wiki(args.num_samples, repo=args.wiki2_hf_repo, split=args.wiki2_hf_split),
            f"2WikimultihopQA_cot_uq_validation_0-{args.num_samples}_src.jsonl",
        ),
        "medqa": (
            lambda: build_medqa(args.num_samples),
            f"MedQA_cot_uq_validation_0-{args.num_samples}_src.jsonl",
        ),
    }

    for dataset_name in args.datasets:
        build_fn, filename = dataset_handlers[dataset_name]
        rows = build_fn()
        _dump_jsonl(output_dir / filename, rows)
        print(f"[OK] {dataset_name}: {len(rows)} samples")


if __name__ == "__main__":
    main()
