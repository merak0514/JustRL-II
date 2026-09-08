"""
Code extraction and format validation.

Extracts code blocks and answers from model output, including TIR-mode extraction.
"""
import re
from typing import List

CODE_PATTERN = re.compile(r"```(?:\w+)?\n(.*?)\n```", re.DOTALL)


def validate_response_structure(processed_str: str) -> bool:
    return "</think>" in processed_str


def try_extract_solution(solution_str: str) -> str:
    answer_pattern = r"<answer>(.*?)</answer>"
    matches = list(re.finditer(answer_pattern, solution_str, re.DOTALL))

    if matches:
        final_answer = matches[-1].group(1).strip()
        return final_answer
    elif "</think>" in solution_str:
        return solution_str.split("</think>")[1].strip()

    return solution_str


def extract_code_from_string(solution_str: str) -> str:
    solution_str = try_extract_solution(solution_str)
    code_blocks = CODE_PATTERN.findall(solution_str)
    return "\n".join(code_blocks).strip()


def tir_code_extract(solution_str: str) -> str:
    """
    TIR-mode code extraction: pull the last python code block out of the LLM output.
    The last match wins, since the LLM's final answer usually comes last.
    """
    patterns = [
        r"```(?:python|py|python3)\s*\n([\s\S]*?)```",
        r"```(?:python|py|python3)\s*([\s\S]*?)```",
        r"```([\s\S]*?)```",
    ]

    code_blocks: List[str] = []
    for pat in patterns:
        matches = re.findall(pat, solution_str, flags=re.IGNORECASE)
        if matches:
            code_blocks = matches
            break

    if not code_blocks:
        print(
            "No ``` ``` code block found. Check that the LLM output marks code with ```python ... ```.",
            solution_str[len(solution_str) - 300:],
        )
        return ""

    code = code_blocks[-1].rstrip()
    return code
