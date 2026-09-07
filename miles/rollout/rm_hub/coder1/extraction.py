"""
代码提取与格式验证模块

从模型输出中提取代码块和答案，支持 TIR 模式的代码提取。
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
    TIR 模式下的代码提取：从 LLM 输出中提取最后一个 python 代码块。
    取最后一个匹配的代码块（LLM 最终答案通常在最后）。
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
            "未找到任何 ``` ``` 代码块。请确认 LLM 输出使用 ```python ... ``` 来标注代码。",
            solution_str[len(solution_str) - 300:],
        )
        return ""

    code = code_blocks[-1].rstrip()
    return code
