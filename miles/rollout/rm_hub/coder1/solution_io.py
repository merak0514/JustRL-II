import json
from typing import Any, Optional, Tuple, List

from .utils import semantic_compare_output

CALL_RESULT_PREFIX = "__CODER1_CALL_RESULT__="


def build_solution_call_wrapper(user_code: str, fn_name: str) -> str:
    """
    Build a Python program that code_exec can run:
    - reads stdin (line by line)
    - calls Solution().{fn_name}(*args)
    - serializes the return value into text that stays as close to stdio output as
      possible, prefixed with a fixed marker so the caller can extract it
    """
    tpl = r"""
# === User Code START ===
{user_code}
# === User Code END ===

_FN_NAME = {fn_name_json}
_PREFIX = {prefix_json}

def _parse_args_from_stdin():
    raw = sys.stdin.read()
    if not raw:
        return []
    lines = [ln for ln in raw.split("\n") if ln != ""]
    inputs = [json.loads(line) for line in lines]
    try:
        if inputs and isinstance(inputs[0], dict):
            inputs = [{{int(k): v for k, v in inputs[0].items()}}]
    except Exception:
        pass
    return inputs

def _to_stdio_text(x):
    if x is None:
        return ""
    if isinstance(x, bool):
        return "true" if x else "false"
    if isinstance(x, (int, float, str)):
        return str(x)
    if isinstance(x, tuple):
        x = list(x)
    if isinstance(x, list):
        if all(not isinstance(i, (list, tuple, dict)) for i in x):
            return " ".join(str(i) for i in x)
        lines = []
        for row in x:
            if isinstance(row, tuple):
                row = list(row)
            if isinstance(row, list):
                lines.append(" ".join(str(i) for i in row))
            else:
                lines.append(str(row))
        return "\\\\n".join(lines)
    try:
        return json.dumps(x, ensure_ascii=False)
    except Exception:
        return str(x)

if __name__ == "__main__":
    args = _parse_args_from_stdin()
    solution = Solution()
    fn = getattr(solution, _FN_NAME)
    out = fn(*args)
    sys.stdout.write(_PREFIX + str(out))
""".lstrip()
    return tpl.format(
        user_code=user_code,
        fn_name_json=json.dumps(fn_name),
        prefix_json=json.dumps(CALL_RESULT_PREFIX),
    )


def extract_call_result_stdout(stdout: Optional[str]) -> Tuple[bool, str, str]:
    if stdout is None:
        return False, "", "empty stdout"
    idx = stdout.rfind(CALL_RESULT_PREFIX)
    if idx < 0:
        return False, stdout, "missing result prefix"
    return True, stdout[idx + len(CALL_RESULT_PREFIX) :], ""


def solution_mode_is_succ(
    *,
    succ: bool,
    output: Optional[str],
    expected_stdout: Any,
    reward_log: List[str],
    max_char_display: int,
) -> Tuple[bool, str]:
    """
    Failure check for Solution (call-based) mode.
    Returns (fail(bool), extracted_text_stripped(str)).
    """
    if not succ:
        err_output = "" if output is None else str(output).strip()[:max_char_display]
        return True, err_output

    extract_done, extract_output, extract_err = extract_call_result_stdout(output)
    if not extract_done:
        reward_log.append("---ExtractError---")
        reward_log.append(f"ExtractError: {extract_err}")
        reward_log.append(f"RawStdout(head): {repr(str(output)[:max_char_display])}")
        err_output = "" if output is None else str(output).strip()[:max_char_display]
        return True, err_output

    extracted_text_stripped = extract_output.strip()
    fail = not semantic_compare_output(extracted_text_stripped, str(expected_stdout))
    return fail, extracted_text_stripped


def solution_mode_extract_text(output: Optional[str]) -> str:
    _, extracted_text, _ = extract_call_result_stdout(output)
    return extracted_text


def append_stdio_mismatch_error_log(
    *,
    reward_log: List[str],
    succ: bool,
    output: Optional[str],
    extracted_output: str,
    expected_stdout: Any,
    max_char_display: int,
    error_msg_prefix: str,
    extra_info: Optional[dict] = None,
) -> None:
    reward_log.append("---Error---")
    if extra_info:
        index = extra_info.get("index", "")
        if index:
            reward_log.append(f"DataSource: index={index}")
    expected_repr = repr(str(expected_stdout).strip())[:500]
    reward_log.append(f"Expected: <gt>{expected_repr}</gt>")

    if succ:
        actual_repr = repr(extracted_output)[:500]
        reward_log.append(f"Actual: <pred>{actual_repr}</pred>")
    else:
        out_s = "" if output is None else str(output)
        actual_repr = (output if out_s.startswith(error_msg_prefix) else repr(out_s))[:500]
        reward_log.append(f"Actual: <pred>{actual_repr}</pred>")
