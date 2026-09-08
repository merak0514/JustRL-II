import json
import os
import time
import traceback
from concurrent.futures import as_completed

import numpy as np

from .utils import _ERROR_MSG_PREFIX, coder1_debug_log, semantic_compare_output
from .solution_io import (
    build_solution_call_wrapper,
    append_stdio_mismatch_error_log,
    solution_mode_is_succ,
)
from .executor import get_global_executor, BatchContext, set_current_batch_ctx
from .extraction import (
    validate_response_structure,
    try_extract_solution,
    extract_code_from_string,
    tir_code_extract,
)

_MAX_CHAR_DISPLAY = 200
os.environ.setdefault("CODER1_DEBUG", "0")
os.environ.setdefault("CODER1_DEBUG_LOG", "/tmp/coder1.log")

CODER1_EXEC = os.environ.get("CODER1_EXEC", "firejail")
if CODER1_EXEC == "firejail":
    from .firejail_exec import code_exec_firejail

    code_exec = code_exec_firejail
else:
    raise ValueError(f"Unknown CODER1_EXEC: {CODER1_EXEC}")

coder1_debug_log(
    f"[bind] CODER1_EXEC={CODER1_EXEC} code_exec={getattr(code_exec, '__module__', '?')}.{getattr(code_exec, '__name__', '?')}"
)


def remote_check_stdio(code, stdin, stdout, batch_ctx):
    if batch_ctx.cancelled:
        return False, "Cancelled", stdin, stdout
    set_current_batch_ctx(batch_ctx)
    try:
        coder1_debug_log(
            f"[call] code_exec={getattr(code_exec, '__module__', '?')}.{getattr(code_exec, '__name__', '?')} stdin_len={0 if stdin is None else len(str(stdin))} code_len={len(code)}"
        )
        succ, output = code_exec(code=code, stdin=stdin)
        coder1_debug_log(
            f"[ret] succ={succ} out_len={0 if output is None else len(str(output))} out_head={repr(str(output)[:200])}"
        )
        return succ, output, stdin, stdout
    except Exception as e:
        tb = traceback.format_exc()
        exc_info = f"{type(e).__name__}: {str(e)}"
        coder1_debug_log(f"[exc] {exc_info}\n{tb}")
        return False, f"[Exception] {exc_info}\n{tb}", stdin, stdout
    finally:
        set_current_batch_ctx(None)


def _compute_score(
    solution_str,
    ground_truth,
    extra_info,
    format_reward=0.1,
    answer_reward=1.0,
    debug=False,
    fail_fast=True,
):
    reward_log = []
    case_log = []

    use_tir = os.environ.get("MINICPM5_TIR", "") == "1"
    if use_tir:
        pass_fmt = True
        solution_code = tir_code_extract(solution_str)
    else:
        pass_fmt = validate_response_structure(solution_str)
        solution_code = extract_code_from_string(solution_str)

    if not pass_fmt or len(solution_code) == 0:
        reward_log.append("\nBad format detected!")
        reward_log.append("Original Model Output:")
        reward_log.append("-" * 32)
        reward_log.append(solution_str[:200])
        reward_log.append("-" * 32)
        return 0.0, "\n".join(reward_log), ["❌ Fail - Extract Code Failed"]

    if isinstance(ground_truth, str):
        ground_truth = json.loads(ground_truth)
    elif isinstance(ground_truth, dict):
        pass
    else:
        raise ValueError(f"Invalid ground_truth: {ground_truth}")

    t_start = time.time()

    if "functional" in ground_truth:
        reward_log.append(solution_code + "\n" + ground_truth["functional"][:200])
    else:
        reward_log.append(solution_code[:200])

    if "pytest" in ground_truth or "functional" in ground_truth:
        if "functional" in ground_truth:
            print("functional pattern are called")
            try:
                succ, output = code_exec(
                    solution_code + "\n" + ground_truth["functional"]
                )
            except Exception as e:
                succ = False
                output = traceback.format_exc()
        else:  # pytest
            print("pytest pattern are called")
            try:
                pytest_code = ground_truth["pytest"]
                if not pytest_code.startswith("import pytest\n"):
                    pytest_code = "import pytest\n" + pytest_code
                succ, output = code_exec(solution_code, pytest=pytest_code)
            except Exception as e:
                succ = False
                output = traceback.format_exc()
        if not succ:
            reward_log.append(
                "!" * 16
                + f"⚠️ Test Execution Failed in {time.time() - t_start:.1f}s"
                + "!" * 16
            )
            reward_log.append(output[:_MAX_CHAR_DISPLAY])
            reward_log.append("-" * 16 + "Failed Prompt" + "-" * 16)
            return 0.0, "\n".join(reward_log), "test execution failed"

    elif "inputs" in ground_truth and "outputs" in ground_truth:
        stdin_list, stdout_list = ground_truth["inputs"], ground_truth["outputs"]

        is_solution_mode = "class Solution" in solution_code
        exec_code = solution_code
        if is_solution_mode:
            fn_name = (
                json.loads(extra_info.get("metadata", "{}")).get("func_name")
                if isinstance(extra_info, dict)
                else None
            )
            if not fn_name:
                print("fn_name not found")
                reward_log.append("found class Solution but extra_info[fn_name] is missing")
                return 0.0, "\n".join(reward_log), "fn_name not found"
            exec_code = build_solution_call_wrapper(solution_code, fn_name)

        executor = get_global_executor()
        batch_ctx = BatchContext()

        futures = [
            executor.submit(remote_check_stdio, exec_code, stdin, stdout, batch_ctx)
            for stdin, stdout in zip(stdin_list, stdout_list)
        ]

        try:
            has_failure = False
            for future in as_completed(futures):
                succ, output, stdin, stdout = future.result()
                output = output.replace("\\r", "")

                if is_solution_mode:
                    fail, extracted_output = solution_mode_is_succ(
                        succ=succ,
                        output=output,
                        expected_stdout=stdout,
                        reward_log=reward_log,
                        max_char_display=_MAX_CHAR_DISPLAY,
                    )
                else:
                    extracted_output = output.strip()
                    if not succ:
                        fail = True
                    else:
                        fail = not semantic_compare_output(
                            extracted_output, str(stdout)
                        )

                if debug:
                    status = "❌ Fail" if fail else "✅ Pass"
                    inputs_display = repr(str(stdin)[:80])
                    outputs_display = repr(extracted_output[:500]) if fail else repr(extracted_output[:100])
                    expected_display = repr(str(stdout).strip()[:100])
                    case_log.append(
                        f"{status} - inputs: {inputs_display}, "
                        f"outputs: {outputs_display}, "
                        f"expected: {expected_display}"
                    )

                if fail:
                    has_failure = True
                    append_stdio_mismatch_error_log(
                        reward_log=reward_log,
                        succ=succ,
                        output=output,
                        extracted_output=extracted_output,
                        expected_stdout=stdout,
                        max_char_display=_MAX_CHAR_DISPLAY,
                        error_msg_prefix=_ERROR_MSG_PREFIX,
                        extra_info=extra_info,
                    )
                    if fail_fast:
                        batch_ctx.cancel_all()
                        for f in futures:
                            if not f.done():
                                f.cancel()
                        return 0.0, "\n".join(reward_log), (case_log if debug else "")

            final_score = 0.0 if has_failure else 1.0
            return final_score, "\n".join(reward_log), (case_log if debug else "")
        except Exception as e:
            batch_ctx.cancel_all()
            for f in futures:
                if not f.done():
                    f.cancel()
            coder1_debug_log(f"[error] Exception in test execution: {e}")
            return 0.0, "\n".join(reward_log) + f"\n[Exception] {e}", (case_log if debug else "")
    else:
        raise ValueError(
            f"Current supports for ground-truth are ['functional', 'inputs/outputs'] -- No idea what's: {ground_truth = }"
        )

    reward_log.append("+" * 16 + "Test Execution Passed! (Output)" + "+" * 16)
    reward_log.append(output[:200])
    return 1.0, "\n".join(reward_log), ""


def _compute_score_eval(solution_str, ground_truth, extra_info=None, fail_fast=True):
    """
    Run the code solution against the test cases and return detailed results.
    Returns: dict with result (1/0), detail list, log list
    """
    if extra_info is None:
        extra_info = {}

    score, _, case_log = _compute_score(
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        debug=True,
        fail_fast=fail_fast,
    )

    all_passed = 1 if score == 1.0 else 0
    total_cases = len(ground_truth.get("inputs", []))

    result = {
        "result": all_passed,
        "detail": [],
        "log": [],
    }

    if isinstance(case_log, list):
        for log_entry in case_log:
            is_passed = 1 if log_entry.startswith("✅") else 0
            result["detail"].append(is_passed)
            result["log"].append(log_entry)
    else:
        result["detail"] = [0] * total_cases
        result["log"] = [str(case_log)] if case_log else ["execution failed"]

    return result


def compute_score(
    solution_str,
    ground_truth,
    extra_info,
    format_reward=0.1,
    answer_reward=1.0,
    debug=False,
    fail_fast=True,
):
    try:
        if isinstance(extra_info, np.ndarray):
            extra_info = extra_info.item()
        score, reward_log, case_log = _compute_score(
            solution_str,
            ground_truth,
            extra_info=extra_info,
            format_reward=format_reward,
            answer_reward=answer_reward,
            debug=debug,
            fail_fast=fail_fast,
        )
        marker = "✅" if score == 1.0 else "❌"
        if debug:
            print(f"=" * 60)
            reward_log = (
                "[Reward Summary] "
                + marker * 1
                + "\n[Reward Log]:\n"
                + reward_log
                + "\n\n"
            )
            print(reward_log + "\n")
            print(f"=" * 60)
        else:
            reward_log = f"[Reward Summary] {marker} /// [Final Reward] = {score}"
            print(reward_log)
        return score, reward_log
    except Exception as e:
        return 0, f"Error: {traceback.format_exc()}"


def compute_score_debug(
    solution_str,
    ground_truth,
    extra_info,
    format_reward=0.1,
    answer_reward=1.0,
    debug=False,
    fail_fast=True,
):
    try:
        if isinstance(extra_info, np.ndarray):
            extra_info = extra_info.item()
        score, reward_log, case_log = _compute_score(
            solution_str,
            ground_truth,
            extra_info=extra_info,
            format_reward=format_reward,
            answer_reward=answer_reward,
            debug=debug,
            fail_fast=fail_fast,
        )
        return score, case_log
    except Exception as e:
        return 0, f"Error: {traceback.format_exc()}"


def get_coder1_reward(response, label, extra_info=None, format_reward=0.0, answer_reward=1.0):
    """
    Code verifier entry point for the miles RL framework.
    Defaults to format_reward=0.0, answer_reward=1.0, matching how retool.py calls it.
    Set MINICPM5_TIR=1 to use TIR-mode code extraction.
    """
    os.environ["MINICPM5_TIR"] = "1"
    try:
        score, reward_log = compute_score(
            response,
            label,
            extra_info=extra_info,
            format_reward=format_reward,
            answer_reward=answer_reward,
            debug=False,
        )
        return score
    except Exception as e:
        return 0
