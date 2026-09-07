"""
Tool-Integrated Reasoning (TIR) generate function for MiniCPM-style models.

Each sample gets its own E2B sandbox that persists across turns, enabling
stateful code execution (variables, imports, etc. carry over between tool
calls within a single sample).

Routing
~~~~~~~
``rollout_mode`` is read from ``sample.metadata["rollout_mode"]`` with a
fallback to the ``--generate-default-rollout-mode`` CLI argument (default
``"multi_turn"``).

* ``"single_turn"`` — delegates to :func:`single_turn.generate`.
* ``"multi_turn"`` — multi-turn loop with E2B sandbox execution.

Dynamic ``max_new_tokens``
~~~~~~~~~~~~~~~~~~~~~~~~~~
Same behaviour as :mod:`unified_generate`: each turn's ``max_new_tokens``
fills the remaining context window when ``--generate-dynamic-max-tokens``
is enabled (the default).

Usage::

    --custom-generate-function-path miles.rollout.generate_hub.tir_generate.generate
    --generate-default-rollout-mode  multi_turn
    --generate-tool-call-parser      minicpm4_xml
"""

import argparse
import json
import logging
import time
import uuid
from copy import deepcopy
from typing import Any

import numpy as np

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_hub.single_turn import generate as single_turn_generate
from miles.rollout.generate_utils.e2b_sandbox import (
    _get_semaphore,
    create_sandbox,
    execute_tool_in_sandbox,
    kill_sandbox,
)
from miles.rollout.generate_utils.generate_endpoint_utils import (
    compute_prompt_ids_from_sample,
    compute_request_payload,
    update_sample_from_response,
)
from miles.rollout.generate_utils.tool_call_utils import (
    create_tool_call_parser,
    update_sample_with_tool_responses,
)
from miles.utils.http_utils import post
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

CODE_INTERPRETER_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "code_interpreter",
            "description": "A tool for executing python code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The python code to execute.",
                    }
                },
                "required": ["code"],
            },
        },
    }
]


# ---------------------------------------------------------------------------
# Tool system-message prefix (lazy-initialised, cached per tokenizer)
# ---------------------------------------------------------------------------

_tool_system_prefix_cache: dict[int, str] = {}


def _get_tool_system_prefix(tokenizer, tools: list[dict[str, Any]]) -> str:
    """Return the system-message prefix that ``apply_chat_template`` adds when
    *tools* are supplied.

    The result is cached per ``tokenizer`` identity so the (relatively
    expensive) double template-render only runs once per worker.
    """
    key = id(tokenizer)
    if key not in _tool_system_prefix_cache:
        dummy = [{"role": "user", "content": "X"}]
        with_tools = tokenizer.apply_chat_template(
            dummy, tools=tools, tokenize=False, add_generation_prompt=True,
        )
        without_tools = tokenizer.apply_chat_template(
            dummy, tools=None, tokenize=False, add_generation_prompt=True,
        )
        idx = with_tools.find(without_tools)
        if idx <= 0:
            logger.warning(
                "Could not derive tool system prefix from chat template; "
                "tool definitions will NOT be injected into the prompt."
            )
            _tool_system_prefix_cache[key] = ""
        else:
            _tool_system_prefix_cache[key] = with_tools[:idx]
            logger.info(
                "Cached tool system prefix (%d chars) for tokenizer %s",
                len(_tool_system_prefix_cache[key]),
                type(tokenizer).__name__,
            )
    return _tool_system_prefix_cache[key]


# ---------------------------------------------------------------------------
# Dynamic max-new-tokens (identical to unified_generate)
# ---------------------------------------------------------------------------

def _compute_dynamic_max_new_tokens(
    args, input_ids_len: int, sampling_params: dict
) -> dict:
    sp = deepcopy(sampling_params)

    if not getattr(args, "generate_dynamic_max_tokens", True):
        return sp

    ctx = getattr(args, "rollout_max_context_len", None)
    if not ctx:
        return sp

    remaining = ctx - input_ids_len
    if remaining <= 0:
        sp["max_new_tokens"] = 0
        return sp

    per_turn_cap = getattr(args, "generate_per_turn_max_tokens", 0)
    if per_turn_cap > 0:
        sp["max_new_tokens"] = min(remaining, per_turn_cap)
    else:
        sp["max_new_tokens"] = remaining

    return sp


# ---------------------------------------------------------------------------
# E2B tool-call execution helpers
# ---------------------------------------------------------------------------

async def _execute_tool_calls_in_sandbox(
    tool_calls,
    sandbox,
    code_timeout: int,
) -> list[dict[str, Any]]:
    """Execute parsed tool calls inside the per-sample E2B sandbox."""
    from sglang.srt.function_call.core_types import ToolCallItem

    tool_messages: list[dict[str, Any]] = []
    for call in tool_calls:
        if isinstance(call, ToolCallItem):
            name = call.name
            params = json.loads(call.parameters) if call.parameters else {}
            tool_call_id = f"call_{uuid.uuid4().hex[:24]}"
        else:
            name = call.function.name
            params = json.loads(call.function.arguments) if call.function.arguments else {}
            tool_call_id = getattr(call, "id", f"call_{uuid.uuid4().hex[:24]}")

        t0 = time.time()
        result = await execute_tool_in_sandbox(sandbox, name, params, code_timeout)
        logger.debug("Tool %s executed in %.2fs", name, time.time() - t0)

        tool_messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": result,
            "name": name,
        })
    return tool_messages


def _pad_routed_experts_if_needed(sample: Sample, args) -> None:
    """Pad ``rollout_routed_experts`` so its first dimension equals
    ``len(sample.tokens) - 1``.

    After tool-response tokens are appended to ``sample.tokens`` the routing
    tensor from the last SGLang call may be shorter.  Padding with -1 keeps
    it consistent with the training code's expectations.
    """
    re = sample.rollout_routed_experts
    if re is None:
        return
    expected = len(sample.tokens) - 1
    if re.shape[0] == expected:
        return
    deficit = expected - re.shape[0]
    if deficit < 0:
        logger.warning("rollout_routed_experts longer than tokens (%d vs %d); truncating", re.shape[0], expected)
        sample.rollout_routed_experts = re[:expected]
        return
    pad = np.full((deficit, re.shape[1], re.shape[2]), fill_value=-1, dtype=re.dtype)
    sample.rollout_routed_experts = np.concatenate([re, pad], axis=0)


# ---------------------------------------------------------------------------
# Core multi-turn generate with E2B sandbox
# ---------------------------------------------------------------------------

async def _multi_turn_tir_generate(input: GenerateFnInput) -> GenerateFnOutput:
    args = input.args
    sample = deepcopy(input.sample)
    tokenizer = input.state.tokenizer

    tool_call_parser = create_tool_call_parser(
        CODE_INTERPRETER_TOOLS,
        getattr(args, "generate_tool_call_parser", "minicpm4_xml"),
    )

    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    # When the data pipeline has already applied the chat template (prompt is
    # a string), tools are NOT injected because compute_prompt_ids_from_sample
    # only passes tools to apply_chat_template for message-list prompts.
    # Fix: prepend the tool system-message prefix to the string prompt.
    if isinstance(sample.prompt, str):
        prefix = _get_tool_system_prefix(tokenizer, CODE_INTERPRETER_TOOLS)
        if prefix and not sample.prompt.startswith(prefix[:40]):
            sample.prompt = prefix + sample.prompt

    prompt_tokens_ids = compute_prompt_ids_from_sample(
        input.state, sample, tools=CODE_INTERPRETER_TOOLS,
    )
    sample.tokens = prompt_tokens_ids.copy()

    # Ensure rollout_log_probs / loss_mask are never None even if no
    # generation happens (e.g. prompt fills the entire context window).
    if sample.rollout_log_probs is None:
        sample.rollout_log_probs = []
    if sample.loss_mask is None:
        sample.loss_mask = []

    max_turns = getattr(args, "generate_max_turns", 16)
    code_timeout = getattr(args, "generate_e2b_code_timeout", 30)
    max_concurrency = getattr(args, "generate_e2b_max_concurrency", 64)
    sample_timeout = getattr(args, "generate_sample_timeout", 120)

    semaphore = _get_semaphore(max_concurrency)
    sample_t0 = time.time()

    async with semaphore:
        sandbox = None
        try:
            sandbox = await create_sandbox()
        except Exception as exc:
            logger.error(
                "Failed to create E2B sandbox (%s: %s)", type(exc).__name__, exc,
            )
            sample.status = Sample.Status.FAILED
            return GenerateFnOutput(samples=sample)

        try:
            for _turn in range(max_turns):
                if time.time() - sample_t0 > sample_timeout:
                    logger.warning(
                        "Sample timeout (%.0fs > %ds) after %d turns",
                        time.time() - sample_t0, sample_timeout, _turn,
                    )
                    if not sample.response:
                        sample.status = Sample.Status.FAILED
                    break

                sp = _compute_dynamic_max_new_tokens(
                    args, len(sample.tokens), input.sampling_params,
                )
                payload, halt_status = compute_request_payload(args, sample.tokens, sp)
                if payload is None:
                    sample.status = halt_status
                    break

                output = await post(url, payload)
                await update_sample_from_response(
                    args, sample, payload=payload, output=output, update_loss_mask=True,
                )

                if output["meta_info"]["finish_reason"]["type"] in ("abort", "length"):
                    break

                _, tool_calls = tool_call_parser.parse_non_stream(output["text"])
                if len(tool_calls) == 0:
                    break

                tool_messages = await _execute_tool_calls_in_sandbox(
                    tool_calls, sandbox, code_timeout,
                )
                update_sample_with_tool_responses(
                    sample, tool_messages, tokenizer=tokenizer,
                )

            _pad_routed_experts_if_needed(sample, args)
        finally:
            if sandbox is not None:
                await kill_sandbox(sandbox)

    return GenerateFnOutput(samples=sample)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    metadata = input.sample.metadata if isinstance(input.sample.metadata, dict) else {}

    default_mode = getattr(input.args, "generate_default_rollout_mode", "multi_turn")
    rollout_mode = metadata.get("rollout_mode", default_mode)

    if rollout_mode == "multi_turn":
        return await _multi_turn_tir_generate(input)
    return await single_turn_generate(input)


# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------

def _add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--generate-default-rollout-mode",
        type=str,
        default="multi_turn",
        choices=["single_turn", "multi_turn"],
        help="Default rollout mode when not specified in sample metadata.",
    )
    parser.add_argument("--generate-max-turns", type=int, default=16)
    parser.add_argument(
        "--generate-tool-call-parser",
        type=str,
        default="minicpm4_xml",
        help="Tool call parser name (passed to SGLang FunctionCallParser).",
    )
    parser.add_argument(
        "--generate-e2b-max-concurrency",
        type=int,
        default=64,
        help="Maximum number of concurrent E2B sandboxes.",
    )
    parser.add_argument(
        "--generate-e2b-code-timeout",
        type=int,
        default=30,
        help="Timeout in seconds for a single code execution in E2B.",
    )
    parser.add_argument(
        "--generate-sample-timeout",
        type=int,
        default=120,
        help="Hard wall-clock timeout in seconds for the entire multi-turn loop of one sample.",
    )
    parser.add_argument(
        "--generate-dynamic-max-tokens",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When enabled (default), each multi-turn generation turn uses "
            "max_new_tokens = rollout_max_context_len - len(input_ids) "
            "instead of the fixed --rollout-max-response-len cap."
        ),
    )
    parser.add_argument(
        "--generate-per-turn-max-tokens",
        type=int,
        default=0,
        help=(
            "Optional per-turn hard cap on generated tokens during dynamic "
            "max-tokens computation. 0 = no cap (use full remaining context)."
        ),
    )


generate.add_arguments = _add_arguments
