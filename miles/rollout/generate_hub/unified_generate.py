"""
Unified generate function that routes between single-turn and multi-turn
based on ``sample.metadata["rollout_mode"]``.

* ``"single_turn"`` (default) — delegates to :func:`single_turn.generate`,
  no tool specs injected.
* ``"multi_turn"`` — runs the tool-calling loop (same logic as
  :mod:`multi_turn`), with tool specs injected only for these samples.

Dynamic ``max_new_tokens``
~~~~~~~~~~~~~~~~~~~~~~~~~~
When ``--generate-dynamic-max-tokens`` is enabled (the default), each turn's
``max_new_tokens`` is computed as::

    remaining = rollout_max_context_len - len(input_ids)
    max_new_tokens = min(remaining, per_turn_cap)    # if per_turn_cap > 0
    max_new_tokens = remaining                        # if per_turn_cap == 0

This lets the model use the full remaining context window on every turn
instead of being capped by ``--rollout-max-response-len``.

Usage::

    --custom-generate-function-path miles.rollout.generate_hub.unified_generate.generate
    --generate-tool-specs-path       my_tools.TOOL_SPECS
    --generate-tool-call-parser      qwen25
    --generate-execute-tool-function-path my_tools.execute_tool_call
"""

import argparse
from copy import deepcopy

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_hub.single_turn import generate as single_turn_generate
from miles.rollout.generate_utils.generate_endpoint_utils import (
    compute_prompt_ids_from_sample,
    compute_request_payload,
    update_sample_from_response,
)
from miles.rollout.generate_utils.tool_call_utils import (
    create_tool_call_parser,
    execute_tool_calls,
    update_sample_with_tool_responses,
)
from miles.utils.http_utils import post
from miles.utils.misc import load_function


def _compute_dynamic_max_new_tokens(
    args, input_ids_len: int, sampling_params: dict
) -> dict:
    """Return a *copy* of ``sampling_params`` with ``max_new_tokens`` adjusted
    to fill the remaining context window.

    When ``generate_dynamic_max_tokens`` is *disabled*, the original
    ``sampling_params`` is returned unchanged (just a deepcopy).
    """
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


async def _multi_turn_generate(input: GenerateFnInput) -> GenerateFnOutput:
    args = input.args
    sample = deepcopy(input.sample)
    tokenizer = input.state.tokenizer

    execute_tool_function = load_function(args.generate_execute_tool_function_path)
    tool_specs = load_function(args.generate_tool_specs_path)
    tool_call_parser = create_tool_call_parser(tool_specs, args.generate_tool_call_parser)

    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    prompt_tokens_ids = compute_prompt_ids_from_sample(input.state, sample, tools=tool_specs)
    sample.tokens = prompt_tokens_ids.copy()

    for _turn in range(args.generate_max_turns):
        sp = _compute_dynamic_max_new_tokens(args, len(sample.tokens), input.sampling_params)
        payload, halt_status = compute_request_payload(args, sample.tokens, sp)
        if payload is None:
            sample.status = halt_status
            break

        output = await post(url, payload)
        await update_sample_from_response(args, sample, payload=payload, output=output, update_loss_mask=True)

        if output["meta_info"]["finish_reason"]["type"] in ("abort", "length"):
            break

        _, tool_calls = tool_call_parser.parse_non_stream(output["text"])
        if len(tool_calls) == 0:
            break

        tool_messages = await execute_tool_calls(tool_calls, execute_tool_function)
        update_sample_with_tool_responses(sample, tool_messages, tokenizer=tokenizer)

    return GenerateFnOutput(samples=sample)


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    metadata = input.sample.metadata if isinstance(input.sample.metadata, dict) else {}
    rollout_mode = metadata.get("rollout_mode", "single_turn")

    if rollout_mode == "multi_turn":
        return await _multi_turn_generate(input)
    return await single_turn_generate(input)


def _add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--generate-max-turns", type=int, default=16)
    parser.add_argument("--generate-tool-specs-path", type=str, default=None)
    parser.add_argument("--generate-tool-call-parser", type=str, default=None)
    parser.add_argument("--generate-execute-tool-function-path", type=str, default=None)
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
