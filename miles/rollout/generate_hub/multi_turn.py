"""
Simple multi-turn generation with tool calling.
"""

import argparse
import logging
from copy import deepcopy

import numpy as np

from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
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

logger = logging.getLogger(__name__)


def _set_turn_metadata(sample, num_turns: int) -> None:
    sample.metadata["num_turns"] = num_turns
    sample.metadata["round_number"] = num_turns


def _pad_routed_experts_if_needed(sample) -> None:
    routed_experts = sample.rollout_routed_experts
    if routed_experts is None:
        return

    expected = len(sample.tokens) - 1
    if routed_experts.shape[0] == expected:
        return

    deficit = expected - routed_experts.shape[0]
    if deficit < 0:
        logger.warning(
            "rollout_routed_experts longer than tokens (%d vs %d); truncating",
            routed_experts.shape[0],
            expected,
        )
        sample.rollout_routed_experts = routed_experts[:expected]
        return

    pad = np.full((deficit, routed_experts.shape[1], routed_experts.shape[2]), -1, dtype=routed_experts.dtype)
    sample.rollout_routed_experts = np.concatenate([routed_experts, pad], axis=0)


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    # ----------------------- Setup -------------------------

    args = input.args
    sample = deepcopy(input.sample)
    tokenizer = input.state.tokenizer
    assert not args.partial_rollout, "Partial rollout is not supported"

    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    execute_tool_function = load_function(args.generate_execute_tool_function_path)

    tool_specs = load_function(args.generate_tool_specs_path)
    tool_call_parser = create_tool_call_parser(tool_specs, args.generate_tool_call_parser)

    multi_samples = []
    num_turns = 0

    # ----------------------- Initial prompts -------------------------

    prompt_tokens_ids = compute_prompt_ids_from_sample(input.state, sample, tools=tool_specs)

    sample.tokens = prompt_tokens_ids.copy()

    for _turn in range(args.generate_max_turns):
        # ----------------------- Call inference endpoint -------------------------

        payload, halt_status = compute_request_payload(args, sample.tokens, input.sampling_params)
        if payload is None:
            sample.status = halt_status
            if args.generate_multi_samples and multi_samples:
                multi_samples[-1].status = halt_status
            break

        if args.generate_multi_samples:
            sample = deepcopy(input.sample)

        output = await post(url, payload)
        await update_sample_from_response(args, sample, payload=payload, output=output, update_loss_mask=True)
        num_turns = _turn + 1
        _set_turn_metadata(sample, num_turns)

        if args.generate_multi_samples:
            turn_sample = deepcopy(sample)
            _set_turn_metadata(turn_sample, num_turns)
            multi_samples.append(turn_sample)

        if output["meta_info"]["finish_reason"]["type"] in ("abort", "length"):
            break

        # ----------------------- Execute tools -------------------------

        _, tool_calls = tool_call_parser.parse_non_stream(output["text"])
        if len(tool_calls) == 0:
            break

        sample.metadata["tool_call_count"] = sample.metadata.get("tool_call_count", 0) + len(tool_calls)
        tool_messages = await execute_tool_calls(tool_calls, execute_tool_function)
        update_sample_with_tool_responses(sample, tool_messages, tokenizer=tokenizer)
        _pad_routed_experts_if_needed(sample)

    if args.generate_multi_samples:
        for idx, turn_sample in enumerate(multi_samples, start=1):
            _set_turn_metadata(turn_sample, idx)
    else:
        _set_turn_metadata(sample, num_turns)

    return GenerateFnOutput(samples=multi_samples if args.generate_multi_samples else sample)


def _add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--generate-max-turns", type=int, default=16)
    parser.add_argument("--generate-tool-specs-path", type=str)
    parser.add_argument("--generate-tool-call-parser", type=str)
    parser.add_argument("--generate-execute-tool-function-path", type=str)
    parser.add_argument("--generate-multi-samples", action="store_true")


generate.add_arguments = _add_arguments
