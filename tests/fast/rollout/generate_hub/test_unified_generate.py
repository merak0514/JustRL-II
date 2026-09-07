"""
Tests for the unified generate function that routes between single-turn and
multi-turn based on ``sample.metadata["rollout_mode"]``.
"""

from itertools import groupby
from unittest.mock import patch

import pytest
from transformers import AutoTokenizer

from miles.rollout.base_types import GenerateFnInput
from miles.rollout.inference_rollout.compatibility import load_generate_function
from miles.rollout.inference_rollout.inference_rollout_common import GenerateState
from miles.utils.async_utils import run
from miles.utils.http_utils import init_http_client
from miles.utils.misc import SingletonMeta
from miles.utils.test_utils.mock_sglang_server import ProcessResult, with_mock_server
from miles.utils.test_utils.mock_tools import SAMPLE_TOOLS, TwoTurnStub
from miles.utils.types import Sample
from tests.fast.fixtures.generation_fixtures import GenerateEnv, GenerateResult, make_sample, with_miles_router

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen3-0.6B"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
UNIFIED_FN_PATH = "miles.rollout.generate_hub.unified_generate.generate"

SINGLE_TURN_PROMPT = "What is 1+7?"
SINGLE_TURN_RESPONSE = "\\boxed{8}"
SINGLE_TURN_PROMPT_TOKENS = TOKENIZER(SINGLE_TURN_PROMPT, add_special_tokens=False)["input_ids"]

MULTI_TURN_PROMPT = [{"role": "user", "content": TwoTurnStub.USER_QUESTION}]

SAMPLING_PARAMS = {"max_new_tokens": 64, "temperature": 0.7}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_unified_args(router_port: int, **overrides):
    argv = [
        "pytest",
        "--train-backend", "fsdp",
        "--rollout-batch-size", "1",
        "--num-rollout", "1",
        "--rollout-num-gpus", "1",
        "--rollout-num-gpus-per-engine", "1",
        "--hf-checkpoint", MODEL_NAME,
        "--prompt-data", "/dev/null",
        "--rm-type", "math",
        "--sglang-router-ip", "127.0.0.1",
        "--sglang-router-port", str(router_port),
        "--rollout-max-response-len", str(overrides.get("rollout_max_response_len", 16)),
        "--custom-generate-function-path", UNIFIED_FN_PATH,
        "--generate-max-turns", str(overrides.get("generate_max_turns", 16)),
        "--generate-tool-specs-path", "miles.utils.test_utils.mock_tools.SAMPLE_TOOLS",
        "--generate-tool-call-parser", "qwen25",
        "--generate-execute-tool-function-path", "miles.utils.test_utils.mock_tools.execute_tool_call",
    ]
    if overrides.get("rollout_max_context_len") is not None:
        argv.extend(["--rollout-max-context-len", str(overrides["rollout_max_context_len"])])
    if overrides.get("generate_dynamic_max_tokens") is False:
        argv.append("--no-generate-dynamic-max-tokens")
    if overrides.get("generate_per_turn_max_tokens") is not None:
        argv.extend(["--generate-per-turn-max-tokens", str(overrides["generate_per_turn_max_tokens"])])

    from miles.utils.arguments import parse_args

    with patch("sys.argv", argv):
        args = parse_args()
    init_http_client(args)
    return args


def _run(env: GenerateEnv, sample: Sample, sampling_params: dict | None = None) -> GenerateResult:
    env.mock_server.request_log.clear()
    generate_fn = load_generate_function(UNIFIED_FN_PATH)
    state = GenerateState(env.args)

    async def _go():
        inp = GenerateFnInput(
            state=state,
            sample=sample,
            sampling_params=(sampling_params or SAMPLING_PARAMS).copy(),
            evaluation=False,
        )
        return (await generate_fn(inp)).samples

    result_sample = run(_go())
    return GenerateResult(sample=result_sample, requests=list(env.mock_server.request_log))


def _token_len(text: str) -> int:
    return len(TOKENIZER(text, add_special_tokens=False)["input_ids"])


def _loss_mask_segments(sample: Sample) -> list[tuple[int, int]]:
    """Return (mask_value, count) segments from the loss_mask."""
    if not sample.loss_mask:
        return []
    return [(val, len(list(g))) for val, g in groupby(sample.loss_mask)]


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def unified_env(request):
    SingletonMeta.clear_all_instances()
    params = getattr(request, "param", {})

    def default_process_fn(_):
        return ProcessResult(text=SINGLE_TURN_RESPONSE, finish_reason="stop")

    with with_mock_server(model_name=MODEL_NAME, process_fn=default_process_fn) as mock_server:
        with with_miles_router(mock_server.url, MODEL_NAME) as router_port:
            args = _make_unified_args(router_port, **params.get("args_kwargs", {}))
            yield GenerateEnv(args=args, mock_server=mock_server)
    SingletonMeta.clear_all_instances()


# ===========================================================================
# Routing tests
# ===========================================================================


class TestRouting:
    """Verify that rollout_mode in metadata selects the right path."""

    def test_default_routes_to_single_turn(self, unified_env):
        sample = make_sample(prompt=SINGLE_TURN_PROMPT)
        result = _run(unified_env, sample)

        assert len(result.requests) == 1
        assert result.sample.status == Sample.Status.COMPLETED
        assert result.sample.response == SINGLE_TURN_RESPONSE
        # single_turn.generate does NOT populate loss_mask
        assert result.sample.loss_mask is None

    def test_explicit_single_turn(self, unified_env):
        sample = make_sample(prompt=SINGLE_TURN_PROMPT)
        sample.metadata = {"rollout_mode": "single_turn"}
        result = _run(unified_env, sample)

        assert len(result.requests) == 1
        assert result.sample.status == Sample.Status.COMPLETED
        assert result.sample.loss_mask is None

    def test_multi_turn_routes_correctly(self, unified_env):
        unified_env.mock_server.process_fn = TwoTurnStub.process_fn

        sample = make_sample(prompt=MULTI_TURN_PROMPT)
        sample.metadata = {"rollout_mode": "multi_turn"}
        result = _run(unified_env, sample)

        assert len(result.requests) == 2
        assert result.sample.status == Sample.Status.COMPLETED
        assert result.sample.loss_mask is not None
        assert 0 in result.sample.loss_mask
        assert 1 in result.sample.loss_mask


# ===========================================================================
# Tool-spec isolation tests
# ===========================================================================


class TestToolSpecIsolation:
    """Single-turn must NOT inject tool specs; multi-turn MUST."""

    def test_single_turn_no_tool_specs_in_prompt(self, unified_env):
        sample = make_sample(prompt=SINGLE_TURN_PROMPT)
        result = _run(unified_env, sample)

        request_input_ids = result.requests[0]["input_ids"]
        assert request_input_ids == SINGLE_TURN_PROMPT_TOKENS

    def test_multi_turn_injects_tool_specs(self, unified_env):
        unified_env.mock_server.process_fn = lambda _: ProcessResult(
            text="The answer is 42.", finish_reason="stop"
        )
        sample = make_sample(prompt=MULTI_TURN_PROMPT)
        sample.metadata = {"rollout_mode": "multi_turn"}
        result = _run(unified_env, sample)

        request_input_ids = result.requests[0]["input_ids"]
        prompt_with_tools = TOKENIZER.apply_chat_template(
            MULTI_TURN_PROMPT, tokenize=False, add_generation_prompt=True, tools=SAMPLE_TOOLS
        )
        expected_ids = TOKENIZER(prompt_with_tools, add_special_tokens=False)["input_ids"]
        assert request_input_ids == expected_ids

        prompt_without_tools = TOKENIZER.apply_chat_template(
            MULTI_TURN_PROMPT, tokenize=False, add_generation_prompt=True
        )
        ids_without_tools = TOKENIZER(prompt_without_tools, add_special_tokens=False)["input_ids"]
        assert len(request_input_ids) > len(ids_without_tools)


# ===========================================================================
# Multi-turn behaviour through the unified interface
# ===========================================================================


class TestMultiTurnBehaviour:
    def test_no_tool_call_exits_first_turn(self, unified_env):
        unified_env.mock_server.process_fn = lambda _: ProcessResult(
            text="Just a plain answer.", finish_reason="stop"
        )
        sample = make_sample(prompt=MULTI_TURN_PROMPT)
        sample.metadata = {"rollout_mode": "multi_turn"}
        result = _run(unified_env, sample)

        assert len(result.requests) == 1
        assert result.sample.status == Sample.Status.COMPLETED

    def test_two_turn_tool_call(self, unified_env):
        unified_env.mock_server.process_fn = TwoTurnStub.process_fn
        sample = make_sample(prompt=MULTI_TURN_PROMPT)
        sample.metadata = {"rollout_mode": "multi_turn"}
        result = _run(unified_env, sample)

        assert len(result.requests) == 2
        assert result.sample.status == Sample.Status.COMPLETED

        segments = _loss_mask_segments(result.sample)
        assert len(segments) == 3
        assert segments[0][0] == 1  # first model output
        assert segments[1][0] == 0  # tool response (observation)
        assert segments[2][0] == 1  # second model output

    def test_abort_exits(self, unified_env):
        unified_env.mock_server.process_fn = lambda _: ProcessResult(
            text="partial...", finish_reason="abort"
        )
        sample = make_sample(prompt=MULTI_TURN_PROMPT)
        sample.metadata = {"rollout_mode": "multi_turn"}
        result = _run(unified_env, sample)

        assert len(result.requests) == 1
        assert result.sample.status == Sample.Status.ABORTED

    def test_length_exits(self, unified_env):
        unified_env.mock_server.process_fn = lambda _: ProcessResult(
            text=TwoTurnStub.FIRST_RESPONSE, finish_reason="length"
        )
        sample = make_sample(prompt=MULTI_TURN_PROMPT)
        sample.metadata = {"rollout_mode": "multi_turn"}
        result = _run(unified_env, sample)

        assert len(result.requests) == 1
        assert result.sample.status == Sample.Status.TRUNCATED

    @pytest.mark.parametrize(
        "unified_env", [{"args_kwargs": {"generate_max_turns": 1}}], indirect=True
    )
    def test_max_turns_respected(self, unified_env):
        unified_env.mock_server.process_fn = TwoTurnStub.process_fn
        sample = make_sample(prompt=MULTI_TURN_PROMPT)
        sample.metadata = {"rollout_mode": "multi_turn"}
        result = _run(unified_env, sample)

        assert len(result.requests) == 1


# ===========================================================================
# Single-turn behaviour through the unified interface
# ===========================================================================


class TestSingleTurnBehaviour:
    def test_finish_stop(self, unified_env):
        sample = make_sample(prompt=SINGLE_TURN_PROMPT)
        result = _run(unified_env, sample)
        assert result.sample.status == Sample.Status.COMPLETED
        assert result.sample.response == SINGLE_TURN_RESPONSE

    def test_finish_abort(self, unified_env):
        unified_env.mock_server.process_fn = lambda _: ProcessResult(text="partial", finish_reason="abort")
        sample = make_sample(prompt=SINGLE_TURN_PROMPT)
        result = _run(unified_env, sample)
        assert result.sample.status == Sample.Status.ABORTED

    def test_finish_length(self, unified_env):
        unified_env.mock_server.process_fn = lambda _: ProcessResult(text="long answer", finish_reason="length")
        sample = make_sample(prompt=SINGLE_TURN_PROMPT)
        result = _run(unified_env, sample)
        assert result.sample.status == Sample.Status.TRUNCATED

    def test_partial_rollout_resuming(self, unified_env):
        """Two consecutive calls on the same sample (partial rollout)."""
        partial_text = "\\boxed"
        remaining_text = "{8}"

        unified_env.mock_server.process_fn = lambda _: ProcessResult(text=partial_text, finish_reason="abort")
        sample = make_sample(prompt=SINGLE_TURN_PROMPT)
        result1 = _run(unified_env, sample)
        assert result1.sample.status == Sample.Status.ABORTED
        assert result1.sample.response == partial_text

        unified_env.mock_server.process_fn = lambda _: ProcessResult(text=remaining_text, finish_reason="stop")
        result2 = _run(unified_env, result1.sample)
        assert result2.sample.status == Sample.Status.COMPLETED
        assert result2.sample.response == partial_text + remaining_text


# ===========================================================================
# Context length tests
# ===========================================================================


class TestContextLength:
    @pytest.mark.parametrize(
        "unified_env",
        [{"args_kwargs": {"rollout_max_context_len": 5}}],
        indirect=True,
    )
    def test_single_turn_prompt_exceeds_max_context_len(self, unified_env):
        sample = make_sample(prompt=SINGLE_TURN_PROMPT)
        result = _run(unified_env, sample)
        assert result.requests == []
        assert result.sample.status == Sample.Status.TRUNCATED

    @pytest.mark.parametrize(
        "unified_env",
        [{"args_kwargs": {"rollout_max_context_len": len(TwoTurnStub.FIRST_PROMPT_TOKEN_IDS)}}],
        indirect=True,
    )
    def test_multi_turn_prompt_exceeds_max_context_len(self, unified_env):
        sample = make_sample(prompt=MULTI_TURN_PROMPT)
        sample.metadata = {"rollout_mode": "multi_turn"}
        result = _run(unified_env, sample)
        assert result.requests == []
        assert result.sample.status == Sample.Status.TRUNCATED

    @pytest.mark.parametrize(
        "unified_env",
        [
            {
                "args_kwargs": {
                    "rollout_max_context_len": len(TwoTurnStub.FIRST_PROMPT_TOKEN_IDS)
                    + _token_len(TwoTurnStub.FIRST_RESPONSE)
                    + _token_len(TwoTurnStub.FIRST_TOOL_RESPONSE)
                }
            }
        ],
        indirect=True,
    )
    def test_multi_turn_second_turn_exceeds_max_context_len(self, unified_env):
        unified_env.mock_server.process_fn = TwoTurnStub.process_fn
        sample = make_sample(prompt=MULTI_TURN_PROMPT)
        sample.metadata = {"rollout_mode": "multi_turn"}
        result = _run(unified_env, sample)

        assert len(result.requests) == 1
        assert result.sample.status == Sample.Status.TRUNCATED


# ===========================================================================
# Dynamic max_new_tokens tests
# ===========================================================================


class TestDynamicMaxTokens:
    """Verify that multi-turn generation uses the full remaining context window."""

    @pytest.mark.parametrize(
        "unified_env",
        [{"args_kwargs": {"rollout_max_context_len": 4096, "rollout_max_response_len": 16}}],
        indirect=True,
    )
    def test_dynamic_budget_exceeds_fixed_cap(self, unified_env):
        """With dynamic enabled, max_new_tokens should be context_len - input_len,
        NOT the fixed rollout_max_response_len (16)."""
        unified_env.mock_server.process_fn = lambda _: ProcessResult(
            text="The answer is 42.", finish_reason="stop"
        )
        sample = make_sample(prompt=MULTI_TURN_PROMPT)
        sample.metadata = {"rollout_mode": "multi_turn"}
        result = _run(unified_env, sample)

        assert len(result.requests) == 1
        req = result.requests[0]
        sent_max = req["sampling_params"]["max_new_tokens"]
        prompt_len = len(req["input_ids"])
        assert sent_max == 4096 - prompt_len, (
            f"Expected dynamic budget {4096 - prompt_len}, got {sent_max}"
        )

    @pytest.mark.parametrize(
        "unified_env",
        [{"args_kwargs": {
            "rollout_max_context_len": 4096,
            "rollout_max_response_len": 16,
            "generate_dynamic_max_tokens": False,
        }}],
        indirect=True,
    )
    def test_dynamic_disabled_uses_fixed_cap(self, unified_env):
        """With dynamic disabled, max_new_tokens stays at the sampling_params value
        (which in real pipelines equals rollout_max_response_len)."""
        unified_env.mock_server.process_fn = lambda _: ProcessResult(
            text="The answer is 42.", finish_reason="stop"
        )
        sample = make_sample(prompt=MULTI_TURN_PROMPT)
        sample.metadata = {"rollout_mode": "multi_turn"}
        result = _run(unified_env, sample)

        assert len(result.requests) == 1
        req = result.requests[0]
        sent_max = req["sampling_params"]["max_new_tokens"]
        # SAMPLING_PARAMS has max_new_tokens=64; dynamic is off, so it stays
        assert sent_max == SAMPLING_PARAMS["max_new_tokens"]

    @pytest.mark.parametrize(
        "unified_env",
        [{"args_kwargs": {
            "rollout_max_context_len": 4096,
            "rollout_max_response_len": 16,
            "generate_per_turn_max_tokens": 512,
        }}],
        indirect=True,
    )
    def test_per_turn_cap_limits_dynamic_budget(self, unified_env):
        """With per_turn_cap=512, dynamic budget should be min(remaining, 512)."""
        unified_env.mock_server.process_fn = lambda _: ProcessResult(
            text="The answer is 42.", finish_reason="stop"
        )
        sample = make_sample(prompt=MULTI_TURN_PROMPT)
        sample.metadata = {"rollout_mode": "multi_turn"}
        result = _run(unified_env, sample)

        assert len(result.requests) == 1
        req = result.requests[0]
        sent_max = req["sampling_params"]["max_new_tokens"]
        prompt_len = len(req["input_ids"])
        expected = min(4096 - prompt_len, 512)
        assert sent_max == expected, f"Expected {expected}, got {sent_max}"

    @pytest.mark.parametrize(
        "unified_env",
        [{"args_kwargs": {"rollout_max_context_len": 4096, "rollout_max_response_len": 16}}],
        indirect=True,
    )
    def test_multi_turn_budget_shrinks_across_turns(self, unified_env):
        """Across turns, max_new_tokens should shrink as input_ids grow."""
        unified_env.mock_server.process_fn = TwoTurnStub.process_fn
        sample = make_sample(prompt=MULTI_TURN_PROMPT)
        sample.metadata = {"rollout_mode": "multi_turn"}
        result = _run(unified_env, sample)

        assert len(result.requests) == 2
        req1, req2 = result.requests
        budget1 = req1["sampling_params"]["max_new_tokens"]
        budget2 = req2["sampling_params"]["max_new_tokens"]
        input_len1 = len(req1["input_ids"])
        input_len2 = len(req2["input_ids"])
        assert input_len2 > input_len1, "Second turn should have more input tokens"
        assert budget1 == 4096 - input_len1
        assert budget2 == 4096 - input_len2
        assert budget2 < budget1, "Budget should shrink as context grows"

    @pytest.mark.parametrize(
        "unified_env",
        [{"args_kwargs": {"rollout_max_response_len": 16}}],
        indirect=True,
    )
    def test_no_context_len_falls_back_to_fixed(self, unified_env):
        """Without rollout_max_context_len, dynamic has no effect; sampling_params value is used."""
        unified_env.mock_server.process_fn = lambda _: ProcessResult(
            text="The answer is 42.", finish_reason="stop"
        )
        sample = make_sample(prompt=MULTI_TURN_PROMPT)
        sample.metadata = {"rollout_mode": "multi_turn"}
        result = _run(unified_env, sample)

        assert len(result.requests) == 1
        req = result.requests[0]
        sent_max = req["sampling_params"]["max_new_tokens"]
        assert sent_max == SAMPLING_PARAMS["max_new_tokens"]

    @pytest.mark.parametrize(
        "unified_env",
        [{"args_kwargs": {"rollout_max_context_len": 4096, "rollout_max_response_len": 16}}],
        indirect=True,
    )
    def test_single_turn_unaffected_by_dynamic(self, unified_env):
        """Dynamic max_tokens only applies to multi-turn; single-turn uses
        sampling_params clamped by context_len via compute_request_payload."""
        sample = make_sample(prompt=SINGLE_TURN_PROMPT)
        result = _run(unified_env, sample)

        assert len(result.requests) == 1
        req = result.requests[0]
        sent_max = req["sampling_params"]["max_new_tokens"]
        # single_turn.generate passes sampling_params through compute_request_payload
        # which does min(sampling_params.max_new_tokens, context_len - input_len)
        prompt_len = len(req["input_ids"])
        expected = min(SAMPLING_PARAMS["max_new_tokens"], 4096 - prompt_len)
        assert sent_max == expected


# ===========================================================================
# Metadata edge cases
# ===========================================================================


class TestMetadataEdgeCases:
    def test_none_metadata_defaults_to_single_turn(self, unified_env):
        sample = make_sample(prompt=SINGLE_TURN_PROMPT)
        sample.metadata = None
        result = _run(unified_env, sample)
        assert result.sample.status == Sample.Status.COMPLETED
        assert result.sample.loss_mask is None

    def test_empty_metadata_defaults_to_single_turn(self, unified_env):
        sample = make_sample(prompt=SINGLE_TURN_PROMPT)
        sample.metadata = {}
        result = _run(unified_env, sample)
        assert result.sample.status == Sample.Status.COMPLETED
        assert result.sample.loss_mask is None

    def test_unknown_rollout_mode_defaults_to_single_turn(self, unified_env):
        sample = make_sample(prompt=SINGLE_TURN_PROMPT)
        sample.metadata = {"rollout_mode": "unknown_mode"}
        result = _run(unified_env, sample)
        assert result.sample.status == Sample.Status.COMPLETED
        assert result.sample.loss_mask is None

    def test_metadata_with_rm_type_and_rollout_mode(self, unified_env):
        """Ensure rm_type and rollout_mode can coexist in metadata."""
        unified_env.mock_server.process_fn = TwoTurnStub.process_fn
        sample = make_sample(prompt=MULTI_TURN_PROMPT)
        sample.metadata = {"rm_type": "math", "rollout_mode": "multi_turn"}
        result = _run(unified_env, sample)

        assert len(result.requests) == 2
        assert result.sample.status == Sample.Status.COMPLETED
