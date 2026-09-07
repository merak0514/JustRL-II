"""
End-to-end tests for the TIR (Tool-Integrated Reasoning) generate function.

Uses a mock SGLang server and mocked E2B sandbox to validate the full
multi-turn tool-calling loop without requiring live infrastructure.
"""

from itertools import groupby
from unittest.mock import AsyncMock, patch

import pytest
from transformers import AutoTokenizer

from miles.rollout.base_types import GenerateFnInput
from miles.rollout.generate_hub.tir_generate import CODE_INTERPRETER_TOOLS
from miles.rollout.inference_rollout.compatibility import load_generate_function
from miles.rollout.inference_rollout.inference_rollout_common import GenerateState
from miles.utils.async_utils import run
from miles.utils.http_utils import init_http_client
from miles.utils.misc import SingletonMeta
from miles.utils.test_utils.mock_sglang_server import ProcessResult, with_mock_server
from miles.utils.types import Sample
from tests.fast.fixtures.generation_fixtures import GenerateEnv, GenerateResult, make_sample, with_miles_router

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen3-0.6B"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
TIR_FN_PATH = "miles.rollout.generate_hub.tir_generate.generate"

SAMPLING_PARAMS = {"max_new_tokens": 64, "temperature": 0.7}


# ---------------------------------------------------------------------------
# Stub: two-turn code_interpreter with qwen25 format for test compatibility
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = TOKENIZER.apply_chat_template(
    [{"role": "user", "content": "dummy"}],
    tokenize=False,
    add_generation_prompt=False,
    tools=CODE_INTERPRETER_TOOLS,
)
_SYSTEM_PROMPT = _SYSTEM_PROMPT[: _SYSTEM_PROMPT.index("<|im_start|>user")]


class CodeInterpreterStub:
    """Two-turn stub: model calls code_interpreter, gets result, gives final answer."""

    USER_QUESTION = "What is the factorial of 10?"
    PROMPT = [{"role": "user", "content": USER_QUESTION}]

    FIRST_RESPONSE = (
        "Let me calculate this.\n"
        "<tool_call>\n"
        '{"name": "code_interpreter", "arguments": {"code": "import math; print(math.factorial(10))"}}\n'
        "</tool_call><|im_end|>\n"
    )

    SECOND_RESPONSE = "The factorial of 10 is 3628800."

    FIRST_PROMPT = (
        _SYSTEM_PROMPT
        + "<|im_start|>user\n"
        + USER_QUESTION
        + "<|im_end|>\n"
        + "<|im_start|>assistant\n"
    )

    TOOL_RESULT = "3628800"
    TOOL_RESPONSE_TEXT = (
        "<|im_start|>user\n"
        "<tool_response>\n"
        "3628800\n"
        "</tool_response><|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    SECOND_PROMPT = FIRST_PROMPT + FIRST_RESPONSE + TOOL_RESPONSE_TEXT

    FIRST_PROMPT_TOKEN_IDS = TOKENIZER(FIRST_PROMPT, add_special_tokens=False)["input_ids"]
    SECOND_PROMPT_TOKEN_IDS = TOKENIZER(SECOND_PROMPT, add_special_tokens=False)["input_ids"]

    @staticmethod
    def process_fn(prompt: str) -> ProcessResult:
        if prompt == CodeInterpreterStub.FIRST_PROMPT:
            return ProcessResult(text=CodeInterpreterStub.FIRST_RESPONSE, finish_reason="stop")
        if prompt == CodeInterpreterStub.SECOND_PROMPT:
            return ProcessResult(text=CodeInterpreterStub.SECOND_RESPONSE, finish_reason="stop")
        raise ValueError(f"Unexpected prompt (len={len(prompt)}): {prompt[:200]}...")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tir_args(router_port: int, **overrides):
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
        "--rollout-max-response-len", str(overrides.get("rollout_max_response_len", 64)),
        "--custom-generate-function-path", TIR_FN_PATH,
        "--generate-default-rollout-mode", overrides.get("generate_default_rollout_mode", "multi_turn"),
        "--generate-max-turns", str(overrides.get("generate_max_turns", 16)),
        "--generate-tool-call-parser", "qwen25",
        "--generate-e2b-max-concurrency", "4",
        "--generate-e2b-code-timeout", "30",
    ]
    if overrides.get("rollout_max_context_len") is not None:
        argv.extend(["--rollout-max-context-len", str(overrides["rollout_max_context_len"])])
    if overrides.get("generate_dynamic_max_tokens") is False:
        argv.append("--no-generate-dynamic-max-tokens")

    from miles.utils.arguments import parse_args

    with patch("sys.argv", argv):
        args = parse_args()
    init_http_client(args)
    return args


def _mock_e2b():
    """Return context manager that patches E2B sandbox calls."""
    mock_sandbox = AsyncMock()
    mock_sandbox.sandbox_id = "mock-sandbox-001"

    async def _mock_execute_tool(sandbox, name, params, code_timeout=600):
        if name == "code_interpreter":
            return CodeInterpreterStub.TOOL_RESULT
        return f"Error: unsupported tool '{name}'"

    return patch.multiple(
        "miles.rollout.generate_hub.tir_generate",
        create_sandbox=AsyncMock(return_value=mock_sandbox),
        kill_sandbox=AsyncMock(),
        execute_tool_in_sandbox=AsyncMock(side_effect=_mock_execute_tool),
    )


def _run(env: GenerateEnv, sample: Sample, sampling_params: dict | None = None) -> GenerateResult:
    env.mock_server.request_log.clear()
    generate_fn = load_generate_function(TIR_FN_PATH)
    state = GenerateState(env.args)

    async def _go():
        inp = GenerateFnInput(
            state=state,
            sample=sample,
            sampling_params=(sampling_params or SAMPLING_PARAMS).copy(),
            evaluation=False,
        )
        return (await generate_fn(inp)).samples

    with _mock_e2b():
        result_sample = run(_go())
    return GenerateResult(sample=result_sample, requests=list(env.mock_server.request_log))


def _loss_mask_segments(sample: Sample) -> list[tuple[int, int]]:
    if not sample.loss_mask:
        return []
    return [(val, len(list(g))) for val, g in groupby(sample.loss_mask)]


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def tir_env(request):
    SingletonMeta.clear_all_instances()
    params = getattr(request, "param", {})

    def default_process_fn(_):
        return ProcessResult(text="\\boxed{42}", finish_reason="stop")

    with with_mock_server(model_name=MODEL_NAME, process_fn=default_process_fn) as mock_server:
        with with_miles_router(mock_server.url, MODEL_NAME) as router_port:
            args = _make_tir_args(router_port, **params.get("args_kwargs", {}))
            yield GenerateEnv(args=args, mock_server=mock_server)
    SingletonMeta.clear_all_instances()


# ===========================================================================
# Routing tests
# ===========================================================================


class TestRouting:
    def test_default_routes_to_multi_turn(self, tir_env):
        """With default mode=multi_turn, even without metadata, should use multi-turn path."""
        tir_env.mock_server.process_fn = lambda _: ProcessResult(
            text="Just a plain answer.", finish_reason="stop"
        )
        sample = make_sample(prompt=CodeInterpreterStub.PROMPT)
        result = _run(tir_env, sample)

        assert len(result.requests) == 1
        assert result.sample.status == Sample.Status.COMPLETED
        assert result.sample.loss_mask is not None

    @pytest.mark.parametrize(
        "tir_env",
        [{"args_kwargs": {"generate_default_rollout_mode": "single_turn"}}],
        indirect=True,
    )
    def test_single_turn_mode_via_cli(self, tir_env):
        sample = make_sample(prompt="What is 1+7?")
        result = _run(tir_env, sample)

        assert len(result.requests) == 1
        assert result.sample.status == Sample.Status.COMPLETED
        assert result.sample.loss_mask is None

    def test_metadata_overrides_default(self, tir_env):
        """sample.metadata rollout_mode=single_turn should override CLI default."""
        sample = make_sample(prompt="What is 1+7?")
        sample.metadata = {"rollout_mode": "single_turn"}
        result = _run(tir_env, sample)

        assert result.sample.loss_mask is None


# ===========================================================================
# Multi-turn tool calling tests
# ===========================================================================


class TestMultiTurnToolCalling:
    def test_no_tool_call_exits_first_turn(self, tir_env):
        tir_env.mock_server.process_fn = lambda _: ProcessResult(
            text="The answer is 42.", finish_reason="stop"
        )
        sample = make_sample(prompt=CodeInterpreterStub.PROMPT)
        result = _run(tir_env, sample)

        assert len(result.requests) == 1
        assert result.sample.status == Sample.Status.COMPLETED

    def test_two_turn_tool_call(self, tir_env):
        tir_env.mock_server.process_fn = CodeInterpreterStub.process_fn
        sample = make_sample(prompt=CodeInterpreterStub.PROMPT)
        result = _run(tir_env, sample)

        assert len(result.requests) == 2
        assert result.sample.status == Sample.Status.COMPLETED

        segments = _loss_mask_segments(result.sample)
        assert len(segments) == 3
        assert segments[0][0] == 1  # first model output (loss=1)
        assert segments[1][0] == 0  # tool response (loss=0)
        assert segments[2][0] == 1  # second model output (loss=1)

    def test_tool_response_in_second_request(self, tir_env):
        """Verify second request's input_ids include the tool response tokens."""
        tir_env.mock_server.process_fn = CodeInterpreterStub.process_fn
        sample = make_sample(prompt=CodeInterpreterStub.PROMPT)
        result = _run(tir_env, sample)

        assert len(result.requests) == 2
        second_input_ids = result.requests[1]["input_ids"]
        assert len(second_input_ids) > len(result.requests[0]["input_ids"])

    def test_abort_exits(self, tir_env):
        tir_env.mock_server.process_fn = lambda _: ProcessResult(
            text="partial...", finish_reason="abort"
        )
        sample = make_sample(prompt=CodeInterpreterStub.PROMPT)
        result = _run(tir_env, sample)

        assert len(result.requests) == 1
        assert result.sample.status == Sample.Status.ABORTED

    def test_length_exits(self, tir_env):
        tir_env.mock_server.process_fn = lambda _: ProcessResult(
            text=CodeInterpreterStub.FIRST_RESPONSE, finish_reason="length"
        )
        sample = make_sample(prompt=CodeInterpreterStub.PROMPT)
        result = _run(tir_env, sample)

        assert len(result.requests) == 1
        assert result.sample.status == Sample.Status.TRUNCATED

    @pytest.mark.parametrize(
        "tir_env",
        [{"args_kwargs": {"generate_max_turns": 1}}],
        indirect=True,
    )
    def test_max_turns_respected(self, tir_env):
        tir_env.mock_server.process_fn = CodeInterpreterStub.process_fn
        sample = make_sample(prompt=CodeInterpreterStub.PROMPT)
        result = _run(tir_env, sample)

        assert len(result.requests) == 1


# ===========================================================================
# Sandbox lifecycle tests
# ===========================================================================


class TestSandboxLifecycle:
    def test_sandbox_created_and_killed(self, tir_env):
        """Verify create_sandbox and kill_sandbox are called exactly once."""
        tir_env.mock_server.process_fn = CodeInterpreterStub.process_fn
        sample = make_sample(prompt=CodeInterpreterStub.PROMPT)

        generate_fn = load_generate_function(TIR_FN_PATH)
        state = GenerateState(tir_env.args)

        mock_sandbox = AsyncMock()
        mock_sandbox.sandbox_id = "test-sandbox"
        mock_create = AsyncMock(return_value=mock_sandbox)
        mock_kill = AsyncMock()

        async def _mock_exec(sandbox, name, params, code_timeout=600):
            return CodeInterpreterStub.TOOL_RESULT

        async def _go():
            inp = GenerateFnInput(
                state=state,
                sample=sample,
                sampling_params=SAMPLING_PARAMS.copy(),
                evaluation=False,
            )
            return (await generate_fn(inp)).samples

        with patch.multiple(
            "miles.rollout.generate_hub.tir_generate",
            create_sandbox=mock_create,
            kill_sandbox=mock_kill,
            execute_tool_in_sandbox=AsyncMock(side_effect=_mock_exec),
        ):
            run(_go())

        mock_create.assert_called_once()
        mock_kill.assert_called_once_with(mock_sandbox)

    def test_sandbox_creation_failure(self, tir_env):
        """If sandbox creation fails, sample status should be FAILED."""
        sample = make_sample(prompt=CodeInterpreterStub.PROMPT)

        generate_fn = load_generate_function(TIR_FN_PATH)
        state = GenerateState(tir_env.args)

        async def _go():
            inp = GenerateFnInput(
                state=state,
                sample=sample,
                sampling_params=SAMPLING_PARAMS.copy(),
                evaluation=False,
            )
            return (await generate_fn(inp)).samples

        with patch.multiple(
            "miles.rollout.generate_hub.tir_generate",
            create_sandbox=AsyncMock(side_effect=RuntimeError("E2B unavailable")),
            kill_sandbox=AsyncMock(),
            execute_tool_in_sandbox=AsyncMock(),
        ):
            result = run(_go())

        assert result.status == Sample.Status.FAILED


# ===========================================================================
# Tool spec injection tests
# ===========================================================================


class TestToolSpecInjection:
    def test_multi_turn_injects_tool_specs(self, tir_env):
        """Multi-turn request should include CODE_INTERPRETER_TOOLS in the prompt."""
        tir_env.mock_server.process_fn = lambda _: ProcessResult(
            text="The answer is 42.", finish_reason="stop"
        )
        sample = make_sample(prompt=CodeInterpreterStub.PROMPT)
        result = _run(tir_env, sample)

        request_input_ids = result.requests[0]["input_ids"]
        prompt_with_tools = TOKENIZER.apply_chat_template(
            CodeInterpreterStub.PROMPT,
            tokenize=False,
            add_generation_prompt=True,
            tools=CODE_INTERPRETER_TOOLS,
        )
        expected_ids = TOKENIZER(prompt_with_tools, add_special_tokens=False)["input_ids"]
        assert request_input_ids == expected_ids

    @pytest.mark.parametrize(
        "tir_env",
        [{"args_kwargs": {"generate_default_rollout_mode": "single_turn"}}],
        indirect=True,
    )
    def test_single_turn_no_tool_specs(self, tir_env):
        """Single-turn should NOT inject tool specs."""
        sample = make_sample(prompt="What is 1+7?")
        result = _run(tir_env, sample)

        request_input_ids = result.requests[0]["input_ids"]
        plain_ids = TOKENIZER("What is 1+7?", add_special_tokens=False)["input_ids"]
        assert request_input_ids == plain_ids
