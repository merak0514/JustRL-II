import pytest
from unittest.mock import MagicMock

from miles.rollout.rm_hub import async_rm, batched_async_rm
from miles.rollout.rm_hub.llm_judge import (
    _compute_reward,
    _format_prompt_as_text,
    _get_score,
    _load_template,
    extract_non_reasoning_content,
    llm_judge_reward,
    parse_judge_kwargs,
)
from miles.utils.async_utils import run
from miles.utils.test_utils.mock_judge_server import with_mock_judge_server
from miles.utils.types import Sample

TEMPLATE_CONTENT = """\
---SYSTEM---
You are a judge.
---USER---
<|User Prompt|>
{prompt}

<|The Start of Assistant A's Answer|>
{response_a}
<|The End of Assistant A's Answer|>

<|The Start of Assistant B's Answer|>
{response_b}
<|The End of Assistant B's Answer|>"""


@pytest.fixture
def template_file(tmp_path):
    path = tmp_path / "judge_template.txt"
    path.write_text(TEMPLATE_CONTENT, encoding="utf-8")
    return str(path)


@pytest.fixture
def mock_args(template_file):
    args = MagicMock()
    args.custom_rm_path = None
    args.rm_type = "llm_judge"
    args.rm_url = None
    args.judge_prompt_template = template_file
    args.judge_model = "mock-judge"
    args.judge_api_key = None
    args.judge_proxy = None
    args.judge_kwargs = "temperature=0.6,max_tokens=32768"
    return args


class TestGetScore:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ('My final verdict is: [[A>>B]]', "A>>B"),
            ('I think [[A>B]] is correct', "A>B"),
            ("[[A=B]]", "A=B"),
            ("The verdict: [[B>A]]", "B>A"),
            ("[[B>>A]]", "B>>A"),
            ('First [[A>B]] then [[B>>A]]', "B>>A"),
            ("- [A>>B]", "A>>B"),
            ("No verdict here", None),
            ("", None),
        ],
    )
    def test_get_score(self, text, expected):
        assert _get_score(text) == expected

    def test_case_insensitive(self):
        assert _get_score("[[a>>b]]") == "A>>B"
        assert _get_score("[[b>a]]") == "B>A"


class TestComputeReward:
    def test_model_wins_both_rounds(self):
        # Round1: A=model, judge says A>>B → model wins
        # Round2: A=ref, judge says A<<B → model (=B) wins
        reward = _compute_reward("A>>B", "A<<B")
        assert reward == pytest.approx(1.0)

    def test_reference_wins_both_rounds(self):
        reward = _compute_reward("A<<B", "A>>B")
        assert reward == pytest.approx(0.0)

    def test_tie_both_rounds(self):
        reward = _compute_reward("A=B", "A=B")
        assert reward == pytest.approx(0.5)

    def test_position_bias_cancels(self):
        # Judge always says A>>B regardless of position → should cancel out
        reward = _compute_reward("A>>B", "A>>B")
        assert reward == pytest.approx(0.5)

    def test_slight_model_win(self):
        # Round1: A=model, A>B; Round2: A=ref, A<B (=B>A, model wins)
        reward = _compute_reward("A>B", "A<B")
        assert reward == pytest.approx(1.0)

    def test_b_relative_verdicts(self):
        # Round1: B>A (ref wins slightly); Round2: B<A (model=B wins slightly)
        reward = _compute_reward("B>A", "B<A")
        assert reward == pytest.approx(0.0)

    def test_none_scores_return_zero(self):
        assert _compute_reward(None, None) == 0.0
        assert _compute_reward("A>>B", None) == 0.0
        assert _compute_reward(None, "A>>B") == 0.0


class TestExtractNonReasoningContent:
    def test_strips_think_blocks(self):
        text = "<think>reasoning here</think>The answer is 42."
        assert extract_non_reasoning_content(text) == "The answer is 42."

    def test_strips_multiline_think(self):
        text = "<think>\nstep 1\nstep 2\n</think>\nFinal answer."
        assert extract_non_reasoning_content(text) == "Final answer."

    def test_no_think_block(self):
        text = "Just a normal response."
        assert extract_non_reasoning_content(text) == "Just a normal response."

    def test_empty_after_strip(self):
        text = "<think>only reasoning</think>"
        assert extract_non_reasoning_content(text) == ""


class TestParseJudgeKwargs:
    def test_basic(self):
        result = parse_judge_kwargs("temperature=0.6,max_tokens=32768")
        assert result == {"temperature": 0.6, "max_tokens": 32768}

    def test_none(self):
        assert parse_judge_kwargs(None) == {}

    def test_empty(self):
        assert parse_judge_kwargs("") == {}

    def test_string_value(self):
        result = parse_judge_kwargs("model=gpt-4,temperature=0.5")
        assert result == {"model": "gpt-4", "temperature": 0.5}


class TestFormatPrompt:
    def test_string_prompt(self):
        assert _format_prompt_as_text("What is 1+1?") == "What is 1+1?"

    def test_message_list_prompt(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "What is 1+1?"},
        ]
        result = _format_prompt_as_text(messages)
        assert "[user]: Hello" in result
        assert "[assistant]: Hi" in result
        assert "[user]: What is 1+1?" in result


class TestTemplateLoading:
    def test_load_system_user_template(self, template_file):
        _load_template.cache_clear()
        system, user = _load_template(template_file)
        assert system is not None
        assert "You are a judge" in system
        assert "{prompt}" in user
        assert "{response_a}" in user
        assert "{response_b}" in user

    def test_load_user_only_template(self, tmp_path):
        _load_template.cache_clear()
        path = tmp_path / "simple.txt"
        path.write_text("Question: {prompt}\nA: {response_a}\nB: {response_b}")
        system, user = _load_template(str(path))
        assert system is None
        assert "{prompt}" in user


class TestLlmJudgeWithMockServer:
    """Integration tests using the mock judge server."""

    def test_consistent_model_wins(self, mock_args):
        _load_template.cache_clear()

        def judge_fn(user_message: str) -> str:
            if "good model answer" in user_message.split("<|The Start of Assistant A's Answer|>")[1].split("<|The End of Assistant A's Answer|>")[0]:
                return "My verdict: [[A>>B]]"
            else:
                return "My verdict: [[A<<B]]"

        with with_mock_judge_server(judge_fn=judge_fn) as server:
            mock_args.rm_url = server.chat_completions_url
            sample = Sample(prompt="What is 1+1?", response="good model answer", label="bad reference")
            reward = run(llm_judge_reward(mock_args, sample))
            assert reward == pytest.approx(1.0)

    def test_consistent_reference_wins(self, mock_args):
        _load_template.cache_clear()

        def judge_fn(user_message: str) -> str:
            if "good reference" in user_message.split("<|The Start of Assistant A's Answer|>")[1].split("<|The End of Assistant A's Answer|>")[0]:
                return "[[A>>B]]"
            else:
                return "[[A<<B]]"

        with with_mock_judge_server(judge_fn=judge_fn) as server:
            mock_args.rm_url = server.chat_completions_url
            sample = Sample(prompt="What is 1+1?", response="bad model answer", label="good reference")
            reward = run(llm_judge_reward(mock_args, sample))
            assert reward == pytest.approx(0.0)

    def test_tie(self, mock_args):
        _load_template.cache_clear()

        with with_mock_judge_server(judge_fn=lambda _: "[[A=B]]") as server:
            mock_args.rm_url = server.chat_completions_url
            sample = Sample(prompt="What is 1+1?", response="answer", label="answer")
            reward = run(llm_judge_reward(mock_args, sample))
            assert reward == pytest.approx(0.5)

    def test_position_bias_cancelled(self, mock_args):
        _load_template.cache_clear()

        with with_mock_judge_server(judge_fn=lambda _: "[[A>>B]]") as server:
            mock_args.rm_url = server.chat_completions_url
            sample = Sample(prompt="Q", response="model", label="reference")
            reward = run(llm_judge_reward(mock_args, sample))
            assert reward == pytest.approx(0.5)

    def test_parse_failure_returns_zero(self, mock_args):
        _load_template.cache_clear()

        with with_mock_judge_server(judge_fn=lambda _: "I cannot decide") as server:
            mock_args.rm_url = server.chat_completions_url
            sample = Sample(prompt="Q", response="model", label="reference")
            reward = run(llm_judge_reward(mock_args, sample))
            assert reward == pytest.approx(0.0)
            assert sample.remove_sample is True

    def test_two_requests_sent(self, mock_args):
        _load_template.cache_clear()

        with with_mock_judge_server(judge_fn=lambda _: "[[A=B]]") as server:
            mock_args.rm_url = server.chat_completions_url
            sample = Sample(prompt="Q", response="model", label="ref")
            run(llm_judge_reward(mock_args, sample))
            assert len(server.request_log) == 2

    def test_system_message_sent(self, mock_args):
        _load_template.cache_clear()

        with with_mock_judge_server(judge_fn=lambda _: "[[A=B]]") as server:
            mock_args.rm_url = server.chat_completions_url
            sample = Sample(prompt="Q", response="A", label="R")
            run(llm_judge_reward(mock_args, sample))
            req = server.request_log[0]
            assert req["messages"][0]["role"] == "system"
            assert "You are a judge" in req["messages"][0]["content"]
            assert req["messages"][1]["role"] == "user"

    def test_judge_kwargs_passed(self, mock_args):
        _load_template.cache_clear()
        mock_args.judge_kwargs = "temperature=0.8,max_tokens=1024"

        with with_mock_judge_server(judge_fn=lambda _: "[[A=B]]") as server:
            mock_args.rm_url = server.chat_completions_url
            sample = Sample(prompt="Q", response="A", label="R")
            run(llm_judge_reward(mock_args, sample))
            req = server.request_log[0]
            assert req["temperature"] == 0.8
            assert req["max_tokens"] == 1024

    def test_think_tags_stripped(self, mock_args):
        _load_template.cache_clear()

        with with_mock_judge_server(judge_fn=lambda _: "[[A=B]]") as server:
            mock_args.rm_url = server.chat_completions_url
            sample = Sample(
                prompt="Q",
                response="<think>internal reasoning</think>The answer is 42.",
                label="<think>ref reasoning</think>The answer is 42.",
            )
            run(llm_judge_reward(mock_args, sample))
            req = server.request_log[0]
            user_content = req["messages"][1]["content"]
            assert "<think>" not in user_content
            assert "internal reasoning" not in user_content
            assert "The answer is 42." in user_content

    def test_via_async_rm(self, mock_args):
        _load_template.cache_clear()

        with with_mock_judge_server(judge_fn=lambda _: "[[A=B]]") as server:
            mock_args.rm_url = server.chat_completions_url
            sample = Sample(prompt="Q", response="A", label="R")
            reward = run(async_rm(mock_args, sample))
            assert reward == pytest.approx(0.5)

    def test_batched_llm_judge(self, mock_args):
        _load_template.cache_clear()

        with with_mock_judge_server(judge_fn=lambda _: "[[A=B]]") as server:
            mock_args.rm_url = server.chat_completions_url
            samples = [
                Sample(prompt="Q1", response="A1", label="R1"),
                Sample(prompt="Q2", response="A2", label="R2"),
                Sample(prompt="Q3", response="A3", label="R3"),
            ]
            rewards = run(batched_async_rm(mock_args, samples))
            assert len(rewards) == 3
            assert all(r == pytest.approx(0.5) for r in rewards)
            assert len(server.request_log) == 6

    def test_api_key_sent_in_header(self, mock_args):
        _load_template.cache_clear()
        mock_args.judge_api_key = "test-secret-key-123"

        with with_mock_judge_server(judge_fn=lambda _: "[[A=B]]") as server:
            mock_args.rm_url = server.chat_completions_url
            sample = Sample(prompt="Q", response="A", label="R")
            run(llm_judge_reward(mock_args, sample))
            for headers in server.header_log:
                assert headers.get("authorization") == "Bearer test-secret-key-123"

    def test_b_relative_verdict(self, mock_args):
        _load_template.cache_clear()

        with with_mock_judge_server(judge_fn=lambda _: "[[B>>A]]") as server:
            mock_args.rm_url = server.chat_completions_url
            sample = Sample(prompt="Q", response="model", label="reference")
            reward = run(llm_judge_reward(mock_args, sample))
            assert reward == pytest.approx(0.5)
