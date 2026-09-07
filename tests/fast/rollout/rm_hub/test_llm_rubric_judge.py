import pytest
from unittest.mock import MagicMock

from miles.rollout.rm_hub import async_rm
from miles.rollout.rm_hub.llm_judge import (
    _load_template,
    _parse_rubric_judge_json,
    _rubric_metadata_lookup,
    _rubric_reward_from_payload,
    llm_rubric_judge_reward,
)
from miles.utils.async_utils import run
from miles.utils.test_utils.mock_judge_server import with_mock_judge_server
from miles.utils.types import Sample

RUBRIC_TEMPLATE = """\
Q: {question_prompt}
Ref: {answer}
Points: {score_point}
Model: {predictions}
N: {max_score}
"""


@pytest.fixture
def rubric_template_file(tmp_path):
    path = tmp_path / "rubric.txt"
    path.write_text(RUBRIC_TEMPLATE, encoding="utf-8")
    return str(path)


@pytest.fixture
def rubric_mock_args(rubric_template_file):
    args = MagicMock()
    args.custom_rm_path = None
    args.rm_type = "llm_rubric_judge"
    args.rm_url = None
    args.judge_prompt_template = rubric_template_file
    args.judge_model = "mock-rubric"
    args.judge_api_key = None
    args.judge_proxy = None
    args.judge_kwargs = "temperature=0,max_tokens=4096"
    return args


class TestRubricMetadataLookup:
    def test_flat_extra_info_style(self):
        m = {"score_point": "1. a; 2. b", "max_score": 2}
        assert _rubric_metadata_lookup(m) is m

    def test_nested_extra_info(self):
        m = {"extra_info": {"score_point": "x", "max_score": 3}}
        assert _rubric_metadata_lookup(m) == {"score_point": "x", "max_score": 3}

    def test_non_dict(self):
        assert _rubric_metadata_lookup(None) == {}


class TestParseRubricJson:
    def test_fenced(self):
        text = 'Here is the result:\n```json\n{"point_scores": [], "total_score": 0}\n```\n'
        d = _parse_rubric_judge_json(text)
        assert d == {"point_scores": [], "total_score": 0}

    def test_raw_object(self):
        d = _parse_rubric_judge_json('prefix {"point_scores": [], "total_score": 1} suffix')
        assert d["total_score"] == 1


class TestRubricRewardFromPayload:
    def test_normalized_by_max_score(self):
        data = {
            "point_scores": [
                {"point": 1, "score": 1},
                {"point": 2, "score": 0},
            ],
            "total_score": 1,
        }
        r, err = _rubric_reward_from_payload(data, 2)
        assert err is None
        assert r == pytest.approx(0.5)

    def test_sum_when_total_missing(self):
        data = {"point_scores": [{"score": 1}, {"score": 1}, {"score": 0}]}
        r, err = _rubric_reward_from_payload(data, 3)
        assert err is None
        assert r == pytest.approx(2 / 3)

    def test_falls_back_to_len_point_scores(self):
        data = {"point_scores": [{"score": 1}], "total_score": 1}
        r, err = _rubric_reward_from_payload(data, 0)
        assert err is None
        assert r == pytest.approx(1.0)


class TestLlmRubricJudgeIntegration:
    def test_full_score(self, rubric_mock_args):
        _load_template.cache_clear()

        def judge_fn(_user_message: str) -> str:
            return (
                '{"point_scores": [{"point": 1, "score": 1}, {"point": 2, "score": 1}], '
                '"total_score": 2}'
            )

        with with_mock_judge_server(judge_fn=judge_fn) as server:
            rubric_mock_args.rm_url = server.chat_completions_url
            sample = Sample(
                prompt="题目",
                response="模型答",
                label="参考答案",
                metadata={"score_point": "p1; p2", "max_score": 2},
            )
            reward = run(llm_rubric_judge_reward(rubric_mock_args, sample))
            assert reward == pytest.approx(1.0)
            assert len(server.request_log) == 1

    def test_parse_failure_marks_remove_sample(self, rubric_mock_args):
        _load_template.cache_clear()

        with with_mock_judge_server(judge_fn=lambda _: "not json") as server:
            rubric_mock_args.rm_url = server.chat_completions_url
            sample = Sample(
                prompt="Q",
                response="A",
                label="L",
                metadata={"score_point": "x", "max_score": 1},
            )
            reward = run(llm_rubric_judge_reward(rubric_mock_args, sample))
            assert reward == 0.0
            assert sample.remove_sample is True

    def test_via_async_rm(self, rubric_mock_args):
        _load_template.cache_clear()

        with with_mock_judge_server(
            judge_fn=lambda _: '{"point_scores": [{"score": 1}], "total_score": 1}'
        ) as server:
            rubric_mock_args.rm_url = server.chat_completions_url
            rubric_mock_args.rm_type = "llm_rubric_judge"
            sample = Sample(
                prompt="Q",
                response="A",
                label="L",
                metadata={"score_point": "one point", "max_score": 1},
            )
            reward = run(async_rm(rubric_mock_args, sample))
            assert reward == pytest.approx(1.0)
