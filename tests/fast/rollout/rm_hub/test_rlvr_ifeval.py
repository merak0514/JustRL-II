"""Tests for RLVR-IFEval rule reward (verl stack: mmengine, opencompass, open_instruct)."""

import pytest

pytest.importorskip("mmengine", reason="OpenCompass utils require mmengine (verl-aligned rlvr_ifeval)")
pytest.importorskip(
    "opencompass.utils.text_postprocessors",
    reason="opencompass on PYTHONPATH + mmengine",
)
pytest.importorskip("open_instruct.ground_truth_utils", reason="open_instruct on PYTHONPATH")

from miles.rollout.rm_hub.rlvr_ifeval import compute_rlvr_ifeval_reward

_VALIDATE_LOWERCASE_GT = (
    '{"func_name": "validate_lowercase", "N": null, "quantifier": null, "end_phrase": null, '
    '"keyword_list": null, "word": null, "forbidden_words": null, "letter": null, "i": null, '
    '"first_word": null, "postscript_marker": null, "options": null, "section_splitter": null, '
    '"original_prompt": null}'
)


class TestRlvrIfevalReward:
    def test_lowercase_pass(self):
        score = compute_rlvr_ifeval_reward("ipv6 is the successor to ipv4.", _VALIDATE_LOWERCASE_GT)
        assert score == 1.0

    def test_lowercase_fail(self):
        score = compute_rlvr_ifeval_reward("IPv6 Uses Capitals", _VALIDATE_LOWERCASE_GT)
        assert score == 0.0

    def test_strips_reasoning_tags(self):
        score = compute_rlvr_ifeval_reward(
            "<think>thinking</think>ipv6 is the successor to ipv4.", _VALIDATE_LOWERCASE_GT
        )
        assert score == 1.0

    def test_none_label(self):
        assert compute_rlvr_ifeval_reward("hello", None) == 0.0

    def test_metadata_ignored(self):
        score = compute_rlvr_ifeval_reward(
            "ipv6 is the successor to ipv4.",
            _VALIDATE_LOWERCASE_GT,
            metadata={"extra": "ignored"},
        )
        assert score == 1.0
