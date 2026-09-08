"""Sample.SpecInfo aggregates the speculative-decoding counters SGLang reports.

The regression this pins: SGLang's tokenizer_manager emits `spec_num_correct_drafts`
and `spec_num_proposed_drafts`, while SpecInfo used to read `spec_accept_token_num` /
`spec_draft_token_num`. Those keys never appeared, so `spec_accept_rate` logged a
constant 0.0 — and the failure was easy to miss because `spec_accept_length`, derived
from `spec_verify_ct` + `completion_tokens`, kept reporting sane values (~4.0).
"""

import pytest

from miles.utils.types import Sample


def _sglang_meta(correct, proposed, verify_ct, completion):
    """meta_info as tokenizer_manager.collect_metrics builds it for a spec request."""
    return {
        "spec_accept_rate": correct / proposed if proposed else 0.0,
        "spec_accept_length": completion / verify_ct if verify_ct else 0.0,
        "spec_num_correct_drafts": correct,
        "spec_num_proposed_drafts": proposed,
        "spec_verify_ct": verify_ct,
        "spec_accepted_drafts": correct,  # backward-compat aliases sglang also sets
        "spec_proposed_drafts": proposed,
        "completion_tokens": completion,
    }


def test_accept_rate_from_sglang_keys():
    info = Sample.SpecInfo()
    # gamma=7 -> 6 proposed drafts per verify step; 3 of them accepted.
    info.add(_sglang_meta(correct=300, proposed=600, verify_ct=100, completion=400))
    assert info.spec_accept_rate == pytest.approx(0.5)
    assert info.spec_accept_length == pytest.approx(4.0)


def test_legacy_keys_still_work():
    info = Sample.SpecInfo()
    info.add(
        {
            "spec_accept_token_num": 120,
            "spec_draft_token_num": 300,
            "spec_verify_ct": 50,
            "completion_tokens": 200,
        }
    )
    assert info.spec_accept_rate == pytest.approx(0.4)
    assert info.spec_accept_length == pytest.approx(4.0)


def test_accumulates_across_requests_and_survives_roundtrip():
    info = Sample.SpecInfo()
    info.add(_sglang_meta(100, 200, 40, 160))
    info.add(_sglang_meta(50, 200, 40, 120))
    assert info.spec_accept_rate == pytest.approx(150 / 400)
    assert info.spec_accept_length == pytest.approx(280 / 80)

    # partial-rollout / buffer round-trip must not lose the counters
    restored = Sample.SpecInfo.from_dict(info.to_dict())
    assert restored.spec_accept_rate == pytest.approx(info.spec_accept_rate)
    assert restored.spec_accept_length == pytest.approx(info.spec_accept_length)


def test_no_spec_fields_is_zero_not_error():
    info = Sample.SpecInfo()
    info.add({"completion_tokens": 128})
    assert info.spec_accept_rate == 0.0
    assert info.spec_accept_length == 0.0
