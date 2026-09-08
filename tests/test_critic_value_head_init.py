"""Regression tests for the critic value-head initialization (needs a megatron env).

Two invariants:
1. The scalar value head (output_size=1) must be zero-initialized by default.
   normal(0, 0.02) on hidden=2048 yields V ~ ±9.7 while the regression targets live
   in a ~±0.35 band; at critic_lr 5e-6 that offset needs ~90 steps to close, which is
   more than any warmup budget, so the actor would train on advantages that are
   mostly noise. With a zero weight the advantage degrades exactly to the critic-free
   form, which is the failure-safe property the recipe relies on. (train.sh seeds the
   *bias* at the expected mean reward on top of this; see --critic-value-bias-init.)
2. Non-scalar output layers (the vocabulary head) are unaffected and keep normal init.

Both only show up on **instantiation**: an earlier version wrote out_features where
output_size was meant, passed py_compile and bash -n, and then died with NameError as
soon as the model was actually built.
"""

from types import SimpleNamespace

import pytest

from miles.backends.megatron_utils.model_provider import LinearForLastLayer


@pytest.fixture
def cfg():
    return SimpleNamespace(sequence_parallel=False)


def test_scalar_value_head_is_zero_initialized(cfg):
    head = LinearForLastLayer(input_size=2048, output_size=1, config=cfg)
    assert float(head.weight.detach().abs().max()) == 0.0
    assert float(head.bias.detach().abs().max()) == 0.0


def test_non_scalar_head_keeps_normal_init(cfg):
    head = LinearForLastLayer(input_size=2048, output_size=128, config=cfg)
    assert float(head.weight.detach().abs().max()) > 0.0


def test_value_head_carries_output_parameter_flag(cfg):
    """muon routes 2D parameters without this flag into Newton-Schulz orthogonalization,
    whose update magnitude is error-independent — a [1, H] value head would then oscillate
    at fixed amplitude around the optimum instead of converging."""
    head = LinearForLastLayer(input_size=2048, output_size=1, config=cfg)
    assert getattr(head.weight, "is_embedding_or_output_parameter", False) is True
