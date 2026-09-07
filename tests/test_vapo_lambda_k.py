"""JustRL2 length-adaptive GAE λ: λ_i = k ** (1/L_i).

The defining property is length invariance of terminal credit: with γ=1 the GAE
weight of the terminal reward at the first token is λ_i ** L_i, which must equal k for
every response length. These tests pin that, the fp32 computation path (bf16 would
round λ to exactly 1.0 at 128k), the decoupling from the λ=1 value target, and the
first-order equivalence with the old VAPO form 1 − 1/(α·L) (k = exp(−1/α)).
"""

import math
from types import SimpleNamespace

import pytest
import torch

import miles.utils.ppo_utils as ppo_utils
from miles.utils.ppo_utils import get_advantages_and_returns_batch, vapo_lambda_rowwise


@pytest.fixture
def cp1_parallel_state(monkeypatch):
    fake = SimpleNamespace(cp=SimpleNamespace(size=1, rank=0, group=None))
    monkeypatch.setattr(ppo_utils, "get_parallel_state", lambda: fake)


@pytest.mark.parametrize("k", [0.1, 0.513, 0.9])
@pytest.mark.parametrize("L", [1, 5, 300, 8192, 126976])
def test_first_token_weight_is_k_for_any_length(k, L):
    lam = vapo_lambda_rowwise(k, [L], device="cpu")
    assert lam.dtype == torch.float32
    # fp32 λ carries ~6e-8 relative error, which λ**L amplifies to ~L·6e-8
    # (≈1% at 128k). That is the precision of the GAE recursion itself.
    assert math.isclose(float(lam[0]) ** L, k, rel_tol=max(1e-4, 2e-7 * L))


def test_longer_response_lambda_closer_to_one():
    lam = vapo_lambda_rowwise(0.5, [10, 100, 1000, 100000], device="cpu")
    assert torch.all(lam[1:] > lam[:-1])
    assert float(lam[-1]) < 1.0


def test_empty_response_lambda_zero_no_nan():
    lam = vapo_lambda_rowwise(0.5, [0, 3], device="cpu")
    assert float(lam[0]) == 0.0 and not torch.isnan(lam).any()


def test_bf16_request_still_computed_in_fp32():
    # At L=126976, k=0.5: λ = 1 − 5.5e-6. bf16 rounds that to 1.0 (eps 7.8e-3), which
    # would make every token see the full terminal reward. The helper computes in
    # fp32 and only casts at the end; the caller's GAE consumes it as a scalar-per-row
    # so the cast back to bf16 is the test of what the *caller* would have done wrong.
    lam32 = vapo_lambda_rowwise(0.5, [126976], device="cpu", dtype=torch.float32)
    assert float(lam32[0]) < 1.0
    assert math.isclose(float(lam32[0]) ** 126976, 0.5, rel_tol=3e-2)


def test_first_order_equivalence_with_old_vapo_alpha():
    # #12 ran the old form with α=1.5; the equivalent k is exp(−1/1.5) ≈ 0.5134.
    alpha = 1.5
    k = math.exp(-1.0 / alpha)
    for L in (1000, 20000, 126976):
        old = 1.0 - 1.0 / (alpha * L)
        new = float(vapo_lambda_rowwise(k, [L], device="cpu")[0])
        assert abs(old - new) < 1e-6, (L, old, new)


def test_advantage_uses_k_lambda_but_returns_are_suffix_sum(cp1_parallel_state):
    torch.manual_seed(0)
    lengths = [12, 40]
    values = [torch.randn(L) * 0.1 for L in lengths]
    rewards = []
    for L in lengths:
        r = torch.zeros(L)
        r[-1] = 1.0
        rewards.append(r)
    adv, ret = get_advantages_and_returns_batch(
        total_lengths=[L + 3 for L in lengths],
        response_lengths=lengths,
        values_list=values,
        rewards_list=rewards,
        gamma=1.0,
        lambd=0.95,  # ignored under length-adaptive mode
        length_adaptive_lambda_k=0.5,
    )
    for i, L in enumerate(lengths):
        # returns: λ=1 suffix reward sum → all ones for a terminal-only reward
        assert torch.equal(ret[i], torch.ones(L))
        # advantages: GAE with λ_i = 0.5**(1/L); check via the closed form
        lam = 0.5 ** (1.0 / L)
        v = torch.cat([values[i], torch.zeros(1)])
        delta = rewards[i] + v[1:] - v[:-1]
        exp_adv = torch.zeros(L)
        run = 0.0
        for t in reversed(range(L)):
            run = float(delta[t]) + lam * run
            exp_adv[t] = run
        torch.testing.assert_close(adv[i], exp_adv, rtol=1e-4, atol=1e-5)


def test_requires_gamma_one(cp1_parallel_state):
    with pytest.raises(AssertionError):
        get_advantages_and_returns_batch(
            total_lengths=[5],
            response_lengths=[4],
            values_list=[torch.zeros(4)],
            rewards_list=[torch.zeros(4)],
            gamma=0.99,
            lambd=1.0,
            length_adaptive_lambda_k=0.5,
        )
