"""--olp-analytic-inject (臂#14, cleancritic 2.0) 的单元测试。

覆盖:
- 注入权重的数学正确性 (λ_inj 值与逐位衰减, 含 clamp 到 0 的短样本分支);
- P=0 样本逐字节零改动;
- padding 区零改动;
- 走 get_advantages_and_returns_batch 全链路 (cp_size=1): 注入差值精确等于解析项,
  returns 完全不动 (vanilla/chunked 与长度自适应解耦两条路径), n=1 与 n=8 两种形状。

CP>1 的分布式一致性无法在单进程单测里构造 (需要 torch.distributed 多进程 + megatron
风格 parallel state + zigzag 切分工具链), 见文件末尾的 skip 用例说明; 结构上注入发生在
全长坐标视图、CP 切分之前, 各 rank 在 allreduce 后持有相同的 full_advantages, 注入是
确定性逐元素运算, 逐 rank 一致性由构造保证。
"""

import math
from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

import miles.utils.ppo_utils as ppo_utils
from miles.utils.ppo_utils import (
    apply_olp_analytic_injection,
    compute_overlong_penalty,
    get_advantages_and_returns_batch,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _expected_weights(length: int, alpha: float, max_len: int) -> torch.Tensor:
    """逐位参考实现 (python 循环, 只用于测试): w[t] = λ^(L-1-t), t < L, 其余 0。"""
    lam = max(0.0, 1.0 - 1.0 / (alpha * length)) if length > 0 else 0.0
    w = torch.zeros(max_len, dtype=torch.float32)
    for t in range(length):
        w[t] = lam ** (length - 1 - t)
    return w


@pytest.fixture
def cp1_parallel_state(monkeypatch):
    """单进程下把 ppo_utils 的 get_parallel_state 换成 cp_size=1 的假状态。"""
    fake = SimpleNamespace(cp=SimpleNamespace(size=1, rank=0, group=None))
    monkeypatch.setattr(ppo_utils, "get_parallel_state", lambda: fake)


# ---------------------------------------------------------------------------
# ① 注入权重的数学正确性
# ---------------------------------------------------------------------------


def test_lambda_inj_value_and_decay_L100_alpha01():
    """L=100, α_inj=0.1 → λ_inj = 1 - 1/(0.1*100) = 0.9, 逐位权重 0.9^(99-t)。"""
    L, alpha, P = 100, 0.1, -1.0
    base = torch.zeros(1, L)
    out = apply_olp_analytic_injection(base, [L], [P], alpha)

    lam = 1.0 - 1.0 / (alpha * L)
    assert math.isclose(lam, 0.9, rel_tol=0, abs_tol=1e-12)
    expected = P * _expected_weights(L, alpha, L).unsqueeze(0)
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-6)

    # 终点 token (d=0) 权重恒为 1: 恰好吃到完整 P
    assert math.isclose(out[0, L - 1].item(), P, rel_tol=0, abs_tol=1e-7)
    # 逐位衰减: 每往前一个 token 幅度缩 λ 倍 (惩罚为负, 幅度用 abs 比)
    ratios = out[0, 1:] / out[0, :-1]
    torch.testing.assert_close(ratios, torch.full((L - 1,), 1.0 / lam), rtol=1e-4, atol=1e-5)
    # α_inj=0.1 语义: 距终点 10%·L 处权重已衰减到 λ^(0.1L) = 0.9^10 ≈ 0.35
    d = int(0.1 * L)
    assert math.isclose(out[0, L - 1 - d].item(), P * lam**d, rel_tol=1e-5, abs_tol=0)


def test_lambda_inj_clamped_to_zero_short_sample():
    """α_inj·L ≤ 1 时 λ_inj clamp 到 0: 只有最后一个 token 吃到 P (0^0=1)。"""
    L, alpha, P = 5, 0.1, -0.8  # α·L = 0.5 → 1 - 1/0.5 = -1 → clamp 0
    base = torch.zeros(1, L)
    out = apply_olp_analytic_injection(base, [L], [P], alpha)
    expected = torch.tensor([[0.0, 0.0, 0.0, 0.0, P]])
    torch.testing.assert_close(out, expected, rtol=0, atol=1e-7)


# ---------------------------------------------------------------------------
# ② P=0 样本零改动 / padding 区零改动
# ---------------------------------------------------------------------------


def test_zero_penalty_sample_bitwise_unchanged():
    torch.manual_seed(0)
    max_len = 64
    base = torch.randn(2, max_len)
    out = apply_olp_analytic_injection(base, [64, 50], [0.0, -0.5], 0.1)
    # P=0 的行逐字节不变
    assert torch.equal(out[0], base[0])
    # P≠0 的行确实变了
    assert not torch.equal(out[1], base[1])


def test_padding_region_untouched():
    torch.manual_seed(1)
    max_len = 16
    lengths = [5, 16]
    base = torch.randn(2, max_len)
    out = apply_olp_analytic_injection(base, lengths, [-1.0, -1.0], 0.5)
    # 样本 0 的 padding 区 [5, 16) 不动
    assert torch.equal(out[0, 5:], base[0, 5:])
    # 响应区 [0, 5) 全部有非零注入 (λ = 1 - 1/(0.5*5) = 0.6 > 0)
    assert torch.all(out[0, :5] != base[0, :5])


def test_empty_response_no_nan():
    """L=0 (空响应, 立即 EOS): 1/(α·0)=inf 路径不得产生 NaN, 整行不动。"""
    base = torch.randn(2, 8)
    out = apply_olp_analytic_injection(base, [0, 8], [0.0, -1.0], 0.1)
    assert not torch.isnan(out).any()
    assert torch.equal(out[0], base[0])


def test_bf16_full_length_weights_computed_in_fp32():
    """bf16 + 128k 长度的回归测试 (评审 CONCERN-A)。

    权重若在 bf16 下计算: λ = 1 − 7.9e-5 会被舍入成 1.0, t_idx/lengths 的大整数
    (>256) 无法精确表示导致终点边界比较错判——实测 L=126976 时 w_end=0、其余全 −1
    的灾难性翻转。加固后权重固定在 fp32 计算、注入项再 cast 回 bf16:
    终点权重恒 1 (w_end = P = −1.0 精确), 远端权重 λ^d 衰减到 ~4.5e-5。
    """
    L, alpha, P = 126976, 0.1, -1.0  # 126976 = 128k recipe 的 rollout_max_response_len
    base = torch.zeros(1, L, dtype=torch.bfloat16)
    out = apply_olp_analytic_injection(base, [L], [P], alpha)

    assert out.dtype == torch.bfloat16
    # 终点 token: d=0 → 权重恰为 1, 注入恰为 P (bf16 精确表示 −1.0)
    assert out[0, -1].item() == -1.0
    # 起点 token: d=L−1 → λ^d ≈ exp(−10) ≈ 4.5e-5, 绝不该是 −1 (bf16 翻转的症状)
    assert out[0, 0].abs().item() < 1e-3
    # 整行与 "fp32 计算 → cast bf16" 的参考实现逐元素一致
    lam32 = torch.clamp(1.0 - 1.0 / (alpha * torch.tensor([float(L)], dtype=torch.float32)), min=0.0)
    d32 = torch.arange(L - 1, -1, -1, dtype=torch.float32)  # d = L-1-t, t ∈ [0, L)
    ref = (P * lam32**d32).to(torch.bfloat16)
    torch.testing.assert_close(out[0], ref, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# rollout 侧: 开关开时 OLP 不进 shaped_rewards / 非标量 reward 硬报错
# (miles.ray.rollout 依赖 sglang, 本地纯 CPU 环境缺依赖时跳过; CPU CI 会跑)
# ---------------------------------------------------------------------------


class _FakeSample:
    def __init__(self, reward, response_length):
        self._reward = reward
        self.response_length = response_length

    def get_reward_value(self, args):
        return self._reward


def _make_rollout_args(**overrides):
    defaults = dict(
        overlong_buffer_len=10,
        rollout_max_response_len=100,
        overlong_penalty_factor=1.0,
        olp_analytic_inject=False,
        length_reward_weight=0.0,
        advantage_estimator="ppo",
        rewards_normalization=False,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def _post_process_rewards(args, samples):
    rollout_mod = pytest.importorskip(
        "miles.ray.rollout",
        reason="miles.ray.rollout 依赖 sglang, 本地环境缺依赖时跳过 (CPU CI 有 stub 可跑)",
    )
    fake_self = SimpleNamespace(args=args, custom_reward_post_process_func=None)
    return rollout_mod.RolloutManager._post_process_rewards(fake_self, samples)


def test_rollout_olp_applied_to_shaped_rewards_when_inject_off():
    samples = [_FakeSample(1.0, 95), _FakeSample(0.0, 50)]  # L=95 → penalty −0.5; L=50 → 0
    raw, shaped = _post_process_rewards(_make_rollout_args(), samples)
    assert raw == [1.0, 0.0]
    assert shaped[0] == pytest.approx(0.5)
    assert shaped[1] == pytest.approx(0.0)


def test_rollout_olp_skipped_in_shaped_rewards_when_inject_on():
    """臂#14 核心: 开关开时 OLP 不加进 shaped_rewards, reward 通道全净。"""
    samples = [_FakeSample(1.0, 95), _FakeSample(0.0, 100)]  # 两个都在惩罚带内
    raw, shaped = _post_process_rewards(_make_rollout_args(olp_analytic_inject=True), samples)
    assert raw == [1.0, 0.0]
    assert shaped == raw  # 逐值等于 raw, 无任何 OLP shaping


def test_rollout_non_scalar_reward_with_inject_raises():
    samples = [_FakeSample({"score": 1.0}, 95)]
    with pytest.raises(ValueError, match="scalar rewards"):
        _post_process_rewards(_make_rollout_args(olp_analytic_inject=True), samples)


# ---------------------------------------------------------------------------
# ③ 全链路 (cp_size=1) — 注入差值精确等于解析项, returns 不动
# ---------------------------------------------------------------------------


def _run_batch(lengths, penalties, alpha, lam_k=None, seed=42):
    torch.manual_seed(seed)
    values = [torch.randn(L) for L in lengths]
    rewards = [torch.randn(L) * 0.1 for L in lengths]
    kwargs = dict(
        total_lengths=[L + 7 for L in lengths],  # prompt_len=7, 注入只看 response 坐标
        response_lengths=lengths,
        values_list=values,
        rewards_list=[r.clone() for r in rewards],
        gamma=1.0,
        lambd=0.95,
        length_adaptive_lambda_k=lam_k,
    )
    adv0, ret0 = get_advantages_and_returns_batch(**kwargs)  # 无注入基线
    adv1, ret1 = get_advantages_and_returns_batch(
        **kwargs, olp_inject_penalties=penalties, olp_inject_alpha=alpha
    )
    return adv0, ret0, adv1, ret1


@pytest.mark.parametrize("lengths", [[100], [100, 3, 77, 128, 1, 64, 100, 55]], ids=["n1", "n8"])
@pytest.mark.parametrize("lam_k", [None, 0.5], ids=["plain_gae", "length_adaptive"])
def test_batch_injection_matches_analytic_and_returns_untouched(
    cp1_parallel_state, lengths, lam_k
):
    alpha = 0.1
    penalties = [(-1.0 if i % 2 == 0 else 0.0) for i in range(len(lengths))]
    adv0, ret0, adv1, ret1 = _run_batch(lengths, penalties, alpha, lam_k=lam_k)

    for i, L in enumerate(lengths):
        expected_delta = penalties[i] * _expected_weights(L, alpha, L)
        torch.testing.assert_close(adv1[i] - adv0[i], expected_delta, rtol=1e-4, atol=1e-5)
        if penalties[i] == 0.0:
            # P=0 样本 advantage 逐字节不变
            assert torch.equal(adv1[i], adv0[i])
        # returns (critic 回归目标) 两种 GAE 路径下都完全不动
        assert torch.equal(ret1[i], ret0[i])


# ---------------------------------------------------------------------------
# compute_overlong_penalty 口径 (惩罚为负, 与 rollout 侧共用同一实现)
# ---------------------------------------------------------------------------


def test_compute_overlong_penalty_sign_and_band():
    args = Namespace(overlong_buffer_len=10, rollout_max_response_len=100, overlong_penalty_factor=1.0)
    assert compute_overlong_penalty(args, 90) == 0.0  # 安全区边界
    assert math.isclose(compute_overlong_penalty(args, 95), -0.5, rel_tol=0, abs_tol=1e-12)
    assert compute_overlong_penalty(args, 100) == -1.0  # 顶格
    assert compute_overlong_penalty(args, 1000) == -1.0  # clamp 在 -factor
    args_off = Namespace(overlong_buffer_len=0, rollout_max_response_len=100, overlong_penalty_factor=1.0)
    assert compute_overlong_penalty(args_off, 100) == 0.0


# ---------------------------------------------------------------------------
# ④ CP>1 一致性 — 单进程单测无法构造, 标注 skip
# ---------------------------------------------------------------------------


@pytest.mark.skip(
    reason=(
        "CP>1 需要 torch.distributed 多进程 + megatron 风格 parallel state + zigzag 切分工具"
        "(cp_utils.local_response_to_full / slice_log_prob_with_cp), 单进程单测环境无法构造。"
        "结构性保证: 注入发生在 get_advantages_and_returns_batch 的全长坐标视图上——CP 各 rank"
        "在 values/rewards allreduce 之后持有相同的 full_advantages, 注入是确定性逐元素运算,"
        "逐 rank 结果一致, 之后才统一走 slice_log_prob_with_cp 切分; cp_size=1 的全链路等价性"
        "已由 test_batch_injection_matches_analytic_and_returns_untouched 覆盖。"
    )
)
def test_cp_gt1_consistency():
    raise NotImplementedError
