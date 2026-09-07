"""--group-center-inject (臂#16, 组中心化注入) 的单元测试。

语义: PPO(VAPO)线上, 在 actor 的 advantage 通道做组内留一中心化——终端标量
a_j = raw_reward_j − V_j (该样本最后一个 loss-mask token 的 critic value), 注入量
P_i = −(Σ_{j∈g, j≠i} a_j)/(n_g−1), GAE 之后按 λ_i^(L_i−1−t) 衰减注入
(λ_i = k^(1/L_i), k = vapo_lambda_k); returns 不动,
critic 照学未中心化 return。

覆盖:
- 留一均值手算小案例 (n=3 与 n=8, 含 mask 尾零样本的 last-loss-mask-token 语义);
- n_g==1 守护 (P=0, 逐字节 no-op);
- 注入等价性: 注入后 advantage == 终端 reward 减去留一均值后重跑 GAE 的 advantage
  (GAE 对 reward 线性, γ=1 时终端脉冲的传播权重恰为 λ_i^(L−1−t)——与注入衰减同一公式);
- bf16 输入不炸且 fp32 中间计算 (长序列 λ 舍入翻转的回归);
- 空响应 (L=0) 与全零 mask 守护;
- 开关 off (group_center=None) 时输出与主线逐字节一致 (no-op 铁律回归);
- flat 位置散射的置换一致性 (单 rank 退化, 验证 DP scatter/allreduce 的索引数学);
- 非整组批 fail-fast;
- 与 --olp-analytic-inject 共存 (两个线性注入相加)。

DP>1 的真实 allreduce 与 CP>1 的分布式一致性无法在单进程单测里构造, 见文件末尾
skip 用例的结构性论证。
"""

from types import SimpleNamespace

import pytest
import torch

import miles.utils.ppo_utils as ppo_utils
from miles.utils.ppo_utils import get_advantages_and_returns_batch


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _expected_weights_alpha(length: int, alpha: float, max_len: int) -> torch.Tensor:
    """OLP 注入 (臂#14, α 参数化) 的逐位参考: w[t] = λ^(L-1-t), λ = clamp(1 − 1/(α·L), 0)。"""
    lam = max(0.0, 1.0 - 1.0 / (alpha * length)) if length > 0 else 0.0
    w = torch.zeros(max_len, dtype=torch.float32)
    for t in range(length):
        w[t] = lam ** (length - 1 - t)
    return w


def _expected_weights(length: int, k: float, max_len: int) -> torch.Tensor:
    """group-center 注入 (k 参数化, 与 GAE 同款 λ_i = k^(1/L)) 的逐位参考:
    w[t] = λ^(L-1-t), 首 token 权重恰为 λ^(L-1) = k^((L-1)/L)。"""
    lam = k ** (1.0 / length) if length > 0 else 0.0
    w = torch.zeros(max_len, dtype=torch.float32)
    for t in range(length):
        w[t] = lam ** (length - 1 - t)
    return w


def _reference_P(raws, values, lengths, masks, n_g):
    """纯 python 参考实现: a_j = raw_j − V_j[最后一个 loss-mask token], 留一均值取负。"""
    a = []
    for j in range(len(raws)):
        L = lengths[j]
        if L == 0:
            a.append(0.0)  # 空响应: a=0, 不污染兄弟基线
            continue
        nz = [t for t in range(len(masks[j])) if float(masks[j][t]) != 0.0]
        last = nz[-1] if nz else L - 1  # 全零 mask 回退终点
        a.append(float(raws[j]) - float(values[j][last]))
    P = []
    for i in range(len(raws)):
        if n_g == 1:
            P.append(0.0)
            continue
        g0 = (i // n_g) * n_g
        siblings = [a[j] for j in range(g0, g0 + n_g) if j != i]
        P.append(-sum(siblings) / (n_g - 1))
    return P


def _make_batch(lengths, dtype=torch.float32, seed=0, zero_base=False):
    torch.manual_seed(seed)
    if zero_base:
        values = [torch.zeros(L, dtype=dtype) for L in lengths]
        rewards = [torch.zeros(L, dtype=dtype) for L in lengths]
    else:
        values = [torch.randn(L, dtype=dtype) for L in lengths]
        rewards = [torch.randn(L, dtype=dtype) * 0.1 for L in lengths]
    masks = [torch.ones(L, dtype=torch.int) for L in lengths]
    return values, rewards, masks


def _gc_ctx(raws, n_g, k, masks, positions=None, stats=None):
    return dict(
        positions=positions if positions is not None else list(range(len(masks))),
        raw_rewards=raws,
        n_samples_per_prompt=n_g,
        k=k,
        loss_masks=masks,
        dp_group=None,
        stats_out=stats,
    )


def _run_pair(lengths, values, rewards, gc, vapo_k=0.5, gamma=1.0, lambd=0.95):
    """同一批数据跑两次: 无注入基线 vs group_center 注入。"""
    kwargs = dict(
        total_lengths=[L + 7 for L in lengths],  # prompt_len=7, 注入只看 response 坐标
        response_lengths=lengths,
        values_list=values,
        rewards_list=[r.clone() for r in rewards],
        gamma=gamma,
        lambd=lambd,
        length_adaptive_lambda_k=vapo_k,
    )
    adv0, ret0 = get_advantages_and_returns_batch(**kwargs)
    adv1, ret1 = get_advantages_and_returns_batch(**kwargs, group_center=gc)
    return adv0, ret0, adv1, ret1


@pytest.fixture
def cp1_parallel_state(monkeypatch):
    """单进程下把 ppo_utils 的 get_parallel_state 换成 cp_size=1 的假状态。"""
    fake = SimpleNamespace(cp=SimpleNamespace(size=1, rank=0, group=None))
    monkeypatch.setattr(ppo_utils, "get_parallel_state", lambda: fake)


# ---------------------------------------------------------------------------
# ① 留一均值手算小案例 (n=3 / n=8, 两条 GAE 路径)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n_g,lengths",
    [
        (3, [10, 7, 5, 40, 22, 13]),
        (8, [100, 3, 77, 128, 1, 64, 100, 55, 90, 10, 33, 47, 5, 120, 8, 60]),
    ],
    ids=["n3", "n8"],
)
@pytest.mark.parametrize("vapo_k", [None, 0.5], ids=["plain_gae", "vapo_decoupled"])
def test_loo_hand_computed(cp1_parallel_state, n_g, lengths, vapo_k):
    alpha = 0.5
    raws = [float(i % 2) for i in range(len(lengths))]  # 0/1 结果奖励
    values, rewards, masks = _make_batch(lengths, seed=42)
    # 样本 0 的 mask 尾部两位清零: 验证 a_j 取的是"最后一个 loss-mask token"的 V,
    # 而注入衰减仍锚定 L−1 (与 P 的取值口径解耦)
    masks[0][-2:] = 0
    stats = {}
    gc = _gc_ctx(raws, n_g, alpha, masks, stats=stats)
    adv0, ret0, adv1, ret1 = _run_pair(lengths, values, rewards, gc, vapo_k=vapo_k)

    P = _reference_P(raws, values, lengths, masks, n_g)
    for i, L in enumerate(lengths):
        expected_delta = P[i] * _expected_weights(L, alpha, L)
        torch.testing.assert_close(adv1[i] - adv0[i], expected_delta, rtol=1e-4, atol=1e-5)
        # returns (critic 回归目标) 完全不动
        assert torch.equal(ret1[i], ret0[i])
    # 统计回填: correction = P 列表, abs = |P| 列表
    assert stats["correction"] == pytest.approx(P, abs=1e-5)
    assert stats["abs"] == pytest.approx([abs(p) for p in P], abs=1e-5)


# ---------------------------------------------------------------------------
# ② n_g==1 守护: 无兄弟可减, P=0, 逐字节 no-op
# ---------------------------------------------------------------------------


def test_ng1_guard_noop(cp1_parallel_state):
    lengths = [12, 40]
    raws = [1.0, 0.0]
    values, rewards, masks = _make_batch(lengths, seed=1)
    stats = {}
    gc = _gc_ctx(raws, n_g=1, k=0.5, masks=masks, stats=stats)
    adv0, ret0, adv1, ret1 = _run_pair(lengths, values, rewards, gc)
    for i in range(len(lengths)):
        assert torch.equal(adv1[i], adv0[i])
        assert torch.equal(ret1[i], ret0[i])
    assert stats["correction"] == [0.0, 0.0]
    assert stats["abs"] == [0.0, 0.0]


# ---------------------------------------------------------------------------
# ③ 注入等价性: == 终端 reward 减去留一均值后重跑 GAE 的 advantage
# ---------------------------------------------------------------------------


def test_injection_equals_terminal_reward_shift_rerun_gae(cp1_parallel_state):
    """GAE 对 reward 线性, γ=1 时终端脉冲 δ 对 advantage_t 的贡献是 δ·λ_i^(L−1−t)——
    与注入衰减 (k=vapo_lambda_k, λ_i=k^(1/L_i)) 同一公式。故注入 P_i 必须精确等价于把终端
    reward 加 P_i (即减去留一均值 μ_i) 后重跑 GAE 的 advantage; 而 returns 若走重跑
    路径会被改动——这正是选择解析注入而非改 reward 的原因。"""
    n_g, alpha = 3, 0.5
    lengths = [40, 25, 33, 60, 5, 12]  # 2 组 × 3; L=5 是 λ=k^(1/5) 较小的短样本
    raws = [1.0, 0.0, 0.0, 1.0, 1.0, 0.0]
    values, rewards, masks = _make_batch(lengths, seed=7)
    gc = _gc_ctx(raws, n_g, alpha, masks)
    adv0, ret0, adv1, ret1 = _run_pair(lengths, values, rewards, gc, vapo_k=alpha)

    P = _reference_P(raws, values, lengths, masks, n_g)
    rewards_ref = [r.clone() for r in rewards]
    for i, L in enumerate(lengths):
        rewards_ref[i][L - 1] += P[i]
    adv_ref, ret_ref = get_advantages_and_returns_batch(
        total_lengths=[L + 7 for L in lengths],
        response_lengths=lengths,
        values_list=values,
        rewards_list=rewards_ref,
        gamma=1.0,
        lambd=0.95,
        length_adaptive_lambda_k=alpha,
    )
    for i in range(len(lengths)):
        torch.testing.assert_close(adv1[i], adv_ref[i], rtol=1e-5, atol=1e-5)
        # 注入路径的 returns 与无注入基线逐字节一致 (critic 照学未中心化 return)
        assert torch.equal(ret1[i], ret0[i])
        # 而"改 reward 重跑"路径的 returns 确实被改了 (P≠0 时)——注入的意义所在
        if P[i] != 0.0:
            assert not torch.equal(ret_ref[i], ret0[i])


# ---------------------------------------------------------------------------
# ④ bf16 输入不炸, 中间计算 fp32 (长序列 λ 舍入翻转回归, 仿 olp bf16 案例)
# ---------------------------------------------------------------------------


def test_bf16_long_length_fp32_intermediates(cp1_parallel_state):
    """L=8192, k=1e-4 → λ = k^(1/L) ≈ 1 − 1.1e-3; bf16 (8 位尾数) 会把它舍入成 1.0,
    若权重/中心化标量在 bf16 里算, 起点 token 会吃到完整 P (灾难性翻转)。fp32 中间
    计算下起点权重 λ^8191 ≈ k = 1e-4。values/rewards 全零 → GAE 基线恒 0,
    输出即注入项本身, 可精确断言。"""
    n_g, alpha, L = 2, 1e-4, 8192
    lengths = [L, L]
    raws = [1.0, 0.0]
    values, rewards, masks = _make_batch(lengths, dtype=torch.bfloat16, zero_base=True)
    gc = _gc_ctx(raws, n_g, alpha, masks)
    adv0, ret0, adv1, ret1 = _run_pair(lengths, values, rewards, gc, vapo_k=alpha)

    # a = raw − V[last] = raw (V≡0) → P_0 = −a_1 = 0, P_1 = −a_0 = −1
    assert adv1[0].dtype == torch.bfloat16
    for i in range(2):
        assert not torch.isnan(adv1[i].float()).any()
        assert torch.equal(ret1[i], ret0[i])
    # P=0 样本逐字节不变
    assert torch.equal(adv1[0], adv0[0])
    # P=−1 样本: 终点吃满 P (bf16 精确表示 −1.0), 起点 ≈ 0 (翻转症状是起点也 −1)
    assert adv1[1][-1].item() == -1.0
    assert abs(adv1[1][0].item()) < 1e-3


# ---------------------------------------------------------------------------
# ⑤ 空响应 (L=0) 与全零 mask 守护
# ---------------------------------------------------------------------------


def test_empty_response_guard(cp1_parallel_state):
    """L=0 (立即 EOS) 样本: a=0 (不用 raw−V, 因该行 full_values 全 0 且无有效终端),
    自身输出为空张量; 兄弟样本的 P 按 a=0 参与留一均值, 全程无 NaN。"""
    n_g, alpha = 3, 0.5
    lengths = [6, 0, 4]
    raws = [1.0, 1.0, 0.0]  # 空响应样本 raw=1.0, 但 a 必须取 0 而非 1.0−V
    values, rewards, masks = _make_batch(lengths, seed=5)
    stats = {}
    gc = _gc_ctx(raws, n_g, alpha, masks, stats=stats)
    adv0, ret0, adv1, ret1 = _run_pair(lengths, values, rewards, gc)

    P = _reference_P(raws, values, lengths, masks, n_g)  # 参考实现同样按 a[1]=0
    assert adv1[1].numel() == 0
    for i, L in enumerate(lengths):
        assert not torch.isnan(adv1[i]).any()
        expected_delta = P[i] * _expected_weights(L, alpha, L)
        torch.testing.assert_close(adv1[i] - adv0[i], expected_delta, rtol=1e-4, atol=1e-5)
        assert torch.equal(ret1[i], ret0[i])
    assert stats["correction"] == pytest.approx(P, abs=1e-5)


def test_all_zero_mask_fallback(cp1_parallel_state):
    """全零 loss mask (remove_sample/env_error/overlong_filtering 置零): last_idx 回退
    L−1, 样本仍以真实 raw−V[L−1] 贡献兄弟基线 (自身梯度已被 mask, 注入惰性无害)。"""
    n_g, alpha = 2, 0.5
    lengths = [5, 5]
    raws = [1.0, 0.0]
    values, rewards, masks = _make_batch(lengths, seed=6)
    masks[1][:] = 0  # 样本 1 全零 mask
    gc = _gc_ctx(raws, n_g, alpha, masks)
    adv0, ret0, adv1, ret1 = _run_pair(lengths, values, rewards, gc)

    P = _reference_P(raws, values, lengths, masks, n_g)  # 参考实现同样回退 L−1
    assert P[0] == pytest.approx(-(0.0 - float(values[1][4])), abs=1e-6)
    for i, L in enumerate(lengths):
        expected_delta = P[i] * _expected_weights(L, alpha, L)
        torch.testing.assert_close(adv1[i] - adv0[i], expected_delta, rtol=1e-4, atol=1e-5)
        assert torch.equal(ret1[i], ret0[i])


# ---------------------------------------------------------------------------
# ⑥ 开关 off: group_center=None 与不传参数的主线调用逐字节一致 (no-op 铁律)
# ---------------------------------------------------------------------------


def test_off_is_bitwise_noop(cp1_parallel_state):
    lengths = [30, 17, 55, 8]
    values, rewards, _ = _make_batch(lengths, seed=9)
    kwargs = dict(
        total_lengths=[L + 7 for L in lengths],
        response_lengths=lengths,
        values_list=values,
        rewards_list=[r.clone() for r in rewards],
        gamma=1.0,
        lambd=0.95,
        length_adaptive_lambda_k=0.5,
    )
    adv_a, ret_a = get_advantages_and_returns_batch(**kwargs)  # 主线 (不传新参数)
    adv_b, ret_b = get_advantages_and_returns_batch(**kwargs, group_center=None)
    for i in range(len(lengths)):
        assert torch.equal(adv_a[i], adv_b[i])
        assert torch.equal(ret_a[i], ret_b[i])


# ---------------------------------------------------------------------------
# ⑦ flat 位置散射: 置换一致性 (单 rank 退化, 验证 DP scatter/allreduce 索引数学)
# ---------------------------------------------------------------------------


def test_scatter_positions_permutation_consistent(cp1_parallel_state):
    """模拟 balance_data 把组成员打散后的本地乱序: 本地序 ≠ flat 序, positions 提供
    映射。单进程持有全量数据即 DP 退化情形——散射进 [N_global] buffer 后按 flat 组号
    (p // n_g) 求组和, 数学上与多 rank 各持互斥 partition 再 allreduce(SUM) 等价
    (partition 互斥且覆盖全批, SUM 恰好拼出完整 a 向量)。断言: 任意本地排列下,
    每个 flat 样本的注入结果与自然序逐字节一致。"""
    n_g, alpha = 3, 0.5
    flat_lengths = [10, 20, 30, 40, 50, 60]
    flat_raws = [1.0, 0.0, 1.0, 0.0, 1.0, 1.0]
    flat_values, flat_rewards, flat_masks = _make_batch(flat_lengths, seed=3)

    # 自然序 (positions = identity)
    gc_id = _gc_ctx(flat_raws, n_g, alpha, flat_masks)
    _, _, adv_id, ret_id = _run_pair(flat_lengths, flat_values, flat_rewards, gc_id)

    # 本地乱序 (positions = 置换)
    perm = [3, 0, 5, 1, 4, 2]
    p_lengths = [flat_lengths[p] for p in perm]
    p_values = [flat_values[p] for p in perm]
    p_rewards = [flat_rewards[p] for p in perm]
    p_masks = [flat_masks[p] for p in perm]
    gc_perm = _gc_ctx(flat_raws, n_g, alpha, p_masks, positions=perm)
    _, _, adv_perm, ret_perm = _run_pair(p_lengths, p_values, p_rewards, gc_perm)

    for i, p in enumerate(perm):
        assert torch.equal(adv_perm[i], adv_id[p])
        assert torch.equal(ret_perm[i], ret_id[p])


# ---------------------------------------------------------------------------
# ⑧ 非整组批 fail-fast (trim/subsample/custom convert 截断组边界的兜底)
# ---------------------------------------------------------------------------


def test_non_whole_groups_raises(cp1_parallel_state):
    lengths = [10, 20, 30, 40]
    raws = [1.0, 0.0, 1.0, 0.0]  # N_global=4, n_g=3 → 不是整组
    values, rewards, masks = _make_batch(lengths, seed=4)
    gc = _gc_ctx(raws, n_g=3, k=0.5, masks=masks)
    with pytest.raises(AssertionError, match="whole groups"):
        _run_pair(lengths, values, rewards, gc)


# ---------------------------------------------------------------------------
# ⑨ 与 --olp-analytic-inject 共存: 差值 = 两解析项之和
# ---------------------------------------------------------------------------


def test_coexists_with_olp_inject(cp1_parallel_state):
    n_g, gc_k, olp_alpha = 3, 0.5, 0.2
    lengths = [30, 44, 15]
    raws = [1.0, 0.0, 1.0]
    olp_pens = [-0.5, 0.0, -1.0]
    values, rewards, masks = _make_batch(lengths, seed=11)
    kwargs = dict(
        total_lengths=[L + 7 for L in lengths],
        response_lengths=lengths,
        values_list=values,
        rewards_list=[r.clone() for r in rewards],
        gamma=1.0,
        lambd=0.95,
        length_adaptive_lambda_k=0.5,
    )
    adv0, ret0 = get_advantages_and_returns_batch(**kwargs)
    gc = _gc_ctx(raws, n_g, gc_k, masks)
    adv2, ret2 = get_advantages_and_returns_batch(
        **kwargs, olp_inject_penalties=olp_pens, olp_inject_alpha=olp_alpha, group_center=gc
    )
    P = _reference_P(raws, values, lengths, masks, n_g)
    for i, L in enumerate(lengths):
        expected = olp_pens[i] * _expected_weights_alpha(L, olp_alpha, L) + P[i] * _expected_weights(L, gc_k, L)
        torch.testing.assert_close(adv2[i] - adv0[i], expected, rtol=1e-4, atol=1e-5)
        assert torch.equal(ret2[i], ret0[i])


# ---------------------------------------------------------------------------
# ⑩ DP>1 / CP>1 一致性 — 单进程单测无法构造, 标注 skip
# ---------------------------------------------------------------------------


@pytest.mark.skip(
    reason=(
        "DP>1 的真实 allreduce 需要 torch.distributed 多进程; 单测环境未初始化 dist 时"
        "注入代码跳过 allreduce, 单进程持有全量数据即退化正确 (散射/组和的索引数学由"
        "test_scatter_positions_permutation_consistent 覆盖: a_buf 按互斥 partition 散射后"
        " allreduce(SUM) 与单进程拼接逐字节等价)。死锁安全性由结构保证: 开关是全局 args"
        "而非数据依赖, 所有 intra_dp rank 无条件恰好执行一次 allreduce (仿 loss.py"
        " normalize_advantages 的 DP-allreduce 不变量)。"
    )
)
def test_dp_gt1_allreduce():
    raise NotImplementedError


@pytest.mark.skip(
    reason=(
        "CP>1 需要 torch.distributed 多进程 + megatron 风格 parallel state + zigzag 切分工具"
        "(cp_utils.local_response_to_full / slice_log_prob_with_cp), 单进程单测环境无法构造。"
        "结构性保证: 注入发生在 get_advantages_and_returns_batch 的全长坐标视图上——CP 各 rank"
        "在 values/rewards allreduce 之后持有相同的 full_values (a_j 从中读取, 任意位置可读),"
        " loss_masks 未做 CP 切分, 故各 CP rank 的 a_local/P_i 一致; 注入是确定性逐元素运算,"
        "逐 rank 结果一致, 之后才统一走 slice_log_prob_with_cp 切分。cp_size=1 的全链路等价性"
        "已由上方用例覆盖。"
    )
)
def test_cp_gt1_consistency():
    raise NotImplementedError
