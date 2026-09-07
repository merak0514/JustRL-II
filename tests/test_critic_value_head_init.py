"""critic value head 初始化的回归测试 (需要 megatron 环境, 在 devspace/CI 上跑)。

守两条:
1. 标量 value head (output_size=1) 必须零初始化。normal(0, 0.02) 在 hidden=2048 上
   给出 V ~ ±9.7, 而组内中心化后的目标只有 ±0.35 —— critic_lr 5e-6 需 ~90 步才能
   填平, 任何 warmup 预算都不够, actor 会在 90% 是噪声的 advantage 上训练
   (job 697926 实测: value_loss 94, value_residual_frac 732, adv mu_w -7.4)。
   零初始化让 adv = R̃ 精确等于 GRPO, 这是 turn_ppo 依赖的 failure-safe 退化性质。
2. 非标量输出层 (词表头) 不受影响, 仍是 normal 初始化。

这两条都只有**实例化**才能验证: 上一版把 output_size 误写成 out_features,
py_compile / bash -n 全部通过, 却在集群建模型时 NameError 秒挂 (job 698467)。
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
    """muon 把"2D 且无此标记"的参数路由进 Newton-Schulz 正交化, 其更新幅度与误差
    无关 —— [1, H] 的 value head 会在最优点附近定幅震荡而非收敛。"""
    head = LinearForLastLayer(input_size=2048, output_size=1, config=cfg)
    assert getattr(head.weight, "is_embedding_or_output_parameter", False) is True
