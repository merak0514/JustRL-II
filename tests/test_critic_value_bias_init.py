"""JustRL2 critic value-head bias init: construction + post-load re-seed.

Two hunks must agree or the prior is silently lost:
  (1) model_provider.LinearForLastLayer fills the scalar head's bias with
      --critic-value-bias-init (weight stays zero-init);
  (2) checkpoint._rezero_critic_value_head, which runs after a policy/base ckpt load
      (finetune=True), re-zeroes the weight but re-fills the bias with the same value
      instead of zeroing it. Without (2) the load path resets the bias to 0.

megatron is stubbed at import time so the real functions run on CPU.
"""

import sys
import types
from types import SimpleNamespace

import pytest
import torch


def _stub(name, **attrs):
    mod = sys.modules.get(name) or types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


@pytest.fixture(scope="module")
def checkpoint_mod():
    if "megatron" not in sys.modules:
        _stub("megatron")
        _stub("megatron.training")
        _stub("megatron.training.checkpointing", load_checkpoint=lambda *a, **k: None, save_checkpoint=lambda *a, **k: None)
        _stub("megatron.training.global_vars", get_args=lambda: None)
        _stub("megatron.core")
        _stub("megatron.core.tensor_parallel")
        _stub("megatron.core.models")
        _stub("megatron.core.models.gpt", GPTModel=object)
        _stub("megatron.core.models.gpt.gpt_layer_specs", get_gpt_decoder_block_spec=None, get_gpt_layer_local_spec=None, get_gpt_layer_with_transformer_engine_spec=None)
        _stub("megatron.core.transformer")
        _stub("megatron.core.transformer.spec_utils", import_module=None)
        _stub("megatron.core.transformer.transformer_config", TransformerConfig=object)
        _stub("megatron.training.arguments", core_transformer_config_from_args=None)
    try:
        from miles.backends.megatron_utils import checkpoint as ck  # noqa: F401
    except ImportError as e:  # a transitive dep this CPU stub set does not cover
        pytest.skip(f"checkpoint module not importable on CPU: {e}")
    return ck


@pytest.fixture(scope="module")
def head_cls(checkpoint_mod):
    from miles.backends.megatron_utils.model_provider import LinearForLastLayer

    return LinearForLastLayer


class _Chunk(torch.nn.Module):
    """Minimal stand-in for a GPTModel chunk: only the scalar head matters."""

    def __init__(self, head):
        super().__init__()
        self.output_layer = head


def _cfg():
    return SimpleNamespace(sequence_parallel=False)


def test_construction_fills_bias_and_zeroes_weight(head_cls):
    head = head_cls(16, 1, config=_cfg(), bias_init=0.52)
    assert torch.all(head.weight == 0)
    assert head.bias.item() == pytest.approx(0.52)
    # step-0 value is exactly the prior, for any input
    assert torch.allclose(head(torch.randn(4, 16))[0], torch.full((4, 1), 0.52))


def test_construction_default_is_zero(head_cls):
    head = head_cls(16, 1, config=_cfg())
    assert head.bias.item() == 0.0


def test_non_scalar_head_ignores_bias_init(head_cls):
    head = head_cls(16, 8, config=_cfg(), bias_init=0.52)
    assert torch.all(head.bias == 0)


def test_rezero_after_policy_load_reseeds_bias(checkpoint_mod, head_cls):
    head = head_cls(16, 1, config=_cfg(), bias_init=0.52)
    # simulate the dist-ckpt reader polluting the head with LM-head row 0 + bias reset
    with torch.no_grad():
        head.weight.fill_(0.189)
        head.bias.zero_()
    args = SimpleNamespace(finetune=True, critic_finetune_load_value_head=False, critic_value_bias_init=0.52)
    checkpoint_mod._rezero_critic_value_head([_Chunk(head)], optimizer=None, args=args)
    assert torch.all(head.weight == 0)
    assert head.bias.item() == pytest.approx(0.52)


def test_rezero_skipped_on_resume(checkpoint_mod, head_cls):
    head = head_cls(16, 1, config=_cfg(), bias_init=0.52)
    with torch.no_grad():
        head.weight.fill_(0.3)
        head.bias.fill_(0.7)
    args = SimpleNamespace(finetune=False, critic_value_bias_init=0.52)
    checkpoint_mod._rezero_critic_value_head([_Chunk(head)], optimizer=None, args=args)
    assert torch.all(head.weight == 0.3) and head.bias.item() == pytest.approx(0.7)


def test_rezero_resyncs_optimizer_master(checkpoint_mod, head_cls):
    head = head_cls(16, 1, config=_cfg(), bias_init=0.52)
    calls = []
    opt = SimpleNamespace(reload_model_params=lambda: calls.append(1))
    args = SimpleNamespace(finetune=True, critic_finetune_load_value_head=False, critic_value_bias_init=0.52, load_main_params_from_ckpt=False)
    checkpoint_mod._rezero_critic_value_head([_Chunk(head)], optimizer=opt, args=args)
    assert calls == [1]
