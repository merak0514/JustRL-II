#!/usr/bin/env python3
"""Minimal reproduction of JustRL2 (#12)'s cc-noLM value-head critic.

A standalone, Megatron-free sanity check that the core piece — a *scalar value head*
(`output_size=1`, zero-initialized) trained on GAE returns — actually learns on a toy
task. It does **not** need the fork submodules, only torch.

What it shows:
  1. **Prior-seeded init** — the weight is zero and the bias is the expected mean reward
     (`--critic-value-bias-init`, 0.52), so the value head starts at V ≡ 0.52 exactly (the
     `LinearForLastLayer` behaviour). A normal-initialized weight would start with
     V ~ N(0, 0.02·√H·rms(h)) and need tens of steps to reach the [0, 1] band; a zero bias
     would open the value loss at E[r²] ≈ 0.5 instead of Var(r) ≈ 0.25 and spend ~25 steps
     learning the offset.
  2. **A/B vs. zero bias** (mirrors the paper's value-head-init figure): both heads are
     trained on the same stream; the seeded one opens at value-loss ≈ Var(r) with a ~3x
     smaller first gradient, the zero one spends its first steps learning the offset.

How to run (container, py>=3.10):

    python3 examples/reproducibility/minicpm5_value_head_demo.py

This mirrors `LinearForLastLayer` (miles/backends/megatron_utils/model_provider.py):
the critic replaces `output_layer` with a zeroed `[1, hidden]` weight + a `[1]` bias
filled with the prior.
"""

from __future__ import annotations

import time

import torch

HIDDEN = 128          # toy hidden size (real MiniCPM5-2.6B: 2048)
# Critic LR. NOTE: for a *converging toy* we use a demonstrative 1e-3; the #12 recipe uses
# --critic-lr 5e-6 over full rollout steps (the head converges over ~30 critic-only steps).
LR = 1e-3
STEPS = 800
SEQ_LEN = 32
REWARD_MEAN = 0.52    # value target band (= the seeded prior; #12 s9 mean reward)
REWARD_STD = 0.2


class ScalarValueHead(torch.nn.Module):
    """The cc-noLM critic's output layer: a single-scalar value head.

    Mirrors `LinearForLastLayer` with `output_size=1`: the weight is `zero_()`-initialized,
    the bias holds the prior, and no LM head exists (the critic has no `vocab` logits).
    """

    def __init__(self, input_size: int, bias_init: float = 0.52) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1, input_size))
        self.bias = torch.nn.Parameter(torch.full((1,), float(bias_init)))
        # In the real mega patch the weight is stamped `is_embedding_or_output_parameter`
        # so muon routes it to the matched-adamw branch; here we just use adam.
        self.weight.is_embedding_or_output_parameter = True

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # #12 keeps the value head in fp32 (`.float()` in the real forward), matching
        # `LinearForLastLayer.forward` for the scalar path.
        return torch.nn.functional.linear(h.float(), self.weight.float(), self.bias.float())


def _run(bias_init: float, H: torch.Tensor, targets: torch.Tensor):
    head = ScalarValueHead(HIDDEN, bias_init=bias_init)
    opt = torch.optim.Adam(head.parameters(), lr=LR)
    losses, gnorms = [], []
    for step in range(STEPS):
        opt.zero_grad()
        loss = torch.mean((head(H[step]) - targets[step]) ** 2)
        loss.backward()
        gnorms.append(torch.cat([p.grad.flatten() for p in head.parameters()]).norm().item())
        opt.step()
        losses.append(loss.item())
    with torch.no_grad():
        v_final = head(H[-1])
    return head, losses, gnorms, v_final


def main() -> None:
    torch.manual_seed(0)
    # Unit-variance "features" standing in for the critic's last hidden state; the
    # return depends weakly on them so there is something to learn beyond the offset.
    H = torch.randn(STEPS, SEQ_LEN, HIDDEN)
    w_true = torch.randn(HIDDEN) * (REWARD_STD / HIDDEN**0.5)
    targets = REWARD_MEAN + H @ w_true + torch.randn(STEPS, SEQ_LEN) * 0.05
    targets = targets.unsqueeze(-1)

    print(f"targets: mean={targets.mean():.3f} var={targets.var():.4f} (E[r^2]={float((targets**2).mean()):.4f})\n")
    results = {}
    for bias_init in (0.0, REWARD_MEAN):
        head, losses, gnorms, v_final = _run(bias_init, H, targets)
        results[bias_init] = (losses, gnorms, v_final)
        tag = f"bias_init={bias_init:<5}"
        print(f"[{tag}] step-0: weight absmax={0.0:.1e}  V={bias_init:.2f}  "
              f"value-loss={losses[0]:.4f}  grad-norm={gnorms[0]:.3f}")
        print(f"[{tag}] final : value-loss={losses[-1]:.4f}  "
              f"V range=[{v_final.min():+.3f}, {v_final.max():+.3f}]  weight absmax={head.weight.abs().max():.2e}")

    l0, g0, _ = results[0.0]
    lb, gb, vb = results[REWARD_MEAN]
    settle = next((i for i, l in enumerate(l0) if l <= lb[0] * 1.5), STEPS)
    print(f"\nzero-init needs {settle} steps just to reach where the seeded head starts "
          f"(loss {l0[0]:.3f} -> {lb[0]:.3f}); first-step grad norm {g0[0]:.2f} vs {gb[0]:.2f} "
          f"({g0[0] / max(gb[0], 1e-9):.1f}x).")

    ok = (
        lb[0] < l0[0] * 0.5                     # seeded head opens at ~Var(r), not ~E[r^2]
        and gb[0] < g0[0]                        # and with a smaller first gradient
        and abs(vb.mean().item() - REWARD_MEAN) < 0.1
        and lb[-1] < lb[0]                       # and still learns the feature-dependent part
    )
    print("\n" + ("PASS — seeding the value-head bias at the mean reward removes the warmup transient."
                  if ok else
                  "FAIL — expected: seeded loss0 << zero-init loss0, smaller grad, V tracks mean."))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
