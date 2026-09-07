#!/usr/bin/env python3
"""Minimal reproduction of JustRL2 (#12)'s cc-noLM value-head critic.

A standalone, Megatron-free sanity check that the core piece — a *scalar value head*
(`output_size=1`, zero-initialized) trained on GAE returns — actually learns on a toy
task. It does **not** need the fork submodules, only torch.

What it shows:
  1. **Zero init** — the value head starts at V ≡ 0 (the `LinearForLastLayer` behaviour).
     A normal-initialized head would start with V ~ N(0, 0.02·√H·rms(h)) and need tens of
     steps just to reach the [0, 1] target band; zero init makes step 0 exactly on-target
     shape so `--normalize-advantages` degrades to whitened reward (GRPO-like failure-safe).
  2. **Value regression** — after a few hundred steps on a fixed reward schedule, the head's
     prediction tracks the return. The convergence curve is the "did the value head learn"
     evidence.

How to run (container, py>=3.10):

    python3 examples/reproducibility/minicpm5_value_head_demo.py

This mirrors `LinearForLastLayer` (miles/backends/megatron_utils/model_provider.py):
the critic replaces `output_layer` with a `[1, hidden]` weight + `[1]` bias, both zeroed.
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
REWARD_MEAN = 0.7     # value target band
REWARD_STD = 0.2


class ScalarValueHead(torch.nn.Module):
    """The cc-noLM critic's output layer: a single-scalar value head.

    Mirrors `LinearForLastLayer` with `output_size=1`: the weight and bias are
    `zero_()`-initialized, and no LM head exists (the critic has no `vocab` logits).
    """

    def __init__(self, input_size: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1, input_size))
        self.bias = torch.nn.Parameter(torch.zeros(1))
        # In the real mega patch the weight is stamped `is_embedding_or_output_parameter`
        # so muon routes it to the matched-adamw branch; here we just use adam.
        self.weight.is_embedding_or_output_parameter = True

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # #12 keeps the value head in fp32 (`.float()` in the real forward), matching
        # `LinearForLastLayer.forward` for the scalar path.
        return torch.nn.functional.linear(h.float(), self.weight.float(), self.bias.float())


def main() -> None:
    torch.manual_seed(0)
    # Unit-variance "features" standing in for the critic's last hidden state (normalized).
    # Each rollout draws hidden states H; value target is a fixed return.
    H = torch.randn(STEPS, SEQ_LEN, HIDDEN)
    targets = torch.full((STEPS, SEQ_LEN, 1), REWARD_MEAN) + torch.randn(STEPS, SEQ_LEN, 1) * REWARD_STD

    head = ScalarValueHead(HIDDEN)
    opt = torch.optim.Adam(head.parameters(), lr=LR)

    # --- report step-0 magnitude (the zero-init claim) ---
    with torch.no_grad():
        v0 = head(H[0])
    w_absmax = head.weight.detach().abs().max().item()
    print(f"[init] value-head weight absmax = {w_absmax:.3e} (0 => zero-init)")
    print(f"[init] step-0 V range          = [{v0.min().item():+.3f}, {v0.max().item():+.3f}]  (targets ~ {REWARD_MEAN})")

    # --- train (critic-only: no policy loss here) ---
    losses = []
    start = time.time()
    for step in range(STEPS):
        opt.zero_grad()
        v = head(H[step])
        loss = torch.mean((v - targets[step]) ** 2)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    with torch.no_grad():
        v_final = head(H[-1])

    print(f"\n[final] steps={STEPS}  lr={LR}  wall={time.time() - start:.2f}s")
    print(f"[final] value-loss  step-0={losses[0]:.4f}  ->  step-{STEPS}={losses[-1]:.4f}")
    print(f"[final] V range     [{v_final.min().item():+.3f}, {v_final.max().item():+.3f}]  (targets ~ {REWARD_MEAN})")
    print(f"[final] weight absmax = {head.weight.detach().abs().max().item():.3e}")

    ok = losses[-1] < losses[0] and abs(v_final.mean().item() - REWARD_MEAN) < 0.2
    print("\n" + ("PASS — the value head started at zero and learned to track the return."
                  if ok else
                  "FALL — did not converge on this toy; check LR / target scale."))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
