# JustRL2 (#12) — config mapping (main_cc_nolm_s9)

The #12 run is **not** a separate config file in the original repo; it is the entry script
`scripts/minicpm5/2_6b/long_rl/run-minicpm5-26b-long-math-128k-ppo.sh` driven by the
environment variables below. This file documents that mapping so an external run can set
the same values.

## Entry environment (the canonical #12 values)

| Env var | #12 value | Meaning |
|---------|-----------|---------|
| `EXP_TAG` | `main_cc_nolm_s9` | Experiment tag, becomes `…_main_cc_nolm_s9` in the run name and `SAVE_DIR` |
| `CRITIC_EXCLUDE_OLP` | `1` | Critic's value-loss regression target excludes the overlong penalty (actor advantage unchanged). Maps to `--critic-exclude-overlong-penalty`. |
| `SGLANG_MEM_FRACTION` | `0.83` | sglang engine memory fraction (`--sglang-mem-fraction-static`). |
| `DYNAMIC_SAMPLING` | `1` | DAPO-style filter: drop zero-variance groups; needs over-sampling to refill. |
| `ROLLOUT_BATCH_SIZE` | `60` | Prompts per rollout batch (→ GBS = 60 × 8 = 480). |
| `VAPO_LAMBDA_K` | `0.513` | **Key.** Length-adaptive GAE: per-sample `λ_i = k^(1/L_i)` (first token always gets fraction k of terminal credit), returns use λ=1 (suffix reward sum). #12 literally ran the old form `1 − 1/(α·L)` with α=1.5, which is `k = e^(−1/1.5) ≈ 0.513` to < 1e-6. |
| `CRITIC_VALUE_BIAS_INIT` | `0.52` | JustRL2 refinement (#12 ran 0): value-head bias seeded at the expected mean reward so `V ≡ 0.52` at step 0; re-applied after the base-ckpt load. Pass `0` to reproduce #12 literally. |
| `NUM_CRITIC_ONLY_STEPS` | `30` | First 30 rollouts train **only** the critic; actor frozen so the zero-initialized value head converges before policy updates. |
| `ENABLE_PARTIAL_ROLLOUT` | `1` | Partial rollout: cap the longest trajectories (long-context + memory-bound). |
| `OVER_SAMPLING_BATCH_SIZE` | `120` | Over-sample prompts (2× the batch), take the first finished `ROLLOUT_BATCH_SIZE` groups, recycle unfinished runs into the buffer. |
| `EPS_CLIP_HIGH` | `0.28` | Upper PPO clip — mild asymmetric clip, GRPO/DAPO-adjacent (adv-clip). |
| `LR` | `1e-6` | Actor learning rate (constant decay). |
| `NUM_ROLLOUT` | `500` | Total rollout steps. |
| `EVAL_INTERVAL` | `1000` | Eval every 1000 steps. |
| `SAVE_INTERVAL` | `5` | Megatron ckpt every 5 steps. |
| `HF_SAVE_INTERVAL` | `5` | HF export every 5 steps (only actor; critic not exported). |
| `--override-opt-param-scheduler` | set | Needed on the first resume run: rebuild the LR scheduler from the CLI value (Megatron asserts if the saved total iters differ). |

Not shown but part of #12's recipe: `N_SAMPLES_PER_PROMPT=8`, `TENSOR_MODEL_PARALLEL_SIZE=1`,
`CONTEXT_PARALLEL_SIZE=4`, actor/critic nodes split evenly (both equal to `WORLD_SIZE/2`).

## The "cc-noLM" name

`main_cc_nolm_s9` = **cc**ritic with **no LM head** (`cc-noLM`), `s9` dataset. The critic is
a separate dense model whose `output_layer` is a **scalar value head** (`output_size=1`),
replacing the LM head. There is no language-modeling loss on the critic — it only regresses
the value (GAE return). The cc-noLM critic (mean-reward-seeded bias) + `k^(1/L)` GAE recipe is what we call **JustRL2**.
See *Technical core* in the README.

## Launch (single node)

```bash
export EXP_TAG=main_cc_nolm_s9
export CRITIC_EXCLUDE_OLP=1
export SGLANG_MEM_FRACTION=0.83
export DYNAMIC_SAMPLING=1
export ROLLOUT_BATCH_SIZE=60
export VAPO_LAMBDA_K=0.513
export CRITIC_VALUE_BIAS_INIT=0.52
export NUM_CRITIC_ONLY_STEPS=30
export ENABLE_PARTIAL_ROLLOUT=1
export OVER_SAMPLING_BATCH_SIZE=120
export EPS_CLIP_HIGH=0.28
export LR=1e-6
export NUM_ROLLOUT=500
export EVAL_INTERVAL=1000
export SAVE_INTERVAL=5
export HF_SAVE_INTERVAL=5
export MODELS_DIR=$PWD/models/cache

bash scripts/minicpm5/2_6b/long_rl/run-minicpm5-26b-long-math-128k-ppo.sh \
  --override-opt-param-scheduler
```

Use a 2×-node, 16×H100 (or larger) setup for the real recipe: `PPO` requires actor and
critic node counts to be equal (the actor-critic NCCL groups are built rank-pairwise), so a
single node with `ACTOR_NUM_NODES` unset is a **debug** only. `WORLD_SIZE` even and
`ACTOR_NUM_NODES == CRITIC_NUM_NODES == WORLD_SIZE/2`.
