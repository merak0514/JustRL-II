# Reproducing the run

## Hardware and topology

The reference run used 16 nodes × 8 H100: 8 actor nodes + 8 critic nodes, SGLang engines
colocated on all GPUs, TP 1 / CP 4, global batch 480 (60 prompts × 8 samples).

PPO in miles requires **actor world size == critic world size** (the actor↔critic NCCL
groups are built rank-pairwise). `justrl2/train.sh` defaults to `WORLD_SIZE/2` nodes each
and refuses to start otherwise. The global batch must be divisible by the actor data
parallel size (`nodes × gpus / TP / CP`); with the defaults that is 480 / 16 — fine.

`justrl2/configs/debug-1node-8gpu.env` splits one node 4 + 4 for a pipeline smoke test.
The reference run never used a single node, so treat that config as untested plumbing,
not as a small-scale recipe.

## Launch

On every node (Ray head is `RANK=0`; `MASTER_ADDR`/`MASTER_PORT`/`WORLD_SIZE`/`RANK` must be
set by your scheduler, or left unset for a single node):

```bash
bash justrl2/train.sh justrl2/configs/minicpm5-2.6b-math-128k.env
```

Override any knob through the environment, e.g. `GAE_LAMBDA_K=0.4 CRITIC_VALUE_BIAS_INIT=0.5`.
Extra arguments after the config are passed straight to `train.py`.

### Startup checks worth reading

- `[critic-value-head] ... after-load ... w_absmax=0.189` then
  `re-zeroed [...] (bias_init=0.52, master params resynced)` — the post-load re-init ran.
- The first rollout's sample responses are coherent text. If they are garbage,
  `MEGATRON_MODEL_PATH` points at an `iter_xxx` subdirectory instead of its parent and
  Megatron silently started from random weights.
- `critic exclude shaping: critic returns computed from rewards without overlong penalty`.

### Resume

Re-run the same command. Actor and critic checkpoints under `SAVE_ROOT/EXP_TAG` and
`SAVE_ROOT/EXP_TAG_critic` are loaded automatically; a half-present pair is a hard error.
If you changed `NUM_ROLLOUT`, add `OVERRIDE_OPT_PARAM_SCHEDULER=1` on the first resume.

## Evaluate

The reference run set `EVAL_INTERVAL=1000` (no in-training eval) and evaluated the HF
exports (`SAVE_ROOT/EXP_TAG/hf/iter_XXXXXXX`, every 5 steps) offline:

```bash
python justrl2/eval.py --model runs/<EXP_TAG>/hf/iter_0000299 \
    --data datasets/aime-2024.jsonl --data datasets/aime-2025.jsonl --data datasets/aime-2026.jsonl \
    --n 16 --temperature 1.0 --top-p 0.95 --max-tokens 126976
```

Reports acc@16, pass@16, mean length and truncation rate per file. AIME differences of
1–2 problems are within noise at n = 16; compare peaks over a window, not single points.

## What the original internal run did differently

Two knobs in the released config are refinements over what that run literally executed:

| | original run | released config |
|---|---|---|
| value-head bias init | 0 | `CRITIC_VALUE_BIAS_INIT=0.52` (validated in a separate arm with 0.5: +0.022 AIME over the shared 120-step window, warmup transient removed) |
| GAE λ | `1 − 1/(α·L)`, α = 1.5 (≡ k ≈ 0.513) | `k^(1/L)`, `GAE_LAMBDA_K=0.5` — same functional form, k rounded from 0.513 to 0.5 |

To reproduce the original literally: `CRITIC_VALUE_BIAS_INIT=0 GAE_LAMBDA_K=0.513`. Everything else
(rollout sizes, clip, warmup, overlong penalty, `CRITIC_EXCLUDE_OLP=1`, LR, dynamic
sampling, partial rollout, DSpark) is as run.

## Determinism

Bitwise reproduction needs SGLang deterministic inference + Megatron deterministic mode
(`--sglang-enable-deterministic-inference --sglang-attention-backend flashinfer
--deterministic-mode`, plus `NCCL_ALGO=Ring NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
CUBLAS_WORKSPACE_CONFIG=:4096:8` in the Ray runtime env) and uninstalling flash-attn 3.
It is slower; the reference numbers were not produced this way.
