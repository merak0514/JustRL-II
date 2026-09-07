<div align="center">

# MiniCPM5-2.6B Math RLARM — #12 (`main_cc_nolm_s9`) Open-Source Reproduction

**Algorithm = JustRL2**: cc-noLM critic (mean-reward-seeded value head) + length-adaptive GAE `λ = k^(1/L)` on 128k-context math RL

</div>

This is an open-source reproduction package for math_recipe experiment **#12**, tag
`main_cc_nolm_s9`: MiniCPM5-2.6B trained on math with a **critic-only (no-LM-head) value
head**, over 128k context, on the `s9` math dataset. We call the algorithm **JustRL2** —
the cc-noLM (value-head-only) critic + length-adaptive GAE recipe that is the point of
#12, with two refinements over what #12 literally ran (see *What #12 ran vs. what ships*). The framework it runs on is **Miles** — a fork of the Apache-2.0
[Miles](https://github.com/radixark/miles) RL framework (SGLang + Megatron-LM) — at commit
**`b62206c9`** on `phx_dev_recipe2`, with the MiniCPM5-2.6B model args and the scalar
value-head critic that are the focus of #12. The codebase keeps the name `miles` for
provenance; only the **algorithm** is branded JustRL2.

> **Scope / honesty note.** This is *not* a byte-exact re-ship of the internal training stack.
> The internal run used three **private forks** (Megatron-LM, mbridge, sglang) that are not
> public. The miles-side code, the Megatron patch, and the complete #12 configuration are
> here; the forks are documented in [`models/README.md`](models/README.md) with three paths
> to obtain them. Public-equivalent weights (base model, DSpark draft) and the `s9` dataset
> are *not* redistributable — the "s9" filtering logic is documented in
> [`data/README.md`](scripts/data/README.md) but the 32k-question filter output is internal.

---

## The JustRL2 recipe in one paragraph

JustRL2 = PPO with a **separate critic** whose `output_layer` is a **scalar value head**
(`output_size=1`) instead of an LM head. The critic regresses the GAE return; the actor
takes the PPO surrogate on the resulting advantages. Three things make it work:

1. The value head's **weight is zero-initialized** and its **bias is seeded at the
   expected mean reward** (`CRITIC_VALUE_BIAS_INIT=0.52`), so at step 0 `V ≡ 0.52` — the
   critic starts as a sensible prior instead of spending ~25 steps learning the offset.
   Both are **re-applied after a policy/base-ckpt load** (so the LM head row 0 can't
   pollute the weight and the load path can't reset the bias) — commit `1faf7ad4c` lineage.
2. A **critic-only warmup** (`NUM_CRITIC_ONLY_STEPS=30`): the actor is frozen while the
   value head converges, so the first policy update sees a sensible baseline.
3. **Length-adaptive GAE** (`VAPO_LAMBDA_K`): per-sample `λ_i = k^(1/L_i)`, so the first
   token always receives fraction `k` of the terminal credit (`λ_i^L_i = k`) regardless of
   response length; the value target stays the λ=1 suffix-reward sum.

Plus partial-rollout + over-sampling for long-context memory, and DSpark speculative
decoding when a draft model is available.

---

## What's in this package

```
miles-opensource/
├── miles/                  # the framework (clean; no internal paths)
├── miles_plugins/
├── train.py                # entry (Ray job submit -> python3 train.py)
├── train_async.py
├── scripts/
│   ├── minicpm5/2_6b/long_rl/run-minicpm5-26b-long-math-128k-ppo.sh   # #12 entry
│   ├── minicpm5/2_6b/minicpm5-26b.sh                                  # model args
│   ├── setup/{env,setup,ray_start,ray_end}.sh                         # sanitized
│   └── data/README.md                                                 # s9 dataset note
├── config/#12-config.md       # the full #12 entry mapping
├── docker/patch/latest/       # megatron.patch + sglang.patch (pip into the forks)
├── models/README.md           # submodule + weight sourcing (the gate)
├── examples/reproducibility/  # run-qwen2.5-0.5B-gsm8k.sh (upstream) + value-head demo
├── tests/                     # incl. test_vapo_lambda_k.py, test_critic_value_bias_init.py
├── requirements.txt
├── pyproject.toml / setup.py / LICENSE (Apache-2.0)
```

---

## Environment

Linux, NVIDIA GPU (H100 class; the #12 run is 16×H100). Python **≥ 3.10** (miles uses 3.10+
syntax — 3.9 fails to import with the `ProcessGroup | None` union error seen on the toy
demo path).

```bash
conda create -n miles_rl python=3.12 -y
conda activate miles_rl
pip install -r requirements.txt            # nvidia CUDA + torch wheel matched to your cluster
bash scripts/setup/setup.sh                # apply megatron.patch to Megatron-LM/
```

`requirements.txt` pulls `math-verify==0.9.0`, `sglang-router`, `transformers`, `ray`,
`tensorboard`, `swanlab`. `setup.sh` also installs `sgl-kernel>=0.3.20` (required by the
sglang fork) and applies `docker/patch/latest/megatron.patch` into `Megatron-LM/` (idempotent
via a marker file). Set `SKIP_PIP_INSTALL=1` if your base image already has deps.

### The three submodules

`Megatron-LM`, `mbridge`, `sglang` are the private forks. See
[`models/README.md`](models/README.md) for the pinned commits (sglang at `fc203f00e`, the
DSpark-capable commit) and the three ways to obtain them. Without them, only the
toy value-head demo and the GAE tests run.

---

## Data

- **Train**: the `s9` math set. `s9` is internal (a 32,412-question filtered subset of a
  larger math corpus). The filtering logic and a public `deepmath`/`deepscaler`/`dapo-math`
  stand-in are described in [`scripts/data/README.md`](scripts/data/README.md); the exact
  selection is not redistributable.
- **Test / eval**: AIME 2024/2025/2026. Set `TEST_FILE` like the original:
  `"aime2026 <path>/aime-2026.jsonl aime2025 <path>/aime-2025.jsonl aime2024 <path>/aime-2024.jsonl"`.

Put them under `DATA_DIR` (default `$WORK_DIR/datasets`) and point `TRAIN_FILE` /
`TEST_FILE` at it.

---

## Run (single node, debug)

```bash
export MODELS_DIR=$PWD/models/cache        # holds HF + Megatron torch_dist ckpt
export DATA_DIR=$PWD/datasets              # holds the jsonl
bash scripts/minicpm5/2_6b/long_rl/run-minicpm5-26b-long-math-128k-ppo.sh
```

**PPO node split.** The script hard-requires `ACTOR_NUM_NODES == CRITIC_NUM_NODES`
(the actor-critic NCCL groups are built rank-pairwise), so a real #12 run uses an even
`WORLD_SIZE`:

```bash
export WORLD_SIZE=16        # 16 nodes
export ACTOR_NUM_NODES=8    # = CRITIC_NUM_NODES (auto = WORLD_SIZE/2)
```

For the full #12 values, source the file in [`config/#12-config.md`](config/#12-config.md),
or just:

```bash
export EXP_TAG=main_cc_nolm_s9 CRITIC_EXCLUDE_OLP=1 SGLANG_MEM_FRACTION=0.83 \
       DYNAMIC_SAMPLING=1 ROLLOUT_BATCH_SIZE=60 VAPO_LAMBDA_K=0.513 CRITIC_VALUE_BIAS_INIT=0.52 \
       NUM_CRITIC_ONLY_STEPS=30 ENABLE_PARTIAL_ROLLOUT=1 OVER_SAMPLING_BATCH_SIZE=120 \
       EPS_CLIP_HIGH=0.28 LR=1e-6 NUM_ROLLOUT=500 EVAL_INTERVAL=1000 \
       SAVE_INTERVAL=5 HF_SAVE_INTERVAL=5
```

The GBS divisibility check: actor DP = `ACTOR_NODES*GPUS/TP/CP`; with the #12 values
(60 × 8 = 480 GBS, CP4, TP1, 6 actor nodes → DP12) this passes.

### Minimal demo (no submodules needed)

```bash
python3 examples/reproducibility/minicpm5_value_head_demo.py
```

A toy scalar value-head check: it asserts the head starts at **exactly the seeded prior**
(zero weight, bias = 0.52) and that gradient descent makes it track a return band. This is
the mechanism (prior-seeded init + value regression) that the 128k critic relies on.

### What #12 ran vs. what ships

Two knobs in this package are the JustRL2 refinements, not what job 825481 literally ran:

| | #12 (job 825481, `b62206c9`) | this package (JustRL2) |
|---|---|---|
| value-head bias init | 0 | `--critic-value-bias-init 0.52` (validated on arm extra-1 with 0.5: +0.022 AIME at the 120-step window, warmup transient removed) |
| length-adaptive λ | `1 − 1/(α·L)`, α=1.5 | `k^(1/L)`, k=0.513 ≡ same λ to <1e-6 at L ≥ 1000 |

To reproduce #12 *literally*, pass `CRITIC_VALUE_BIAS_INIT=0`; the λ change is numerically
a no-op at these lengths. `--vapo-lambda-alpha` no longer exists — use `--vapo-lambda-k`
with `k = exp(−1/α)`.

## Naming

- **JustRL2** — the *algorithm*: cc-noLM scalar value-head critic (mean-reward-seeded bias) +
  length-adaptive GAE `λ = k^(1/L)` + critic-only warmup. #12 is its first full run.
- **Miles** — the *framework* the algorithm runs on (Apache-2.0, upstream
  [radixark/miles](https://github.com/radixark/miles)). The codebase keeps its name; we do
  **not** rename the framework to JustRL2, so its provenance and upstream diffs stay
  auditable.

---

## Technical core (why it works)

### 1. cc-noLM critic — scalar value head, no LM head

The critic is a **separate dense model of the same shape** as the actor, but its
`output_layer` is replaced by `LinearForLastLayer(input_size=hidden, output_size=1)`
([`model_provider.py`](miles/backends/megatron_utils/model_provider.py) lines 27–77,
274). It produces one scalar `V` per token. There is **no LM head** and **no token-prediction
loss** — the critic only regresses the GAE return. Removing the LM head avoids the shape
mismatch/pollution that would occur if the value head had to share (or be confused with) the
`[vocab, hidden]` LM head.

### 2. Value head init: zero weight + mean-reward bias, re-applied after load (the critical fix)

- `LinearForLastLayer.__init__` zeroes the `[1, hidden]` weight and fills the `[1]` bias
  with `--critic-value-bias-init` (JustRL2 default **0.52**, the expected mean reward on
  the s9 math mix). With a zero weight `V == bias` at step 0, so the bias *is* the
  critic's prior. With a normal `N(0, 0.02)` weight the value output has std
  `0.02·√H·rms(h) ≈ 9.7` while targets live in `[0,1]`; with a zero bias the value loss
  opens at `E[r²] ≈ 0.5` and the critic gradient norm at 130–140, and the head spends its
  first ~25 steps learning the offset *while the policy already updates against it*.
  Seeded at the mean reward the loss opens at `Var(r) ≈ 0.25` and the gradient norm stays
  under 40 from the first step — the whole transient disappears at zero cost.
- After loading a **policy/base checkpoint**, `checkpoint.py:_rezero_critic_value_head`
  re-zeroes the weight, **re-fills the bias** with the same value, and resyncs the fp32
  master weights. Without this, Megatron's dist-ckpt reader fills the `[1,H]` head from
  the overlapping LM-head region (measured `w_absmax 0.189` → `V ~ ±8`) and the load
  path resets the bias to 0 — **completely destroying the run**. Commit `1faf7ad4c`
  exists precisely to fix this first-training pollution. **If you port this pipeline and
  skip `_rezero_critic_value_head`, the value head will be polluted and the prior lost.**
  Verify at startup: the log line `[critic-value-head] re-zeroed [...] (bias_init=0.52, master
  params resynced)`.

### 3. Length-adaptive GAE (`vapo_lambda_k`, γ=1)

`get_advantages_and_returns_batch` ([`ppo_utils.py`](miles/utils/ppo_utils.py)) uses a
per-sample λ from `vapo_lambda_rowwise`:

```
λ_i = k ^ (1 / L_i)
```

We want the fraction of terminal credit that propagates back to the first token to stay a
constant `k` regardless of response length. With γ=1 the GAE weight of the terminal reward
at the first token is `λ_i^L_i`, and this choice makes it exactly `k`: longer responses get a
λ closer to 1, so credit propagation does not weaken with length. The **advantage** uses
`λ_i` row-wise; the **value target** (`returns`) is the `λ = 1` suffix-reward sum, so the
critic target is decoupled from the advantage discount. λ is computed in fp32 — at 128k
lengths `λ = 1 − O(1e-5)`, which bf16 rounds to exactly 1.0.

Relation to the VAPO form: `1 − 1/(α·L)` is the first-order expansion of `k^(1/L)` with
`k = e^(−1/α)`; the #12 run's `α = 1.5` corresponds to `k ≈ 0.513` (the two differ by
< 1e-6 at L ≥ 1000, see `tests/test_vapo_lambda_k.py`). The same λ also drives the optional
`--group-center-inject` decay.

### 4. Partial rollout + over-sampling

`ENABLE_PARTIAL_ROLLOUT=1` with `ROLLOUT_BATCH_SIZE=60` / `OVER_SAMPLING_BATCH_SIZE=120`
caps the longest trajectories (128k context + memory-bound) by **over-sampling prompts**,
taking the first completed `rollout_batch_size` groups, and recycling unfinished runs into
the next step's buffer (`--partial-rollout`).

### 5. Critic warmup

`NUM_CRITIC_ONLY_STEPS=30`: for the first 30 rollout steps only the critic updates
(`actor.py` gates the policy update by `rollout_id >= num_critic_only_steps`). The value
head converges before the actor starts taking PPO steps — this is what makes #12 stable.

### 6. DSpark speculative decoding

`DSPARK_DRAFT_MODEL_PATH` points at a MiniCPM5-2.6B **Draft-5L** model; when set, sglang
runs DSpark speculative decoding (block size 7). Needs the sglang fork at `fc203f00e`.
Empty → plain (non-speculative) decoding; the DSpark block is then a no-op.

---

## Determinism / reproducibility

The upstream [`examples/reproducibility/README.md`](examples/reproducibility/README.md)
documents bitwise reproduction (SGLang deterministic + Megatron deterministic + the NCCL /
CUBLAS env overrides). Useful for a clean comparison run.

## License

Apache-2.0 (the Miles framework). The MiniCPM5 model weights and the `s9` data are subject
to their own licenses and are not redistributed here.
