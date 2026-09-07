# JustRL2 — the method

JustRL2 is PPO for long-context (128k) math reasoning with a **separate critic that has no
language-model head**, a **mean-reward-seeded value head**, and a **length-adaptive GAE λ**.
This page explains each piece and points at the code; the full write-up with experiments
and ablations is the blog post [JustRL-II: Scaling Small LLMs to 128k Reasoning with a
Critic](https://panhaoxuan.notion.site/justrl-ii-scaling-small-llms-to-128k-reasoning-with-a-critic)
([中文](https://panhaoxuan.notion.site/justrl-ii-small-llms-to-128k-reasoning-with-a-critic-cn)). File paths are relative to the repo
root; the framework is [Miles](https://github.com/radixark/miles) (SGLang rollout +
Megatron-LM training).

## 1. cc-noLM critic: a scalar value head instead of an LM head

The critic is a second copy of the policy architecture whose `output_layer` is replaced by
`LinearForLastLayer(input_size=hidden, output_size=1)` —
[`miles/backends/megatron_utils/model_provider.py`](../miles/backends/megatron_utils/model_provider.py).
It emits one scalar `V` per token and is trained only on the value loss; there is no
token-prediction loss on the critic at all. Keeping the `[vocab, hidden]` LM head around
was the source of the pollution bug in §2, and it costs memory for nothing.

The critic runs as its own Ray training group with the same world size as the actor
(`--critic-num-nodes == --actor-num-nodes`; NCCL groups are built rank-pairwise), colocated
with the SGLang engines.

## 2. Value-head init: zero weight, bias = expected mean reward — re-applied after load

**Construction.** For the scalar head the weight is zero-initialised and the bias is filled
with `--critic-value-bias-init` (0.52 in the released config). With a zero weight, `V ≡ bias`
at step 0, so the bias *is* the critic's prior.

Why not the usual `N(0, 0.02)` weight: on a 2048-wide hidden state that gives
`std(V) ≈ 0.02·√H·rms(h) ≈ 9.7` against targets in `[0, 1]`. Why not a zero bias: the value
loss then opens at `E[r²] ≈ 0.5` with a first critic gradient norm of 130–140, and the head
spends its first ~25 steps learning the offset *while the policy already updates against it*.
Seeded at the mean reward the loss opens at `Var(r) ≈ 0.25` and the gradient norm stays
under 40 from step 0 — the transient is gone at zero cost (blog, value-head-init figure).

**After loading the base checkpoint.** The critic starts from the policy/base checkpoint.
Megatron's dist-ckpt reader fills the `[1, H]` head from the overlapping region of the
`[vocab, H]` LM head (measured `w_absmax 0.189`, i.e. `V ~ ±8`) and the load path resets the
bias. `_rezero_critic_value_head` in
[`miles/backends/megatron_utils/checkpoint.py`](../miles/backends/megatron_utils/checkpoint.py)
therefore runs after every finetune-style load: it zeroes the weight again, **re-fills the
bias** with the same value, and calls `optimizer.reload_model_params()` so the fp32 master
copy matches (otherwise the first optimizer step would restore the polluted values).
On resume (`finetune=False`) it does nothing — the trained head from the critic's own
checkpoint is kept.

Verify at startup:
```
[critic-value-head] re-zeroed ['...output_layer.weight', '...output_layer.bias'] after policy-ckpt load (bias_init=0.52, master params resynced)
```
If you port this to another trainer and skip the post-load step, the run is silently broken.

## 3. Length-adaptive GAE: λ_i = k^(1/L_i), returns at λ = 1

`get_advantages_and_returns_batch` in
[`miles/utils/ppo_utils.py`](../miles/utils/ppo_utils.py) computes a per-sample λ from
`length_adaptive_lambda`:

```
λ_i = k ^ (1 / L_i)          (L_i = response length, k = --gae-lambda-k)
```

With γ = 1 the GAE weight of the terminal reward at the first token is `λ_i^L_i`, and this
choice makes it exactly `k` for every length: the fraction of terminal credit that reaches
the beginning of the response is a constant instead of decaying with length. Longer responses
get a λ closer to 1, so credit propagation does not weaken on 100k-token solutions.

Two details:

- **Decoupled targets.** The *advantage* uses `λ_i`; the *value target* (`returns`) is the
  λ = 1 suffix-reward sum. The critic regresses an unbiased return while the advantage uses
  the length-adaptive discount. This requires γ = 1 (asserted).
- **fp32.** At 128k, `λ = 1 − O(1e-5)`, which bf16 rounds to exactly 1.0; λ is computed in
  fp32 and only cast when handed to the GAE scan.

`k = 0.5` is the release value: half of the terminal credit reaches the first token. The
original internal run used the first-order form `1 − 1/(α·L)` with α = 1.5, which is
`k = exp(−1/1.5) ≈ 0.513` (the two parametrisations agree to < 1e-6 per token at L ≥ 1000,
see `tests/test_gae_lambda_k.py`); 0.5 is that value rounded.

## 4. Critic-only warmup

`--num-critic-only-steps 30`: for the first 30 rollouts only the critic is updated
(`actor.py` gates the policy update on `rollout_id >= num_critic_only_steps`). Combined with
§2 the value head is a usable baseline before the first policy step.

## 5. Length control: DAPO soft overlong penalty, kept out of the critic's target

The reward that feeds the actor's GAE is `r − penalty(L)` with the DAPO soft overlong
penalty on the last 20 % of the 126 976-token budget (`--overlong-buffer-len 25395`,
`--overlong-penalty-factor 1.0`). With `--critic-exclude-overlong-penalty` the critic's
regression target is the reward **without** that penalty (`critic_rewards` in
`miles/ray/rollout.py`), so the value head models "will this be correct" rather than a
mixture of correctness and length. The actor's advantages are unchanged.

## 6. Rollout: over-sampling, dynamic sampling, partial rollout

- `--over-sampling-batch-size 120` with `--rollout-batch-size 60`: sample 2× the prompts,
  train on the first 60 finished groups.
- `--dynamic-sampling-filter-path …check_reward_nonzero_std`: drop groups whose 8 samples
  are all right or all wrong (DAPO).
- `--partial-rollout`: unfinished long trajectories are carried into the next step's buffer
  instead of blocking the step; truncated importance sampling (`--use-tis`, IcePop clip
  `[0.5, 5]`) corrects for the resulting off-policy-ness.

## 7. Optional: DSpark speculative decoding

Set `DSPARK_DRAFT_MODEL_PATH` to a MiniCPM-2B Draft-5L model and SGLang runs DSpark
speculative decoding (block size 7). Requires the sglang fork commit in
`third_party/README.md`. Off by default; when off the launcher passes nothing.
