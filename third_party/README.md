# Submodules / model forks

JustRL2 trains against **three private forks** of public frameworks. The
private commits are pinned by the `gitlink` SHAs below; the forks live on the
`codeup.aliyun.com:modelbest/scaling` internal group and are **not** public. This package
ships the miles-side code and the Megatron patch, but **not** the forks themselves.

## What you must obtain

| Submodule | Private commit (gitlink) | Public upstream it forks |
|-----------|--------------------------|--------------------------|
| `Megatron-LM` | `afedb9da1a8b7daf8d071608b75442db87a6aa0d` (`main_0129_lzh_minicpm5_moe`) | NVIDIA Megatron-LM |
| `mbridge` | `d2131a184c1c782c9e1a885c8b813d18f1d1c7e6` (`minicpm`) | model-to-Megatron bridge (see below) |
| `sglang` | `fc203f00e76e5a07d459c77a5c825cb512a237cd` (`minicpm`, DSpark-capable) | sglang-project/sglang |

The launch script expects the checkout at `<repo>/Megatron-LM`, `<repo>/mbridge`,
`<repo>/sglang` and a `PYTHONPATH` of `.:Megatron-LM:mbridge:sglang/python`.

## Getting these three, in order of effort

### Option A — publish the private forks
Recommended if the modelbest forks can be released. Convert each `git@codeup…` URL to a
public clone URL, then:

```bash
git clone --recurse-submodules <public-miles-url> miles
cd miles
# (submodule URLs in .gitmodules must be repointed to the public ones)
git submodule update --init --recursive
git -C Megatron-LM checkout afedb9da1a8b7daf8d071608b75442db87a6aa0d
git -C mbridge       checkout d2131a184c1c782c9e1a885c8b813d18f1d1c7e6
git -C sglang        checkout fc203f00e76e5a07d459c77a5c825cb512a237cd   # DSpark spec decode
```

### Option B — reconstruct against public upstreams + the shipped patch
The miles-side patches are included here:

- **Megatron**: `third_party/patches/megatron.patch` — applied by `justrl2/setup/setup.sh`.
  This is what wires miles' **scalar value-head critic** (`LinearForLastLayer`) and the
  MiniCPM5 family into Megatron. Clone NVIDIA Megatron-LM, apply the patch, and the
  `afedb9da…` fork's miles-facing behavior is recovered. The fork also carries MiniCPM5
  architecture support (dense Llama-arch with QKV GQA, RMSNorm, svg activated) — if your
  upstream lacks the MiniCPM5 config, port the `minicpm5` config/attention bits.
- **sglang**: `third_party/patches/sglang.patch` mirrors the fork. The **DSpark** speculative
  decoding (`--sglang-speculative-algorithm DSPARK`, draft model, block size, draft window)
  is a newer fork feature than the base sglang release; to reproduce it you need the
  sglang commit `fc203f00e` (or an upstream that already merged DSpark).
- **mbridge**: a thin bridge that loads a HuggingFace checkpoint into Megatron and back
  (`model_provider.megatron_to_hf_mode == "bridge"`). Reconstruct it from
  `miles/backends/megatron_utils/model_provider.py` + `megatron_bridge_utils.py`.

### Option C — minimal reconstruction (skip the forks)
The pieces that make the JustRL2 critic work are **in the miles tree**, not the forks:

- `miles/backends/megatron_utils/model_provider.py` → `LinearForLastLayer` (scalar value
  head: zero weight, mean-reward bias).
- `miles/backends/megatron_utils/checkpoint.py` → `_rezero_critic_value_head` (re-init after
  policy-ckpt load + fp32 master resync).
- `miles/utils/ppo_utils.py` → length-adaptive GAE λ=k^(1/L), partial-rollout, over-sampling.

If you only need to study those, you don't strictly need the forks; you need them to *run*
the full Megatron rollout.

## Model weights

`bash justrl2/prepare_model.sh` downloads `openbmb/MiniCPM5-2.6B` (HF format) and converts
it to the Megatron `torch_dist` layout with `tools/convert_hf_to_torch_dist.py`; with
`WITH_DSPARK=1` it also fetches the `openbmb/MiniCPM5-2.6B-DSpark-5L` draft model for
speculative decoding. `train.sh` refuses to start if either model path is missing, so a
typo cannot fall through to a random-init run. Always point `MEGATRON_MODEL_PATH` at the
parent `torch_dist` directory, not an `iter_xxx` subdirectory.
