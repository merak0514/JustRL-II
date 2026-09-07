# Submodules / model forks

The #12 pipeline reproduces against **three private forks** of public frameworks. The
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

- **Megatron**: `docker/patch/latest/megatron.patch` — applied by `scripts/setup/setup.sh`.
  This is what wires miles' **scalar value-head critic** (`LinearForLastLayer`) and the
  MiniCPM5 family into Megatron. Clone NVIDIA Megatron-LM, apply the patch, and the
  `afedb9da…` fork's miles-facing behavior is recovered. The fork also carries MiniCPM5
  architecture support (dense Llama-arch with QKV GQA, RMSNorm, svg activated) — if your
  upstream lacks the MiniCPM5 config, port the `minicpm5` config/attention bits.
- **sglang**: `docker/patch/latest/sglang.patch` mirrors the fork. The **DSpark** speculative
  decoding (`--sglang-speculative-algorithm DSPARK`, draft model, block size, draft window)
  is a newer fork feature than the base sglang release; to reproduce it you need the
  sglang commit `fc203f00e` (or an upstream that already merged DSpark).
- **mbridge**: a thin bridge that loads a HuggingFace checkpoint into Megatron and back
  (`model_provider.megatron_to_hf_mode == "bridge"`). Reconstruct it from
  `miles/backends/megatron_utils/model_provider.py` + `megatron_bridge_utils.py`.

### Option C — minimal reconstruction (skip the forks)
The pieces that make #12's critic work are **in the miles tree**, not the forks:

- `miles/backends/megatron_utils/model_provider.py` → `LinearForLastLayer` (scalar value
  head, zero-init).
- `miles/backends/megatron_utils/checkpoint.py` → `_rezero_critic_value_head` (re-zero after
  policy-ckpt load + fp32 master resync).
- `miles/utils/ppo_utils.py` → VAPO length-adaptive GAE, partial-rollout, over-sampling.

If you only need to study those, you don't strictly need the forks; you need them to *run*
the full Megatron rollout.

## Model weights (public)

The base model is MiniCPM5-2.6B (dense, Llama-arch). #12 uses:

- **HF checkpoint** (base, for the HF/`--hf-checkpoint` path)
- **Megatron `torch_dist` checkpoint** (for `--ref-load` / `--tokenizer`; must be a `tp1`
  torch-dist conversion)

Both need to be downloadable from a public hub. Because MiniCPM5-2.6B may not be on
HuggingFace under that exact name, treat these as "download from your model hub / ModelScope /
HF and place in `MODELS_DIR`":

```bash
export MODELS_DIR=$PWD/models/cache
mkdir -p "$MODELS_DIR"
# HF  -> $MODELS_DIR/MiniCPM5-2.6B-0426-24000           (config.json + *.safetensors)
# Mg  -> $MODELS_DIR/MiniCPM5-2.6B-0426-24000-tp1-torch-dists
```

The `run` script will refuse to start if `MODELS_DIR` is missing, so a wrong path cannot
silently fall through to a fresh-init run.

**DSpark draft model** (`DSPARK_DRAFT_MODEL_PATH`): a MiniCPM5-2.6B **Draft-5L** (5-layer
decoder) used for speculative decoding, `block_size=7` in its config. This is also model-best
internal; get a public-equivalent 5-layer MiniCPM5 draft or pin the DSpark draft checkpoint
you have. Without it, leave `DSPARK_DRAFT_MODEL_PATH` empty — the DSpark block is then a
no-op and the run is plain (non-speculative) decoding.
