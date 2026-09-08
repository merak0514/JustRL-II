# Frameworks: Megatron-LM and SGLang

JustRL2 trains with Megatron-LM and samples with SGLang, both through the
[Miles](https://github.com/radixark/miles) RL framework that lives in `miles/` here.
`justrl2/train.sh` expects them checked out (or symlinked) as `<repo>/Megatron-LM` and
`<repo>/sglang`, and puts `.:Megatron-LM:sglang/python` on `PYTHONPATH`.

## Recommended: the community Miles image

The `Dockerfile` at the repo root builds on `radixark/miles:dev`, which already contains
the versions the Miles project tests together:

| component | what the image ships | where |
|---|---|---|
| Megatron-LM | `radixark/Megatron-LM` @ `miles-main` (Miles' maintained fork of NVIDIA Megatron-LM) | `/root/Megatron-LM` |
| SGLang | `sglang-miles` branch of SGLang | `/sgl-workspace/sglang` |
| mbridge | pip package (`ISEEKYAN/mbridge`, HF↔Megatron weight mapping), used only by `tools/convert_hf_to_torch_dist.py`; its `LlamaBridge` covers the base model | site-packages |
| TransformerEngine, apex, flash-attn, sgl-kernel, Ray, torch | pinned by the image | — |

The Dockerfile symlinks the two checkouts into the repo root, installs `miles/` in
editable mode and adds `math-verify`. Nothing else is needed:

```bash
docker build -t justrl2 .
docker run --gpus all --ipc=host --network=host -it justrl2
```

Image tags and how they are built: <https://github.com/radixark/miles/tree/main/docker>.

## Building on stock NVIDIA Megatron-LM instead

If you cannot use the image, the Miles-specific Megatron changes are collected in
`third_party/patches/megatron.patch` (dist-checkpointing, TE extensions, GPT layer specs,
the distributed optimizer, `--override-opt_param-scheduler`, …). Clone NVIDIA Megatron-LM at
the version the patch was cut against (the Miles `nightly-dev-20260113` line), then:

```bash
APPLY_MEGATRON_PATCH=1 bash justrl2/setup/setup.sh     # idempotent, marker file
```

`third_party/patches/sglang.patch` is the corresponding Miles diff on top of the SGLang
release the image is based on; it is provided for reference — use the `sglang-miles`
branch rather than re-applying it.

## The base model is plain Llama

`openbmb/JustRL-II-base-model` is a stock `LlamaForCausalLM` (42 layers, hidden 2048,
16 heads / 2 KV heads, vocab 130560, rope θ = 5e6, 65536 positions in `config.json`;
`justrl2/model_args/minicpm5-2b.sh` mirrors this). No custom modeling code is needed on
either the Megatron or the SGLang side. Two EOS ids are configured
(`eos_token_id = [1, 130073]`); SGLang picks both up from `config.json`. The 128k
generation budget is enabled by `--sglang-context-length 131072` plus the
`max_position_embeddings` override that `train.sh` passes.

## What is *not* in the community stack

**DSpark speculative decoding.** The reference run used a DSpark-capable SGLang build
and a 5-layer draft model to speed up 128k rollouts. That scheduler is not in the
`sglang-miles` branch; leave `DSPARK_DRAFT_MODEL_PATH` empty (the default) and rollouts
run with plain decoding. Throughput is lower; the training recipe and its results do not
depend on it.

## Model weights

`bash justrl2/prepare_model.sh` downloads `openbmb/JustRL-II-base-model` and converts it
to the Megatron `torch_dist` layout with `tools/convert_hf_to_torch_dist.py` (needs one
GPU and the `mbridge` package from the image). `train.sh` refuses to start if either model path or either framework directory is
missing. Always point `MEGATRON_MODEL_PATH` at the parent `torch_dist` directory, not an
`iter_xxx` subdirectory — Megatron would then silently start from random weights.
