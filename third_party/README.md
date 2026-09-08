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

## Bare metal: clone the two frameworks

If you are not using the image, clone the current version of each next to the repo root.
There is nothing to pin — the recipe uses only stable interfaces of both projects, so the
latest of each is the right choice:

```bash
git clone https://github.com/radixark/Megatron-LM.git -b miles-main Megatron-LM
git clone https://github.com/sgl-project/sglang.git   -b sglang-miles sglang
```

`radixark/Megatron-LM` is Miles' maintained fork of NVIDIA Megatron-LM; it already carries
the dist-checkpointing, TE and distributed-optimizer changes the recipe needs, so no patch
step is required. `third_party/patches/` keeps the equivalent diffs against stock NVIDIA
Megatron-LM and stock SGLang for reference only — you do not need them for a normal setup.

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
