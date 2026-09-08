#!/bin/bash
# Download the JustRL2 model weights from the Hugging Face Hub and convert the base
# model to the Megatron torch_dist format that training loads (--ref-load).
#
#   bash justrl2/prepare_model.sh [models_dir]
#
# Produces under <models_dir> (default ./models):
#   JustRL-II-base-model/             HF checkpoint (openbmb/JustRL-II-base-model,
#                                     stock LlamaForCausalLM, bf16)          -> HF_MODEL_DIR
#   JustRL-II-base-model-torch_dist/  Megatron ckpt, TP1                     -> MEGATRON_MODEL_PATH
#
# The conversion needs Megatron-LM and the megatron-bridge package (both in the
# radixark/miles image, see Dockerfile) and one GPU.
# Point MEGATRON_MODEL_PATH at the *parent* torch_dist directory, never at an iter_xxx
# subdirectory: Megatron then silently starts from random weights.

set -eo pipefail
MODELS_DIR=${1:-$PWD/models}
HF_BASE=${HF_BASE:-openbmb/JustRL-II-base-model}
BASE_DIR_NAME=$(basename "$HF_BASE")
mkdir -p "$MODELS_DIR"

# `hf` is the current CLI; `huggingface-cli` is the pre-0.34 name. Accept either so the
# script does not depend on how new the huggingface_hub in your image happens to be.
if command -v hf >/dev/null 2>&1; then
  hf download "$HF_BASE" --local-dir "$MODELS_DIR/$BASE_DIR_NAME"
elif command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli download "$HF_BASE" --local-dir "$MODELS_DIR/$BASE_DIR_NAME"
else
  echo "FATAL: neither 'hf' nor 'huggingface-cli' found; pip install -U huggingface_hub" >&2
  exit 1
fi

if [ -f "$MODELS_DIR/$BASE_DIR_NAME-torch_dist/latest_checkpointed_iteration.txt" ]; then
  echo "torch_dist checkpoint already exists, skipping conversion"
else
  source justrl2/model_args/minicpm5-2b.sh
  # The converter is a single process, but it reads WORLD_SIZE from the environment (and
  # asserts it is <= num_layers). Pin it, or a scheduler that already exported WORLD_SIZE
  # for the training job makes this one process configure itself for that many ranks.
  WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 \
  PYTHONPATH=.:Megatron-LM python tools/convert_hf_to_torch_dist.py \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "$MODELS_DIR/$BASE_DIR_NAME" \
    --save "$MODELS_DIR/$BASE_DIR_NAME-torch_dist"
fi

echo
echo "export MODELS_DIR=$MODELS_DIR    # train.sh defaults derive from this"
