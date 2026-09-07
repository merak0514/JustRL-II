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

hf download "$HF_BASE" --local-dir "$MODELS_DIR/$BASE_DIR_NAME"

if [ -f "$MODELS_DIR/$BASE_DIR_NAME-torch_dist/latest_checkpointed_iteration.txt" ]; then
  echo "torch_dist checkpoint already exists, skipping conversion"
else
  source justrl2/model_args/minicpm-2b.sh
  PYTHONPATH=.:Megatron-LM python tools/convert_hf_to_torch_dist.py \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "$MODELS_DIR/$BASE_DIR_NAME" \
    --save "$MODELS_DIR/$BASE_DIR_NAME-torch_dist"
fi

echo
echo "export MODELS_DIR=$MODELS_DIR    # train.sh defaults derive from this"
