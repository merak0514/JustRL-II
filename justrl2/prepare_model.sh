#!/bin/bash
# Download the JustRL2 model weights from the Hugging Face Hub and convert the base
# model to the Megatron torch_dist format that training loads (--ref-load).
#
#   bash justrl2/prepare_model.sh [models_dir]
#
# Produces under <models_dir> (default ./models):
#   MiniCPM-2B/               HF checkpoint  (openbmb/MiniCPM-2B)      -> HF_MODEL_DIR
#   MiniCPM-2B-torch_dist/    Megatron ckpt, TP1                          -> MEGATRON_MODEL_PATH
#   MiniCPM-2B-DSpark-5L/     optional 5-layer draft for DSpark            -> DSPARK_DRAFT_MODEL_PATH
#
# The conversion needs the Megatron-LM submodule (see third_party/README.md) and one GPU.
# Point MEGATRON_MODEL_PATH at the *parent* torch_dist directory, never at an iter_xxx
# subdirectory: Megatron then silently starts from random weights.

set -eo pipefail
MODELS_DIR=${1:-$PWD/models}
HF_BASE=${HF_BASE:-openbmb/MiniCPM-2B}
HF_DRAFT=${HF_DRAFT:-openbmb/MiniCPM-2B-DSpark-5L}
WITH_DSPARK=${WITH_DSPARK:-0}
mkdir -p "$MODELS_DIR"

hf download "$HF_BASE" --local-dir "$MODELS_DIR/MiniCPM-2B"
if [ "$WITH_DSPARK" = 1 ]; then
  hf download "$HF_DRAFT" --local-dir "$MODELS_DIR/MiniCPM-2B-DSpark-5L"
fi

if [ -f "$MODELS_DIR/MiniCPM-2B-torch_dist/latest_checkpointed_iteration.txt" ]; then
  echo "torch_dist checkpoint already exists, skipping conversion"
else
  source justrl2/model_args/minicpm-2b.sh
  PYTHONPATH=.:Megatron-LM:mbridge python tools/convert_hf_to_torch_dist.py \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "$MODELS_DIR/MiniCPM-2B" \
    --save "$MODELS_DIR/MiniCPM-2B-torch_dist"
fi

echo
echo "export MODELS_DIR=$MODELS_DIR    # train.sh defaults derive from this"
[ "$WITH_DSPARK" = 1 ] && echo "export DSPARK_DRAFT_MODEL_PATH=$MODELS_DIR/MiniCPM-2B-DSpark-5L"
true
