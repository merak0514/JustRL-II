# MiniCPM5-2B model args for Megatron (dense Llama architecture).
# Mirrors config.json of openbmb/JustRL-II-base-model, the RL initialization
# checkpoint of MiniCPM5-2B (openbmb/MiniCPM5-2B).
MODEL_ARGS=(
   --swiglu
   --disable-bias-linear
   --num-layers 42
   --hidden-size 2048
   --ffn-hidden-size 6144
   --num-attention-heads 16
   --group-query-attention
   --num-query-groups 2
   --kv-channels 128
   --vocab-size 130560
   --make-vocab-size-divisible-by 64
   --position-embedding-type rope
   --rotary-percent 1.0
   --rotary-base "${MODEL_ARGS_ROTARY_BASE:-5000000}"
   --max-position-embeddings 65536      # inert: miles sets it from --seq-length (arguments.py)
   --normalization RMSNorm
   --norm-epsilon 1e-6
   --untie-embeddings-and-output-weights
   --no-masked-softmax-fusion
   --no-rope-fusion
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
)
