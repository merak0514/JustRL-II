# MiniCPM5-2.6B model args (Llama dense architecture)
# Source:
# MiniCPM5-2.6B-0426_job_327123_step_24000_fusion_think/config.json
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
   --max-position-embeddings 65536
   --normalization RMSNorm
   --norm-epsilon 1e-6
   --untie-embeddings-and-output-weights
   --no-masked-softmax-fusion
   --no-rope-fusion
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
)
