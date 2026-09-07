unset http_proxy
unset HTTP_PROXY
unset https_proxy
unset HTTPS_PROXY
unset no_proxy
unset NO_PROXY
unset PYTORCH_CUDA_ALLOC_CONF

export WORK_DIR=`pwd`
export PYTHONUNBUFFERED=1
export HF_TRUST_REMOTE_CODE="1"
# export NCCL_SOCKET_NTHREADS=2
# export NCCL_NSOCKS_PERTHREAD=8
export PYTHONPATH=/root/.cache/huggingface/modules:${WORK_DIR}/Megatron-LM:${PYTHONPATH:-}
export USE_FP8_GEMM=0
export NCCL_DEBUG=WARN
# export CUDA_LAUNCH_BLOCKING=1

# Override via env to match your node (e.g. export GPUS_PER_NODE=8 for an H100 node).
if [ -z "${GPUS_PER_NODE+x}" ]; then
    export GPUS_PER_NODE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
else
    export GPUS_PER_NODE="$GPUS_PER_NODE"
fi

# If RANK does not exist, set default values for distributed training.
# For a single node this gives a self-contained 1-node setup; set MASTER_ADDR
# yourself when launching a multi-node cluster.
if [ -z "${RANK+x}" ]; then
    export RANK=0
    export WORLD_SIZE=1
    export MASTER_ADDR=$(hostname)
    export MASTER_PORT=23456
fi

# Single node: default to 1 actor node so launch scripts can split GPUs within
# the node (e.g. 2 train + 6 rollout). Leave ACTOR_NUM_NODES unset for multi-node
# so each launch script's node-ratio decides it.
if [ -z "${ACTOR_NUM_NODES+x}" ] && [ "${WORLD_SIZE}" -le 1 ]; then
    export ACTOR_NUM_NODES=1
fi

export HYDRA_FULL_ERROR=1
export VLLM_USE_V1=1
# export VLLM_LOGGING_LEVEL=DEBUG
export NCCL_DEBUG=WARN # TRACE or INFO or WARN

# Ray log rotation caps (guard against unbounded growth on long runs).
export RAY_BACKEND_LOG_LEVEL=${RAY_BACKEND_LOG_LEVEL:-info}
export RAY_ROTATION_MAX_BYTES=${RAY_ROTATION_MAX_BYTES:-134217728}
export RAY_ROTATION_BACKUP_COUNT=${RAY_ROTATION_BACKUP_COUNT:-2}
export RAY_CGRAPH_submit_timeout=3000
export RAY_CGRAPH_get_timeout=3000

echo "PyTorch NCCL version:"
python -c "import torch; print(torch.cuda.nccl.version())"
echo "NCCL related environment variables:"
env | grep -E "NCCL*|LD_LIBRARY_PATH"

export PROJECT_NAME=${PROJECT_NAME:-miles_open}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-math_ppo}

ulimit -n 1048576 || true

hostname -i
nvidia-smi
free -h

# SwanLab / WandB tracking. Set credentials via env (SWANLAB_API_KEY / WANDB_API_KEY);
# no secret is hardcoded here. Log directories are chosen by train.sh (under SAVE_DIR).
export SWANLAB_MODE=${SWANLAB_MODE:-cloud}

# Set http_proxy/https_proxy/no_proxy yourself if your cluster needs an egress
# proxy. This file intentionally ships with none.

rm -f /etc/pip.conf 2>/dev/null || true

mkdir -p /root/.cache/huggingface/modules/transformers_modules/

export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/ray/rank_${RANK}}
export TMPDIR=${TMPDIR:-/tmp}
mkdir -p $RAY_TMPDIR
mkdir -p $TMPDIR

export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/triton_cache/${JOB_ID:-default}_rank${RANK}}
mkdir -p $TRITON_CACHE_DIR
export HF_HOME=${HF_HOME:-/tmp/huggingface}
mkdir -p $HF_HOME
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-/tmp/torch_extensions}
mkdir -p $TORCH_EXTENSIONS_DIR
export TORCH_HOME=${TORCH_HOME:-/tmp/torch_home}
mkdir -p $TORCH_HOME

export RAY_health_check_failure_threshold=10
export RAY_health_check_period_ms=5000
