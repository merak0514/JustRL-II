#!/bin/bash
# JustRL2 launcher: MiniCPM5-2B math RL with a scalar value-head critic,
# length-adaptive GAE (lambda_i = k^(1/L_i)) and a mean-reward-seeded value head.
#
#   bash justrl2/train.sh justrl2/configs/minicpm5-2b-math-128k.env [extra miles args...]
#
# All knobs live in the .env file (every line there is a default that an exported
# shell variable overrides). This script only turns them into miles arguments and
# submits the Ray job. Run it from the repo root on every node of the cluster; the
# usual torchrun-style RANK / WORLD_SIZE / MASTER_ADDR / MASTER_PORT env decides who
# is the Ray head.
#
# Resuming: point SAVE_ROOT/EXP_TAG at an existing run — actor and critic checkpoints
# are picked up automatically (both must exist; a half-present pair is a hard error).

set -eo pipefail
CONFIG=${1:?usage: bash justrl2/train.sh <config.env> [extra miles args]}
shift
source "$CONFIG"

export WORK_DIR=$(pwd)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source justrl2/setup/env.sh
bash justrl2/setup/setup.sh
set -x

# ---- sanity: model / data must exist (a wrong --ref-load silently random-inits) ----
for d in "$HF_MODEL_DIR" "$MEGATRON_MODEL_PATH" Megatron-LM sglang; do
  [ -e "$d" ] || { echo "FATAL: $d does not exist (models: justrl2/prepare_model.sh; frameworks: third_party/README.md)" >&2; exit 1; }
done
for f in $TRAIN_FILE; do
  [ -f "$f" ] || { echo "FATAL: train file $f does not exist (see justrl2/prepare_data.py)" >&2; exit 1; }
done

export SGLANG_PATH=${WORK_DIR}/sglang
# EXTRA_PYTHONPATH: optional extra entries (e.g. a HF dynamic-modules cache for checkpoints
# that ship custom modeling code); appended for the driver and every Ray worker.
export PYTHONPATH=.:Megatron-LM:${SGLANG_PATH}/python${EXTRA_PYTHONPATH:+:$EXTRA_PYTHONPATH}
source justrl2/model_args/minicpm5-2b.sh

# ---- topology: PPO needs actor and critic world sizes equal (rank-pairwise NCCL groups) ----
ACTOR_NUM_NODES=${ACTOR_NUM_NODES:-$((WORLD_SIZE / 2))}
CRITIC_NUM_NODES=${CRITIC_NUM_NODES:-$((WORLD_SIZE - ACTOR_NUM_NODES))}
ACTOR_NUM_GPUS_PER_NODE=${ACTOR_NUM_GPUS_PER_NODE:-$GPUS_PER_NODE}
CRITIC_NUM_GPUS_PER_NODE=${CRITIC_NUM_GPUS_PER_NODE:-$ACTOR_NUM_GPUS_PER_NODE}
if [ "$WORLD_SIZE" -eq 1 ]; then
  ACTOR_NUM_NODES=1; CRITIC_NUM_NODES=1
  [ $((ACTOR_NUM_GPUS_PER_NODE + CRITIC_NUM_GPUS_PER_NODE)) -le "$GPUS_PER_NODE" ] \
    || { echo "FATAL: single node: actor+critic GPUs exceed GPUS_PER_NODE=$GPUS_PER_NODE" >&2; exit 1; }
fi
[ $((ACTOR_NUM_NODES * ACTOR_NUM_GPUS_PER_NODE)) -eq $((CRITIC_NUM_NODES * CRITIC_NUM_GPUS_PER_NODE)) ] \
  || { echo "FATAL: actor world size must equal critic world size" >&2; exit 1; }

GLOBAL_BATCH_SIZE=$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))
ACTOR_DP=$((ACTOR_NUM_NODES * ACTOR_NUM_GPUS_PER_NODE / TENSOR_MODEL_PARALLEL_SIZE / CONTEXT_PARALLEL_SIZE))
[ $((GLOBAL_BATCH_SIZE % ACTOR_DP)) -eq 0 ] \
  || { echo "FATAL: GBS=$GLOBAL_BATCH_SIZE not divisible by actor DP=$ACTOR_DP; adjust ROLLOUT_BATCH_SIZE" >&2; exit 1; }

# ---- checkpoints ----------------------------------------------------------------
SAVE_DIR=${SAVE_DIR:-${SAVE_ROOT}/${EXP_TAG}}
CRITIC_SAVE_DIR=${CRITIC_SAVE_DIR:-${SAVE_DIR%/}_critic}
EXP_NAME="${EXP_TAG}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$SAVE_DIR" "$CRITIC_SAVE_DIR"
# Logs go next to the checkpoints (shared storage), never to the pod-local disk: a full
# local disk makes the logger block and training stall silently while the job shows Running.
export TENSORBOARD_DIR=${TENSORBOARD_DIR:-${SAVE_DIR}/tensorboard}
export SWANLAB_LOG_DIR=${SWANLAB_LOG_DIR:-${SAVE_DIR}/swanlog}
mkdir -p "$TENSORBOARD_DIR" "$SWANLAB_LOG_DIR"

# Rotation: Megatron keeps the latest torch_dist checkpoint plus every iteration that is a
# multiple of --save-retain-interval, and rotates the rest away. Rollout ids are 0-based
# (iter_0000009/19/...), so a retain interval that is a large multiple of SAVE_INTERVAL never
# hits a milestone and exactly one actor + one critic checkpoint stays on disk — 63 GB total
# here, instead of 63 GB per save. Long-term archiving is the HF export below, not this.
# Megatron asserts save_retain_interval % save_interval == 0 on a fresh start; fail early
# with a readable message instead of dying inside argument validation an hour into the queue.
[ $((SAVE_RETAIN_INTERVAL % SAVE_INTERVAL)) -eq 0 ] \
  || { echo "FATAL: SAVE_RETAIN_INTERVAL=$SAVE_RETAIN_INTERVAL must be a multiple of SAVE_INTERVAL=$SAVE_INTERVAL" >&2; exit 1; }
CKPT_ARGS=(
  --hf-checkpoint "$HF_MODEL_DIR"
  --ref-load "$MEGATRON_MODEL_PATH"
  --load "$SAVE_DIR"
  --save "$SAVE_DIR"
  --save-interval "$SAVE_INTERVAL"
  --save-retain-interval "$SAVE_RETAIN_INTERVAL"
  --critic-save "$CRITIC_SAVE_DIR"
)
if [ "$HF_SAVE_INTERVAL" != "0" ]; then
  CKPT_ARGS+=(--save-hf "${SAVE_DIR}/hf/iter_{rollout_id:07d}" --save-hf-interval "$HF_SAVE_INTERVAL")
fi
# First run: --critic-load is left unset, so miles defaults it to --load (arguments.py),
# which the finetune fallback has already pointed at --ref-load because SAVE_DIR holds no
# latest_checkpointed_iteration.txt. The critic therefore starts from the base model and
# checkpoint.py re-zeroes the value-head weight / re-seeds its bias after that load.
# Resume: both actor and critic checkpoints must exist and are loaded explicitly.
ACTOR_HAS_CKPT=$([ -f "${SAVE_DIR}/latest_checkpointed_iteration.txt" ] && echo 1 || echo 0)
CRITIC_HAS_CKPT=$([ -f "${CRITIC_SAVE_DIR}/latest_checkpointed_iteration.txt" ] && echo 1 || echo 0)
if [ "$ACTOR_HAS_CKPT" = 1 ] && [ "$CRITIC_HAS_CKPT" = 1 ]; then
  CKPT_ARGS+=(--critic-load "$CRITIC_SAVE_DIR")
elif [ "$ACTOR_HAS_CKPT" != "$CRITIC_HAS_CKPT" ]; then
  echo "FATAL: actor ckpt present=$ACTOR_HAS_CKPT but critic ckpt present=$CRITIC_HAS_CKPT in $SAVE_DIR / $CRITIC_SAVE_DIR;" \
       "resume needs both (or neither). Remove the stray one." >&2
  exit 1
fi
# Needed on the first resume after changing NUM_ROLLOUT (Megatron asserts on the saved schedule).
[ "${OVERRIDE_OPT_PARAM_SCHEDULER:-0}" = 1 ] && CKPT_ARGS+=(--override-opt_param-scheduler)

# ---- rollout ------------------------------------------------------------------
ROLLOUT_ARGS=(
  --prompt-data $TRAIN_FILE
  --input-key prompt
  --label-key label
  --apply-chat-template
  --rollout-shuffle
  --rm-type math
  --num-rollout "$NUM_ROLLOUT"
  --rollout-batch-size "$ROLLOUT_BATCH_SIZE"
  --n-samples-per-prompt "$N_SAMPLES_PER_PROMPT"
  --rollout-max-prompt-len "$ROLLOUT_MAX_PROMPT_LEN"
  --rollout-max-response-len "$ROLLOUT_MAX_RESPONSE_LEN"
  --rollout-max-context-len "$ROLLOUT_MAX_CONTEXT_LEN"
  --rollout-temperature "$ROLLOUT_TEMPERATURE"
  --rollout-top-p 1.0
  --global-batch-size "$GLOBAL_BATCH_SIZE"
  --balance-data
  --over-sampling-batch-size "$OVER_SAMPLING_BATCH_SIZE"
)
[ "$DYNAMIC_SAMPLING" = 1 ] && ROLLOUT_ARGS+=(--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std)
[ "$ENABLE_PARTIAL_ROLLOUT" = 1 ] && ROLLOUT_ARGS+=(--partial-rollout)

# ---- the recipe ------------------------------------------------------------------
PPO_ARGS=(
  --advantage-estimator ppo
  --gamma "$GAMMA"
  --lambd 1.0                              # unused when --gae-lambda-k is set
  --gae-lambda-k "$GAE_LAMBDA_K"
  --critic-value-bias-init "$CRITIC_VALUE_BIAS_INIT"
  --num-critic-only-steps "$NUM_CRITIC_ONLY_STEPS"
  --critic-lr "$CRITIC_LR"
  --critic-lr-warmup-iters "$CRITIC_LR_WARMUP_ITERS"
  --normalize-advantages
  --calculate-per-token-loss
  --eps-clip "$EPS_CLIP"
  --eps-clip-high "$EPS_CLIP_HIGH"
  --value-clip "$VALUE_CLIP"
  --kl-loss-coef 0.0
  --kl-loss-type low_var_kl
  --entropy-coef 0.0
  --overlong-buffer-len "$OVERLONG_BUFFER_LEN"
  --overlong-penalty-factor "$OVERLONG_PENALTY_FACTOR"
  --use-tis
  --custom-tis-function-path miles.backends.training_utils.loss.icepop_function
  --tis-clip-low "$TIS_CLIP_LOW"
  --tis-clip "$TIS_CLIP"
)
[ "$CRITIC_EXCLUDE_OLP" = 1 ] && PPO_ARGS+=(--critic-exclude-overlong-penalty)

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr "$LR"
  --lr-decay-style constant
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98
)

PERF_ARGS=(
  --train-backend megatron
  --tensor-model-parallel-size "$TENSOR_MODEL_PARALLEL_SIZE"
  --context-parallel-size "$CONTEXT_PARALLEL_SIZE"
  --sequence-parallel
  --pipeline-model-parallel-size 1
  --expert-model-parallel-size 1
  --expert-tensor-parallel-size 1
  --attention-backend flash
  --recompute-granularity full          # do not switch to selective at 128k: single 127k samples OOM
  --recompute-method uniform
  --recompute-num-layers 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu "$MAX_TOKENS_PER_GPU"
  # Passed for parity with the reference run; miles logs it as UNUSED and reads it nowhere.
  --log-probs-max-tokens-per-gpu $((MAX_TOKENS_PER_GPU * CONTEXT_PARALLEL_SIZE))
  --log-probs-chunk-size "$LOG_PROBS_CHUNK_SIZE"
  --attention-dropout 0.0
  --hidden-dropout 0.0
)

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine 1
  --sglang-mem-fraction-static "$SGLANG_MEM_FRACTION"
  --sglang-max-running-requests "$SGLANG_MAX_RUNNING_REQUESTS"
  --sglang-schedule-conservativeness 1.2
  --sglang-attention-backend "$SGLANG_ATTENTION_BACKEND"
  --sglang-log-level warning
  --sglang-json-model-override-args "{\"max_position_embeddings\": ${ROLLOUT_MAX_CONTEXT_LEN}}"
  --sglang-context-length "$ROLLOUT_MAX_CONTEXT_LEN"
  --sglang-max-prefill-tokens "$ROLLOUT_MAX_CONTEXT_LEN"
  --sglang-max-total-tokens "$SGLANG_MAX_TOTAL_TOKENS"
  --sglang-enable-metrics
  --log-rollout-mbu
)
DSPARK_ENV=""
if [ -n "$DSPARK_DRAFT_MODEL_PATH" ]; then
  SGLANG_ARGS+=(
    --sglang-speculative-algorithm DSPARK
    --sglang-speculative-draft-model-path "$DSPARK_DRAFT_MODEL_PATH"
    --sglang-speculative-dspark-block-size "$DSPARK_BLOCK_SIZE"
    --sglang-speculative-draft-window-size "$DSPARK_DRAFT_WINDOW"
  )
  DSPARK_ENV='"SGLANG_RAGGED_VERIFY_MODE": "static",'
fi

EVAL_ARGS=(
  --eval-prompt-data $TEST_FILE
  --eval-interval "$EVAL_INTERVAL"
  --eval-temperature "$EVAL_TEMPERATURE"
  --eval-top-p "$EVAL_TOP_P"
  --eval-max-response-len "$ROLLOUT_MAX_RESPONSE_LEN"
  --n-samples-per-eval-prompt "$N_SAMPLES_PER_EVAL_PROMPT"
)

# ---- post-hoc analysis dumps -------------------------------------------------------
# DUMP_DIR unset (the default) means no dump at all. When set, miles derives
# rollout_data/{rollout_id}.pt (per-sample trajectories: prompt, response, reward) and
# train_data/{rollout_id}_{rank}.pt (per-token values, advantages, returns, log-probs)
# from --dump-details, and the same directory receives the IcePop/TIS scatter samples.
# Budget: ~1 GB per step for the two data dumps at this batch/length, plus ~0.5 GB per
# step of policy_loss_debug, which every rank writes on every microbatch and which is the
# one subdirectory that is safe to delete afterwards.
DUMP_ARGS=()
if [ -n "$DUMP_DIR" ]; then
  mkdir -p "$DUMP_DIR"
  DUMP_ARGS+=(--dump-details "$DUMP_DIR" --icepop-dump-dir "$DUMP_DIR")
  [ -n "$ICEPOP_DUMP_STEPS" ] && DUMP_ARGS+=(--icepop-dump-steps $ICEPOP_DUMP_STEPS)
fi

TRACK_ARGS=(--use-tensorboard)
[ "$USE_WANDB" = 1 ] && TRACK_ARGS+=(--use-wandb --wandb-project "${WANDB_PROJECT:-justrl2}")
# The project name is read from MILES_SWANLAB_PROJECT, not SWANLAB_PROJECT: the swanlab SDK
# parses SWANLAB_* env vars at import time and a plain string there fails validation.
if [ "$USE_SWANLAB" = 1 ]; then
  unset SWANLAB_PROJECT
  TRACK_ARGS+=(--use-swanlab --swanlab-project "${MILES_SWANLAB_PROJECT:-justrl2}" --swanlab-experiment-name "$EXP_NAME")
fi

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -c 'NV[0-9]' || true); NVLINK_COUNT=${NVLINK_COUNT:-0}
RUNTIME_ENV_JSON="{
  \"env_vars\": {${DSPARK_ENV}
    \"PYTHONPATH\": \"${PYTHONPATH}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"$([ "$NVLINK_COUNT" -gt 0 ] && echo 1 || echo 0)\",
    \"NVTE_FUSED_ATTN\": \"0\",
    \"NVTE_UNFUSED_ATTN\": \"0\",
    \"SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN\": \"1\",
    \"SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION\": \"10000\",
    \"SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK\": \"1\",
    \"SGLANG_LOGITS_PROCESSER_CHUNK_SIZE\": \"2048\",
    \"TRITON_CACHE_DIR\": \"${TRITON_CACHE_DIR}\",
    \"TMPDIR\": \"${TMPDIR}\",
    \"MILES_EXPERIMENTAL_ROLLOUT_REFACTOR\": \"1\"
  }
}"

bash justrl2/setup/ray_start.sh
echo "============================================================"
echo "JustRL2  exp=${EXP_NAME}"
echo "actor ${ACTOR_NUM_NODES}x${ACTOR_NUM_GPUS_PER_NODE}  critic ${CRITIC_NUM_NODES}x${CRITIC_NUM_GPUS_PER_NODE}  TP${TENSOR_MODEL_PARALLEL_SIZE} CP${CONTEXT_PARALLEL_SIZE}  GBS=${GLOBAL_BATCH_SIZE}"
echo "k=${GAE_LAMBDA_K}  bias_init=${CRITIC_VALUE_BIAS_INIT}  critic_only=${NUM_CRITIC_ONLY_STEPS}  exclude_olp=${CRITIC_EXCLUDE_OLP}"
echo "save=${SAVE_DIR}  retain=${SAVE_RETAIN_INTERVAL}  dump=${DUMP_DIR:-off}"
echo "============================================================"

ray job submit --address="http://${MASTER_ADDR}:8265" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 train.py \
  --actor-num-nodes "$ACTOR_NUM_NODES" \
  --actor-num-gpus-per-node "$ACTOR_NUM_GPUS_PER_NODE" \
  --critic-num-nodes "$CRITIC_NUM_NODES" \
  --critic-num-gpus-per-node "$CRITIC_NUM_GPUS_PER_NODE" \
  --colocate \
  --skip-eval-before-train \
  "${MODEL_ARGS[@]}" "${CKPT_ARGS[@]}" "${ROLLOUT_ARGS[@]}" "${PPO_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" "${PERF_ARGS[@]}" "${SGLANG_ARGS[@]}" "${EVAL_ARGS[@]}" \
  "${DUMP_ARGS[@]}" "${TRACK_ARGS[@]}" "$@"

bash justrl2/setup/ray_end.sh
echo "JustRL2 done: ${EXP_NAME}"
