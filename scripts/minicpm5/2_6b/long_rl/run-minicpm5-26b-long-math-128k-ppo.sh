#!/bin/bash
# MiniCPM5 2.6b long RL, PPO (critic-based GAE) 版本, colocate。
# 派生自 run-minicpm5-26b-long-math-128k-partial-rollout.sh，差异：
#   1. --advantage-estimator ppo：独立 critic 训练组（GAE advantage，不做组内归一化）
#   2. 节点对半分：actor N/2 + critic N/2（代码硬约束 actor/critic world size 必须相等，
#      connect() 按 rank 逐对建 NCCL group）
#   3. critic ckpt：--critic-save 指向 ${SAVE_DIR}_critic；首训不传 --critic-load
#      （继承 finetune fallback 后的底模），续训显式传 critic 自己的目录
#   4. 去掉 GRPO 专属项：dynamic sampling filter / over-sampling / disable-grpo-std-normalization
#      （PPO 下零方差组仍有 critic baseline 提供的梯度，过滤反而丢数据）
#   5. partial rollout 默认关（首版减少变量），ENABLE_PARTIAL_ROLLOUT=1 可开
# overlong soft punishment 与数据集与 GRPO 基准保持一致（可从 entry 环境变量覆盖）。

set -x
unset http_proxy HTTP_PROXY https_proxy HTTPS_PROXY no_proxy NO_PROXY

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CLUSTER=${CLUSTER:-ningxia_h100}

source scripts/setup/env.sh
bash scripts/setup/setup.sh
export SLIME_SCRIPT_NUM_GPUS=$GPUS_PER_NODE

BASE_DIR=${BASE_DIR:-$WORK_DIR/runs/$PROJECT_NAME/$JOB_ID}
mkdir -p "$BASE_DIR"

# Model / dataset roots. Set these to point at wherever you downloaded the public
# weights and data (see README.md / models/ / scripts/data). A missing model root
# is a hard error here so a typo can never silently fall through to a fresh-init
# training run (see memory: ref-load 子目录默随机初始化).
export MODELS_DIR=${MODELS_DIR:-$WORK_DIR/models/cache}
export DATA_DIR=${DATA_DIR:-$WORK_DIR/datasets}
if [ ! -e "$MODELS_DIR" ]; then
  echo "FATAL: MODELS_DIR=$MODELS_DIR does not exist. Set MODELS_DIR to the dir holding the MiniCPM5 weights/torch_dist ckpt (see README)." >&2
  exit 1
fi

export SGLANG_PATH=${WORK_DIR}/sglang
export ARG_MODE=true_on_policy
export WANDB_API_KEY=${WANDB_API_KEY:-i_dont_have_wandb_key}
USE_WANDB=${USE_WANDB:-0}

if [ "${SKIP_KILL:-0}" != "1" ]; then
  pkill -9 sglang || true
  sleep 3
  ray stop --force || true
  pkill -9 ray || true
  pkill -f "python3 train.py" || true
  sleep 3
  pkill -9 ray || true
  pkill -f "python3 train.py" || true
  pkill -9 redis || true
else
  echo "SKIP_KILL=1: skip pkill/ray stop cleanup"
fi

set -ex
export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || echo 0)
HAS_NVLINK=$([[ "$NVLINK_COUNT" -gt 0 ]] && echo 1 || echo 0)
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

HF_MODEL_DIR=${HF_MODEL_DIR:-$MODELS_DIR}
HF_MODULES_DIR=${HF_MODULES_DIR:-/root/.cache/huggingface/modules}
HF_DYNAMIC_MODULE_NAME="$(basename "${HF_MODEL_DIR}" | sed 's/\./_dot_/g; s/-/_/g')"
HF_DYNAMIC_MODULE_DIR="${HF_MODULES_DIR}/transformers_modules/${HF_DYNAMIC_MODULE_NAME}"
mkdir -p "${HF_DYNAMIC_MODULE_DIR}"
touch "${HF_MODULES_DIR}/transformers_modules/__init__.py" "${HF_DYNAMIC_MODULE_DIR}/__init__.py"
for py_file in configuration_minicpm.py modeling_minicpm.py; do
  if [ -f "${HF_MODEL_DIR}/${py_file}" ]; then
    cp "${HF_MODEL_DIR}/${py_file}" "${HF_DYNAMIC_MODULE_DIR}/"
  fi
done

export MODEL_PATH=${HF_MODEL_DIR}
export MEGATRON_MODEL_PATH=${MEGATRON_MODEL_PATH:-${MODELS_DIR}/MiniCPM5-2.6B-0426-24000-tp1-torch-dists}   # torch_dist Megatron ckpt; see README
export CKPT_PATH=${CKPT_PATH:-${BASE_DIR}/checkpoints}
export PYTHONPATH=.:Megatron-LM/:mbridge:${SGLANG_PATH}/python:${HF_MODULES_DIR}:${WORK_DIR}/examples/retool_v2
source "${WORK_DIR}/scripts/minicpm5/2_6b/minicpm5-26b.sh"

MODEL_ARGS_SELECTED=("${MODEL_ARGS[@]}")
MODEL_TAG=minicpm5_26b
TENSOR_MODEL_PARALLEL_SIZE=${TENSOR_MODEL_PARALLEL_SIZE:-1}
CONTEXT_PARALLEL_SIZE=${CONTEXT_PARALLEL_SIZE:-4}
ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}
EXPERT_MODEL_PARALLEL_SIZE=${EXPERT_MODEL_PARALLEL_SIZE:-1}
OPTIMIZER=${OPTIMIZER:-adam}

# --- DSpark 投机采样挂点（默认关闭）-----------------------------------------
# DSPARK_DRAFT_MODEL_PATH 为空 => 完全不启用；此时生成的 sglang 参数列表与
# runtime env 与加本块之前完全一致。在跑的 128k 臂复用本脚本，默认路径不能被影响。
#
# draft 与 target 必须配对，错配会掉接受率。draft 模型（MiniCPM5-2.6B-DSpark-5L，5 层 decoder，
# config.json 里 block_size=7）需自备，见 README「DSpark 投机采样」一节：DSPARK_DRAFT_MODEL_PATH
# 指向本地下载的 draft 目录。日期前缀=target 底模日期的命名约定：draft 必须是按你实际使用的
# 底模训出来的那一份。参考接受率：0.40 / 平均接受长度 3.80 @ gamma 7（128k midtrain 配对）。
#
# DSPARK_BLOCK_SIZE 默认 7：三个 draft 的 config.json 都是 block_size=7（已核对），不要改。
# 128k 显存：ragged verify 的 graph capture 比纯 decode 更吃显存（short_rl 的 dspark
# 脚本为此把 mem-fraction 从 0.85 调到 0.80）。本脚本的 SGLANG_MEM_FRACTION（见下方，
# 默认 0.88）本身就是 env 可覆盖的；启用 DSpark 后若 OOM，从那里调低，默认值保持不动。
DSPARK_DRAFT_MODEL_PATH=${DSPARK_DRAFT_MODEL_PATH:-}
DSPARK_BLOCK_SIZE=${DSPARK_BLOCK_SIZE:-7}
DSPARK_DRAFT_WINDOW=${DSPARK_DRAFT_WINDOW:-4096}
SGLANG_RAGGED_VERIFY_MODE=${SGLANG_RAGGED_VERIFY_MODE:-static}

SGLANG_EXTRA_ARGS=()
DSPARK_ENV_JSON_EXTRA=""
if [ -n "${DSPARK_DRAFT_MODEL_PATH}" ]; then
  SGLANG_EXTRA_ARGS+=(
    --sglang-speculative-algorithm DSPARK
    --sglang-speculative-draft-model-path "${DSPARK_DRAFT_MODEL_PATH}"
    --sglang-speculative-dspark-block-size "${DSPARK_BLOCK_SIZE}"
    --sglang-speculative-draft-window-size "${DSPARK_DRAFT_WINDOW}"
  )
  DSPARK_ENV_JSON_EXTRA="\"SGLANG_RAGGED_VERIFY_MODE\": \"${SGLANG_RAGGED_VERIFY_MODE}\","
fi

# ---- PPO 节点划分: actor / critic 对半分（world size 必须相等）----
ACTOR_NUM_NODES=${ACTOR_NUM_NODES:-$((WORLD_SIZE / 2))}
CRITIC_NUM_NODES=${CRITIC_NUM_NODES:-$((WORLD_SIZE - ACTOR_NUM_NODES))}
if [ "${ACTOR_NUM_NODES}" != "${CRITIC_NUM_NODES}" ]; then
  echo "FATAL: PPO 要求 actor 与 critic world size 相等 (actor_critic group 按 rank 逐对建立)," \
       "当前 ACTOR_NUM_NODES=${ACTOR_NUM_NODES} CRITIC_NUM_NODES=${CRITIC_NUM_NODES}" >&2
  exit 1
fi

TASK_TAG=long_math_128k_ppo
# Public-repro dataset layout: point TRAIN_FILE / TEST_FILE at your downloaded files.
TRAIN_FILE=${TRAIN_FILE:-"${DATA_DIR}/dapo-math-17k.jsonl ${DATA_DIR}/deepmath.jsonl ${DATA_DIR}/deepscaler.jsonl"}
TEST_FILE=${TEST_FILE:-"aime2026 ${DATA_DIR}/aime-2026.jsonl aime2025 ${DATA_DIR}/aime-2025.jsonl aime2024 ${DATA_DIR}/aime-2024.jsonl"}
NUM_ROLLOUT=${NUM_ROLLOUT:-800}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-64}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-8}
ROLLOUT_MAX_PROMPT_LEN=${ROLLOUT_MAX_PROMPT_LEN:-2048}
ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-126976}
ROLLOUT_MAX_CONTEXT_LEN=${ROLLOUT_MAX_CONTEXT_LEN:-131072}
SGLANG_CONTEXT_LEN=${SGLANG_CONTEXT_LEN:-131072}
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-16000}
LOGPROBS_MAX_TOKENS_PER_GPU=${LOGPROBS_MAX_TOKENS_PER_GPU:-$((MAX_TOKENS_PER_GPU * CONTEXT_PARALLEL_SIZE))}
SGLANG_MAX_TOTAL_TOKENS=${SGLANG_MAX_TOTAL_TOKENS:-1441792}   # ~11 full-128k seqs/engine; only safe with logit_chunk
SGLANG_MAX_RUNNING_REQUESTS=${SGLANG_MAX_RUNNING_REQUESTS:-64}


export TENSORBOARD_DIR=${TENSORBOARD_DIR:-${BASE_DIR}/exp_log/tensorboard/${MODEL_TAG}_${TASK_TAG}}

GLOBAL_BATCH_SIZE=$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))
# Megatron 约束: GBS 必须被 actor DP (= actor_gpus / TP / CP) 整除。
# 96 卡 (actor 6 节点, CP4 -> DP12) 时用 ROLLOUT_BATCH_SIZE=60 (GBS=480)。
ACTOR_DP=$((ACTOR_NUM_NODES * GPUS_PER_NODE / TENSOR_MODEL_PARALLEL_SIZE / CONTEXT_PARALLEL_SIZE))
if [ $((GLOBAL_BATCH_SIZE % ACTOR_DP)) -ne 0 ]; then
  echo "FATAL: GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} 不能被 actor DP=${ACTOR_DP} 整除" \
       "(actor_nodes=${ACTOR_NUM_NODES} x ${GPUS_PER_NODE}gpu / TP${TENSOR_MODEL_PARALLEL_SIZE} / CP${CONTEXT_PARALLEL_SIZE})。" \
       "请调整 ROLLOUT_BATCH_SIZE 或 N_SAMPLES_PER_PROMPT。" >&2
  exit 1
fi
LOG_PROBS_CHUNK_SIZE=${LOG_PROBS_CHUNK_SIZE:-2048}
SGLANG_MEM_FRACTION=${SGLANG_MEM_FRACTION:-0.88}   # only safe with logit_chunk (flattens logprob fp32 peak)
SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK=${SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK:-1}
SGLANG_LOGITS_PROCESSER_CHUNK_SIZE=${SGLANG_LOGITS_PROCESSER_CHUNK_SIZE:-2048}
SGLANG_MAX_PREFILL_TOKENS=${SGLANG_MAX_PREFILL_TOKENS:-${SGLANG_CONTEXT_LEN}}
SGLANG_SCHEDULE_CONSERVATIVENESS=${SGLANG_SCHEDULE_CONSERVATIVENESS:-1.2}
EVAL_MAX_RESPONSE_LEN=${EVAL_MAX_RESPONSE_LEN:-${ROLLOUT_MAX_RESPONSE_LEN}}
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
# EXP_TAG: 可选实验分组标签 (如 ppo_v0 / ppo_smoke)
EXP_NAME="${MODEL_TAG}_${TASK_TAG}${EXP_TAG:+_${EXP_TAG}}${JOB_ID:+_job${JOB_ID}}_${RUN_TAG}"

# 断点续训（actor 侧与 GRPO 基准一致）:
#   首训: SAVE_DIR 无 ckpt -> miles 自动 finetune, load 回退到 --ref-load(底模)
#   续训: SAVE_DIR 已有 iter_* -> 自动加载 model+optim+rng 并从 loaded+1 继续
SAVE_ROOT=${SAVE_ROOT:-$WORK_DIR/runs/ppo_ablation}
SAVE_DIR=${SAVE_DIR:-${SAVE_ROOT}/${EXP_TAG:-baseline}}
ABLATION_TAG="$(basename "$(dirname "${SAVE_ROOT}")")-$(basename "${SAVE_ROOT}")"
CKPT_ARGS=(
  --hf-checkpoint $MODEL_PATH
  --ref-load $MEGATRON_MODEL_PATH
  --load $SAVE_DIR
  --save $SAVE_DIR
  --save-interval ${SAVE_INTERVAL:-10}
  --save-retain-interval ${SAVE_RETAIN_INTERVAL:-1000}
)

# 长期归档: 每 HF_SAVE_INTERVAL 步额外导出一份 HF 格式权重 (critic 不导出, 评测只需 actor) (bf16 safetensors,
# 2.6B 约 5GB/份, 不含 optimizer state) 到 ${SAVE_DIR}/hf/iter_XXXXXXX; 不参与 retain 轮转,
# 长期保留, 可直接 from_pretrained / 起 sglang 跑 eval 刷榜。需为 SAVE_INTERVAL 的整数倍
# (导出只在 ckpt save 时刻发生, 否则启动即 assert); 最后一个 rollout 必导出。置 0 关闭。
# 注: retain 默认 1000 且 miles iter 编号 0-based (iter_0000009/19/...), 里程碑永不命中,
# torch_dist 实际只留最新一份 —— 长期保留由本 HF 归档负责。
if [ "${HF_SAVE_INTERVAL:-50}" != "0" ]; then
  CKPT_ARGS+=(
    --save-hf "${SAVE_DIR}/hf/iter_{rollout_id:07d}"
    --save-hf-interval ${HF_SAVE_INTERVAL:-50}
  )
fi

# ---- critic ckpt 逻辑（load 语义与 actor 不对称，见 miles/utils/arguments.py:2195-2286）----
# --critic-save 没有默认回退，必须显式设；critic 与 actor 分目录。
# --critic-load:
#   首训: 不传 -> 继承 finetune fallback 后的 args.load = --ref-load(底模 backbone)，
#          底模 LM 头会按重叠区灌进 [1,H] value head，由 checkpoint.py 的
#          _rezero_critic_value_head 在加载后归零并重同步 fp32 主权重
#   续训: critic 目录已有 ckpt 时必须显式传，否则会默认继承 actor 的 SAVE_DIR（错误 ckpt）
CRITIC_SAVE_DIR=${CRITIC_SAVE_DIR:-${SAVE_DIR%/}_critic}
CKPT_ARGS+=(--critic-save ${CRITIC_SAVE_DIR})
ACTOR_HAS_CKPT=$([ -f "${SAVE_DIR}/latest_checkpointed_iteration.txt" ] && echo 1 || echo 0)
CRITIC_HAS_CKPT=$([ -f "${CRITIC_SAVE_DIR}/latest_checkpointed_iteration.txt" ] && echo 1 || echo 0)
if [ "${CRITIC_HAS_CKPT}" = "1" ] && [ "${ACTOR_HAS_CKPT}" = "1" ]; then
  CKPT_ARGS+=(--critic-load ${CRITIC_SAVE_DIR})
elif [ "${CRITIC_HAS_CKPT}" = "1" ]; then
  echo "FATAL: critic 已有 ckpt (${CRITIC_SAVE_DIR}) 而 actor 没有 (${SAVE_DIR})，" \
       "续训状态不一致（此时全局 finetune=True，会丢掉 critic 的 optimizer 状态，" \
       "且 _rezero_critic_value_head 会把训练过的 value head 归零）。" \
       "请人工处理：补回 actor ckpt，或清掉 critic 目录重训。" >&2
  exit 1
elif [ -n "${CRITIC_WARM_LOAD:-}" ] && [ "${ACTOR_HAS_CKPT}" = "0" ]; then
  # 首训热启 critic: 从外部已 warmup 的 critic ckpt 只读加载 weights+value head
  # (finetune 语义, 不带 optimizer), 可配 NUM_CRITIC_ONLY_STEPS 缩短/跳过 warmup。
  # --critic-finetune-load-value-head 同时豁免 _rezero_critic_value_head（warm head 保留）
  CKPT_ARGS+=(--critic-load ${CRITIC_WARM_LOAD} --critic-finetune-load-value-head)
elif [ "${ACTOR_HAS_CKPT}" = "1" ]; then
  echo "FATAL: actor 已有 ckpt (${SAVE_DIR}) 而 critic 没有 (${CRITIC_SAVE_DIR})，" \
       "续训状态不一致（此时全局 finetune=False，critic 指底模会尝试加载 optimizer 而崩）。" \
       "请人工处理：删掉 actor ckpt 重训，或补一份 critic ckpt。" >&2
  exit 1
fi

# 续训延长总步数时 (NUM_ROLLOUT 与 checkpoint 里训练时的值不同)，Megatron 加载
# opt_param_scheduler 会因 total iters 不一致直接 assert (实测 288000 vs 96000)。
# 置 1 用命令行值重建 scheduler (lr-decay-style=constant，覆盖无实际副作用)。仅首个恢复 run 需要。
if [ "${OVERRIDE_OPT_PARAM_SCHEDULER:-0}" = "1" ]; then
  CKPT_ARGS+=(--override-opt_param-scheduler)
fi

ROLLOUT_ARGS=(
  --prompt-data $TRAIN_FILE
  --input-key ${INPUT_KEY:-prompt}
  --label-key ${LABEL_KEY:-label}
  --apply-chat-template
  --rollout-shuffle
  --num-rollout ${NUM_ROLLOUT}
  --rollout-batch-size ${ROLLOUT_BATCH_SIZE}
  --n-samples-per-prompt ${N_SAMPLES_PER_PROMPT}
  --rollout-max-prompt-len ${ROLLOUT_MAX_PROMPT_LEN}
  --rollout-max-response-len ${ROLLOUT_MAX_RESPONSE_LEN}
  --rollout-max-context-len ${ROLLOUT_MAX_CONTEXT_LEN}
  --rollout-temperature ${ROLLOUT_TEMPERATURE:-1.0}
  --rollout-top-p ${ROLLOUT_TOP_P:-1.0}
  --global-batch-size ${GLOBAL_BATCH_SIZE}
  --balance-data
)

# PPO 默认不做 dynamic sampling 过滤（零方差组在 critic baseline 下仍有梯度），
# 保留普通采样 + math RM。DYNAMIC_SAMPLING=1 开 DAPO 式过滤：
# 丢弃组内 reward 零方差组（全对/全错），需 over-sampling 补齐被过滤的组。
ROLLOUT_ARGS+=(
  --rm-type math
)
if [ "${DYNAMIC_SAMPLING:-0}" = "1" ]; then
  ROLLOUT_ARGS+=(--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std)
  # partial rollout 关闭时 over-sampling 不会在下面的 partial 分支里加，这里补上
  if [ "${ENABLE_PARTIAL_ROLLOUT:-0}" != "1" ]; then
    ROLLOUT_ARGS+=(--over-sampling-batch-size ${OVER_SAMPLING_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE * 2))})
  fi
fi
CUSTOM_ARGS=(
  # IcePop/TIS 可视化 dump（与 GRPO 基准一致）
  --icepop-dump-dir ${SAVE_DIR}/${EXP_NAME}
  --dump-details ${SAVE_DIR}/${EXP_NAME}/dump_details
)

PERF_ARGS=(
  --train-backend megatron
  --tensor-model-parallel-size ${TENSOR_MODEL_PARALLEL_SIZE}
  --context-parallel-size ${CONTEXT_PARALLEL_SIZE}
  --sequence-parallel
  --pipeline-model-parallel-size 1
  --expert-model-parallel-size ${EXPERT_MODEL_PARALLEL_SIZE}
  --expert-tensor-parallel-size 1
  --attention-backend ${ATTENTION_BACKEND:-flash}
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU}
  --log-probs-max-tokens-per-gpu ${LOGPROBS_MAX_TOKENS_PER_GPU}
  --log-probs-chunk-size ${LOG_PROBS_CHUNK_SIZE}
)

PPO_ARGS=(
  --advantage-estimator ppo
  --gamma ${GAMMA:-1.0}
  --lambd ${LAMBD:-1.0}
  --normalize-advantages
  --kl-loss-coef 0.00
  --kl-loss-type low_var_kl
  --entropy-coef 0.00
  --calculate-per-token-loss
  --eps-clip 0.2
  --eps-clip-high ${EPS_CLIP_HIGH:-0.2}   # PPO 首版用经典对称 clip；clip-higher 是 GRPO/DAPO 技巧
  --value-clip ${VALUE_CLIP:-0.2}
  # critic warmup: value head weight 零初始化 + bias 初始化到平均奖励 (CRITIC_VALUE_BIAS_INIT),
  # value_loss 从 ~Var(r)≈0.25 起步; fresh 默认 20 步足够收敛; 热启 (CRITIC_WARM_LOAD) 传 5
  --num-critic-only-steps ${NUM_CRITIC_ONLY_STEPS:-20}
  --critic-lr ${CRITIC_LR:-5e-6}
  --critic-lr-warmup-iters ${CRITIC_LR_WARMUP_ITERS:-10}
)

# JustRL2 长度自适应解耦 GAE: GAE_LAMBDA_K 给出首 token 拿到的终端 credit 比例 k,
# 逐样本 λ_i = k^(1/L_i) (#12 实跑的旧形式 1−1/(α·L), α=1.5 等价于 k=e^(−1/1.5)≈0.513)。
if [ -n "${GAE_LAMBDA_K:-}" ]; then
  PPO_ARGS+=(--gae-lambda-k ${GAE_LAMBDA_K})
fi
# JustRL2 critic value-head bias 初始化 (weight 零初始化, 故 step-0 的 V ≡ bias): 设为期望
# 平均奖励, 省掉 critic 头 ~25 步学 offset 的过渡期。加载底模后 checkpoint.py 会重新填回。
PPO_ARGS+=(--critic-value-bias-init ${CRITIC_VALUE_BIAS_INIT:-0.52})

PPO_ARGS+=(
  --use-tis
  --custom-tis-function-path miles.backends.training_utils.loss.icepop_function
  --tis-clip-low 0.5
  --tis-clip 5.0
)

# DAPO Soft Overlong Punishment（与 GRPO 基准保持一致；作用在 shaped reward、进 GAE 之前）
PPO_ARGS+=(
  --overlong-buffer-len ${OVERLONG_BUFFER_LEN:-25395}   # 126976 * 20%
  --overlong-penalty-factor ${OVERLONG_PENALTY_FACTOR:-1.0}
)

# 臂#10: critic 的 value-loss 回归目标剔除 overlong penalty (actor 的 advantage 不变)。默认关。
if [ "${CRITIC_EXCLUDE_OLP:-0}" = "1" ]; then
  PPO_ARGS+=(--critic-exclude-overlong-penalty)
fi

# 臂#11: critic 的 value-loss 回归目标剔除组内相对长度奖励 lenrw (actor 的 advantage 不变)。默认关。
# 与 CRITIC_EXCLUDE_OLP 独立; 两个同开时 critic 回归目标即 raw task reward (不含 olp 不含 lenrw)。
if [ "${CRITIC_EXCLUDE_LENRW:-0}" = "1" ]; then
  PPO_ARGS+=(--critic-exclude-length-reward)
fi

# 臂#14: OLP 解析注入 (cleancritic 2.0) — OLP 离开学习通道 (reward/GAE/critic 全净, 等价 noolp),
# advantage 在 GAE 算完后按 λ_inj^d 解析注入, 回传半径由 OLP_INJECT_ALPHA 独立控制 (默认 0.1,
# 即只有末尾 ~10%·L 的 token 显著感到惩罚)。与 CRITIC_EXCLUDE_OLP 互斥 (2.0 是其超集, 同开会在
# 参数校验阶段报错二选一); 需 OVERLONG_BUFFER_LEN>0 (本脚本默认已开)。默认关。
if [ "${OLP_ANALYTIC_INJECT:-0}" = "1" ]; then
  PPO_ARGS+=(--olp-analytic-inject --olp-inject-alpha ${OLP_INJECT_ALPHA:-0.1})
fi

# 臂#16: 组中心化注入 — actor 的 advantage 通道做组内留一中心化: 终端标量 a_j = raw_reward_j − V_j
# (最后一个 loss-mask token 的 critic value), P_i = −(组内其余成员 a_j 之和)/(n−1), GAE 之后按
# λ_i 衰减注入 (k 复用 GAE_LAMBDA_K, 参数校验要求其非空); critic 照学未中心化
# return。a_j 用 raw_reward (纯任务奖励), 与 OLP_ANALYTIC_INJECT 可共存 (两个线性注入相加)。默认关。
if [ "${GROUP_CENTER_INJECT:-0}" = "1" ]; then
  PPO_ARGS+=(--group-center-inject)
fi

# Kimi k1.5 组内相对长度奖励 (仅正确分支, 与 GRPO 脚本同款; 作用在 shaped reward)。
# 默认 0 关闭——不设 LENGTH_REWARD_WEIGHT 时严格 no-op。
if [ "${LENGTH_REWARD_WEIGHT:-0}" != "0" ]; then
  PPO_ARGS+=(
    --length-reward-weight ${LENGTH_REWARD_WEIGHT}
    --length-reward-min-spread ${LENGTH_REWARD_MIN_SPREAD:-2000}
    --length-reward-budget-floor ${LENGTH_REWARD_BUDGET_FLOOR:-0}
  )
fi

# PPO 首版默认关 partial rollout（减少与 critic 路径叠加的变量）
if [ "${ENABLE_PARTIAL_ROLLOUT:-0}" = "1" ]; then
  PPO_ARGS+=(--partial-rollout)
  # partial rollout 需配合 over-sampling 才能截断长尾：超发 prompt，取先完成的
  # rollout_batch_size 组，未完成的轨迹回收到 buffer 下一步续写（TIS 缓解跨版本 off-policy）
  ROLLOUT_ARGS+=(--over-sampling-batch-size ${OVER_SAMPLING_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE * 2))})
fi

if [ "$OPTIMIZER" = "muon" ]; then
  OPTIMIZER_ARGS=(
    --optimizer muon
    --lr ${LR:-1e-6}
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
    --muon-matched-adamw-rms 0.2
    --muon-momentum 0.95
    --muon-ns-steps 5
    --muon-coefficient-type simple
  )
else
  OPTIMIZER_ARGS=(
    --optimizer adam
    --lr ${LR:-1e-6}
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
  )
fi

WANDB_ARGS=(--use-tensorboard)
if [ "${USE_SWANLAB:-1}" = "1" ]; then
  # 项目名优先 MILES_SWANLAB_PROJECT: SWANLAB_PROJECT 环境变量会被镜像内
  # swanlab SDK 的 pydantic EnvSettingsSource 捕获并在 import 时解析失败
  # (574061 复现 SettingsError: error parsing value for field "project")
  WANDB_ARGS+=(--use-swanlab --swanlab-project ${MILES_SWANLAB_PROJECT:-${SWANLAB_PROJECT:-minicpm5-${ABLATION_TAG}}} --swanlab-experiment-name ${EXP_NAME})
  unset SWANLAB_PROJECT
fi
if [ "${USE_WANDB}" = "1" ]; then
  WANDB_ARGS+=(--use-wandb --wandb-project ${WANDB_PROJECT:-miles-minicpm5})
  if [ -n "${WANDB_TEAM:-}" ]; then
    WANDB_ARGS+=(--wandb-team "${WANDB_TEAM}")
  fi
fi

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine ${ROLLOUT_NUM_GPUS_PER_ENGINE}
  --sglang-mem-fraction-static ${SGLANG_MEM_FRACTION}
  --sglang-max-running-requests ${SGLANG_MAX_RUNNING_REQUESTS}
  --sglang-schedule-conservativeness ${SGLANG_SCHEDULE_CONSERVATIVENESS}
  --sglang-attention-backend fa3
  # 引擎日志降到 warning: 32 引擎的 INFO 级 decode/请求日志经 log_to_driver 汇聚到 master,
  # 是 622523 临时存储超限 (~1.1MB/s 持续写) 的主力之一; 错误/重试仍是 WARNING+ 不受影响
  --sglang-log-level ${SGLANG_LOG_LEVEL:-warning}
  --sglang-json-model-override-args "{\"max_position_embeddings\": ${SGLANG_CONTEXT_LEN}}"
  --sglang-context-length ${SGLANG_CONTEXT_LEN}
  --sglang-max-prefill-tokens ${SGLANG_MAX_PREFILL_TOKENS}
  --sglang-max-total-tokens ${SGLANG_MAX_TOTAL_TOKENS}
  --sglang-enable-metrics
  --log-rollout-mbu
  "${SGLANG_EXTRA_ARGS[@]}"
)

EVAL_ARGS=(
  --eval-prompt-data $TEST_FILE
  --eval-interval ${EVAL_INTERVAL:-5}
  --eval-temperature ${EVAL_TEMPERATURE:-1.0}
  --eval-top-p ${EVAL_TOP_P:-0.95}
  --eval-max-response-len ${EVAL_MAX_RESPONSE_LEN}
  --n-samples-per-eval-prompt ${N_SAMPLES_PER_EVAL_PROMPT:-4}
)

mkdir -p ${SAVE_DIR}/${EXP_NAME}/dump_details ${SAVE_DIR}/nccl_trace
MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
)

RUNTIME_ENV_JSON="{
  \"env_vars\": {${DSPARK_ENV_JSON_EXTRA}
    \"PYTHONPATH\":  \".:./Megatron-LM:./mbridge:${SGLANG_PATH}/python:${WORK_DIR}/examples/retool_v2:/root/.cache/huggingface/modules\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"TORCH_NCCL_TRACE_BUFFER_SIZE\": \"2000\",
    \"TORCH_NCCL_DUMP_ON_TIMEOUT\": \"1\",
    \"TORCH_NCCL_DEBUG_INFO_TEMP_FILE\": \"${SAVE_DIR}/nccl_trace/rank_\",
    \"SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN\": \"1\",
    \"NVTE_DEBUG\": \"${NVTE_DEBUG:-0}\",
    \"NVTE_DEBUG_LEVEL\": \"${NVTE_DEBUG_LEVEL:-0}\",
    \"NVTE_FUSED_ATTN\": \"0\",
    \"NVTE_UNFUSED_ATTN\": \"0\",
    \"SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION\": \"10000\",
    \"SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK\": \"${SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK}\",
    \"SGLANG_LOGITS_PROCESSER_CHUNK_SIZE\": \"${SGLANG_LOGITS_PROCESSER_CHUNK_SIZE}\",
    \"TRITON_CACHE_DIR\": \"${TRITON_CACHE_DIR}\",
    \"TMPDIR\": \"${TMPDIR}\",
    \"MILES_EXPERIMENTAL_ROLLOUT_REFACTOR\": \"1\"
  }
}"

bash scripts/setup/ray_start.sh

echo "============================================================"
echo "${MODEL_TAG} ${TASK_TAG}: colocate PPO RL"
echo "rollouts=${NUM_ROLLOUT}, batch=${ROLLOUT_BATCH_SIZE}, samples=${N_SAMPLES_PER_PROMPT}"
echo "actor_nodes=${ACTOR_NUM_NODES}, critic_nodes=${CRITIC_NUM_NODES}, gpus_per_node=${GPUS_PER_NODE}"
echo "TP=${TENSOR_MODEL_PARALLEL_SIZE}, CP=${CONTEXT_PARALLEL_SIZE}, EP=${EXPERT_MODEL_PARALLEL_SIZE}, sglang_tp=${ROLLOUT_NUM_GPUS_PER_ENGINE}"
echo "prompt_len=${ROLLOUT_MAX_PROMPT_LEN}, response_len=${ROLLOUT_MAX_RESPONSE_LEN}, context_len=${ROLLOUT_MAX_CONTEXT_LEN}"
echo "critic_only_steps=${NUM_CRITIC_ONLY_STEPS:-15}, critic_lr=${CRITIC_LR:-5e-6}, critic_save=${CRITIC_SAVE_DIR}"
echo "partial_rollout=${ENABLE_PARTIAL_ROLLOUT:-0}, dynamic_sampling=${DYNAMIC_SAMPLING:-0}"
echo "EXP=${EXP_NAME}"
echo "BASE_DIR=${BASE_DIR}"
echo "============================================================"

ray job submit --address="http://${MASTER_ADDR}:8265" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 train.py \
  --actor-num-nodes ${ACTOR_NUM_NODES} \
  --actor-num-gpus-per-node ${GPUS_PER_NODE} \
  --critic-num-nodes ${CRITIC_NUM_NODES} \
  --critic-num-gpus-per-node ${GPUS_PER_NODE} \
  --colocate \
  --skip-eval-before-train \
  "${MODEL_ARGS_SELECTED[@]}" \
  "${CKPT_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${PPO_ARGS[@]}" \
  "${WANDB_ARGS[@]}" \
  "${PERF_ARGS[@]}" \
  "${SGLANG_ARGS[@]}" \
  "${EVAL_ARGS[@]}" \
  "${CUSTOM_ARGS[@]}" \
  "${MISC_ARGS[@]}" "$@"

bash $WORK_DIR/scripts/setup/ray_end.sh
echo "${MODEL_TAG} ${TASK_TAG} done! EXP=${EXP_NAME}"
