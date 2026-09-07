import asyncio
import logging
import time

import ray

from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH, GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS

from miles.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from miles.utils.arguments import parse_args
from miles.utils.async_utils import eager_create_task
from miles.utils.logging_utils import configure_logger
from miles.utils.misc import should_run_periodic_action
from miles.utils.tracking_utils import finish_tracking, init_tracking

logger = logging.getLogger(__name__)


def _vram_phase(rollout_id: int, phase: str):
    logger.info(f"===== VRAM-Phase rollout={rollout_id} phase={phase} t={time.time():.1f} =====")


async def train(args):
    configure_logger()
    # allocate the GPUs
    pgs = create_placement_groups(args)
    init_tracking(args)

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    # create the actor and critic models
    actor_model, critic_model = await create_training_models(args, pgs, rollout_manager)

    if args.offload_rollout:
        await rollout_manager.onload_weights.remote()

    # always update weight first so that sglang has the loaded weights from training.
    await actor_model.update_weights()

    if args.check_weight_update_equal:
        await rollout_manager.check_weights.remote(action="compare")

    if args.offload_rollout:
        await rollout_manager.onload_kv.remote()

    # special case for eval-only
    if args.num_rollout == 0 and args.eval_interval is not None:
        await rollout_manager.eval.remote(rollout_id=0)

    async def offload_train():
        if args.offload_train:
            if args.use_critic:
                await critic_model.offload()
                if rollout_id >= args.num_critic_only_steps:
                    await actor_model.offload()
            else:
                await actor_model.offload()
        else:
            await actor_model.clear_memory()

    async def save(rollout_id):
        # 即使在 critic-only warmup 期间也保存 actor（权重虽未更新，但 resume 的
        # start_rollout_id 由 actor ckpt 推断；不存 actor 会导致重启后回到 rollout 0，
        # 而 critic 却带着旧状态续训，产生错位）。
        await actor_model.save_model(
            rollout_id,
            force_sync=rollout_id == args.num_rollout - 1,
        )
        if args.use_critic:
            await critic_model.save_model(
                rollout_id,
                force_sync=rollout_id == args.num_rollout - 1,
            )
        if args.rollout_global_dataset:
            await rollout_manager.save.remote(rollout_id)

    consecutive_high_no_grad = 0

    # train loop.
    # note that for async training, one can change the position of the sync operation(ray.get).
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        if args.eval_interval is not None and rollout_id == 0 and not args.skip_eval_before_train:
            await rollout_manager.eval.remote(rollout_id)

        _vram_phase(rollout_id, "generate_start")
        rollout_data_ref = await rollout_manager.generate.remote(rollout_id)
        rollout_metrics = await rollout_manager.get_latest_rollout_metrics.remote()
        _vram_phase(rollout_id, "generate_end")

        no_grad = rollout_metrics.get("grpo_metrics/no_grad")
        if args.stop_on_no_grad_threshold is not None:
            if no_grad is None:
                consecutive_high_no_grad = 0
            elif no_grad > args.stop_on_no_grad_threshold:
                consecutive_high_no_grad += 1
                logger.warning(
                    "rollout %s grpo_metrics/no_grad=%.4f exceeded threshold %.4f (%s/%s)",
                    rollout_id,
                    no_grad,
                    args.stop_on_no_grad_threshold,
                    consecutive_high_no_grad,
                    args.stop_on_no_grad_patience,
                )
                if consecutive_high_no_grad >= args.stop_on_no_grad_patience:
                    logger.warning(
                        "Stopping early before actor update because no_grad stayed above threshold for %s rollouts",
                        consecutive_high_no_grad,
                    )
                    break
            else:
                consecutive_high_no_grad = 0

        if args.offload_rollout:
            _vram_phase(rollout_id, "offload_rollout_start")
            offload_tags = [GPU_MEMORY_TYPE_CUDA_GRAPH]
            if "kv_cache" in args.offload_rollout_level:
                offload_tags.append(GPU_MEMORY_TYPE_KV_CACHE)
            if "weight" in args.offload_rollout_level:
                offload_tags.append(GPU_MEMORY_TYPE_WEIGHTS)
            await rollout_manager.offload.remote(tags=offload_tags)
            _vram_phase(rollout_id, "offload_rollout_end")

        _vram_phase(rollout_id, "train_start")
        if args.use_critic:
            critic_task = await eager_create_task(critic_model.train(rollout_id, rollout_data_ref))
            if rollout_id >= args.num_critic_only_steps:
                await actor_model.train(rollout_id, rollout_data_ref)
            await critic_task
        else:
            await actor_model.train(rollout_id, rollout_data_ref)
        _vram_phase(rollout_id, "train_end")

        if should_run_periodic_action(rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout):
            await save(rollout_id)

        _vram_phase(rollout_id, "offload_train_start")
        await offload_train()
        _vram_phase(rollout_id, "offload_train_end")

        if args.offload_rollout:
            _vram_phase(rollout_id, "onload_weights_start")
            await rollout_manager.onload_weights.remote()
            _vram_phase(rollout_id, "onload_weights_end")

        _vram_phase(rollout_id, "update_weights_start")
        await actor_model.update_weights()
        _vram_phase(rollout_id, "update_weights_end")

        if args.offload_rollout:
            _vram_phase(rollout_id, "onload_kv_start")
            await rollout_manager.onload_kv.remote()
            _vram_phase(rollout_id, "onload_kv_end")

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            await rollout_manager.eval.remote(rollout_id)

    await rollout_manager.dispose.remote()


if __name__ == "__main__":
    args = parse_args()
    try:
        asyncio.run(train(args))
    finally:
        finish_tracking()
