import logging
import os
from argparse import Namespace
from collections.abc import Callable
from copy import deepcopy

import torch
import torch.distributed as dist

from miles.utils import tracking_utils
from miles.utils.metric_utils import compute_rollout_step
from miles.utils.timer import Timer

logger = logging.getLogger(__name__)


def _get_peak_tflops_bf16() -> float:
    """Return per-GPU peak BF16 TensorCore TFLOPS for MFU estimation."""
    v = os.environ.get("MILES_PEAK_TFLOPS_BF16", "").strip()
    if v:
        try:
            return float(v)
        except Exception:
            return 0.0

    try:
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0) or ""
            if "H100" in name.upper():
                return 989.0
    except Exception:
        pass

    return 0.0


def log_perf_data_raw(
    rollout_id: int, args: Namespace, is_primary_rank: bool, compute_total_fwd_flops: Callable
) -> None:
    timer_instance = Timer()
    log_dict_raw = deepcopy(timer_instance.log_dict())
    timer_instance.reset()

    if not is_primary_rank:
        return

    log_dict = {f"perf/{key}_time": val for key, val in log_dict_raw.items()}

    if ("perf/actor_train_time" in log_dict) and (compute_total_fwd_flops is not None):
        total_fwd_flops = compute_total_fwd_flops(seq_lens=timer_instance.seq_lens)

        world_size = dist.get_world_size() if dist.is_initialized() else 1
        total_tokens = sum(timer_instance.seq_lens)
        train_time = log_dict["perf/actor_train_time"]
        
        actor_train_tflops = 3 * total_fwd_flops / train_time if train_time > 0 else 0
        tok_per_s = total_tokens / train_time if train_time > 0 else 0

        peak = _get_peak_tflops_bf16()
        actor_train_mfu = (actor_train_tflops / peak) if (peak and peak > 0) else None
        
        logger.info(
            f"[TFLOPS_DEBUG] rollout={rollout_id} world_size={world_size} "
            f"tokens={total_tokens} train_time={train_time:.2f}s "
            f"tflops={actor_train_tflops:.2f} tok/s={tok_per_s:.0f}"
            + (f" mfu={actor_train_mfu:.4f}" if actor_train_mfu is not None else "")
        )

        if "perf/log_probs_time" in log_dict:
            log_dict["perf/log_probs_tflops"] = total_fwd_flops / log_dict["perf/log_probs_time"]

        if "perf/ref_log_probs_time" in log_dict:
            log_dict["perf/ref_log_probs_tflops"] = total_fwd_flops / log_dict["perf/ref_log_probs_time"]

        if log_dict["perf/actor_train_time"] > 0:
            log_dict["perf/actor_train_tflops"] = 3 * total_fwd_flops / log_dict["perf/actor_train_time"]
            log_dict["perf/actor_train_tok_per_s"] = sum(timer_instance.seq_lens) / log_dict["perf/actor_train_time"]
            log_dict["perf/actor_train_tflops_8gpu"] = log_dict["perf/actor_train_tflops"] * world_size
            log_dict["perf/actor_train_tok_per_s_8gpu"] = log_dict["perf/actor_train_tok_per_s"] * world_size

            peak = _get_peak_tflops_bf16()
            if peak and peak > 0:
                log_dict["perf/actor_train_mfu"] = log_dict["perf/actor_train_tflops"] / peak
                log_dict["perf/actor_train_mfu_8gpu"] = log_dict["perf/actor_train_tflops_8gpu"] / (peak * world_size)

    if "perf/train_wait_time" in log_dict and "perf/train_time" in log_dict:
        total_time = log_dict["perf/train_wait_time"] + log_dict["perf/train_time"]
        if total_time > 0:
            log_dict["perf/step_time"] = total_time
            log_dict["perf/wait_time_ratio"] = log_dict["perf/train_wait_time"] / total_time

    logger.info(f"perf {rollout_id}: {log_dict}")

    step = compute_rollout_step(args, rollout_id)
    log_dict["rollout/step"] = step
    tracking_utils.log(args, log_dict, step_key="rollout/step")
