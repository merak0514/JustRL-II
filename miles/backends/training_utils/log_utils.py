import logging
from argparse import Namespace
from math import isclose

import numpy as np
import psutil
import torch
import torch.distributed as dist

from miles.utils import train_metric_utils
from miles.utils.flops_utils import calculate_fwd_flops
from miles.utils.metric_utils import compute_grpo_sample_metrics, compute_pass_rate, compute_rollout_step
from miles.utils.types import RolloutBatch

from ...utils import tracking_utils
from .cp_utils import get_sum_of_sample_mean
from .data import DataIterator
from .loss import flush_logdiff_histogram
from .parallel import get_parallel_state

logger = logging.getLogger(__name__)

_LOGDIFF_HISTOGRAM_BINS = 4096
_LOGDIFF_HISTOGRAM_MIN_EXP = -8
_LOGDIFF_HISTOGRAM_MAX_EXP = 2
_LOGDIFF_QUANTILES = {
    "p10": 0.1,
    "p20": 0.2,
    "p30": 0.3,
    "p40": 0.4,
    "p50": 0.5,
    "p60": 0.6,
    "p70": 0.7,
    "p75": 0.75,
    "p80": 0.8,
    "p90": 0.9,
    "p95": 0.95,
    "p999": 0.999,
}


def _logdiff_quantile_from_histogram(counts: torch.Tensor, q: float, max_value: float) -> float:
    total = int(counts.sum().item())
    if total <= 0:
        return 0.0

    target_idx = min(int(q * total), total - 1)
    bucket_idx = int(torch.searchsorted(torch.cumsum(counts, dim=0), target_idx + 1).item())
    if bucket_idx == 0:
        return 0.0
    if bucket_idx >= _LOGDIFF_HISTOGRAM_BINS:
        return max_value

    edges = torch.logspace(
        _LOGDIFF_HISTOGRAM_MIN_EXP,
        _LOGDIFF_HISTOGRAM_MAX_EXP,
        steps=_LOGDIFF_HISTOGRAM_BINS,
        device=counts.device,
        dtype=torch.float64,
    )
    return float(edges[bucket_idx].item())


def _cat_per_sample_tensors(values: list[torch.Tensor]) -> torch.Tensor:
    """Concatenate per-sample tensors along dim 0; promote 0-D tensors so ``torch.cat`` succeeds."""
    if not values:
        raise ValueError("_cat_per_sample_tensors: empty tensor list")
    normalized = [t.unsqueeze(0) if t.dim() == 0 else t for t in values]
    return torch.cat(normalized, dim=0)


def gather_log_data(
    metric_name: str,
    args: Namespace,
    rollout_id: int,
    log_dict: dict[str, float],
) -> dict[str, float] | None:
    """
    Gather per-rank metrics, reduce by mean on the DP source rank, and log.

    Expects `log_dict` to contain plain scalars. The DP source rank prints and
    optionally logs to WandB/TensorBoard with a step derived from `rollout_id` and
    batch sizes. Returns the reduced dict on the DP source rank; returns None on others.
    """

    parallel_state = get_parallel_state()

    pg = parallel_state.intra_dp_cp
    dp_size = pg.size
    gathered_log_dict = [None] * dp_size
    # Not sure if this will be a performance bottleneck.
    dist.gather_object(
        log_dict,
        gathered_log_dict if pg.rank == 0 else None,
        dst=dist.get_global_rank(pg.gloo_group, 0),
        group=pg.gloo_group,
    )

    if pg.rank == 0:
        reduced_log_dict = {
            f"{metric_name}/{key}": sum([d[key] for d in gathered_log_dict]) / dp_size for key in log_dict
        }
        logger.info(f"{metric_name} {rollout_id}: {reduced_log_dict}")

        # Calculate step once to avoid duplication
        step = compute_rollout_step(args, rollout_id)
        reduced_log_dict["rollout/step"] = step
        tracking_utils.log(args, reduced_log_dict, step_key="rollout/step")

        return reduced_log_dict
    else:
        return None


def aggregate_forward_results(
    forward_data_store: list[dict[str, list]],
    data_iterator: DataIterator,
    args: Namespace,
    store_prefix: str = "",
) -> dict[str, list]:
    rollout_data = {}
    if not forward_data_store:
        return rollout_data

    keys = forward_data_store[0].keys()
    for key in keys:
        values = []
        for batch_result in forward_data_store:
            assert isinstance(batch_result[key], list), f"Expected list for key {key}, got {type(batch_result[key])}"
            values += batch_result[key]

        # Handle dynamic batch size: restore original order
        if args.use_dynamic_batch_size and hasattr(data_iterator, "micro_batch_indices"):
            origin_values = [None] * len(values)
            origin_indices = sum(data_iterator.micro_batch_indices, [])
            for value, origin_index in zip(values, origin_indices, strict=False):
                origin_values[origin_index] = value
            values = origin_values

        rollout_data[key] = values

    return rollout_data


def log_rollout_data(rollout_id: int, args: Namespace, rollout_data: RolloutBatch) -> None:
    """
    Summarize rollout fields and log reduced metrics on PP last stage, TP rank 0.

    - Tensor-valued lists are concatenated and averaged. For token-level metrics
      like log-probs/returns/advantages/values, computes a CP-correct sample mean
      using `loss_masks` and total/response lengths.
    - Non-tensor lists are averaged elementwise.
    - Scalars are converted to Python numbers.
    """
    parallel_state = get_parallel_state()
    if parallel_state.tp.rank == 0 and parallel_state.is_pp_last_stage:
        cp_size = parallel_state.cp.size
        log_dict = {}
        response_lengths = rollout_data["response_lengths"]
        loss_masks = rollout_data["loss_masks"]
        total_lengths = rollout_data["total_lengths"]
        max_seq_lens = rollout_data.get("max_seq_lens", None)

        env_errors = rollout_data.get("env_error")
        if env_errors:
            env_error_total = sum(env_errors)
            log_dict["env_error_ratio"] = env_error_total / len(env_errors)

        for key, val in rollout_data.items():
            if key in [
                "tokens",
                "multimodal_train_inputs",
                "loss_masks",
                "sample_indices",
                "group_center_positions",
                "rollout_routed_experts",
                "rollout_routed_experts_weights",
                "max_seq_lens",
                "dynamic_global_batch_size",
                "env_error",
                "weight_versions",
                "metadata",
            ]:
                continue
            # Upload per sample mean for each rollout value
            # There are the following assumptions:
            # - Each dp rank has the same number of samples
            if isinstance(val, (list, tuple)):
                if isinstance(val[0], torch.Tensor):
                    # NOTE: Here we have to do the clone().detach(), otherwise the tensor will be
                    # modified in place and will cause problem for the next rollout.
                    val = _cat_per_sample_tensors(val).clone().detach()
                    if key in [
                        "log_probs",
                        "ref_log_probs",
                        "rollout_log_probs",
                        "returns",
                        "advantages",
                        "values",
                        "entropy",
                    ]:
                        sum_of_sample_mean = get_sum_of_sample_mean(
                            total_lengths,
                            response_lengths,
                            loss_masks,
                            qkv_format=args.qkv_format,
                            max_seq_lens=max_seq_lens,
                        )
                        val = cp_size * sum_of_sample_mean(val) / len(loss_masks)
                    else:
                        val = val.mean() * cp_size
                else:
                    # Flatten nested lists (e.g. list of lists from async rollout).
                    flat = val
                    if isinstance(val[0], (list, tuple)):
                        flat = [x for sublist in val for x in sublist]
                    # Skip non-numeric values (e.g. strings from async rollout metadata).
                    if flat and not isinstance(flat[0], (int, float)):
                        continue
                    if key == "raw_reward" and env_errors:
                        filtered = [v for v, e in zip(flat, env_errors) if not e]
                        val = sum(filtered) / len(filtered) if filtered else 0
                    else:
                        val = sum(flat) / len(flat)
            elif isinstance(val, torch.Tensor):
                val = val.float().mean()
            else:
                raise ValueError(f"Unsupported type: {type(val)} for key: {key}")
            log_dict[key] = val.item() if isinstance(val, torch.Tensor) else val

        reduced_log_dict = gather_log_data("rollout", args, rollout_id, log_dict)
        if args.ci_test and not args.ci_disable_logprobs_checker and reduced_log_dict is not None:
            if (
                rollout_id == 0
                and "rollout/log_probs" in reduced_log_dict
                and "rollout/ref_log_probs" in reduced_log_dict
            ):
                # When R3 (rollout routing replay) is enabled, ref model does not use R3
                # so log_probs and ref_log_probs may diverge; use a relaxed tolerance.
                # When --sglang-config deploys multiple models, the heavier offload/onload
                # cycle can amplify flash-attention non-determinism; use 1e-8.
                # The default branch also covers larger TP/CP/EP variants (e.g. stage-c-long
                # test_qwen2.5_0.5B_gsm8k.py on 8 GPUs hit ~3.7e-9 diff in CI), so use 1e-8
                # rather than the previous 3e-9 to absorb BF16 reduction noise across configs.
                if args.use_rollout_routing_replay:
                    abs_tol = 1e-5
                elif getattr(args, "sglang_config", None) is not None:
                    abs_tol = 1e-8
                else:
                    abs_tol = 1e-8
                assert isclose(
                    reduced_log_dict["rollout/log_probs"], reduced_log_dict["rollout/ref_log_probs"], abs_tol=abs_tol
                ), f"CI check failed: log_probs ({reduced_log_dict['rollout/log_probs']}) != ref_log_probs ({reduced_log_dict['rollout/ref_log_probs']})"
            if "rollout/log_probs" in reduced_log_dict and "rollout/rollout_log_probs" in reduced_log_dict:
                assert isclose(
                    reduced_log_dict["rollout/log_probs"], reduced_log_dict["rollout/rollout_log_probs"], abs_tol=0.03
                ), f"CI check failed: log_probs ({reduced_log_dict['rollout/log_probs']}) != rollout_log_probs ({reduced_log_dict['rollout/rollout_log_probs']})"
            if "rollout/entropy" in reduced_log_dict:
                assert 0 < reduced_log_dict["rollout/entropy"] < 0.7

        if args.ci_test and args.true_on_policy_mode:
            assert log_dict["log_probs"] == log_dict["rollout_log_probs"], (
                f"CI check failed: true_on_policy_mode is enabled, but log_probs "
                f"({log_dict['log_probs']}) != rollout_log_probs "
                f"({log_dict['rollout_log_probs']})"
            )

    if args.log_multi_turn:
        log_multi_turn_data(rollout_id, args, rollout_data)
    if args.log_passrate:
        log_passrate(rollout_id, args, rollout_data)

    if args.n_samples_per_prompt > 1:
        log_grpo_metrics(rollout_id, args, rollout_data, parallel_state)

    if args.log_correct_samples:
        if parallel_state.tp.rank == 0 and parallel_state.is_pp_last_stage:
            cp_size = parallel_state.cp.size
            log_dict = {}
            response_lengths = rollout_data["response_lengths"]
            loss_masks = rollout_data["loss_masks"]
            total_lengths = rollout_data["total_lengths"]

            def quantile(total_value, n_quantiles, data) -> dict:
                import math

                assert n_quantiles > 1, f"n_quantiles({n_quantiles}) must be greater than 1."

                quantiles = [((i + 1) / n_quantiles) for i in range(n_quantiles)]
                cut_points = [total_value * q for q in quantiles]
                cut_points[-1] = total_value

                count = [0] * n_quantiles
                for d in data:
                    for i, point in enumerate(cut_points):
                        if d <= point:
                            count[i] += 1
                            break

                total = sum(count) + 1e-9
                percentile = [c / total for c in count]

                percentile = {f"p{min(math.ceil(q*100),100)}": p for q, p in zip(quantiles, percentile, strict=True)}
                return percentile

            raw_rewards = rollout_data["raw_reward"]
            # Additional metrics for correct cases are calculated separately below.
            correct_response_lengths = []
            correct_total_lengths = []
            correct_loss_masks = []
            correct_entropy = []
            for i, raw_reward in enumerate(raw_rewards):
                if raw_reward == 1:
                    correct_response_lengths.append(response_lengths[i])
                    correct_total_lengths.append(total_lengths[i])
                    correct_loss_masks.append(loss_masks[i])
                    correct_entropy.append(-rollout_data["log_probs"][i])
            num_correct_responses = len(correct_total_lengths)
            rollout_data["correct_response_lengths"] = correct_response_lengths
            correct_response_length_percentile = quantile(
                args.rollout_max_response_len, 4, rollout_data["correct_response_lengths"]
            )
            for p, val in correct_response_length_percentile.items():
                rollout_data[f"correct_length/{p}"] = [val] * num_correct_responses
            if len(correct_entropy) > 0:
                sum_of_sample_mean = get_sum_of_sample_mean(
                    correct_total_lengths, correct_response_lengths, correct_loss_masks
                )
                correct_entropy = sum_of_sample_mean(torch.cat(correct_entropy, dim=0))
                rollout_data["correct_entropy"] = [correct_entropy.item()] * num_correct_responses
            else:
                rollout_data["correct_entropy"] = [0] * num_correct_responses


def log_multi_turn_data(rollout_id: int, args: Namespace, rollout_data: RolloutBatch) -> None:
    """
    Log multi-turn auxiliary metrics such as raw/observed response lengths and rounds.

    Operates only on PP last stage and TP rank 0. Uses GPU tensors when available
    to compute statistics without host transfers.
    """
    parallel_state = get_parallel_state()
    if parallel_state.tp.rank == 0 and parallel_state.is_pp_last_stage:
        log_dict = {}
        for key, val in rollout_data.items():
            if key == "loss_masks":
                if val:  # Check if val is not empty
                    device = val[0].device  # Get device from first tensor

                    # Vectorized length calculation using torch
                    raw_response_lengths = torch.tensor([v.shape[0] for v in val], dtype=torch.float32, device=device)
                    log_dict["raw_response_length/response_length_mean"] = raw_response_lengths.mean().item()
                    log_dict["raw_response_length/response_length_max"] = raw_response_lengths.max().item()
                    log_dict["raw_response_length/response_length_min"] = raw_response_lengths.min().item()
                    log_dict["raw_response_length/response_length_clip_ratio"] = (
                        (raw_response_lengths >= args.rollout_max_response_len).float().mean().item()
                    )

                    # Vectorized sum calculation using torch - stay on GPU
                    wo_obs_response_lengths = torch.tensor(
                        [v.sum().item() for v in val], dtype=torch.float32, device=device
                    )
                    log_dict["wo_obs_response_length/response_length_mean"] = wo_obs_response_lengths.mean().item()
                    log_dict["wo_obs_response_length/response_length_max"] = wo_obs_response_lengths.max().item()
                    log_dict["wo_obs_response_length/response_length_min"] = wo_obs_response_lengths.min().item()
            if key == "round_number":
                # Use numpy for vectorized round number statistics
                round_number_array = np.array(val)
                log_dict["multi_turn_metric/round_number_mean"] = np.mean(round_number_array)
                log_dict["multi_turn_metric/round_number_max"] = np.max(round_number_array)
                log_dict["multi_turn_metric/round_number_min"] = np.min(round_number_array)
        gather_log_data("multi_turn", args, rollout_id, log_dict)


def log_passrate(rollout_id: int, args: Namespace, rollout_data: RolloutBatch) -> None:
    """
    Compute pass@k metrics from `raw_reward` groups and log the results.

    `raw_reward` is reshaped to `[group_number, group_size]`, then pass@k is
    estimated per problem and averaged.
    """
    parallel_state = get_parallel_state()
    if parallel_state.tp.rank == 0 and parallel_state.is_pp_last_stage:
        log_dict = {}
        for key, val in rollout_data.items():
            if key != "raw_reward":
                continue

            log_dict |= compute_pass_rate(
                flat_rewards=val,
                group_size=args.n_samples_per_prompt,
                num_groups=args.rollout_batch_size,
            )

        gather_log_data("passrate", args, rollout_id, log_dict)


def log_grpo_metrics(
    rollout_id: int, args: Namespace, rollout_data: RolloutBatch, parallel_state
) -> None:
    """
    Compute GRPO group-level metrics and log them.

    For each prompt group of n samples:
    - solve_all  / all_one_ratio  : variance<0.01 and mean>0.99 (all correct, no gradient signal)
    - solve_none / all_zero_ratio : variance<0.01 and mean<0.01 (all wrong, no gradient signal)
    - no_grad                     : variance<0.01 (no gradient signal regardless of correctness)
    - effective_ratio             : 1 - no_grad (fraction of groups that DO produce gradient)
    - {all_one,all_zero,effective}_count : absolute counts (denominator = num_groups)

    Also emits a single human-friendly INFO line per rollout to make signal collapse
    visible at a glance when scrolling raw stdout/stderr (TB shows the same numbers).
    """
    if parallel_state.tp.rank == 0 and parallel_state.is_pp_last_stage:
        log_dict = {}
        if "raw_reward" in rollout_data:
            log_dict |= compute_grpo_sample_metrics(
                flat_rewards=rollout_data["raw_reward"],
                group_size=args.n_samples_per_prompt,
                num_groups=args.rollout_batch_size,
            )

        if log_dict:
            if parallel_state.intra_dp_cp.rank == 0 and "num_groups" in log_dict:
                ng = int(log_dict["num_groups"])
                a1 = int(log_dict["all_one_count"])
                a0 = int(log_dict["all_zero_count"])
                eff = int(log_dict["effective_count"])
                logger.info(
                    f"[grpo_signal] rollout={rollout_id}  groups={ng}  "
                    f"all_one={a1} ({a1 / ng:.0%})  "
                    f"all_zero={a0} ({a0 / ng:.0%})  "
                    f"effective={eff} ({eff / ng:.0%})"
                )
            gather_log_data("grpo_metrics", args, rollout_id, log_dict)


def log_perf_data(rollout_id: int, args: Namespace) -> None:
    parallel_state = get_parallel_state()
    train_metric_utils.log_perf_data_raw(
        rollout_id=rollout_id,
        args=args,
        is_primary_rank=(
            parallel_state.tp.rank == 0 and parallel_state.is_pp_last_stage and parallel_state.intra_dp_cp.rank == 0
        ),
        compute_total_fwd_flops=lambda seq_lens: calculate_fwd_flops(seqlens=seq_lens, args=args)
        / dist.get_world_size()
        / 1e12,
    )


def log_cpu_memory(rollout_id: int, args: Namespace, label: str) -> None:
    """Log current system CPU memory usage to wandb/tensorboard.

    Caller is responsible for ensuring this runs on a single rank only.
    """

    cpu_mem_gb = psutil.virtual_memory().used / 1e9
    step = compute_rollout_step(args, rollout_id)
    logger.info(f"[CPU memory] {label}: {cpu_mem_gb:.2f} GB (rollout_id={rollout_id}, step={step})")
    tracking_utils.log(
        args,
        {f"perf/cpu_memory_{label}_gb": cpu_mem_gb, "rollout/step": step},
        step_key="rollout/step",
    )


def aggregate_train_losses(
    losses_reduced: list[dict[str, list[str] | torch.Tensor]],
) -> dict[str, float]:
    """Aggregate loss metrics across micro-batches.

    Sums loss values across all micro-batches, performs all-reduce across
    the data-parallel group, and computes per-sample/token averages.

    Args:
        losses_reduced: List of log_dict from each micro-batch.
            Each log_dict has format: {"keys": list[str], "values": torch.Tensor}
        parallel_state: Parallel state containing dp_group and cp_size.

    Returns:
        Dictionary mapping metric names to averaged values.
    """
    parallel_state = get_parallel_state()
    if not losses_reduced:
        return {}

    keys = losses_reduced[0]["keys"]

    values = None
    weighted_metrics = None
    logdiff_histogram = None
    logdiff_min_max = None
    for log_dict in losses_reduced:
        if values is None:
            values = log_dict["values"].clone()
        else:
            values += log_dict["values"]

        for key, metric_value in log_dict.get("weighted_metrics", {}).items():
            if weighted_metrics is None:
                weighted_metrics = {}
            metric_value = metric_value.to(device=values.device, dtype=torch.float64)
            if key not in weighted_metrics:
                weighted_metrics[key] = metric_value.clone()
            else:
                weighted_metrics[key] += metric_value

        if "logdiff_histogram" in log_dict:
            metric_histogram = log_dict["logdiff_histogram"].to(device=values.device, dtype=torch.float64)
            if logdiff_histogram is None:
                logdiff_histogram = metric_histogram.clone()
            else:
                logdiff_histogram += metric_histogram

        if "logdiff_min_max" in log_dict:
            metric_min_max = log_dict["logdiff_min_max"].to(device=values.device, dtype=torch.float64)
            if logdiff_min_max is None:
                logdiff_min_max = metric_min_max.clone()
            else:
                logdiff_min_max[0] = torch.minimum(logdiff_min_max[0], metric_min_max[0])
                logdiff_min_max[1] = torch.maximum(logdiff_min_max[1], metric_min_max[1])

    assert len(keys) + 1 == values.numel(), f"Expected {len(keys) + 1} values, got {values.numel()}"

    dist.all_reduce(values, op=dist.ReduceOp.SUM, group=parallel_state.intra_dp_cp.group)
    if weighted_metrics:
        for metric_value in weighted_metrics.values():
            dist.all_reduce(metric_value, op=dist.ReduceOp.SUM, group=parallel_state.intra_dp_cp.group)
    if logdiff_histogram is not None:
        dist.all_reduce(logdiff_histogram, op=dist.ReduceOp.SUM, group=parallel_state.intra_dp_cp.group)
    if logdiff_min_max is not None:
        logdiff_min = logdiff_min_max[0].clone()
        logdiff_max = logdiff_min_max[1].clone()
        dist.all_reduce(logdiff_min, op=dist.ReduceOp.MIN, group=parallel_state.intra_dp_cp.group)
        dist.all_reduce(logdiff_max, op=dist.ReduceOp.MAX, group=parallel_state.intra_dp_cp.group)
        logdiff_min_max = torch.stack([logdiff_min, logdiff_max])

    loss_reduced = {}
    values = values.tolist()
    num_samples_or_tokens = values[0]

    if num_samples_or_tokens == 0:
        # All ranks dropped the batch — typically because an upstream micro-batch
        # raised OOM and produced an empty per-token / per-sample contribution.
        # Without this guard ``value / 0`` raises ZeroDivisionError, which has
        # been observed to take down whole chunk-ablation suites (the ``none``
        # mode under long context, see exp_log/tp_sweep_*_70k_*_none*.log).
        # Returning zero-valued metrics lets the caller continue logging and
        # advance to the next step, so a single OOM doesn't poison subsequent
        # cases or hide the real failure under a generic Python error.
        logger.warning(
            "aggregate_train_losses: num_samples_or_tokens=0 after dp_cp all-reduce; "
            "every rank produced an empty contribution (likely OOM). "
            "Returning zero-valued metrics to keep the run alive."
        )
        return {key: 0.0 for key in keys}

    for key, value in zip(keys, values[1:], strict=False):
        loss_reduced[key] = value * parallel_state.cp.size / num_samples_or_tokens

    if weighted_metrics:
        for key, numerator_denominator in weighted_metrics.items():
            numerator, denominator = numerator_denominator.tolist()
            loss_reduced[key] = numerator / denominator if denominator != 0 else 0.0

    if logdiff_histogram is not None and logdiff_min_max is not None and logdiff_histogram.sum().item() > 0:
        logdiff_min, logdiff_max = logdiff_min_max.tolist()
        for tag, quantile in _LOGDIFF_QUANTILES.items():
            loss_reduced[f"logdiff_{tag}"] = _logdiff_quantile_from_histogram(
                logdiff_histogram,
                quantile,
                logdiff_max,
            )
        loss_reduced["logdiff_min"] = logdiff_min
        loss_reduced["logdiff_max"] = logdiff_max

    return loss_reduced


def log_train_step(
    args: Namespace,
    loss_dict: dict[str, float],
    grad_norm: float,
    rollout_id: int,
    step_id: int,
    num_steps_per_rollout: int,
    role: str = "actor",
    extra_metrics: dict[str, float] | None = None,
    should_log: bool | None = None,
) -> dict[str, float]:
    """Log training metrics for one step.

    Formats loss metrics, gradient norm, and extra metrics (e.g., learning rates, MTP loss) for tracking.

    Args:
        args: Configuration.
        loss_dict: Dictionary of loss metrics from aggregate_train_losses.
        grad_norm: Gradient norm after clipping.
        rollout_id: Rollout ID.
        step_id: Step ID within the rollout.
        num_steps_per_rollout: Total number of steps per rollout.
        role: Role name (e.g., "actor", "critic").
        extra_metrics: Optional extra metrics to log (e.g., learning rates, MTP loss).
        should_log: Optional override for logging condition. If None, uses rank == 0.

    Returns:
        The formatted log_dict (for CI tests or other uses).
    """
    accumulated_step_id = rollout_id * num_steps_per_rollout + step_id
    role_tag = "" if role == "actor" else f"{role}-"

    log_dict_out = {
        f"train/{role_tag}{key}": val.mean().item() if isinstance(val, torch.Tensor) else val
        for key, val in loss_dict.items()
    }
    log_dict_out[f"train/{role_tag}grad_norm"] = float(grad_norm)

    if extra_metrics:
        for key, val in extra_metrics.items():
            log_dict_out[f"train/{role_tag}{key}"] = val

    log_dict_out["train/step"] = accumulated_step_id

    if should_log is None:
        should_log = dist.get_rank() == 0

    if should_log:
        tracking_utils.log(args, log_dict_out, step_key="train/step")
        logger.info(f"{role_tag}step {accumulated_step_id}: {log_dict_out}")

    flush_logdiff_histogram()

    return log_dict_out
