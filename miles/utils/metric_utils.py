import math
from typing import Any, Literal

import numpy as np


def dict_add_prefix(d: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {f"{prefix}{k}": v for k, v in d.items()}


def compute_pass_rate(
    flat_rewards: list[float],
    group_size: int,
    num_groups: int | None = None,
):
    if group_size == 1:
        return {}

    if num_groups is None:
        num_groups = len(flat_rewards) // group_size

    pass_rate_name_list = [2**i for i in range(int(math.log2(group_size)) + 1)]

    assert len(flat_rewards) == num_groups * group_size, f"{len(flat_rewards)=} {num_groups=} {group_size=}"
    rewards_of_group = np.array(flat_rewards).reshape(num_groups, group_size)

    log_dict = {}
    for k in pass_rate_name_list:
        num_correct = np.sum(rewards_of_group == 1, axis=1)
        num_samples = np.full(num_groups, group_size)

        pass_k_estimates = _estimate_pass_at_k(num_samples, num_correct, k)

        pass_k = np.mean(pass_k_estimates)
        log_dict[f"pass@{k}"] = pass_k

    return log_dict


def _estimate_pass_at_k(num_samples, num_correct, k):
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n, c, k):
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples, num_correct, strict=False)])


def compute_statistics(values: list[float], percentiles: list[int] | None = None) -> dict[str, float]:
    values = np.array(values)
    result = {
        "mean": np.mean(values).item(),
        "median": np.median(values).item(),
        "max": np.max(values).item(),
        "min": np.min(values).item(),
    }
    if percentiles:
        for p in percentiles:
            result[f"p{p}"] = np.percentile(values, p).item()
    return result


def compression_ratio(
    data: str | bytes,
    *,
    encoding: str = "utf-8",
    algorithm: Literal["zlib", "gzip", "bz2", "lzma"] = "zlib",
    level: int = 9,
) -> tuple[float, float]:
    if isinstance(data, str):
        raw = data.encode(encoding)
    else:
        raw = data

    original = len(raw)
    if original == 0:
        return float("inf"), 0.0

    if algorithm == "zlib":
        import zlib

        compressed = zlib.compress(raw, level)
    elif algorithm == "gzip":
        import gzip

        compressed = gzip.compress(raw, compresslevel=level)
    elif algorithm == "bz2":
        import bz2

        compressed = bz2.compress(raw, compresslevel=level)
    elif algorithm == "lzma":
        import lzma

        compressed = lzma.compress(raw, preset=level)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    comp_len = len(compressed)
    if comp_len == 0:
        return float("inf"), 100.0

    ratio = original / comp_len
    savings_pct = 100.0 * (1.0 - comp_len / original)
    return ratio, savings_pct


def has_repetition(text: str):
    if len(text) > 10000 and compression_ratio(text[-10000:])[0] > 10:
        return True
    else:
        return False


def compute_grpo_sample_metrics(
    flat_rewards: list[float],
    group_size: int,
    num_groups: int | None = None,
    var_threshold: float = 0.01,
    high_mean_threshold: float = 0.99,
    low_mean_threshold: float = 0.01,
) -> dict[str, float]:
    """
    Compute GRPO group-level diagnostics.

    Legacy keys (kept for back-compat with NO_GRAD_STOP_THRESHOLD early-stop and
    existing tensorboard dashboards):
      - solve_all  : ratio of groups with var<thr and mean>hi  (== all_one_ratio for binary reward)
      - solve_none : ratio of groups with var<thr and mean<lo  (== all_zero_ratio for binary reward)
      - no_grad    : ratio of groups with var<thr (no gradient signal)

    Newly added keys (more direct names + absolute counts so a quick glance at
    console / TB tells you how many of the N groups actually contribute gradient):
      - all_one_ratio     : same as solve_all, plain English alias
      - all_zero_ratio    : same as solve_none, plain English alias
      - effective_ratio   : 1 - no_grad, fraction of groups that DO produce gradient
      - all_one_count     : absolute number of all-one groups (out of num_groups)
      - all_zero_count    : absolute number of all-zero groups
      - effective_count   : absolute number of groups that produce gradient
      - num_groups        : total groups in this rollout (handy denominator)
    """
    if group_size <= 1:
        return {}

    if num_groups is None:
        num_groups = len(flat_rewards) // group_size

    assert len(flat_rewards) == num_groups * group_size, f"{len(flat_rewards)=} {num_groups=} {group_size=}"
    rewards_of_group = np.array(flat_rewards).reshape(num_groups, group_size)

    group_vars = np.var(rewards_of_group, axis=1)
    group_means = np.mean(rewards_of_group, axis=1)

    low_var_mask = group_vars < var_threshold
    all_one_mask = low_var_mask & (group_means > high_mean_threshold)
    all_zero_mask = low_var_mask & (group_means < low_mean_threshold)

    all_one_count = int(all_one_mask.sum())
    all_zero_count = int(all_zero_mask.sum())
    no_grad_count = int(low_var_mask.sum())
    effective_count = int(num_groups - no_grad_count)

    solve_all = all_one_count / num_groups
    solve_none = all_zero_count / num_groups
    no_grad = no_grad_count / num_groups
    effective_ratio = 1.0 - no_grad

    return {
        # ---- legacy (do NOT rename or remove: NO_GRAD_STOP early-stop reads no_grad) ----
        "solve_all": float(solve_all),
        "solve_none": float(solve_none),
        "no_grad": float(no_grad),
        # ---- direct-name aliases for diagnostics ----
        "all_one_ratio": float(solve_all),
        "all_zero_ratio": float(solve_none),
        "effective_ratio": float(effective_ratio),
        # ---- absolute counts (denominator = num_groups, same on every rank for a global rollout batch) ----
        "all_one_count": float(all_one_count),
        "all_zero_count": float(all_zero_count),
        "effective_count": float(effective_count),
        "num_groups": float(num_groups),
    }


def compute_rollout_step(args, rollout_id):
    if args.wandb_always_use_train_step:
        return rollout_id * args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size
    return rollout_id
