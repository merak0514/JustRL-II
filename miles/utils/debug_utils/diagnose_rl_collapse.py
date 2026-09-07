"""Diagnose RLVR/GRPO collapse from logs and rollout debug dumps.

This utility is intentionally offline-only. It parses the scalar dictionaries
already emitted by MILES logs plus optional ``--save-debug-rollout-data`` dumps
and classifies the observed cliff into a small set of actionable buckets.
"""

from __future__ import annotations

import argparse
import ast
import glob
import json
import math
import re
import statistics
import zlib
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
DICT_RE = re.compile(r"\{.*\}")
PERF_RE = re.compile(r"\bperf\s+(\d+):\s+(\{.*\})")
ROLLOUT_RE = re.compile(r"\brollout\s+(\d+):\s+(\{.*\})")
GRPO_RE = re.compile(r"\bgrpo_metrics\s+(\d+):\s+(\{.*\})")
TRAIN_RE = re.compile(r"\bstep\s+(\d+):\s+(\{.*\})")
GRPO_SIGNAL_RE = re.compile(
    r"\[grpo_signal\]\s+rollout=(?P<step>\d+)\s+groups=(?P<groups>\d+)\s+"
    r"all_one=(?P<all_one>\d+).*?all_zero=(?P<all_zero>\d+).*?effective=(?P<effective>\d+)"
)


@dataclass
class StepMetrics:
    step: int
    values: dict[str, float] = field(default_factory=dict)
    sources: set[str] = field(default_factory=set)

    def update(self, data: dict[str, Any], source: str) -> None:
        self.sources.add(source)
        for key, value in data.items():
            numeric = _to_float(value)
            if numeric is not None and math.isfinite(numeric):
                self.values[key] = numeric


@dataclass
class Diagnosis:
    collapse_step: int | None
    dominant_cause: str
    confidence: str
    evidence: list[str]
    recommendations: list[str]


def _strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _safe_literal_dict(text: str) -> dict[str, Any] | None:
    match = DICT_RE.search(text)
    if not match:
        return None
    try:
        value = ast.literal_eval(match.group(0))
    except (SyntaxError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _merge_prefixed(step: StepMetrics, data: dict[str, Any], source: str, prefix: str | None = None) -> None:
    if prefix:
        data = {k if k.startswith(prefix) else f"{prefix}{k}": v for k, v in data.items()}
    step.update(data, source)


def parse_log_file(path: str | Path) -> dict[int, StepMetrics]:
    metrics: dict[int, StepMetrics] = {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = _strip_ansi(line)

            for regex, source in [(PERF_RE, "perf"), (ROLLOUT_RE, "rollout"), (GRPO_RE, "grpo")]:
                match = regex.search(line)
                if match:
                    step_id = int(match.group(1))
                    data = _safe_literal_dict(match.group(2))
                    if data is None:
                        continue
                    step = metrics.setdefault(step_id, StepMetrics(step=step_id))
                    _merge_prefixed(step, data, source)
                    break

            match = TRAIN_RE.search(line)
            if match and ("log_utils.py" in line or "model.py" in line):
                step_id = int(match.group(1))
                data = _safe_literal_dict(match.group(2))
                if data is not None:
                    step = metrics.setdefault(step_id, StepMetrics(step=step_id))
                    _merge_prefixed(step, data, "train")

            match = GRPO_SIGNAL_RE.search(line)
            if match:
                step_id = int(match.group("step"))
                groups = float(match.group("groups"))
                step = metrics.setdefault(step_id, StepMetrics(step=step_id))
                step.update(
                    {
                        "grpo_metrics/num_groups": groups,
                        "grpo_metrics/all_one_count": float(match.group("all_one")),
                        "grpo_metrics/all_zero_count": float(match.group("all_zero")),
                        "grpo_metrics/effective_count": float(match.group("effective")),
                    },
                    "grpo_signal",
                )
                if groups > 0:
                    step.update(
                        {
                            "grpo_metrics/all_one_ratio": float(match.group("all_one")) / groups,
                            "grpo_metrics/all_zero_ratio": float(match.group("all_zero")) / groups,
                            "grpo_metrics/effective_ratio": float(match.group("effective")) / groups,
                            "grpo_metrics/no_grad": 1.0 - float(match.group("effective")) / groups,
                        },
                        "grpo_signal",
                    )
    return metrics


def parse_logs(paths: list[str]) -> dict[int, StepMetrics]:
    merged: dict[int, StepMetrics] = {}
    for pattern in paths:
        for path in sorted(glob.glob(pattern)):
            for step_id, step_metrics in parse_log_file(path).items():
                merged_step = merged.setdefault(step_id, StepMetrics(step=step_id))
                merged_step.update(step_metrics.values, f"log:{path}")
                merged_step.sources.update(step_metrics.sources)
    return merged


def _get_metric(step: StepMetrics, names: list[str]) -> float | None:
    for name in names:
        if name in step.values:
            return step.values[name]
    return None


def _metric_series(steps: list[StepMetrics], names: list[str]) -> list[tuple[int, float]]:
    series = []
    for step in steps:
        value = _get_metric(step, names)
        if value is not None:
            series.append((step.step, value))
    return series


def _find_collapse_step(steps: list[StepMetrics], reward_floor: float, reward_drop_ratio: float) -> int | None:
    rewards = _metric_series(steps, ["rollout/raw_reward", "rollout/raw_reward_mean", "raw_reward_mean"])
    if not rewards:
        return None
    best_so_far = rewards[0][1]
    for step_id, reward in rewards:
        if reward <= reward_floor and best_so_far > reward_floor:
            return step_id
        if best_so_far > 0 and reward <= best_so_far * reward_drop_ratio:
            return step_id
        best_so_far = max(best_so_far, reward)
    return None


def _nearest_step(steps_by_id: dict[int, StepMetrics], step_id: int | None) -> StepMetrics | None:
    if step_id is None or not steps_by_id:
        return None
    if step_id in steps_by_id:
        return steps_by_id[step_id]
    nearest = min(steps_by_id, key=lambda candidate: abs(candidate - step_id))
    return steps_by_id[nearest]


def diagnose_metrics(
    metrics: dict[int, StepMetrics],
    *,
    collapse_step: int | None = None,
    window: int = 5,
    reward_floor: float = 0.02,
    reward_drop_ratio: float = 0.35,
) -> Diagnosis:
    steps = [metrics[k] for k in sorted(metrics)]
    if not steps:
        return Diagnosis(None, "no_data", "low", ["没有解析到可用指标。"], ["先启用训练日志或 debug rollout dump。"])

    if collapse_step is None:
        collapse_step = _find_collapse_step(steps, reward_floor, reward_drop_ratio)

    steps_by_id = {s.step: s for s in steps}
    focus = _nearest_step(steps_by_id, collapse_step) if collapse_step is not None else steps[-1]
    start = max(0, steps.index(focus) - window) if focus in steps else 0
    before = steps[start : steps.index(focus)] if focus in steps else []

    evidence: list[str] = []
    scores = Counter()

    reward = _get_metric(focus, ["rollout/raw_reward", "rollout/raw_reward_mean", "raw_reward_mean"])
    if reward is not None:
        evidence.append(f"focus_step={focus.step} reward={reward:.4g}")

    no_grad = _get_metric(focus, ["grpo_metrics/no_grad"])
    all_zero = _get_metric(focus, ["grpo_metrics/all_zero_ratio", "grpo_metrics/solve_none"])
    all_one = _get_metric(focus, ["grpo_metrics/all_one_ratio", "grpo_metrics/solve_all"])
    if no_grad is not None:
        evidence.append(f"grpo no_grad={no_grad:.2%}")
        if no_grad >= 0.5:
            scores["grpo_signal_starvation"] += 3
        elif no_grad >= 0.25:
            scores["grpo_signal_starvation"] += 1
    if all_zero is not None:
        evidence.append(f"all_zero/solve_none={all_zero:.2%}")
        if all_zero >= 0.35:
            scores["grpo_signal_starvation"] += 2
    if all_one is not None:
        evidence.append(f"all_one/solve_all={all_one:.2%}")
        if all_one >= 0.35:
            scores["grpo_signal_starvation"] += 1

    trunc = _get_metric(focus, ["rollout/truncated_ratio", "rollout/truncated"])
    response_len = _get_metric(focus, ["rollout/response_len/mean", "rollout/response_lengths"])
    if trunc is not None:
        evidence.append(f"truncated_ratio={trunc:.2%}")
        if trunc >= 0.25:
            scores["length_truncation_feedback"] += 3
    if response_len is not None and before:
        prev_lengths = [
            value
            for step in before
            if (value := _get_metric(step, ["rollout/response_len/mean", "rollout/response_lengths"])) is not None
        ]
        if prev_lengths:
            prev_mean = statistics.mean(prev_lengths)
            evidence.append(f"response_len_mean={response_len:.1f}, previous_window_mean={prev_mean:.1f}")
            if prev_mean > 0 and response_len >= prev_mean * 1.25:
                scores["length_truncation_feedback"] += 2

    env_error = _get_metric(focus, ["rollout/env_error_ratio", "env_error_ratio"])
    if env_error is not None:
        evidence.append(f"env_error_ratio={env_error:.2%}")
        if env_error >= 0.05:
            scores["reward_or_env_pipeline"] += 3

    ppo_kl = _get_metric(focus, ["train/ppo_kl"])
    clipfrac = _get_metric(focus, ["train/pg_clipfrac"])
    logdiff = _get_metric(focus, ["train/train_rollout_logprob_abs_diff"])
    grad_norm = _get_metric(focus, ["train/grad_norm"])
    if ppo_kl is not None:
        evidence.append(f"ppo_kl={ppo_kl:.4g}")
        if abs(ppo_kl) >= 0.05:
            scores["kl_ratio_update_instability"] += 2
    if clipfrac is not None:
        evidence.append(f"pg_clipfrac={clipfrac:.2%}")
        if clipfrac >= 0.4:
            scores["kl_ratio_update_instability"] += 2
    if logdiff is not None:
        evidence.append(f"train_rollout_logprob_abs_diff={logdiff:.4g}")
        if logdiff >= 0.08:
            scores["kl_ratio_update_instability"] += 2
    if grad_norm is not None:
        evidence.append(f"grad_norm={grad_norm:.4g}")
        if grad_norm >= 5:
            scores["kl_ratio_update_instability"] += 1

    entropy = _get_metric(focus, ["rollout/entropy", "train/entropy_loss"])
    if entropy is not None:
        evidence.append(f"entropy_or_entropy_loss={entropy:.4g}")
        previous_entropy = [
            value
            for step in before
            if (value := _get_metric(step, ["rollout/entropy", "train/entropy_loss"])) is not None
        ]
        if previous_entropy and statistics.mean(previous_entropy) > 0 and entropy < statistics.mean(previous_entropy) * 0.5:
            scores["entropy_or_mode_collapse"] += 3
    else:
        evidence.append("没有解析到非零 entropy 曲线，无法把问题直接定性为熵崩塌。")

    if not scores:
        scores["unknown_or_needs_samples"] += 1

    dominant, score = scores.most_common(1)[0]
    confidence = "high" if score >= 4 else "medium" if score >= 2 else "low"
    return Diagnosis(collapse_step, dominant, confidence, evidence, _recommendations(dominant))


def _recommendations(cause: str) -> list[str]:
    if cause == "grpo_signal_starvation":
        return [
            "启用或保留 mixed-group dynamic sampling，目标是 grpo_metrics/effective_ratio 长期高于 0.75。",
            "对长期 all-pass/all-fail prompt 做难度分桶或移出当前训练池。",
            "保留 stop-on-no-grad early stop，避免在无梯度 batch 上继续更新。",
        ]
    if cause == "length_truncation_feedback":
        return [
            "把 response length、truncated_ratio 纳入 kill-switch；cliff 前先回滚 checkpoint。",
            "加入软长度惩罚或截断惩罚，避免 binary reward 在长 CoT 截断时突然变成全 0。",
            "降低 late-stage LR/temperature/entropy_coef，或用分段 schedule 在高 reward 后降探索。",
        ]
    if cause == "kl_ratio_update_instability":
        return [
            "降低 LR 或缩短每个 rollout 的 actor 更新幅度，并加 target-KL/clipfrac 阈值。",
            "长响应优先 A/B GSPO，减少 token-level ratio 在 16k response 上的方差。",
            "继续监控 train_rollout_logprob_abs_diff；异常时检查 rollout/train logprob 对齐。",
        ]
    if cause == "reward_or_env_pipeline":
        return [
            "先修 reward parser/sandbox/env_error，不要先调算法参数。",
            "dump env_error 样本并分桶统计错误原因、超时和无效格式。",
            "给 reward 服务加熔断和重试上限，避免错误样本进入训练。",
        ]
    if cause == "entropy_or_mode_collapse":
        return [
            "加入 entropy/top1/unique-response 监控，确认是真正分布收缩而不是 reward 表观归零。",
            "增加 entropy bonus 或 temperature 的同时观察 truncation，避免更长 CoT 反向伤害 reward。",
            "必要时加入 KL-to-reference 或行为克隆混合，防止策略远离可用基座。",
        ]
    return [
        "补齐 debug rollout dump，并从 cliff 前 checkpoint 固定 seed 复现 3-5 个 rollout。",
        "优先验证 reward 管道、长度/truncation、GRPO group 方差，再调 entropy。",
    ]


def _reward_value(sample: dict[str, Any], reward_key: str | None) -> float | None:
    reward = sample.get("reward")
    if isinstance(reward, dict):
        if reward_key is None:
            return None
        reward = reward.get(reward_key)
    return _to_float(reward)


def _status_value(sample: dict[str, Any]) -> str:
    status = sample.get("status", "unknown")
    if isinstance(status, str):
        return status
    value = getattr(status, "value", None)
    return str(value or status)


def _has_repetition(text: str) -> bool:
    if len(text) <= 10000:
        return False
    tail = text[-10000:].encode("utf-8", errors="ignore")
    if not tail:
        return False
    compressed = zlib.compress(tail, 9)
    return len(tail) / max(len(compressed), 1) > 10


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    idx = min(int(q * (len(values) - 1)), len(values) - 1)
    return values[idx]


def _group_samples(samples: list[dict[str, Any]], group_size: int) -> list[list[dict[str, Any]]]:
    keyed: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for idx, sample in enumerate(samples):
        key = sample.get("group_index")
        if key is None:
            key = idx // group_size
        keyed[key].append(sample)
    return [keyed[k] for k in sorted(keyed)]


def analyze_samples(samples: list[dict[str, Any]], *, group_size: int, reward_key: str | None) -> dict[str, Any]:
    rewards = [_reward_value(s, reward_key) for s in samples]
    valid_rewards = [r for r in rewards if r is not None]
    lengths = [float(s.get("response_length") or 0) for s in samples]
    statuses = Counter(_status_value(s) for s in samples)
    env_errors = sum(1 for s in samples if isinstance(s.get("metadata"), dict) and s["metadata"].get("env_error"))
    responses = [str(s.get("response") or "") for s in samples]
    unique_responses = len(set(responses))

    groups = _group_samples(samples, group_size)
    all_one = all_zero = no_grad = effective = 0
    for group in groups:
        group_rewards = [_reward_value(s, reward_key) for s in group]
        group_rewards = [r for r in group_rewards if r is not None]
        if not group_rewards:
            continue
        mean_reward = statistics.mean(group_rewards)
        same_reward = max(group_rewards) == min(group_rewards)
        if same_reward:
            no_grad += 1
            if mean_reward > 0.99:
                all_one += 1
            elif mean_reward < 0.01:
                all_zero += 1
        else:
            effective += 1

    group_count = all_one + all_zero + effective
    return {
        "num_samples": len(samples),
        "reward_mean": statistics.mean(valid_rewards) if valid_rewards else None,
        "reward_min": min(valid_rewards) if valid_rewards else None,
        "reward_max": max(valid_rewards) if valid_rewards else None,
        "response_len_mean": statistics.mean(lengths) if lengths else None,
        "response_len_p95": _percentile(lengths, 0.95),
        "status_counts": dict(statuses),
        "truncated_ratio": statuses.get("truncated", 0) / len(samples) if samples else 0.0,
        "env_error_ratio": env_errors / len(samples) if samples else 0.0,
        "repetition_ratio": sum(1 for text in responses if _has_repetition(text)) / len(samples) if samples else 0.0,
        "unique_response_ratio": unique_responses / len(samples) if samples else 0.0,
        "group_count": group_count,
        "all_one_count": all_one,
        "all_zero_count": all_zero,
        "effective_count": effective,
        "no_grad_ratio": no_grad / group_count if group_count else None,
        "all_zero_ratio": all_zero / group_count if group_count else None,
        "all_one_ratio": all_one / group_count if group_count else None,
        "effective_ratio": effective / group_count if group_count else None,
    }


def load_rollout_dump(path: str | Path) -> list[dict[str, Any]]:
    import torch

    pack = torch.load(path, weights_only=False)
    samples = pack.get("samples", pack)
    if not isinstance(samples, list):
        raise ValueError(f"{path} does not contain a list of samples")
    result = []
    for sample in samples:
        if hasattr(sample, "to_dict"):
            sample = sample.to_dict()
        if not isinstance(sample, dict):
            raise ValueError(f"{path} contains unsupported sample type: {type(sample)!r}")
        result.append(sample)
    return result


def analyze_rollout_dumps(patterns: list[str], *, group_size: int, reward_key: str | None) -> dict[str, Any]:
    per_file = {}
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            samples = load_rollout_dump(path)
            per_file[path] = analyze_samples(samples, group_size=group_size, reward_key=reward_key)
    return per_file


def build_summary(
    metrics: dict[int, StepMetrics],
    diagnosis: Diagnosis,
    dump_metrics: dict[str, Any],
    *,
    max_steps: int = 12,
) -> dict[str, Any]:
    steps = [metrics[k] for k in sorted(metrics)]
    reward_series = _metric_series(steps, ["rollout/raw_reward", "rollout/raw_reward_mean", "raw_reward_mean"])
    start = max(0, len(reward_series) - max_steps)
    compact_steps = []
    for step_id, reward in reward_series[start:]:
        step = metrics[step_id]
        compact_steps.append(
            {
                "step": step_id,
                "reward": reward,
                "truncated_ratio": _get_metric(step, ["rollout/truncated_ratio", "rollout/truncated"]),
                "response_len_mean": _get_metric(step, ["rollout/response_len/mean", "rollout/response_lengths"]),
                "no_grad": _get_metric(step, ["grpo_metrics/no_grad"]),
                "ppo_kl": _get_metric(step, ["train/ppo_kl"]),
                "pg_clipfrac": _get_metric(step, ["train/pg_clipfrac"]),
                "logdiff": _get_metric(step, ["train/train_rollout_logprob_abs_diff"]),
            }
        )

    return {
        "diagnosis": {
            "collapse_step": diagnosis.collapse_step,
            "dominant_cause": diagnosis.dominant_cause,
            "confidence": diagnosis.confidence,
            "evidence": diagnosis.evidence,
            "recommendations": diagnosis.recommendations,
        },
        "recent_reward_steps": compact_steps,
        "rollout_dumps": dump_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", action="append", default=[], help="Training log path or glob. Can be repeated.")
    parser.add_argument("--rollout-dump", action="append", default=[], help="Debug rollout dump path or glob. Can be repeated.")
    parser.add_argument("--collapse-step", type=int, default=None, help="Known collapse rollout/step id.")
    parser.add_argument("--window", type=int, default=5, help="Number of previous steps used for local comparisons.")
    parser.add_argument("--group-size", type=int, default=16, help="n_samples_per_prompt for sample-level GRPO grouping.")
    parser.add_argument("--reward-key", default=None, help="Reward key when sample.reward is a dict.")
    parser.add_argument("--output-json", default=None, help="Optional path to write the structured report.")
    args = parser.parse_args()

    metrics = parse_logs(args.log)
    dump_metrics = analyze_rollout_dumps(args.rollout_dump, group_size=args.group_size, reward_key=args.reward_key)
    diagnosis = diagnose_metrics(metrics, collapse_step=args.collapse_step, window=args.window)
    summary = build_summary(metrics, diagnosis, dump_metrics)

    text = json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True)
    print(text)
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
