from __future__ import annotations

import math
import os
import re
import threading
import time
from dataclasses import asdict, dataclass
from typing import Callable

import torch


PROMETHEUS_LINE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(?P<labels>[^}]*)\})?\s+(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)$"
)
PROMETHEUS_LABEL_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')

TARGET_METRIC_NAMES = {
    "sglang:num_running_reqs",
    "sglang:num_used_tokens",
    "sglang:gen_throughput",
    "sglang:num_queue_reqs",
    "sglang:token_usage",
}


@dataclass(frozen=True)
class DecodeMetricsSnapshot:
    timestamp: float
    num_running_reqs: float
    num_used_tokens: float
    gen_throughput: float
    num_queue_reqs: float
    token_usage: float


@dataclass(frozen=True)
class EngineDecodeMetricsSummary:
    enabled: bool
    error: str | None
    sample_count: int
    window_count: int
    wall_time_s: float
    active_time_s: float
    decoded_tokens: float
    sum_decoded_tokens_over_batch: float
    sum_decoded_tokens_times_ctx: float
    weighted_running_req_time: float
    weighted_ctx_time: float
    weighted_live_tokens_time: float
    weighted_throughput_time: float


@dataclass(frozen=True)
class RolloutModelMBUStats:
    hidden_size: int
    num_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    dense_layers: int
    moe_layers: int
    shared_weight_params_per_pass: int
    num_experts: int
    moe_router_topk: int
    expert_weight_params_per_expert: int


def _parse_prometheus_labels(raw_labels: str | None) -> dict[str, str]:
    if not raw_labels:
        return {}
    labels = {}
    for key, value in PROMETHEUS_LABEL_RE.findall(raw_labels):
        labels[key] = value.encode("utf-8").decode("unicode_escape")
    return labels


def _extract_metric_value(metrics_text: str, metric_name: str) -> float | None:
    values: list[float] = []
    for raw_line in metrics_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or metric_name not in line:
            continue
        match = PROMETHEUS_LINE_RE.match(line)
        if not match or match.group("name") != metric_name:
            continue
        labels = _parse_prometheus_labels(match.group("labels"))
        if labels.get("tp_rank", "0") != "0":
            continue
        if labels.get("pp_rank", "0") != "0":
            continue
        values.append(float(match.group("value")))
    if not values:
        return None
    return sum(values)


def parse_sglang_metrics_snapshot(metrics_text: str, timestamp: float | None = None) -> DecodeMetricsSnapshot | None:
    parsed = {}
    for metric_name in TARGET_METRIC_NAMES:
        parsed_name = metric_name.split(":", 1)[1]
        parsed[parsed_name] = _extract_metric_value(metrics_text, metric_name)
    if parsed["num_running_reqs"] is None or parsed["num_used_tokens"] is None or parsed["gen_throughput"] is None:
        return None
    return DecodeMetricsSnapshot(
        timestamp=time.time() if timestamp is None else timestamp,
        num_running_reqs=max(parsed["num_running_reqs"], 0.0),
        num_used_tokens=max(parsed["num_used_tokens"], 0.0),
        gen_throughput=max(parsed["gen_throughput"], 0.0),
        num_queue_reqs=max(parsed["num_queue_reqs"] or 0.0, 0.0),
        token_usage=max(parsed["token_usage"] or 0.0, 0.0),
    )


def summarize_decode_metrics_snapshots(
    snapshots: list[DecodeMetricsSnapshot],
) -> EngineDecodeMetricsSummary:
    if len(snapshots) < 2:
        return EngineDecodeMetricsSummary(
            enabled=True,
            error=None,
            sample_count=len(snapshots),
            window_count=0,
            wall_time_s=0.0,
            active_time_s=0.0,
            decoded_tokens=0.0,
            sum_decoded_tokens_over_batch=0.0,
            sum_decoded_tokens_times_ctx=0.0,
            weighted_running_req_time=0.0,
            weighted_ctx_time=0.0,
            weighted_live_tokens_time=0.0,
            weighted_throughput_time=0.0,
        )

    window_count = 0
    wall_time_s = 0.0
    active_time_s = 0.0
    decoded_tokens = 0.0
    sum_decoded_tokens_over_batch = 0.0
    sum_decoded_tokens_times_ctx = 0.0
    weighted_running_req_time = 0.0
    weighted_ctx_time = 0.0
    weighted_live_tokens_time = 0.0
    weighted_throughput_time = 0.0

    for prev, current in zip(snapshots, snapshots[1:], strict=False):
        dt = max(current.timestamp - prev.timestamp, 0.0)
        if dt <= 0:
            continue

        window_count += 1
        wall_time_s += dt

        running_req = max(current.num_running_reqs, 0.0)
        live_tokens = max(current.num_used_tokens, 0.0)
        throughput = max(current.gen_throughput, 0.0)
        if running_req <= 0:
            continue

        active_time_s += dt
        ctx_per_req = live_tokens / running_req
        decoded_tokens_interval = throughput * dt

        decoded_tokens += decoded_tokens_interval
        sum_decoded_tokens_over_batch += decoded_tokens_interval / running_req
        sum_decoded_tokens_times_ctx += decoded_tokens_interval * ctx_per_req
        weighted_running_req_time += running_req * dt
        weighted_ctx_time += ctx_per_req * dt
        weighted_live_tokens_time += live_tokens * dt
        weighted_throughput_time += throughput * dt

    return EngineDecodeMetricsSummary(
        enabled=True,
        error=None,
        sample_count=len(snapshots),
        window_count=window_count,
        wall_time_s=wall_time_s,
        active_time_s=active_time_s,
        decoded_tokens=decoded_tokens,
        sum_decoded_tokens_over_batch=sum_decoded_tokens_over_batch,
        sum_decoded_tokens_times_ctx=sum_decoded_tokens_times_ctx,
        weighted_running_req_time=weighted_running_req_time,
        weighted_ctx_time=weighted_ctx_time,
        weighted_live_tokens_time=weighted_live_tokens_time,
        weighted_throughput_time=weighted_throughput_time,
    )


class SGLangMetricsPoller:
    def __init__(self, fetch_metrics_text: Callable[[], str], poll_interval_s: float):
        self._fetch_metrics_text = fetch_metrics_text
        self._poll_interval_s = max(poll_interval_s, 0.1)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._snapshots: list[DecodeMetricsSnapshot] = []
        self._lock = threading.Lock()
        self._error: str | None = None
        self._enabled = False

    def _append_snapshot(self) -> None:
        snapshot = parse_sglang_metrics_snapshot(self._fetch_metrics_text(), timestamp=time.time())
        if snapshot is None:
            raise RuntimeError("failed to parse required SGLang scheduler metrics from /metrics")
        with self._lock:
            self._snapshots.append(snapshot)

    def _poll_loop(self) -> None:
        try:
            self._append_snapshot()
        except Exception as exc:  # pragma: no cover - best effort metrics path
            self._error = str(exc)
            self._enabled = False
            return

        while not self._stop_event.wait(self._poll_interval_s):
            try:
                self._append_snapshot()
            except Exception as exc:  # pragma: no cover - best effort metrics path
                self._error = str(exc)

        try:
            self._append_snapshot()
        except Exception as exc:  # pragma: no cover - best effort metrics path
            self._error = str(exc)

    def start(self) -> dict[str, object]:
        self._stop_event.clear()
        self._snapshots = []
        self._error = None
        self._enabled = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        return {"enabled": True, "poll_interval_s": self._poll_interval_s}

    def stop(self) -> dict[str, object]:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self._poll_interval_s * 3 + 1.0)
        with self._lock:
            snapshots = list(self._snapshots)
        if not self._enabled:
            return asdict(
                EngineDecodeMetricsSummary(
                    enabled=False,
                    error=self._error,
                    sample_count=len(snapshots),
                    window_count=0,
                    wall_time_s=0.0,
                    active_time_s=0.0,
                    decoded_tokens=0.0,
                    sum_decoded_tokens_over_batch=0.0,
                    sum_decoded_tokens_times_ctx=0.0,
                    weighted_running_req_time=0.0,
                    weighted_ctx_time=0.0,
                    weighted_live_tokens_time=0.0,
                    weighted_throughput_time=0.0,
                )
            )
        summary = summarize_decode_metrics_snapshots(snapshots)
        if self._error is not None:
            summary = EngineDecodeMetricsSummary(**{**asdict(summary), "error": self._error})
        return asdict(summary)


def _infer_peak_bandwidth_tb_per_gpu() -> float:
    env_value = os.environ.get("MILES_PEAK_BANDWIDTH_TB_PER_GPU", "").strip()
    if env_value:
        try:
            return float(env_value)
        except Exception:
            return 0.0

    try:
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0).upper()
            if "H100" in name or "H800" in name:
                return 3.35
    except Exception:
        pass
    return 0.0


def _count_dense_and_moe_layers(args) -> tuple[int, int]:
    num_layers = int(args.num_layers)
    num_experts = getattr(args, "num_experts", None)
    if num_experts in (None, 0):
        return num_layers, 0

    moe_layer_freq = getattr(args, "moe_layer_freq", None)
    if moe_layer_freq is None:
        return 0, num_layers
    if isinstance(moe_layer_freq, list):
        dense_layers = sum(1 for value in moe_layer_freq if value == 0)
        moe_layers = sum(1 for value in moe_layer_freq if value != 0)
        return dense_layers, moe_layers
    if moe_layer_freq <= 0:
        return num_layers, 0

    dense_layers = sum(1 for i in range(num_layers) if i % moe_layer_freq != 0)
    moe_layers = num_layers - dense_layers
    return dense_layers, moe_layers


def derive_rollout_model_mbu_stats(args) -> RolloutModelMBUStats:
    hidden_size = int(args.hidden_size)
    num_layers = int(args.num_layers)
    num_attention_heads = int(args.num_attention_heads)
    num_key_value_heads = int(getattr(args, "num_query_groups", num_attention_heads) or num_attention_heads)
    head_dim = int(getattr(args, "kv_channels", None) or (hidden_size // num_attention_heads))
    vocab_size = int(args.vocab_size)
    use_gated_attention = bool(
        getattr(args, "use_gated_attention", False) or getattr(args, "attention_output_gate", False)
    )
    tie_word_embeddings = not bool(getattr(args, "untie_embeddings_and_output_weights", False))
    dense_layers, moe_layers = _count_dense_and_moe_layers(args)

    q_proj_out = num_attention_heads * head_dim * (2 if use_gated_attention else 1)
    kv_proj_out = num_key_value_heads * head_dim
    attention_params_per_layer = hidden_size * (
        q_proj_out + kv_proj_out + kv_proj_out + num_attention_heads * head_dim
    )

    mlp_multiplier = 3 if bool(getattr(args, "swiglu", False)) else 2
    dense_mlp_params_per_layer = hidden_size * int(args.ffn_hidden_size) * mlp_multiplier

    num_experts = getattr(args, "num_experts", None)
    if num_experts in (None, 0):
        num_experts = None
    moe_ffn_hidden_size = int(getattr(args, "moe_ffn_hidden_size", getattr(args, "ffn_hidden_size", 0)) or 0)
    moe_router_topk = int(getattr(args, "moe_router_topk", 0) or 0)
    moe_shared_expert_intermediate_size = int(getattr(args, "moe_shared_expert_intermediate_size", 0) or 0)

    expert_mlp_params_per_expert = hidden_size * moe_ffn_hidden_size * mlp_multiplier
    shared_expert_params_per_layer = hidden_size * moe_shared_expert_intermediate_size * mlp_multiplier
    router_params_per_layer = hidden_size * (num_experts or 0)

    shared_weight_params_per_pass = (
        dense_layers * (attention_params_per_layer + dense_mlp_params_per_layer + 2 * hidden_size)
        + moe_layers * (attention_params_per_layer + shared_expert_params_per_layer + router_params_per_layer + 2 * hidden_size)
        + hidden_size
        + (0 if tie_word_embeddings else hidden_size * vocab_size)
    )

    return RolloutModelMBUStats(
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        dense_layers=dense_layers,
        moe_layers=moe_layers,
        shared_weight_params_per_pass=shared_weight_params_per_pass,
        num_experts=int(num_experts or 0),
        moe_router_topk=moe_router_topk,
        expert_weight_params_per_expert=expert_mlp_params_per_expert,
    )


def _estimate_moe_expert_weight_bytes_total(
    summary: dict[str, object],
    model_stats: RolloutModelMBUStats,
    dtype_bytes: int,
    *,
    bound_mode: str,
) -> float:
    if model_stats.moe_layers <= 0 or model_stats.num_experts <= 0 or model_stats.expert_weight_params_per_expert <= 0:
        return 0.0

    decode_passes = float(summary.get("sum_decoded_tokens_over_batch", 0.0) or 0.0)
    decoded_tokens = float(summary.get("decoded_tokens", 0.0) or 0.0)
    if decode_passes <= 0 or decoded_tokens <= 0:
        return 0.0

    avg_batch_size = max(decoded_tokens / decode_passes, 0.0)
    if bound_mode == "lower":
        if model_stats.moe_router_topk <= 0:
            return 0.0
        distinct_experts = min(float(model_stats.moe_router_topk), float(model_stats.num_experts))
    elif bound_mode == "upper":
        if model_stats.moe_router_topk <= 0:
            return 0.0
        distinct_experts = min(avg_batch_size * model_stats.moe_router_topk, float(model_stats.num_experts))
    elif bound_mode == "dense_equivalent":
        distinct_experts = float(model_stats.num_experts)
    else:
        raise ValueError(f"Unsupported MoE expert bound mode: {bound_mode}")

    # Treat MoE expert traffic as a per-pass batch estimate instead of scaling it linearly with tokens.
    expert_weight_params_per_pass = (
        model_stats.moe_layers * distinct_experts * model_stats.expert_weight_params_per_expert
    )
    return decode_passes * expert_weight_params_per_pass * dtype_bytes


def infer_rollout_kv_replication_factor(args, num_key_value_heads: int) -> float:
    tp_size = int(getattr(args, "rollout_num_gpus_per_engine", 1) or 1)
    if num_key_value_heads <= 0 or tp_size <= 0:
        return 1.0
    expanded = math.ceil(num_key_value_heads / tp_size) * tp_size
    return expanded / num_key_value_heads


def compute_rollout_step_mbu_metrics(args, engine_summaries: list[dict[str, object]], rollout_time_s: float) -> dict[str, float]:
    valid_summaries = [summary for summary in engine_summaries if summary.get("enabled", False)]
    if not valid_summaries:
        return {}

    model_stats = derive_rollout_model_mbu_stats(args)
    dtype_bytes = int(getattr(args, "rollout_mbu_dtype_bytes", 2))
    peak_bandwidth_tb_per_gpu = getattr(args, "rollout_mbu_peak_bandwidth_tb_per_gpu", None)
    if peak_bandwidth_tb_per_gpu is None:
        peak_bandwidth_tb_per_gpu = _infer_peak_bandwidth_tb_per_gpu()
    peak_bandwidth_tb_per_gpu = float(peak_bandwidth_tb_per_gpu or 0.0)

    total_decoded_tokens = sum(float(summary["decoded_tokens"]) for summary in valid_summaries)
    total_tokens_over_batch = sum(float(summary["sum_decoded_tokens_over_batch"]) for summary in valid_summaries)
    total_tokens_times_ctx = sum(float(summary["sum_decoded_tokens_times_ctx"]) for summary in valid_summaries)
    total_active_time = sum(float(summary["active_time_s"]) for summary in valid_summaries)
    total_weighted_running_req_time = sum(float(summary["weighted_running_req_time"]) for summary in valid_summaries)
    total_weighted_ctx_time = sum(float(summary["weighted_ctx_time"]) for summary in valid_summaries)
    total_weighted_live_tokens_time = sum(float(summary["weighted_live_tokens_time"]) for summary in valid_summaries)
    total_weighted_throughput_time = sum(float(summary["weighted_throughput_time"]) for summary in valid_summaries)

    metrics = {
        "perf/rollout_decode_window_count": float(sum(int(summary["window_count"]) for summary in valid_summaries)),
        "perf/rollout_decode_sample_count": float(sum(int(summary["sample_count"]) for summary in valid_summaries)),
        "perf/rollout_decode_tokens": total_decoded_tokens,
        "perf/rollout_decode_tokens_per_sec": total_decoded_tokens / rollout_time_s if rollout_time_s > 0 else 0.0,
        "perf/rollout_decode_busy_time": total_active_time / max(len(valid_summaries), 1),
        "perf/rollout_decode_busy_ratio": (
            (total_active_time / max(len(valid_summaries), 1)) / rollout_time_s if rollout_time_s > 0 else 0.0
        ),
        "perf/rollout_decode_avg_running_req": (
            total_weighted_running_req_time / total_active_time if total_active_time > 0 else 0.0
        ),
        "perf/rollout_decode_avg_ctx_per_req": (
            total_weighted_ctx_time / total_active_time if total_active_time > 0 else 0.0
        ),
        "perf/rollout_decode_avg_live_tokens": (
            total_weighted_live_tokens_time / total_active_time if total_active_time > 0 else 0.0
        ),
        "perf/rollout_decode_avg_gen_throughput_per_engine": (
            total_weighted_throughput_time / total_active_time if total_active_time > 0 else 0.0
        ),
    }

    if rollout_time_s <= 0 or total_decoded_tokens <= 0 or total_tokens_over_batch <= 0:
        return metrics

    shared_weight_bytes_per_pass = model_stats.shared_weight_params_per_pass * dtype_bytes
    shared_weight_bytes_total = shared_weight_bytes_per_pass * total_tokens_over_batch
    active_expert_weight_bytes_lower_total = sum(
        _estimate_moe_expert_weight_bytes_total(
            summary,
            model_stats,
            dtype_bytes,
            bound_mode="lower",
        )
        for summary in valid_summaries
    )
    active_expert_weight_bytes_upper_total = sum(
        _estimate_moe_expert_weight_bytes_total(
            summary,
            model_stats,
            dtype_bytes,
            bound_mode="upper",
        )
        for summary in valid_summaries
    )
    dense_equivalent_expert_weight_bytes_total = sum(
        _estimate_moe_expert_weight_bytes_total(
            summary,
            model_stats,
            dtype_bytes,
            bound_mode="dense_equivalent",
        )
        for summary in valid_summaries
    )
    kv_replication_factor = infer_rollout_kv_replication_factor(args, model_stats.num_key_value_heads)
    kv_bytes_per_token_coeff = (
        2.0
        * model_stats.num_layers
        * model_stats.num_key_value_heads
        * model_stats.head_dim
        * dtype_bytes
        * kv_replication_factor
    )

    weight_bytes_lower_total = shared_weight_bytes_total + active_expert_weight_bytes_lower_total
    weight_bytes_upper_total = shared_weight_bytes_total + active_expert_weight_bytes_upper_total
    weight_bytes_dense_equivalent_total = shared_weight_bytes_total + dense_equivalent_expert_weight_bytes_total
    kv_bytes_total = kv_bytes_per_token_coeff * total_tokens_times_ctx

    metrics["perf/rollout_decode_weight_bytes_per_token_lower_bound"] = weight_bytes_lower_total / total_decoded_tokens
    metrics["perf/rollout_decode_weight_bytes_per_token_upper_bound"] = weight_bytes_upper_total / total_decoded_tokens
    # Keep the legacy metric as the MoE overlap upper bound for dashboard compatibility.
    metrics["perf/rollout_decode_weight_bytes_per_token"] = metrics[
        "perf/rollout_decode_weight_bytes_per_token_upper_bound"
    ]
    metrics["perf/rollout_decode_kv_bytes_per_token"] = kv_bytes_total / total_decoded_tokens
    metrics["perf/rollout_decode_kv_replication_factor"] = kv_replication_factor

    total_rollout_gpus = len(valid_summaries) * int(getattr(args, "rollout_num_gpus_per_engine", 1) or 1)
    if peak_bandwidth_tb_per_gpu <= 0 or total_rollout_gpus <= 0:
        return metrics

    deployed_peak_bandwidth_bytes_per_s = peak_bandwidth_tb_per_gpu * total_rollout_gpus * 1e12
    metrics["perf/rollout_decode_peak_bandwidth_tb_per_gpu"] = peak_bandwidth_tb_per_gpu
    metrics["perf/rollout_decode_peak_bandwidth_tbps_all_gpus"] = peak_bandwidth_tb_per_gpu * total_rollout_gpus

    metrics["perf/rollout_decode_mbu_lower_bound"] = (
        weight_bytes_lower_total + kv_bytes_total
    ) / (
        rollout_time_s * deployed_peak_bandwidth_bytes_per_s
    )
    metrics["perf/rollout_decode_mbu_upper_bound"] = (
        weight_bytes_upper_total + kv_bytes_total
    ) / (
        rollout_time_s * deployed_peak_bandwidth_bytes_per_s
    )
    metrics["perf/rollout_decode_mbu"] = metrics["perf/rollout_decode_mbu_upper_bound"]

    if model_stats.moe_layers > 0:
        metrics["perf/rollout_decode_weight_bytes_per_token_dense_equivalent"] = (
            weight_bytes_dense_equivalent_total / total_decoded_tokens
        )
        metrics["perf/rollout_decode_mbu_dense_equivalent"] = (
            weight_bytes_dense_equivalent_total + kv_bytes_total
        ) / (
            rollout_time_s * deployed_peak_bandwidth_bytes_per_s
        )

    return metrics
