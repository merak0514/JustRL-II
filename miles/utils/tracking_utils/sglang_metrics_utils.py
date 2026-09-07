"""
Shared helpers to fetch and parse SGLang Prometheus metrics.

Used by rollout logging (per-rollout ``sglang/*`` snapshots).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

AVG_METRICS = {
    "sglang:gen_throughput",
    "sglang:token_usage",
    "sglang:cache_hit_rate",
    "sglang:new_token_ratio",
    "sglang:utilization",
}
SUM_METRICS = {
    "sglang:num_running_reqs",
    "sglang:num_queue_reqs",
    "sglang:num_used_tokens",
    "sglang:max_total_num_tokens",
    "sglang:num_retracted_reqs",
    "sglang:num_paused_reqs",
}


def parse_prometheus_metrics(
    metrics_text: str,
    *,
    avg_keys: set[str] = AVG_METRICS,
    sum_keys: set[str] = SUM_METRICS,
) -> dict[str, float]:
    """Parse Prometheus exposition text into a flat ``short_name -> value`` dict."""
    accum: dict[str, list[float]] = {k: [] for k in avg_keys | sum_keys}

    for line in metrics_text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        name = parts[0].split("{")[0]
        if name not in accum:
            continue
        try:
            accum[name].append(float(parts[-1]))
        except ValueError:
            continue

    result: dict[str, float] = {}
    for key, vals in accum.items():
        if not vals:
            continue
        short = key.replace("sglang:", "")
        if key in avg_keys:
            result[short] = sum(vals) / len(vals)
        else:
            result[short] = sum(vals)
    return result


def _aggregate_worker_metrics_texts(metrics_texts: list[str]) -> dict[str, float]:
    """Aggregate raw per-worker ``/metrics`` bodies (matches pre-refactor rollout.py)."""
    all_keys = AVG_METRICS | SUM_METRICS
    accum: dict[str, list[float]] = {k: [] for k in all_keys}
    n_workers = 0

    for text in metrics_texts:
        n_workers += 1
        for line in text.splitlines():
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            name = parts[0].split("{")[0]
            if name not in all_keys:
                continue
            try:
                accum[name].append(float(parts[-1]))
            except ValueError:
                continue

    if n_workers == 0:
        return {}

    result: dict[str, float] = {}
    for key, vals in accum.items():
        if not vals:
            continue
        short = key.replace("sglang:", "")
        if key in AVG_METRICS:
            result[short] = sum(vals) / len(vals)
        else:
            result[short] = sum(vals)

    result["num_engines"] = float(n_workers)
    return result


def list_sglang_worker_urls(router_ip: str, router_port: int, timeout: float = 5.0) -> list[str]:
    import requests as sync_requests

    router_url = f"http://{router_ip}:{router_port}"
    try:
        resp = sync_requests.get(f"{router_url}/workers", timeout=timeout)
        resp.raise_for_status()
        return [w["url"] for w in resp.json()["workers"]]
    except Exception:
        resp = sync_requests.get(f"{router_url}/list_workers", timeout=timeout)
        resp.raise_for_status()
        return resp.json()["urls"]


def fetch_sglang_worker_metrics(router_ip: str, router_port: int, timeout: float = 10.0) -> dict[str, float]:
    """Fetch key Prometheus metrics from each SGLang engine worker ``/metrics`` endpoint."""
    try:
        urls = list_sglang_worker_urls(router_ip, router_port, timeout=min(timeout, 5.0))
    except Exception as exc:
        logger.warning("Failed to list sglang workers: %s", exc)
        return {}

    metrics_texts: list[str] = []
    for url in urls:
        try:
            import requests

            resp = requests.get(f"{url.rstrip('/')}/metrics", timeout=timeout)
            resp.raise_for_status()
            metrics_texts.append(resp.text)
        except Exception as exc:
            logger.debug("Failed to fetch metrics from %s: %s", url, exc)

    return _aggregate_worker_metrics_texts(metrics_texts)
