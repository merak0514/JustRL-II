"""
SwanLab tracking backend for miles.
"""

from __future__ import annotations

import logging
import os
from copy import deepcopy
from typing import Any

from miles.utils.env_report import decode_env_report

logger = logging.getLogger(__name__)


def _compute_config_for_logging(args) -> dict:
    output = deepcopy(args.__dict__)
    whitelist_env_vars = ["SLURM_JOB_ID"]
    output["env_vars"] = {k: v for k, v in os.environ.items() if k in whitelist_env_vars}
    if env_report_raw := getattr(args, "env_report", None):
        if launcher_report := decode_env_report(env_report_raw):
            output["launcher_env_report"] = launcher_report
    return output


def _resolve_project(args) -> str:
    return (
        getattr(args, "swanlab_project", None)
        or os.environ.get("SWANLAB_PROJ_NAME")
        or getattr(args, "wandb_project", None)
        or "miles"
    )


def _resolve_experiment_name(args) -> str:
    return (
        getattr(args, "swanlab_experiment_name", None)
        or os.environ.get("SWANLAB_EXP_NAME")
        or getattr(args, "wandb_group", None)
        or "miles-run"
    )


def _get_run_id(run) -> str | None:
    if run is None:
        return None
    for attr in ("id", "run_id"):
        if run_id := getattr(run, attr, None):
            return str(run_id)
    try:
        import swanlab

        active = swanlab.get_run()
        if active is not None and getattr(active, "id", None):
            return str(active.id)
    except Exception:
        pass
    return None


_STEP_AXIS_KEYS = ("train/step", "rollout/step", "eval/step")


def _init_swanlab_common() -> None:
    """Record step-axis metrics for SwanLab UI custom X-axis (WandB define_metric equivalent).

    SwanLab has no ``define_metric`` API; logging ``train/step``, ``rollout/step``, and
    ``eval/step`` as scalars lets the dashboard bind chart X-axes to the correct step counter.
    """
    logger.debug(
        "SwanLab step axes: %s (set chart X-axis to these keys in the SwanLab UI when needed)",
        ", ".join(_STEP_AXIS_KEYS),
    )


def init_swanlab(args, *, primary: bool = True, **kwargs) -> None:
    if not getattr(args, "use_swanlab", False):
        args.swanlab_run_id = None
        return

    import swanlab

    api_key = getattr(args, "swanlab_api_key", None) or os.environ.get("SWANLAB_API_KEY")
    if api_key:
        swanlab.login(api_key)

    logdir = getattr(args, "swanlab_log_dir", None) or os.environ.get("SWANLAB_LOG_DIR", "swanlog")
    mode = getattr(args, "swanlab_mode", None) or os.environ.get("SWANLAB_MODE", "cloud")
    os.makedirs(logdir, exist_ok=True)

    project = _resolve_project(args)
    experiment_name = _resolve_experiment_name(args)
    run_id = getattr(args, "swanlab_run_id", None) or os.environ.get("SWANLAB_RUN_ID")
    resume = os.environ.get("SWANLAB_RESUME")

    init_kwargs: dict[str, Any] = {
        "project": project,
        "experiment_name": experiment_name,
        "config": {"FRAMEWORK": "miles", **_compute_config_for_logging(args)},
        "logdir": logdir,
        "mode": mode,
    }

    if primary:
        if run_id:
            init_kwargs["id"] = run_id
            if resume:
                init_kwargs["resume"] = resume
        run = swanlab.init(**init_kwargs)
        _init_swanlab_common()
        args.swanlab_run_id = _get_run_id(run)
        logger.info(
            "SwanLab primary run started (project=%s, experiment=%s, id=%s)",
            project,
            experiment_name,
            args.swanlab_run_id,
        )
    else:
        shared_id = run_id or getattr(args, "swanlab_run_id", None)
        if not shared_id:
            logger.warning("SwanLab secondary init skipped: missing swanlab_run_id on args.")
            return
        init_kwargs["parallel"] = "shared"
        init_kwargs["id"] = shared_id
        if resume:
            init_kwargs["resume"] = resume
        swanlab.init(**init_kwargs)
        _init_swanlab_common()
        logger.info("SwanLab secondary attached to run id=%s", shared_id)


def log_metrics(metrics: dict[str, Any], step: int | None = None, step_key: str | None = None) -> None:
    import swanlab

    if step is None and step_key:
        raw = metrics.get(step_key)
        if raw is not None:
            step = int(raw)

    step_axes = {k: float(metrics[k]) for k in _STEP_AXIS_KEYS if k in metrics}
    data = {k: v for k, v in metrics.items() if not k.endswith("/step")}

    if step_axes and step is not None:
        swanlab.log(step_axes, step=step)

    if data:
        swanlab.log(data, step=step)


def log_histogram(tag: str, values, step: int | None = None) -> None:
    import numpy as np
    import swanlab

    if hasattr(values, "detach"):
        values = values.detach()
    if hasattr(values, "cpu"):
        values = values.cpu()
    if hasattr(values, "numpy"):
        arr = values.numpy()
    else:
        arr = np.asarray(values)
    arr = np.asarray(arr).astype(np.float64).reshape(-1)
    if arr.size == 0:
        return

    swanlab.log(
        {
            f"{tag}/count": float(arr.size),
            f"{tag}/mean": float(np.mean(arr)),
            f"{tag}/std": float(np.std(arr)),
            f"{tag}/min": float(np.min(arr)),
            f"{tag}/max": float(np.max(arr)),
            f"{tag}/p50": float(np.percentile(arr, 50)),
            f"{tag}/p99": float(np.percentile(arr, 99)),
        },
        step=step,
    )

    try:
        counts, bin_edges = np.histogram(arr, bins=32)
        labels = [f"{bin_edges[i]:.4g}" for i in range(len(counts))]
        bar = swanlab.echarts.Bar()
        bar.add_xaxis(labels)
        bar.add_yaxis("count", counts.tolist())
        swanlab.log({tag: bar}, step=step)
    except Exception:
        logger.debug("SwanLab echarts histogram for %s skipped", tag, exc_info=True)


def finish() -> None:
    import swanlab

    try:
        swanlab.finish()
    except Exception:
        logger.exception("Error finishing SwanLab run")
