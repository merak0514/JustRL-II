"""Lightweight, step-gated dumper for the IcePop / TIS per-token scatter.

Captures per-token ``(train_log_prob, rollout_log_prob)`` on active response
tokens at a few chosen rollout steps, so a scatter of ``pi_train`` vs
``pi_infer`` (colored by whether IcePop keeps or masks the token) can be drawn
offline from the dumped tensors.

Design notes:
- Gated by ``args.icepop_dump_dir`` (falls back to ``args.dump_details``); a
  no-op otherwise, so it costs nothing on normal runs.
- Only fires when the current rollout id is in ``args.icepop_dump_steps``.
- Stores raw log-probs (float16) plus the clip thresholds; ``pi`` and the
  keep/mask flag are re-derived offline, so thresholds can be changed later.
- Records are buffered per rollout across microbatches and flushed once at the
  end of the training step, yielding one small file per rank per step.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

_CURRENT_ROLLOUT_ID: int | None = None
_CURRENT_STEP_ID: int | None = None
# rollout_id -> list of (train_lp_cpu, rollout_lp_cpu) chunks for this rank
_BUFFER: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = {}


def _dump_dir(args) -> str | None:
    return getattr(args, "icepop_dump_dir", None) or getattr(args, "dump_details", None)


def _enabled(args) -> bool:
    return _dump_dir(args) is not None and bool(getattr(args, "icepop_dump_steps", None))


def _is_target_step(args) -> bool:
    return _CURRENT_ROLLOUT_ID is not None and _CURRENT_ROLLOUT_ID in set(args.icepop_dump_steps)


def set_current_train_step(rollout_id: int, step_id: int) -> None:
    """Record the current (rollout_id, step_id) so the loss-side recorder can gate on it."""
    global _CURRENT_ROLLOUT_ID, _CURRENT_STEP_ID
    _CURRENT_ROLLOUT_ID = rollout_id
    _CURRENT_STEP_ID = step_id


def record_icepop_scatter(
    args,
    *,
    train_log_probs: torch.Tensor,
    rollout_log_probs: torch.Tensor,
    active_tokens: torch.Tensor,
) -> None:
    """Buffer active-token log-probs for the current step (per microbatch call).

    ``train_log_probs``, ``rollout_log_probs`` and ``active_tokens`` must be
    token-aligned 1D tensors (the CP-local, concatenated layout used by the
    policy loss). Only tokens with ``active_tokens == True`` are kept.
    """
    if not _enabled(args) or not _is_target_step(args):
        return

    mask = active_tokens.reshape(-1).bool()
    if mask.numel() == 0 or not bool(mask.any()):
        return

    train_lp = train_log_probs.reshape(-1)[mask].detach().float()
    rollout_lp = rollout_log_probs.reshape(-1)[mask].detach().float()

    # Bound per-call memory; final subsample happens at flush.
    headroom = max(int(getattr(args, "icepop_dump_max_tokens_per_rank", 12000)) * 4, 1)
    n = train_lp.numel()
    if n > headroom:
        idx = torch.randperm(n, device=train_lp.device)[:headroom]
        train_lp = train_lp[idx]
        rollout_lp = rollout_lp[idx]

    _BUFFER.setdefault(_CURRENT_ROLLOUT_ID, []).append((train_lp.cpu(), rollout_lp.cpu()))


def flush_icepop_scatter(args) -> None:
    """Write the buffered samples for the current rollout to disk (one file per rank)."""
    if not _enabled(args):
        return
    rollout_id = _CURRENT_ROLLOUT_ID
    chunks = _BUFFER.pop(rollout_id, None) if rollout_id is not None else None
    if not chunks:
        return

    train_lp = torch.cat([c[0] for c in chunks])
    rollout_lp = torch.cat([c[1] for c in chunks])

    cap = max(int(getattr(args, "icepop_dump_max_tokens_per_rank", 12000)), 1)
    n = train_lp.numel()
    if n > cap:
        idx = torch.randperm(n)[:cap]
        train_lp = train_lp[idx]
        rollout_lp = rollout_lp[idx]

    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    out_dir = Path(_dump_dir(args)) / "icepop_scatter" / f"step_{rollout_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"rank_{rank}.pt"

    try:
        torch.save(
            {
                "rollout_id": rollout_id,
                "rank": rank,
                "num_tokens": int(n),
                "train_log_probs": train_lp.to(torch.float16),
                "rollout_log_probs": rollout_lp.to(torch.float16),
                "tis_clip_low": float(getattr(args, "tis_clip_low", 0.0)),
                "tis_clip": float(getattr(args, "tis_clip", float("inf"))),
            },
            path,
        )
        logger.info(f"Dumped IcePop scatter samples ({min(n, cap)} tokens) to {path}")
    except OSError as exc:
        logger.warning(f"Failed to dump IcePop scatter samples to {path}: {exc}")
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
