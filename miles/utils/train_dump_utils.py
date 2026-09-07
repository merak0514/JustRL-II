import logging
import os
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

_DEBUG_DUMP_MAX_STRING_BYTES = int(os.environ.get("MILES_DEBUG_DUMP_MAX_STRING_BYTES", 16 * 1024 * 1024))
_DEBUG_DUMP_PICKLE_PROTOCOL = 4

_SKIP_DEBUG_TRAIN_DATA_KEYS = {
    "rollout_routed_experts",
    "rollout_routed_experts_weights",
}

def _sanitize_debug_dump(value: Any, *, stats: dict[str, int] | None = None) -> Any:
    if stats is None:
        stats = {"truncated": 0}

    if isinstance(value, str) and len(value) > _DEBUG_DUMP_MAX_STRING_BYTES:
        marker = f"\n...[truncated debug dump string: original_chars={len(value)}]...\n"
        keep = max((_DEBUG_DUMP_MAX_STRING_BYTES - len(marker)) // 2, 128)
        stats["truncated"] += 1
        return value[:keep] + marker + value[-keep:]
    if isinstance(value, bytes) and len(value) > _DEBUG_DUMP_MAX_STRING_BYTES:
        marker = f"\n...[truncated debug dump bytes: original_bytes={len(value)}]...\n".encode()
        keep = max((_DEBUG_DUMP_MAX_STRING_BYTES - len(marker)) // 2, 128)
        stats["truncated"] += 1
        return value[:keep] + marker + value[-keep:]
    if isinstance(value, dict):
        return {key: _sanitize_debug_dump(item, stats=stats) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_debug_dump(item, stats=stats) for item in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_debug_dump(item, stats=stats) for item in value)

    return value


def safe_torch_save_debug(obj: Any, path: str | Path) -> None:
    stats = {"truncated": 0}
    safe_obj = _sanitize_debug_dump(obj, stats=stats)
    if stats["truncated"]:
        logger.warning("Truncated %s oversized debug dump fields before saving %s", stats["truncated"], path)
    torch.save(safe_obj, path, pickle_protocol=_DEBUG_DUMP_PICKLE_PROTOCOL)


def save_debug_train_data(args, *, rollout_id, rollout_data):
    if (path_template := args.save_debug_train_data) is not None:
        rank = torch.distributed.get_rank()
        path = Path(path_template.format(rollout_id=rollout_id, rank=rank))
        logger.info(f"Save debug train data to {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            safe_torch_save_debug(
                dict(
                    rollout_id=rollout_id,
                    rank=rank,
                    rollout_data=rollout_data,
                ),
                path,
            )
        except OSError as e:
            logger.warning(f"Failed to save debug train data to {path}: {e}")
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
