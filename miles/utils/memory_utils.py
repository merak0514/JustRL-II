import gc
import logging
import os
import time

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

_VRAM_MONITOR_START_TIME = time.monotonic()


def clear_memory(clear_host_memory: bool = False):
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    if clear_host_memory:
        torch._C._host_emptyCache()


def available_memory():
    device = torch.cuda.current_device()
    free, total = torch.cuda.mem_get_info(device)
    return {
        "gpu": str(device),
        "total_GB": _byte_to_gb(total),
        "free_GB": _byte_to_gb(free),
        "used_GB": _byte_to_gb(total - free),
        "allocated_GB": _byte_to_gb(torch.cuda.memory_allocated(device)),
        "reserved_GB": _byte_to_gb(torch.cuda.memory_reserved(device)),
    }


def _byte_to_gb(n: int):
    return round(n / (1024**3), 2)


def print_memory(msg, clear_before_print: bool = False):
    if clear_before_print:
        clear_memory()

    memory_info = available_memory()
    # Need to print for all ranks, b/c different rank can have different behaviors
    logger.info(
        f"[Rank {dist.get_rank()}] Memory-Usage {msg}{' (cleared before print)' if clear_before_print else ''}: {memory_info}"
    )
    return memory_info


def print_memory_extended(msg: str, clear_before_print: bool = False, reset_peak: bool = False):
    """Enhanced memory snapshot with peak stats, nvidia-smi process-level VRAM, and elapsed time."""
    if clear_before_print:
        clear_memory()

    device = torch.cuda.current_device()
    free, total = torch.cuda.mem_get_info(device)
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)

    elapsed = time.monotonic() - _VRAM_MONITOR_START_TIME

    nvidia_smi_used_gb = _get_nvidia_smi_used_gb(device)

    rank = dist.get_rank() if dist.is_initialized() else -1
    info = (
        f"[Rank {rank}] VRAM-Monitor [{msg}] t={elapsed:.1f}s | "
        f"alloc={_byte_to_gb(allocated)}GB peak_alloc={_byte_to_gb(peak_allocated)}GB | "
        f"reserved={_byte_to_gb(reserved)}GB peak_reserved={_byte_to_gb(peak_reserved)}GB | "
        f"free={_byte_to_gb(free)}GB total={_byte_to_gb(total)}GB | "
        f"nvidia_smi_used={nvidia_smi_used_gb}GB"
    )
    if clear_before_print:
        info += " (cleared)"
    logger.info(info)

    if reset_peak:
        torch.cuda.reset_peak_memory_stats(device)

    return {
        "gpu": str(device),
        "allocated_GB": _byte_to_gb(allocated),
        "peak_allocated_GB": _byte_to_gb(peak_allocated),
        "reserved_GB": _byte_to_gb(reserved),
        "peak_reserved_GB": _byte_to_gb(peak_reserved),
        "free_GB": _byte_to_gb(free),
        "total_GB": _byte_to_gb(total),
        "nvidia_smi_used_GB": nvidia_smi_used_gb,
        "elapsed_s": round(elapsed, 1),
    }


def _get_nvidia_smi_used_gb(device_index: int) -> float:
    """Get the process-level GPU memory usage via pynvml (same data as nvidia-smi)."""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        pynvml.nvmlShutdown()
        return _byte_to_gb(mem_info.used)
    except Exception:
        return -1.0
