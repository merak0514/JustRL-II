"""In-training export of the actor model to a HuggingFace-format directory.

Reuses the raw Megatron->HF weight gather path (the same collectives that push
weights into the rollout engines), so the export works under any TP/PP/CP/EP/DP
layout and produces exact HF-named, unquantized tensors. Rank 0 streams them to
sharded safetensors plus the usual `model.safetensors.index.json`, then copies
config/tokenizer assets from the base HF checkpoint so the directory is directly
loadable with `from_pretrained`.
"""

import json
import logging
import os
import shutil
import time
from pathlib import Path

import safetensors.torch
import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


@torch.no_grad()
def save_hf_model_raw(args, rollout_id: int, hf_weight_iterator, weights_getter) -> None:
    """Export current weights as a HuggingFace-format directory.

    This function is collective: every training rank must call it at the same
    time (the weight iterator runs broadcast/all_gather among all ranks). Only
    rank 0 writes files. The export is written to a `.tmp` directory and renamed
    on success, so a crash mid-export never leaves a directory that looks
    complete. Failures raise instead of being swallowed: a silently missing
    archive is worse than a visible error.
    """
    path = Path(args.save_hf.format(rollout_id=rollout_id))
    tmp_path = path.parent / f"{path.name}.tmp"
    is_writer = dist.get_rank() == 0

    if is_writer:
        logger.info(f"Saving HF-format model to {path}")
        if tmp_path.exists():
            shutil.rmtree(tmp_path)
        tmp_path.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    writer = _ShardedSafetensorsWriter(tmp_path) if is_writer else None
    for hf_named_tensors in hf_weight_iterator.get_hf_weight_chunks(weights_getter()):
        if writer is not None:
            for name, tensor in hf_named_tensors:
                # The tensors are (views of) this chunk's gather buffers; copying
                # to CPU here both bounds GPU memory to one chunk and gives each
                # tensor its own storage, which safetensors requires.
                writer.add(name, tensor.detach().contiguous().cpu())

    if writer is not None:
        writer.finalize()
        _copy_hf_assets(args.hf_checkpoint, tmp_path)
        if path.exists():
            shutil.rmtree(path)
        os.replace(tmp_path, path)
        logger.info(
            f"Saved HF-format model ({writer.total_size / 1024**3:.1f} GiB) to {path} "
            f"in {time.time() - start_time:.1f}s"
        )


class _ShardedSafetensorsWriter:
    """Stream tensors into safetensors shards of at most `max_shard_size` bytes.

    Shards are written under temporary names as they fill up (the final
    `model-XXXXX-of-XXXXX.safetensors` names need the total count), then renamed
    in `finalize()`, which also writes the index. Peak CPU memory is one shard.
    """

    def __init__(self, output_dir: Path, max_shard_size: int = 5 * 1024**3):
        self.output_dir = output_dir
        self.max_shard_size = max_shard_size
        self.total_size = 0
        self._current: dict[str, torch.Tensor] = {}
        self._current_size = 0
        self._flushed_shards: list[list[str]] = []
        self._seen_names: set[str] = set()

    def add(self, name: str, tensor: torch.Tensor) -> None:
        if name in self._seen_names:
            raise ValueError(f"Duplicate tensor name in HF export: {name}")
        self._seen_names.add(name)

        size = tensor.numel() * tensor.element_size()
        if self._current and self._current_size + size > self.max_shard_size:
            self._flush()
        self._current[name] = tensor
        self._current_size += size
        self.total_size += size

    def _flush(self) -> None:
        shard_id = len(self._flushed_shards)
        safetensors.torch.save_file(
            self._current,
            str(self.output_dir / _tmp_shard_name(shard_id)),
            metadata={"format": "pt"},
        )
        self._flushed_shards.append(list(self._current))
        self._current = {}
        self._current_size = 0

    def finalize(self) -> None:
        if self._current:
            self._flush()
        if not self._flushed_shards:
            raise RuntimeError("HF export produced no tensors; refusing to write an empty model directory.")

        num_shards = len(self._flushed_shards)
        weight_map = {}
        for shard_id, names in enumerate(self._flushed_shards):
            final_name = f"model-{shard_id + 1:05d}-of-{num_shards:05d}.safetensors"
            os.replace(self.output_dir / _tmp_shard_name(shard_id), self.output_dir / final_name)
            for name in names:
                weight_map[name] = final_name

        index = {"metadata": {"total_size": self.total_size}, "weight_map": weight_map}
        with open(self.output_dir / "model.safetensors.index.json", "w") as f:
            json.dump(index, f, indent=2)


def _tmp_shard_name(shard_id: int) -> str:
    return f"shard-{shard_id:05d}.safetensors.part"


def _copy_hf_assets(origin_hf_dir: str, output_dir: Path) -> None:
    """Copy config/tokenizer/etc. (everything except weights) from the base HF dir."""
    if origin_hf_dir is None or not os.path.isdir(origin_hf_dir):
        logger.warning(
            f"hf_checkpoint {origin_hf_dir!r} is not a local directory; "
            "the exported HF model will miss config/tokenizer files."
        )
        return
    for filename in os.listdir(origin_hf_dir):
        src = os.path.join(origin_hf_dir, filename)
        if not os.path.isfile(src):
            continue
        if filename.endswith((".safetensors", ".bin", ".pt", ".pth", ".index.json")):
            continue
        if filename == "config.json":
            # This export path always writes unquantized weights (the iterator is
            # built with quantization_config=None), so a quantized base model's
            # config must not claim otherwise.
            with open(src) as f:
                config = json.load(f)
            if config.pop("quantization_config", None) is not None:
                logger.warning(
                    "Stripped quantization_config from the exported config.json: "
                    "the raw --save-hf export writes unquantized weights."
                )
            with open(os.path.join(str(output_dir), filename), "w") as f:
                json.dump(config, f, indent=2)
            continue
        shutil.copy(src, os.path.join(str(output_dir), filename))
