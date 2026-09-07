import os
from argparse import Namespace
from collections.abc import Callable, Iterator
from typing import Any

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint

from miles.utils.distributed_utils import distributed_masked_whiten
from miles.utils.misc import load_function
from miles.utils.ppo_utils import (
    calculate_log_probs_and_entropy,
    compute_approx_kl,
    compute_gspo_kl,
    compute_opsm_mask,
    compute_overlong_penalty,
    compute_policy_loss,
    get_advantages_and_returns_batch,
    get_grpo_returns,
    get_reinforce_plus_plus_baseline_advantages,
    get_reinforce_plus_plus_returns,
)
from miles.utils.types import RolloutBatch

from .cp_utils import (
    _allgather_cp_redistribute,
    all_gather_with_cp,
    get_local_response_loss_masks,
    get_logits_and_tokens_offset_with_cp,
    get_sum_of_sample_mean,
    slice_loss_masks_for_local_cp,
)
from .parallel import ParallelState, get_parallel_state

try:
    from flash_attn.ops.triton.cross_entropy import cross_entropy_loss

    _FLASH_ATTN_CROSS_ENTROPY_AVAILABLE = True
except ImportError:
    _FLASH_ATTN_CROSS_ENTROPY_AVAILABLE = False

_logdiff_histogram_buffer: list[torch.Tensor] = []
_LOGDIFF_HISTOGRAM_BINS = 4096
_LOGDIFF_HISTOGRAM_MIN_EXP = -8
_LOGDIFF_HISTOGRAM_MAX_EXP = 2


def _accumulate_logdiff_histogram(values: torch.Tensor) -> None:
    _logdiff_histogram_buffer.append(values.detach().cpu())


def flush_logdiff_histogram() -> torch.Tensor | None:
    """Return concatenated abs-diff values collected during the last train step and clear the buffer."""
    if not _logdiff_histogram_buffer:
        return None
    data = torch.cat(_logdiff_histogram_buffer)
    _logdiff_histogram_buffer.clear()
    return data


def _make_logdiff_histogram(values: torch.Tensor) -> torch.Tensor:
    """Build fixed log-spaced histogram counts for distributed logdiff quantiles."""
    clean_values = torch.nan_to_num(values.detach().float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    edges = torch.logspace(
        _LOGDIFF_HISTOGRAM_MIN_EXP,
        _LOGDIFF_HISTOGRAM_MAX_EXP,
        steps=_LOGDIFF_HISTOGRAM_BINS,
        device=clean_values.device,
    )
    buckets = torch.bucketize(clean_values, edges)
    return torch.bincount(buckets, minlength=_LOGDIFF_HISTOGRAM_BINS + 1).to(dtype=torch.float64)


def _squeeze_trailing_singleton(t: torch.Tensor) -> torch.Tensor:
    """Remove a trailing size-1 dim from (..., 1) without turning 1D (1,) into a 0-D scalar.

    ``squeeze(-1)`` on shape ``(1,)`` yields a scalar and breaks ``torch.cat`` for GRPO
    advantages and rollout logging.
    """
    if t.dim() > 1 and t.size(-1) == 1:
        return t.squeeze(-1)
    return t


import logging as _logging

_diag_logger = _logging.getLogger("logdiff_diag")


def _log_extreme_logdiff_diagnostic(
    flat_diff: torch.Tensor,
    old_log_probs: torch.Tensor,
    rollout_log_probs: torch.Tensor,
    batch: dict,
    args,
    top_k: int = 10,
) -> None:
    """Log detailed info about the top-K tokens with the highest logdiff."""
    try:
        n = flat_diff.numel()
        topk_vals, topk_flat_idxs = flat_diff.topk(min(top_k, n))

        old_lp = old_log_probs.detach().float()
        roll_lp = rollout_log_probs.detach().float()

        ref_list = batch["rollout_log_probs"]
        sample_boundaries = []
        offset = 0
        for i, lp in enumerate(ref_list):
            seg_len = lp.size(0)
            sample_boundaries.append((offset, offset + seg_len, i))
            offset += seg_len

        total_lengths = batch.get("total_lengths", [])
        response_lengths = batch.get("response_lengths", [])

        lines = [f"\n{'='*70}", f"LOGDIFF EXTREME DIAGNOSTIC (top {min(top_k, n)} tokens)", f"{'='*70}"]
        for rank, (val, flat_idx) in enumerate(zip(topk_vals.tolist(), topk_flat_idxs.tolist())):
            sample_idx = -1
            pos_in_sample = -1
            for (start, end, si) in sample_boundaries:
                if start <= flat_idx < end:
                    sample_idx = si
                    pos_in_sample = flat_idx - start
                    break

            train_lp_val = old_lp[flat_idx].item() if flat_idx < old_lp.numel() else float('nan')
            roll_lp_val = roll_lp[flat_idx].item() if flat_idx < roll_lp.numel() else float('nan')

            tlen = total_lengths[sample_idx] if sample_idx < len(total_lengths) else -1
            rlen = response_lengths[sample_idx] if sample_idx < len(response_lengths) else -1

            tokens_list = batch.get("unconcat_tokens")
            token_id = -1
            if tokens_list and 0 <= sample_idx < len(tokens_list):
                tok_tensor = tokens_list[sample_idx]
                prompt_len = tlen - rlen if tlen > 0 and rlen > 0 else 0
                abs_pos = prompt_len + pos_in_sample
                if 0 <= abs_pos < tok_tensor.numel():
                    token_id = tok_tensor[abs_pos].item()

            frac_in_resp = pos_in_sample / rlen if rlen > 0 else -1

            lines.append(
                f"  #{rank}: diff={val:.4f} | sample={sample_idx} "
                f"pos_in_resp={pos_in_sample}/{rlen} ({frac_in_resp:.1%}) "
                f"total_len={tlen} "
                f"| train_lp={train_lp_val:.6f} rollout_lp={roll_lp_val:.6f} "
                f"| token_id={token_id}"
            )

        cp_size = getattr(args, "context_parallel_size", 1)
        lines.append(f"  CP size: {cp_size}")
        if cp_size > 1 and response_lengths:
            chunk_sz = response_lengths[0] // (2 * cp_size) if response_lengths[0] > 0 else 0
            lines.append(f"  Approx CP chunk size per sample: {chunk_sz}")

        lines.append(f"{'='*70}")
        _diag_logger.warning("\n".join(lines))
    except Exception as e:
        _diag_logger.warning(f"Diagnostic failed: {e}")


# ---------------------------------------------------------------------------
# Verl-style log_probs and entropy (no TP; used when TP size == 1)
# ---------------------------------------------------------------------------


def _logprobs_from_logits(
    logits: torch.Tensor, labels: torch.Tensor, inplace_backward: bool = True
) -> torch.Tensor:
    """Per-token log-probabilities via Flash Attention fused cross-entropy when
    available, otherwise a memory-efficient row-wise fallback."""
    if _FLASH_ATTN_CROSS_ENTROPY_AVAILABLE:
        batch_dim = logits.shape[:-1]
        last_dim = logits.shape[-1]
        logits_flat = logits.reshape(-1, last_dim)
        labels_flat = labels.reshape(-1)
        output = cross_entropy_loss(logits_flat, labels_flat, inplace_backward=inplace_backward)
        assert isinstance(output, tuple), (
            "please make sure flash-attn>=2.4.3 where cross_entropy_loss returns Tuple[losses, z_losses]."
        )
        return (-output[0]).view(*batch_dim)
    return _logprobs_from_logits_v2(logits, labels)


def _logprobs_from_logits_v2(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Memory-efficient log-probability using row-wise processing.
    Uses logsumexp for float32/float64, row-wise log_softmax for bf16."""
    if logits.size(0) == 0:
        return logits.new_zeros((0,))
    if logits.dtype in (torch.float32, torch.float64):
        logits_labels = torch.gather(logits, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        logsumexp_values = torch.stack([torch.logsumexp(logit, dim=-1) for logit in logits])
        return logits_labels - logsumexp_values
    logprobs_labels = []
    for row_logits, row_labels in zip(logits, labels, strict=True):
        row_logprobs = F.log_softmax(row_logits, dim=-1)
        logprobs_labels.append(row_logprobs.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1))
    return torch.stack(logprobs_labels)


def _entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Shannon entropy: H = logsumexp(logits) - sum(p * logits)."""
    logits = logits.float()
    pd = F.softmax(logits, dim=-1)
    return torch.logsumexp(logits, dim=-1) - (pd * logits).sum(dim=-1)


def _entropy_from_logits_chunked(logits: torch.Tensor, chunk_size: int = 2048) -> torch.Tensor:
    """Memory-efficient entropy by processing rows in chunks."""
    if logits.size(0) == 0:
        return logits.new_zeros((0,))
    entropy = torch.zeros(logits.size(0), device=logits.device, dtype=torch.float32)
    for i in range(0, logits.size(0), chunk_size):
        chunk = logits[i : i + chunk_size].float()
        pd = F.softmax(chunk, dim=-1)
        entropy[i : i + chunk_size] = torch.logsumexp(chunk, dim=-1) - (pd * chunk).sum(dim=-1)
    return entropy


def get_responses(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    max_seq_lens: list[int] | None = None,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield response-aligned `(logits_chunk, tokens_chunk)` pairs per sample.

    After squeezing batch dimension and applying temperature scaling, this
    function extracts the logits and tokens corresponding to response segments
    for each sample. When context parallelism is disabled, it slices directly
    from the concatenated sequence. With context parallelism enabled, it
    handles split sequences across ranks.

    Args:
        logits: Model outputs with shape `[1, T, V]` (policy) or `[1, T, 1]`
            (value). Can be fp32/fp16/bf16. For long-context prompts, keeping
            full logits in half precision avoids OOM; we only upcast the sliced
            response chunks (small) to fp32 for stable log-prob computation.
        args: Configuration containing `rollout_temperature` for scaling.
        unconcat_tokens: List of token tensors (prompt+response) per sample.
        total_lengths: Total sequence lengths (prompt+response) per sample.
        response_lengths: Response segment lengths per sample.

    Yields:
        Tuple of `(logits_chunk, tokens_chunk)` where `logits_chunk` is shape
        `[R, V]` (policy) or `[R, 1]` (value) and `tokens_chunk` is shape `[R]`
        (1D int64), both aligned to response tokens for one sample.
    """
    parallel_state = get_parallel_state()
    qkv_format = args.qkv_format

    assert logits.is_floating_point(), f"{logits.dtype}"
    assert len(logits.shape) == 3, f"{logits.shape}"

    if qkv_format == "thd":
        assert logits.size(0) == 1, f"{logits.shape}"
        logits = logits.squeeze(0)
    else:
        assert max_seq_lens is not None
        logits = logits.view(-1, logits.size(-1))

    # IMPORTANT(long-context): never apply temperature scaling to the full logits
    # tensor here (it would allocate a full-size copy). Apply scaling to the
    # per-sample response slice instead.

    cp_size = parallel_state.cp.size
    end = 0
    seq_start = 0
    for i, (tokens, total_length, response_length) in enumerate(
        zip(unconcat_tokens, total_lengths, response_lengths, strict=False)
    ):
        max_seq_len = max_seq_lens[i] if max_seq_lens is not None else None

        if cp_size == 1:
            if qkv_format == "bshd":
                end = max_seq_len * i + total_length
                start = end - response_length
                logits_chunk = logits[start - 1 : end - 1]
            else:
                end += total_length
                start = end - response_length
                logits_chunk = logits[start - 1 : end - 1]
            tokens_chunk = tokens[-response_length:]
        elif args.allgather_cp:
            # DSA: global concat then contiguous CP split. Each rank owns logits for
            # global positions [chunk_start, chunk_end).
            logits_local_len = logits.size(0)
            cp_rank = parallel_state.cp.rank
            chunk_start = cp_rank * logits_local_len
            chunk_end = chunk_start + logits_local_len

            prompt_length = total_length - response_length
            resp_token_start = seq_start + prompt_length
            resp_token_end = seq_start + total_length
            logit_global_start = resp_token_start - 1
            logit_global_end = resp_token_end - 1

            s = max(logit_global_start, chunk_start)
            e = min(logit_global_end, chunk_end)
            if e <= s:
                logits_chunk = logits[0:0]
                tokens_chunk = tokens[0:0]
            else:
                logits_chunk = logits[s - chunk_start : e - chunk_start]
                tokens_chunk = tokens[(s + 1) - seq_start : (e + 1) - seq_start]
            assert logits_chunk.size(0) == tokens_chunk.size(0), f"{logits_chunk.size(0)} vs {tokens_chunk.size(0)}"
        else:
            # TODO: this is super ugly... do better abstraction.
            chunk_size, chunks_offset, logits_offset, tokens_offset = get_logits_and_tokens_offset_with_cp(
                total_length, response_length, qkv_format, max_seq_len
            )

            logits_0, logits_1 = logits[end : end + chunk_size], logits[end + chunk_size : end + 2 * chunk_size]
            end += 2 * chunk_size

            logits_0 = logits_0[logits_offset[0][0] - chunks_offset[0][0] : logits_offset[0][1] - chunks_offset[0][0]]
            tokens_0 = tokens[tokens_offset[0][0] : tokens_offset[0][1]]

            logits_1 = logits_1[logits_offset[1][0] - chunks_offset[1][0] : logits_offset[1][1] - chunks_offset[1][0]]
            tokens_1 = tokens[tokens_offset[1][0] : tokens_offset[1][1]]

            assert logits_0.size(0) == tokens_0.size(0), f"{logits_0.size(0)} vs {tokens_0.size(0)}"
            assert logits_1.size(0) == tokens_1.size(0), f"{logits_1.size(0)} vs {tokens_1.size(0)}"

            logits_chunk = torch.cat([logits_0, logits_1], dim=0)
            tokens_chunk = torch.cat([tokens_0, tokens_1], dim=0)

        # Apply temperature scaling and upcast only on the sliced chunk.
        if args.rollout_temperature != 1.0:
            logits_chunk = logits_chunk / args.rollout_temperature
        if args.true_on_policy_mode:
            if getattr(args, "bf16", False):
                logits_chunk = logits_chunk.to(torch.bfloat16)
            elif getattr(args, "fp16", False):
                logits_chunk = logits_chunk.to(torch.float16)
        elif logits_chunk.dtype != torch.float32:
            logits_chunk = logits_chunk.float()

        seq_start += total_length

        yield logits_chunk, tokens_chunk


def _get_log_probs_from_hidden_states_chunked(
    hidden_states: torch.Tensor,
    *,
    args: Namespace,
    parallel_state: ParallelState,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    with_entropy: bool = False,
    max_seq_lens: list[int] | None = None,
    output_layer,
    output_weight,
) -> dict[str, list[torch.Tensor]]:
    """Compute log-probs from hidden states with a selectable chunked implementation.

    Controlled by ``MILES_CHUNKED_LOGPROBS_IMPL``:
    - ``modular``: current operator module in ``chunked_cross_entropy.py``.
    - ``inline``: pre-refactor inline chunk loop kept for A/B memory tests.

    The modular path delegates to :mod:`chunked_cross_entropy`, which keeps
    peak logits memory at ``O(chunk_size * V)`` with standard autograd.
    """
    impl = os.environ.get("MILES_CHUNKED_LOGPROBS_IMPL", "modular").strip().lower()
    if impl == "inline":
        return _get_log_probs_from_hidden_states_chunked_inline(
            hidden_states,
            args=args,
            parallel_state=parallel_state,
            unconcat_tokens=unconcat_tokens,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            with_entropy=with_entropy,
            max_seq_lens=max_seq_lens,
            output_layer=output_layer,
            output_weight=output_weight,
        )
    if impl != "modular":
        raise ValueError(
            f"Unknown MILES_CHUNKED_LOGPROBS_IMPL={impl!r}; expected 'modular' or 'inline'."
        )

    from .chunked_cross_entropy import chunked_log_probs_from_hidden_states

    return chunked_log_probs_from_hidden_states(
        hidden_states,
        args=args,
        parallel_state=parallel_state,
        unconcat_tokens=unconcat_tokens,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=with_entropy,
        max_seq_lens=max_seq_lens,
        output_layer=output_layer,
        output_weight=output_weight,
    )


def _get_log_probs_from_hidden_states_chunked_inline(
    hidden_states: torch.Tensor,
    *,
    args: Namespace,
    parallel_state: ParallelState,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    with_entropy: bool = False,
    max_seq_lens: list[int] | None = None,
    output_layer,
    output_weight,
) -> dict[str, list[torch.Tensor]]:
    """Pre-refactor inline chunk loop kept for chunk-operator A/B experiments.

    This reproduces the pre-``chunked_cross_entropy.py`` implementation so we
    can compare:
    1. inline chunk loop in ``loss.py``
    2. modular chunk operator in ``chunked_cross_entropy.py``
    3. no chunk path (``log_probs_chunk_size <= 0``)
    """
    qkv_format = args.qkv_format
    chunk_size = args.log_probs_chunk_size
    tp_size = parallel_state.tp.size
    cp_size = parallel_state.cp.size
    sp_enabled = getattr(args, "sequence_parallel", False) and tp_size > 1

    if qkv_format == "thd":
        hs = hidden_states.squeeze(1)
    else:
        hs = hidden_states.transpose(0, 1).contiguous().view(-1, hidden_states.size(-1))

    old_sequence_parallel = getattr(output_layer, "sequence_parallel", None)
    if sp_enabled:
        from megatron.core.tensor_parallel import gather_from_sequence_parallel_region

        hs = gather_from_sequence_parallel_region(hs, tensor_parallel_output_grad=True, group=parallel_state.tp.group)
        if old_sequence_parallel is not None:
            output_layer.sequence_parallel = False

    try:
        log_probs_list: list[torch.Tensor] = []
        entropy_list: list[torch.Tensor | None] = []

        end = 0
        for i, (tokens, total_length, response_length) in enumerate(
            zip(unconcat_tokens, total_lengths, response_lengths, strict=False)
        ):
            max_seq_len = max_seq_lens[i] if max_seq_lens is not None else None

            if cp_size == 1:
                if qkv_format == "bshd":
                    e = max_seq_len * i + total_length
                    s = e - response_length
                else:
                    end += total_length
                    e = end
                    s = e - response_length

                hs_response = hs[s - 1 : e - 1]
                tokens_response = tokens[-response_length:]
            else:
                cp_chunk_sz, chunks_offset, logits_offset, tokens_offset = get_logits_and_tokens_offset_with_cp(
                    total_length, response_length, parallel_state, qkv_format, max_seq_len
                )

                hs_0 = hs[end : end + cp_chunk_sz]
                hs_1 = hs[end + cp_chunk_sz : end + 2 * cp_chunk_sz]
                end += 2 * cp_chunk_sz

                hs_0 = hs_0[logits_offset[0][0] - chunks_offset[0][0] : logits_offset[0][1] - chunks_offset[0][0]]
                tokens_0 = tokens[tokens_offset[0][0] : tokens_offset[0][1]]

                hs_1 = hs_1[logits_offset[1][0] - chunks_offset[1][0] : logits_offset[1][1] - chunks_offset[1][0]]
                tokens_1 = tokens[tokens_offset[1][0] : tokens_offset[1][1]]

                assert hs_0.size(0) == tokens_0.size(0), f"CP chunk 0: hs {hs_0.size(0)} vs tokens {tokens_0.size(0)}"
                assert hs_1.size(0) == tokens_1.size(0), f"CP chunk 1: hs {hs_1.size(0)} vs tokens {tokens_1.size(0)}"

                hs_response = torch.cat([hs_0, hs_1], dim=0)
                tokens_response = torch.cat([tokens_0, tokens_1], dim=0)

            if hs_response.size(0) == 0:
                log_probs_list.append(hs_response.new_zeros((0,)))
                entropy_list.append(hs_response.new_zeros((0,)) if with_entropy else None)
                continue

            lp_parts: list[torch.Tensor] = []
            ent_parts: list[torch.Tensor] = []

            for c_start in range(0, hs_response.size(0), chunk_size):
                c_end = min(c_start + chunk_size, hs_response.size(0))
                hs_chunk = hs_response[c_start:c_end]
                tk_chunk = tokens_response[c_start:c_end]

                if output_weight is not None:
                    hs_chunk = hs_chunk.to(dtype=output_weight.dtype, copy=False)
                chunk_logits, _ = output_layer(hs_chunk, weight=output_weight)

                if args.rollout_temperature != 1.0:
                    chunk_logits = chunk_logits / args.rollout_temperature
                if chunk_logits.dtype != torch.float32:
                    chunk_logits = chunk_logits.float()

                if tp_size == 1:
                    log_prob = _logprobs_from_logits(chunk_logits, tk_chunk)
                    if with_entropy:
                        with torch.no_grad():
                            entropy = _entropy_from_logits_chunked(chunk_logits, chunk_size=chunk_size)
                    else:
                        entropy = None
                else:
                    log_prob, entropy = calculate_log_probs_and_entropy(
                        chunk_logits,
                        tk_chunk,
                        parallel_state.tp.group,
                        with_entropy=with_entropy,
                        chunk_size=-1,
                        true_on_policy=getattr(args, "true_on_policy_mode", False),
                        vocab_size=getattr(args, "vocab_size", None),
                    )
                lp_parts.append(log_prob)
                if entropy is not None:
                    ent_parts.append(entropy)
                if not torch.is_grad_enabled():
                    del chunk_logits

            log_probs_list.append(_squeeze_trailing_singleton(torch.cat(lp_parts, dim=0)))
            entropy_list.append(torch.cat(ent_parts, dim=0) if ent_parts else None)
    finally:
        if sp_enabled and old_sequence_parallel is not None:
            output_layer.sequence_parallel = old_sequence_parallel

    res: dict[str, list] = {"log_probs": log_probs_list}
    if with_entropy:
        res["entropy"] = entropy_list
    return res


def get_log_probs_and_entropy(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    with_entropy: bool = False,
    non_loss_data: bool = True,
    max_seq_lens: list[int] | None = None,
    _chunked_logits_params: dict | None = None,
) -> dict[str, list[torch.Tensor]]:
    """Compute per-token log-probabilities (and optionally entropy) on responses.

    For each sample, extracts response-aligned logits and tokens, then computes
    log-probabilities via softmax across the tensor-parallel group. Log-probs
    are squeezed from `[R, 1]` to `[R]`. Entropy values are always appended
    (even when `with_entropy=False`), but only included in the result dict
    when requested.

    Args:
        logits: Policy logits with shape `[1, T, V]`.
        args: Configuration (temperature applied in `get_responses`).
        unconcat_tokens: List of token tensors per sample.
        total_lengths: Total sequence lengths per sample.
        response_lengths: Response segment lengths per sample.
        with_entropy: If True, include "entropy" key in result.
        non_loss_data: Unused; kept for API compatibility.

    Returns:
        Dict with key "log_probs" mapping to a list of `[R]` tensors per
        sample. If `with_entropy` is True, also includes "entropy" key with
        a list of `[R]` tensors.
    """
    parallel_state = get_parallel_state()
    assert non_loss_data

    if _chunked_logits_params is not None:
        return _get_log_probs_from_hidden_states_chunked(
            logits,
            args=args,
            parallel_state=parallel_state,
            unconcat_tokens=unconcat_tokens,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            with_entropy=with_entropy,
            max_seq_lens=max_seq_lens,
            **_chunked_logits_params,
        )

    tp_size = parallel_state.tp.size
    log_probs_list = []
    entropy_list = []
    for logits_chunk, tokens_chunk in get_responses(
        logits,
        args=args,
        unconcat_tokens=unconcat_tokens,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        max_seq_lens=max_seq_lens,
    ):
        if tp_size == 1 and not getattr(args, "true_on_policy_mode", False):
            if logits_chunk.dtype != torch.float32:
                logits_chunk = logits_chunk.float()
            chunk_size = args.log_probs_chunk_size
            if logits_chunk.size(0) == 0:
                log_prob = logits_chunk.new_zeros((0,))
                entropy = logits_chunk.new_zeros((0,)) if with_entropy else None
            elif chunk_size > 0:
                lp_parts: list[torch.Tensor] = []
                ent_parts: list[torch.Tensor] = []
                for c_start in range(0, logits_chunk.size(0), chunk_size):
                    c_end = min(c_start + chunk_size, logits_chunk.size(0))
                    lc = logits_chunk[c_start:c_end]
                    tc = tokens_chunk[c_start:c_end]
                    lp_parts.append(_logprobs_from_logits(lc, tc))
                    if with_entropy:
                        with torch.no_grad():
                            ent_parts.append(_entropy_from_logits(lc))
                log_prob = torch.cat(lp_parts, dim=0)
                entropy = torch.cat(ent_parts, dim=0) if with_entropy else None
            else:
                log_prob = _logprobs_from_logits(logits_chunk, tokens_chunk)
                if with_entropy:
                    with torch.no_grad():
                        entropy = _entropy_from_logits_chunked(logits_chunk, chunk_size=2048)
                else:
                    entropy = None
        else:
            log_prob, entropy = calculate_log_probs_and_entropy(
                logits_chunk,
                tokens_chunk,
                    parallel_state.tp.group,
                with_entropy=with_entropy,
                chunk_size=args.log_probs_chunk_size,
                true_on_policy=getattr(args, "true_on_policy_mode", False),
                    vocab_size=getattr(args, "vocab_size", None),
            )

        log_probs_list.append(_squeeze_trailing_singleton(log_prob))
        entropy_list.append(entropy)

    res = {
        "log_probs": log_probs_list,
    }
    if with_entropy:
        res["entropy"] = entropy_list

    # we need to turn the all gather kv into zigzag ring attn kv
    if args.allgather_cp:
        _allgather_cp_redistribute(
            res,
            logits=logits,
            args=args,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            max_seq_lens=max_seq_lens,
        )

    return res


def get_values(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    with_entropy: bool = False,
    non_loss_data: bool = True,
    max_seq_lens: list[int] | None = None,
) -> dict[str, list[torch.Tensor]]:
    """Extract per-token value predictions over response tokens.

    For each sample, extracts response-aligned chunks from the value head
    output and squeezes the final dimension from `[R, 1]` to `[R]`.

    Args:
        logits: Value head output with shape `[1, T, 1]`.
        args: Configuration (passed to `get_responses` which uses
            `rollout_temperature` even though values don't need temperature).
        unconcat_tokens: List of token tensors per sample.
        total_lengths: Total sequence lengths per sample.
        response_lengths: Response segment lengths per sample.
        with_entropy: Unused; kept for signature compatibility.
        non_loss_data: Unused; kept for signature compatibility.

    Returns:
        Dict with key "values" mapping to a list of `[R]` value tensors
        per sample.
    """
    value_list = []
    for logits_chunk, _ in get_responses(
        logits,
        args=args,
        unconcat_tokens=unconcat_tokens,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        max_seq_lens=max_seq_lens,
    ):
        assert logits_chunk.size(-1) == 1, f"{logits_chunk.shape}"
        value_list.append(logits_chunk.squeeze(-1))

    res = {
        "values": value_list,
    }

    if args.allgather_cp:
        _allgather_cp_redistribute(
            res,
            logits=logits,
            args=args,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            max_seq_lens=max_seq_lens,
        )

    return res


def compute_advantages_and_returns(args: Namespace, rollout_data: RolloutBatch) -> None:
    """Compute advantages and returns in-place based on `args.advantage_estimator`.

    This function extracts rewards, log-probs, values, and masks from
    `rollout_data`, computes KL divergences, then applies the chosen advantage
    estimator. Supported methods: "grpo", "gspo", "ppo", "reinforce_plus_plus",
    and "reinforce_plus_plus_baseline". When `args.normalize_advantages` is
    True, advantages are whitened across the data-parallel group using masked
    statistics.

    Early returns if both `log_probs` and `values` are None (intermediate
    pipeline stages).

    Args:
        args: Configuration specifying estimator type, KL coefficient,
            normalization settings, and other hyperparameters.
        rollout_data: Dict containing input lists ("log_probs", "ref_log_probs",
            "rewards", "values", "response_lengths", "loss_masks",
            "total_lengths"). Modified in-place to add "advantages" and
            "returns" keys, each mapping to lists of tensors per sample.
    """
    parallel_state = get_parallel_state()
    log_probs: list[torch.Tensor] = rollout_data.get("rollout_log_probs" if args.use_rollout_logprobs else "log_probs")
    ref_log_probs: list[torch.Tensor] = rollout_data.get("ref_log_probs")
    rewards: list[float] = rollout_data.get("rewards")
    values: None | list[torch.Tensor] = rollout_data.get("values")
    response_lengths: list[int] = rollout_data.get("response_lengths")
    loss_masks: list[torch.Tensor] = rollout_data.get("loss_masks")
    total_lengths: list[int] = rollout_data.get("total_lengths")
    max_seq_lens: list[int] | None = rollout_data.get("max_seq_lens", None)

    # return when not the last pp stage.
    if log_probs is None and values is None:
        return

    if args.kl_coef == 0 or not log_probs:
        # when kl_coef is 0, we won't compute ref_log_prob
        xs = log_probs if log_probs is not None else values
        kl = [torch.zeros_like(x, dtype=torch.float32, device=x.device) for x in xs]
    else:
        kl = [
            compute_approx_kl(
                log_probs[i],
                ref_log_probs[i],
                kl_loss_type=args.kl_loss_type,
            )
            for i in range(len(log_probs))
        ]

    if args.advantage_estimator in ["grpo", "gspo"]:
        rewards = torch.tensor(rewards, dtype=torch.float32, device=kl[0].device)
        returns = get_grpo_returns(rewards, kl)
        # TODO: is the copy necessary?
        advantages = [r for r in returns]

    elif args.advantage_estimator == "ppo":
        old_rewards = rewards
        rewards = []
        kl_coef = -args.kl_coef
        cp_rank = parallel_state.cp.rank
        for reward, k in zip(old_rewards, kl, strict=False):
            k *= kl_coef
            # 空响应样本 (response_length=0, 如立即 EOS) 的 k 为空张量, 跳过终点
            # 奖励注入即可: 其 advantage/returns 均为空, 对梯度零贡献, 不会 crash。
            if cp_rank == 0 and k.numel() > 0:
                k[-1] += reward
            rewards.append(k)
        _diag_logger.info(
            "[adv-diag] rank=%s enter GAE B=%s max_resp=%s",
            dist.get_rank(), len(response_lengths), max(response_lengths) if response_lengths else 0,
        )
        # --olp-analytic-inject (臂#14, cleancritic 2.0): OLP 不在 reward 里 (rollout 侧已
        # 跳过施加), 这里按 response_length 逐样本重算 P_i——compute_overlong_penalty 是
        # (args, L) 的确定性函数, 与 rollout 侧口径完全一致——交给 get_advantages_and_
        # returns_batch 在 GAE 之后的全长坐标视图上注入 advantage; returns 不动。注入发生
        # 在下方 normalize_advantages 白化之前: 注入项与 GAE advantage 一起白化, 与 OLP 走
        # reward 通道时(基线臂)受到的处理一致, 跨臂可比。顺带把注入统计写进 rollout_data
        # (per-sample float 列表), 由 log_rollout_data 自动以 rollout/olp_inject_penalized_ratio
        # (P≠0 样本占比) 与 rollout/olp_inject_mean_penalty (P 均值) 上报——与旧口径
        # rollout/overlong/penalized_ratio 同一 rollout/step 轴, 便于对齐观察。critic 进程
        # 也会走到这里: 其 advantage 无消费方 (value loss 只用 returns), 注入是惰性无害的;
        # 统计键在 critic 进程不会被 log_rollout_data 上报。
        olp_inject_penalties = None
        if getattr(args, "olp_analytic_inject", False):
            olp_inject_penalties = [compute_overlong_penalty(args, length) for length in response_lengths]
            rollout_data["olp_inject_penalized_ratio"] = [float(p != 0.0) for p in olp_inject_penalties]
            rollout_data["olp_inject_mean_penalty"] = [float(p) for p in olp_inject_penalties]
        # --group-center-inject (臂#16): actor 的 advantage 通道做组内留一中心化——终端标量
        # a_j = raw_reward_j − V_j (该样本最后一个 loss-mask token 的 critic value), 注入量
        # P_i = −(组内其余成员 a_j 之和)/(n_g−1), 在 GAE 之后按 vapo λ_i 衰减注入 (数学上
        # 等价于终端 reward 减去组内留一基线后重跑 GAE 的 advantage); returns 不动, critic
        # 照学未中心化 return。a_j 用 raw_reward (纯任务奖励, 不含 OLP/length shaping)——
        # 与 --olp-analytic-inject 可共存 (两个线性注入对 full_advantages 相加), 对基线
        # OLP-in-reward 臂语义也一致。rollout_data["raw_reward"] 不按 partition 切分, 每个
        # rank 持有全局 flat 序完整列表; 本地样本的 flat 位置由 process_rollout_data 保留的
        # group_center_positions 提供。组内求和的 DP allreduce 在 get_advantages_and_
        # returns_batch 内无条件执行 (集合通信不变量见彼处注释, 仿下方 normalize_advantages
        # 的 DP-allreduce 写法)。注入项与 GAE advantage 一起被白化, 跨臂可比。统计经
        # log_rollout_data 以 rollout/group_center_correction (P 均值) 与 rollout/
        # group_center_abs (|P| 均值) 上报。critic 进程照走: advantage 无消费方, 注入惰性
        # 无害, allreduce 在 critic world 的 intra_dp 组内自我对称; 统计键不被 critic 上报。
        group_center_ctx = None
        group_center_stats: dict = {}
        if getattr(args, "group_center_inject", False):
            raw_full = rollout_data.get("raw_reward")
            positions = rollout_data.get("group_center_positions")
            assert raw_full is not None and positions is not None, (
                "--group-center-inject: rollout_data is missing raw_reward/group_center_positions "
                "(custom convert_samples_to_train_data or a debug data path that bypasses "
                "process_rollout_data?)"
            )
            assert all(isinstance(r, (int, float)) for r in raw_full), (
                "--group-center-inject requires scalar raw rewards (got non-scalar entries)"
            )
            group_center_ctx = dict(
                positions=positions,
                raw_rewards=raw_full,
                n_samples_per_prompt=args.n_samples_per_prompt,
                alpha=args.vapo_lambda_alpha,
                loss_masks=loss_masks,
                dp_group=parallel_state.intra_dp.group,
                stats_out=group_center_stats,
            )
        advantages, returns = get_advantages_and_returns_batch(
            total_lengths,
            response_lengths,
            values,
            rewards,
            args.gamma,
            args.lambd,
            length_adaptive_lambda_alpha=args.vapo_lambda_alpha,
            olp_inject_penalties=olp_inject_penalties,
            olp_inject_alpha=getattr(args, "olp_inject_alpha", 0.1),
            group_center=group_center_ctx,
        )
        if group_center_ctx is not None:
            rollout_data["group_center_correction"] = group_center_stats["correction"]
            rollout_data["group_center_abs"] = group_center_stats["abs"]
        _diag_logger.info("[adv-diag] rank=%s GAE done", dist.get_rank())

    elif args.advantage_estimator == "reinforce_plus_plus":
        rewards = torch.tensor(rewards, dtype=torch.float32, device=kl[0].device)
        returns = get_reinforce_plus_plus_returns(
            rewards=rewards,
            kl=kl,
            loss_masks=loss_masks,
            response_lengths=response_lengths,
            total_lengths=total_lengths,
            kl_coef=args.kl_coef,
            gamma=args.gamma,
        )
        advantages = [r for r in returns]

    elif args.advantage_estimator == "reinforce_plus_plus_baseline":
        rewards = torch.tensor(rewards, dtype=torch.float32, device=kl[0].device)
        advantages = get_reinforce_plus_plus_baseline_advantages(
            rewards=rewards,
            kl=kl,
            loss_masks=loss_masks,
            kl_coef=args.kl_coef,
        )
        returns = advantages

    elif args.advantage_estimator == "on_policy_distillation":
        student_log_probs = log_probs
        teacher_log_probs = rollout_data.get("teacher_log_probs")
        response_lengths = rollout_data.get("response_lengths")
        device = student_log_probs[0].device
        teacher_log_probs = [t_log_prob.to(device=device) for t_log_prob in teacher_log_probs]
        teacher_log_probs = [
            t_log_prob[-response_length:]
            for t_log_prob, response_length in zip(teacher_log_probs, response_lengths, strict=False)
        ]
        advantages = [
            teacher_log_prob - student_log_prob
            for teacher_log_prob, student_log_prob in zip(teacher_log_probs, student_log_probs, strict=False)
        ]
        returns = advantages

    else:
        raise NotImplementedError(f"advantage_estimator {args.advantage_estimator} is not supported. ")

    # TODO: OpenRLHF always does advantages normalization but veRL doesn't seem to do it.
    if args.normalize_advantages:
        all_advs = torch.cat(advantages)
        cp_size = parallel_state.cp.size
        if cp_size == 1:
            all_masks = torch.cat(loss_masks)
        else:
            mask_chunks = []
            for i in range(len(advantages)):
                total_len = total_lengths[i]
                response_len = response_lengths[i]
                prompt_len = total_len - response_len
                max_seq_len = max_seq_lens[i] if max_seq_lens is not None else None

                _, _, _, token_offsets = get_logits_and_tokens_offset_with_cp(
                    total_len, response_len, args.qkv_format, max_seq_len
                )

                # Convert global offsets to response-space offsets
                s0, e0 = token_offsets[0]
                s1, e1 = token_offsets[1]
                res_s0, res_e0 = max(0, s0 - prompt_len), max(0, e0 - prompt_len)
                res_s1, res_e1 = max(0, s1 - prompt_len), max(0, e1 - prompt_len)

                local_mask_parts = []
                full_mask = loss_masks[i]
                if res_e0 > res_s0:
                    local_mask_parts.append(full_mask[res_s0:res_e0])
                if res_e1 > res_s1:
                    local_mask_parts.append(full_mask[res_s1:res_e1])

                # Concatenate the parts to form the final mask chunk for this rank and this sequence
                local_mask_chunk = (
                    torch.cat(local_mask_parts)
                    if local_mask_parts
                    else torch.tensor([], device=all_advs.device, dtype=full_mask.dtype)
                )
                mask_chunks.append(local_mask_chunk)

            all_masks = torch.cat(mask_chunks)

        # 集合通信不变量：distributed_masked_whiten 含 DP-group allreduce，所有 rank
        # 必须无条件参与。CP 切片下本地分片可能为空，空 tensor 参与是安全的
        # （贡献 [0,0,0] 统计量）；若按本地是否为空跳过，会死锁整个 DP group。
        assert (
            all_advs.size() == all_masks.size()
        ), f"Shape mismatch before whitening: advantages {all_advs.size()}, masks {all_masks.size()}"
        dp_group = parallel_state.intra_dp.group

        _diag_logger.info(
            "[adv-diag] rank=%s enter whiten adv_numel=%s mask_numel=%s",
            dist.get_rank(), all_advs.numel(), all_masks.numel(),
        )
        whitened_advs_flat = distributed_masked_whiten(
            all_advs,
            all_masks,
            process_group=dp_group,
            shift_mean=True,
        )
        _diag_logger.info("[adv-diag] rank=%s whiten done", dist.get_rank())
        chunk_lengths = [chunk.size(0) for chunk in advantages]
        advantages = list(torch.split(whitened_advs_flat, chunk_lengths))

    rollout_data["advantages"] = advantages
    rollout_data["returns"] = returns


def vanilla_tis_function(
    args,
    *,
    pg_loss: torch.Tensor,
    train_log_probs: list[torch.Tensor],
    rollout_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    **kwargs: Any,
) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]:
    rollout_log_probs = torch.cat(rollout_log_probs, dim=0)
    old_log_probs = torch.cat(train_log_probs, dim=0)
    tis = torch.exp(old_log_probs - rollout_log_probs)
    tis_abs = (torch.exp(old_log_probs - rollout_log_probs) - 1).abs()
    tis_weights = torch.clamp(tis, min=args.tis_clip_low, max=args.tis_clip)
    tis_clipfrac = (tis_weights != tis).float()
    metrics = {
        "tis": tis.clone().detach(),
        "tis_clipfrac": tis_clipfrac.clone().detach(),
        "tis_abs": tis_abs.clone().detach(),
    }
    pg_loss = pg_loss * tis_weights
    return pg_loss, loss_masks, metrics


def compute_ess_ratio_contribution(
    ppo_kl: torch.Tensor,
    loss_masks: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    qkv_format: str,
    max_seq_lens: list[int] | None,
    calculate_per_token_loss: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return an ESS contribution compatible with ``aggregate_train_losses``.

    ESS needs full-sample sums before applying the nonlinear ratio.  Under CP we
    reconstruct those sums first, then let only CP rank 0 emit the final
    contribution so the generic CP metric aggregation does not double count it.
    """
    parallel_state = get_parallel_state()
    cp_size = parallel_state.cp.size

    local_masks = slice_loss_masks_for_local_cp(
        loss_masks,
        total_lengths,
        response_lengths,
        qkv_format,
        max_seq_lens,
    )
    local_lengths = [mask.size(0) for mask in local_masks]
    is_weights_per_sample = (-ppo_kl.detach().float()).exp().split(local_lengths, dim=0)

    partial_sums = torch.zeros(len(loss_masks), 2, device=ppo_kl.device, dtype=torch.float32)
    for i, (weights, mask) in enumerate(zip(is_weights_per_sample, local_masks, strict=False)):
        if weights.numel() != mask.numel():
            raise ValueError(f"ESS weight/mask length mismatch for sample {i}: {weights.numel()} vs {mask.numel()}")
        masked_weights = weights * mask.to(device=weights.device, dtype=weights.dtype)
        partial_sums[i, 0] = masked_weights.sum()
        partial_sums[i, 1] = (masked_weights * masked_weights).sum()

    if cp_size > 1:
        dist.all_reduce(partial_sums, op=dist.ReduceOp.SUM, group=parallel_state.cp.group)

    ess_ratio_sum = torch.zeros((), device=ppo_kl.device, dtype=torch.float32)
    ess_ratio_weight = torch.zeros((), device=ppo_kl.device, dtype=torch.float32)
    for i, loss_mask in enumerate(loss_masks):
        num_valid_tokens = torch.clamp_min(loss_mask.to(device=ppo_kl.device, dtype=torch.float32).sum(), 1)
        sum_w = partial_sums[i, 0]
        sum_w2 = partial_sums[i, 1]
        ess_ratio = (sum_w * sum_w) / (num_valid_tokens * torch.clamp_min(sum_w2, 1e-8))
        ess_ratio_sum += ess_ratio * num_valid_tokens if calculate_per_token_loss else ess_ratio
        if calculate_per_token_loss:
            ess_ratio_weight += num_valid_tokens

    if cp_size > 1 and parallel_state.cp.rank != 0:
        ess_ratio_sum = ess_ratio_sum * 0
        ess_ratio_weight = ess_ratio_weight * 0

    return ess_ratio_sum, ess_ratio_weight if calculate_per_token_loss else None


def icepop_function(
    args,
    *,
    pg_loss: torch.Tensor,
    train_log_probs: list[torch.Tensor],
    rollout_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    **kwargs: Any,
) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]:
    rollout_log_probs = torch.cat(rollout_log_probs, dim=0)
    old_log_probs = torch.cat(train_log_probs, dim=0)
    ice_ratio = torch.exp(old_log_probs - rollout_log_probs)
    ice_abs = (torch.exp(old_log_probs - rollout_log_probs) - 1).abs()
    ice_weight = torch.where(
        (ice_ratio >= args.tis_clip_low) & (ice_ratio <= args.tis_clip), ice_ratio, torch.zeros_like(ice_ratio)
    )
    ice_clipfrac = (ice_weight != ice_ratio).float()
    metrics = {
        "tis": ice_ratio.clone().detach(),
        "tis_clipfrac": ice_clipfrac.clone().detach(),
        "tis_abs": ice_abs.clone().detach(),
    }
    pg_loss = pg_loss * ice_weight
    return pg_loss, loss_masks, metrics


def policy_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute policy loss (PPO/GSPO) and metrics.

    Computes current log-probabilities and entropy from model logits, then
    calculates PPO-style clipped policy gradient loss. For GSPO, gathers
    full sequences via context-parallel all-gather before computing per-sample
    KL. Optionally applies TIS (Truncated Importance Sampling) correction and
    adds KL loss term if configured.

    Args:
        args: Configuration controlling advantage estimator, clipping thresholds,
            entropy/KL coefficients, and TIS settings.
        batch: Mini-batch containing "advantages", "log_probs" (old policy),
            "unconcat_tokens", "response_lengths", "total_lengths", "loss_masks",
            and optionally "ref_log_probs" and "rollout_log_probs".
        logits: Policy logits with shape `[1, T, V]`.
        sum_of_sample_mean: Reduction function that averages per-sample values.

    Returns:
        Tuple of `(loss, metrics)` where `loss` is a scalar tensor and `metrics`
        is a dict containing detached scalars: "loss", "pg_loss",
        "entropy_loss", "pg_clipfrac", "ppo_kl". Additional keys "kl_loss",
        "tis", "ois", "tis_clipfrac" are included when the respective features
        are enabled.
    """
    parallel_state = get_parallel_state()
    advantages = torch.cat(batch["advantages"], dim=0)
    old_log_probs = batch["rollout_log_probs"] if args.use_rollout_logprobs else batch["log_probs"]

    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]
    max_seq_lens = batch.get("max_seq_lens", None)

    _need_entropy = args.entropy_coef != 0
    _chunked_logits_params = batch.get("_chunked_logits_params")
    log_probs_and_entropy = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=_need_entropy,
        max_seq_lens=max_seq_lens,
        _chunked_logits_params=_chunked_logits_params,
    )

    log_probs = log_probs_and_entropy["log_probs"]
    train_log_probs_list = log_probs
    old_log_probs_list = old_log_probs

    # Pre-gather log probs if needed by OPSM or GSPO to avoid duplicate gathering
    need_full_log_probs = args.use_opsm or args.advantage_estimator == "gspo"

    full_log_probs = None
    full_old_log_probs = None
    if need_full_log_probs:
        full_log_probs = [
            all_gather_with_cp(log_prob, total_length, response_length)
            for log_prob, total_length, response_length in zip(
                log_probs, total_lengths, response_lengths, strict=False
            )
        ]
        full_old_log_probs = [
            all_gather_with_cp(old_log_prob, total_length, response_length)
            for old_log_prob, total_length, response_length in zip(
                old_log_probs, total_lengths, response_lengths, strict=False
            )
        ]

    # Compute OPSM mask if enabled
    if args.use_opsm:
        opsm_mask, opsm_clipfrac = compute_opsm_mask(
            args=args,
            full_log_probs=full_log_probs,
            full_old_log_probs=full_old_log_probs,
            advantages=batch["advantages"],
            loss_masks=batch["loss_masks"],
        )

    # Compute KL divergence (GSPO uses sequence-level KL, others use per-token KL)
    if args.advantage_estimator == "gspo":
        ppo_kl = compute_gspo_kl(
            full_log_probs=full_log_probs,
            full_old_log_probs=full_old_log_probs,
            local_log_probs=log_probs,
            loss_masks=batch["loss_masks"],
        )
        old_log_probs = torch.cat(old_log_probs, dim=0)
        log_probs = torch.cat(log_probs, dim=0)
    else:
        old_log_probs = torch.cat(old_log_probs, dim=0)
        log_probs = torch.cat(log_probs, dim=0)
        ppo_kl = old_log_probs - log_probs

    local_loss_mask_list = get_local_response_loss_masks(
        total_lengths,
        response_lengths,
        batch["loss_masks"],
        args.qkv_format,
        max_seq_lens,
    )
    local_loss_masks = torch.cat(local_loss_mask_list, dim=0).to(device=ppo_kl.device)
    active_tokens = local_loss_masks.bool()
    ppo_kl = torch.where(
        active_tokens,
        torch.nan_to_num(ppo_kl, nan=0.0, posinf=0.0, neginf=0.0),
        ppo_kl.new_zeros(()),
    )
    advantages = torch.where(
        active_tokens,
        torch.nan_to_num(advantages, nan=0.0, posinf=0.0, neginf=0.0),
        advantages.new_zeros(()),
    )

    pg_loss, pg_clipfrac = compute_policy_loss(ppo_kl, advantages, args.eps_clip, args.eps_clip_high)

    if getattr(args, "dump_details", None) is not None:
        from .debug_dump import maybe_dump_policy_loss_debug

        maybe_dump_policy_loss_debug(
            args=args,
            batch=batch,
            train_log_probs=train_log_probs_list,
            old_log_probs=old_log_probs_list,
            rollout_log_probs=batch.get("rollout_log_probs"),
            advantages=batch["advantages"],
            local_loss_masks=local_loss_mask_list,
            ppo_kl=ppo_kl,
            pg_loss=pg_loss,
        )

    if args.use_opsm:
        pg_loss = pg_loss * opsm_mask

    # Apply off-policy correction using importance sampling if enabled
    if args.get_mismatch_metrics or args.use_tis:
        # NOTE:
        # `tis_func` may apply rejection-sampling style masking (RS) and return `modified_response_masks`.
        # We rebuild `sum_of_sample_mean` with those masks to correct denominators for loss/backprop.
        #
        # However, mismatch/TIS/RS metrics (e.g., "truncate_fraction") are often defined over the
        # *pre-RS* valid tokens. If we aggregate metrics with `modified_response_masks`, the rejected
        # tokens are excluded from the denominator and the metric can be artificially driven to 0.
        # Keep a copy of the original reducer (based on `batch["loss_masks"]`) for metric aggregation.
        sum_of_sample_mean_for_mismatch_metrics = sum_of_sample_mean

        assert "rollout_log_probs" in batch, "rollout_log_probs must be provided for TIS"

        ois = (-ppo_kl).exp()
        tis_kwargs = {
            "args": args,
            "pg_loss": pg_loss,
            "train_log_probs": batch["log_probs"],
            "rollout_log_probs": batch["rollout_log_probs"],
            "loss_masks": batch["loss_masks"],
            "total_lengths": total_lengths,
            "response_lengths": response_lengths,
            "parallel_state": parallel_state,
            "max_seq_lens": max_seq_lens,
        }

        if args.custom_tis_function_path is not None:
            tis_func = load_function(args.custom_tis_function_path)
        else:
            tis_func = vanilla_tis_function
        pg_loss, modified_response_masks, tis_metrics = tis_func(**tis_kwargs)

        # [decouple IS and rejection] Rebuild sum_of_sample_mean with modified_response_masks for denominator correction
        # modified_response_masks will be sliced with cp in get_sum_of_sample_mean
        sum_of_sample_mean = get_sum_of_sample_mean(
            total_lengths,
            response_lengths,
            modified_response_masks,
            args.calculate_per_token_loss,
            args.qkv_format,
            max_seq_lens,
        )

    # Determine pg_loss reducer: use custom if specified, otherwise default
    if getattr(args, "custom_pg_loss_reducer_function_path", None) is not None:
        custom_pg_loss_reducer_func = load_function(args.custom_pg_loss_reducer_function_path)
        # Determine which loss_masks to use for pg_loss reducer
        pg_loss_masks = modified_response_masks if (args.get_mismatch_metrics or args.use_tis) else batch["loss_masks"]
        pg_loss_reducer = custom_pg_loss_reducer_func(
            total_lengths, response_lengths, pg_loss_masks, args.calculate_per_token_loss
        )
    else:
        pg_loss_reducer = sum_of_sample_mean

    # ESS (Effective Sample Size) ratio from per-token IS weights
    # w = π_new/π_old = exp(-ppo_kl).  A value of 1.0 is on-policy; near 0
    # means the per-token weights are highly concentrated.
    ess_ratio_sum, ess_ratio_weight = compute_ess_ratio_contribution(
        ppo_kl=ppo_kl,
        loss_masks=batch["loss_masks"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        qkv_format=args.qkv_format,
        max_seq_lens=max_seq_lens,
        calculate_per_token_loss=args.calculate_per_token_loss,
    )

    pg_loss = pg_loss_reducer(pg_loss)
    pg_clipfrac = sum_of_sample_mean(pg_clipfrac)
    ppo_kl = sum_of_sample_mean(ppo_kl)

    # entropy loss (skip expensive computation when entropy_coef == 0)
    if _need_entropy:
        entropy = log_probs_and_entropy["entropy"]
        entropy = torch.cat(entropy, dim=0)
        entropy_loss = sum_of_sample_mean(entropy)
    else:
        entropy_loss = pg_loss.new_zeros(())

    loss = pg_loss - args.entropy_coef * entropy_loss

    positive_lm_loss = None
    positive_lm_weight_mean = None
    if getattr(args, "positive_lm_loss_coef", 0) > 0:
        # VAPO positive-example LM loss: 对答对样本 (returns ≡ R > 0.5) 的 token 加 NLL。
        # 逐样本权重由 rollout 侧生成(difficulty_weight 开启时为 1 - group_pass_rate);
        # 权重缺失回退 w ≡ 0(fail-safe 关闭, 防止无权重数据静默退回全量自模仿)。
        returns_flat = torch.cat(batch["returns"], dim=0)
        positive_tokens = active_tokens & (returns_flat > 0.5)
        nll = torch.where(positive_tokens, -log_probs, log_probs.new_zeros(()))
        sample_weights = batch.get("positive_lm_weight")
        if sample_weights is not None:
            assert len(sample_weights) == len(batch["returns"]), (
                f"positive_lm_weight count {len(sample_weights)} != samples {len(batch['returns'])}"
            )
            weight_flat = torch.cat(
                [
                    log_probs.new_full((ret.numel(),), float(w))
                    for w, ret in zip(sample_weights, batch["returns"], strict=True)
                ]
            )
            nll = nll * weight_flat
            # 以 _weighted/(分子,分母) 形式上报得到真实样本均值; 直接报标量会在
            # per-token-loss 模式下被按 token 数归一化, 观测值缩小 ~3 万倍
            positive_lm_weight_mean = torch.stack(
                [
                    log_probs.new_tensor(float(sum(sample_weights))),
                    log_probs.new_tensor(float(len(sample_weights))),
                ]
            )
        else:
            nll = nll * 0.0
        positive_lm_loss = sum_of_sample_mean(nll)
        loss = loss + args.positive_lm_loss_coef * positive_lm_loss

    if args.use_kl_loss:
        ref_log_probs = batch["ref_log_probs"]
        ref_log_probs = torch.cat(ref_log_probs, dim=0)
        importance_ratio = None
        if args.use_unbiased_kl:
            importance_ratio = torch.exp(log_probs - old_log_probs)
        kl = compute_approx_kl(
            log_probs,
            ref_log_probs,
            kl_loss_type=args.kl_loss_type,
            importance_ratio=importance_ratio,
        )
        kl = torch.where(
            active_tokens,
            torch.nan_to_num(kl, nan=0.0, posinf=0.0, neginf=0.0),
            kl.new_zeros(()),
        )
        kl_loss = sum_of_sample_mean(kl)

        if args.kl_loss_coef != 0:
            loss = loss + args.kl_loss_coef * kl_loss

    # make sure the gradient could backprop correctly.
    if log_probs.numel() == 0:
        loss += 0 * logits.sum()

    disable_logdiff_metrics = os.environ.get("MILES_DISABLE_LOGDIFF_METRICS", "0").lower() in {"1", "true", "yes"}

    train_scored_log_probs = log_probs
    train_rollout_logprob_abs_diff = None
    train_rollout_kl = None
    if not disable_logdiff_metrics and "rollout_log_probs" in batch and batch["rollout_log_probs"]:
        rollout_log_probs = torch.cat(batch["rollout_log_probs"], dim=0)

        if getattr(args, "icepop_dump_dir", None) is not None or getattr(args, "dump_details", None) is not None:
            from miles.utils.debug_utils.icepop_dump import record_icepop_scatter

            # Use ``old_log_probs`` (== batch["log_probs"], the same tensor IcePop/TIS
            # feeds into its ratio) so the offline keep/mask decision matches training.
            record_icepop_scatter(
                args,
                train_log_probs=old_log_probs,
                rollout_log_probs=rollout_log_probs,
                active_tokens=active_tokens,
            )

        abs_diff_raw = (train_scored_log_probs - rollout_log_probs).abs()
        abs_diff = torch.where(
            active_tokens,
            torch.nan_to_num(abs_diff_raw, nan=0.0, posinf=0.0, neginf=0.0),
            abs_diff_raw.new_zeros(()),
        )
        train_rollout_logprob_abs_diff = sum_of_sample_mean(abs_diff)

        # KL(rollout || train) at sampled tokens via Schulman k3 with per-token clamp [-10, 10]
        rollout_train_kl = compute_approx_kl(rollout_log_probs, train_scored_log_probs, kl_loss_type="low_var_kl")
        rollout_train_kl = torch.where(
            active_tokens,
            torch.nan_to_num(rollout_train_kl, nan=0.0, posinf=0.0, neginf=0.0),
            rollout_train_kl.new_zeros(()),
        )
        train_rollout_kl = sum_of_sample_mean(rollout_train_kl)

    reported_loss = {
        "loss": loss.clone().detach(),
        "pg_loss": pg_loss.clone().detach(),
        "entropy_loss": entropy_loss.clone().detach(),
        "pg_clipfrac": pg_clipfrac.clone().detach(),
        "ppo_kl": ppo_kl.clone().detach(),
    }
    if positive_lm_loss is not None:
        reported_loss["positive_lm_loss"] = positive_lm_loss.clone().detach()
    if positive_lm_weight_mean is not None:
        reported_loss["_weighted/positive_lm_weight_mean"] = positive_lm_weight_mean.clone().detach()
    if ess_ratio_weight is None:
        reported_loss["ess_ratio"] = ess_ratio_sum.squeeze()
    else:
        reported_loss["_weighted/ess_ratio"] = torch.stack([ess_ratio_sum.squeeze(), ess_ratio_weight.squeeze()])

    if train_rollout_logprob_abs_diff is not None:
        reported_loss["train_rollout_logprob_abs_diff"] = train_rollout_logprob_abs_diff.clone().detach()
    if train_rollout_kl is not None:
        reported_loss["train_rollout_kl"] = train_rollout_kl.clone().detach()

        with torch.no_grad():
            metric_diff = torch.where(
                active_tokens,
                torch.nan_to_num(abs_diff_raw, nan=0.0, posinf=0.0, neginf=0.0),
                abs_diff_raw.new_zeros(()),
            )
            flat_diff = metric_diff[active_tokens].detach().float()

            if flat_diff.numel() > 0:
                if args.calculate_per_token_loss:
                    reported_loss["_logdiff_histogram"] = _make_logdiff_histogram(flat_diff)
                    reported_loss["_logdiff_min_max"] = torch.stack([flat_diff.min(), flat_diff.max()]).to(
                        dtype=torch.float64
                    )
                else:
                    sorted_diff = flat_diff.sort().values
                    n = sorted_diff.numel()
                    for tag, q in [("p50", 0.5), ("p75", 0.75), ("p90", 0.9), ("p95", 0.95), ("p999", 0.999)]:
                        idx = min(int(q * n), n - 1)
                        reported_loss[f"logdiff_{tag}"] = sorted_diff[idx]
                    reported_loss["logdiff_min"] = sorted_diff[0]
                    reported_loss["logdiff_max"] = sorted_diff[-1]

                # Per-sample token-level max: ignore response tokens masked out of the loss
                # (for example tool-response tokens in multi-turn rollouts).
                ref_list = batch["rollout_log_probs"]
                offset = 0
                per_sample_max_list = []
                for sample_lp, sample_mask in zip(ref_list, local_loss_mask_list, strict=True):
                    seg_len = sample_lp.size(0)
                    sample_mask = sample_mask.to(device=metric_diff.device, dtype=torch.bool)
                    sample_diff = metric_diff[offset : offset + seg_len]
                    if sample_mask.any():
                        per_sample_max_list.append(sample_diff[sample_mask].max())
                    elif args.calculate_per_token_loss:
                        per_sample_max_list.append(flat_diff.new_full((), float("-inf")))
                    offset += seg_len
                if per_sample_max_list:
                    per_sample_max = torch.stack(per_sample_max_list)
                    if args.calculate_per_token_loss:
                        if parallel_state.cp.size > 1:
                            dist.all_reduce(per_sample_max, op=dist.ReduceOp.MAX, group=parallel_state.cp.group)
                        finite_sample_max = per_sample_max[torch.isfinite(per_sample_max)]
                        numerator = finite_sample_max.sum() if finite_sample_max.numel() > 0 else flat_diff.new_zeros(())
                        denominator = flat_diff.new_tensor(float(finite_sample_max.numel()))
                        if parallel_state.cp.size > 1 and parallel_state.cp.rank != 0:
                            numerator = numerator * 0
                            denominator = denominator * 0
                        reported_loss["_weighted/logdiff_token_max_per_sample"] = torch.stack(
                            [numerator.squeeze(), denominator.squeeze()]
                        )
                    else:
                        reported_loss["logdiff_token_max_per_sample"] = per_sample_max.mean()

                _accumulate_logdiff_histogram(flat_diff)

                # Extreme-value diagnostic: log details about top-K worst tokens
                if metric_diff.max().item() > 0.5:
                    _log_extreme_logdiff_diagnostic(
                        flat_diff=metric_diff.detach().float(),
                        old_log_probs=old_log_probs,
                        rollout_log_probs=rollout_log_probs,
                        batch=batch,
                        args=args,
                    )

    if args.use_kl_loss:
        reported_loss["kl_loss"] = kl_loss.clone().detach()

    if args.get_mismatch_metrics or args.use_tis:
        # Aggregate mismatch/TIS/RS related metrics with the *pre-RS* masks.
        # See comment above where `sum_of_sample_mean_for_mismatch_metrics` is defined.
        reported_loss["ois"] = sum_of_sample_mean_for_mismatch_metrics(ois).clone().detach()
        # Assume all metrics are already cloned and detached
        for metric_key, metric_value in tis_metrics.items():
            key_name = f"{metric_key}"
            reported_loss[key_name] = sum_of_sample_mean_for_mismatch_metrics(metric_value)

    if args.use_opsm:
        reported_loss["opsm_clipfrac"] = opsm_clipfrac

    return loss, reported_loss


def value_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute clipped value loss and metrics.

    Extracts current value predictions from `logits`, compares them against
    stored old values with clipping, and computes the maximum of clipped and
    unclipped squared errors (PPO-style value clipping).

    Args:
        args: Configuration containing `value_clip` threshold.
        batch: Mini-batch with "values" (old predictions), "returns",
            "unconcat_tokens", "total_lengths", and "response_lengths".
        logits: Value head output with shape `[1, T, 1]`.
        sum_of_sample_mean: Reduction function that averages per-sample values.

    Returns:
        Tuple of `(loss, metrics)` where `loss` is a scalar tensor and
        `metrics` contains detached scalars "value_loss" and "value_clipfrac".
    """
    old_values = torch.cat(batch["values"], dim=0)

    values = get_values(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
        max_seq_lens=batch.get("max_seq_lens", None),
    )
    values = torch.cat([value.flatten() for value in values["values"]], dim=0)

    returns = torch.cat(batch["returns"], dim=0)

    values_clipfrac = torch.abs(values - old_values) > args.value_clip
    values_clipped = old_values + (values - old_values).clamp(-args.value_clip, args.value_clip)
    surr1 = (values_clipped - returns) ** 2
    surr2 = (values - returns) ** 2
    loss = torch.max(surr1, surr2)

    loss = sum_of_sample_mean(loss)
    values_clipfrac = sum_of_sample_mean(values_clipfrac.float())

    # make sure the gradient could backprop correctly.
    if values.numel() == 0:
        loss += 0 * values.sum()

    reported_loss = {
        "value_loss": loss.clone().detach(),
        "value_clipfrac": values_clipfrac.clone().detach(),
    }

    return loss, reported_loss


def sft_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute supervised fine-tuning loss over response tokens.

    Computes log-probabilities of the ground-truth tokens in the response
    segments and returns the negative log-likelihood as the loss.

    Args:
        args: Configuration (passed through to helpers).
        batch: Mini-batch with "unconcat_tokens", "response_lengths", and
            "total_lengths".
        logits: Policy logits with shape `[1, T, V]`.
        sum_of_sample_mean: Reduction function that averages per-sample values.

    Returns:
        Tuple of `(loss, metrics)` where `metrics` contains a single detached
        scalar "loss".
    """
    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]

    _chunked_logits_params = batch.get("_chunked_logits_params")
    log_probs_and_entropy = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=False,
        max_seq_lens=batch.get("max_seq_lens", None),
        _chunked_logits_params=_chunked_logits_params,
    )

    log_probs = log_probs_and_entropy["log_probs"]
    log_probs = torch.cat(log_probs, dim=0)
    loss = -sum_of_sample_mean(log_probs)

    # make sure the gradient could backprop correctly.
    if log_probs.numel() == 0:
        loss += 0 * logits.sum()

    return (
        loss,
        {
            "loss": loss.clone().detach(),
        },
    )


def loss_function(
    args: Namespace,
    batch: RolloutBatch,
    num_microbatches: int,
    logits: torch.Tensor,
    apply_megatron_loss_scaling: bool = False,
) -> tuple[torch.Tensor, int | torch.Tensor, dict[str, list[str] | torch.Tensor]]:
    """Dispatch to the configured loss and rescale for Megatron integration.

    Selects one of "policy_loss", "value_loss", "sft_loss", or a custom loss
    function based on `args.loss_type`, computes the loss and metrics, then
    rescales the loss by micro-batch and parallelism factors to integrate with
    Megatron's gradient accumulation.

    Args:
        args: Configuration specifying `loss_type`, `calculate_per_token_loss`,
            `global_batch_size`, and optionally `custom_loss_function_path`.
        batch: Mini-batch with "loss_masks", "response_lengths", and other
            keys required by the selected loss function.
        num_microbatches: Number of gradient accumulation steps.
        logits: Model outputs (policy or value head).

    Returns:
        Tuple of `(scaled_loss, normalizer, logging_dict)` where:
        - `scaled_loss` is the loss tensor (scalar) rescaled for Megatron.
        - `normalizer` is `num_tokens` (scalar tensor) if
          `args.calculate_per_token_loss` is True, else `1` (int).
        - `logging_dict` has keys "keys" (list of str metric names) and
          "values" (1D tensor: [count, metric1, metric2, ...]).
    """
    parallel_state = get_parallel_state()
    num_tokens = sum([torch.clamp_min(loss_mask.sum(), 1) for loss_mask in batch["loss_masks"]])
    num_samples = len(batch["response_lengths"])

    sum_of_sample_mean = get_sum_of_sample_mean(
        batch["total_lengths"],
        batch["response_lengths"],
        batch["loss_masks"],
        args.calculate_per_token_loss,
        args.qkv_format,
        batch.get("max_seq_lens", None),
    )

    match args.loss_type:
        case "policy_loss":
            func = policy_loss_function
        case "value_loss":
            func = value_loss_function
        case "sft_loss":
            func = sft_loss_function
        case "custom_loss":
            func = load_function(args.custom_loss_function_path)
        case _:
            raise ValueError(f"Unknown loss type: {args.loss_type}")

    if args.recompute_loss_function:
        loss, log = checkpoint(
            func,
            args,
            batch,
            logits,
            sum_of_sample_mean,
            use_reentrant=True,
        )
    else:
        loss, log = func(args, batch, logits, sum_of_sample_mean)

    # With allgather-CP, some CP ranks may have no loss-contributing tokens (e.g., all
    # padding). Without this, gradient doesn't flow through their attention path, so
    # the CP gather's backward (reduce-scatter) is not called, deadlocking other CP
    # ranks that call it. Adding this zero loss forces autograd to traverse the full
    # graph on every rank without changing gradient values.
    if parallel_state.cp.size > 1 and args.allgather_cp:
        loss = loss + 0 * logits.sum()

    # Here we need to divide by cp_size because to cancel the multiply in Megatron.
    assert args.use_dynamic_global_batch_size == ("dynamic_global_batch_size" in batch)
    global_batch_size = batch.get("dynamic_global_batch_size", args.global_batch_size)
    if not args.calculate_per_token_loss:
        if apply_megatron_loss_scaling:
            loss_parallel_size = (
                parallel_state.intra_dp.size
                if args.true_on_policy_mode and parallel_state.is_ulysses_cp
                else parallel_state.intra_dp_cp.size
            )
            loss = loss * num_microbatches / global_batch_size * loss_parallel_size
        else:
            loss = loss / global_batch_size * parallel_state.intra_dp.size
    else:
        if apply_megatron_loss_scaling:
            loss = loss * parallel_state.cp.size

    weighted_metrics = {
        key.removeprefix("_weighted/"): value.detach()
        for key, value in log.items()
        if key.startswith("_weighted/")
    }
    metric_items = [
        (key, value)
        for key, value in log.items()
        if not key.startswith("_") and not key.startswith("_weighted/")
    ]
    logging_dict = {
        "keys": [key for key, _ in metric_items],
        "values": torch.tensor(
            [
                num_samples if not args.calculate_per_token_loss else num_tokens,
            ]
            + [value for _, value in metric_items],
            device=logits.device,
        ),
    }
    if weighted_metrics:
        logging_dict["weighted_metrics"] = weighted_metrics
    if "_logdiff_histogram" in log:
        logging_dict["logdiff_histogram"] = log["_logdiff_histogram"].detach()
    if "_logdiff_min_max" in log:
        logging_dict["logdiff_min_max"] = log["_logdiff_min_max"].detach()

    return (
        loss,
        torch.tensor(num_tokens if args.calculate_per_token_loss else 1, device=logits.device),
        logging_dict,
    )
