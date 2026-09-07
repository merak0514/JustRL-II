# Adapt from https://github.com/OpenRLHF/OpenRLHF/blob/10c733694ed9fbb78a0a2ff6a05efc7401584d46/openrlhf/models/utils.py
# and https://github.com/OpenRLHF/OpenRLHF/blob/10c733694ed9fbb78a0a2ff6a05efc7401584d46/openrlhf/trainer/ppo_utils/experience_maker.py

from argparse import Namespace
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

from miles.backends.training_utils.parallel import get_parallel_state


_TOP_LOGPROB_BWD_DUMP_COUNTER = 0


def _maybe_dump_top_logprob_backward(name: str, tensor: torch.Tensor) -> None:
    dump_dir = None
    try:
        import os

        dump_dir = os.environ.get("MILES_LOGPROB_BACKWARD_DEBUG_DIR")
    except Exception:
        dump_dir = None
    if not dump_dir or not tensor.requires_grad:
        return

    def hook(grad: torch.Tensor) -> torch.Tensor:
        global _TOP_LOGPROB_BWD_DUMP_COUNTER
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        stats = {
            "rank": rank,
            "name": name,
            "shape": tuple(grad.shape),
            "dtype": str(grad.dtype),
            "numel": grad.numel(),
            "finite": torch.isfinite(grad).sum().item(),
            "nan": torch.isnan(grad).sum().item(),
            "inf": torch.isinf(grad).sum().item(),
        }
        finite = grad[torch.isfinite(grad)]
        if finite.numel() > 0:
            finite_f = finite.float()
            stats.update(
                {
                    "max_abs_finite": finite_f.abs().max().item(),
                    "min_finite": finite_f.min().item(),
                    "max_finite": finite_f.max().item(),
                }
            )
        else:
            stats.update({"max_abs_finite": None, "min_finite": None, "max_finite": None})
        counter = _TOP_LOGPROB_BWD_DUMP_COUNTER
        _TOP_LOGPROB_BWD_DUMP_COUNTER += 1
        path = Path(dump_dir) / f"rank_{rank}_{counter:05d}_{name}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"stats": stats, "grad": grad.detach().cpu()}, path)
        print(f"[MILES_LOGPROB_BACKWARD_DEBUG] {stats} wrote {path}", flush=True)
        return grad

    tensor.register_hook(hook)


@torch.compile(dynamic=True)
def compute_approx_kl(
    log_probs: torch.Tensor,
    log_probs_base: torch.Tensor,
    kl_loss_type: str,
    importance_ratio: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Compute the approximate KL divergence between two distributions.
    Schulman blog: http://joschu.net/blog/kl-approx.html

    Args:
        log_probs: Log probabilities of the new distribution.
        log_probs_base: Log probabilities of the base distribution.
        kl_loss_type: Type of KL estimator (k1, k2, k3, low_var_kl).
        importance_ratio: Optional IS ratio (π_θ/π_old) for unbiased KL estimation.
    """
    log_ratio = log_probs.float() - log_probs_base.float()

    if kl_loss_type == "k1":
        kl = log_ratio
    elif kl_loss_type == "k2":
        kl = log_ratio**2 / 2.0
    elif kl_loss_type in ["k3", "low_var_kl"]:
        # The non negative kl approximation in
        # http://joschu.net/blog/kl-approx.html
        # Besides non negative, it is also unbiased and have lower variance.
        log_ratio = -log_ratio
        kl = log_ratio.exp() - 1 - log_ratio
    else:
        raise ValueError(f"Unknown kl_loss_type: {kl_loss_type}")

    # Apply IS ratio for unbiased KL estimation (DeepSeek-V3.2)
    if importance_ratio is not None:
        kl = importance_ratio * kl

    # Clamp only for low_var_kl for numerical stability
    if kl_loss_type == "low_var_kl":
        kl = torch.clamp(kl, min=-10, max=10)

    return kl


def compute_opsm_mask(
    args: Namespace,
    full_log_probs: list[torch.Tensor],
    full_old_log_probs: list[torch.Tensor],
    advantages: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Off-Policy Sequence Masking (OPSM) mask.

    Args:
        args: Configuration containing `opsm_delta` threshold.
        full_log_probs: Current policy log-probs per sample.
        full_old_log_probs: Old policy log-probs per sample.
        advantages: Advantage values per sample.
        loss_masks: Loss masks per sample.

    Returns:
        Tuple of `(opsm_mask, opsm_clipfrac)` where `opsm_mask` is a
        concatenated tensor of per-token masks and
        `opsm_clipfrac` is the count of masked sequences.
    """
    opsm_mask_list = []
    device = advantages[0].device
    opsm_clipfrac = torch.tensor(0.0, device=device)

    for full_log_prob, full_old_log_prob, advantage, loss_mask in zip(
        full_log_probs, full_old_log_probs, advantages, loss_masks, strict=False
    ):
        # Calculate sequence-level KL
        seq_kl = ((full_old_log_prob - full_log_prob) * loss_mask).sum() / torch.clamp_min(loss_mask.sum(), 1)

        # Create mask: 0 if (advantage < 0 and seq_kl > delta), else 1
        mask = ((advantage < 0) & (seq_kl > args.opsm_delta)).float()
        opsm_clipfrac += mask.sum() / torch.clamp_min(loss_mask.sum(), 1)

        opsm_mask_list.append(1 - mask)

    opsm_mask = torch.cat(opsm_mask_list, dim=0)
    return opsm_mask, opsm_clipfrac


def compute_gspo_kl(
    full_log_probs: list[torch.Tensor],
    full_old_log_probs: list[torch.Tensor],
    local_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
) -> torch.Tensor:
    """Compute GSPO-style per-sequence KL divergence.

    Args:
        full_log_probs: Current policy log-probs per sample (full or CP-local).
        full_old_log_probs: Old policy log-probs per sample (full or CP-local).
        local_log_probs: Local (CP-local) log-probs for expansion shape reference.
        loss_masks: Loss masks per sample.

    Returns:
        Concatenated tensor of per-token KL values where each token in a
        sequence has the same KL value (the sequence-level KL).
    """
    # Compute sequence-level KL and expand to per-token
    ppo_kl = [
        ((old_logprob - log_prob) * loss_mask).sum() / torch.clamp_min(loss_mask.sum(), 1)
        for log_prob, old_logprob, loss_mask in zip(full_log_probs, full_old_log_probs, loss_masks, strict=False)
    ]
    ppo_kl = [kl.expand_as(log_prob) for kl, log_prob in zip(ppo_kl, local_log_probs, strict=False)]
    ppo_kl = torch.cat(ppo_kl, dim=0)

    return ppo_kl


@torch.compile(dynamic=True)
def compute_policy_loss(
    ppo_kl: torch.Tensor,
    advantages: torch.Tensor,
    eps_clip: float,
    eps_clip_high: float,
    eps_clip_c: float | None = None,
):
    ratio = (-ppo_kl).exp()
    pg_losses1 = -ratio * advantages
    pg_losses2 = -ratio.clamp(1 - eps_clip, 1 + eps_clip_high) * advantages
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    clipfrac = torch.gt(pg_losses2, pg_losses1).float()

    if eps_clip_c is not None:
        assert (
            eps_clip_c > 1.0
        ), f"The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0, but get the value: {eps_clip_c}."
        pg_losses3 = -eps_clip_c * advantages
        clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
        pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    else:
        pg_losses = clip_pg_losses1

    return pg_losses, clipfrac


def compute_log_probs(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    process_group: dist.ProcessGroup | None,
    *,
    true_on_policy_mode: bool = False,
    vocab_size: int | None = None,
):
    if true_on_policy_mode:
        full_logits = _gather_true_on_policy_full_logits(logits, process_group, vocab_size=vocab_size)
        log_probs = torch.log_softmax(full_logits, dim=-1)
        return log_probs.gather(dim=-1, index=tokens.unsqueeze(-1)).squeeze(-1)

    # TODO: when megatron is not installed, fall back to naive implementation
    from megatron.core.fusions.fused_cross_entropy import fused_vocab_parallel_cross_entropy

    # convert to [seq_len, batch_size, vocab_size] as expected by fused_vocab_parallel_cross_entropy
    logits = logits.unsqueeze(1)
    tokens = tokens.unsqueeze(1)
    return -fused_vocab_parallel_cross_entropy(logits, tokens, process_group)


def _prepare_true_on_policy_full_logits(
    logits_or_shards: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
    *,
    vocab_size: int | None = None,
) -> torch.Tensor:
    if isinstance(logits_or_shards, (list, tuple)):
        full_logits = torch.cat([shard.contiguous() for shard in logits_or_shards], dim=-1)
    else:
        full_logits = logits_or_shards.contiguous()

    if vocab_size is not None and full_logits.size(-1) > vocab_size:
        full_logits = full_logits[..., :vocab_size]

    return full_logits


def _gather_true_on_policy_full_logits(
    logits: torch.Tensor,
    process_group: dist.ProcessGroup | None,
    *,
    vocab_size: int | None = None,
) -> torch.Tensor:
    if process_group is None:
        return _prepare_true_on_policy_full_logits(logits, vocab_size=vocab_size)

    tp_size = dist.get_world_size(process_group)
    if tp_size <= 1:
        return _prepare_true_on_policy_full_logits(logits, vocab_size=vocab_size)

    full_logits = _ReplicatedLossAllGatherLastDim.apply(logits.contiguous(), process_group)
    return _prepare_true_on_policy_full_logits(full_logits, vocab_size=vocab_size)


def _split_replicated_loss_gather_grad(
    grad_output: torch.Tensor,
    *,
    rank: int,
    world_size: int,
    local_last_dim: int,
) -> torch.Tensor:
    if world_size <= 1:
        return grad_output.contiguous()

    expected_last_dim = local_last_dim * world_size
    if grad_output.size(-1) != expected_last_dim:
        raise RuntimeError(
            "True-on-policy replicated-loss gather backward expected the full padded "
            f"vocab dimension to be {expected_last_dim}, got {grad_output.size(-1)}."
        )

    return torch.narrow(grad_output, dim=-1, start=rank * local_last_dim, length=local_last_dim).contiguous()


class _ReplicatedLossAllGatherLastDim(torch.autograd.Function):
    """All-gather vocab shards for a loss replicated on every TP rank.

    Megatron's standard all-gather autograd uses reduce-scatter in backward,
    which is correct when each rank contributes a distinct output gradient. In
    the true-on-policy logprob path every TP rank computes the same scalar loss
    from the gathered full vocabulary, so reduce-scatter would sum identical
    gradients and scale the local logits gradient by TP size.
    """

    @staticmethod
    def forward(ctx, input_: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
        world_size = group.size()
        ctx.group = group
        ctx.local_last_dim = input_.shape[-1]
        ctx.world_size = world_size

        if world_size == 1:
            return input_.contiguous()

        from megatron.core.tensor_parallel.mappings import dist_all_gather_func

        gather_shape = list(input_.shape)
        gather_shape[0] *= world_size
        gathered = torch.empty(gather_shape, dtype=input_.dtype, device=input_.device)
        dist_all_gather_func(gathered, input_.contiguous(), group=group)
        return torch.cat(gathered.chunk(world_size, dim=0), dim=-1).contiguous()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return (
            _split_replicated_loss_gather_grad(
                grad_output,
                rank=ctx.group.rank(),
                world_size=ctx.world_size,
                local_last_dim=ctx.local_last_dim,
            ),
            None,
        )


# from https://github.com/volcengine/verl/blob/0bdf7f469854815177e73dcfe9e420836c952e6e/verl/utils/megatron/tensor_parallel.py#L99
class _VocabParallelEntropy(torch.autograd.Function):

    @staticmethod
    def forward(ctx, vocab_parallel_logits: torch.Tensor, process_group: dist.ProcessGroup) -> torch.Tensor:

        @torch.compile(dynamic=True)
        def mul_reduce(a, b):
            return (a * b).sum(dim=-1, keepdim=True)

        logits_max = vocab_parallel_logits.max(dim=-1, keepdim=True).values
        dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=process_group)
        normalized_vocab_parallel_logits = vocab_parallel_logits - logits_max
        normalized_exp_logits = normalized_vocab_parallel_logits.exp_()
        normalized_sum_exp_logits = normalized_exp_logits.sum(dim=-1, keepdim=True)
        dist.all_reduce(normalized_sum_exp_logits, group=process_group)
        softmax_logits = normalized_exp_logits.div_(normalized_sum_exp_logits)
        sum_softmax_times_logits = mul_reduce(softmax_logits, vocab_parallel_logits)
        dist.all_reduce(sum_softmax_times_logits, group=process_group)
        entropy = logits_max + normalized_sum_exp_logits.log() - sum_softmax_times_logits
        ctx.save_for_backward(vocab_parallel_logits, softmax_logits, sum_softmax_times_logits)
        return entropy.squeeze(dim=-1)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        vocab_parallel_logits, softmax_logits, sum_softmax_times_logits = ctx.saved_tensors
        # Out-of-place ops: do NOT modify saved vocab_parallel_logits so the
        # caller can safely reuse the same clone for both entropy and log_prob
        # in calculate_log_probs_and_entropy.
        workspace = vocab_parallel_logits - sum_softmax_times_logits
        grad = softmax_logits * workspace * grad_output.unsqueeze(dim=-1) * (-1)
        return grad, None


def compute_entropy_from_logits(logits: torch.Tensor, process_group) -> torch.Tensor:
    return _VocabParallelEntropy.apply(logits, process_group)


def get_grpo_returns(
    rewards: torch.Tensor,
    kl: list[torch.Tensor],
):
    returns = []
    for i in range(len(rewards)):
        returns.append(torch.ones_like(kl[i]) * rewards[i])
    return returns


def get_reinforce_plus_plus_returns(
    rewards: torch.Tensor,
    kl: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    response_lengths: list[int],
    total_lengths: list[int],
    kl_coef: float,
    gamma: float,
) -> list[torch.Tensor]:
    """
    Calculates discounted returns for REINFORCE++ (https://arxiv.org/pdf/2501.03262)

    Args:
        rewards (Tensor): A tensor of scalar rewards for each sequence.
        kl (List[Tensor]): List of per-token KL divergence tensors for sequence chunks.
        loss_masks (List[Tensor]): List of response-only loss masks for each full sequence.
        response_lengths (List[int]): The full length of each response sequence.
        total_lengths (List[int]): The full length of each sequence (prompt + response).
        kl_coef (float): Coefficient for the KL penalty.
        gamma (float): The discount factor.

    Returns:
        List[torch.Tensor]: A list of return (G_t) tensors for the
                            local sequence chunks owned by the current GPU rank.
    """

    cp_size = get_parallel_state().cp.size

    final_returns_chunks = []
    for i in range(len(rewards)):
        local_kl_chunk = kl[i]
        total_len, response_len = total_lengths[i], response_lengths[i]

        if cp_size > 1:
            # Step 1,2:Gather all chunks and token_offsets from all ranks and reconstruct the full response tensor by splitting and placing each part
            from miles.backends.training_utils.cp_utils import all_gather_with_cp

            full_kl_response = all_gather_with_cp(local_kl_chunk, total_len, response_len)
        else:
            full_kl_response = local_kl_chunk

        # Step 3: Compute returns on full response kl tensor.
        token_level_rewards = -kl_coef * full_kl_response
        full_mask = loss_masks[i]
        assert full_mask.sum().item() > 0, f"Sequence at index {i} is fully masked."
        last_idx = full_mask.nonzero(as_tuple=True)[0][-1]
        token_level_rewards[last_idx] += rewards[i]

        returns_for_seq = torch.zeros_like(token_level_rewards)
        running_return = 0.0
        for t in reversed(range(token_level_rewards.size(0))):
            # G_t = r_t + gamma * G_{t+1}
            running_return = token_level_rewards[t] + gamma * running_return
            returns_for_seq[t] = running_return

        # Step 4: Pick up the results corresponding to our local chunk's parts.
        if cp_size > 1:
            from miles.backends.training_utils.cp_utils import slice_log_prob_with_cp

            local_returns_chunk = slice_log_prob_with_cp(returns_for_seq, total_len, response_len)
        else:
            local_returns_chunk = returns_for_seq

        final_returns_chunks.append(local_returns_chunk)

    return final_returns_chunks


def get_reinforce_plus_plus_baseline_advantages(
    rewards: torch.Tensor,
    kl: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    kl_coef: float,
) -> list[torch.Tensor]:
    """
    Calculates the unwhitened advantages for the REINFORCE++-baseline algorithm.
    Broadcasting the scalar (reward - group_baseline) to each token.

    Args:
        rewards (Tensor): A tensor of scalar rewards, where the group-wise
                                baseline has already been subtracted.
        kl (list[Tensor]): A list of per-token KL divergence tensors. Used to
                                 get the shape for broadcasting.
        loss_masks (list[Tensor]): A list of per-token loss masks.
        kl_coef (float): Coefficient for the KL penalty.

    Returns:
        list[Tensor]: A list of tensors containing the unwhitened advantages.
    """
    # Broadcast to get unwhitened advantages
    unwhitened_advantages = [
        torch.ones_like(kl_tensor) * reward_val - kl_coef * kl_tensor
        for kl_tensor, reward_val in zip(kl, rewards, strict=False)
    ]

    return unwhitened_advantages


def get_advantages_and_returns(
    total_len: int,
    response_len: int,
    values: torch.Tensor,
    rewards: torch.Tensor,
    gamma: float,
    lambd: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Function that computes advantages and returns from rewards and values.
    Calculated as in the original PPO paper: https://arxiv.org/abs/1707.06347
    Note that rewards may include a KL divergence loss term.

    Advantages looks like this:
    Adv1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
            - V1 + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

    Returns looks like this:
    Ret1 =  R1 + γ * λ * R2     + γ^2 * λ^2 * R3       + ...
                + γ * (1 - λ) V2 + γ^2 * λ * (1 - λ) V3 + ...

    Input:
    - values: Tensor of shape (response_size,)
    - rewards: Tensor of shape (response_size,)

    Output:
    - advantages: Tensor of shape (response_size,)
    - returns: Tensor of shape (response_size,)
    """

    cp_size = get_parallel_state().cp.size
    if cp_size > 1:
        from miles.backends.training_utils.cp_utils import all_gather_with_cp

        full_rewards = all_gather_with_cp(rewards, total_len, response_len)
        full_values = all_gather_with_cp(values, total_len, response_len)
    else:
        full_rewards = rewards
        full_values = values

    lastgaelam = 0
    advantages_reversed = []

    for t in reversed(range(response_len)):
        nextvalues = full_values[t + 1] if t < response_len - 1 else 0.0
        delta = full_rewards[t] + gamma * nextvalues - full_values[t]
        lastgaelam = delta + gamma * lambd * lastgaelam
        advantages_reversed.append(lastgaelam)
    full_advantages = torch.tensor(advantages_reversed[::-1], dtype=full_values.dtype, device=full_values.device)
    full_returns = full_advantages + full_values

    if cp_size > 1:
        from miles.backends.training_utils.cp_utils import slice_log_prob_with_cp

        advantages = slice_log_prob_with_cp(full_advantages, total_len, response_len)
        returns = slice_log_prob_with_cp(full_returns, total_len, response_len)
    else:
        advantages = full_advantages
        returns = full_returns

    return advantages.detach(), returns


def compute_overlong_penalty(args, response_length: int) -> float:
    """DAPO Soft Overlong Punishment length penalty for a single sample.

    Returns 0 when the response is within the safe zone, and a value in
    [-overlong_penalty_factor, 0) that decreases linearly once the response
    length enters the buffer band ``[max_len - buffer_len, max_len]``.
    Returns 0 when soft overlong punishment is disabled (buffer_len <= 0).
    See https://arxiv.org/abs/2503.14476 .

    历史上住在 miles/ray/rollout.py; 为了让训练侧 (--olp-analytic-inject, 臂#14)
    与 rollout 侧共用同一份实现而挪到这里 (本模块无 megatron/sglang 依赖, 双方
    都能 import, 无循环依赖)。
    """
    buffer_len = getattr(args, "overlong_buffer_len", 0) or 0
    if buffer_len <= 0:
        return 0.0
    max_len = args.rollout_max_response_len
    factor = float(getattr(args, "overlong_penalty_factor", 1.0))
    expected_len = max_len - buffer_len
    if response_length <= expected_len:
        return 0.0
    penalty = (expected_len - response_length) / buffer_len * factor
    return max(penalty, -factor)


def length_adaptive_lambda(k: float, response_lengths, device, dtype=torch.float32) -> torch.Tensor:
    """JustRL2 length-adaptive GAE λ: per-sample ``λ_i = k ** (1 / L_i)``.

    ``k`` is the fraction of terminal credit that reaches the first token: with γ=1 the
    GAE weight of the terminal reward at distance d is ``λ_i**d``, so the first token
    always sees ``λ_i**L_i == k`` regardless of response length. Longer responses get λ
    closer to 1, so credit propagation does not weaken with length. (The earlier
    form ``1 − 1/(α·L)`` is the first-order expansion of this with ``k = exp(−1/α)``.)

    Computed in float32 regardless of ``dtype``: at 128k lengths ``k**(1/L)`` is
    ``1 − 8e-6``-ish, which bf16 rounds to exactly 1.0. ``L_i == 0`` (empty response)
    gives λ=0; that row is fully masked downstream, so the value is irrelevant.
    """
    assert 0.0 < k <= 1.0, f"--gae-lambda-k must be in (0, 1], got {k}"
    lengths = torch.tensor(response_lengths, device=device, dtype=torch.float32)
    lam = torch.where(lengths > 0, float(k) ** (1.0 / lengths.clamp(min=1.0)), torch.zeros_like(lengths))
    return lam.to(dtype)


def apply_olp_analytic_injection(
    full_advantages: torch.Tensor,
    response_lengths,
    penalties,
    inject_alpha: "float | None" = None,
    lam_rowwise: "torch.Tensor | None" = None,
) -> torch.Tensor:
    """--olp-analytic-inject (臂#14, cleancritic 2.0) 的解析注入核心。

    ``lam_rowwise`` 若给出 ([B] 逐样本 λ_i, 如 length_adaptive_lambda 的输出) 则直接用它做
    衰减, 忽略 ``inject_alpha``; 否则按 α 参数化 λ_i = clamp(1 − 1/(α_inj·L_i), 0)。

    在全长坐标视图 full_advantages [B, max_len] 上, 对每个样本 i 把 OLP 惩罚
    P_i 以指数衰减权重注入 advantage:

        full_advantages[i, t] += P_i * λ_i^(L_i - 1 - t),  t ∈ [0, L_i)

    其中 λ_i = clamp(1 − 1/(α_inj·L_i), min=0) (注: 这是臂#14 自己的 α 参数化, 与
    --gae-lambda-k 的 k^(1/L) 不同): α_inj=0.1 即只有末尾 ~10%·L 的 token 显著感到惩罚 (终点 token
    权重恒为 1, 距终点 d 每远一个 token 衰减 λ 倍); α_inj·L ≤ 1 时 λ=0, 只罚
    最后一个 token (0^0=1)。P_i=0 的样本加数恒为 0, 逐字节不变; padding 区
    (t ≥ L_i) 被 mask 成 0, 不动。全程逐元素张量运算, 无 python 循环。

    Args:
        full_advantages: [B, max_len] 全长坐标 advantage (GAE 输出)。
        response_lengths: list[int], 每个样本的完整 response 长度 (非 CP 分片长度)。
        penalties: list[float], 每个样本的 OLP 惩罚 P_i (compute_overlong_penalty
            的输出, 惩罚为负, 无惩罚为 0)。
        inject_alpha: 注入衰减半径 α_inj (> 0)。

    Returns:
        注入后的 full_advantages (新张量, 不原地修改输入)。
    """
    B, max_len = full_advantages.shape
    assert len(response_lengths) == B and len(penalties) == B
    device = full_advantages.device
    dtype = full_advantages.dtype

    # 权重计算固定在 float32 进行, 与 full_advantages 的 dtype 无关: bf16 下
    # λ = 1 − 1/(α·L) 在 128k 长度上是 1 − 7.9e-5, 会被舍入成 1.0; t_idx/lengths
    # 这类大整数 (>256) bf16 也无法精确表示, 终点边界比较会错判 (实测 L=126976 时
    # w_end=0、其余位置全 −1 的灾难性翻转)。算完再把注入项 .to(dtype) 回写。
    compute_dtype = torch.float32
    penalties_t = torch.tensor(penalties, device=device, dtype=compute_dtype)  # [B]
    lengths_t = torch.tensor(response_lengths, device=device, dtype=compute_dtype)  # [B]
    if lam_rowwise is not None:
        lam = lam_rowwise.to(device=device, dtype=compute_dtype)  # [B]
    else:
        assert inject_alpha is not None, "apply_olp_analytic_injection needs inject_alpha or lam_rowwise"
        # L=0 (空响应) 时 1/(α·0)=inf → 1-inf=-inf → clamp 到 0; 该行 valid mask 全 False,
        # 注入量恒 0, 不会产生 NaN。
        lam = torch.clamp(1.0 - 1.0 / (inject_alpha * lengths_t), min=0.0)  # [B]
    t_idx = torch.arange(max_len, device=device, dtype=compute_dtype).unsqueeze(0)  # [1, max_len]
    # 距终点距离 d = L_i - 1 - t; padding 区 d<0, 先 clamp 到 0 避免 0^负数=inf,
    # 再用 valid mask 归零 (clamp 后 padding 区权重是 λ^0=1, 必须靠 mask 拦住)。
    dist_to_end = (lengths_t - 1.0).unsqueeze(1) - t_idx  # [B, max_len]
    valid = t_idx < lengths_t.unsqueeze(1)  # [B, max_len]
    weights = torch.where(
        valid,
        lam.unsqueeze(1) ** dist_to_end.clamp(min=0.0),
        torch.zeros((), device=device, dtype=compute_dtype),
    )
    return full_advantages + (penalties_t.unsqueeze(1) * weights).to(dtype)


def get_advantages_and_returns_batch(
    total_lengths,
    response_lengths,
    values_list,
    rewards_list,
    gamma,
    lambd,
    chunked: bool = True,
    length_adaptive_lambda_k: "float | None" = None,
    olp_inject_penalties: "list[float] | None" = None,
    olp_inject_alpha: float = 0.1,
    group_center: "dict | None" = None,
):
    """
    Batched GAE with CP support.
    Input:
        total_lengths:     list[int], each sample's total_len
        response_lengths:  list[int], each sample's response_len
        values_list:       list[Tensor], each shape = [resp_len_i]
        rewards_list:      list[Tensor], same shape
        length_adaptive_lambda_k: 若设置 (JustRL2 length-adaptive 解耦 GAE), advantage 用
            逐样本 λ_i = k^(1/L_i) (首 token 恒拿到终端 credit 的 k 倍, 与长度无关),
            而 returns 用 λ=1 的无偏目标 (γ=1 时即奖励后缀和), 二者解耦。
        olp_inject_penalties: 若非 None (--olp-analytic-inject, 臂#14), 每个样本的
            OLP 惩罚 P_i; GAE 算完后在全长坐标上按 λ_inj^d 注入 advantage,
            returns 不动 (见 apply_olp_analytic_injection)。
        olp_inject_alpha: 注入衰减半径 α_inj (仅在 olp_inject_penalties 非 None 时用)。
        group_center: 若非 None (--group-center-inject, 臂#16), 组内留一中心化注入
            的上下文 dict, 键:
              positions:  list[int], 本地样本在全局 flat 序里的位置 (process_rollout_data
                          保留的 DP partition, 组内 n_g 条样本在 flat 序里连续);
              raw_rewards: list[float], 全局 flat 序完整 raw_reward 列表 (长度 N_global,
                          每个 DP rank 都持有整份);
              n_samples_per_prompt: int, 组大小 n_g;
              k:          float, 注入衰减 k (= args.gae_lambda_k, 与 GAE 同款 λ_i = k^(1/L_i));
              loss_masks: list[Tensor], 本地样本全长 response 坐标 loss mask;
              dp_group:   ProcessGroup | None, 组内求和的 allreduce 组 (intra_dp);
              stats_out:  dict | None, 回填统计 {"correction": [P_i], "abs": [|P_i|]}。
            GAE 之后按 λ_i^(L_i-1-t) 把 P_i 注入 advantage, returns 不动。
    Output:
        advantages_list:   list[Tensor], each shape = [resp_len_i]
        returns_list:      list[Tensor], same shape
    """

    with torch.no_grad():
        B = len(response_lengths)
        assert B == len(values_list)
        assert B == len(rewards_list)

        cp_size = get_parallel_state().cp.size
        device = values_list[0].device
        dtype = values_list[0].dtype

        # pad to max_len for batched GAE
        max_len = max(response_lengths)

        full_values = torch.zeros(B, max_len, device=device, dtype=dtype)
        full_rewards = torch.zeros(B, max_len, device=device, dtype=dtype)

        if cp_size > 1:
            from miles.backends.training_utils.cp_utils import local_response_to_full

            # 集合通信不变量：CP 组内所有 rank 必须执行完全相同的通信序列。
            # 这里先把各样本的本地分片放回全长坐标（纯本地），再对整个 batch 只做
            # 两次固定的 allreduce（values/rewards 各一次），而不是逐样本 2B 次——
            # 消除任何数据依赖的通信次数偏差。
            for i, (total_len, resp_len, v, r) in enumerate(
                zip(total_lengths, response_lengths, values_list, rewards_list, strict=False)
            ):
                full_values[i, :resp_len] = local_response_to_full(v, total_len, resp_len)
                full_rewards[i, :resp_len] = local_response_to_full(r.to(dtype), total_len, resp_len)

            cp_group = get_parallel_state().cp.group
            dist.all_reduce(full_values, group=cp_group)
            dist.all_reduce(full_rewards, group=cp_group)
        else:
            for i in range(B):
                L = response_lengths[i]
                full_values[i, :L] = values_list[i][:L]
                full_rewards[i, :L] = rewards_list[i][:L]

        # fp32 copy of λ_i kept for the group-center injection below: at 128k
        # lengths λ = 1 − O(1e-5), which bf16 rounds to exactly 1.0 (every token
        # would then see the full injected scalar). chunked_gae casts λ to the
        # rewards dtype itself, so the GAE path is unchanged from before.
        lam32 = None
        if length_adaptive_lambda_k:
            assert gamma == 1.0, "length-adaptive decoupled GAE requires gamma == 1.0 (returns = suffix reward sum)"
            lam32 = length_adaptive_lambda(length_adaptive_lambda_k, response_lengths, device)
            lambd_rowwise = lam32.to(dtype)
            full_advantages, _ = chunked_gae(
                rewards=full_rewards,
                values=full_values,
                gamma=gamma,
                lambd=lambd_rowwise,
            )
            full_returns = torch.flip(torch.cumsum(torch.flip(full_rewards, dims=[1]), dim=1), dims=[1])
        elif not chunked:
            full_advantages, full_returns = vanilla_gae(
                rewards=full_rewards,
                values=full_values,
                gamma=gamma,
                lambd=lambd,
            )
        else:
            full_advantages, full_returns = chunked_gae(
                rewards=full_rewards,
                values=full_values,
                gamma=gamma,
                lambd=lambd,
            )

        # --olp-analytic-inject (臂#14): 在 GAE 算完之后、切回 per-sample/CP 分片之前注入。
        # 此处 full_advantages 是全长坐标 [B, max_len]: CP>1 时各 rank 的 full_rewards/
        # full_values 经上面的 allreduce 后完全一致, GAE 与注入都是确定性逐元素运算,
        # 逐 rank 结果一致, 之后统一走 slice_log_prob_with_cp 切分——天然规避 zigzag
        # 分片坐标问题。full_returns 已在注入前算好且此后不再改动: returns (critic 的
        # 回归目标) 保持无 OLP (开关开启时 rollout 侧 reward 通道本就没加 OLP)。
        if olp_inject_penalties is not None:
            full_advantages = apply_olp_analytic_injection(
                full_advantages, response_lengths, olp_inject_penalties, olp_inject_alpha
            )

        # --group-center-inject (臂#16): 同样在 GAE 之后、CP slice 之前的全长坐标视图上
        # 注入 (与臂#14 是两个独立的线性加法, 天然可共存)。终端标量 a_j = raw_reward_j −
        # V_j (该样本最后一个 loss-mask token 的 critic value), 注入量为组内留一均值取负
        # P_i = −(Σ_{j∈g, j≠i} a_j)/(n_g−1), 衰减 λ_i 与长度自适应解耦 GAE 同款
        # (k=gae_lambda_k, λ_i=k^(1/L_i))——γ=1 时 GAE 对终端 reward 脉冲的传播权重恰为
        # λ_i^(L_i−1−t), 故注入在数学上等价于 "终端 reward 减去组内留一均值后重跑 GAE"
        # 的 advantage; 但 full_returns 已在此前算好且不再动, critic 照学未中心化 return。
        #
        # 组结构与通信: 全局 flat 序里同组 n_g 条样本连续 (data_source 逐 prompt 深拷贝
        # + sglang_rollout 按组首样本 index 排序 + chain 展平), 但 DP 切分 (balance_data
        # 的长度均衡分区、或跨步切分) 会把组成员打散到各 rank——组内求和必须走 DP 集合
        # 通信: 各 rank 把本地 a_j 按 flat 位置散射进稠密 [N_global] fp32 buffer, 一次
        # all_reduce(SUM) 拼出完整 a 向量 (partition 互斥且覆盖全批, 无重叠无遗漏)。
        #
        # 集合通信不变量 (仿 loss.py normalize_advantages 的 DP-allreduce 注释): 开关是
        # 全局 args 而非数据依赖, 所有 intra_dp rank 无条件走到这里恰好一次, 不会死锁;
        # 千万不要把这段包进任何数据依赖的条件分支。CP>1 时 a_local 源于上面 CP-allreduce
        # 后的 full_values (全长坐标、逐 CP rank 一致), loss_masks 未做 CP 切分, 故各 CP
        # rank 的 a_local/a_buf 一致, 各自在自己的 intra_dp 组内 allreduce, 结果逐 rank
        # 一致, 无需跨 CP 通信。单测 (无 torch.distributed 初始化) 下单进程即持有全量
        # 数据, 跳过 allreduce 即为退化正确。
        if group_center is not None:
            n_g = int(group_center["n_samples_per_prompt"])
            raw_full = group_center["raw_rewards"]
            N_global = len(raw_full)
            positions = group_center["positions"]
            assert len(positions) == B, (
                f"group_center_inject: positions ({len(positions)}) and local batch ({B}) mismatch"
            )
            assert N_global % n_g == 0, (
                f"group_center_inject: global batch ({N_global}) is not whole groups of {n_g} "
                "(trim/subsample/custom convert_samples_to_train_data broke group contiguity?)"
            )
            # a_j 固定 fp32 计算 (full_values 可能是 bf16; 与 apply_olp_analytic_injection
            # 的 fp32 中间计算口径一致)。
            a_local = torch.zeros(B, device=device, dtype=torch.float32)
            for i in range(B):
                L = response_lengths[i]
                if L == 0:
                    # 空响应 (立即 EOS): full_values 该行全 0, 取 a=0——不污染兄弟样本的
                    # 基线, 自身注入也被 apply 的 valid mask 归零。
                    continue
                nz = group_center["loss_masks"][i].nonzero(as_tuple=True)[0]
                # 全零 mask (remove_sample/env_error/overlong_filtering 置零) 无 nonzero:
                # 回退终点 L−1。该样本自身梯度已被 mask, 但仍以真实 raw−V 贡献兄弟基线。
                last_idx = int(nz[-1].item()) if nz.numel() > 0 else L - 1
                a_local[i] = float(raw_full[positions[i]]) - full_values[i, last_idx].float()
            pos_t = torch.tensor(list(positions), device=device, dtype=torch.long)
            a_buf = torch.zeros(N_global, device=device, dtype=torch.float32)
            a_buf[pos_t] = a_local
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(a_buf, group=group_center["dp_group"])
            if n_g > 1:
                # flat 序里组连续 → 位置 p 的组号是 p // n_g; 留一均值 = (组和 − 自身)/(n_g−1)
                g_sum = a_buf.view(-1, n_g).sum(dim=1)  # [N_global / n_g]
                p_local = -(g_sum[pos_t // n_g] - a_local) / (n_g - 1)
            else:
                p_local = torch.zeros_like(a_local)  # n_g==1: 无兄弟, P=0, 严格 no-op
            p_list = [float(p) for p in p_local.tolist()]
            gc_lam = lam32 if lam32 is not None else length_adaptive_lambda(
                group_center["k"], response_lengths, device
            )
            full_advantages = apply_olp_analytic_injection(
                full_advantages, response_lengths, p_list, lam_rowwise=gc_lam
            )
            stats_out = group_center.get("stats_out")
            if stats_out is not None:
                stats_out["correction"] = p_list
                stats_out["abs"] = [abs(p) for p in p_list]

        advantages_list = []
        returns_list = []

        if cp_size > 1:
            from miles.backends.training_utils.cp_utils import slice_log_prob_with_cp

            for total_len, resp_len, adv_row, ret_row in zip(
                total_lengths,
                response_lengths,
                full_advantages,
                full_returns,
                strict=False,
            ):
                adv_full = adv_row  # shape = [resp_len_i padded to max_len]
                ret_full = ret_row

                adv_sliced = slice_log_prob_with_cp(adv_full[:resp_len], total_len, resp_len)
                ret_sliced = slice_log_prob_with_cp(ret_full[:resp_len], total_len, resp_len)

                advantages_list.append(adv_sliced)
                returns_list.append(ret_sliced)

        else:
            for i in range(B):
                L = response_lengths[i]
                advantages_list.append(full_advantages[i, :L])
                returns_list.append(full_returns[i, :L])

    return advantages_list, returns_list


def vanilla_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    lambd: float,
):
    B, T = rewards.shape
    device = rewards.device
    dtype = rewards.dtype

    lastgaelam = torch.zeros(B, device=device, dtype=dtype)
    adv_rev = []

    for t in reversed(range(T)):
        next_value = values[:, t + 1] if t < T - 1 else 0.0
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        lastgaelam = delta + gamma * lambd * lastgaelam
        adv_rev.append(lastgaelam)

    full_advantages = torch.stack(adv_rev[::-1], dim=1)  # [B, max_len]
    full_returns = full_advantages + values  # [B, max_len]
    return full_advantages, full_returns


def chunked_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    lambd: "float | torch.Tensor",
    chunk_size: int = 128,
):
    """
    Compute Generalized Advantage Estimation (GAE) using a FlashLinearAttention-
    inspired algorithm: parallel prefix scan within chunks and recurrent state
    propagation across chunks.

    This reduces the sequential dependency length from O(T) to O(T / chunk_size),
    while keeping chunk computations fully parallelizable (O(C^2) per chunk).

    Args:
        rewards (Tensor): [B, T] reward sequence.
        values (Tensor):  [B, T] value predictions. The next-value of the final
                          step is assumed to be zero (standard PPO convention).
        gamma (float): discount factor.
        lambd (float | Tensor): GAE lambda; a scalar, or a [B] tensor for
                          per-sample lambda (length-adaptive GAE).
        chunk_size (int): sequence chunk length for parallel scan.

    Returns:
        advantages (Tensor): [B, T] computed advantages.
        returns (Tensor):    [B, T] advantages + values.
    """

    # -------------------------------------------------------------------------
    # Validate inputs
    # -------------------------------------------------------------------------
    assert rewards.ndim == 2 and values.ndim == 2
    B, T = rewards.shape
    assert values.shape == (B, T)

    device = rewards.device
    dtype = rewards.dtype

    rowwise_lambda = isinstance(lambd, torch.Tensor)
    if rowwise_lambda:
        assert lambd.shape == (B,), f"per-sample lambd must be [B], got {lambd.shape}"
        lambd = lambd.to(device=device, dtype=dtype)

    # -------------------------------------------------------------------------
    # Build δ_t = r_t + γ * V_{t+1} - V_t   with V_{T} = 0
    # -------------------------------------------------------------------------
    next_values = torch.cat(
        [values[:, 1:], torch.zeros(B, 1, device=device, dtype=dtype)],
        dim=1,
    )
    deltas = rewards + gamma * next_values - values

    # Reformulate backward GAE as a forward scan on the reversed sequence:
    #   S[i] = Δ[i] + w * S[i - 1],   w = γλ
    w = gamma * lambd
    deltas_rev = torch.flip(deltas, dims=[1])  # [B, T]

    # -------------------------------------------------------------------------
    # Pad to a multiple of chunk_size
    # -------------------------------------------------------------------------
    if T % chunk_size != 0:
        pad = chunk_size - (T % chunk_size)
        deltas_rev = F.pad(deltas_rev, (0, pad))
    else:
        pad = 0

    B, T_pad = deltas_rev.shape
    n_chunks = T_pad // chunk_size

    deltas_chunks = deltas_rev.view(B, n_chunks, chunk_size)

    # -------------------------------------------------------------------------
    # Construct the intra-chunk parallel scan kernel M
    #
    # For a chunk Δ[0..C-1], we want:
    #   S_local[t] = sum_{k=0..t} w^(t-k) * Δ[k]
    #
    # This is implemented as:
    #   S_local = Δ @ M
    #
    # where:
    #   M[i, j] = w^(j - i)    if j >= i
    #             0            otherwise
    # -------------------------------------------------------------------------
    idx = torch.arange(chunk_size, device=device)
    row = idx[:, None]
    col = idx[None, :]
    diff = col - row
    mask = diff >= 0

    if rowwise_lambda:
        # 0**0 == 1 in torch, so w == 0 rows degenerate to the identity kernel.
        diff_f = diff.clamp(min=0).to(dtype)
        M = torch.where(mask, w[:, None, None] ** diff_f, torch.zeros((), device=device, dtype=dtype))  # [B, C, C]
        pow_vec = w[:, None] ** torch.arange(1, chunk_size + 1, device=device, dtype=dtype)  # [B, C]
    else:
        M = torch.zeros(chunk_size, chunk_size, device=device, dtype=dtype)

        if w == 0.0:
            M[mask & (diff == 0)] = 1.0
        else:
            M[mask] = w ** diff[mask].to(dtype)

        # pow_vec[t] = w^(t+1), used to inject the recurrent state s_prev
        if w == 0.0:
            pow_vec = torch.zeros(chunk_size, device=device, dtype=dtype)
        else:
            pow_vec = w ** torch.arange(1, chunk_size + 1, device=device, dtype=dtype)

    # -------------------------------------------------------------------------
    # Parallel compute local chunk results (assuming initial state = 0)
    # -------------------------------------------------------------------------
    if rowwise_lambda:
        S_local_chunks = torch.einsum("bnc,bcd->bnd", deltas_chunks, M)
    else:
        deltas_flat = deltas_chunks.reshape(B * n_chunks, chunk_size)
        S_local_flat = deltas_flat @ M
        S_local_chunks = S_local_flat.view(B, n_chunks, chunk_size)

    # Effective length of each chunk (the last chunk may be padded)
    lengths = [chunk_size] * n_chunks
    if pad > 0:
        lengths[-1] = chunk_size - pad

    # -------------------------------------------------------------------------
    # Recurrent propagation between chunks
    #
    # Each chunk contributes:
    #   S_global[t] = S_local[t] + w^(t+1) * s_prev
    #
    # And updates:
    #   s_prev = S_global[last_t]
    # -------------------------------------------------------------------------
    S_rev = deltas_rev.new_zeros(B, T_pad)
    s_prev = torch.zeros(B, device=device, dtype=dtype)

    for c in range(n_chunks):
        Lc = lengths[c]
        start = c * chunk_size
        end = start + Lc

        S_local = S_local_chunks[:, c, :Lc]
        if rowwise_lambda:
            S_global = S_local + s_prev.unsqueeze(1) * pow_vec[:, :Lc]
        else:
            S_global = S_local + s_prev.unsqueeze(1) * pow_vec[:Lc]

        S_rev[:, start:end] = S_global
        s_prev = S_global[:, -1]  # state for next chunk

    # Remove padding and flip back to original time order
    if pad > 0:
        S_rev = S_rev[:, :T]

    advantages = torch.flip(S_rev, dims=[1])
    returns = advantages + values

    return advantages, returns


def calculate_log_probs_and_entropy(
    logits,
    tokens,
    tp_group,
    with_entropy: bool = False,
    chunk_size: int = -1,
    true_on_policy: bool = False,
    vocab_size: int | None = None,
):
    if true_on_policy:
        return _calculate_log_probs_and_entropy_true_on_policy(
            logits,
            tokens,
            tp_group,
            with_entropy=with_entropy,
            chunk_size=chunk_size,
            vocab_size=vocab_size,
        )

    logits = logits.contiguous()
    # fused_vocab_parallel_cross_entropy (used by compute_log_probs) modifies
    # the logits tensor in-place.  Entropy forward only reads the input, and
    # we compute it under torch.no_grad() so no autograd graph is built —
    # this avoids retaining the large [T, V/TP] saved_tensors from
    # _VocabParallelEntropy and saves significant VRAM.
    entropy = None
    if logits.size(0) != 0:
        if chunk_size > 0:
            num_chunks = (logits.size(0) - 1) // chunk_size + 1
            tokens_chunks = tokens.chunk(num_chunks, dim=0)
            logits_chunks = logits.chunk(num_chunks, dim=0)
            log_probs = []
            entropys = []
            for tokens_chunk, logits_chunk in zip(tokens_chunks, logits_chunks, strict=True):
                if with_entropy:
                    with torch.no_grad():
                        entropys.append(compute_entropy_from_logits(logits_chunk, tp_group))
                log_probs.append(compute_log_probs(logits_chunk.clone(), tokens_chunk, tp_group))
            log_prob = torch.cat(log_probs, dim=0)
            if with_entropy:
                entropy = torch.cat(entropys, dim=0)
        else:
            if with_entropy:
                with torch.no_grad():
                    entropy = compute_entropy_from_logits(logits, tp_group)
                log_prob = compute_log_probs(logits.clone(), tokens, tp_group)
            else:
                log_prob = compute_log_probs(logits, tokens, tp_group)
    else:
        log_prob = logits.new_zeros((0,))
        if with_entropy:
            entropy = logits.new_zeros((0,))

    return log_prob, entropy


def _calculate_log_probs_and_entropy_true_on_policy(
    logits: torch.Tensor,
    tokens: torch.Tensor,
    tp_group: dist.ProcessGroup | None,
    with_entropy: bool = False,
    chunk_size: int = -1,
    vocab_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """True-on-policy log-prob and entropy computation matching SGLang's scoring contract.

    Args:
        logits: Aligned local logits of shape ``[R, V_local]`` (already
            response-sliced and temperature-scaled by ``get_responses``).
        tokens: Target tokens of shape ``[R]``.
        tp_group: Tensor-parallel process group for vocab gather.
        with_entropy: If True, also compute entropy.
        chunk_size: When > 0, process logits in chunks of this size along dim-0
            to avoid allocating a full ``[R, V]`` intermediate tensor (which can
            OOM for long responses with large vocabularies).
        vocab_size: Real tokenizer vocab size. If provided, padded logits are
            truncated after the full-vocab gather and before ``log_softmax``.

    Returns:
        Tuple of ``(log_probs, entropy)`` where *log_probs* has shape ``[R]``
        and *entropy* has shape ``[R]`` or is ``None``.
    """
    if logits.size(0) == 0:
        log_prob = logits.new_zeros((0,))
        entropy = logits.new_zeros((0,)) if with_entropy else None
        return log_prob, entropy

    if chunk_size > 0 and logits.size(0) > chunk_size:
        num_chunks = (logits.size(0) - 1) // chunk_size + 1
        logits_chunks = logits.chunk(num_chunks, dim=0)
        tokens_chunks = tokens.chunk(num_chunks, dim=0)
        log_prob_parts = []
        entropy_parts = []
        for lg, tk in zip(logits_chunks, tokens_chunks, strict=True):
            lp, ent = _calculate_log_probs_and_entropy_true_on_policy(
                lg,
                tk,
                tp_group,
                with_entropy=with_entropy,
                chunk_size=-1,
                vocab_size=vocab_size,
            )
            log_prob_parts.append(lp)
            if ent is not None:
                entropy_parts.append(ent)
        log_prob = torch.cat(log_prob_parts, dim=0)
        entropy = torch.cat(entropy_parts, dim=0) if with_entropy else None
        return log_prob, entropy

    full_logits = _gather_true_on_policy_full_logits(logits, tp_group, vocab_size=vocab_size)
    _maybe_dump_top_logprob_backward("full_logits", full_logits)
    log_probs_full = torch.log_softmax(full_logits, dim=-1)
    _maybe_dump_top_logprob_backward("log_probs_full", log_probs_full)
    log_prob = torch.gather(log_probs_full, dim=-1, index=tokens.unsqueeze(-1)).squeeze(-1)
    _maybe_dump_top_logprob_backward("log_prob", log_prob)

    entropy = None
    if with_entropy:
        probs = log_probs_full.exp()
        entropy = -(probs * log_probs_full).sum(dim=-1)

    return log_prob, entropy
