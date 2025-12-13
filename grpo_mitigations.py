"""
GRPO Credit Assignment Mitigations

This module provides 4 alternative update_policy implementations with different
credit assignment strategies:

1. Outcome-Conditional Advantage: Discount advantage for <think> section tokens
2. Inverse Log Prob Weighting: Weight by 1/(|log_prob| + epsilon)
3. Attention-Based Credit: Weight by attention from answer tokens
4. Entropy-Based Masking: Mask bottom X% entropy tokens (from paper 2506.01939)

Each can be used as a drop-in replacement for update_policy in grpo_instrumented.py
"""

import dataclasses
import math
import re
from collections import defaultdict
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from data_types import Episode
from grpo_instrumented import (
    get_credit_logger,
    normalize_rewards_per_group,
    compute_entropy,
)


# ============================================================
# Mitigation 1: Outcome-Conditional Advantage
# ============================================================

def find_section_boundaries(token_ids: List[int], tokenizer) -> Tuple[int, int]:
    """
    Find the boundaries of <think> and <answer> sections.

    Returns:
        (answer_start, answer_end): Indices of answer section
        If no answer found, returns (len(tokens), len(tokens))
    """
    text = tokenizer.detokenize(token_ids)

    # Find <answer> tag position in text
    answer_match = re.search(r'<answer>', text, re.IGNORECASE)
    if not answer_match:
        # No answer tag found, treat all as thinking
        return len(token_ids), len(token_ids)

    # Approximate token position from character position
    # This is imprecise but good enough for discount application
    char_pos = answer_match.start()

    # Decode incrementally to find token position
    cumulative_text = ""
    answer_token_start = len(token_ids)
    for i, tid in enumerate(token_ids):
        cumulative_text = tokenizer.detokenize(token_ids[:i+1])
        if len(cumulative_text) >= char_pos:
            answer_token_start = i
            break

    return answer_token_start, len(token_ids)


def update_policy_outcome_conditional(
    model,
    optimizer,
    episodes: List[Episode],
    micro_batch_size: int,
    pad_token_id: int,
    max_grad_norm: float,
    device: torch.device,
    dtype: torch.dtype,
    tokenizer,  # Required for section detection
    step: int = 0,
    log_tokens: bool = True,
    think_discount: float = 0.5,  # Discount factor for <think> section
):
    """
    GRPO update with Outcome-Conditional Advantage.

    Tokens in <think> section get discounted advantage (think_discount * advantage).
    Tokens in <answer> section get full advantage.

    Args:
        think_discount: Factor to multiply advantage for think tokens (default 0.5)
    """
    original_rewards = {id(ep): ep.reward for ep in episodes}
    episodes = normalize_rewards_per_group(episodes)
    episodes.sort(key=lambda x: len(x.prefix_token_ids) + len(x.generated_token_ids))
    num_micro_batches = math.ceil(len(episodes) / micro_batch_size)
    num_target_tokens = sum(len(episode.generated_token_ids) for episode in episodes)
    entropy = 0.0

    logger = get_credit_logger() if log_tokens else None
    episode_counter = 0

    for i in range(0, len(episodes), micro_batch_size):
        print(
            f"\r* Computing policy gradient (outcome-conditional): {i:>2d}/{len(episodes):>2d}",
            flush=True,
            end="",
        )
        j = min(i + micro_batch_size, len(episodes))
        batch_episodes = episodes[i:j]
        batch_lengths = [
            len(episode.prefix_token_ids) + len(episode.generated_token_ids)
            for episode in batch_episodes
        ]
        batch_max_length = max(batch_lengths)
        batch_token_ids = [
            episode.prefix_token_ids
            + episode.generated_token_ids
            + [pad_token_id] * (batch_max_length - batch_lengths[k])
            for k, episode in enumerate(batch_episodes)
        ]
        batch_masks = [
            [0] * len(episode.prefix_token_ids)
            + [1] * len(episode.generated_token_ids)
            + [0] * (batch_max_length - batch_lengths[k])
            for k, episode in enumerate(batch_episodes)
        ]
        batch_advantages = [episode.reward for episode in batch_episodes]
        batch_token_ids = torch.tensor(batch_token_ids, device=device, dtype=torch.long)
        batch_masks = torch.tensor(batch_masks, device=device, dtype=torch.bool)
        batch_advantages_base = torch.tensor(
            batch_advantages, device=device, dtype=torch.float32
        )

        # ============================================================
        # MITIGATION: Create per-token advantage weights
        # ============================================================
        # Shape: (batch_size, seq_len)
        advantage_weights = torch.ones(
            len(batch_episodes), batch_max_length - 1,
            device=device, dtype=torch.float32
        )

        for batch_idx, episode in enumerate(batch_episodes):
            prompt_len = len(episode.prefix_token_ids)
            gen_len = len(episode.generated_token_ids)

            # Find answer section boundary
            answer_start, _ = find_section_boundaries(
                episode.generated_token_ids, tokenizer
            )

            # Set weights: think tokens get discount, answer tokens get 1.0
            for pos in range(gen_len):
                # Position in the advantage_weights tensor (shifted by prompt)
                weight_pos = prompt_len + pos - 1  # -1 because of target shift
                if weight_pos >= 0 and weight_pos < batch_max_length - 1:
                    if pos < answer_start:
                        # In think section
                        advantage_weights[batch_idx, weight_pos] = think_discount
                    else:
                        # In answer section
                        advantage_weights[batch_idx, weight_pos] = 1.0

        with torch.autocast(device_type=device.type, dtype=dtype):
            input_token_ids = batch_token_ids[:, :-1]
            target_token_ids = batch_token_ids[:, 1:]
            target_masks = batch_masks[:, 1:]
            logits = model.forward(input_token_ids).float()

        log_probs = -torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_token_ids.reshape(-1),
            ignore_index=pad_token_id,
            reduction="none",
        ).reshape(input_token_ids.shape[0], -1)

        with torch.no_grad():
            token_entropy = compute_entropy(logits)
            entropy = entropy + (token_entropy * target_masks).sum() / num_target_tokens

        # Apply weighted advantages
        # effective_advantage[b, t] = base_advantage[b] * weight[b, t]
        effective_advantages = batch_advantages_base[:, None] * advantage_weights

        # Logging
        if log_tokens and logger is not None:
            with torch.no_grad():
                for batch_idx, episode in enumerate(batch_episodes):
                    gen_len = len(episode.generated_token_ids)
                    prompt_len = len(episode.prefix_token_ids)
                    start_idx = prompt_len - 1 if prompt_len > 0 else 0
                    end_idx = start_idx + gen_len
                    token_log_probs = log_probs[batch_idx, start_idx:end_idx]

                    # Log with effective (weighted) advantage
                    eff_adv = effective_advantages[batch_idx, start_idx:end_idx]

                    logger.log_tokens(
                        step=step,
                        episode_idx=episode_counter,
                        token_ids=episode.generated_token_ids,
                        log_probs=token_log_probs[:len(episode.generated_token_ids)],
                        advantage=batch_advantages_base[batch_idx].item(),  # Log original
                        reward=episode.reward_info.get('reward', episode.reward) if hasattr(episode, 'reward_info') else episode.reward,
                        normalized_reward=episode.reward,
                    )
                    episode_counter += 1

        # Compute loss with per-token weighted advantages
        obj = log_probs * effective_advantages
        obj = (obj * target_masks).sum() / num_target_tokens
        loss = -obj
        loss.backward()

    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=max_grad_norm
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    metrics = {
        "loss": loss.item(),
        "grad_norm": grad_norm.item(),
        "entropy": entropy.item(),
    }

    if log_tokens and logger is not None:
        logger.log_step_summary(step, episodes, metrics)

    return metrics


# ============================================================
# Mitigation 2: Inverse Log Prob Weighting
# ============================================================

def update_policy_inverse_logprob(
    model,
    optimizer,
    episodes: List[Episode],
    micro_batch_size: int,
    pad_token_id: int,
    max_grad_norm: float,
    device: torch.device,
    dtype: torch.dtype,
    step: int = 0,
    log_tokens: bool = True,
    epsilon: float = 0.1,  # Smoothing factor to avoid division by zero
):
    """
    GRPO update with Inverse Log Prob Weighting.

    Standard GRPO: loss = advantage * log_prob
    This mitigation: loss = advantage * log_prob * (1 / (|log_prob| + epsilon))
                         = advantage * sign(log_prob) * |log_prob| / (|log_prob| + epsilon)

    This counteracts the effect where low-confidence tokens get more gradient.

    Args:
        epsilon: Smoothing factor (default 0.1). Larger = more smoothing.
    """
    original_rewards = {id(ep): ep.reward for ep in episodes}
    episodes = normalize_rewards_per_group(episodes)
    episodes.sort(key=lambda x: len(x.prefix_token_ids) + len(x.generated_token_ids))
    num_micro_batches = math.ceil(len(episodes) / micro_batch_size)
    num_target_tokens = sum(len(episode.generated_token_ids) for episode in episodes)
    entropy = 0.0

    logger = get_credit_logger() if log_tokens else None
    episode_counter = 0

    for i in range(0, len(episodes), micro_batch_size):
        print(
            f"\r* Computing policy gradient (inverse-logprob): {i:>2d}/{len(episodes):>2d}",
            flush=True,
            end="",
        )
        j = min(i + micro_batch_size, len(episodes))
        batch_episodes = episodes[i:j]
        batch_lengths = [
            len(episode.prefix_token_ids) + len(episode.generated_token_ids)
            for episode in batch_episodes
        ]
        batch_max_length = max(batch_lengths)
        batch_token_ids = [
            episode.prefix_token_ids
            + episode.generated_token_ids
            + [pad_token_id] * (batch_max_length - batch_lengths[k])
            for k, episode in enumerate(batch_episodes)
        ]
        batch_masks = [
            [0] * len(episode.prefix_token_ids)
            + [1] * len(episode.generated_token_ids)
            + [0] * (batch_max_length - batch_lengths[k])
            for k, episode in enumerate(batch_episodes)
        ]
        batch_advantages = [episode.reward for episode in batch_episodes]
        batch_token_ids = torch.tensor(batch_token_ids, device=device, dtype=torch.long)
        batch_masks = torch.tensor(batch_masks, device=device, dtype=torch.bool)
        batch_advantages = torch.tensor(
            batch_advantages, device=device, dtype=torch.float32
        )

        with torch.autocast(device_type=device.type, dtype=dtype):
            input_token_ids = batch_token_ids[:, :-1]
            target_token_ids = batch_token_ids[:, 1:]
            target_masks = batch_masks[:, 1:]
            logits = model.forward(input_token_ids).float()

        log_probs = -torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_token_ids.reshape(-1),
            ignore_index=pad_token_id,
            reduction="none",
        ).reshape(input_token_ids.shape[0], -1)

        with torch.no_grad():
            token_entropy = compute_entropy(logits)
            entropy = entropy + (token_entropy * target_masks).sum() / num_target_tokens

        # ============================================================
        # MITIGATION: Inverse log prob weighting
        # ============================================================
        # weight = 1 / (|log_prob| + epsilon)
        # This makes all tokens contribute more equally regardless of confidence
        inverse_weights = 1.0 / (log_probs.abs() + epsilon)

        # Logging
        if log_tokens and logger is not None:
            with torch.no_grad():
                for batch_idx, episode in enumerate(batch_episodes):
                    gen_len = len(episode.generated_token_ids)
                    prompt_len = len(episode.prefix_token_ids)
                    start_idx = prompt_len - 1 if prompt_len > 0 else 0
                    end_idx = start_idx + gen_len
                    token_log_probs = log_probs[batch_idx, start_idx:end_idx]

                    logger.log_tokens(
                        step=step,
                        episode_idx=episode_counter,
                        token_ids=episode.generated_token_ids,
                        log_probs=token_log_probs[:len(episode.generated_token_ids)],
                        advantage=batch_advantages[batch_idx].item(),
                        reward=episode.reward_info.get('reward', episode.reward) if hasattr(episode, 'reward_info') else episode.reward,
                        normalized_reward=episode.reward,
                    )
                    episode_counter += 1

        # Compute weighted loss
        obj = log_probs * batch_advantages[:, None] * inverse_weights
        obj = (obj * target_masks).sum() / num_target_tokens
        loss = -obj
        loss.backward()

    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=max_grad_norm
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    metrics = {
        "loss": loss.item(),
        "grad_norm": grad_norm.item(),
        "entropy": entropy.item(),
    }

    if log_tokens and logger is not None:
        logger.log_step_summary(step, episodes, metrics)

    return metrics


# ============================================================
# Mitigation 3: Attention-Based Credit
# ============================================================

def compute_attention_credit(
    model,
    input_token_ids: torch.Tensor,
    target_masks: torch.Tensor,
    episodes: List[Episode],
    tokenizer,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Compute attention-based credit weights.

    For each token, compute how much the answer tokens attend to it.
    Tokens that the answer attends to more get higher credit.

    Returns:
        attention_credit: (batch_size, seq_len) tensor of credit weights
    """
    batch_size, seq_len = input_token_ids.shape
    attention_credit = torch.ones(batch_size, seq_len, device=device, dtype=torch.float32)

    # Get the last layer's attention module
    last_layer = model.layers[-1]
    attn = last_layer.self_attn

    with torch.no_grad():
        # Get embeddings
        h = model.embed_tokens(input_token_ids)
        pos = torch.arange(0, seq_len, device=input_token_ids.device, dtype=torch.int32)
        pos_emb = model.rotary_emb(h, pos[None, :])

        # Run through all layers except last to get pre-attention hidden states
        for layer in model.layers[:-1]:
            h = layer(h, pos_emb)

        # Get Q, K for last layer
        h_norm = last_layer.input_layernorm(h)
        xq = attn.q_proj(h_norm)
        xk = attn.k_proj(h_norm)

        xq = xq.view(batch_size, seq_len, attn.n_heads, attn.head_dim)
        xk = xk.view(batch_size, seq_len, attn.n_kv_heads, attn.head_dim)

        cos, sin = pos_emb
        from qwen2_model import apply_rotary_pos_emb
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin, unsqueeze_dim=2)

        # Expand KV heads if using GQA
        if attn.n_kv_heads < attn.n_heads:
            n_rep = attn.n_heads // attn.n_kv_heads
            xk = xk.unsqueeze(3).expand(-1, -1, -1, n_rep, -1).reshape(
                batch_size, seq_len, attn.n_heads, attn.head_dim
            )

        # Compute attention scores: (batch, heads, seq, seq)
        xq = xq.transpose(1, 2)  # (batch, heads, seq, dim)
        xk = xk.transpose(1, 2)  # (batch, heads, seq, dim)

        scale = 1.0 / math.sqrt(attn.head_dim)
        attn_scores = torch.matmul(xq, xk.transpose(-2, -1)) * scale

        # Apply causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1
        )
        attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)  # (batch, heads, seq, seq)

        # Average over heads
        attn_weights = attn_weights.mean(dim=1)  # (batch, seq, seq)

        # For each batch, find answer section and compute credit
        for b, episode in enumerate(episodes):
            prompt_len = len(episode.prefix_token_ids)
            gen_len = len(episode.generated_token_ids)

            # Find answer section
            answer_start, _ = find_section_boundaries(
                episode.generated_token_ids, tokenizer
            )
            answer_start_abs = prompt_len + answer_start

            if answer_start_abs < seq_len:
                # Credit = how much answer tokens attend to each position
                # Sum attention from all answer positions to each earlier position
                answer_positions = list(range(answer_start_abs, prompt_len + gen_len))
                if answer_positions:
                    # attention_credit[b, t] = sum of attention from answer positions to t
                    for ans_pos in answer_positions:
                        if ans_pos < seq_len:
                            attention_credit[b, :ans_pos] += attn_weights[b, ans_pos, :ans_pos]

                    # Normalize
                    attention_credit[b] = attention_credit[b] / (len(answer_positions) + 1)

    return attention_credit


def update_policy_attention_credit(
    model,
    optimizer,
    episodes: List[Episode],
    micro_batch_size: int,
    pad_token_id: int,
    max_grad_norm: float,
    device: torch.device,
    dtype: torch.dtype,
    tokenizer,  # Required for section detection
    step: int = 0,
    log_tokens: bool = True,
    credit_scale: float = 2.0,  # Scale factor for attention credit
):
    """
    GRPO update with Attention-Based Credit Assignment.

    Tokens are weighted by how much the answer section attends to them.
    Tokens the answer "looks at" more get more credit.

    Args:
        credit_scale: Scale factor for attention credit (default 2.0)
    """
    original_rewards = {id(ep): ep.reward for ep in episodes}
    episodes = normalize_rewards_per_group(episodes)
    episodes.sort(key=lambda x: len(x.prefix_token_ids) + len(x.generated_token_ids))
    num_micro_batches = math.ceil(len(episodes) / micro_batch_size)
    num_target_tokens = sum(len(episode.generated_token_ids) for episode in episodes)
    entropy = 0.0

    logger = get_credit_logger() if log_tokens else None
    episode_counter = 0

    for i in range(0, len(episodes), micro_batch_size):
        print(
            f"\r* Computing policy gradient (attention-credit): {i:>2d}/{len(episodes):>2d}",
            flush=True,
            end="",
        )
        j = min(i + micro_batch_size, len(episodes))
        batch_episodes = episodes[i:j]
        batch_lengths = [
            len(episode.prefix_token_ids) + len(episode.generated_token_ids)
            for episode in batch_episodes
        ]
        batch_max_length = max(batch_lengths)
        batch_token_ids = [
            episode.prefix_token_ids
            + episode.generated_token_ids
            + [pad_token_id] * (batch_max_length - batch_lengths[k])
            for k, episode in enumerate(batch_episodes)
        ]
        batch_masks = [
            [0] * len(episode.prefix_token_ids)
            + [1] * len(episode.generated_token_ids)
            + [0] * (batch_max_length - batch_lengths[k])
            for k, episode in enumerate(batch_episodes)
        ]
        batch_advantages = [episode.reward for episode in batch_episodes]
        batch_token_ids = torch.tensor(batch_token_ids, device=device, dtype=torch.long)
        batch_masks = torch.tensor(batch_masks, device=device, dtype=torch.bool)
        batch_advantages = torch.tensor(
            batch_advantages, device=device, dtype=torch.float32
        )

        with torch.autocast(device_type=device.type, dtype=dtype):
            input_token_ids = batch_token_ids[:, :-1]
            target_token_ids = batch_token_ids[:, 1:]
            target_masks = batch_masks[:, 1:]
            logits = model.forward(input_token_ids).float()

        log_probs = -torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_token_ids.reshape(-1),
            ignore_index=pad_token_id,
            reduction="none",
        ).reshape(input_token_ids.shape[0], -1)

        with torch.no_grad():
            token_entropy = compute_entropy(logits)
            entropy = entropy + (token_entropy * target_masks).sum() / num_target_tokens

        # ============================================================
        # MITIGATION: Attention-based credit
        # ============================================================
        with torch.no_grad():
            attention_credit = compute_attention_credit(
                model, input_token_ids, target_masks, batch_episodes,
                tokenizer, device, dtype
            )
            # Scale and shift to [0.5, 1.5] range approximately
            attention_credit = 0.5 + attention_credit * credit_scale
            attention_credit = attention_credit.clamp(0.1, 3.0)

        # Logging
        if log_tokens and logger is not None:
            with torch.no_grad():
                for batch_idx, episode in enumerate(batch_episodes):
                    gen_len = len(episode.generated_token_ids)
                    prompt_len = len(episode.prefix_token_ids)
                    start_idx = prompt_len - 1 if prompt_len > 0 else 0
                    end_idx = start_idx + gen_len
                    token_log_probs = log_probs[batch_idx, start_idx:end_idx]

                    logger.log_tokens(
                        step=step,
                        episode_idx=episode_counter,
                        token_ids=episode.generated_token_ids,
                        log_probs=token_log_probs[:len(episode.generated_token_ids)],
                        advantage=batch_advantages[batch_idx].item(),
                        reward=episode.reward_info.get('reward', episode.reward) if hasattr(episode, 'reward_info') else episode.reward,
                        normalized_reward=episode.reward,
                    )
                    episode_counter += 1

        # Compute weighted loss
        obj = log_probs * batch_advantages[:, None] * attention_credit
        obj = (obj * target_masks).sum() / num_target_tokens
        loss = -obj
        loss.backward()

    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=max_grad_norm
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    metrics = {
        "loss": loss.item(),
        "grad_norm": grad_norm.item(),
        "entropy": entropy.item(),
    }

    if log_tokens and logger is not None:
        logger.log_step_summary(step, episodes, metrics)

    return metrics


# ============================================================
# Mitigation 4: Entropy-Based Masking (from paper 2506.01939)
# ============================================================

def update_policy_entropy_masking(
    model,
    optimizer,
    episodes: List[Episode],
    micro_batch_size: int,
    pad_token_id: int,
    max_grad_norm: float,
    device: torch.device,
    dtype: torch.dtype,
    step: int = 0,
    log_tokens: bool = True,
    mask_percentile: float = 80.0,  # Mask bottom X% entropy tokens (paper uses 80%)
):
    """
    GRPO update with Entropy-Based Masking.

    Based on paper "Beyond the 80/20 Rule" (2506.01939):
    - High-entropy tokens are "forking" tokens (decision points)
    - Low-entropy tokens are "following" tokens (deterministic continuations)
    - Only top 20% high-entropy tokens drive RLVR learning

    This implementation masks the bottom mask_percentile% of tokens by entropy,
    focusing gradient signal on the high-entropy forking tokens.

    Args:
        mask_percentile: Percentage of low-entropy tokens to mask (default 80.0)
    """
    original_rewards = {id(ep): ep.reward for ep in episodes}
    episodes = normalize_rewards_per_group(episodes)
    episodes.sort(key=lambda x: len(x.prefix_token_ids) + len(x.generated_token_ids))
    num_micro_batches = math.ceil(len(episodes) / micro_batch_size)
    num_target_tokens = sum(len(episode.generated_token_ids) for episode in episodes)
    entropy_total = 0.0

    # Track stats for logging
    total_tokens = 0
    masked_tokens = 0

    logger = get_credit_logger() if log_tokens else None
    episode_counter = 0

    for i in range(0, len(episodes), micro_batch_size):
        print(
            f"\r* Computing policy gradient (entropy-masking): {i:>2d}/{len(episodes):>2d}",
            flush=True,
            end="",
        )
        j = min(i + micro_batch_size, len(episodes))
        batch_episodes = episodes[i:j]
        batch_lengths = [
            len(episode.prefix_token_ids) + len(episode.generated_token_ids)
            for episode in batch_episodes
        ]
        batch_max_length = max(batch_lengths)
        batch_token_ids = [
            episode.prefix_token_ids
            + episode.generated_token_ids
            + [pad_token_id] * (batch_max_length - batch_lengths[k])
            for k, episode in enumerate(batch_episodes)
        ]
        batch_masks = [
            [0] * len(episode.prefix_token_ids)
            + [1] * len(episode.generated_token_ids)
            + [0] * (batch_max_length - batch_lengths[k])
            for k, episode in enumerate(batch_episodes)
        ]
        batch_advantages = [episode.reward for episode in batch_episodes]
        batch_token_ids = torch.tensor(batch_token_ids, device=device, dtype=torch.long)
        batch_masks = torch.tensor(batch_masks, device=device, dtype=torch.bool)
        batch_advantages = torch.tensor(
            batch_advantages, device=device, dtype=torch.float32
        )

        with torch.autocast(device_type=device.type, dtype=dtype):
            input_token_ids = batch_token_ids[:, :-1]
            target_token_ids = batch_token_ids[:, 1:]
            target_masks = batch_masks[:, 1:]
            logits = model.forward(input_token_ids).float()

        log_probs = -torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_token_ids.reshape(-1),
            ignore_index=pad_token_id,
            reduction="none",
        ).reshape(input_token_ids.shape[0], -1)

        with torch.no_grad():
            token_entropy = compute_entropy(logits)
            entropy_total = entropy_total + (token_entropy * target_masks).sum() / num_target_tokens

        # ============================================================
        # MITIGATION: Entropy-based masking
        # ============================================================
        with torch.no_grad():
            # Get valid entropy values (where target_masks is True)
            valid_entropies = token_entropy[target_masks]

            if len(valid_entropies) > 0:
                # Find entropy threshold for masking
                threshold = torch.quantile(valid_entropies, mask_percentile / 100.0)

                # Create entropy mask: 1 for high-entropy (keep), 0 for low-entropy (mask)
                entropy_mask = (token_entropy >= threshold).float()

                # Combine with target_masks
                combined_mask = target_masks.float() * entropy_mask

                # Track stats
                total_tokens += target_masks.sum().item()
                masked_tokens += (target_masks.float() * (1 - entropy_mask)).sum().item()
            else:
                combined_mask = target_masks.float()

        # Logging
        if log_tokens and logger is not None:
            with torch.no_grad():
                for batch_idx, episode in enumerate(batch_episodes):
                    gen_len = len(episode.generated_token_ids)
                    prompt_len = len(episode.prefix_token_ids)
                    start_idx = prompt_len - 1 if prompt_len > 0 else 0
                    end_idx = start_idx + gen_len
                    token_log_probs = log_probs[batch_idx, start_idx:end_idx]
                    token_entropies = token_entropy[batch_idx, start_idx:end_idx]

                    logger.log_tokens(
                        step=step,
                        episode_idx=episode_counter,
                        token_ids=episode.generated_token_ids,
                        log_probs=token_log_probs[:len(episode.generated_token_ids)],
                        advantage=batch_advantages[batch_idx].item(),
                        reward=episode.reward_info.get('reward', episode.reward) if hasattr(episode, 'reward_info') else episode.reward,
                        normalized_reward=episode.reward,
                        entropies=token_entropies[:len(episode.generated_token_ids)],
                    )
                    episode_counter += 1

        # Compute loss with entropy mask applied
        # Only high-entropy tokens contribute to gradient
        obj = log_probs * batch_advantages[:, None]
        obj = (obj * combined_mask).sum() / max(combined_mask.sum(), 1.0)
        loss = -obj
        loss.backward()

    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=max_grad_norm
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    # Calculate mask ratio
    mask_ratio = masked_tokens / total_tokens if total_tokens > 0 else 0.0

    metrics = {
        "loss": loss.item(),
        "grad_norm": grad_norm.item(),
        "entropy": entropy_total.item(),
        "entropy_mask_ratio": mask_ratio,  # How many tokens were masked
    }

    if log_tokens and logger is not None:
        logger.log_step_summary(step, episodes, metrics)

    return metrics


# ============================================================
# Factory function to select mitigation
# ============================================================

def get_update_policy_fn(mitigation: str = "none"):
    """
    Get the appropriate update_policy function for the given mitigation.

    Args:
        mitigation: One of:
            - "none": Standard GRPO (from grpo_instrumented)
            - "outcome_conditional": Outcome-Conditional Advantage
            - "inverse_logprob": Inverse Log Prob Weighting
            - "attention_credit": Attention-Based Credit
            - "entropy_masking": Entropy-Based Masking (paper 2506.01939)

    Returns:
        update_policy function
    """
    from grpo_instrumented import update_policy as update_policy_standard

    mitigations = {
        "none": update_policy_standard,
        "standard": update_policy_standard,
        "outcome_conditional": update_policy_outcome_conditional,
        "inverse_logprob": update_policy_inverse_logprob,
        "attention_credit": update_policy_attention_credit,
        "entropy_masking": update_policy_entropy_masking,
    }

    if mitigation not in mitigations:
        raise ValueError(
            f"Unknown mitigation: {mitigation}. "
            f"Choose from: {list(mitigations.keys())}"
        )

    return mitigations[mitigation]
