"""
Instrumented GRPO implementation with per-token credit assignment logging.

This module extends the base GRPO implementation to log:
- Per-token log probabilities
- Per-token advantages (broadcast from episode-level)
- Per-token policy gradient contributions: advantage * log_prob
- Token IDs and positions for semantic analysis
"""

import dataclasses
import gc
import math
from collections import defaultdict
from typing import Callable, List, Optional
from dataclasses import dataclass, field

import numpy as np
import torch

from data_types import Episode, MiniBatch
from qwen2_model import Transformer
from tokenizer import Tokenizer


@dataclass
class TokenGradientLog:
    """Log entry for a single token's gradient contribution."""
    step: int
    episode_idx: int
    token_position: int  # Position within generated tokens (0-indexed)
    token_id: int
    log_prob: float
    advantage: float  # Episode-level advantage (same for all tokens in episode)
    contribution: float  # advantage * log_prob
    reward: float  # Original reward before normalization
    normalized_reward: float  # After group normalization
    entropy: float = 0.0  # Token entropy at generation time (for forking/following analysis)


@dataclass
class CreditAssignmentLogger:
    """Accumulates per-token gradient information across training."""
    logs: List[TokenGradientLog] = field(default_factory=list)
    step_summaries: List[dict] = field(default_factory=list)

    def log_tokens(
        self,
        step: int,
        episode_idx: int,
        token_ids: List[int],
        log_probs: torch.Tensor,  # (seq_len,)
        advantage: float,
        reward: float,
        normalized_reward: float,
        entropies: Optional[torch.Tensor] = None,  # (seq_len,) - per-token entropy
    ):
        """Log gradient info for all tokens in an episode."""
        log_probs_np = log_probs.detach().cpu().numpy()
        entropies_np = entropies.detach().cpu().numpy() if entropies is not None else None

        for pos, (token_id, log_prob) in enumerate(zip(token_ids, log_probs_np)):
            contribution = advantage * log_prob
            entropy_val = float(entropies_np[pos]) if entropies_np is not None and pos < len(entropies_np) else 0.0
            self.logs.append(TokenGradientLog(
                step=step,
                episode_idx=episode_idx,
                token_position=pos,
                token_id=int(token_id),
                log_prob=float(log_prob),
                advantage=float(advantage),
                contribution=float(contribution),
                reward=float(reward),
                normalized_reward=float(normalized_reward),
                entropy=entropy_val,
            ))

    def log_step_summary(self, step: int, episodes: List[Episode], metrics: dict):
        """Log summary statistics for a training step."""
        summary = {
            'step': step,
            'num_episodes': len(episodes),
            'mean_reward': np.mean([e.reward for e in episodes]),
            'std_reward': np.std([e.reward for e in episodes]),
            'mean_response_len': np.mean([len(e.generated_token_ids) for e in episodes]),
            **metrics
        }
        self.step_summaries.append(summary)

    def save(self, path: str):
        """Save logs to file."""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                'logs': self.logs,
                'step_summaries': self.step_summaries,
            }, f)
        print(f"Saved {len(self.logs)} token logs to {path}")

    @classmethod
    def load(cls, path: str) -> 'CreditAssignmentLogger':
        """Load logs from file."""
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
        logger = cls()
        logger.logs = data['logs']
        logger.step_summaries = data['step_summaries']
        return logger

    def to_dataframe(self):
        """Convert logs to pandas DataFrame for analysis."""
        import pandas as pd
        return pd.DataFrame([dataclasses.asdict(log) for log in self.logs])


# Global logger instance
_credit_logger: Optional[CreditAssignmentLogger] = None


def get_credit_logger() -> CreditAssignmentLogger:
    """Get the global credit assignment logger."""
    global _credit_logger
    if _credit_logger is None:
        _credit_logger = CreditAssignmentLogger()
    return _credit_logger


def reset_credit_logger():
    """Reset the global logger."""
    global _credit_logger
    _credit_logger = CreditAssignmentLogger()
    return _credit_logger


# ============================================================
# Original GRPO functions (unchanged)
# ============================================================

@torch.no_grad()
def rollout(
    model: Transformer,
    batch: MiniBatch,
    tokenizer: Tokenizer,
    max_gen_len: int,
    num_answer_per_question: int,
    reward_function: Callable,
    device: torch.device,
    dtype: torch.dtype,
) -> List[Episode]:
    """Generate rollouts (unchanged from original)."""
    end_token = tokenizer.eos_token
    end_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    prefix_token_ids = batch.prefix_token_ids
    bsz = len(batch.prefix) * num_answer_per_question
    min_prompt_len = min(len(t) for t in prefix_token_ids)
    max_prompt_len = max(len(t) for t in prefix_token_ids)
    total_len = max_gen_len + max_prompt_len
    model.init_kv_cache(
        max_batch_size=bsz,
        max_seq_len=total_len,
        device=device,
        dtype=dtype,
    )
    tokens = torch.full((bsz, total_len), pad_token_id, dtype=torch.long, device=device)
    for k, t in enumerate(prefix_token_ids):
        offset = k * num_answer_per_question
        for i in range(num_answer_per_question):
            tokens[offset + i, : len(t)] = torch.tensor(
                t, dtype=torch.long, device=device
            )

    prev_pos = 0
    input_text_mask = tokens != pad_token_id
    assert min_prompt_len < total_len
    is_finished = torch.zeros((bsz,), dtype=torch.bool, device=device)

    for cur_pos in range(min_prompt_len, total_len):
        print(
            f"\r* Generating trajectories: {cur_pos-min_prompt_len:>4d}/{total_len-min_prompt_len:>4d}",
            flush=True,
            end="",
        )
        with torch.autocast(device_type=device.type, dtype=dtype):
            logits = model.inference(tokens[:, prev_pos:cur_pos], prev_pos)
        probs = torch.softmax(logits[:, -1], dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        next_token = next_token.reshape(-1)
        next_token = torch.where(
            input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
        )
        next_token = torch.where(is_finished, pad_token_id, next_token)
        tokens[:, cur_pos] = next_token
        if end_token_id is not None:
            is_end_token = next_token == end_token_id
            is_generated_token = ~input_text_mask[:, cur_pos]
            is_finished = is_finished | (is_end_token & is_generated_token)
        prev_pos = cur_pos
        if is_finished.all():
            break
    model.del_kv_cache()
    gc.collect()
    torch.cuda.empty_cache()
    is_finished_list = is_finished.tolist()
    tokens_list = tokens.tolist()

    episodes = []
    for i in range(bsz // num_answer_per_question):
        for j in range(num_answer_per_question):
            idx = i * num_answer_per_question + j
            generated_token_ids = tokens_list[idx][len(batch.prefix_token_ids[i]) :]
            if pad_token_id in generated_token_ids:
                generated_token_ids = generated_token_ids[
                    : generated_token_ids.index(pad_token_id)
                ]
            generated_text = tokenizer.detokenize(generated_token_ids)
            rewards = reward_function(
                response=generated_text,
                numbers=batch.numbers[i],
                target=batch.target[i],
                end_token=end_token,
            )
            episode = Episode(
                prefix=batch.prefix[i],
                text=batch.prefix[i] + generated_text,
                prefix_token_ids=batch.prefix_token_ids[i],
                prefix_tokens=batch.prefix_tokens[i],
                generated_token_ids=generated_token_ids,
                is_finished=is_finished_list[idx],
                reward=rewards["reward"],
                reward_info=rewards["reward_info"],
            )
            episodes.append(episode)
    print("\r", end=" " * 100, flush=True)
    return episodes


def normalize_rewards_per_group(episodes: List[Episode]) -> List[Episode]:
    """Normalize rewards per group (unchanged from original)."""
    groups = defaultdict(list)
    for episode in episodes:
        groups[tuple(episode.prefix)].append(episode)
    output = []
    for group in groups.values():
        group_rewards = [item.reward for item in group]
        mean_reward = np.mean(group_rewards)
        std_reward = np.std(group_rewards)
        for episode in group:
            normalized_reward = (episode.reward - mean_reward) / (std_reward + 1e-4)
            episode = dataclasses.replace(episode, reward=normalized_reward)
            output.append(episode)
    return output


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Compute entropy (unchanged from original)."""
    probs = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(probs * logits, dim=-1)
    return entropy


# ============================================================
# INSTRUMENTED update_policy with per-token logging
# ============================================================

def update_policy(
    model,
    optimizer,
    episodes: List[Episode],
    micro_batch_size: int,
    pad_token_id: int,
    max_grad_norm: float,
    device: torch.device,
    dtype: torch.dtype,
    step: int = 0,  # Added: current training step
    log_tokens: bool = True,  # Added: whether to log per-token info
):
    """
    Update the policy using the GRPO algorithm.

    INSTRUMENTED VERSION: Logs per-token gradient contributions.
    """
    # Store original rewards before normalization
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
            f"\r* Computing policy gradient: {i:>2d}/{len(episodes):>2d}",
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
        batch_advantages = [episode.reward for episode in batch_episodes]  # normalized reward = advantage
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

        # Compute per-token log probabilities
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
        # INSTRUMENTATION: Log per-token contributions with entropy
        # ============================================================
        if log_tokens and logger is not None:
            with torch.no_grad():
                # Per-token contribution before masking
                per_token_obj = log_probs * batch_advantages[:, None]

                for batch_idx, episode in enumerate(batch_episodes):
                    # Get the generated token portion only
                    gen_len = len(episode.generated_token_ids)
                    prompt_len = len(episode.prefix_token_ids)

                    # Extract log_probs for generated tokens only
                    # Note: log_probs are for predicting token at position t from position t-1
                    # So for generated tokens starting at prompt_len, their log_probs are at indices [prompt_len-1:]
                    start_idx = prompt_len - 1 if prompt_len > 0 else 0
                    end_idx = start_idx + gen_len

                    token_log_probs = log_probs[batch_idx, start_idx:end_idx]
                    # Extract entropy for generated tokens (same indexing as log_probs)
                    token_entropies = token_entropy[batch_idx, start_idx:end_idx]

                    # Find original reward (before normalization)
                    # We need to match by episode content since sorting changed order
                    original_reward = 0.0  # Default
                    for orig_ep_id, orig_reward in original_rewards.items():
                        # Match by generated tokens
                        pass  # Skip for now, just use normalized

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
        # ============================================================

        obj = log_probs * batch_advantages[:, None]
        obj = (obj * target_masks).sum() / num_target_tokens
        loss = -obj
        loss.backward()

    # Update the policy
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

    # Log step summary
    if log_tokens and logger is not None:
        logger.log_step_summary(step, episodes, metrics)

    return metrics
