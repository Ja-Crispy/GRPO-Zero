"""
Training script for GRPO with credit assignment mitigations.

Supports:
- none/standard: Original GRPO (baseline)
- outcome_conditional: Discount advantage for <think> section
- inverse_logprob: Weight by 1/(|log_prob| + epsilon)
- attention_credit: Weight by attention from answer tokens

Usage:
    python train_mitigations.py --config config_instrumented.yaml --mitigation outcome_conditional
    python train_mitigations.py --config config_instrumented.yaml --mitigation inverse_logprob --epsilon 0.1
    python train_mitigations.py --config config_instrumented.yaml --mitigation attention_credit
"""

import html
import time
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from countdown_task import CountdownTasksDataset, reward_function
from grpo_instrumented import rollout, get_credit_logger, reset_credit_logger
from grpo_mitigations import get_update_policy_fn
from optimizer import MemoryEfficientAdamW
from qwen2_model import Transformer
from tokenizer import Tokenizer


def evaluate(model, tokenizer, device, dtype, config):
    test_dataset = CountdownTasksDataset(
        data_path=config["data"]["path"],
        tokenizer=tokenizer,
        split="test",
        test_size=config["data"]["test_size"],
    )
    generator = torch.Generator(device=device)
    dataloader = DataLoader(
        test_dataset,
        shuffle=False,
        collate_fn=CountdownTasksDataset.collate_fn,
        generator=generator,
        batch_size=config["training"]["batch_size"] // 2,
        drop_last=False,
    )
    success = []
    for batch in dataloader:
        episodes = rollout(
            model=model,
            tokenizer=tokenizer,
            batch=batch,
            max_gen_len=config["training"]["max_gen_len"] * 2,
            num_answer_per_question=1,
            reward_function=reward_function,
            device=device,
            dtype=dtype,
        )
        success.extend([episode.reward_info["answer_reward"] for episode in episodes])
    return np.mean(success)


def main(args):
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Mitigation-specific parameters
    mitigation = args.mitigation
    mitigation_params = {
        "outcome_conditional": {
            "think_discount": args.think_discount,
        },
        "inverse_logprob": {
            "epsilon": args.epsilon,
        },
        "attention_credit": {
            "credit_scale": args.credit_scale,
        },
        "entropy_weighted": {
            "entropy_temp": args.entropy_temp,
        },
    }

    pretrained_model_path = Path(config["model"]["pretrained_model_path"])
    device = torch.device(config["model"]["device"])
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(config["model"]["dtype"], torch.bfloat16)
    torch.set_default_device(device)
    torch.random.manual_seed(config["training"]["random_seed"])
    BATCH_SIZE = config["training"]["batch_size"]
    NUM_QUESTIONS_PER_BATCH = config["training"]["num_questions_per_batch"]
    NUM_ANSWERS_PER_QUESTION = BATCH_SIZE // NUM_QUESTIONS_PER_BATCH

    # Create log directory with mitigation name
    current_time = datetime.now().strftime(r"%Y%m%d-%H%M%S")
    log_suffix = f"_{mitigation}" if mitigation != "none" else ""
    tb_writer = SummaryWriter(log_dir=f"{config['training']['log_dir']}{log_suffix}/{current_time}")

    tokenizer = Tokenizer(str(pretrained_model_path / "tokenizer.json"))

    train_dataset = CountdownTasksDataset(
        data_path=config["data"]["path"],
        tokenizer=tokenizer,
        split="train",
        test_size=config["data"]["test_size"],
    )
    generator = torch.Generator(device=device)
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=CountdownTasksDataset.collate_fn,
        generator=generator,
        batch_size=NUM_QUESTIONS_PER_BATCH,
    )

    model = Transformer.from_pretrained(pretrained_model_path, device=device).train()

    optimizer = MemoryEfficientAdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        betas=config["training"]["betas"],
        enabled=config["training"]["memory_efficient_adamw"],
    )

    # Get the appropriate update_policy function
    update_policy = get_update_policy_fn(mitigation)

    # Initialize credit assignment logger
    credit_logger = reset_credit_logger()
    credit_log_dir = Path(config["training"].get("credit_log_dir", "credit_logs") + log_suffix)
    credit_log_dir.mkdir(parents=True, exist_ok=True)
    credit_save_interval = config["training"].get("credit_save_interval", 10)
    max_steps = config["training"].get("max_steps", None)

    start_time = time.time()
    ckpt_dir = Path(config["training"]["ckpt_dir"] + log_suffix)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"=" * 60)
    print(f"GRPO Training with Mitigation: {mitigation.upper()}")
    print(f"=" * 60)
    if mitigation in mitigation_params:
        print(f"Mitigation parameters: {mitigation_params[mitigation]}")
    print(f"Run ID: {current_time}")
    print(f"Credit logs: {credit_log_dir}")
    print(f"Checkpoints: {ckpt_dir}")
    print(f"Model: {pretrained_model_path}")
    print(f"Batch size: {BATCH_SIZE}, Questions per batch: {NUM_QUESTIONS_PER_BATCH}")
    print("-" * 60)

    for step, batch in enumerate(train_dataloader, start=1):
        episodes = rollout(
            model=model,
            tokenizer=tokenizer,
            batch=batch,
            max_gen_len=config["training"]["max_gen_len"],
            num_answer_per_question=NUM_ANSWERS_PER_QUESTION,
            reward_function=reward_function,
            device=device,
            dtype=dtype,
        )
        if config["training"]["skip_unfinished_episodes"]:
            episodes = [episode for episode in episodes if episode.is_finished]

        # Build kwargs for update_policy based on mitigation
        update_kwargs = {
            "model": model,
            "optimizer": optimizer,
            "episodes": episodes,
            "micro_batch_size": config["training"]["micro_batch_size"],
            "pad_token_id": tokenizer.pad_token_id,
            "max_grad_norm": config["training"]["max_grad_norm"],
            "device": device,
            "dtype": dtype,
            "step": step,
            "log_tokens": True,
        }

        # Add mitigation-specific parameters
        if mitigation == "outcome_conditional":
            update_kwargs["tokenizer"] = tokenizer
            update_kwargs["think_discount"] = args.think_discount
        elif mitigation == "inverse_logprob":
            update_kwargs["epsilon"] = args.epsilon
        elif mitigation == "attention_credit":
            update_kwargs["tokenizer"] = tokenizer
            update_kwargs["credit_scale"] = args.credit_scale
        elif mitigation == "entropy_weighted":
            update_kwargs["entropy_temp"] = args.entropy_temp

        results = update_policy(**update_kwargs)
        torch.cuda.synchronize()
        end_time = time.time()
        duration = end_time - start_time
        start_time = end_time

        # Compute and log metrics
        reward = [episode.reward for episode in episodes]
        formatted_reward = [
            episode.reward_info["format_reward"] for episode in episodes
        ]
        answer_reward = [episode.reward_info["answer_reward"] for episode in episodes]
        num_finished_episodes = sum(episode.is_finished for episode in episodes)
        mean_reward = np.mean(reward)
        std_reward = np.std(reward)
        success_rate = np.mean(answer_reward)
        format_reward = np.mean(formatted_reward)
        grad_norm = results["grad_norm"]
        entropy = results["entropy"]
        lr = optimizer.param_groups[0]["lr"]
        loss = results["loss"]
        mean_response_len = np.mean(
            [len(episode.generated_token_ids) for episode in episodes]
        )

        logger = get_credit_logger()
        num_tokens_logged = len(logger.logs)

        print(
            f"\rStep {step} [{mitigation}], reward: {mean_reward:.2f}, "
            f"success: {success_rate:.2f}, "
            f"grad_norm: {grad_norm:.2f}, duration: {duration:.2f}, "
            f"len: {mean_response_len:.0f}"
        )

        if step % config["training"]["eval_interval"] == 0:
            eval_success_rate = evaluate(model, tokenizer, device, dtype, config)
            print(f"\rEval success rate: {eval_success_rate:.2f}" + " " * 100)
            tb_writer.add_scalar("success_rate/eval", eval_success_rate, step)

        # TensorBoard logging
        tb_writer.add_scalar("loss", loss, step)
        tb_writer.add_scalar("mean_reward", mean_reward, step)
        tb_writer.add_scalar("std_reward", std_reward, step)
        tb_writer.add_scalar("success_rate/train", success_rate, step)
        tb_writer.add_scalar("format_reward", format_reward, step)
        tb_writer.add_scalar("grad_norm", grad_norm, step)
        tb_writer.add_scalar("duration", duration, step)
        tb_writer.add_scalar("num_finished_episodes", num_finished_episodes, step)
        tb_writer.add_scalar("learning_rate", lr, step)
        tb_writer.add_scalar("mean_response_len", mean_response_len, step)
        tb_writer.add_scalar("entropy", entropy, step)
        tb_writer.add_scalar("credit/num_tokens_logged", num_tokens_logged, step)

        for i, episode in enumerate(episodes[:4]):
            text = html.escape(episode.text)
            tb_writer.add_text(f"text_{i}", f"<pre>{text}</pre>", step)

        # Save credit logs periodically
        if step % credit_save_interval == 0:
            log_path = credit_log_dir / f"credit_logs_step_{step:06d}.pkl"
            logger.save(str(log_path))
            print(f"Saved credit logs to {log_path}")

        # Save checkpoint
        if step % config["training"]["ckpt_save_interval"] == 0:
            output_file = ckpt_dir / f"ckpt_{step:06d}.pt"
            torch.save(model.state_dict(), output_file)
            print(f"Saved checkpoint to {output_file}")

        # Check max steps
        if max_steps and step >= max_steps:
            print(f"Reached max_steps ({max_steps}), stopping training")
            break

    # Final save
    final_log_path = credit_log_dir / f"credit_logs_final.pkl"
    logger.save(str(final_log_path))
    print(f"\n{'=' * 60}")
    print(f"Training complete!")
    print(f"Mitigation: {mitigation}")
    print(f"Final credit logs: {final_log_path}")
    print(f"Total tokens logged: {len(logger.logs)}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config_instrumented.yaml",
                        help="Path to config file")
    parser.add_argument("--mitigation", type=str, default="none",
                        choices=["none", "standard", "outcome_conditional", "inverse_logprob", "attention_credit", "entropy_weighted"],
                        help="Credit assignment mitigation to use")

    # Outcome-conditional parameters
    parser.add_argument("--think_discount", type=float, default=0.5,
                        help="Discount factor for think section (outcome_conditional)")

    # Inverse log prob parameters
    parser.add_argument("--epsilon", type=float, default=0.1,
                        help="Smoothing factor (inverse_logprob)")

    # Attention credit parameters
    parser.add_argument("--credit_scale", type=float, default=2.0,
                        help="Scale factor for attention credit (attention_credit)")

    # Entropy weighting parameters
    parser.add_argument("--entropy_temp", type=float, default=1.0,
                        help="Temperature for entropy weighting - lower=more focus on high-entropy (entropy_weighted)")

    args = parser.parse_args()
    main(args)
