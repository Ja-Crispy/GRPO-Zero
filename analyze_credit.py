"""
Credit Assignment Analysis Script

Analyzes the per-token gradient contributions logged during GRPO training.
This is the CORRECT approach - measuring advantage × log_prob per token.
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from grpo_instrumented import CreditAssignmentLogger


def load_logs(log_path: str) -> CreditAssignmentLogger:
    """Load credit assignment logs from file."""
    return CreditAssignmentLogger.load(log_path)


def classify_token(token_text: str) -> str:
    """
    Classify token as 'reasoning', 'filler', or 'structure'.

    Categories:
    - reasoning: numbers, operators, mathematical expressions
    - structure: tags like <think>, </think>, <answer>, etc.
    - filler: everything else (connecting words, etc.)
    """
    token_text = token_text.strip()

    # Structure tokens (tags)
    if re.match(r'^</?(?:think|answer|THINK|ANSWER)>$', token_text):
        return 'structure'

    # Mathematical tokens
    if re.match(r'^[\d+\-*/()=]+$', token_text):
        return 'reasoning'

    # Numbers (including as part of larger token)
    if re.search(r'\d', token_text):
        return 'reasoning'

    # Operators
    if token_text in ['+', '-', '*', '/', '=', '(', ')', 'equals', 'plus', 'minus', 'times', 'divided']:
        return 'reasoning'

    # Filler words
    return 'filler'


def analyze_by_token_type(df: pd.DataFrame, tokenizer=None) -> Dict:
    """Analyze credit assignment by token type."""

    # If tokenizer provided, decode tokens
    if tokenizer is not None:
        df['token_text'] = df['token_id'].apply(lambda x: tokenizer.detokenize([x]))
        df['token_type'] = df['token_text'].apply(classify_token)
    else:
        # Without tokenizer, classify by token_id patterns (less accurate)
        df['token_type'] = 'unknown'

    results = {}
    for token_type in df['token_type'].unique():
        subset = df[df['token_type'] == token_type]
        results[token_type] = {
            'count': len(subset),
            'mean_contribution': subset['contribution'].mean(),
            'std_contribution': subset['contribution'].std(),
            'mean_abs_contribution': subset['contribution'].abs().mean(),
            'mean_log_prob': subset['log_prob'].mean(),
            'mean_advantage': subset['advantage'].mean(),
        }

    return results


def analyze_by_position(df: pd.DataFrame, num_bins: int = 10) -> pd.DataFrame:
    """Analyze credit assignment by relative position in sequence."""

    # Group by episode to get max position per episode
    episode_lengths = df.groupby(['step', 'episode_idx'])['token_position'].max() + 1

    # Merge back to get relative position
    df_with_len = df.merge(
        episode_lengths.reset_index().rename(columns={'token_position': 'episode_length'}),
        on=['step', 'episode_idx']
    )
    df_with_len['relative_position'] = df_with_len['token_position'] / df_with_len['episode_length']
    df_with_len['position_bin'] = pd.cut(df_with_len['relative_position'], bins=num_bins, labels=False)

    # Aggregate by position bin
    position_stats = df_with_len.groupby('position_bin').agg({
        'contribution': ['mean', 'std', lambda x: x.abs().mean()],
        'log_prob': 'mean',
        'advantage': 'mean',
        'token_position': 'count'
    }).round(6)

    position_stats.columns = ['mean_contribution', 'std_contribution', 'mean_abs_contribution',
                              'mean_log_prob', 'mean_advantage', 'count']

    return position_stats


def analyze_by_reward(df: pd.DataFrame) -> Dict:
    """Compare credit assignment for correct vs incorrect episodes."""

    # Episodes with positive reward vs negative
    positive_episodes = df[df['normalized_reward'] > 0]
    negative_episodes = df[df['normalized_reward'] <= 0]

    results = {
        'positive_reward': {
            'num_episodes': positive_episodes[['step', 'episode_idx']].drop_duplicates().shape[0],
            'num_tokens': len(positive_episodes),
            'mean_contribution': positive_episodes['contribution'].mean(),
            'mean_abs_contribution': positive_episodes['contribution'].abs().mean(),
            'mean_log_prob': positive_episodes['log_prob'].mean(),
        },
        'negative_reward': {
            'num_episodes': negative_episodes[['step', 'episode_idx']].drop_duplicates().shape[0],
            'num_tokens': len(negative_episodes),
            'mean_contribution': negative_episodes['contribution'].mean(),
            'mean_abs_contribution': negative_episodes['contribution'].abs().mean(),
            'mean_log_prob': negative_episodes['log_prob'].mean(),
        }
    }

    return results


def analyze_contribution_consistency(df: pd.DataFrame) -> Dict:
    """
    Analyze if tokens get consistent credit direction across episodes.

    This is what we SHOULD have measured before:
    - t-statistic of CONTRIBUTION (advantage × log_prob), not raw gradient
    """

    # Group by token position and compute t-statistic
    position_groups = df.groupby('token_position')

    t_stats = []
    for pos, group in position_groups:
        if len(group) > 1:
            contributions = group['contribution'].values
            if contributions.std() > 0:
                t_stat = contributions.mean() / (contributions.std() / np.sqrt(len(contributions)))
            else:
                t_stat = 0
            t_stats.append({
                'position': pos,
                't_stat': t_stat,
                'mean_contribution': contributions.mean(),
                'count': len(group)
            })

    t_df = pd.DataFrame(t_stats)

    results = {
        'high_t_stat_ratio': (t_df['t_stat'].abs() > 3.0).mean(),
        'low_t_stat_ratio': (t_df['t_stat'].abs() < 1.5).mean(),
        'mean_abs_t_stat': t_df['t_stat'].abs().mean(),
        'positions_analyzed': len(t_df),
    }

    return results


def plot_results(df: pd.DataFrame, output_dir: Path):
    """Generate visualization plots."""

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Contribution by position
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Position analysis
    position_stats = analyze_by_position(df)
    ax = axes[0, 0]
    ax.bar(range(len(position_stats)), position_stats['mean_abs_contribution'])
    ax.set_xlabel('Position Bin (0=start, 9=end)')
    ax.set_ylabel('Mean |Contribution|')
    ax.set_title('Credit Assignment by Position')

    # Contribution distribution
    ax = axes[0, 1]
    ax.hist(df['contribution'], bins=50, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Contribution (advantage × log_prob)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Per-Token Contributions')
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)

    # Log prob vs advantage scatter
    ax = axes[1, 0]
    sample = df.sample(min(5000, len(df)))  # Sample for plotting
    colors = np.where(sample['advantage'] > 0, 'green', 'red')
    ax.scatter(sample['log_prob'], sample['advantage'], c=colors, alpha=0.3, s=1)
    ax.set_xlabel('Log Probability')
    ax.set_ylabel('Advantage')
    ax.set_title('Log Prob vs Advantage (green=positive, red=negative)')

    # Contribution over training steps
    ax = axes[1, 1]
    step_stats = df.groupby('step').agg({
        'contribution': ['mean', lambda x: x.abs().mean()],
    })
    step_stats.columns = ['mean_contribution', 'mean_abs_contribution']
    ax.plot(step_stats.index, step_stats['mean_abs_contribution'], label='Mean |Contribution|')
    ax.plot(step_stats.index, step_stats['mean_contribution'], label='Mean Contribution', alpha=0.7)
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Contribution')
    ax.set_title('Credit Assignment Over Training')
    ax.legend()
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / 'credit_analysis.png', dpi=150)
    plt.close()

    print(f"Saved plots to {output_dir / 'credit_analysis.png'}")


def main():
    parser = argparse.ArgumentParser(description='Analyze credit assignment logs')
    parser.add_argument('--log_path', type=str, required=True,
                        help='Path to credit assignment log file (.pkl)')
    parser.add_argument('--output_dir', type=str, default='analysis_output',
                        help='Directory for output files')
    parser.add_argument('--tokenizer_path', type=str, default=None,
                        help='Path to tokenizer.json for token decoding')
    args = parser.parse_args()

    print(f"Loading logs from {args.log_path}...")
    logger = load_logs(args.log_path)

    print(f"Loaded {len(logger.logs)} token logs from {len(logger.step_summaries)} steps")

    # Convert to DataFrame
    df = logger.to_dataframe()

    print("\n" + "="*60)
    print("CREDIT ASSIGNMENT ANALYSIS")
    print("="*60)

    # Basic statistics
    print("\n### Basic Statistics ###")
    print(f"Total tokens logged: {len(df)}")
    print(f"Unique steps: {df['step'].nunique()}")
    print(f"Unique episodes: {df[['step', 'episode_idx']].drop_duplicates().shape[0]}")
    print(f"Mean tokens per episode: {len(df) / df[['step', 'episode_idx']].drop_duplicates().shape[0]:.1f}")

    # Contribution statistics
    print("\n### Contribution Statistics ###")
    print(f"Mean contribution: {df['contribution'].mean():.6f}")
    print(f"Std contribution: {df['contribution'].std():.6f}")
    print(f"Mean |contribution|: {df['contribution'].abs().mean():.6f}")
    print(f"Min contribution: {df['contribution'].min():.6f}")
    print(f"Max contribution: {df['contribution'].max():.6f}")

    # Position analysis
    print("\n### Position Analysis ###")
    position_stats = analyze_by_position(df)
    print(position_stats)

    # Reward analysis
    print("\n### Reward-Based Analysis ###")
    reward_analysis = analyze_by_reward(df)
    for key, value in reward_analysis.items():
        print(f"\n{key}:")
        for k, v in value.items():
            print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

    # Consistency analysis
    print("\n### Contribution Consistency (T-Statistics) ###")
    consistency = analyze_contribution_consistency(df)
    for k, v in consistency.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    # Generate plots
    output_dir = Path(args.output_dir)
    plot_results(df, output_dir)

    # Save processed data
    df.to_csv(output_dir / 'credit_data.csv', index=False)
    print(f"\nSaved processed data to {output_dir / 'credit_data.csv'}")

    # Key findings
    print("\n" + "="*60)
    print("KEY FINDINGS")
    print("="*60)

    # Position effect
    early_contrib = df[df['token_position'] < df['token_position'].quantile(0.2)]['contribution'].abs().mean()
    late_contrib = df[df['token_position'] > df['token_position'].quantile(0.8)]['contribution'].abs().mean()
    print(f"\n1. Position Effect:")
    print(f"   Early 20% mean |contribution|: {early_contrib:.6f}")
    print(f"   Late 20% mean |contribution|: {late_contrib:.6f}")
    print(f"   Ratio (early/late): {early_contrib/late_contrib:.2f}x")

    # Advantage effect
    print(f"\n2. Advantage Effect:")
    pos_adv = df[df['advantage'] > 0]['contribution'].mean()
    neg_adv = df[df['advantage'] <= 0]['contribution'].mean()
    print(f"   Positive advantage mean contribution: {pos_adv:.6f}")
    print(f"   Negative advantage mean contribution: {neg_adv:.6f}")
    print(f"   (Positive should be positive, negative should be negative)")

    # Overall conclusion
    print(f"\n3. Overall Credit Assignment Pattern:")
    print(f"   The contribution = advantage × log_prob")
    print(f"   - When advantage > 0 (better than average): tokens get POSITIVE update")
    print(f"   - When advantage < 0 (worse than average): tokens get NEGATIVE update")
    print(f"   - All tokens in an episode get the SAME advantage (episode-level)")


if __name__ == "__main__":
    main()
