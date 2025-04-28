# core/visualization.py
import logging
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np


def visualize_cache_performance(evaluation_results, output_dir="cache_eval_results"):
    """Generate visualizations of cache performance metrics"""
    logger = logging.getLogger(__name__)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Extract data
    hit_history = evaluation_results.get('hit_history', [])
    moving_hit_rates = evaluation_results.get('moving_hit_rates', [])
    rewards = evaluation_results.get('rewards', [])
    final_hit_rate = evaluation_results.get('final_hit_rate', 0)
    device_used = evaluation_results.get('device_used', 'unknown')

    # Extract new reasoning/in_cache columns if present
    step_reasoning = evaluation_results.get('step_reasoning', [])
    in_cache = evaluation_results.get('in_cache', [])

    if not hit_history:
        logger.warning("No hit history data to visualize")
        return None

    # Create figure with multiple subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

    # Plot hit/miss pattern
    steps = list(range(1, len(hit_history) + 1))
    ax1.scatter(steps, hit_history, s=10, alpha=0.5, label='Hit/Miss (1/0)')
    ax1.plot(steps, moving_hit_rates, 'r-', label='Moving Avg Hit Rate')
    ax1.set_title(f'Cache Hit/Miss Pattern (Final Hit Rate: {final_hit_rate:.4f})')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Cache Hit (1) / Miss (0)')
    ax1.set_ylim(-0.1, 1.1)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()

    # Plot rewards
    if rewards:
        cumulative_rewards = np.cumsum(rewards)
        ax2.plot(steps, rewards, 'b-', alpha=0.5, label='Step Reward')
        ax2.plot(steps, cumulative_rewards / max(1, max(cumulative_rewards)) * max(rewards),
                 'g-', label='Normalized Cumulative Reward')
        ax2.set_title('RL Rewards per Step')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Reward Value')
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend()

    # Add a table with reasoning and in_cache for first 10 steps (if available)
    if step_reasoning and in_cache:
        from matplotlib.table import Table
        ax_table = fig.add_axes([0.1, -0.25, 0.8, 0.18])  # [left, bottom, width, height]
        ax_table.axis('off')
        n_rows = min(10, len(step_reasoning))
        col_labels = ["Step", "Hit/Miss", "In Cache", "Reasoning"]
        cell_text = []
        for i in range(n_rows):
            hitmiss = "Hit" if hit_history[i] else "Miss"
            cell_text.append([
                str(i + 1),
                hitmiss,
                str(in_cache[i]),
                step_reasoning[i][:60] + ("..." if len(step_reasoning[i]) > 60 else "")
            ])
        table = ax_table.table(
            cellText=cell_text,
            colLabels=col_labels,
            loc='center',
            cellLoc='left'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.2)

    # Add metadata
    plt.tight_layout()

    # Add more evaluation metrics to the figure annotation
    evaluation_time = evaluation_results.get('evaluation_time_seconds', 0)
    total_reward = evaluation_results.get('total_reward', 0)
    avg_inference = evaluation_results.get('avg_inference_time_ms', 0)

    plt.figtext(
        0.5, 0.01,
        f"Evaluated on {device_used.upper()} | Avg Inference: {avg_inference:.2f}ms | "
        f"Evaluation Time: {evaluation_time:.2f}s | Total Reward: {total_reward:.2f} | "
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ha="center", fontsize=10,
        bbox={"facecolor": "orange", "alpha": 0.2, "pad": 5}
    )

    # Save figure
    filename = f"cache_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=100)
    plt.close()

    logger.info(f"Cache visualization saved to {filepath}")
    return filepath

