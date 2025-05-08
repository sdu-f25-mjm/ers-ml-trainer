# core/visualization.py
import logging
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import threading


def visualize_cache_performance(evaluation_results, output_dir="cache_eval_results"):
    """Generate visualizations of cache performance metrics and other learning diagnostics"""
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

    # --- Fix: Ensure rewards are all numbers ---
    rewards = [r if r is not None else 0.0 for r in rewards]

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
    fig.set_constrained_layout(True)

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

    plot_paths = {"performance_plot": filepath}

    # --- Visualize Q-value estimation if available ---
    q_values_history = evaluation_results.get('q_values_history')
    game_name = evaluation_results.get('game_name', 'Unknown')
    if q_values_history is not None and len(q_values_history) > 0:
        try:
            q_values_plot_path = visualize_double_dqn_value_estimation(
                q_values_history, game_name=game_name, output_dir=output_dir
            )
            if q_values_plot_path:
                logger.info(f"Q-value estimation plot saved to {q_values_plot_path}")
                plot_paths["q_values_plot"] = q_values_plot_path
        except Exception as e:
            logger.warning(f"Failed to visualize Q-value estimation: {e}")
    else:
        logger.info("No Q-value history available for Q-value plot.")

    # --- Plot rewards over episodes if available ---
    episode_rewards = evaluation_results.get('episode_rewards')
    rewards_plot_path = None
    if episode_rewards is not None and len(episode_rewards) > 0:
        try:
            rewards_plot_path = plot_rewards_over_episodes(
                episode_rewards, output_dir=output_dir
            )
            plot_paths["episode_rewards_plot"] = rewards_plot_path
            logger.info(f"Episode rewards plot saved to {rewards_plot_path}")
        except Exception as e:
            logger.warning(f"Failed to plot rewards over episodes: {e}")
    else:
        logger.info("No episode rewards available for episode rewards plot.")

    # --- Plot learning diagnostics if available ---
    try:
        diagnostics_paths = plot_learning_diagnostics(
            evaluation_results,
            output_dir=output_dir,
            title_prefix="Learning Diagnostics",
            episode_rewards_plot_path=rewards_plot_path  # Pass the already generated plot path
        )
        if diagnostics_paths:
            logger.info(f"Learning diagnostics plots saved: {diagnostics_paths}")
        plot_paths.update(diagnostics_paths)
    except Exception as e:
        logger.warning(f"Failed to plot learning diagnostics: {e}")

    return plot_paths


def visualize_double_dqn_value_estimation(q_values_history, game_name="Unknown", output_dir="cache_eval_results"):
    """
    Visualize Double DQN Q-value estimation statistics (mean, max, min) over time.

    Args:
        q_values_history: list of lists or 2D np.ndarray, shape (steps, actions)
        game_name: str, name of the game or experiment
        output_dir: directory to save the plot

    Returns:
        Path to the saved plot image.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    from datetime import datetime
    import logging

    logger = logging.getLogger(__name__)
    os.makedirs(output_dir, exist_ok=True)

    q_values_history = np.array(q_values_history)
    steps = np.arange(q_values_history.shape[0])

    plt.figure(figsize=(10, 5))
    plt.plot(steps, np.mean(q_values_history, axis=1), label="Mean Q-value")
    plt.plot(steps, np.max(q_values_history, axis=1), label="Max Q-value", linestyle='--')
    plt.plot(steps, np.min(q_values_history, axis=1), label="Min Q-value", linestyle=':')
    plt.xlabel("Step")
    plt.ylabel("Q-value")
    plt.title(f"Double DQN Q-value Estimation: {game_name}")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    filename = f"double_dqn_q_values_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=100)
    plt.close()
    logger.info(f"Double DQN Q-value estimation plot saved to {filepath}")
    return filepath


def plot_rewards_over_episodes(episode_rewards, output_dir="cache_eval_results", title="Rewards Over Episodes"):
    """
    Plot the total reward per episode.
    Args:
        episode_rewards: List of total rewards per episode.
        output_dir: Directory to save the plot.
        title: Plot title.
    Returns:
        Path to the saved plot image.
    """
    logger = logging.getLogger(__name__)
    os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(episode_rewards) + 1), episode_rewards, marker='o', label="Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title(title)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    filename = f"rewards_over_episodes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=100)
    plt.close()
    logger.info(f"Rewards over episodes plot saved to {filepath}")
    return filepath


def plot_learning_diagnostics(
    training_history,
    output_dir="cache_eval_results",
    title_prefix="Learning Diagnostics",
    episode_rewards_plot_path=None
):
    """
    Plot additional diagnostics for the learning process if data is available.
    Args:
        training_history: dict with optional keys: 'losses', 'q_values_history', 'entropies', 'episode_rewards'
        output_dir: directory to save plots
        title_prefix: prefix for plot titles
        episode_rewards_plot_path: if provided, use this path instead of generating a new plot
    Returns:
        Dict of plot file paths
    """
    logger = logging.getLogger(__name__)
    os.makedirs(output_dir, exist_ok=True)
    plot_paths = {}

    # Plot loss curve if available
    losses = training_history.get('losses')
    if losses is not None and len(losses) > 0:
        plt.figure(figsize=(10, 4))
        plt.plot(losses, label="Loss")
        plt.xlabel("Training Step")
        plt.ylabel("Loss")
        plt.title(f"{title_prefix}: Loss Curve")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        loss_path = os.path.join(output_dir, f"loss_curve_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(loss_path, dpi=100)
        plt.close()
        plot_paths['loss_curve'] = loss_path
        logger.info(f"Loss curve plot saved to {loss_path}")

    # Plot Q-value statistics if available
    q_values_history = training_history.get('q_values_history')
    if q_values_history is not None and len(q_values_history) > 0:
        q_values_history = np.array(q_values_history)
        steps = np.arange(q_values_history.shape[0])
        plt.figure(figsize=(10, 4))
        plt.plot(steps, np.mean(q_values_history, axis=1), label="Mean Q-value")
        plt.plot(steps, np.max(q_values_history, axis=1), label="Max Q-value", linestyle='--')
        plt.plot(steps, np.min(q_values_history, axis=1), label="Min Q-value", linestyle=':')
        plt.xlabel("Step")
        plt.ylabel("Q-value")
        plt.title(f"{title_prefix}: Q-value Statistics")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        qval_path = os.path.join(output_dir, f"q_value_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(qval_path, dpi=100)
        plt.close()
        plot_paths['q_value_stats'] = qval_path
        logger.info(f"Q-value statistics plot saved to {qval_path}")

    # Plot policy entropy if available
    entropies = training_history.get('entropies')
    if entropies is not None and len(entropies) > 0:
        plt.figure(figsize=(10, 4))
        plt.plot(entropies, label="Policy Entropy")
        plt.xlabel("Training Step")
        plt.ylabel("Entropy")
        plt.title(f"{title_prefix}: Policy Entropy")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        entropy_path = os.path.join(output_dir, f"policy_entropy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        plt.savefig(entropy_path, dpi=100)
        plt.close()
        plot_paths['policy_entropy'] = entropy_path
        logger.info(f"Policy entropy plot saved to {entropy_path}")

    # Plot episode rewards if available and not already plotted
    episode_rewards = training_history.get('episode_rewards')
    if episode_rewards is not None and len(episode_rewards) > 0:
        if episode_rewards_plot_path:
            plot_paths['episode_rewards'] = episode_rewards_plot_path
        else:
            plot_paths['episode_rewards'] = plot_rewards_over_episodes(
                episode_rewards, output_dir=output_dir, title=f"{title_prefix}: Rewards Over Episodes"
            )

    return plot_paths
