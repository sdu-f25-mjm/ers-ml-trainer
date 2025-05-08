# core/visualization.py
import logging
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import threading


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

    # --- NEW: Visualize Q-value estimation if available ---
    q_values_plot_path = None
    q_values_history = evaluation_results.get('q_values_history')
    game_name = evaluation_results.get('game_name', 'Unknown')
    if q_values_history is not None:
        try:
            q_values_plot_path = visualize_double_dqn_value_estimation(
                q_values_history, game_name=game_name, output_dir=output_dir
            )
            if q_values_plot_path:
                logger.info(f"Q-value estimation plot saved to {q_values_plot_path}")
        except Exception as e:
            logger.warning(f"Failed to visualize Q-value estimation: {e}")

    # Return both plot paths if Q-values plot was generated
    if q_values_plot_path:
        return {"performance_plot": filepath, "q_values_plot": q_values_plot_path}
    return filepath


def visualize_double_dqn_value_estimation(q_values_history, game_name="Unknown", output_dir="dqn_value_vis"):
    """
    Visualize value estimation (Q-values) over time for Double DQN on Atari games.
    Args:
        q_values_history: List of lists/arrays of Q-values per step (shape: [steps, actions])
        game_name: Name of the Atari game (e.g., 'Alien', 'Space Invaders', etc.)
        output_dir: Directory to save the plot
    """
    logger = logging.getLogger(__name__)
    os.makedirs(output_dir, exist_ok=True)

    q_values_history = np.array(q_values_history)  # shape: [steps, actions]
    if q_values_history.ndim != 2 or q_values_history.shape[0] == 0:
        logger.warning("q_values_history must be a 2D array with shape [steps, actions] and nonzero steps.")
        return None

    steps = np.arange(q_values_history.shape[0])

    plt.figure(figsize=(12, 6))
    for action in range(q_values_history.shape[1]):
        plt.plot(steps, q_values_history[:, action], label=f"Action {action}")

    plt.title(f"Double DQN Value Estimation - {game_name}")
    plt.xlabel("Step")
    plt.ylabel("Estimated Q-Value")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    filename = f"dqn_value_{game_name.replace(' ', '_').lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=100)
    plt.close()
    logger.info(f"Double DQN value estimation visualization saved to {filepath}")
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


class RealTimeTrainingPlotter:
    """
    Real-time plotter for RL training metrics (e.g., hit rate, reward).
    Call .update(step, hit_rate, reward) from your training loop.

    Usage Example:
    --------------
    from core.visualization import RealTimeTrainingPlotter

    plotter = RealTimeTrainingPlotter()
    plotter.start()

    for step in range(total_steps):
        # ... RL training logic ...
        plotter.update(step, current_hit_rate, current_reward)

    plotter.stop()
    """
    def __init__(self, title="RL Training Progress", interval=500):
        self.hit_rates = []
        self.rewards = []
        self.steps = []
        self.lock = threading.Lock()
        self.fig, self.ax = plt.subplots(2, 1, figsize=(10, 8))
        self.ani = animation.FuncAnimation(
            self.fig, self._animate, interval=interval, blit=False
        )
        self.fig.suptitle(title)
        self._running = False

    def start(self):
        self._running = True
        threading.Thread(target=plt.show, daemon=True).start()

    def stop(self):
        self._running = False
        plt.close(self.fig)

    def update(self, step, hit_rate, reward):
        with self.lock:
            self.steps.append(step)
            self.hit_rates.append(hit_rate)
            self.rewards.append(reward)

    def _animate(self, frame):
        with self.lock:
            self.ax[0].clear()
            self.ax[1].clear()
            self.ax[0].plot(self.steps, self.hit_rates, label="Hit Rate", color="blue")
            self.ax[0].set_ylabel("Hit Rate")
            self.ax[0].set_xlabel("Step")
            self.ax[0].legend()
            self.ax[0].grid(True, linestyle='--', alpha=0.7)
            self.ax[1].plot(self.steps, self.rewards, label="Reward", color="green")
            self.ax[1].set_ylabel("Reward")
            self.ax[1].set_xlabel("Step")
            self.ax[1].legend()
            self.ax[1].grid(True, linestyle='--', alpha=0.7)
        return self.ax
