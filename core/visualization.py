import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------------------------------------------------
# Configure module‐level logger with UTF-8 encoding
# -----------------------------------------------------------------------------
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
file_handler = logging.FileHandler(os.path.join(log_dir, "application.log"), encoding="utf-8")
stream_handler = logging.StreamHandler()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[file_handler, stream_handler],
)
logger = logging.getLogger(__name__)


def visualize_cache_performance(
    evaluation_results: Dict[str, Any],
    output_dir: str = "cache_eval_results",
) -> Dict[str, str]:
    """
    Generate visualizations of cache performance metrics, action distribution,
    cache occupancy, hit-rate windows, reward distributions, Q-values, and
    inference time CDF.

    Returns a dict mapping plot identifiers to file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Core data
    hits: List[int] = evaluation_results.get("hit_history", [])
    moving: List[float] = (
        evaluation_results.get("moving_hit_rates")
        or evaluation_results.get("moving_hit_rate")
        or []
    )
    rewards: List[float] = [float(r or 0.0) for r in evaluation_results.get("rewards", [])]
    final_hit_rate: float = evaluation_results.get("final_hit_rate", 0.0)
    device: str = evaluation_results.get("device_used", "UNKNOWN")

    # Optional diagnostics
    reasoning: List[str] = evaluation_results.get("step_reasoning", [])
    in_cache: List[bool] = evaluation_results.get("in_cache", [])
    actions: List[int] = evaluation_results.get("actions", [])
    occupancy: List[int] = evaluation_results.get("cache_occupancy", [])
    q_hist: Optional[List[List[float]]] = evaluation_results.get("q_values_history")
    ep_rewards: Optional[List[float]] = evaluation_results.get("episode_rewards")
    losses: Optional[List[float]] = evaluation_results.get("losses")
    entropies: Optional[List[float]] = evaluation_results.get("entropies")
    inf_times: List[float] = evaluation_results.get("inference_times", [])

    plot_paths: Dict[str, str] = {}
    steps = np.arange(1, len(hits) + 1)

    # --- 1) Main performance plot: Hit/Miss and Rewards ---
    fig, (ax_hits, ax_rew) = plt.subplots(2, 1, figsize=(10, 12), constrained_layout=True)

    # Hit/Miss
    ax_hits.scatter(steps, hits, s=10, alpha=0.5, label="Hit/Miss (1/0)")
    if moving:
        ax_hits.plot(steps, moving, color="red", label="Moving Hit Rate")
    ax_hits.set_title(f"Cache Hits (final rate={final_hit_rate:.3f}) on {device.upper()}")
    ax_hits.set_xlabel("Step")
    ax_hits.set_ylabel("Hit (1) / Miss (0)")
    ax_hits.set_ylim(-0.1, 1.1)
    ax_hits.grid(True, linestyle="--", alpha=0.7)
    ax_hits.legend()

    # Rewards per step + normalized cumulative
    ax_rew.plot(steps, rewards, color="blue", alpha=0.5, label="Step Reward")
    if rewards:
        cum = np.cumsum(rewards)
        norm = cum / (cum.max() or 1.0) * max(rewards)
        ax_rew.plot(steps, norm, color="green", linestyle="-", label="Normalized Cumulative")
    ax_rew.set_title("Rewards per Step")
    ax_rew.set_xlabel("Step")
    ax_rew.set_ylabel("Reward")
    ax_rew.grid(True, linestyle="--", alpha=0.7)
    ax_rew.legend()

    # Diagnostics table (first 10 steps)
    if reasoning and in_cache:
        n = min(10, len(hits))
        cell_text = []
        for i in range(n):
            cell_text.append([
                str(i + 1),
                "Hit" if hits[i] else "Miss",
                "Yes" if in_cache[i] else "No",
                reasoning[i][:50] + ("…" if len(reasoning[i]) > 50 else ""),
            ])
        tbl_ax = fig.add_axes([0.1, -0.25, 0.8, 0.2])
        tbl_ax.axis("off")
        table = tbl_ax.table(
            cellText=cell_text,
            colLabels=["Step", "Hit/Miss", "In Cache", "Reason"],
            loc="center"
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.2)

    # Footer annotation
    avg_inf = evaluation_results.get("avg_inference_ms") or np.mean(inf_times or [0.0])
    eval_time = evaluation_results.get("evaluation_time_seconds", 0.0)
    total_reward = evaluation_results.get("total_reward", sum(rewards))
    footer = (
        f"Device: {device.upper()} | Avg Inference: {avg_inf:.2f}ms | "
        f"Eval Time: {eval_time:.2f}s | Total Reward: {total_reward:.2f} | "
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    plt.figtext(0.5, 0.01, footer, ha="center", fontsize=9, bbox=dict(facecolor="yellow", alpha=0.2))

    perf_path = os.path.join(output_dir, f"cache_performance_{timestamp}.png")
    plt.savefig(perf_path, dpi=100)
    plt.close(fig)
    logger.info(f"Saved performance plot: {perf_path}")
    plot_paths["performance_plot"] = perf_path

    # --- 2) Action Distribution ---
    if actions:
        plt.figure(figsize=(6, 4))
        plt.hist(actions, bins=[-0.5, 0.5, 1.5], rwidth=0.6, color="skyblue")
        plt.xticks([0, 1], ["Skip", "Cache"])
        plt.title("Action Distribution")
        plt.ylabel("Count")
        act_path = os.path.join(output_dir, f"action_dist_{timestamp}.png")
        plt.savefig(act_path, dpi=100)
        plt.close()
        logger.info(f"Saved action distribution: {act_path}")
        plot_paths["action_distribution"] = act_path

    # --- 3) Cache Occupancy Over Time ---
    if occupancy:
        plt.figure(figsize=(8, 3))
        plt.plot(steps, occupancy, label="Cache Size")
        max_cap = evaluation_results.get("cache_size", None)
        if max_cap:
            plt.axhline(max_cap, color="red", linestyle="--", label="Max Capacity")
        plt.title("Cache Occupancy Over Time")
        plt.xlabel("Step")
        plt.ylabel("Items in Cache")
        plt.grid(True, linestyle="--", alpha=0.7)
        occ_path = os.path.join(output_dir, f"cache_occupancy_{timestamp}.png")
        plt.savefig(occ_path, dpi=100)
        plt.close()
        logger.info(f"Saved cache occupancy: {occ_path}")
        plot_paths["cache_occupancy"] = occ_path

    # --- 4) Hit Rate per Window ---
    window = evaluation_results.get("window_size", 50)
    if hits and len(hits) >= window:
        rates = [np.mean(hits[i : i + window]) for i in range(0, len(hits), window)]
        plt.figure(figsize=(8, 3))
        plt.bar(range(len(rates)), rates, color="orange", alpha=0.7)
        plt.title(f"Hit Rate per {window}-step Window")
        plt.xlabel("Window Index")
        plt.ylabel("Hit Rate")
        wr_path = os.path.join(output_dir, f"hit_rate_window_{timestamp}.png")
        plt.savefig(wr_path, dpi=100)
        plt.close()
        logger.info(f"Saved hit-rate window plot: {wr_path}")
        plot_paths["hit_rate_window"] = wr_path

    # --- 5) Reward Distribution ---
    if rewards:
        plt.figure(figsize=(6, 4))
        plt.hist(rewards, bins=30, color="purple", alpha=0.7)
        plt.title("Reward Distribution")
        plt.xlabel("Reward")
        plt.ylabel("Frequency")
        rd_path = os.path.join(output_dir, f"reward_dist_{timestamp}.png")
        plt.savefig(rd_path, dpi=100)
        plt.close()
        logger.info(f"Saved reward distribution: {rd_path}")
        plot_paths["reward_distribution"] = rd_path

    # --- 6) Q-Value Statistics ---
    if q_hist:
        arr = np.array(q_hist)
        steps_q = np.arange(arr.shape[0])
        plt.figure(figsize=(10, 4))
        plt.plot(steps_q, arr.mean(axis=1), label="Mean Q")
        plt.plot(steps_q, arr.max(axis=1), "--", label="Max Q")
        plt.plot(steps_q, arr.min(axis=1), ":", label="Min Q")
        plt.title("Q-Value Statistics Over Time")
        plt.xlabel("Step")
        plt.ylabel("Q-Value")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)
        qv_path = os.path.join(output_dir, f"q_values_{timestamp}.png")
        plt.savefig(qv_path, dpi=100)
        plt.close()
        logger.info(f"Saved Q-value stats: {qv_path}")
        plot_paths["q_value_stats"] = qv_path

    # --- 7) Inference Time CDF ---
    if inf_times:
        sorted_t = np.sort(inf_times)
        cdf = np.arange(len(sorted_t)) / len(sorted_t)
        plt.figure(figsize=(6, 4))
        plt.plot(sorted_t, cdf, marker=".", linestyle="-")
        plt.title("Inference Time CDF")
        plt.xlabel("Inference (ms)")
        plt.ylabel("CDF")
        it_path = os.path.join(output_dir, f"inference_cdf_{timestamp}.png")
        plt.savefig(it_path, dpi=100)
        plt.close()
        logger.info(f"Saved inference time CDF: {it_path}")
        plot_paths["inference_cdf"] = it_path

    return plot_paths