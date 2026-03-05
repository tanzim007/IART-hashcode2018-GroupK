"""
Visualization module.

  - plot_evolution()   : score history of a single run
  - plot_comparison()  : side-by-side comparison of all algorithms
  - plot_multi_instance(): heatmap / bar chart across instances
"""
import os
from typing import Dict, List, Any

import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for all OS)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


COLORS = {
    "hill_climbing":       "#e74c3c",
    "simulated_annealing": "#3498db",
    "tabu_search":         "#2ecc71",
    "genetic_algorithm":   "#f39c12",
}

LABELS = {
    "hill_climbing":       "Hill Climbing",
    "simulated_annealing": "Simulated Annealing",
    "tabu_search":         "Tabu Search",
    "genetic_algorithm":   "Genetic Algorithm",
}


def _savefig(fig, path: str):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"  [Plot] Saved → {path}")


# ── single algorithm evolution ───────────────────────────────────────────────

def plot_evolution(
    history: List[int],
    algo_name: str,
    instance_name: str,
    save_path: str = None,
) -> None:
    """Plot score evolution over iterations for one algorithm."""
    fig, ax = plt.subplots(figsize=(8, 4))
    color = COLORS.get(algo_name, "#888888")
    label = LABELS.get(algo_name, algo_name)

    ax.plot(history, color=color, linewidth=1.5, label=label)
    ax.set_title(f"{label} – {instance_name}", fontsize=13)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best Score")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.grid(True, alpha=0.3)
    ax.legend()

    if save_path:
        _savefig(fig, save_path)
    else:
        plt.show()


# ── multi-algorithm comparison ───────────────────────────────────────────────

def plot_comparison(
    summary: Dict[str, Any],
    instance_name: str,
    save_path: str = None,
) -> None:
    """
    Two-panel figure:
      Left  – score history of all algorithms (normalised x-axis 0–100%)
      Right – bar chart of final scores
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Algorithm Comparison – {instance_name}", fontsize=14)

    # --- left: evolution curves ---
    for algo, data in summary.items():
        hist = data["history"]
        # normalise x to 0-100% so different iteration counts are comparable
        xs = np.linspace(0, 100, len(hist))
        color = COLORS.get(algo, "#888888")
        label = LABELS.get(algo, algo)
        ax1.plot(xs, hist, color=color, linewidth=1.8, label=label)

    ax1.set_xlabel("Progress (%)")
    ax1.set_ylabel("Best Score")
    ax1.set_title("Score Evolution")
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)

    # --- right: final score bar chart ---
    algos  = list(summary.keys())
    scores = [summary[a]["score"] for a in algos]
    colors = [COLORS.get(a, "#888888") for a in algos]
    bars = ax2.bar([LABELS.get(a, a) for a in algos], scores, color=colors, edgecolor="white", linewidth=0.8)

    ax2.set_ylabel("Final Score")
    ax2.set_title("Final Score Comparison")
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax2.tick_params(axis="x", rotation=15)
    ax2.grid(axis="y", alpha=0.3)

    # label each bar
    for bar, score in zip(bars, scores):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.01,
            f"{score:,}",
            ha="center", va="bottom", fontsize=9
        )

    plt.tight_layout()
    if save_path:
        _savefig(fig, save_path)
    else:
        plt.show()


# ── time vs score scatter ────────────────────────────────────────────────────

def plot_time_vs_score(
    summary: Dict[str, Any],
    instance_name: str,
    save_path: str = None,
) -> None:
    """Scatter plot of runtime vs final score."""
    fig, ax = plt.subplots(figsize=(7, 5))

    for algo, data in summary.items():
        color = COLORS.get(algo, "#888888")
        label = LABELS.get(algo, algo)
        ax.scatter(data["time"], data["score"], color=color, s=120, label=label, zorder=3)
        ax.annotate(label, (data["time"], data["score"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)

    ax.set_xlabel("Runtime (s)")
    ax.set_ylabel("Final Score")
    ax.set_title(f"Time vs Score – {instance_name}")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    if save_path:
        _savefig(fig, save_path)
    else:
        plt.show()


# ── multi-instance grouped bar ───────────────────────────────────────────────

def plot_multi_instance(
    all_results: Dict[str, Dict[str, Any]],
    save_path: str = None,
) -> None:
    """
    Grouped bar chart: x-axis = instances, groups = algorithms.

    all_results = { instance_name: { algo_name: {score, time, ...}, ... } }
    """
    instances = list(all_results.keys())
    algos     = list(ALGORITHM_FNS_ORDER)

    x = np.arange(len(instances))
    width = 0.2

    fig, ax = plt.subplots(figsize=(max(8, len(instances) * 2.5), 5))

    for i, algo in enumerate(algos):
        scores = [all_results[inst].get(algo, {}).get("score", 0) for inst in instances]
        offset = (i - len(algos) / 2 + 0.5) * width
        bars = ax.bar(x + offset, scores, width,
                      label=LABELS.get(algo, algo),
                      color=COLORS.get(algo, "#888888"),
                      edgecolor="white", linewidth=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(instances, rotation=15)
    ax.set_ylabel("Score")
    ax.set_title("Algorithm Scores Across Instances")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        _savefig(fig, save_path)
    else:
        plt.show()


# order used in multi-instance chart
ALGORITHM_FNS_ORDER = [
    "hill_climbing",
    "simulated_annealing",
    "tabu_search",
    "genetic_algorithm",
]
