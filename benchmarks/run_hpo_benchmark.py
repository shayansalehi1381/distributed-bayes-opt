"""Real-World Machine Learning Hyperparameter Optimization (HPO) Benchmark.

Compares three optimization paradigms for tuning a Gradient Boosting Regressor
on the California Housing dataset:
1. Standard Random Search
2. Sequential (Single-Worker) Bayesian Optimization
3. Asynchronous Distributed Bayesian Optimization (4 Parallel Workers)

Outputs:
- Console performance metrics and parallel speedup statistics.
- High-aesthetic 1x3 visualization: `benchmarks/hpo_benchmark_results.png`
  (Convergence vs Evaluations, Convergence vs Wall-Clock Time, and Worker Gantt Chart).
"""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pathlib
import sys
# Ensure project root is in sys.path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import time
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
import torch
from torch import Tensor

from src.distributed import AsyncBayesOptCoordinator, DistributedConfig

# ---------------------------------------------------------------------------
# Benchmark Setup
# ---------------------------------------------------------------------------

TOTAL_EVALS = 32
N_WORKERS = 4
SEED = 42

print("Loading California Housing dataset for HPO benchmark...")
X_raw, y_raw = fetch_california_housing(return_X_y=True)
# Subsample 2500 samples for rapid yet realistic cross-validation
rng = np.random.default_rng(SEED)
idx_subset = rng.choice(len(X_raw), size=2500, replace=False)
X_data, y_data = X_raw[idx_subset], y_raw[idx_subset]

# Parameter Bounds:
# Dim 0: log10(learning_rate) in [-2.0, -0.5] -> lr in [0.01, 0.316]
# Dim 1: max_depth in [2.0, 8.0]
# Dim 2: n_estimators in [20.0, 120.0]
# Dim 3: subsample in [0.5, 1.0]
BOUNDS = torch.tensor(
    [
        [-2.0, -0.5],
        [2.0, 8.0],
        [20.0, 120.0],
        [0.5, 1.0],
    ],
    dtype=torch.float64,
)

def evaluate_hpo_objective(x: Tensor) -> float:
    """Evaluate 3-fold cross-validated R^2 score for Gradient Boosting."""
    lr = float(10 ** x[0].item())
    max_depth = int(round(x[1].item()))
    n_estimators = int(round(x[2].item()))
    subsample = float(x[3].item())

    model = GradientBoostingRegressor(
        learning_rate=lr,
        max_depth=max_depth,
        n_estimators=n_estimators,
        subsample=subsample,
        random_state=SEED,
    )
    # 3-fold cross validation score (R^2)
    scores = cross_val_score(model, X_data, y_data, cv=3, scoring="r2", n_jobs=1)
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Baseline 1: Random Search
# ---------------------------------------------------------------------------

def run_random_search(n_evals: int) -> tuple[list[float], list[float], list[float]]:
    """Execute Uniform Random Search."""
    print("\n[1/3] Running Uniform Random Search Baseline...")
    start_time = time.perf_counter()
    history_y: list[float] = []
    history_times: list[float] = []
    best_so_far: list[float] = []
    current_best = -float("inf")

    torch.manual_seed(SEED)
    lo = BOUNDS[:, 0]
    hi = BOUNDS[:, 1]

    for step in range(1, n_evals + 1):
        u = torch.rand(4, dtype=torch.float64)
        x_cand = lo + u * (hi - lo)
        y_val = evaluate_hpo_objective(x_cand)

        history_y.append(y_val)
        history_times.append(time.perf_counter() - start_time)
        current_best = max(current_best, y_val)
        best_so_far.append(current_best)

    total_time = time.perf_counter() - start_time
    print(f"  Random Search finished: Best R^2 = {current_best:.4f} in {total_time:.2f}s")
    return history_y, best_so_far, history_times


# ---------------------------------------------------------------------------
# Baseline 2: Sequential Bayesian Optimization (1 Worker)
# ---------------------------------------------------------------------------

def run_sequential_bo(n_evals: int) -> tuple[list[float], list[float], list[float]]:
    """Execute Sequential Bayesian Optimization."""
    print("\n[2/3] Running Sequential Bayesian Optimization (1 Worker)...")
    cfg = DistributedConfig(
        dim=4,
        bounds=BOUNDS,
        n_workers=1,
        total_evaluations=n_evals,
        n_initial_points=6,
        surrogate="exact",
        surrogate_refit_freq=2,
    )
    coord = AsyncBayesOptCoordinator(evaluate_hpo_objective, cfg)
    start_time = time.perf_counter()
    res = coord.optimize(verbose=False)
    total_time = time.perf_counter() - start_time

    history_y = [rec.y for rec in res.all_evaluations]
    history_times = [rec.finish_time - res.all_evaluations[0].submit_time for rec in res.all_evaluations]
    best_so_far = []
    c_best = -float("inf")
    for y in history_y:
        c_best = max(c_best, y)
        best_so_far.append(c_best)

    print(f"  Sequential BO finished: Best R^2 = {res.best_y:.4f} in {total_time:.2f}s")
    return history_y, best_so_far, history_times


# ---------------------------------------------------------------------------
# Method 3: Distributed Asynchronous Bayesian Optimization (4 Workers)
# ---------------------------------------------------------------------------

def run_distributed_bo(n_evals: int, n_workers: int):
    """Execute Distributed Asynchronous Bayesian Optimization."""
    print(f"\n[3/3] Running Distributed Asynchronous BO ({n_workers} Workers)...")
    cfg = DistributedConfig(
        dim=4,
        bounds=BOUNDS,
        n_workers=n_workers,
        total_evaluations=n_evals,
        n_initial_points=6,
        surrogate="exact",
        surrogate_refit_freq=2,
    )
    coord = AsyncBayesOptCoordinator(evaluate_hpo_objective, cfg)
    start_time = time.perf_counter()
    res = coord.optimize(verbose=True)
    total_time = time.perf_counter() - start_time

    history_y = [rec.y for rec in res.all_evaluations]
    base_t = res.all_evaluations[0].submit_time
    history_times = [rec.finish_time - base_t for rec in res.all_evaluations]
    best_so_far = []
    c_best = -float("inf")
    for y in history_y:
        c_best = max(c_best, y)
        best_so_far.append(c_best)

    print(f"  Distributed BO finished: Best R^2 = {res.best_y:.4f} in {total_time:.2f}s (Speedup: {res.speedup_ratio:.2f}x)")
    return history_y, best_so_far, history_times, res


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_benchmark_results(
    rs_best: list[float],
    rs_times: list[float],
    seq_best: list[float],
    seq_times: list[float],
    dist_best: list[float],
    dist_times: list[float],
    dist_res,
    save_path: str = "benchmarks/hpo_benchmark_results.png",
) -> None:
    """Render a comprehensive 1x3 publication-grade comparison figure."""
    plt.style.use("dark_background")
    BG = "#0d1117"
    PANEL_BG = "#161b22"
    GRID = "#30363d"
    C_RS = "#8b949e"      # Muted grey (Random Search)
    C_SEQ = "#58a6ff"     # Blue (Sequential BO)
    C_DIST = "#3fb950"    # Emerald green (Distributed Async BO)

    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3,
        figsize=(17, 5.2),
        facecolor=BG,
        constrained_layout=True,
    )
    fig.suptitle(
        "Hyperparameter Optimization Benchmark (California Housing — Gradient Boosting)",
        color="white",
        fontsize=15,
        fontweight="bold",
        y=1.03,
    )

    eval_steps = list(range(1, len(rs_best) + 1))

    # --- Panel 1: Regret vs Function Evaluations ---
    ax1.set_facecolor(PANEL_BG)
    ax1.plot(eval_steps, rs_best, label="Random Search", color=C_RS, linestyle="--", linewidth=1.8)
    ax1.plot(eval_steps, seq_best, label="Sequential BO (1 Worker)", color=C_SEQ, linewidth=2.2)
    ax1.plot(eval_steps, dist_best, label=f"Distributed Async BO ({N_WORKERS} Workers)", color=C_DIST, linewidth=2.5)
    ax1.set_title("Sample Efficiency: R² vs Evaluations", color="white", fontsize=12, pad=10)
    ax1.set_xlabel("Number of Evaluations", color="#c9d1d9", fontsize=10)
    ax1.set_ylabel("Best Validation R² Score", color="#c9d1d9", fontsize=10)
    ax1.grid(True, color=GRID, linestyle="--", alpha=0.7)
    ax1.legend(loc="lower right", facecolor=PANEL_BG, edgecolor=GRID, fontsize=9)
    ax1.tick_params(colors="#c9d1d9")
    for s in ax1.spines.values(): s.set_color(GRID)

    # --- Panel 2: Regret vs Wall-Clock Time ---
    ax2.set_facecolor(PANEL_BG)
    ax2.plot(rs_times, rs_best, label="Random Search", color=C_RS, linestyle="--", linewidth=1.8)
    ax2.plot(seq_times, seq_best, label="Sequential BO", color=C_SEQ, linewidth=2.2)
    ax2.plot(dist_times, dist_best, label=f"Distributed Async BO", color=C_DIST, linewidth=2.5)
    ax2.set_title("Time Efficiency: R² vs Wall-Clock (s)", color="white", fontsize=12, pad=10)
    ax2.set_xlabel("Elapsed Wall-Clock Time (seconds)", color="#c9d1d9", fontsize=10)
    ax2.set_ylabel("Best Validation R² Score", color="#c9d1d9", fontsize=10)
    ax2.grid(True, color=GRID, linestyle="--", alpha=0.7)
    ax2.legend(loc="lower right", facecolor=PANEL_BG, edgecolor=GRID, fontsize=9)
    ax2.tick_params(colors="#c9d1d9")
    for s in ax2.spines.values(): s.set_color(GRID)

    # --- Panel 3: Asynchronous Worker Gantt / Concurrency Timeline ---
    ax3.set_facecolor(PANEL_BG)
    base_t = dist_res.all_evaluations[0].submit_time
    worker_colors = ["#f778ba", "#79c0ff", "#d2a8ff", "#ffa657"]

    for rec in dist_res.all_evaluations:
        start_t = rec.submit_time - base_t
        dur = rec.duration
        w_id = rec.worker_id
        ax3.barh(
            y=w_id,
            width=dur,
            left=start_t,
            height=0.6,
            color=worker_colors[w_id % len(worker_colors)],
            alpha=0.85,
            edgecolor=PANEL_BG,
            linewidth=0.8,
        )

    ax3.set_title(f"Async Concurrency: {dist_res.speedup_ratio:.2f}x Speedup", color="white", fontsize=12, pad=10)
    ax3.set_xlabel("Wall-Clock Timeline (seconds)", color="#c9d1d9", fontsize=10)
    ax3.set_ylabel("Worker Thread ID", color="#c9d1d9", fontsize=10)
    ax3.set_yticks(list(range(N_WORKERS)))
    ax3.set_yticklabels([f"Worker {i}" for i in range(N_WORKERS)], color="#c9d1d9")
    ax3.grid(True, color=GRID, linestyle="--", alpha=0.7, axis="x")
    ax3.tick_params(colors="#c9d1d9")
    for s in ax3.spines.values(): s.set_color(GRID)

    out_p = pathlib.Path(save_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_p, dpi=160, facecolor=BG)
    plt.close(fig)
    print(f"\n[OK] Benchmark plot saved to -> {out_p}")


def main() -> None:
    print("=" * 80)
    print("  Real-World Machine Learning Benchmark Suite")
    print(f"  Dataset: California Housing | Model: GradientBoostingRegressor")
    print(f"  Budget: {TOTAL_EVALS} evaluations per method")
    print("=" * 80)

    # 1. Random Search
    rs_y, rs_best, rs_t = run_random_search(TOTAL_EVALS)

    # 2. Sequential BO
    seq_y, seq_best, seq_t = run_sequential_bo(TOTAL_EVALS)

    # 3. Distributed Async BO
    dist_y, dist_best, dist_t, dist_res = run_distributed_bo(TOTAL_EVALS, N_WORKERS)

    # Plot
    plot_benchmark_results(
        rs_best, rs_t,
        seq_best, seq_t,
        dist_best, dist_t,
        dist_res,
        save_path="benchmarks/hpo_benchmark_results.png"
    )

    print("\n" + "=" * 80)
    print("  Benchmark Summary Comparison:")
    print(f"  * Random Search Final Best R^2        : {rs_best[-1]:.4f}  (Time: {rs_t[-1]:.2f}s)")
    print(f"  * Sequential BO Final Best R^2        : {seq_best[-1]:.4f}  (Time: {seq_t[-1]:.2f}s)")
    print(f"  * Distributed Async BO Final Best R^2 : {dist_best[-1]:.4f}  (Time: {dist_t[-1]:.2f}s)")
    print(f"  * Parallel Speedup Achieved           : {dist_res.speedup_ratio:.2f}x over sequential execution")
    print("=" * 80)


if __name__ == "__main__":
    main()
