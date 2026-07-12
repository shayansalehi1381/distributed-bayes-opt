"""run_pipeline.py — Step 5: Full Integrated Pipeline

Demonstrates the integration of the Scalable Sparse Gaussian Process with the
Double DQN RL decision pipeline. The RL agent selects points to sample, and the
Sparse GP updates its global surrogate model in real-time, augmenting the agent's
state space with posterior mean and variance.
"""

from __future__ import annotations

import os
# Suppress Windows/Anaconda duplicate-OpenMP-runtime conflict between
# PyTorch and NumPy/matplotlib.  Must be set before importing either.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import math
import pathlib

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import torch
from torch import Tensor

from src.environment import ContinuousEnvironment, EnvConfig
from src.rl_agent import DoubleDQNAgent
from src.sparse_gp import SparseGPRegressor, greedy_inducing_init

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DIM = 2
N_ACTIONS = 50
BUDGET = 20
N_EPISODES = 10
TOTAL_STEPS = N_EPISODES * BUDGET
BATCH_SIZE = 32

DTYPE = torch.float64
DEVICE = "cpu"
SEED = 42

def linear_epsilon(step: int) -> float:
    """Linearly anneal epsilon from 1.0 to 0.1 over TOTAL_STEPS."""
    fraction = min(step / max(TOTAL_STEPS - 1, 1), 1.0)
    return 1.0 + fraction * (0.1 - 1.0)

def get_augmented_state(base_state: Tensor, x_query: Tensor, gp: SparseGPRegressor | None) -> Tensor:
    """Augment the base RL state with the GP's posterior mean and variance."""
    if gp is None or gp._y_train is None:
        mean, var = 0.0, 1.0
    else:
        with torch.no_grad():
            mean_t, var_t = gp.posterior_diag(x_query.unsqueeze(0))
            mean, var = float(mean_t.squeeze()), float(var_t.squeeze())
            
    gp_stats = torch.tensor([mean, var], dtype=DTYPE, device=DEVICE)
    return torch.cat([base_state, gp_stats])

def plot_pipeline_convergence(
    rewards: list[float],
    losses: list[tuple[int, float]],
    save_path: str = "pipeline_convergence.png"
) -> None:
    plt.style.use("dark_background")
    ACCENT = "#7eb8f7"
    ACCENT2 = "#f7a27e"
    BG = "#0f1923"
    GRID_COL = "#2e3a4e"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), facecolor=BG, constrained_layout=True)
    fig.suptitle("Step 5: RL + Sparse GP Pipeline Convergence", color="white", fontsize=14, fontweight="bold")

    eps = list(range(1, len(rewards) + 1))
    
    # Left: Rewards
    ax1.set_facecolor(BG)
    ax1.plot(eps, rewards, color=ACCENT, marker="o", linewidth=2, markersize=5)
    ax1.fill_between(eps, rewards, alpha=0.15, color=ACCENT)
    ax1.set_title("Total Reward per Episode", color="white")
    ax1.set_xlabel("Episode", color="#aab4be")
    ax1.set_ylabel("Reward", color="#aab4be")
    ax1.grid(True, color=GRID_COL, linestyle="--")
    ax1.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax1.tick_params(colors="#aab4be")
    for spine in ax1.spines.values(): spine.set_color(GRID_COL)

    # Right: Losses
    if losses:
        loss_eps, loss_vals = zip(*losses)
        ax2.set_facecolor(BG)
        ax2.plot(loss_eps, loss_vals, color=ACCENT2, marker="s", linewidth=2, markersize=5)
        ax2.fill_between(loss_eps, loss_vals, alpha=0.15, color=ACCENT2)
        ax2.set_title("DQN Huber Loss", color="white")
        ax2.set_xlabel("Episode", color="#aab4be")
        ax2.set_ylabel("Loss", color="#aab4be")
        ax2.grid(True, color=GRID_COL, linestyle="--")
        ax2.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax2.tick_params(colors="#aab4be")
        for spine in ax2.spines.values(): spine.set_color(GRID_COL)

    fig.savefig(save_path, dpi=150, facecolor=BG)
    plt.close(fig)
    print(f"\nPlot saved to -> {pathlib.Path(save_path).resolve()}")

def main() -> None:
    print("=" * 80)
    print("  Step 5: Full Integrated Pipeline (Sparse GP + Double DQN)")
    print("=" * 80)

    # 1. Initialize Environment
    cfg = EnvConfig(dim=DIM, n_actions=N_ACTIONS, budget=BUDGET, noise_std=0.05, seed=SEED, dtype=DTYPE, device=DEVICE)
    env = ContinuousEnvironment(config=cfg)

    # Augmented state dimension (+2 for GP mean and variance)
    state_dim = env.state_dim + 2
    
    # 2. Initialize Agent
    agent = DoubleDQNAgent(
        state_dim=state_dim,
        n_actions=N_ACTIONS,
        hidden_dim=128,
        n_layers=2,
        buffer_capacity=10_000,
        dtype=DTYPE,
        device=DEVICE
    )

    print(f"\nEnvironment State Dim: {env.state_dim}")
    print(f"Augmented State Dim  : {state_dim} (Added GP Mean and Variance)")
    print("-" * 80)
    print(f"{'Ep':>3} | {'Step':>4} | {'Act':>3} | {'Reward':>8} | {'GP Mean':>8} | {'GP Var':>8} | {'Loss':>8}")
    print("-" * 80)

    global_step = 0
    all_rewards = []
    all_losses = []
    
    # Global datasets for the surrogate model
    X_history: list[Tensor] = []
    Y_history: list[Tensor] = []

    for ep in range(1, N_EPISODES + 1):
        # Seed differently per episode
        base_state = env.reset(seed=SEED + ep * 100)
        
        # Add the first randomly initialized point
        X_history.append(env._x_last.clone())
        Y_history.append(torch.tensor(env._y_last, dtype=DTYPE, device=DEVICE))
        
        # Initial GP fitting
        gp = None
        x_train = torch.stack(X_history)
        x_unique = torch.unique(x_train, dim=0)
        if len(x_unique) >= 2:
            y_train = torch.stack(Y_history)
            m = min(16, len(x_unique))
            z = greedy_inducing_init(x_unique, m=m)
            gp = SparseGPRegressor(z, learn_inducing=True, dtype=DTYPE, device=DEVICE)
            # Use collapsed VFE bound since datasets are small (up to ~200 points)
            gp.fit_collapsed(x_train, y_train, n_iters=15, verbose=False)
            
        state = get_augmented_state(base_state, env._x_last, gp)
        
        total_reward = 0.0
        episode_loss = None
        
        for step in range(BUDGET):
            epsilon = linear_epsilon(global_step)
            
            # Action informed by GP-augmented state
            action = agent.select_action(state, epsilon=epsilon)
            
            # Step environment
            next_base_state, reward, done, info = env.step(action)
            total_reward += reward
            
            x_new = info["x"]
            y_new = info["y_noisy"]
            
            X_history.append(x_new.clone())
            Y_history.append(torch.tensor(y_new, dtype=DTYPE, device=DEVICE))
            
            # Update global surrogate model in real-time
            x_train = torch.stack(X_history)
            y_train = torch.stack(Y_history)
            x_unique = torch.unique(x_train, dim=0)
            
            gp_mean, gp_var = 0.0, 1.0
            if len(x_unique) >= 2:
                m = min(16, len(x_unique))
                z = greedy_inducing_init(x_unique, m=m)
                gp = SparseGPRegressor(z, learn_inducing=True, dtype=DTYPE, device=DEVICE)
                gp.fit_collapsed(x_train, y_train, n_iters=15, verbose=False)
                
                with torch.no_grad():
                    mean_t, var_t = gp.posterior_diag(x_new.unsqueeze(0))
                    gp_mean, gp_var = float(mean_t.squeeze()), float(var_t.squeeze())
                    
            next_state = get_augmented_state(next_base_state, x_new, gp)
            
            # RL Orchestrator stores transition and learns
            agent.store(state, action, reward, next_state, done)
            
            loss = None
            if len(agent.replay_buffer) >= BATCH_SIZE:
                loss = agent.update(batch_size=BATCH_SIZE)
                if loss is not None:
                    episode_loss = loss
                    
            # Logging
            loss_str = f"{loss:.4f}" if loss is not None else "     n/a"
            print(f"{ep:3d} | {step+1:4d} | {action:3d} | {reward:8.4f} | {gp_mean:8.4f} | {gp_var:8.4f} | {loss_str:>8}")
            
            state = next_state
            global_step += 1
            if done:
                break
                
        all_rewards.append(total_reward)
        if episode_loss is not None:
            all_losses.append((ep, episode_loss))

    print("-" * 80)
    print("Optimization Complete.")
    
    # 5. Save Summary Graph
    plot_pipeline_convergence(all_rewards, all_losses, "pipeline_convergence.png")
    print("Done [OK]")

if __name__ == "__main__":
    main()
