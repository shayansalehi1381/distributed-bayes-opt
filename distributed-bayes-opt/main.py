"""main.py — Double DQN demonstration loop for distributed-bayes-opt.

Runs a complete training session across 20 episodes, where the Double DQN
agent learns to navigate the ContinuousEnvironment (a noisy Branin-like
objective) using epsilon-greedy exploration with linear decay.

Usage
-----
    python main.py

No arguments required.  Adjust the constants at the top of the file to
experiment with different settings.
"""

from __future__ import annotations

import math
import os

# Suppress Windows/Anaconda duplicate-OpenMP-runtime conflict between
# PyTorch and NumPy/matplotlib.  Must be set before importing either.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import pathlib

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — no display required
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import torch

from src.environment import ContinuousEnvironment, EnvConfig
from src.rl_agent import DoubleDQNAgent

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

# Environment
DIM: int = 2            # Search-space dimensionality
N_ACTIONS: int = 50     # Discrete candidate pool size
BUDGET: int = 20        # Steps per episode (function-evaluation budget)
NOISE_STD: float = 0.05 # Observation noise

# Training
N_EPISODES: int = 20    # Total number of training episodes
BATCH_SIZE: int = 32    # Replay-buffer mini-batch size

# Epsilon-greedy schedule (linear decay from EPS_START → EPS_END over all steps)
EPS_START: float = 1.0
EPS_END: float = 0.10
TOTAL_STEPS: int = N_EPISODES * BUDGET  # used for linear decay denominator

# Agent
HIDDEN_DIM: int = 128
N_LAYERS: int = 2
LR: float = 1e-3
GAMMA: float = 0.99
TARGET_UPDATE_FREQ: int = 50   # hard target-net sync every N gradient steps
BUFFER_CAPACITY: int = 10_000

SEED: int = 42
DTYPE = torch.float64
DEVICE = "cpu"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def linear_epsilon(step: int) -> float:
    """Linearly anneal epsilon from EPS_START to EPS_END."""
    fraction = min(step / max(TOTAL_STEPS - 1, 1), 1.0)
    return EPS_START + fraction * (EPS_END - EPS_START)


def plot_training_performance(
    rewards: list[float],
    losses: list[tuple[int, float]],
    save_path: str | pathlib.Path = "rl_training_performance.png",
) -> None:
    """Render and save a 1x2 training-performance figure.

    Parameters
    ----------
    rewards:
        Total reward per episode (length = N_EPISODES).
    losses:
        List of (episode_number, huber_loss) tuples — episodes where the
        buffer was not yet warm are omitted (no n/a entries).
    save_path:
        Destination path for the PNG output.
    """
    # ---- style ----
    plt.style.use("dark_background")
    ACCENT   = "#7eb8f7"   # soft blue  — reward line
    ACCENT2  = "#f7a27e"   # soft amber — loss line
    GRID_COL = "#2e3a4e"
    BG       = "#0f1923"

    fig, (ax_r, ax_l) = plt.subplots(
        1, 2,
        figsize=(12, 4.5),
        facecolor=BG,
        constrained_layout=True,
        gridspec_kw={"wspace": 0.35},
    )
    fig.suptitle(
        "Double DQN — Training Performance",
        fontsize=15,
        fontweight="bold",
        color="white",
        y=1.02,
    )

    episodes_all = list(range(1, len(rewards) + 1))
    loss_eps    = [ep  for ep, _ in losses]
    loss_vals   = [val for _, val in losses]

    # ---- left: Total Reward ----
    ax_r.set_facecolor(BG)
    ax_r.plot(
        episodes_all, rewards,
        color=ACCENT, linewidth=2.0, marker="o",
        markersize=5, markerfacecolor="white", markeredgewidth=1.2,
        zorder=3,
    )
    ax_r.fill_between(episodes_all, rewards, alpha=0.12, color=ACCENT)
    ax_r.set_title("Episode vs. Total Reward", color="white", fontsize=12, pad=10)
    ax_r.set_xlabel("Episode", color="#aab4be", fontsize=10)
    ax_r.set_ylabel("Total Reward", color="#aab4be", fontsize=10)
    ax_r.tick_params(colors="#aab4be")
    ax_r.spines[:].set_color(GRID_COL)
    ax_r.grid(True, color=GRID_COL, linewidth=0.6, linestyle="--")
    ax_r.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # ---- right: Huber Loss ----
    ax_l.set_facecolor(BG)
    ax_l.plot(
        loss_eps, loss_vals,
        color=ACCENT2, linewidth=2.0, marker="s",
        markersize=5, markerfacecolor="white", markeredgewidth=1.2,
        zorder=3,
    )
    ax_l.fill_between(loss_eps, loss_vals, alpha=0.12, color=ACCENT2)
    ax_l.set_title("Episode vs. Huber Loss", color="white", fontsize=12, pad=10)
    ax_l.set_xlabel("Episode", color="#aab4be", fontsize=10)
    ax_l.set_ylabel("Huber Loss", color="#aab4be", fontsize=10)
    ax_l.tick_params(colors="#aab4be")
    ax_l.spines[:].set_color(GRID_COL)
    ax_l.grid(True, color=GRID_COL, linewidth=0.6, linestyle="--")
    ax_l.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"\nPlot saved -> {pathlib.Path(save_path).resolve()}")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("  Distributed Bayesian Optimization — RL Demo (Step 4)")
    print("=" * 60)

    # ---- Environment ----
    cfg = EnvConfig(
        dim=DIM,
        n_actions=N_ACTIONS,
        budget=BUDGET,
        noise_std=NOISE_STD,
        seed=SEED,
        dtype=DTYPE,
        device=DEVICE,
    )
    env = ContinuousEnvironment(config=cfg)

    state_dim: int = env.state_dim  # 2 * DIM + 3

    print(f"\nEnvironment : {env}")
    print(f"State dim   : {state_dim}  (= 2×{DIM} + 3)")

    # ---- Agent ----
    agent = DoubleDQNAgent(
        state_dim=state_dim,
        n_actions=env.n_actions,
        hidden_dim=HIDDEN_DIM,
        n_layers=N_LAYERS,
        lr=LR,
        gamma=GAMMA,
        target_update_freq=TARGET_UPDATE_FREQ,
        buffer_capacity=BUFFER_CAPACITY,
        buffer_seed=SEED,
        dtype=DTYPE,
        device=DEVICE,
    )

    print(f"Agent       : {agent}\n")
    print(f"{'Episode':>8}  {'Total Reward':>13}  {'Best Reward':>11}  "
          f"{'Epsilon':>8}  {'Huber Loss':>11}")
    print("-" * 60)

    global_step: int = 0  # tracks total environment steps for epsilon decay

    # ---- tracking lists for visualization ----
    all_rewards: list[float] = []
    all_losses:  list[tuple[int, float]] = []  # (episode, loss) — n/a episodes omitted

    for episode in range(1, N_EPISODES + 1):
        state = env.reset(seed=SEED + episode)

        total_reward: float = 0.0
        best_reward: float = -math.inf
        episode_loss: float | None = None

        for _ in range(BUDGET):
            epsilon = linear_epsilon(global_step)

            # 1. Action selection
            action = agent.select_action(state, epsilon=epsilon)

            # 2. Environment step
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            best_reward = max(best_reward, reward)

            # 3. Store transition
            agent.store(state, action, reward, next_state, done)

            # 4. Update (only once buffer is warm enough)
            if len(agent.replay_buffer) >= BATCH_SIZE:
                loss = agent.update(batch_size=BATCH_SIZE)
                if loss is not None:
                    episode_loss = loss  # keep the last loss value for logging

            state = next_state
            global_step += 1

            if done:
                break

        # ---- Per-episode summary ----
        loss_str = f"{episode_loss:.6f}" if episode_loss is not None else "     n/a"
        print(
            f"{episode:>8}  {total_reward:>13.4f}  {best_reward:>11.4f}  "
            f"{epsilon:>8.4f}  {loss_str:>11}"
        )

        # ---- record for plotting ----
        all_rewards.append(total_reward)
        if episode_loss is not None:
            all_losses.append((episode, episode_loss))

    print("-" * 60)
    print(f"\nTraining complete.  Final agent state: {agent}")

    # ---- visualization ----
    plot_training_performance(
        rewards=all_rewards,
        losses=all_losses,
        save_path="rl_training_performance.png",
    )

    print("\nDone [OK]")


if __name__ == "__main__":
    main()
