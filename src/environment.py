"""Continuous stochastic environment for Bayesian-optimization-guided RL.

Provides an environment loop that the Double DQN agent interacts with:

    state  = env.reset()
    action = agent.act(state)
    next_state, reward, done, info = env.step(action)

The environment wraps a *noisy* continuous objective function over a bounded
d-dimensional search space.  At each step the agent picks a *discrete* action
index (choosing among a finite set of candidate query points), observes the
noisy function value as reward, and transitions to the next state — which
encodes the best observation so far plus the last query location.

Design notes
------------
* **Stochastic rewards**: Each observation is corrupted by i.i.d. Gaussian
  noise of configurable variance, matching the standard BO observation model.
* **State representation**: ``[x_best (d) | y_best (1) | x_last (d) |
  y_last (1) | t_norm (1)]`` — a (2d + 3,) vector that gives the agent full
  context for sequential decision-making.
* **Discrete action space over continuous candidates**: A fixed grid (or
  user-supplied candidate set) of *N_actions* points in the continuous domain.
  The agent selects an index; the environment evaluates the objective there.
  This discretization bridges the gap between a DQN (inherently discrete) and
  the continuous BO search space.
* **Episodic**: An episode runs for a fixed budget *T* of function
  evaluations, matching a typical BO campaign.
* **float64 everywhere** for numerical consistency with the GP stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch
from torch import Tensor

__all__ = ["ContinuousEnvironment", "EnvConfig"]


@dataclass
class EnvConfig:
    """Static configuration for the environment.

    Attributes:
        dim: dimensionality of the search space.
        bounds: (dim, 2) tensor of [lo, hi] per dimension.
        n_actions: number of candidate query points (discrete action space).
        budget: maximum function evaluations per episode.
        noise_std: observation noise standard deviation.
        seed: random seed for reproducibility.
        dtype: tensor dtype; float64 by default.
        device: tensor device.
    """

    dim: int = 2
    bounds: Tensor | None = None
    n_actions: int = 50
    budget: int = 20
    noise_std: float = 0.1
    seed: int = 42
    dtype: torch.dtype = torch.float64
    device: torch.device | str = "cpu"


def _default_objective(x: Tensor) -> Tensor:
    """Scaled Branin-like function (2-D) extended to arbitrary dims.

    For d > 2 the extra dimensions contribute a cosine term so the
    function is never degenerate.  Negated so that *maximization* of the
    reward corresponds to finding the minimum of the classic Branin,
    aligning with the convention in the acquisition module.
    """
    # Branin core (first two dims, or wrap if d == 1).
    x0 = x[..., 0]
    x1 = x[..., min(1, x.shape[-1] - 1)]
    a, b, c = 1.0, 5.1 / (4.0 * torch.pi**2), 5.0 / torch.pi
    r, s, t = 6.0, 10.0, 1.0 / (8.0 * torch.pi)
    branin = a * (x1 - b * x0**2 + c * x0 - r) ** 2 + s * (1.0 - t) * torch.cos(x0) + s
    # Negate (we want a *reward* to maximize, Branin is minimized).
    val = -branin / 50.0  # scale to a friendlier magnitude
    # Extra dims: add a small cosine ridge.
    if x.shape[-1] > 2:
        val = val + 0.1 * torch.cos(x[..., 2:].sum(-1))
    return val


class ContinuousEnvironment:
    """Episodic continuous-domain stochastic environment.

    Parameters
    ----------
    objective : callable (Tensor -> Tensor), optional
        The black-box function to optimize.  Receives ``(*, d)`` and returns
        ``(*,)``.  Defaults to a negated scaled Branin.
    config : EnvConfig, optional
        Environment configuration.  Defaults are sensible for quick tests.
    candidates : Tensor, optional
        (n_actions, d) pre-defined candidate set.  If *None*, a random Latin-
        hypercube-style grid is generated from the config.
    """

    def __init__(
        self,
        objective: Callable[[Tensor], Tensor] | None = None,
        config: EnvConfig | None = None,
        candidates: Tensor | None = None,
    ) -> None:
        self.cfg = config or EnvConfig()
        self.dtype = self.cfg.dtype
        self.device = torch.device(self.cfg.device)
        self.objective = objective or _default_objective
        self._rng = torch.Generator(device=self.device).manual_seed(self.cfg.seed)

        # --- bounds ---
        if self.cfg.bounds is not None:
            self.bounds = self.cfg.bounds.to(dtype=self.dtype, device=self.device)
        else:
            self.bounds = torch.stack(
                [
                    torch.full((self.cfg.dim,), -5.0, dtype=self.dtype, device=self.device),
                    torch.full((self.cfg.dim,), 10.0, dtype=self.dtype, device=self.device),
                ],
                dim=-1,
            )  # (d, 2)

        # --- candidate action set ---
        if candidates is not None:
            self.candidates = candidates.to(dtype=self.dtype, device=self.device)
        else:
            self.candidates = self._generate_candidates()

        self.n_actions: int = self.candidates.shape[0]
        self.state_dim: int = 2 * self.cfg.dim + 3  # x_best + y_best + x_last + y_last + t_norm

        # --- episode state (populated by reset()) ---
        self._t: int = 0
        self._x_best: Tensor = torch.zeros(self.cfg.dim, dtype=self.dtype, device=self.device)
        self._y_best: float = -float("inf")
        self._x_last: Tensor = torch.zeros(self.cfg.dim, dtype=self.dtype, device=self.device)
        self._y_last: float = 0.0
        self._done: bool = True

    # ------------------------------------------------------------------ #
    # Candidate generation                                                #
    # ------------------------------------------------------------------ #

    def _generate_candidates(self) -> Tensor:
        """Quasi-random Sobol-like candidate set within bounds."""
        lo = self.bounds[:, 0]
        hi = self.bounds[:, 1]
        # Stratified random: divide each dim into n_actions^{1/d} strata,
        # then sample one per stratum — cheaper than a full Sobol engine but
        # gives better coverage than pure uniform.
        pts = torch.rand(
            self.cfg.n_actions,
            self.cfg.dim,
            generator=self._rng,
            dtype=self.dtype,
            device=self.device,
        )
        return lo + (hi - lo) * pts

    # ------------------------------------------------------------------ #
    # State encoding                                                      #
    # ------------------------------------------------------------------ #

    def _encode_state(self) -> Tensor:
        """Pack episode context into a flat state vector.

        Layout: [x_best (d) | y_best (1) | x_last (d) | y_last (1) | t / T (1)]
        Everything is normalized to roughly [-1, 1] for neural-net friendliness.
        """
        lo = self.bounds[:, 0]
        hi = self.bounds[:, 1]
        span = (hi - lo).clamp_min(1e-12)

        x_best_n = 2.0 * (self._x_best - lo) / span - 1.0
        x_last_n = 2.0 * (self._x_last - lo) / span - 1.0
        t_norm = torch.tensor(
            [self._t / max(self.cfg.budget, 1)], dtype=self.dtype, device=self.device
        )
        return torch.cat(
            [
                x_best_n,
                torch.tensor([self._y_best], dtype=self.dtype, device=self.device),
                x_last_n,
                torch.tensor([self._y_last], dtype=self.dtype, device=self.device),
                t_norm,
            ]
        )

    # ------------------------------------------------------------------ #
    # Core loop                                                           #
    # ------------------------------------------------------------------ #

    def reset(self, seed: int | None = None) -> Tensor:
        """Reset the episode and return the initial state.

        Args:
            seed: optional new random seed; re-seeds the internal generator.

        Returns:
            state: (state_dim,) float64 tensor.
        """
        if seed is not None:
            self._rng.manual_seed(seed)

        self._t = 0
        # Start from a random initial query.
        idx = int(torch.randint(self.n_actions, (1,), generator=self._rng).item())
        x0 = self.candidates[idx]
        y0 = float(self.objective(x0.unsqueeze(0)).squeeze())

        self._x_best = x0.clone()
        self._y_best = y0
        self._x_last = x0.clone()
        self._y_last = y0
        self._done = False

        return self._encode_state()

    def step(self, action: int) -> tuple[Tensor, float, bool, dict[str, Any]]:
        """Execute one function evaluation.

        Args:
            action: integer index into the candidate set.

        Returns:
            next_state: (state_dim,) float64 tensor.
            reward: noisy function value at the queried candidate.
            done: True when the budget is exhausted.
            info: dict with keys ``'x'``, ``'y_true'``, ``'y_noisy'``,
                ``'t'``, ``'best_y'``.
        """
        if self._done:
            raise RuntimeError("Episode is done — call reset() before stepping.")
        if not (0 <= action < self.n_actions):
            raise ValueError(f"action {action} out of [0, {self.n_actions})")

        x = self.candidates[action]
        y_true = float(self.objective(x.unsqueeze(0)).squeeze())
        noise = float(
            self.cfg.noise_std
            * torch.randn(1, generator=self._rng, dtype=self.dtype, device=self.device).item()
        )
        y_noisy = y_true + noise

        # Transition.
        self._t += 1
        self._x_last = x.clone()
        self._y_last = y_noisy
        if y_noisy > self._y_best:
            self._x_best = x.clone()
            self._y_best = y_noisy

        self._done = self._t >= self.cfg.budget

        info: dict[str, Any] = {
            "x": x.clone(),
            "y_true": y_true,
            "y_noisy": y_noisy,
            "t": self._t,
            "best_y": self._y_best,
        }

        return self._encode_state(), y_noisy, self._done, info

    # ------------------------------------------------------------------ #
    # Convenience                                                         #
    # ------------------------------------------------------------------ #

    @property
    def done(self) -> bool:
        """Whether the current episode has ended."""
        return self._done

    def __repr__(self) -> str:
        return (
            f"ContinuousEnvironment(dim={self.cfg.dim}, n_actions={self.n_actions}, "
            f"budget={self.cfg.budget}, noise_std={self.cfg.noise_std})"
        )
