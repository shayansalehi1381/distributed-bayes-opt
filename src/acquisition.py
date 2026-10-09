"""Acquisition functions for Bayesian optimization over a GP posterior.

All acquisitions consume the posterior mean mu(x) and variance sigma^2(x)
of the `GaussianProcessRegressor` from Step 1 and are fully vectorized over
candidate batches. Convention: **maximization** of the objective f; every
acquisition returns values to be maximized.

Implemented:

* **Expected Improvement (EI)** — with the full analytical gradient. For
  improvement I(x) = max(f(x) - f* - xi, 0) and z = (mu - f* - xi) / sigma:

      EI(x) = (mu - f* - xi) * Phi(z) + sigma * phi(z)

  Closed-form partials (used for the analytical chain rule):

      dEI/dmu    = Phi(z)
      dEI/dsigma = phi(z)

  and through the GP posterior, dmu/dx and dsigma/dx follow from kernel
  derivatives. `ExpectedImprovement.gradient()` returns the exact
  analytical dEI/dx for the RBF kernel, verified in tests against autograd.

* **Upper Confidence Bound (UCB)** — mu + sqrt(beta_t) * sigma with the
  *dynamic* trade-off schedule of Srinivas et al. (GP-UCB, 2010):

      beta_t = 2 log(|D| t^2 pi^2 / (6 delta))

  which yields sublinear regret guarantees; beta grows O(log t) so
  exploration never vanishes.

* **Thompson Sampling (TS)** — draws joint posterior sample paths via
  `GaussianProcessRegressor.sample_posterior` (Step 1) and returns the
  argmax per path; a randomized acquisition whose choice probability equals
  the posterior probability of being the maximizer.

Normal pdf/cdf are implemented from scratch via `torch.erf` — no
distribution objects from high-level libraries.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from src.gp_regression import GaussianProcessRegressor

__all__ = [
    "ExpectedImprovement",
    "UpperConfidenceBound",
    "ThompsonSampling",
]

_SQRT2 = math.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def _phi(z: Tensor) -> Tensor:
    """Standard normal pdf."""
    return _INV_SQRT_2PI * torch.exp(-0.5 * z**2)


def _Phi(z: Tensor) -> Tensor:
    """Standard normal cdf via the error function."""
    return 0.5 * (1.0 + torch.erf(z / _SQRT2))


class ExpectedImprovement:
    """EI(x) = E[max(f(x) - f* - xi, 0)] under the GP posterior (maximization).

    Args:
        model: fitted GaussianProcessRegressor.
        best_f: incumbent value f*. If None, taken as max of the training
            targets at evaluation time.
        xi: exploration offset; larger xi discounts small improvements.
    """

    def __init__(
        self,
        model: GaussianProcessRegressor,
        best_f: float | None = None,
        xi: float = 0.0,
    ) -> None:
        self.model = model
        self.best_f = best_f
        self.xi = float(xi)

    def _incumbent(self) -> float:
        if self.best_f is not None:
            return self.best_f
        if self.model._y_train is None:
            raise RuntimeError("Model must be fitted or best_f provided.")
        return float(self.model._y_train.max())

    def __call__(self, x: Tensor) -> Tensor:
        """EI values, shape (m,). Zero (not NaN) where sigma == 0."""
        with torch.no_grad():
            mean, var = self.model.posterior_diag(x)
        return self._ei(mean, var.sqrt())

    def _ei(self, mean: Tensor, sigma: Tensor) -> Tensor:
        improve = mean - self._incumbent() - self.xi
        # Guard sigma=0: z is finite via clamp, and the sigma*phi term
        # vanishes; EI degenerates to max(improve, 0) as it must.
        safe_sigma = sigma.clamp_min(1e-12)
        z = improve / safe_sigma
        ei = improve * _Phi(z) + safe_sigma * _phi(z)
        return torch.where(sigma > 0, ei, improve.clamp_min(0.0)).clamp_min(0.0)

    def gradient(self, x: Tensor) -> Tensor:
        """Full analytical gradient dEI/dx, shape (m, d).

        Chain rule with the closed-form partials

            dEI/dx = Phi(z) * dmu/dx + phi(z) * dsigma/dx

        (the z-dependent inner terms cancel exactly — the classic EI
        gradient identity). dmu/dx and dsigma/dx are computed analytically
        from RBF kernel derivatives:

            dk(x, xi)/dx = -k(x, xi) * (x - xi) / l^2
            dmu/dx       = sum_i alpha_i dk(x, xi)/dx
            dsigma^2/dx  = -2 K_x' Kinv k_x   =>  dsigma/dx = dsigma^2/dx / (2 sigma)
        """
        model = self.model
        model._require_fitted()
        xq, _ = model._validate_inputs(x)
        assert model._x_train is not None and model._alpha is not None
        assert model._chol is not None
        xt = model._x_train                                     # (n, d)
        ls2 = (model.kernel.lengthscale.detach() ** 2)          # (d,) or (1,)

        with torch.no_grad():
            k_star = model.kernel(xt, xq)                       # (n, m)
            # dk/dxq: (m, n, d) = k(xq, xi) * (xi - xq) / l^2
            diff = (xt.unsqueeze(0) - xq.unsqueeze(1)) / ls2    # (m, n, d)
            dk = k_star.T.unsqueeze(-1) * diff                  # (m, n, d)

            dmu = torch.einsum("mnd,n->md", dk, model._alpha.reshape(-1))

            # sigma^2 = k_xx - k_*^T Kinv k_*  with k_xx constant for RBF:
            # dsigma^2/dx = -2 (Kinv k_*)^T dk/dx
            kinv_kstar = torch.cholesky_solve(k_star, model._chol)  # (n, m)
            dvar = -2.0 * torch.einsum("mnd,nm->md", dk, kinv_kstar)

            mean, var = model.posterior_diag(xq)
            sigma = var.sqrt().clamp_min(1e-12)
            dsigma = dvar / (2.0 * sigma.unsqueeze(-1))

            z = (mean - self._incumbent() - self.xi) / sigma
            grad = _Phi(z).unsqueeze(-1) * dmu + _phi(z).unsqueeze(-1) * dsigma
        return grad


class UpperConfidenceBound:
    """GP-UCB: alpha(x) = mu(x) + sqrt(beta_t) * sigma(x) (maximization).

    The trade-off parameter follows the dynamic schedule of Srinivas et al.
    (2010), Theorem 1 (finite candidate sets):

        beta_t = 2 log(|D| * t^2 * pi^2 / (6 * delta))

    Growing beta ~ O(log t) keeps exploring forever at a diminishing rate —
    the ingredient behind GP-UCB's sublinear cumulative regret. Call
    `step()` once per BO iteration to advance t, or pass a fixed `beta` to
    disable the schedule.

    Args:
        model: fitted GaussianProcessRegressor.
        beta: fixed trade-off; overrides the schedule when given.
        delta: schedule confidence level (regret bound holds w.p. 1-delta).
        domain_size: |D|, the candidate-set cardinality in the schedule.
    """

    def __init__(
        self,
        model: GaussianProcessRegressor,
        beta: float | None = None,
        delta: float = 0.1,
        domain_size: int = 1000,
    ) -> None:
        self.model = model
        self.fixed_beta = beta
        self.delta = float(delta)
        self.domain_size = int(domain_size)
        self.t = 1

    @property
    def beta(self) -> float:
        if self.fixed_beta is not None:
            return self.fixed_beta
        return 2.0 * math.log(
            self.domain_size * self.t**2 * math.pi**2 / (6.0 * self.delta)
        )

    def step(self) -> None:
        """Advance the iteration counter of the beta schedule."""
        self.t += 1

    def __call__(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            mean, var = self.model.posterior_diag(x)
        return mean + math.sqrt(self.beta) * var.sqrt()


class ThompsonSampling:
    """Thompson sampling over a candidate set (maximization).

    Draws joint sample paths f ~ posterior via the Step 1
    `sample_posterior` (full covariance, so samples respect correlations
    between candidates — crucial for TS to be exact on the discrete set)
    and selects each path's argmax. Selection frequencies converge to the
    posterior probability of each candidate being the maximizer.
    """

    def __init__(self, model: GaussianProcessRegressor) -> None:
        self.model = model

    def __call__(
        self, x: Tensor, n_samples: int = 1, generator: torch.Generator | None = None
    ) -> Tensor:
        """Sampled function values, shape (n_samples, m) — one path per row."""
        return self.model.sample_posterior(x, n_samples=n_samples, generator=generator)

    def select(
        self, x: Tensor, n_select: int = 1, generator: torch.Generator | None = None
    ) -> Tensor:
        """Indices into `x` chosen by n_select independent posterior draws.

        With n_select > 1 this is the standard parallel/batch TS: each draw
        is an independent path, so duplicates are possible by design.
        Returns shape (n_select,).
        """
        paths = self(x, n_samples=n_select, generator=generator)  # (s, m)
        return paths.argmax(dim=-1)
