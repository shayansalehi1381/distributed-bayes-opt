"""Advanced stochastic process sampling — from scratch.

Two complementary samplers for the Bayesian optimization stack:

1. **Metropolis–Hastings MCMC** (`MetropolisHastings`, `sample_gp_hyperposterior`)
   Random-walk MH over an arbitrary unnormalized log-density. Used to sample
   the *hyperparameter posterior* of the GP,

       p(theta | X, y)  ∝  p(y | X, theta) · p(theta),

   where theta = (lengthscales, signal variance, noise variance). This gives
   a fully Bayesian treatment of kernel uncertainty instead of a single
   type-II ML point estimate: acquisition values can then be averaged over
   hyperparameter samples (marginalized acquisition).

   Sampling happens in the model's *unconstrained* (raw, pre-softplus)
   parameterization, so the chain never needs boundary handling; the prior
   is placed on the constrained values with the softplus log-Jacobian
   correction applied automatically.

2. **Langevin dynamics via Euler–Maruyama** (`LangevinSampler`)
   The overdamped Langevin SDE

       dX_t = ∇ log π(X_t) dt + sqrt(2) dW_t

   has π as its stationary distribution. Its Euler–Maruyama discretization

       x_{k+1} = x_k + eps · ∇ log π(x_k) + sqrt(2 eps) · z_k,  z_k ~ N(0, I)

   is ULA (unadjusted Langevin). With an optional MH correction it becomes
   MALA, which removes the discretization bias exactly. Gradients come from
   autograd, so any differentiable log-density works — including GP
   posterior-based acquisition surfaces for gradient-based candidate
   generation in continuous spaces.

Both samplers are fully vectorized over independent chains: `n_chains`
states are propagated as one (c, d) tensor per step, with per-chain
accept/reject masks — no Python loop over chains.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor

from src.gp_regression import GaussianProcessRegressor

__all__ = [
    "MetropolisHastings",
    "LangevinSampler",
    "MCMCResult",
    "sample_gp_hyperposterior",
    "potential_scale_reduction",
]

LogDensity = Callable[[Tensor], Tensor]


@dataclass(frozen=True)
class MCMCResult:
    """Output of an MCMC run.

    Attributes:
        samples: (n_samples, n_chains, d) retained post-burn-in draws.
        acceptance_rate: per-chain fraction of accepted proposals, (n_chains,).
        log_probs: (n_samples, n_chains) log-density of each retained draw.
    """

    samples: Tensor
    acceptance_rate: Tensor
    log_probs: Tensor

    def flat(self) -> Tensor:
        """Pooled samples across chains, shape (n_samples * n_chains, d)."""
        return self.samples.reshape(-1, self.samples.shape[-1])


def _as_batch(x: Tensor) -> Tensor:
    return x.unsqueeze(0) if x.ndim == 1 else x


class MetropolisHastings:
    """Random-walk Metropolis–Hastings over an unnormalized log-density.

    Proposal: x' = x + step_size * z, z ~ N(0, I) — symmetric, so the
    acceptance ratio reduces to the Metropolis form

        a(x -> x') = min(1, pi(x') / pi(x)).

    `log_density` must accept a batch (c, d) and return (c,) log-densities
    (unnormalized is fine); all chains are advanced in one vectorized step.

    Args:
        log_density: batched unnormalized log-density.
        step_size: isotropic proposal scale. Rule of thumb: tune toward
            ~0.23 acceptance in high dimensions, ~0.44 in 1-D.
        adapt: if True, adapt step_size during burn-in toward
            `target_acceptance` via Robbins–Monro on the log-scale
            (adaptation stops at sampling time, preserving detailed balance
            for the retained draws).
    """

    def __init__(
        self,
        log_density: LogDensity,
        step_size: float = 0.1,
        adapt: bool = True,
        target_acceptance: float = 0.3,
    ) -> None:
        self.log_density = log_density
        self.step_size = float(step_size)
        self.adapt = adapt
        self.target_acceptance = float(target_acceptance)

    def sample(
        self,
        initial: Tensor,
        n_samples: int = 1000,
        burn_in: int = 500,
        thin: int = 1,
        generator: torch.Generator | None = None,
    ) -> MCMCResult:
        """Run the chain(s).

        Args:
            initial: (d,) or (n_chains, d) starting states.
            n_samples: retained draws per chain (after burn-in and thinning).
            burn_in: discarded warm-up iterations (step size adapts here).
            thin: keep every `thin`-th post-burn-in state.
            generator: torch RNG for reproducibility.

        Returns:
            MCMCResult with samples of shape (n_samples, n_chains, d).
        """
        x = _as_batch(torch.as_tensor(initial).detach().clone())
        n_chains, d = x.shape
        logp = self.log_density(x)
        if logp.shape != (n_chains,):
            raise ValueError(
                f"log_density must map (c, d) -> (c,); got {tuple(logp.shape)} for c={n_chains}"
            )

        step = self.step_size
        total = burn_in + n_samples * thin
        kept, kept_logp = [], []
        accepted = torch.zeros(n_chains, dtype=x.dtype)

        for it in range(total):
            z = torch.randn(n_chains, d, dtype=x.dtype, device=x.device, generator=generator)
            proposal = x + step * z
            logp_prop = self.log_density(proposal)
            # Metropolis accept: log u < logp' - logp   (vectorized per chain)
            log_u = torch.rand(n_chains, dtype=x.dtype, device=x.device, generator=generator).log()
            accept = log_u < (logp_prop - logp)
            x = torch.where(accept.unsqueeze(-1), proposal, x)
            logp = torch.where(accept, logp_prop, logp)

            if it < burn_in:
                if self.adapt:
                    # Robbins–Monro on log(step): decaying gain guarantees
                    # adaptation vanishes; direction follows the mean
                    # acceptance deviation from target.
                    gain = min(1.0, 10.0 / (it + 1) ** 0.6)
                    rate = float(accept.to(x.dtype).mean())
                    step = float(step * math.exp(gain * (rate - self.target_acceptance)))
            else:
                accepted += accept.to(x.dtype)
                if (it - burn_in) % thin == 0:
                    kept.append(x.clone())
                    kept_logp.append(logp.clone())

        self.step_size = step  # expose the adapted scale
        return MCMCResult(
            samples=torch.stack(kept),
            acceptance_rate=accepted / (n_samples * thin),
            log_probs=torch.stack(kept_logp),
        )


class LangevinSampler:
    """Langevin dynamics via Euler–Maruyama discretization of the SDE

        dX_t = ∇ log π(X_t) dt + sqrt(2) dW_t.

    One Euler–Maruyama step with step size eps:

        x' = x + eps * ∇ log π(x) + sqrt(2 eps) * z,   z ~ N(0, I).

    Without correction this is ULA — asymptotically biased at O(eps), but
    cheap and gradient-guided. With `metropolis_adjust=True` each step is
    accepted/rejected under the *asymmetric* Gaussian proposal density

        q(x' | x) = N(x' ; x + eps ∇ log π(x), 2 eps I),

    giving MALA, whose stationary distribution is exactly π.

    Gradients are obtained through autograd, so `log_density` only needs to
    be differentiable — no manual gradient required.
    """

    def __init__(
        self,
        log_density: LogDensity,
        step_size: float = 1e-2,
        metropolis_adjust: bool = True,
        grad_clip: float | None = 1e3,
    ) -> None:
        self.log_density = log_density
        self.step_size = float(step_size)
        self.metropolis_adjust = metropolis_adjust
        self.grad_clip = grad_clip

    def _logp_and_grad(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Batched (c,) log-density and (c, d) score ∇ log π via autograd."""
        x = x.detach().requires_grad_(True)
        logp = self.log_density(x)
        (grad,) = torch.autograd.grad(logp.sum(), x)
        if self.grad_clip is not None:
            # Per-chain norm clip: keeps EM steps stable in heavy-tailed regions.
            norm = grad.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            grad = grad * (norm.clamp_max(self.grad_clip) / norm)
        return logp.detach(), grad.detach()

    @staticmethod
    def _log_q(x_to: Tensor, x_from: Tensor, grad_from: Tensor, eps: float) -> Tensor:
        """log q(x_to | x_from) under the EM proposal N(x_from + eps*grad, 2*eps I),
        dropping the constant normalizer (cancels in the MH ratio)."""
        mean = x_from + eps * grad_from
        return -((x_to - mean) ** 2).sum(-1) / (4.0 * eps)

    def sample(
        self,
        initial: Tensor,
        n_samples: int = 1000,
        burn_in: int = 500,
        thin: int = 1,
        generator: torch.Generator | None = None,
    ) -> MCMCResult:
        """Run ULA/MALA chains; same conventions as MetropolisHastings.sample."""
        x = _as_batch(torch.as_tensor(initial).detach().clone())
        n_chains, d = x.shape
        eps = self.step_size
        logp, grad = self._logp_and_grad(x)

        total = burn_in + n_samples * thin
        kept, kept_logp = [], []
        accepted = torch.zeros(n_chains, dtype=x.dtype)

        for it in range(total):
            z = torch.randn(n_chains, d, dtype=x.dtype, device=x.device, generator=generator)
            proposal = x + eps * grad + math.sqrt(2.0 * eps) * z
            logp_prop, grad_prop = self._logp_and_grad(proposal)

            if self.metropolis_adjust:
                # MALA ratio: pi(x')q(x|x') / pi(x)q(x'|x)
                log_alpha = (
                    logp_prop
                    - logp
                    + self._log_q(x, proposal, grad_prop, eps)
                    - self._log_q(proposal, x, grad, eps)
                )
                log_u = torch.rand(
                    n_chains, dtype=x.dtype, device=x.device, generator=generator
                ).log()
                accept = log_u < log_alpha
            else:
                accept = torch.ones(n_chains, dtype=torch.bool, device=x.device)

            x = torch.where(accept.unsqueeze(-1), proposal, x)
            logp = torch.where(accept, logp_prop, logp)
            grad = torch.where(accept.unsqueeze(-1), grad_prop, grad)

            if it >= burn_in:
                accepted += accept.to(x.dtype)
                if (it - burn_in) % thin == 0:
                    kept.append(x.clone())
                    kept_logp.append(logp.clone())

        return MCMCResult(
            samples=torch.stack(kept),
            acceptance_rate=accepted / (n_samples * thin),
            log_probs=torch.stack(kept_logp),
        )


# --------------------------------------------------------------------- #
# GP hyperparameter posterior sampling                                   #
# --------------------------------------------------------------------- #

def _gp_log_posterior_factory(
    model: GaussianProcessRegressor,
    x: Tensor,
    y: Tensor,
    prior_mean: float = 0.0,
    prior_std: float = 1.5,
) -> tuple[LogDensity, Tensor]:
    """Build log p(theta_raw | X, y) over the model's raw (unconstrained)
    hyperparameters, plus the current raw vector as the chain's start point.

    Prior: independent N(prior_mean, prior_std^2) on log-constrained values
    — i.e. a log-normal-style weakly informative prior on lengthscales and
    variances — with the softplus change-of-variables Jacobian included so
    the density is correct in raw space.
    """
    params = model.parameters()  # [raw_lengthscale (d_ls,), raw_sf2 (), raw_sn2 ()]
    shapes = [p.shape for p in params]
    sizes = [p.numel() for p in params]
    theta0 = torch.cat([p.detach().reshape(-1) for p in params])

    def set_raw(theta: Tensor) -> None:
        offset = 0
        for p, shape, size in zip(params, shapes, sizes):
            with torch.no_grad():
                p.copy_(theta[offset : offset + size].reshape(shape))
            offset += size

    def log_posterior(thetas: Tensor) -> Tensor:
        thetas = _as_batch(thetas)
        out = torch.empty(thetas.shape[0], dtype=thetas.dtype)
        for i, theta in enumerate(thetas):  # LML is O(n^3); loop, don't batch kernels
            set_raw(theta)
            try:
                with torch.no_grad():
                    lml = model.log_marginal_likelihood(x, y)
            except torch.linalg.LinAlgError:
                out[i] = -float("inf")
                continue
            # softplus(theta) has log-Jacobian log(sigmoid(theta)) per coord.
            constrained = torch.nn.functional.softplus(theta)
            log_prior = (
                -0.5 * ((constrained.log() - prior_mean) / prior_std) ** 2
            ).sum() + torch.nn.functional.logsigmoid(theta).sum() - constrained.log().sum()
            out[i] = lml + log_prior
        set_raw(theta0)  # leave the model untouched between density calls
        return out

    return log_posterior, theta0


def sample_gp_hyperposterior(
    model: GaussianProcessRegressor,
    x_train: Tensor,
    y_train: Tensor,
    n_samples: int = 200,
    n_chains: int = 4,
    burn_in: int = 300,
    step_size: float = 0.15,
    generator: torch.Generator | None = None,
) -> MCMCResult:
    """Sample the GP hyperparameter posterior p(theta | X, y) with MH.

    Chains start from the model's current raw hyperparameters, jittered per
    chain for overdispersed initialization (as R-hat diagnostics require).
    Samples live in raw (unconstrained) space; map through softplus to get
    lengthscales/variances. The model's parameters are restored afterwards.
    """
    x, y = model._validate_inputs(x_train, y_train)
    assert y is not None
    log_post, theta0 = _gp_log_posterior_factory(model, x, y)
    init = theta0.unsqueeze(0) + 0.5 * torch.randn(
        n_chains, theta0.numel(), dtype=theta0.dtype, generator=generator
    )
    mh = MetropolisHastings(log_post, step_size=step_size, adapt=True)
    return mh.sample(init, n_samples=n_samples, burn_in=burn_in, generator=generator)


def potential_scale_reduction(samples: Tensor) -> Tensor:
    """Gelman–Rubin R-hat from (n_samples, n_chains, d) draws.

    R-hat compares between-chain and within-chain variance; values near 1
    indicate the chains have mixed. Requires >= 2 chains.
    """
    n, m, _ = samples.shape
    if m < 2:
        raise ValueError("R-hat needs at least 2 chains")
    chain_means = samples.mean(0)                     # (m, d)
    grand_mean = chain_means.mean(0)                  # (d,)
    b = n / (m - 1) * ((chain_means - grand_mean) ** 2).sum(0)   # between
    w = samples.var(0, correction=1).mean(0)                     # within
    var_hat = (n - 1) / n * w + b / n
    return (var_hat / w).sqrt()
