"""Sparse Gaussian Processes with inducing points — from scratch.

Implements the variational sparse GP framework in two flavors:

1. **VFE / Titsias (2009)** — `SparseGPRegressor.elbo_collapsed()`
   The collapsed variational bound for Gaussian likelihoods,

       L = log N(y | 0, Q_nn + sigma_n^2 I) - 1/(2 sigma_n^2) tr(K_nn - Q_nn),

   where Q_nn = K_nm K_mm^{-1} K_mn is the Nystrom approximation. The trace
   term penalizes inducing sets that fail to explain the training inputs,
   which is what makes VFE a *lower bound on the exact log marginal
   likelihood* — unlike FITC, it cannot overfit by pinching noise.

2. **SVGP / Hensman et al. (2013)** — `SparseGPRegressor.elbo_minibatch()`
   The uncollapsed bound with an explicit Gaussian variational posterior
   q(u) = N(m, S) over inducing values:

       L = sum_i E_{q(f_i)}[log p(y_i | f_i)] - KL(q(u) || p(u)),

   whose data term decomposes over observations, enabling unbiased
   mini-batch estimates (rescale by N/B) — this is Stochastic Variational
   Inference: training cost per step is O(B m^2 + m^3), independent of N.

Both share the sparse predictive equations. With A = K_mm^{-1} K_mu(x):

    q(f*) mean:      mu(x)      = A^T m
    q(f*) variance:  sigma^2(x) = k(x,x) - A^T (K_mm - S) A

Implementation notes
--------------------
* Everything is parameterized for unconstrained autograd training:
  softplus for noise (shared convention with Step 1's kernel), and S = L_S
  L_S^T with L_S a raw lower-triangular Cholesky factor whose diagonal goes
  through softplus for positivity.
* Inducing locations Z are plain learnable tensors, optimized jointly with
  kernel hyperparameters and variational parameters.
* All K_mm operations go through a jittered Cholesky (never an explicit
  inverse), float64 by default — K_mm conditioning degrades quickly as
  inducing points cluster during optimization, which is exactly when
  float32 breaks.
* Whitened parameterization is deliberately NOT used, to keep the math
  1:1 with the published bounds; the jittered Cholesky plus float64 is
  sufficient at portfolio scale (m <= a few thousand).

The predictive interface (`posterior_diag`, `predict`, `sample_posterior`,
`_y_train`) duck-types `GaussianProcessRegressor`, so every acquisition
function from Step 2 (EI values, UCB, Thompson sampling) composes with this
model unchanged. `EI.gradient()` relies on exact-GP internals; for sparse
models use autograd through `posterior_diag` (tested).
"""

from __future__ import annotations

import math
from typing import Iterator

import torch
from torch import Tensor

from src.gp_regression import GPPosterior, RBFKernel, _inv_softplus, _softplus

__all__ = ["SparseGPRegressor", "greedy_inducing_init"]

_LOG_2PI = math.log(2.0 * math.pi)


def greedy_inducing_init(x: Tensor, m: int, generator: torch.Generator | None = None) -> Tensor:
    """Pick m well-spread rows of x by greedy farthest-point selection.

    A cheap k-center heuristic: start from a random point, then repeatedly
    add the point farthest from the current set. O(n m) — far better spread
    than uniform subsampling when data is clustered, and deterministic given
    the generator.
    """
    n = x.shape[0]
    if m >= n:
        return x.clone()
    first = int(torch.randint(n, (1,), generator=generator))
    chosen = [first]
    d2 = ((x - x[first]) ** 2).sum(-1)
    for _ in range(m - 1):
        nxt = int(d2.argmax())
        chosen.append(nxt)
        d2 = torch.minimum(d2, ((x - x[nxt]) ** 2).sum(-1))
    return x[chosen].clone()


class SparseGPRegressor:
    """Sparse GP regression with learnable inducing points (VFE / SVGP).

    Args:
        inducing_points: (m, d) initial inducing locations Z; copied and
            promoted to a learnable parameter.
        kernel: covariance function; defaults to isotropic RBF.
        noise_variance: initial Gaussian observation noise sigma_n^2.
        learn_inducing: if False, Z stays fixed at its initialization.
        jitter: base diagonal jitter for the K_mm Cholesky.
        dtype / device: float64 strongly recommended — K_mm conditioning
            degrades as inducing points coalesce during training.
    """

    def __init__(
        self,
        inducing_points: Tensor,
        kernel: RBFKernel | None = None,
        noise_variance: float = 1e-2,
        learn_inducing: bool = True,
        jitter: float = 1e-6,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = "cpu",
    ) -> None:
        self.dtype = dtype
        self.device = torch.device(device)
        z = torch.as_tensor(inducing_points, dtype=dtype, device=self.device).detach().clone()
        if z.ndim == 1:
            z = z.unsqueeze(-1)
        self.inducing_points = z.requires_grad_(learn_inducing)
        m = z.shape[0]

        self.kernel = kernel or RBFKernel(dtype=dtype, device=device)
        self.raw_noise_variance = torch.tensor(
            _inv_softplus(float(noise_variance)), dtype=dtype, device=self.device, requires_grad=True
        )
        self.jitter = float(jitter)

        # Variational posterior q(u) = N(m, S), S = L L^T. Initialized at the
        # prior: m = 0, S = K_mm would be ideal, but identity is the standard
        # cheap init; raw diag is softplus-inverted so S starts at I.
        self.variational_mean = torch.zeros(m, dtype=dtype, device=self.device, requires_grad=True)
        raw_l = torch.zeros(m, m, dtype=dtype, device=self.device)
        raw_l.diagonal().fill_(_inv_softplus(1.0))
        self.raw_variational_chol = raw_l.requires_grad_(True)

        self._y_train: Tensor | None = None  # incumbent source for EI (duck-typed)
        self._y_mean: Tensor = torch.zeros((), dtype=dtype, device=self.device)

    # ------------------------------------------------------------------ #
    # Parameters & constrained views                                     #
    # ------------------------------------------------------------------ #

    @property
    def noise_variance(self) -> Tensor:
        return _softplus(self.raw_noise_variance)

    @property
    def variational_chol(self) -> Tensor:
        """Lower-triangular L_S with softplus-positive diagonal."""
        l = self.raw_variational_chol.tril(-1)
        return l + torch.diag_embed(_softplus(self.raw_variational_chol.diagonal()))

    def parameters(self, variational: bool = True) -> list[Tensor]:
        params = self.kernel.parameters() + [self.raw_noise_variance]
        if self.inducing_points.requires_grad:
            params.append(self.inducing_points)
        if variational:
            params += [self.variational_mean, self.raw_variational_chol]
        return params

    def _validate_inputs(self, x: Tensor, y: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        x = torch.as_tensor(x, dtype=self.dtype, device=self.device)
        if x.ndim == 1:
            x = x.unsqueeze(-1)
        if x.ndim != 2:
            raise ValueError(f"X must have shape (n, d), got {tuple(x.shape)}")
        if y is not None:
            y = torch.as_tensor(y, dtype=self.dtype, device=self.device).reshape(-1)
            if y.shape[0] != x.shape[0]:
                raise ValueError(f"X has {x.shape[0]} rows but y has {y.shape[0]} entries")
        return x, y

    def _kmm_chol(self) -> Tensor:
        """Jittered Cholesky of K_mm (escalating jitter, as in Step 1)."""
        z = self.inducing_points
        kmm = self.kernel(z)
        eye = torch.eye(z.shape[0], dtype=self.dtype, device=self.device)
        jitter = self.jitter
        for _ in range(8):
            chol, info = torch.linalg.cholesky_ex(kmm + jitter * eye)
            if int(info) == 0:
                return chol
            jitter *= 10.0
        raise torch.linalg.LinAlgError(
            f"K_mm Cholesky failed at jitter={jitter:.1e}; inducing points have "
            "likely collapsed onto each other."
        )

    # ------------------------------------------------------------------ #
    # Titsias (2009) collapsed VFE bound                                  #
    # ------------------------------------------------------------------ #

    def elbo_collapsed(self, x: Tensor, y: Tensor) -> Tensor:
        """Collapsed variational free energy (exact optimal q(u) plugged in).

            L = log N(y | 0, Q_nn + s^2 I) - 1/(2 s^2) tr(K_nn - Q_nn)

        Evaluated in O(n m^2) via the Woodbury/matrix-determinant lemmas on
        B = I + A A^T / s^2 with A = L_mm^{-1} K_mu (so Q_nn = A^T A):

            log|Q_nn + s^2 I| = log|B| + n log s^2
            (Q_nn + s^2 I)^{-1} = (I - A^T B^{-1} A / s^2) / s^2

        Requires the full (x, y); for mini-batch SVI use elbo_minibatch().
        """
        x, y = self._validate_inputs(x, y)
        assert y is not None
        n = x.shape[0]
        s2 = self.noise_variance
        resid = y - y.mean()

        l_mm = self._kmm_chol()
        kmu = self.kernel(self.inducing_points, x)                   # (m, n)
        a = torch.linalg.solve_triangular(l_mm, kmu, upper=False)     # (m, n)

        m = a.shape[0]
        b = torch.eye(m, dtype=self.dtype, device=self.device) + (a @ a.T) / s2
        l_b = torch.linalg.cholesky(b)

        # log|Q + s^2 I| = 2 sum log diag(L_B) + n log s^2
        logdet = 2.0 * l_b.diagonal().log().sum() + n * s2.log()
        # quadratic form via c = L_B^{-1} A resid / s^2
        c = torch.linalg.solve_triangular(l_b, a @ resid.unsqueeze(-1), upper=False) / s2
        quad = (resid @ resid) / s2 - (c**2).sum()
        # trace term: tr(K_nn - Q_nn) = sum k(x_i,x_i) - sum A^2
        trace = self.kernel.diag(x).sum() - (a**2).sum()

        return -0.5 * (logdet + quad + n * _LOG_2PI) - trace / (2.0 * s2)

    # ------------------------------------------------------------------ #
    # Hensman et al. (2013) uncollapsed bound — SVI                       #
    # ------------------------------------------------------------------ #

    def _kl_qu_pu(self, l_mm: Tensor) -> Tensor:
        """KL(q(u) || p(u)) between N(m, S) and N(0, K_mm), Cholesky form:

            1/2 [ tr(K^{-1} S) + m^T K^{-1} m - m + log|K| - log|S| ]
        """
        l_s = self.variational_chol
        m = l_s.shape[0]
        # tr(K^{-1} S) = || L_mm^{-1} L_S ||_F^2
        inv_ls = torch.linalg.solve_triangular(l_mm, l_s, upper=False)
        trace = (inv_ls**2).sum()
        mu = torch.linalg.solve_triangular(l_mm, self.variational_mean.unsqueeze(-1), upper=False)
        quad = (mu**2).sum()
        logdet_k = 2.0 * l_mm.diagonal().log().sum()
        logdet_s = 2.0 * l_s.diagonal().log().sum()
        return 0.5 * (trace + quad - m + logdet_k - logdet_s)

    def _predictive_qf(self, x: Tensor, l_mm: Tensor) -> tuple[Tensor, Tensor]:
        """Marginal q(f(x)) = N(mu, var) under q(u); returns ((b,), (b,)).

        With A = K_mm^{-1} K_mu (computed as two triangular solves):
            mu  = A^T m
            var = k_xx - a^T K_mm^{-1} a + a^T K_mm^{-1} S K_mm^{-1} a
        """
        kmu = self.kernel(self.inducing_points, x)                     # (m, b)
        v = torch.linalg.solve_triangular(l_mm, kmu, upper=False)      # L^{-1} K_mu
        a = torch.linalg.solve_triangular(l_mm.T, v, upper=True)       # K_mm^{-1} K_mu

        mean = a.T @ self.variational_mean
        proj_s = self.variational_chol.T @ a                            # L_S^T K^{-1} K_mu
        var = self.kernel.diag(x) - (v**2).sum(0) + (proj_s**2).sum(0)
        return mean, var.clamp_min(0.0)

    def elbo_minibatch(self, x_batch: Tensor, y_batch: Tensor, n_total: int) -> Tensor:
        """Unbiased SVI estimate of the Hensman bound from a mini-batch:

            N/B * sum_{i in batch} E_{q(f_i)}[log N(y_i | f_i, s^2)] - KL(q||p)

        The Gaussian expected log-likelihood is closed-form:

            E[log N(y|f, s^2)] = -1/2 log(2 pi s^2)
                                 - ((y - mu_i)^2 + var_i) / (2 s^2).

        Cost per step: O(B m^2 + m^3), independent of N — this is what lets
        the model train on arbitrarily large datasets.
        """
        x, y = self._validate_inputs(x_batch, y_batch)
        assert y is not None
        b = x.shape[0]
        s2 = self.noise_variance

        l_mm = self._kmm_chol()
        mean, var = self._predictive_qf(x, l_mm)
        resid = y - self._y_mean  # centered by the stored constant mean

        ell = -0.5 * (_LOG_2PI + s2.log()) - ((resid - mean) ** 2 + var) / (2.0 * s2)
        return (n_total / b) * ell.sum() - self._kl_qu_pu(l_mm)

    # ------------------------------------------------------------------ #
    # Training loops                                                      #
    # ------------------------------------------------------------------ #

    def fit_collapsed(
        self, x: Tensor, y: Tensor, n_iters: int = 200, lr: float = 0.05, verbose: bool = False
    ) -> list[float]:
        """Maximize the Titsias collapsed bound (full-batch) over kernel
        hyperparameters, noise, and inducing locations. q(u) is implicit
        (analytically optimal), so after training we recover the explicit
        optimal q(u) once for prediction."""
        x, y = self._validate_inputs(x, y)
        assert y is not None
        self._y_train = y
        self._y_mean = y.mean()
        opt = torch.optim.Adam(self.parameters(variational=False), lr=lr)
        history: list[float] = []
        for it in range(n_iters):
            opt.zero_grad()
            loss = -self.elbo_collapsed(x, y)
            loss.backward()
            opt.step()
            history.append(-float(loss.detach()))
            if verbose and (it % max(1, n_iters // 10) == 0 or it == n_iters - 1):
                print(f"iter {it:4d} | ELBO {history[-1]:12.4f}")
        self._set_optimal_qu(x, y)
        return history

    def fit_svi(
        self,
        x: Tensor,
        y: Tensor,
        n_epochs: int = 30,
        batch_size: int = 512,
        lr: float = 0.05,
        generator: torch.Generator | None = None,
        verbose: bool = False,
    ) -> list[float]:
        """Stochastic Variational Inference: Adam on the mini-batch ELBO over
        ALL parameters (kernel, noise, Z, variational m and L_S).

        Returns per-epoch average ELBO estimates (rescaled to full-data
        units). Memory per step is O(B m + m^2) — N never enters a kernel.
        """
        x, y = self._validate_inputs(x, y)
        assert y is not None
        n = x.shape[0]
        self._y_train = y
        self._y_mean = y.mean().detach()
        opt = torch.optim.Adam(self.parameters(variational=True), lr=lr)
        history: list[float] = []
        for epoch in range(n_epochs):
            perm = torch.randperm(n, generator=generator)
            epoch_elbo, n_batches = 0.0, 0
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                opt.zero_grad()
                loss = -self.elbo_minibatch(x[idx], y[idx], n_total=n)
                loss.backward()
                opt.step()
                epoch_elbo += -float(loss.detach())
                n_batches += 1
            history.append(epoch_elbo / n_batches)
            if verbose and (epoch % max(1, n_epochs // 10) == 0 or epoch == n_epochs - 1):
                print(f"epoch {epoch:3d} | avg ELBO {history[-1]:14.2f}")
        return history

    @torch.no_grad()
    def _set_optimal_qu(self, x: Tensor, y: Tensor) -> None:
        """Set q(u) to the Titsias-optimal Gaussian for the current
        hyperparameters (used after collapsed training so the SVGP-form
        predictive equations apply):

            Sigma = K_mm (K_mm + K_mu K_um / s^2)^{-1} K_mm
            m_u   = Sigma K_mm^{-1} K_mu y / s^2   (on centered y)
        """
        s2 = self.noise_variance
        l_mm = self._kmm_chol()
        kmu = self.kernel(self.inducing_points, x)
        a = torch.linalg.solve_triangular(l_mm, kmu, upper=False)      # (m, n)
        m = a.shape[0]
        b = torch.eye(m, dtype=self.dtype, device=self.device) + (a @ a.T) / s2
        l_b = torch.linalg.cholesky(b)
        # S = L_mm B^{-1} L_mm^T. Note L_mm L_B^{-T} is NOT triangular (lower
        # times upper), so assemble S and take a fresh Cholesky for storage.
        w = torch.linalg.solve_triangular(l_b, l_mm.T, upper=False)   # L_B^{-1} L_mm^T
        s_opt = w.T @ w                                                # = L_mm B^{-1} L_mm^T
        eye = torch.eye(m, dtype=self.dtype, device=self.device)
        chol_s = torch.linalg.cholesky(s_opt + self.jitter * eye)
        resid = (y - y.mean()).unsqueeze(-1)
        # m_u = L_mm B^{-1} A resid / s^2
        tmp = torch.cholesky_solve(a @ resid, l_b) / s2
        m_u = (l_mm @ tmp).reshape(-1)

        self.variational_mean.copy_(m_u)
        raw = chol_s.tril(-1).clone()
        raw.diagonal().copy_(torch.tensor(
            [_inv_softplus(float(v)) for v in chol_s.diagonal().clamp_min(1e-12)],
            dtype=self.dtype, device=self.device,
        ))
        self.raw_variational_chol.copy_(raw)

    # ------------------------------------------------------------------ #
    # Prediction — duck-types GaussianProcessRegressor                    #
    # ------------------------------------------------------------------ #

    def _require_fitted(self) -> None:
        if self._y_train is None:
            raise RuntimeError("Call fit_svi()/fit_collapsed() before predicting.")

    def posterior_diag(self, x_test: Tensor) -> tuple[Tensor, Tensor]:
        """Predictive mean and diagonal variance, differentiable w.r.t.
        x_test — same contract as GaussianProcessRegressor.posterior_diag,
        so Step 2 acquisition functions compose unchanged."""
        self._require_fitted()
        x, _ = self._validate_inputs(x_test)
        mean, var = self._predictive_qf(x, self._kmm_chol())
        return self._y_mean + mean, var

    @torch.no_grad()
    def predict(self, x_test: Tensor, return_full_cov: bool = True) -> GPPosterior:
        """Full predictive posterior q(f*) = N(mu, Sigma) at test inputs.

            Sigma = K_** - A^T (K_mm - S) A,   A = K_mm^{-1} K_m*
        """
        self._require_fitted()
        x, _ = self._validate_inputs(x_test)
        l_mm = self._kmm_chol()
        kmu = self.kernel(self.inducing_points, x)
        v = torch.linalg.solve_triangular(l_mm, kmu, upper=False)
        a = torch.linalg.solve_triangular(l_mm.T, v, upper=True)

        mean = self._y_mean + a.T @ self.variational_mean
        proj_s = self.variational_chol.T @ a
        var = (self.kernel.diag(x) - (v**2).sum(0) + (proj_s**2).sum(0)).clamp_min(0.0)

        if return_full_cov:
            cov = self.kernel(x) - v.T @ v + proj_s.T @ proj_s
            cov = 0.5 * (cov + cov.T)
        else:
            cov = torch.diag_embed(var)
        return GPPosterior(mean=mean, covariance=cov, variance=var)

    @torch.no_grad()
    def sample_posterior(
        self, x_test: Tensor, n_samples: int = 1, generator: torch.Generator | None = None
    ) -> Tensor:
        """Joint samples f* ~ q(f*), shape (n_samples, m*) — Thompson-ready."""
        post = self.predict(x_test, return_full_cov=True)
        eye = torch.eye(post.mean.shape[0], dtype=self.dtype, device=self.device)
        cov = post.covariance + self.jitter * eye
        chol = torch.linalg.cholesky(cov)
        z = torch.randn(
            n_samples, post.mean.shape[0], dtype=self.dtype, device=self.device, generator=generator
        )
        return post.mean.unsqueeze(0) + z @ chol.T
