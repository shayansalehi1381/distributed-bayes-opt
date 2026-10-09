"""Scalable Gaussian Process Regression from scratch.

Implements exact GP regression with an RBF (squared-exponential) kernel using
only PyTorch base tensor operations — no GPyTorch / scikit-learn. The model
follows Rasmussen & Williams, "Gaussian Processes for Machine Learning"
(2006), Algorithm 2.1:

    Posterior mean:        mu_*  = K_*^T (K + sigma_n^2 I)^{-1} y
    Posterior covariance:  Sigma_* = K_** - K_*^T (K + sigma_n^2 I)^{-1} K_*
    Log marginal lik.:     log p(y|X) = -1/2 y^T alpha - sum(log diag(L)) - n/2 log(2*pi)

Numerical strategy
------------------
* All linear algebra goes through a Cholesky factorization of the noisy
  kernel matrix (K + sigma_n^2 I = L L^T). We never form an explicit inverse:
  solves use `torch.cholesky_solve`, which is both faster and far better
  conditioned than `torch.linalg.inv`.
* An adaptive jitter is added to the diagonal and escalated geometrically if
  the factorization fails, which is the standard fix for near-singular kernel
  matrices arising from duplicate or tightly clustered inputs.
* Hyperparameters (lengthscale, signal variance, noise variance) are stored
  as unconstrained raw parameters mapped through softplus, so gradient-based
  maximization of the exact log marginal likelihood is unconstrained and
  autograd-friendly.
* float64 is the default dtype: kernel matrices are notoriously
  ill-conditioned in float32.
* Prediction is chunked over test points so memory stays O(n * chunk) rather
  than O(n * m) for m test points.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

__all__ = ["RBFKernel", "GaussianProcessRegressor", "GPPosterior"]

_LOG_2PI = math.log(2.0 * math.pi)


def _softplus(x: Tensor) -> Tensor:
    """Numerically stable softplus: log(1 + exp(x))."""
    return torch.nn.functional.softplus(x)


def _inv_softplus(y: float) -> float:
    """Inverse of softplus for initializing raw parameters. y must be > 0."""
    if y <= 0:
        raise ValueError(f"softplus-constrained value must be positive, got {y}")
    # log(exp(y) - 1), stable for large y where it approaches y.
    return y + math.log(-math.expm1(-y))


class RBFKernel:
    """RBF / squared-exponential kernel with ARD support.

        k(x, x') = sigma_f^2 * exp(-1/2 * sum_d (x_d - x'_d)^2 / l_d^2)

    Parameters are held as unconstrained tensors (softplus-transformed on
    read) so they can be optimized directly with autograd.

    Args:
        lengthscale: initial lengthscale(s). Scalar for isotropic, or one per
            input dimension for ARD (automatic relevance determination).
        signal_variance: initial sigma_f^2, the prior marginal variance.
        ard_dims: if given, expands a scalar lengthscale to this many dims.
        dtype / device: tensor placement; float64 strongly recommended.
    """

    def __init__(
        self,
        lengthscale: float | list[float] = 1.0,
        signal_variance: float = 1.0,
        ard_dims: int | None = None,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = "cpu",
    ) -> None:
        if isinstance(lengthscale, (int, float)):
            ls = [float(lengthscale)] * (ard_dims or 1)
        else:
            ls = [float(v) for v in lengthscale]
        self.raw_lengthscale = torch.tensor(
            [_inv_softplus(v) for v in ls], dtype=dtype, device=device, requires_grad=True
        )
        self.raw_signal_variance = torch.tensor(
            _inv_softplus(float(signal_variance)), dtype=dtype, device=device, requires_grad=True
        )

    @property
    def lengthscale(self) -> Tensor:
        return _softplus(self.raw_lengthscale)

    @property
    def signal_variance(self) -> Tensor:
        return _softplus(self.raw_signal_variance)

    def parameters(self) -> list[Tensor]:
        return [self.raw_lengthscale, self.raw_signal_variance]

    def _sq_dist(self, x1: Tensor, x2: Tensor) -> Tensor:
        """Pairwise squared Euclidean distance of lengthscale-scaled inputs.

        Uses the expansion ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b computed
        via a single matmul — O(n*m*d) with BLAS instead of materializing the
        (n, m, d) difference tensor. Clamped at 0 to kill negative values
        from floating-point cancellation.
        """
        x1 = x1 / self.lengthscale
        x2 = x2 / self.lengthscale
        x1_sq = (x1**2).sum(-1, keepdim=True)          # (n, 1)
        x2_sq = (x2**2).sum(-1, keepdim=True).T        # (1, m)
        d2 = x1_sq + x2_sq - 2.0 * (x1 @ x2.T)
        return d2.clamp_min_(0.0)

    def __call__(self, x1: Tensor, x2: Tensor | None = None) -> Tensor:
        """Evaluate the kernel matrix K(x1, x2) of shape (n, m)."""
        if x2 is None:
            x2 = x1
        return self.signal_variance * torch.exp(-0.5 * self._sq_dist(x1, x2))

    def diag(self, x: Tensor) -> Tensor:
        """k(x_i, x_i) for each row — O(n), avoids the full matrix."""
        return self.signal_variance.expand(x.shape[0]).clone()


@dataclass(frozen=True)
class GPPosterior:
    """Posterior predictive distribution at a set of test inputs."""

    mean: Tensor        # (m,)
    covariance: Tensor  # (m, m) full posterior covariance
    variance: Tensor    # (m,)  diagonal of `covariance`, clamped >= 0

    @property
    def stddev(self) -> Tensor:
        return self.variance.sqrt()


class GaussianProcessRegressor:
    """Exact GP regression with Cholesky-based inference.

    Complexity: O(n^3) fit (one Cholesky), O(n^2) per test point for the
    predictive covariance, O(n) memory per test point during chunked
    prediction.

    Args:
        kernel: covariance function; defaults to an isotropic RBF.
        noise_variance: initial observation noise sigma_n^2 (softplus-
            parameterized, learnable).
        jitter: base diagonal jitter for Cholesky stability.
        dtype / device: tensor placement; float64 strongly recommended.
    """

    def __init__(
        self,
        kernel: RBFKernel | None = None,
        noise_variance: float = 1e-2,
        jitter: float = 1e-8,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = "cpu",
    ) -> None:
        self.dtype = dtype
        self.device = torch.device(device)
        self.kernel = kernel or RBFKernel(dtype=dtype, device=device)
        self.raw_noise_variance = torch.tensor(
            _inv_softplus(float(noise_variance)), dtype=dtype, device=self.device, requires_grad=True
        )
        self.jitter = float(jitter)

        # Cached training state (set by fit()).
        self._x_train: Tensor | None = None
        self._y_train: Tensor | None = None
        self._y_mean: Tensor | None = None
        self._chol: Tensor | None = None   # L with K + sigma_n^2 I = L L^T
        self._alpha: Tensor | None = None  # (K + sigma_n^2 I)^{-1} (y - y_mean)

    @property
    def noise_variance(self) -> Tensor:
        return _softplus(self.raw_noise_variance)

    def parameters(self) -> list[Tensor]:
        return self.kernel.parameters() + [self.raw_noise_variance]

    # ------------------------------------------------------------------ #
    # Core linear algebra                                                #
    # ------------------------------------------------------------------ #

    def _robust_cholesky(self, mat: Tensor) -> Tensor:
        """Cholesky with geometrically escalating diagonal jitter.

        Kernel matrices are PSD in exact arithmetic but often indefinite in
        floating point. Retrying with jitter * 10^k is the standard remedy.
        """
        eye = torch.eye(mat.shape[0], dtype=mat.dtype, device=mat.device)
        jitter = self.jitter
        for _ in range(8):
            chol, info = torch.linalg.cholesky_ex(mat + jitter * eye)
            if int(info) == 0:
                return chol
            jitter *= 10.0
        raise torch.linalg.LinAlgError(
            f"Cholesky failed even with jitter={jitter:.1e}; kernel matrix is "
            "numerically singular (duplicate inputs or degenerate lengthscale?)."
        )

    def _validate_inputs(self, x: Tensor, y: Tensor | None = None) -> tuple[Tensor, Tensor | None]:
        x = torch.as_tensor(x, dtype=self.dtype, device=self.device)
        if x.ndim == 1:
            x = x.unsqueeze(-1)
        if x.ndim != 2:
            raise ValueError(f"X must have shape (n, d), got {tuple(x.shape)}")
        if y is not None:
            y = torch.as_tensor(y, dtype=self.dtype, device=self.device).reshape(-1)
            if y.shape[0] != x.shape[0]:
                raise ValueError(
                    f"X has {x.shape[0]} rows but y has {y.shape[0]} entries"
                )
        return x, y

    # ------------------------------------------------------------------ #
    # Fitting                                                            #
    # ------------------------------------------------------------------ #

    def fit(self, x_train: Tensor, y_train: Tensor) -> "GaussianProcessRegressor":
        """Condition the GP on observations: factorize K + sigma_n^2 I once.

        The targets are centered by their empirical mean (a constant mean
        function), which materially improves conditioning when y is offset
        far from zero.
        """
        x, y = self._validate_inputs(x_train, y_train)
        assert y is not None
        self._x_train, self._y_train = x, y
        self._y_mean = y.mean()
        with torch.no_grad():
            k = self.kernel(x)
            noisy_k = k + self.noise_variance * torch.eye(
                x.shape[0], dtype=self.dtype, device=self.device
            )
            self._chol = self._robust_cholesky(noisy_k)
            resid = (y - self._y_mean).unsqueeze(-1)
            self._alpha = torch.cholesky_solve(resid, self._chol)
        return self

    def _require_fitted(self) -> None:
        if self._chol is None:
            raise RuntimeError("Call fit(x_train, y_train) before predicting.")

    # ------------------------------------------------------------------ #
    # Prediction                                                         #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def predict(
        self,
        x_test: Tensor,
        return_full_cov: bool = True,
        chunk_size: int = 2048,
    ) -> GPPosterior:
        """Posterior predictive mean and covariance at test inputs.

        Args:
            x_test: (m, d) test locations.
            return_full_cov: if False, only the (m,) diagonal variance is
                computed — O(m n^2) time but O(chunk * n) memory, which is
                what scalable downstream acquisition functions need.
            chunk_size: number of test points processed per block.

        Returns:
            GPPosterior with `mean` (m,), `variance` (m,), and `covariance`
            ((m, m) if return_full_cov else diagonal embedding).
        """
        self._require_fitted()
        x, _ = self._validate_inputs(x_test)
        assert self._x_train is not None and self._alpha is not None
        assert self._chol is not None and self._y_mean is not None

        means, variances, v_blocks = [], [], []
        for start in range(0, x.shape[0], chunk_size):
            xb = x[start : start + chunk_size]
            k_star = self.kernel(self._x_train, xb)              # (n, b)
            means.append(self._y_mean + k_star.T @ self._alpha)  # (b, 1)
            # v = L^{-1} K_*  =>  K_*^T (K + s^2 I)^{-1} K_* = v^T v
            v = torch.linalg.solve_triangular(self._chol, k_star, upper=False)
            variances.append(self.kernel.diag(xb) - (v**2).sum(0))
            if return_full_cov:
                v_blocks.append(v)

        mean = torch.cat(means).reshape(-1)
        variance = torch.cat(variances).clamp_min(0.0)

        if return_full_cov:
            v_all = torch.cat(v_blocks, dim=-1)                  # (n, m)
            covariance = self.kernel(x) - v_all.T @ v_all
            # Symmetrize to remove floating-point asymmetry.
            covariance = 0.5 * (covariance + covariance.T)
        else:
            covariance = torch.diag_embed(variance)

        return GPPosterior(mean=mean, covariance=covariance, variance=variance)

    def posterior_diag(self, x_test: Tensor) -> tuple[Tensor, Tensor]:
        """Posterior mean and *diagonal* variance, differentiable w.r.t. x_test.

        Unlike predict() this is not wrapped in no_grad, so downstream
        acquisition functions can differentiate through mu(x) and sigma^2(x)
        via autograd (e.g. to cross-check analytical gradients). Uses the
        cached Cholesky factor from fit().
        """
        self._require_fitted()
        x, _ = self._validate_inputs(x_test)
        assert self._x_train is not None and self._alpha is not None
        assert self._chol is not None and self._y_mean is not None
        k_star = self.kernel(self._x_train, x)                    # (n, m)
        mean = self._y_mean + (k_star.T @ self._alpha).reshape(-1)
        v = torch.linalg.solve_triangular(self._chol, k_star, upper=False)
        var = (self.kernel.diag(x) - (v**2).sum(0)).clamp_min(0.0)
        return mean, var

    @torch.no_grad()
    def sample_posterior(
        self, x_test: Tensor, n_samples: int = 1, generator: torch.Generator | None = None
    ) -> Tensor:
        """Draw joint samples f_* ~ N(mu_*, Sigma_*). Returns (n_samples, m)."""
        post = self.predict(x_test, return_full_cov=True)
        chol = self._robust_cholesky(post.covariance)
        z = torch.randn(
            n_samples, post.mean.shape[0], dtype=self.dtype, device=self.device, generator=generator
        )
        return post.mean.unsqueeze(0) + z @ chol.T

    # ------------------------------------------------------------------ #
    # Marginal likelihood & hyperparameter optimization                  #
    # ------------------------------------------------------------------ #

    def log_marginal_likelihood(self, x: Tensor | None = None, y: Tensor | None = None) -> Tensor:
        """Exact log p(y | X, theta), differentiable w.r.t. hyperparameters.

        Rebuilds the Cholesky factor inside the autograd graph (fit() caches
        it under no_grad for speed, which is fine for prediction but not for
        gradients).
        """
        if x is None or y is None:
            self._require_fitted()
            x, y = self._x_train, self._y_train
        else:
            x, y = self._validate_inputs(x, y)
        assert x is not None and y is not None

        n = x.shape[0]
        k = self.kernel(x) + self.noise_variance * torch.eye(n, dtype=self.dtype, device=self.device)
        chol = self._robust_cholesky(k)
        resid = (y - y.mean()).unsqueeze(-1)
        alpha = torch.cholesky_solve(resid, chol)
        # -1/2 y^T alpha - sum(log L_ii) - n/2 log(2 pi)
        return (
            -0.5 * (resid * alpha).sum()
            - chol.diagonal().log().sum()
            - 0.5 * n * _LOG_2PI
        )

    def optimize_hyperparameters(
        self,
        x_train: Tensor,
        y_train: Tensor,
        n_iters: int = 200,
        lr: float = 0.1,
        verbose: bool = False,
    ) -> list[float]:
        """Type-II maximum likelihood: maximize log p(y|X) with Adam.

        Returns the trace of negative log marginal likelihood per iteration.
        Refits the cached Cholesky/alpha with the optimized hyperparameters
        before returning, so the model is immediately ready for prediction.
        """
        x, y = self._validate_inputs(x_train, y_train)
        assert y is not None
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        history: list[float] = []
        for it in range(n_iters):
            optimizer.zero_grad()
            nll = -self.log_marginal_likelihood(x, y)
            nll.backward()
            optimizer.step()
            history.append(float(nll.detach()))
            if verbose and (it % max(1, n_iters // 10) == 0 or it == n_iters - 1):
                print(
                    f"iter {it:4d} | nll {history[-1]:10.4f} | "
                    f"ls {self.kernel.lengthscale.detach().cpu().numpy().round(4)} | "
                    f"sf2 {float(self.kernel.signal_variance):.4f} | "
                    f"sn2 {float(self.noise_variance):.6f}"
                )
        self.fit(x, y)
        return history
