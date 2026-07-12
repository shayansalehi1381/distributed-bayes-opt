"""Tests for the from-scratch GP regression implementation.

The mathematical correctness tests verify our Cholesky-based inference
against a direct (naive) evaluation of the textbook equations, and check
the statistical properties a correct GP must satisfy (interpolation of
noise-free data, prior reversion far from data, calibrated posterior
contraction).
"""

import math

import pytest
import torch

from src.gp_regression import GaussianProcessRegressor, RBFKernel

torch.manual_seed(0)
DTYPE = torch.float64


def _make_data(n=40, d=2, noise=0.05, seed=1):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, d, generator=g, dtype=DTYPE) * 4.0 - 2.0
    f = torch.sin(x[:, 0] * 2.0) + 0.5 * torch.cos(x[:, 1] * 3.0)
    y = f + noise * torch.randn(n, generator=g, dtype=DTYPE)
    return x, y, f


# --------------------------------------------------------------------- #
# Kernel                                                                #
# --------------------------------------------------------------------- #

class TestRBFKernel:
    def test_diagonal_equals_signal_variance(self):
        k = RBFKernel(lengthscale=0.7, signal_variance=2.5)
        x = torch.randn(10, 3, dtype=DTYPE)
        K = k(x)
        assert torch.allclose(K.diagonal(), torch.full((10,), 2.5, dtype=DTYPE), atol=1e-10)

    def test_symmetry_and_psd(self):
        k = RBFKernel()
        x = torch.randn(30, 2, dtype=DTYPE)
        K = k(x)
        assert torch.allclose(K, K.T, atol=1e-12)
        eigvals = torch.linalg.eigvalsh(K)
        assert eigvals.min() > -1e-8  # PSD up to float tolerance

    def test_matches_naive_computation(self):
        """Vectorized ||a-b||^2 expansion must agree with the direct formula."""
        ls, sf2 = 0.9, 1.7
        k = RBFKernel(lengthscale=ls, signal_variance=sf2)
        x1 = torch.randn(8, 3, dtype=DTYPE)
        x2 = torch.randn(5, 3, dtype=DTYPE)
        K = k(x1, x2)
        naive = torch.empty(8, 5, dtype=DTYPE)
        for i in range(8):
            for j in range(5):
                d2 = ((x1[i] - x2[j]) ** 2).sum() / ls**2
                naive[i, j] = sf2 * math.exp(-0.5 * float(d2))
        assert torch.allclose(K, naive, atol=1e-10)

    def test_ard_lengthscales(self):
        k = RBFKernel(lengthscale=[0.5, 5.0])
        base = torch.zeros(1, 2, dtype=DTYPE)
        # Same offset along a short vs long lengthscale dim: short decays more.
        k_short = k(base, torch.tensor([[1.0, 0.0]], dtype=DTYPE)).detach()
        k_long = k(base, torch.tensor([[0.0, 1.0]], dtype=DTYPE)).detach()
        assert float(k_short) < float(k_long)


# --------------------------------------------------------------------- #
# Posterior correctness                                                 #
# --------------------------------------------------------------------- #

class TestPosteriorMath:
    def test_matches_naive_textbook_equations(self):
        """Cholesky path must agree with explicit-inverse textbook formulas."""
        x, y, _ = _make_data(n=25)
        xs = torch.rand(7, 2, dtype=DTYPE) * 4.0 - 2.0

        model = GaussianProcessRegressor(noise_variance=0.05, jitter=0.0)
        model.fit(x, y)
        post = model.predict(xs)

        # Naive computation with explicit inverse.
        K = model.kernel(x) + model.noise_variance * torch.eye(25, dtype=DTYPE)
        Ks = model.kernel(x, xs)
        Kss = model.kernel(xs)
        K_inv = torch.linalg.inv(K)
        ym = y.mean()
        mean_naive = ym + Ks.T @ K_inv @ (y - ym)
        cov_naive = Kss - Ks.T @ K_inv @ Ks

        assert torch.allclose(post.mean, mean_naive, atol=1e-8)
        assert torch.allclose(post.covariance, cov_naive, atol=1e-8)
        assert torch.allclose(post.variance, cov_naive.diagonal(), atol=1e-8)

    def test_interpolates_noise_free_data(self):
        """With sigma_n^2 -> 0 the posterior mean must pass through the data
        and the posterior variance must vanish there."""
        x, _, f = _make_data(n=20, noise=0.0)
        model = GaussianProcessRegressor(noise_variance=1e-10)
        model.fit(x, f)
        post = model.predict(x)
        assert torch.allclose(post.mean, f, atol=1e-4)
        assert post.variance.max() < 1e-4

    def test_reverts_to_prior_far_from_data(self):
        x, y, _ = _make_data(n=20)
        model = GaussianProcessRegressor(noise_variance=0.05)
        model.fit(x, y)
        far = torch.full((3, 2), 100.0, dtype=DTYPE)
        post = model.predict(far)
        # Mean -> constant mean function (y mean), variance -> sigma_f^2.
        assert torch.allclose(post.mean, y.mean().expand(3), atol=1e-6)
        assert torch.allclose(post.variance, model.kernel.signal_variance.detach().expand(3), atol=1e-6)

    def test_variance_contracts_with_more_data(self):
        """More observations can only reduce predictive uncertainty."""
        x, y, _ = _make_data(n=40)
        xs = torch.rand(10, 2, dtype=DTYPE) * 4.0 - 2.0
        m_small = GaussianProcessRegressor(noise_variance=0.05).fit(x[:10], y[:10])
        m_big = GaussianProcessRegressor(noise_variance=0.05).fit(x, y)
        assert (m_big.predict(xs).variance <= m_small.predict(xs).variance + 1e-9).all()

    def test_full_cov_consistent_with_diag_only_path(self):
        x, y, _ = _make_data(n=30)
        xs = torch.rand(50, 2, dtype=DTYPE) * 4.0 - 2.0
        model = GaussianProcessRegressor().fit(x, y)
        full = model.predict(xs, return_full_cov=True)
        diag = model.predict(xs, return_full_cov=False, chunk_size=7)  # force chunking
        assert torch.allclose(full.mean, diag.mean, atol=1e-10)
        assert torch.allclose(full.variance, diag.variance, atol=1e-10)

    def test_posterior_covariance_is_psd(self):
        x, y, _ = _make_data(n=30)
        xs = torch.rand(20, 2, dtype=DTYPE) * 4.0 - 2.0
        post = GaussianProcessRegressor().fit(x, y).predict(xs)
        eigvals = torch.linalg.eigvalsh(post.covariance)
        assert eigvals.min() > -1e-8


# --------------------------------------------------------------------- #
# Log marginal likelihood & hyperparameter learning                     #
# --------------------------------------------------------------------- #

class TestMarginalLikelihood:
    def test_matches_naive_formula(self):
        x, y, _ = _make_data(n=15)
        model = GaussianProcessRegressor(noise_variance=0.05, jitter=0.0)
        lml = model.log_marginal_likelihood(x, y)

        K = model.kernel(x) + model.noise_variance * torch.eye(15, dtype=DTYPE)
        resid = y - y.mean()
        naive = (
            -0.5 * resid @ torch.linalg.solve(K.detach(), resid)
            - 0.5 * torch.logdet(K.detach())
            - 0.5 * 15 * math.log(2 * math.pi)
        )
        assert torch.allclose(lml.detach(), naive, atol=1e-8)

    def test_gradients_flow_to_all_hyperparameters(self):
        x, y, _ = _make_data(n=15)
        model = GaussianProcessRegressor()
        (-model.log_marginal_likelihood(x, y)).backward()
        for p in model.parameters():
            assert p.grad is not None
            assert torch.isfinite(p.grad).all()

    def test_optimization_improves_likelihood_and_recovers_noise(self):
        x, y, _ = _make_data(n=60, noise=0.1, seed=3)
        model = GaussianProcessRegressor(noise_variance=0.5)  # deliberately wrong init
        history = model.optimize_hyperparameters(x, y, n_iters=150, lr=0.1)
        assert history[-1] < history[0] - 1.0  # NLL decreased materially
        learned_noise = float(model.noise_variance.detach())
        assert 0.1**2 * 0.2 < learned_noise < 0.1**2 * 5.0  # right order of magnitude


# --------------------------------------------------------------------- #
# Robustness & API                                                      #
# --------------------------------------------------------------------- #

class TestRobustness:
    def test_duplicate_inputs_do_not_crash(self):
        """Duplicated rows make K singular; jitter escalation must handle it."""
        x = torch.tensor([[0.0], [0.0], [1.0], [1.0]], dtype=DTYPE)
        y = torch.tensor([0.1, 0.1, 0.9, 0.9], dtype=DTYPE)
        model = GaussianProcessRegressor(noise_variance=1e-12)
        post = model.fit(x, y).predict(torch.tensor([[0.5]], dtype=DTYPE))
        assert torch.isfinite(post.mean).all() and torch.isfinite(post.variance).all()

    def test_1d_input_promoted(self):
        x = torch.linspace(0, 1, 10, dtype=DTYPE)  # (n,) not (n, 1)
        y = torch.sin(x)
        post = GaussianProcessRegressor().fit(x, y).predict(torch.tensor([0.5], dtype=DTYPE))
        assert post.mean.shape == (1,)

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError, match="fit"):
            GaussianProcessRegressor().predict(torch.zeros(3, 2, dtype=DTYPE))

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="rows"):
            GaussianProcessRegressor().fit(torch.zeros(5, 2), torch.zeros(4))

    def test_posterior_samples_shape_and_moments(self):
        x, y, _ = _make_data(n=25)
        xs = torch.rand(6, 2, dtype=DTYPE) * 4.0 - 2.0
        model = GaussianProcessRegressor().fit(x, y)
        g = torch.Generator().manual_seed(42)
        samples = model.sample_posterior(xs, n_samples=4000, generator=g)
        assert samples.shape == (4000, 6)
        post = model.predict(xs)
        # Sample moments should match the analytic posterior within MC error.
        assert torch.allclose(samples.mean(0), post.mean, atol=0.05)
        assert torch.allclose(samples.var(0), post.variance, atol=0.05)

    def test_numpy_inputs_accepted(self):
        import numpy as np

        x = np.random.default_rng(0).uniform(-1, 1, size=(15, 2))
        y = np.sin(x[:, 0])
        post = GaussianProcessRegressor().fit(x, y).predict(np.zeros((2, 2)))
        assert post.mean.shape == (2,)
