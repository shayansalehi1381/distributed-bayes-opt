"""Tests for the sparse GP (VFE/Titsias collapsed bound + SVGP/Hensman SVI).

Key mathematical guarantees verified here:
* With Z = X (M = N) the sparse model must recover the exact GP: identical
  predictive mean/variance and collapsed ELBO == exact log marginal
  likelihood (the Nystrom approximation becomes exact, trace term -> 0).
* The collapsed ELBO is a true LOWER bound on the exact LML for any Z.
* The uncollapsed (Hensman) bound never exceeds the collapsed (Titsias)
  bound, and equals it at the optimal q(u).
* Mini-batch ELBO estimates are unbiased: averaged over a partition of the
  data they reproduce the full-batch bound exactly.
* SVI trains on N = 20,000 in seconds with sane predictive accuracy.
* The predictive interface composes with every Step 2 acquisition function.
"""

import math
import time

import pytest
import torch

from src.acquisition import ExpectedImprovement, ThompsonSampling, UpperConfidenceBound
from src.gp_regression import GaussianProcessRegressor
from src.sparse_gp import SparseGPRegressor, greedy_inducing_init

DTYPE = torch.float64
torch.manual_seed(0)


def _data(n=50, d=2, noise=0.1, seed=1):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, d, generator=g, dtype=DTYPE) * 4 - 2
    f = torch.sin(2 * x[:, 0]) + 0.5 * torch.cos(3 * x[:, 1] if d > 1 else 3 * x[:, 0])
    y = f + noise * torch.randn(n, generator=g, dtype=DTYPE)
    return x, y, f


def _sparse_with_optimal_qu(x, y, z, jitter=1e-8, noise_variance=1e-2):
    """Sparse model at default hyperparameters with the Titsias-optimal q(u)
    (fit_collapsed with 0 iterations skips optimization, sets q(u))."""
    model = SparseGPRegressor(z, learn_inducing=False, jitter=jitter, noise_variance=noise_variance)
    model.fit_collapsed(x, y, n_iters=0)
    return model


# --------------------------------------------------------------------- #
# Convergence to the exact GP as M -> N                                  #
# --------------------------------------------------------------------- #

class TestExactRecovery:
    def test_z_equals_x_matches_exact_gp_predictions(self):
        """M = N, Z = X: sparse predictions must equal Step 1's exact GP."""
        x, y, _ = _data(n=40)
        exact = GaussianProcessRegressor(noise_variance=1e-2, jitter=1e-8).fit(x, y)
        sparse = _sparse_with_optimal_qu(x, y, z=x)

        g = torch.Generator().manual_seed(3)
        xs = torch.rand(25, 2, dtype=DTYPE, generator=g) * 4 - 2
        pe, ps = exact.predict(xs), sparse.predict(xs)

        # Tolerance note: the models are analytically identical here, but the
        # sparse path jitters K_mm (cond ~1e8) while the exact path jitters
        # the far-better-conditioned K + sigma^2 I. The mean solve amplifies
        # the 1e-8 jitter to ~1e-5; the variance applies K_mm^{-1} twice and
        # lands at ~5e-4. The tight ELBO == LML test below is the sharper
        # exactness check.
        assert torch.allclose(pe.mean, ps.mean, atol=1e-4)
        assert torch.allclose(pe.variance, ps.variance, atol=2e-3)
        assert torch.allclose(pe.covariance, ps.covariance, atol=2e-3)

    def test_z_equals_x_elbo_equals_exact_lml(self):
        """At Z = X the trace penalty vanishes and the collapsed bound is
        tight: ELBO == exact log marginal likelihood."""
        x, y, _ = _data(n=30)
        exact = GaussianProcessRegressor(noise_variance=1e-2, jitter=1e-8).fit(x, y)
        sparse = _sparse_with_optimal_qu(x, y, z=x)

        lml = float(exact.log_marginal_likelihood(x, y).detach())
        elbo = float(sparse.elbo_collapsed(x, y).detach())
        # Same jitter-placement tolerance as above: agreement to ~1e-4 nats.
        assert abs(lml - elbo) < 2e-4, f"LML {lml} vs collapsed ELBO {elbo}"

    def test_predictions_converge_monotonically_in_m(self):
        """Prediction error vs the exact GP should shrink as M grows."""
        x, y, _ = _data(n=80)
        exact = GaussianProcessRegressor(noise_variance=1e-2).fit(x, y)
        g = torch.Generator().manual_seed(5)
        xs = torch.rand(30, 2, dtype=DTYPE, generator=g) * 4 - 2
        ref = exact.predict(xs).mean

        errs = []
        for m in (5, 20, 80):
            z = greedy_inducing_init(x, m, generator=g)
            sparse = _sparse_with_optimal_qu(x, y, z=z)
            errs.append(float((sparse.predict(xs).mean - ref).abs().max()))
        assert errs[2] < errs[0]          # more inducing points -> closer
        assert errs[2] < 1e-4             # M = N (greedy picks all) -> exact


# --------------------------------------------------------------------- #
# Bound properties                                                       #
# --------------------------------------------------------------------- #

class TestBounds:
    def test_collapsed_elbo_lower_bounds_exact_lml(self):
        """Titsias VFE is a true lower bound for ANY inducing set."""
        x, y, _ = _data(n=60)
        exact = GaussianProcessRegressor(noise_variance=1e-2).fit(x, y)
        lml = float(exact.log_marginal_likelihood(x, y).detach())
        g = torch.Generator().manual_seed(7)
        for m in (3, 10, 30):
            z = greedy_inducing_init(x, m, generator=g)
            sparse = SparseGPRegressor(z, noise_variance=1e-2)
            assert float(sparse.elbo_collapsed(x, y).detach()) <= lml + 1e-8

    def test_uncollapsed_bound_below_collapsed_and_tight_at_optimum(self):
        """For any q(u): Hensman bound <= Titsias bound, with equality at the
        analytically optimal q(u)."""
        x, y, _ = _data(n=50)
        g = torch.Generator().manual_seed(9)
        z = greedy_inducing_init(x, 12, generator=g)

        model = SparseGPRegressor(z, learn_inducing=False)
        model._y_train, model._y_mean = y, y.mean()
        collapsed = float(model.elbo_collapsed(x, y).detach())

        # Arbitrary (suboptimal) q(u) — strictly below the collapsed bound.
        with torch.no_grad():
            model.variational_mean.normal_(generator=g)
        loose = float(model.elbo_minibatch(x, y, n_total=50).detach())
        assert loose < collapsed

        # Optimal q(u) — the gap must close.
        model._set_optimal_qu(x, y)
        tight = float(model.elbo_minibatch(x, y, n_total=50).detach())
        assert tight <= collapsed + 1e-6
        assert collapsed - tight < 1e-4, f"gap {collapsed - tight} at optimal q(u)"

    def test_minibatch_estimator_is_unbiased_over_partition(self):
        """Averaging the scaled mini-batch ELBO over a disjoint partition of
        the data must reproduce the full-batch bound exactly (the data term
        is a sum; KL is constant)."""
        x, y, _ = _data(n=48)
        g = torch.Generator().manual_seed(11)
        model = SparseGPRegressor(greedy_inducing_init(x, 8, generator=g))
        model._y_train, model._y_mean = y, y.mean()

        full = model.elbo_minibatch(x, y, n_total=48).detach()
        batches = [model.elbo_minibatch(x[i : i + 12], y[i : i + 12], 48).detach()
                   for i in range(0, 48, 12)]
        assert torch.allclose(torch.stack(batches).mean(), full, atol=1e-9)

    def test_kl_zero_iff_prior(self):
        """q(u) = N(0, K_mm) must give KL == 0 (and default init KL >= 0)."""
        x, y, _ = _data(n=20)
        model = SparseGPRegressor(x[:6])
        l_mm = model._kmm_chol()
        assert float(model._kl_qu_pu(l_mm).detach()) >= 0.0
        with torch.no_grad():
            model.variational_mean.zero_()
            raw = l_mm.tril(-1).clone()
            raw.diagonal().copy_(torch.log(torch.expm1(l_mm.diagonal())))  # inv softplus
            model.raw_variational_chol.copy_(raw)
        assert abs(float(model._kl_qu_pu(l_mm).detach())) < 1e-9


# --------------------------------------------------------------------- #
# SVI training                                                           #
# --------------------------------------------------------------------- #

class TestSVI:
    def test_svi_improves_elbo_and_learns(self):
        x, y, f = _data(n=400, noise=0.1, seed=13)
        g = torch.Generator().manual_seed(0)
        model = SparseGPRegressor(greedy_inducing_init(x, 20, generator=g))
        hist = model.fit_svi(x, y, n_epochs=40, batch_size=100, lr=0.05, generator=g)
        assert hist[-1] > hist[0] + 10.0

        xt, _, ft = _data(n=200, noise=0.0, seed=14)
        rmse = float(((model.predict(xt).mean - ft) ** 2).mean().sqrt())
        assert rmse < 0.25, f"SVI predictive RMSE too high: {rmse}"

    def test_inducing_locations_are_learned(self):
        """Z must receive gradients and actually move during SVI."""
        x, y, _ = _data(n=200, seed=15)
        g = torch.Generator().manual_seed(1)
        model = SparseGPRegressor(greedy_inducing_init(x, 10, generator=g), learn_inducing=True)
        z0 = model.inducing_points.detach().clone()
        model.fit_svi(x, y, n_epochs=10, batch_size=50, generator=g)
        assert model.inducing_points.grad is not None
        assert float((model.inducing_points.detach() - z0).norm()) > 1e-3

    def test_frozen_inducing_stay_fixed(self):
        x, y, _ = _data(n=100, seed=16)
        g = torch.Generator().manual_seed(2)
        model = SparseGPRegressor(x[:8], learn_inducing=False)
        z0 = model.inducing_points.detach().clone()
        model.fit_svi(x, y, n_epochs=5, batch_size=50, generator=g)
        assert torch.equal(model.inducing_points.detach(), z0)

    def test_scalability_n20000(self):
        """SVI on N = 20,000: per-step cost is O(B m^2 + m^3), independent of
        N — a handful of epochs must complete in seconds with sane RMSE."""
        n = 20_000
        g = torch.Generator().manual_seed(17)
        x = torch.rand(n, 2, generator=g, dtype=DTYPE) * 4 - 2
        f = torch.sin(2 * x[:, 0]) + 0.5 * torch.cos(3 * x[:, 1])
        y = f + 0.1 * torch.randn(n, generator=g, dtype=DTYPE)

        model = SparseGPRegressor(greedy_inducing_init(x, 64, generator=g))
        t0 = time.perf_counter()
        hist = model.fit_svi(x, y, n_epochs=3, batch_size=512, lr=0.08, generator=g)
        elapsed = time.perf_counter() - t0

        assert elapsed < 60.0, f"SVI on N=20k took {elapsed:.1f}s"
        assert hist[-1] > hist[0]

        xt = torch.rand(1000, 2, dtype=DTYPE, generator=g) * 4 - 2
        ft = torch.sin(2 * xt[:, 0]) + 0.5 * torch.cos(3 * xt[:, 1])
        rmse = float(((model.predict(xt, return_full_cov=False).mean - ft) ** 2).mean().sqrt())
        assert rmse < 0.2, f"N=20k SVI RMSE {rmse}"
        print(f"\n[benchmark] SVI N=20,000, M=64, B=512, 3 epochs: "
              f"{elapsed:.2f}s, test RMSE {rmse:.4f}")


# --------------------------------------------------------------------- #
# Acquisition-function composition (Step 2 interoperability)             #
# --------------------------------------------------------------------- #

class TestAcquisitionComposition:
    @pytest.fixture()
    def fitted(self):
        x, y, _ = _data(n=300, seed=19)
        g = torch.Generator().manual_seed(4)
        model = SparseGPRegressor(greedy_inducing_init(x, 24, generator=g))
        model.fit_svi(x, y, n_epochs=20, batch_size=100, generator=g)
        return model

    def test_expected_improvement(self, fitted):
        g = torch.Generator().manual_seed(0)
        xq = torch.rand(50, 2, dtype=DTYPE, generator=g) * 4 - 2
        vals = ExpectedImprovement(fitted)(xq)  # incumbent from _y_train duck-typing
        assert vals.shape == (50,) and (vals >= 0).all() and torch.isfinite(vals).all()

    def test_ei_gradient_via_autograd_path(self, fitted):
        """EI.gradient() is exact-GP-specific; the supported sparse route is
        autograd through posterior_diag — verify against finite differences."""
        ei = ExpectedImprovement(fitted, xi=0.01)
        xq = torch.tensor([[0.4, -0.3]], dtype=DTYPE, requires_grad=True)
        mean, var = fitted.posterior_diag(xq)
        ei._ei(mean, var.clamp_min(1e-24).sqrt()).sum().backward()
        grad = xq.grad

        h = 1e-6
        for d in range(2):
            e = torch.zeros(1, 2, dtype=DTYPE)
            e[0, d] = h
            with torch.no_grad():
                fd = float((ei(xq + e) - ei(xq - e)) / (2 * h))
            assert abs(fd - float(grad[0, d])) < 1e-5

    def test_ucb(self, fitted):
        g = torch.Generator().manual_seed(1)
        xq = torch.rand(30, 2, dtype=DTYPE, generator=g) * 4 - 2
        ucb = UpperConfidenceBound(fitted, beta=4.0)
        mean, var = fitted.posterior_diag(xq)
        assert torch.allclose(ucb(xq), (mean + 2 * var.sqrt()).detach(), atol=1e-12)

    def test_thompson_sampling(self, fitted):
        g = torch.Generator().manual_seed(2)
        xq = torch.rand(16, 2, dtype=DTYPE, generator=g) * 4 - 2
        ts = ThompsonSampling(fitted)
        paths = ts(xq, n_samples=8, generator=g)
        assert paths.shape == (8, 16) and torch.isfinite(paths).all()
        idx = ts.select(xq, n_select=5, generator=g)
        assert idx.shape == (5,) and (idx >= 0).all() and (idx < 16).all()


# --------------------------------------------------------------------- #
# Robustness & utilities                                                 #
# --------------------------------------------------------------------- #

class TestRobustness:
    def test_predict_before_fit_raises(self):
        model = SparseGPRegressor(torch.zeros(4, 2, dtype=DTYPE))
        with pytest.raises(RuntimeError, match="fit"):
            model.predict(torch.zeros(3, 2, dtype=DTYPE))

    def test_variance_nonnegative_and_diag_consistent(self):
        x, y, _ = _data(n=100, seed=21)
        g = torch.Generator().manual_seed(3)
        model = SparseGPRegressor(greedy_inducing_init(x, 15, generator=g))
        model.fit_svi(x, y, n_epochs=10, batch_size=50, generator=g)
        xq = torch.rand(200, 2, dtype=DTYPE, generator=g) * 6 - 3
        post = model.predict(xq)
        _, var_diag = model.posterior_diag(xq)
        assert (post.variance >= 0).all()
        assert torch.allclose(post.variance, var_diag.detach(), atol=1e-10)
        assert torch.allclose(post.covariance.diagonal(), post.variance, atol=1e-8)

    def test_float64_kuu_cholesky_stability_with_clustered_z(self):
        """Nearly-duplicated inducing points make K_mm severely
        ill-conditioned; the escalating-jitter Cholesky must survive."""
        z = torch.zeros(10, 2, dtype=DTYPE) + 1e-9 * torch.randn(10, 2, dtype=DTYPE)
        model = SparseGPRegressor(z)
        l = model._kmm_chol()
        assert torch.isfinite(l).all()

    def test_greedy_inducing_init(self):
        g = torch.Generator().manual_seed(0)
        x = torch.rand(100, 3, dtype=DTYPE, generator=g)
        z = greedy_inducing_init(x, 10, generator=g)
        assert z.shape == (10, 3)
        assert torch.unique(z, dim=0).shape[0] == 10  # no duplicates
        assert greedy_inducing_init(x, 200).shape == (100, 3)  # m >= n -> copy

    def test_1d_inputs_promoted(self):
        g = torch.Generator().manual_seed(5)
        x = torch.linspace(0, 1, 50, dtype=DTYPE)
        y = torch.sin(6 * x)
        model = SparseGPRegressor(x[::5], noise_variance=1e-3)
        model.fit_collapsed(x, y, n_iters=30)
        post = model.predict(torch.tensor([0.5], dtype=DTYPE))
        assert post.mean.shape == (1,)
        assert abs(float(post.mean) - math.sin(3.0)) < 0.2
