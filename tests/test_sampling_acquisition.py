"""Tests for the sampling (MCMC / Langevin) and acquisition layers.

Statistical convergence is tested on targets with known moments (Gaussians
with specified mean/covariance) using seeded generators, and cross-checked
with R-hat diagnostics. Acquisition math is verified against Monte-Carlo
estimates and autograd.
"""

import math

import pytest
import torch

from src.acquisition import (
    ExpectedImprovement,
    ThompsonSampling,
    UpperConfidenceBound,
    _Phi,
    _phi,
)
from src.gp_regression import GaussianProcessRegressor
from src.sampling import (
    LangevinSampler,
    MetropolisHastings,
    potential_scale_reduction,
    sample_gp_hyperposterior,
)

DTYPE = torch.float64
torch.manual_seed(0)


def _gauss_logp(mean: torch.Tensor, cov: torch.Tensor):
    """Batched log-density of N(mean, cov), up to the (irrelevant) constant."""
    prec = torch.linalg.inv(cov)

    def logp(x: torch.Tensor) -> torch.Tensor:
        d = x - mean
        return -0.5 * torch.einsum("ci,ij,cj->c", d, prec, d)

    return logp


def _fit_gp(n=30, seed=1):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n, 2, generator=g, dtype=DTYPE) * 4 - 2
    y = torch.sin(2 * x[:, 0]) + 0.5 * torch.cos(3 * x[:, 1]) + 0.05 * torch.randn(
        n, generator=g, dtype=DTYPE
    )
    return GaussianProcessRegressor(noise_variance=0.05).fit(x, y), x, y


# --------------------------------------------------------------------- #
# Metropolis–Hastings                                                    #
# --------------------------------------------------------------------- #

class TestMetropolisHastings:
    def test_recovers_gaussian_moments(self):
        """Chains on a correlated 2-D Gaussian must recover mean & covariance."""
        mean = torch.tensor([1.0, -2.0], dtype=DTYPE)
        cov = torch.tensor([[1.0, 0.6], [0.6, 0.5]], dtype=DTYPE)
        g = torch.Generator().manual_seed(7)
        init = torch.randn(8, 2, dtype=DTYPE, generator=g)  # 8 overdispersed chains

        mh = MetropolisHastings(_gauss_logp(mean, cov), step_size=0.5)
        res = mh.sample(init, n_samples=3000, burn_in=1000, generator=g)
        flat = res.flat()

        assert torch.allclose(flat.mean(0), mean, atol=0.08)
        emp_cov = torch.cov(flat.T)
        assert torch.allclose(emp_cov, cov, atol=0.12)

    def test_chains_mix_rhat_near_one(self):
        mean = torch.zeros(2, dtype=DTYPE)
        cov = torch.eye(2, dtype=DTYPE)
        g = torch.Generator().manual_seed(11)
        init = 3.0 * torch.randn(6, 2, dtype=DTYPE, generator=g)
        res = MetropolisHastings(_gauss_logp(mean, cov), step_size=0.8).sample(
            init, n_samples=2000, burn_in=800, generator=g
        )
        rhat = potential_scale_reduction(res.samples)
        assert (rhat < 1.1).all(), f"chains failed to mix: R-hat={rhat}"

    def test_adaptation_hits_target_acceptance(self):
        g = torch.Generator().manual_seed(3)
        mh = MetropolisHastings(
            _gauss_logp(torch.zeros(4, dtype=DTYPE), torch.eye(4, dtype=DTYPE)),
            step_size=50.0,  # absurd init; adaptation must rescue it
            target_acceptance=0.3,
        )
        res = mh.sample(
            torch.zeros(4, 4, dtype=DTYPE), n_samples=1500, burn_in=1500, generator=g
        )
        rate = float(res.acceptance_rate.mean())
        assert 0.15 < rate < 0.5, f"acceptance {rate} far from 0.3 target"

    def test_shapes_and_log_density_validation(self):
        logp = _gauss_logp(torch.zeros(2, dtype=DTYPE), torch.eye(2, dtype=DTYPE))
        res = MetropolisHastings(logp).sample(
            torch.zeros(3, 2, dtype=DTYPE), n_samples=50, burn_in=10, thin=2
        )
        assert res.samples.shape == (50, 3, 2)
        assert res.log_probs.shape == (50, 3)
        assert res.flat().shape == (150, 2)

        bad = lambda x: x.sum()  # scalar instead of (c,)
        with pytest.raises(ValueError, match="log_density"):
            MetropolisHastings(bad).sample(torch.zeros(3, 2, dtype=DTYPE), n_samples=1)


# --------------------------------------------------------------------- #
# Langevin dynamics (Euler–Maruyama)                                     #
# --------------------------------------------------------------------- #

class TestLangevin:
    def test_mala_recovers_gaussian_moments(self):
        mean = torch.tensor([0.5, -1.0, 2.0], dtype=DTYPE)
        cov = torch.diag(torch.tensor([1.0, 0.25, 2.0], dtype=DTYPE))
        g = torch.Generator().manual_seed(21)
        init = torch.randn(8, 3, dtype=DTYPE, generator=g)

        mala = LangevinSampler(_gauss_logp(mean, cov), step_size=0.15, metropolis_adjust=True)
        res = mala.sample(init, n_samples=3000, burn_in=800, generator=g)
        flat = res.flat()

        assert torch.allclose(flat.mean(0), mean, atol=0.1)
        assert torch.allclose(flat.var(0), cov.diagonal(), atol=0.25)
        # MALA at moderate step sizes should accept most proposals.
        assert float(res.acceptance_rate.mean()) > 0.5

    def test_ula_small_step_approximates_target(self):
        """Unadjusted Langevin is O(eps)-biased; with small eps the moments
        must still land close to the target."""
        mean = torch.tensor([1.0], dtype=DTYPE)
        cov = torch.eye(1, dtype=DTYPE)
        g = torch.Generator().manual_seed(5)
        ula = LangevinSampler(_gauss_logp(mean, cov), step_size=5e-3, metropolis_adjust=False)
        res = ula.sample(
            torch.zeros(10, 1, dtype=DTYPE), n_samples=4000, burn_in=2000, generator=g
        )
        flat = res.flat()
        assert abs(float(flat.mean()) - 1.0) < 0.1
        assert abs(float(flat.var()) - 1.0) < 0.15
        # ULA never rejects by construction.
        assert torch.allclose(res.acceptance_rate, torch.ones(10, dtype=DTYPE))

    def test_gradients_via_autograd_match_analytical(self):
        """The internal score computation must equal the closed-form
        Gaussian score -Sigma^{-1}(x - mu)."""
        mean = torch.tensor([1.0, -1.0], dtype=DTYPE)
        cov = torch.tensor([[2.0, 0.3], [0.3, 1.0]], dtype=DTYPE)
        sampler = LangevinSampler(_gauss_logp(mean, cov), grad_clip=None)
        x = torch.tensor([[0.5, 0.5], [-1.0, 2.0]], dtype=DTYPE)
        _, grad = sampler._logp_and_grad(x)
        expected = -(x - mean) @ torch.linalg.inv(cov).T
        assert torch.allclose(grad, expected, atol=1e-10)

    def test_mala_beats_ula_bias_at_large_step(self):
        """At an aggressive step size, the MH correction must reduce the
        variance bias relative to plain ULA (ULA overdisperses ~ (1-eps/2)^-1)."""
        logp = _gauss_logp(torch.zeros(1, dtype=DTYPE), torch.eye(1, dtype=DTYPE))
        g1 = torch.Generator().manual_seed(9)
        g2 = torch.Generator().manual_seed(9)
        eps = 0.6
        kw = dict(n_samples=6000, burn_in=1000)
        v_ula = float(
            LangevinSampler(logp, eps, metropolis_adjust=False)
            .sample(torch.zeros(6, 1, dtype=DTYPE), generator=g1, **kw).flat().var()
        )
        v_mala = float(
            LangevinSampler(logp, eps, metropolis_adjust=True)
            .sample(torch.zeros(6, 1, dtype=DTYPE), generator=g2, **kw).flat().var()
        )
        assert abs(v_mala - 1.0) < abs(v_ula - 1.0)


# --------------------------------------------------------------------- #
# GP hyperparameter posterior                                            #
# --------------------------------------------------------------------- #

class TestHyperposterior:
    def test_samples_finite_and_model_restored(self):
        model, x, y = _fit_gp(n=25)
        before = [p.detach().clone() for p in model.parameters()]
        g = torch.Generator().manual_seed(2)
        res = sample_gp_hyperposterior(
            model, x, y, n_samples=60, n_chains=3, burn_in=120, generator=g
        )
        assert res.samples.shape == (60, 3, 3)  # 1 iso lengthscale + sf2 + sn2
        assert torch.isfinite(res.samples).all()
        assert torch.isfinite(res.log_probs).all()
        # Model hyperparameters must be untouched by the sampling run.
        for p, b in zip(model.parameters(), before):
            assert torch.allclose(p.detach(), b)

    def test_posterior_concentrates_near_mle(self):
        """The posterior mode should give predictions comparable to the
        type-II ML fit: mean posterior LML within a few nats of the MLE."""
        model, x, y = _fit_gp(n=30)
        model.optimize_hyperparameters(x, y, n_iters=100, lr=0.1)
        lml_star = float(model.log_marginal_likelihood(x, y).detach())
        g = torch.Generator().manual_seed(4)
        res = sample_gp_hyperposterior(
            model, x, y, n_samples=100, n_chains=3, burn_in=200, step_size=0.1, generator=g
        )
        # log_probs include the prior; compare against the same quantity at the MLE.
        assert float(res.log_probs.max()) > lml_star - 15.0


# --------------------------------------------------------------------- #
# R-hat diagnostic                                                       #
# --------------------------------------------------------------------- #

class TestRhat:
    def test_identical_distribution_near_one(self):
        g = torch.Generator().manual_seed(0)
        samples = torch.randn(2000, 4, 3, dtype=DTYPE, generator=g)
        assert (potential_scale_reduction(samples) < 1.05).all()

    def test_separated_chains_large(self):
        g = torch.Generator().manual_seed(0)
        samples = torch.randn(500, 2, 1, dtype=DTYPE, generator=g)
        samples[:, 1] += 10.0  # chain 2 stuck in a different mode
        assert float(potential_scale_reduction(samples)) > 3.0

    def test_single_chain_raises(self):
        with pytest.raises(ValueError, match="2 chains"):
            potential_scale_reduction(torch.zeros(10, 1, 2, dtype=DTYPE))


# --------------------------------------------------------------------- #
# Expected Improvement                                                   #
# --------------------------------------------------------------------- #

class TestExpectedImprovement:
    def test_matches_monte_carlo(self):
        """Analytical EI must equal E[max(f - f*, 0)] estimated by sampling
        the GP posterior."""
        model, x, y = _fit_gp()
        ei = ExpectedImprovement(model)
        g = torch.Generator().manual_seed(0)
        xq = torch.rand(5, 2, dtype=DTYPE, generator=g) * 4 - 2

        analytical = ei(xq)
        f_star = float(y.max())
        paths = model.sample_posterior(xq, n_samples=200_000, generator=g)
        mc = (paths - f_star).clamp_min(0.0).mean(0)
        assert torch.allclose(analytical, mc, atol=5e-3)

    def test_nonnegative_and_zero_at_dominated_points(self):
        model, x, y = _fit_gp()
        g = torch.Generator().manual_seed(1)
        xq = torch.rand(200, 2, dtype=DTYPE, generator=g) * 4 - 2
        vals = ExpectedImprovement(model)(xq)
        assert (vals >= 0).all()
        # A huge incumbent makes improvement essentially impossible.
        assert ExpectedImprovement(model, best_f=1e3)(xq).max() < 1e-10

    def test_zero_variance_degenerates_correctly(self):
        """At sigma=0, EI must equal max(mu - f*, 0) with no NaNs."""
        ei = ExpectedImprovement.__new__(ExpectedImprovement)
        ei.best_f, ei.xi = 0.0, 0.0
        mean = torch.tensor([0.5, -0.5], dtype=DTYPE)
        sigma = torch.zeros(2, dtype=DTYPE)
        out = ei._ei(mean, sigma)
        assert torch.allclose(out, torch.tensor([0.5, 0.0], dtype=DTYPE))
        assert torch.isfinite(out).all()

    def test_analytical_gradient_matches_autograd(self):
        """The closed-form dEI/dx must agree with autograd through the
        differentiable posterior path."""
        model, x, y = _fit_gp()
        ei = ExpectedImprovement(model, xi=0.01)
        g = torch.Generator().manual_seed(2)
        xq = (torch.rand(6, 2, dtype=DTYPE, generator=g) * 4 - 2).requires_grad_(True)

        mean, var = model.posterior_diag(xq)
        vals = ei._ei(mean, var.clamp_min(1e-24).sqrt())
        vals.sum().backward()
        autograd_grad = xq.grad

        analytical = ei.gradient(xq.detach())
        assert torch.allclose(analytical, autograd_grad, atol=1e-8), (
            f"max err {(analytical - autograd_grad).abs().max()}"
        )

    def test_gradient_matches_finite_differences(self):
        model, x, y = _fit_gp()
        ei = ExpectedImprovement(model)
        xq = torch.tensor([[0.3, -0.7]], dtype=DTYPE)
        grad = ei.gradient(xq)
        h = 1e-6
        for d in range(2):
            e = torch.zeros_like(xq)
            e[0, d] = h
            fd = float((ei(xq + e) - ei(xq - e)) / (2 * h))
            assert abs(fd - float(grad[0, d])) < 1e-5


# --------------------------------------------------------------------- #
# Upper Confidence Bound                                                 #
# --------------------------------------------------------------------- #

class TestUCB:
    def test_value_formula(self):
        model, x, y = _fit_gp()
        ucb = UpperConfidenceBound(model, beta=4.0)
        g = torch.Generator().manual_seed(3)
        xq = torch.rand(10, 2, dtype=DTYPE, generator=g) * 4 - 2
        mean, var = model.posterior_diag(xq)
        assert torch.allclose(ucb(xq), (mean + 2.0 * var.sqrt()).detach(), atol=1e-12)

    def test_dynamic_beta_schedule_grows_logarithmically(self):
        model, *_ = _fit_gp()
        ucb = UpperConfidenceBound(model, delta=0.1, domain_size=500)
        betas = []
        for _ in range(50):
            betas.append(ucb.beta)
            ucb.step()
        betas = torch.tensor(betas, dtype=DTYPE)
        assert (betas.diff() > 0).all()          # strictly increasing
        assert (betas.diff().diff() < 0).all()   # concave: O(log t) growth
        # Exact Srinivas et al. formula at t=1.
        expected = 2 * math.log(500 * 1 * math.pi**2 / (6 * 0.1))
        assert abs(betas[0].item() - expected) < 1e-12

    def test_higher_beta_prefers_uncertain_points(self):
        model, x, y = _fit_gp()
        # "near": beside the BEST training point -> high mean, tiny sigma.
        # "far": prior regime -> mean reverts to y.mean(), sigma = sigma_f.
        near = x[y.argmax() : y.argmax() + 1] + 0.01
        far = torch.full((1, 2), 50.0, dtype=DTYPE)
        both = torch.cat([near, far])
        exploit = UpperConfidenceBound(model, beta=1e-6)(both)
        explore = UpperConfidenceBound(model, beta=100.0)(both)
        assert exploit.argmax() == 0 and explore.argmax() == 1


# --------------------------------------------------------------------- #
# Thompson Sampling                                                      #
# --------------------------------------------------------------------- #

class TestThompsonSampling:
    def test_selection_frequency_matches_posterior_argmax_probability(self):
        """TS pick frequencies must converge to P(candidate = argmax f)."""
        model, x, y = _fit_gp()
        ts = ThompsonSampling(model)
        g = torch.Generator().manual_seed(0)
        xq = torch.rand(8, 2, dtype=DTYPE, generator=g) * 4 - 2

        n = 40_000
        picks = ts.select(xq, n_select=n, generator=g)
        freq = torch.bincount(picks, minlength=8).double() / n

        g2 = torch.Generator().manual_seed(123)
        ref = model.sample_posterior(xq, n_samples=n, generator=g2).argmax(-1)
        ref_freq = torch.bincount(ref, minlength=8).double() / n
        assert torch.allclose(freq, ref_freq, atol=0.015)

    def test_shapes(self):
        model, *_ = _fit_gp()
        ts = ThompsonSampling(model)
        xq = torch.rand(12, 2, dtype=DTYPE) * 4 - 2
        assert ts(xq, n_samples=5).shape == (5, 12)
        idx = ts.select(xq, n_select=3)
        assert idx.shape == (3,) and (idx >= 0).all() and (idx < 12).all()

    def test_dominant_candidate_wins(self):
        """A candidate at a training point with a much higher target should
        be picked almost always."""
        g = torch.Generator().manual_seed(6)
        x = torch.rand(20, 1, generator=g, dtype=DTYPE)
        y = torch.zeros(20, dtype=DTYPE)
        y[7] = 5.0
        model = GaussianProcessRegressor(noise_variance=1e-4).fit(x, y)
        xq = torch.cat([x[7:8], torch.rand(5, 1, generator=g, dtype=DTYPE)])
        picks = ThompsonSampling(model).select(xq, n_select=500, generator=g)
        assert float((picks == 0).double().mean()) > 0.95


# --------------------------------------------------------------------- #
# Normal pdf/cdf primitives                                              #
# --------------------------------------------------------------------- #

class TestNormalPrimitives:
    def test_cdf_pdf_identities(self):
        z = torch.linspace(-5, 5, 101, dtype=DTYPE)
        assert torch.allclose(_Phi(z) + _Phi(-z), torch.ones_like(z), atol=1e-12)
        assert float(_Phi(torch.tensor(0.0, dtype=DTYPE))) == pytest.approx(0.5)
        # dPhi/dz = phi via finite differences.
        h = 1e-6
        fd = (_Phi(z + h) - _Phi(z - h)) / (2 * h)
        assert torch.allclose(fd, _phi(z), atol=1e-8)
