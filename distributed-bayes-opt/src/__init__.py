"""Distributed Bayesian Optimization Framework — core package."""

from src.acquisition import ExpectedImprovement, ThompsonSampling, UpperConfidenceBound
from src.gp_regression import GaussianProcessRegressor, GPPosterior, RBFKernel
from src.sampling import (
    LangevinSampler,
    MCMCResult,
    MetropolisHastings,
    potential_scale_reduction,
    sample_gp_hyperposterior,
)
from src.sparse_gp import SparseGPRegressor, greedy_inducing_init

__all__ = [
    "GaussianProcessRegressor",
    "GPPosterior",
    "RBFKernel",
    "SparseGPRegressor",
    "greedy_inducing_init",
    "ExpectedImprovement",
    "UpperConfidenceBound",
    "ThompsonSampling",
    "MetropolisHastings",
    "LangevinSampler",
    "MCMCResult",
    "sample_gp_hyperposterior",
    "potential_scale_reduction",
]
__version__ = "0.3.0"
