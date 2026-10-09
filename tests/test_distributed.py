"""Comprehensive unit and integration tests for the Asynchronous Distributed Engine."""

from __future__ import annotations

import time
import pytest
import torch
from torch import Tensor

from src.distributed import (
    AsyncBayesOptCoordinator,
    DistributedConfig,
    DistributedResult,
    EvaluationRecord,
)


def test_distributed_config_defaults() -> None:
    """Verify DistributedConfig default properties and bounds."""
    cfg = DistributedConfig(dim=3, n_workers=4, total_evaluations=20)
    assert cfg.dim == 3
    assert cfg.n_workers == 4
    assert cfg.total_evaluations == 20
    assert cfg.surrogate == "exact"
    assert cfg.dtype == torch.float64


def test_async_coordinator_convergence() -> None:
    """Verify asynchronous optimization convergence on a negated sphere objective."""
    # Negated sphere function: global maximum at [0.0, 0.0] with f(x) = 0.0
    def sphere_obj(x: Tensor) -> float:
        return float(-torch.sum(x**2))

    bounds = torch.tensor([[-2.0, 2.0], [-2.0, 2.0]], dtype=torch.float64)
    cfg = DistributedConfig(
        dim=2,
        bounds=bounds,
        n_workers=2,
        total_evaluations=16,
        n_initial_points=4,
        candidate_pool_size=300,
    )

    coordinator = AsyncBayesOptCoordinator(sphere_obj, cfg)
    result = coordinator.optimize(verbose=False)

    assert isinstance(result, DistributedResult)
    assert len(result.all_evaluations) == 16
    assert result.best_y > -1.0, f"Expected convergence near 0, got {result.best_y}"
    assert result.best_x.shape == (2,)


def test_parallel_speedup_concurrency() -> None:
    """Verify that multi-worker execution achieves real parallel speedup."""
    sleep_time = 0.06

    def delayed_objective(x: Tensor) -> float:
        time.sleep(sleep_time)
        return float(-torch.sum(x**2))

    n_workers = 4
    n_evals = 8
    cfg = DistributedConfig(
        dim=2,
        n_workers=n_workers,
        total_evaluations=n_evals,
        n_initial_points=4,
        candidate_pool_size=100,
    )

    coordinator = AsyncBayesOptCoordinator(delayed_objective, cfg)
    result = coordinator.optimize(verbose=False)

    # If run sequentially: n_evals * sleep_time = 8 * 0.06 = 0.48s
    # With 4 workers: ~ 2 * 0.06 = 0.12s + overhead
    cumulative_time = sum(rec.duration for rec in result.all_evaluations)
    assert cumulative_time >= n_evals * (sleep_time * 0.8)
    assert result.speedup_ratio > 1.8, f"Expected speedup > 1.8x, got {result.speedup_ratio}x"


def test_sparse_surrogate_distributed() -> None:
    """Verify distributed optimization operating with Sparse GP surrogate."""
    def simple_obj(x: Tensor) -> float:
        return float(torch.sin(x[0]) + torch.cos(x[1]))

    cfg = DistributedConfig(
        dim=2,
        n_workers=2,
        total_evaluations=12,
        n_initial_points=4,
        surrogate="sparse",
        n_inducing=8,
        candidate_pool_size=200,
    )

    coordinator = AsyncBayesOptCoordinator(simple_obj, cfg)
    result = coordinator.optimize(verbose=False)

    assert len(result.all_evaluations) == 12
    assert result.best_y is not None


def test_fault_tolerance() -> None:
    """Verify coordinator gracefully handles worker evaluation exceptions."""
    counter = 0

    def intermittent_failing_obj(x: Tensor) -> float:
        nonlocal counter
        counter += 1
        if counter == 3:
            raise RuntimeError("Simulated worker transient failure")
        return float(-torch.norm(x))

    cfg = DistributedConfig(
        dim=2,
        n_workers=2,
        total_evaluations=8,
        n_initial_points=3,
        candidate_pool_size=100,
    )

    coordinator = AsyncBayesOptCoordinator(intermittent_failing_obj, cfg)
    result = coordinator.optimize(verbose=False)

    # All 8 evaluations completed without crash
    assert len(result.all_evaluations) == 8
    # The failed job was assigned a penalty and did not break the pipeline
    assert any(rec.y <= -1e5 for rec in result.all_evaluations)
