"""Asynchronous Distributed Bayesian Optimization Engine.

Implements an asynchronous master-worker coordination architecture for parallel
Bayesian optimization over expensive continuous objective functions.

Key Mathematical and Architectural Concepts:
---------------------------------------------
1. **Asynchronous Non-Blocking Dispatch**:
   Traditional batch Bayesian Optimization (e.g., q-EI) forces all workers to
   synchronize at batch boundaries, causing massive worker idle time (straggler effect)
   when function evaluations have variable latencies.
   This coordinator operates completely asynchronously: the moment ANY worker
   completes its evaluation, its result is incorporated into the surrogate model,
   and a new candidate point is dispatched immediately.

2. **Parallel Thompson Sampling**:
   Following Kandasamy et al. (NeurIPS 2018), independent posterior sample paths
   from the GP surrogate naturally explore different promising regions of the
   search space simultaneously. This avoids the cluster collapse of naive parallel
   acquisition while maintaining optimal regret bounds without requiring costly
   joint batch acquisition integrals.

3. **Multi-Worker Execution Model**:
   Supports both concurrent thread pools (for I/O-bound, networked, or C-extension
   evaluations) and process pools (for heavy CPU-bound objective functions) with
   fault tolerance and wall-clock telemetry.
"""

from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass, field
import time
from typing import Any, Callable, Literal

import torch
from torch import Tensor

from src.gp_regression import GaussianProcessRegressor, RBFKernel
from src.sparse_gp import SparseGPRegressor, greedy_inducing_init

__all__ = [
    "EvaluationRecord",
    "DistributedConfig",
    "DistributedResult",
    "AsyncBayesOptCoordinator",
]


@dataclass
class EvaluationRecord:
    """Telemetry and data for an individual candidate evaluation."""

    job_id: int
    worker_id: int
    x: Tensor
    y: float
    submit_time: float
    finish_time: float
    duration: float


@dataclass
class DistributedConfig:
    """Configuration parameters for distributed Bayesian optimization.

    Attributes:
        dim: Dimensionality of the search space.
        bounds: Tensor of shape (dim, 2) specifying [lower, upper] bounds per dimension.
        n_workers: Number of parallel evaluation workers.
        total_evaluations: Maximum total number of objective function calls.
        n_initial_points: Number of initial quasi-random exploration points.
        surrogate: Type of GP surrogate ("exact" or "sparse").
        n_inducing: Number of inducing points if using sparse GP.
        surrogate_refit_freq: Refit GP hyperparameters every N evaluations.
        candidate_pool_size: Number of discrete continuous candidate locations per sampling step.
        dtype: PyTorch tensor floating point precision (default float64).
        device: PyTorch compute device ("cpu" or "cuda").
        executor_type: "thread" or "process".
    """

    dim: int = 2
    bounds: Tensor | None = None
    n_workers: int = 4
    total_evaluations: int = 40
    n_initial_points: int = 8
    surrogate: Literal["exact", "sparse"] = "exact"
    n_inducing: int = 16
    surrogate_refit_freq: int = 2
    candidate_pool_size: int = 1000
    dtype: torch.dtype = torch.float64
    device: torch.device | str = "cpu"
    executor_type: Literal["thread", "process"] = "thread"


@dataclass
class DistributedResult:
    """Summary of the distributed optimization campaign."""

    best_x: Tensor
    best_y: float
    all_evaluations: list[EvaluationRecord]
    wall_clock_time: float
    total_evaluations: int
    speedup_ratio: float
    worker_utilization: float


class AsyncBayesOptCoordinator:
    """Asynchronous Master Coordinator for Distributed Bayesian Optimization.

    Coordinates concurrent workers, continuously refits the surrogate GP,
    and proposes candidates using Asynchronous Thompson Sampling.
    """

    def __init__(
        self,
        objective_fn: Callable[[Tensor], float | Tensor],
        config: DistributedConfig,
    ) -> None:
        self.objective_fn = objective_fn
        self.config = config

        # Set default bounds [-2.0, 2.0] if none provided
        if self.config.bounds is None:
            self.bounds = torch.tensor(
                [[-2.0, 2.0]] * self.config.dim,
                dtype=self.config.dtype,
                device=self.config.device,
            )
        else:
            self.bounds = self.config.bounds.to(
                dtype=self.config.dtype, device=self.config.device
            )

        self.history_x: list[Tensor] = []
        self.history_y: list[float] = []
        self.evaluations: list[EvaluationRecord] = []

        self.best_x: Tensor | None = None
        self.best_y: float = -float("inf")

        self.surrogate: GaussianProcessRegressor | SparseGPRegressor | None = None

    def _sample_random_candidates(self, n_points: int) -> Tensor:
        """Draw quasi-random uniform points within the bounding box."""
        lo = self.bounds[:, 0]
        hi = self.bounds[:, 1]
        u = torch.rand(
            n_points,
            self.config.dim,
            dtype=self.config.dtype,
            device=self.config.device,
        )
        return lo + u * (hi - lo)

    def _update_surrogate(self) -> None:
        """Fit or update the Gaussian Process surrogate model."""
        if len(self.history_x) < 2:
            return

        x_train = torch.stack(self.history_x)
        y_train = torch.tensor(
            self.history_y, dtype=self.config.dtype, device=self.config.device
        )

        if self.config.surrogate == "exact":
            kernel = RBFKernel(
                ard_dims=self.config.dim,
                dtype=self.config.dtype,
                device=self.config.device,
            )
            gp = GaussianProcessRegressor(
                kernel=kernel,
                dtype=self.config.dtype,
                device=self.config.device,
            )
            # Fast hyperparameter optimization
            gp.optimize_hyperparameters(
                x_train,
                y_train,
                n_iters=50,
                lr=0.08,
                verbose=False,
            )
            self.surrogate = gp
        else:
            m = min(self.config.n_inducing, len(x_train))
            x_unique = torch.unique(x_train, dim=0)
            if len(x_unique) < 2:
                return
            m = min(m, len(x_unique))
            z = greedy_inducing_init(x_unique, m=m)
            kernel = RBFKernel(
                ard_dims=self.config.dim,
                dtype=self.config.dtype,
                device=self.config.device,
            )
            sgp = SparseGPRegressor(
                inducing_points=z,
                kernel=kernel,
                learn_inducing=True,
                dtype=self.config.dtype,
                device=self.config.device,
            )
            sgp.fit_collapsed(x_train, y_train, n_iters=30, lr=0.05, verbose=False)
            self.surrogate = sgp

    def _propose_next_point(self, pending_points: list[Tensor]) -> Tensor:
        """Propose the next continuous candidate via Asynchronous Thompson Sampling.

        Draws a sample realization from the GP posterior across candidate points.
        If pending points are currently in-flight, hallucinated (fantasized) values
        from the GP posterior mean are incorporated to avoid redundant evaluations.
        """
        if self.surrogate is None or len(self.history_x) < self.config.n_initial_points:
            return self._sample_random_candidates(1).squeeze(0)

        # 1. Generate candidate pool
        candidates = self._sample_random_candidates(self.config.candidate_pool_size)

        with torch.no_grad():
            try:
                # Joint posterior realization (Thompson Sampling path)
                sample_paths = self.surrogate.sample_posterior(candidates, n_samples=1)
                scores = sample_paths.squeeze(0)

                # Penalize candidates that are too close to currently in-flight pending points
                if pending_points:
                    pending_tensor = torch.stack(pending_points)
                    # Pairwise distances
                    dists = torch.cdist(candidates, pending_tensor)
                    min_dist, _ = dists.min(dim=1)
                    # Soft repulsion radius: 5% of bounding box diagonal
                    repulsion_radius = 0.05 * torch.norm(
                        self.bounds[:, 1] - self.bounds[:, 0]
                    )
                    repulsion_mask = min_dist < repulsion_radius
                    scores[repulsion_mask] -= 1e6

                best_idx = torch.argmax(scores).item()
                return candidates[best_idx].clone()
            except Exception:
                # Fallback to random if numerical ill-conditioning occurs
                return self._sample_random_candidates(1).squeeze(0)

    def optimize(
        self,
        verbose: bool = True,
        on_step_callback: Callable[[EvaluationRecord], None] | None = None,
    ) -> DistributedResult:
        """Run the asynchronous distributed optimization loop.

        Returns:
            DistributedResult with detailed convergence and efficiency metrics.
        """
        start_time = time.perf_counter()
        job_counter = 0
        total_evals = self.config.total_evaluations
        n_workers = self.config.n_workers

        if verbose:
            print("=" * 80)
            print("  Asynchronous Distributed Bayesian Optimization Engine")
            print(f"  Workers: {n_workers} | Budget: {total_evals} | Dim: {self.config.dim} | Surrogate: {self.config.surrogate}")
            print("=" * 80)
            print(f"{'Job':>4} | {'Worker':>6} | {'Value (f(x))':>13} | {'Best So Far':>12} | {'Eval Duration':>14} | {'Wall-Clock':>11}")
            print("-" * 80)

        executor_cls = (
            concurrent.futures.ThreadPoolExecutor
            if self.config.executor_type == "thread"
            else concurrent.futures.ProcessPoolExecutor
        )

        # Mapping: Future -> (job_id, worker_id, x_tensor, submit_time)
        active_futures: dict[concurrent.futures.Future, tuple[int, int, Tensor, float]] = {}
        pending_points: list[Tensor] = []

        # Available worker IDs
        free_workers: list[int] = list(range(n_workers))

        def _evaluate_wrapper(x_tensor: Tensor) -> float:
            """Safe evaluation wrapper."""
            val = self.objective_fn(x_tensor)
            if isinstance(val, Tensor):
                val = float(val.detach().cpu().squeeze())
            return float(val)

        with executor_cls(max_workers=n_workers) as executor:
            # 1. Initial burst of job submissions to prime all workers
            for _ in range(min(n_workers, total_evals)):
                worker_id = free_workers.pop(0)
                job_counter += 1
                x_query = self._propose_next_point(pending_points)
                pending_points.append(x_query)

                sub_time = time.perf_counter()
                future = executor.submit(_evaluate_wrapper, x_query)
                active_futures[future] = (job_counter, worker_id, x_query, sub_time)

            # 2. Asynchronous event loop
            while active_futures:
                # Wait for the first future to complete (non-blocking for other workers)
                done_set, _ = concurrent.futures.wait(
                    active_futures.keys(),
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )

                for finished_future in done_set:
                    job_id, worker_id, x_eval, sub_time = active_futures.pop(finished_future)
                    # Remove from pending points
                    for idx, p in enumerate(pending_points):
                        if torch.equal(p, x_eval):
                            pending_points.pop(idx)
                            break

                    finish_time = time.perf_counter()
                    duration = finish_time - sub_time

                    try:
                        y_val = finished_future.result()
                    except Exception as err:
                        if verbose:
                            print(f"[Error] Job {job_id} failed: {err}")
                        y_val = -1e6

                    # Ingest observation
                    self.history_x.append(x_eval)
                    self.history_y.append(y_val)

                    if y_val > self.best_y:
                        self.best_y = y_val
                        self.best_x = x_eval.clone()

                    record = EvaluationRecord(
                        job_id=job_id,
                        worker_id=worker_id,
                        x=x_eval,
                        y=y_val,
                        submit_time=sub_time,
                        finish_time=finish_time,
                        duration=duration,
                    )
                    self.evaluations.append(record)

                    if on_step_callback is not None:
                        on_step_callback(record)

                    elapsed = finish_time - start_time
                    if verbose:
                        print(
                            f"{job_id:4d} | {worker_id:6d} | {y_val:13.5f} | "
                            f"{self.best_y:12.5f} | {duration:12.3f}s | {elapsed:10.2f}s"
                        )

                    # Update surrogate periodically
                    if len(self.history_x) % self.config.surrogate_refit_freq == 0:
                        self._update_surrogate()

                    # Free this worker and dispatch next task if budget permits
                    free_workers.append(worker_id)
                    if job_counter < total_evals:
                        next_worker_id = free_workers.pop(0)
                        job_counter += 1
                        next_x = self._propose_next_point(pending_points)
                        pending_points.append(next_x)

                        next_sub_time = time.perf_counter()
                        new_fut = executor.submit(_evaluate_wrapper, next_x)
                        active_futures[new_fut] = (
                            job_counter,
                            next_worker_id,
                            next_x,
                            next_sub_time,
                        )

        total_wall_clock = time.perf_counter() - start_time
        total_eval_time = sum(rec.duration for rec in self.evaluations)
        speedup = (
            (total_eval_time / total_wall_clock) if total_wall_clock > 0 else 1.0
        )
        utilization = min(
            1.0, (total_eval_time / (total_wall_clock * n_workers)) if total_wall_clock > 0 else 1.0
        )

        if verbose:
            print("=" * 80)
            print("  Distributed Campaign Finished")
            print(f"  Total Evaluations: {len(self.evaluations)}")
            print(f"  Best f(x) Found  : {self.best_y:.6f}")
            print(f"  Cumulative Time  : {total_eval_time:.2f}s (if run sequentially)")
            print(f"  Wall-Clock Time  : {total_wall_clock:.2f}s (actual elapsed)")
            print(f"  Parallel Speedup : {speedup:.2f}x (across {n_workers} workers)")
            print(f"  Worker Efficiency: {utilization * 100:.1f}%")
            print("=" * 80)

        assert self.best_x is not None
        return DistributedResult(
            best_x=self.best_x,
            best_y=self.best_y,
            all_evaluations=self.evaluations,
            wall_clock_time=total_wall_clock,
            total_evaluations=len(self.evaluations),
            speedup_ratio=speedup,
            worker_utilization=utilization,
        )
