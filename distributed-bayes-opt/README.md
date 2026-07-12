# Distributed Bayesian Optimization Framework

**via Advanced Stochastic Processes and Reinforcement Learning**

A research-grade framework implementing Bayesian optimization from first principles — exact Gaussian Process regression, advanced acquisition strategies, scalable sparse models, and a reinforcement learning (Double DQN) continuous decision pipeline. Core math is implemented from scratch with NumPy/PyTorch base operations to make the underlying theory explicit.

## Architecture Overview

The framework is built in four incrementally composed layers:

1. **Exact GPR (Step 1)**: RBF/ARD kernel, Cholesky inference, log marginal likelihood, type-II ML training.
2. **Acquisition & Sampling (Step 2)**: EI (analytical gradients), GP-UCB (dynamic beta), Thompson sampling (joint posterior paths), Metropolis-Hastings MCMC, and Langevin dynamics.
3. **Scalable Sparse GP (Step 3)**: Titsias VFE (collapsed bound) + SVGP/Hensman SVI, learnable inducing points. Achieves $O(nm^2)$ scalability.
4. **Double DQN RL Pipeline (Step 4)**: A continuous stochastic environment wrapping the BO search space, solved by a PyTorch-based Double DQN agent with a robust replay buffer and Huber loss optimizations, operating purely in `float64`.

## Benchmark Results

- **Sparse GP Scalability**: The Stochastic Variational Inference (SVI) module seamlessly scales to massive datasets. Using mini-batch SVI, training $N = 20,000$ points converges in **~6.35 seconds** on a standard CPU.
- **RL Integration**: The Continuous Environment and Double DQN agent pass full interaction loop integration tests natively in `float64`, perfectly marrying discrete action sequences with a continuous problem space smoothly.

## Project structure

```text
distributed-bayes-opt/
├── config/              # YAML experiment & model configurations
│   └── gp_config.yaml
├── src/                 # Core library
│   ├── gp_regression.py # Exact GPR
│   ├── sampling.py      # MCMC & Langevin dynamics
│   ├── acquisition.py   # Acquisition functions
│   ├── sparse_gp.py     # Sparse GP via VFE / SVI
│   ├── environment.py   # Continuous BO stochastic environment
│   └── rl_agent.py      # Double DQN agent and replay buffer
├── tests/               # Comprehensive Pytest suite
├── notebooks/           # Analysis & demonstration notebooks
└── pyproject.toml
```

## Roadmap

- [x] **Step 1** — Exact GP regression from scratch (RBF/ARD kernel, Cholesky-based posterior mean & covariance, differentiable log marginal likelihood, Adam hyperparameter optimization)
- [x] **Step 2** — Stochastic sampling & acquisition layer (MCMC, Langevin dynamics, EI, GP-UCB, Thompson sampling)
- [x] **Step 3** — Scalable sparse GPs via inducing points (Titsias VFE, SVGP/Hensman SVI, learnable inducing locations)
- [x] **Step 4** — Double DQN continuous RL decision pipeline (Episodic stochastic environment, target networks, replay buffer)
- [ ] **Step 5** — Distributed asynchronous optimization loop

## Installation

```bash
pip install -e ".[dev]"
```

## Quick start

### Bayesian optimization primitives (Step 1 & 2)

```python
import torch
from src.gp_regression import GaussianProcessRegressor
from src.acquisition import ExpectedImprovement

x = torch.rand(50, 2, dtype=torch.float64) * 4 - 2
y = torch.sin(2 * x[:, 0]) + 0.5 * torch.cos(3 * x[:, 1]) + 0.05 * torch.randn(50, dtype=torch.float64)

gp = GaussianProcessRegressor()
gp.optimize_hyperparameters(x, y, n_iters=200)

cands = torch.rand(512, 2, dtype=torch.float64) * 4 - 2
ei = ExpectedImprovement(gp)
best = cands[ei(cands).argmax()] 
```

### Scalable sparse GPs (Step 3)

```python
from src.sparse_gp import SparseGPRegressor, greedy_inducing_init

# N = 20,000 processed in ~6.35s
z = greedy_inducing_init(x_big, m=64)
sgp = SparseGPRegressor(z, learn_inducing=True)
sgp.fit_svi(x_big, y_big, n_epochs=30, batch_size=512)
```

### Reinforcement Learning Pipeline (Step 4)

```python
from src.environment import ContinuousEnvironment, EnvConfig
from src.rl_agent import DoubleDQNAgent

config = EnvConfig(dim=2, n_actions=50, budget=20)
env = ContinuousEnvironment(config=config)
agent = DoubleDQNAgent(state_dim=env.state_dim, n_actions=env.n_actions)

state = env.reset()
for _ in range(config.budget):
    action = agent.select_action(state, epsilon=0.1)
    next_state, reward, done, info = env.step(action)
    agent.store(state, action, reward, next_state, done)
    agent.update(batch_size=64)
    state = next_state
```

## Mathematical foundations

Given observations $(X, \mathbf{y})$ with $y = f(x) + \varepsilon$, $\varepsilon \sim \mathcal{N}(0, \sigma_n^2)$,
and RBF kernel $k(x, x') = \sigma_f^2 \exp\!\big(-\tfrac{1}{2}\sum_d (x_d - x'_d)^2 / \ell_d^2\big)$:

$$\mu_* = K_*^\top (K + \sigma_n^2 I)^{-1} \mathbf{y}, \qquad
\Sigma_* = K_{**} - K_*^\top (K + \sigma_n^2 I)^{-1} K_*$$

Inference uses a single Cholesky factorization $K + \sigma_n^2 I = LL^\top$ (never an explicit inverse), adaptive jitter for numerical robustness, and float64 throughout. Hyperparameters $(\ell, \sigma_f^2, \sigma_n^2)$ are learned by maximizing the exact log marginal likelihood with autograd.

## Testing

```bash
pytest tests/ -v
```

The suite verifies the Cholesky inference path against direct textbook formulas, checks statistical invariants (noise-free interpolation, prior reversion, variance contraction, PSD posterior covariance), validates Monte-Carlo moments of posterior samples, and guarantees robust PyTorch float64 interactions for Double DQN components.
