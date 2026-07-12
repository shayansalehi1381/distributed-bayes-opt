"""Double DQN agent for Bayesian-optimization-guided reinforcement learning.

Public API
----------
* ``QNetwork``       -- float64 MLP mapping states to per-action Q-values.
* ``ReplayBuffer``   -- fixed-capacity circular buffer with uniform sampling.
* ``Transition``     -- NamedTuple holding one (s, a, r, s', done) experience.
* ``DoubleDQNAgent`` -- Double DQN agent with epsilon-greedy policy, Adam
                        optimiser, Huber loss, and hard target-network sync.
"""

from __future__ import annotations

import random
from collections import deque
from typing import NamedTuple

import torch
import torch.nn as nn
from torch import Tensor

__all__ = ["QNetwork", "ReplayBuffer", "Transition", "DoubleDQNAgent"]


# ---------------------------------------------------------------------------
# Transition container
# ---------------------------------------------------------------------------

class Transition(NamedTuple):
    """A single (s, a, r, s_next, done) experience tuple.

    All tensors are 1-D (un-batched) and stored in float64 on the same device
    as the network to avoid dtype/device mismatches during sampling.

    Attributes:
        state:      (state_dim,) float64 observation before the action.
        action:     scalar int64 tensor -- index into the discrete action set.
        reward:     scalar float64 tensor.
        next_state: (state_dim,) float64 observation after the action.
        done:       scalar float64 tensor (1.0 = episode ended, 0.0 otherwise).
    """

    state: Tensor
    action: Tensor
    reward: Tensor
    next_state: Tensor
    done: Tensor


# ---------------------------------------------------------------------------
# Q-Network
# ---------------------------------------------------------------------------

class QNetwork(nn.Module):
    """Multi-layer perceptron that estimates Q-values for all discrete actions.

    The network is registered entirely in **float64** so that it is numerically
    consistent with the GP / BO stack and the ContinuousEnvironment state
    vectors (which are float64 by design).

    Architecture (default):
        Linear(state_dim -> hidden) -> LayerNorm -> ReLU
        Linear(hidden    -> hidden) -> LayerNorm -> ReLU
        Linear(hidden    -> n_actions)            (raw Q-values, no activation)

    Parameters
    ----------
    state_dim : int
        Dimensionality of the flat state vector produced by
        ContinuousEnvironment._encode_state().  Equals 2*d + 3.
    n_actions : int
        Number of discrete candidate actions (= EnvConfig.n_actions).
    hidden_dim : int
        Width of each hidden layer.  Defaults to 128.
    n_layers : int
        Number of hidden layers.  Must be >= 1.  Defaults to 2.
    dtype : torch.dtype
        Tensor dtype for all parameters.  Should be torch.float64 (default)
        to match the environment and replay buffer.
    device : torch.device or str
        Device for all parameters.  Defaults to CPU.

    Examples
    --------
    >>> net = QNetwork(state_dim=7, n_actions=50)
    >>> s = torch.zeros(7, dtype=torch.float64)
    >>> q = net(s)
    >>> q.shape
    torch.Size([50])
    >>> q.dtype
    torch.float64
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_dim: int = 128,
        n_layers: int = 2,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()

        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}.")

        self.state_dim = state_dim
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim

        # Build the MLP as a Sequential stack.
        layers: list[nn.Module] = []
        in_features = state_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            in_features = hidden_dim
        layers.append(nn.Linear(in_features, n_actions))

        self.net = nn.Sequential(*layers)

        # Cast all parameters to the requested dtype and move to device.
        self.to(dtype=dtype, device=device)

    def forward(self, state: Tensor) -> Tensor:
        """Compute Q-values for every action.

        Parameters
        ----------
        state : Tensor
            (state_dim,) or (batch, state_dim) float64 tensor.

        Returns
        -------
        Tensor
            (n_actions,) or (batch, n_actions) float64 Q-values.
        """
        return self.net(state)


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Fixed-capacity circular experience replay buffer with uniform sampling.

    Transitions are stored as individual Transition named-tuples and retrieved
    as batches of stacked tensors -- ready to feed directly into the Q-network
    without extra dtype/device conversion.

    Parameters
    ----------
    capacity : int
        Maximum number of transitions to keep.  When full, the oldest
        experience is silently evicted (FIFO, via collections.deque).
    device : torch.device or str
        Device on which sampled batch tensors will be placed.
    dtype : torch.dtype
        Floating-point dtype for state, reward, and done tensors.
        Defaults to torch.float64.
    seed : int, optional
        Optional integer seed for the Python random module used during
        sampling, enabling reproducible mini-batches.

    Examples
    --------
    >>> buf = ReplayBuffer(capacity=1000, seed=0)
    >>> buf.push(
    ...     state=torch.zeros(7, dtype=torch.float64),
    ...     action=0,
    ...     reward=1.5,
    ...     next_state=torch.ones(7, dtype=torch.float64),
    ...     done=False,
    ... )
    >>> len(buf)
    1
    >>> batch = buf.sample(1)
    >>> batch.state.shape
    torch.Size([1, 7])
    """

    def __init__(
        self,
        capacity: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float64,
        seed: int | None = None,
    ) -> None:
        self.capacity = capacity
        self.device = torch.device(device)
        self.dtype = dtype
        self._buffer: deque[Transition] = deque(maxlen=capacity)

        if seed is not None:
            random.seed(seed)

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def push(
        self,
        state: Tensor,
        action: int,
        reward: float,
        next_state: Tensor,
        done: bool,
    ) -> None:
        """Add a single transition to the buffer.

        Parameters
        ----------
        state : Tensor
            (state_dim,) float64 tensor -- observation before the step.
        action : int
            Integer index of the chosen action.
        reward : float
            Scalar reward received.
        next_state : Tensor
            (state_dim,) float64 tensor -- observation after the step.
        done : bool
            True if the episode ended after this transition.
        """
        t = Transition(
            state=state.to(dtype=self.dtype, device=self.device).detach(),
            action=torch.tensor(action, dtype=torch.int64, device=self.device),
            reward=torch.tensor(reward, dtype=self.dtype, device=self.device),
            next_state=next_state.to(dtype=self.dtype, device=self.device).detach(),
            done=torch.tensor(float(done), dtype=self.dtype, device=self.device),
        )
        self._buffer.append(t)

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def sample(self, batch_size: int) -> Transition:
        """Draw a random mini-batch (without replacement) from the buffer.

        Parameters
        ----------
        batch_size : int
            Number of transitions to sample.  Must not exceed len(self).

        Returns
        -------
        Transition
            A Transition of *stacked* tensors, each with a leading batch dim:
                state      -> (batch_size, state_dim)
                action     -> (batch_size,)
                reward     -> (batch_size,)
                next_state -> (batch_size, state_dim)
                done       -> (batch_size,)

        Raises
        ------
        ValueError
            If batch_size is larger than the number of stored transitions.
        """
        if batch_size > len(self):
            raise ValueError(
                f"Requested batch_size={batch_size} but buffer only contains "
                f"{len(self)} transitions."
            )

        transitions = random.sample(list(self._buffer), batch_size)

        return Transition(
            state=torch.stack([t.state for t in transitions]),
            action=torch.stack([t.action for t in transitions]),
            reward=torch.stack([t.reward for t in transitions]),
            next_state=torch.stack([t.next_state for t in transitions]),
            done=torch.stack([t.done for t in transitions]),
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Current number of stored transitions."""
        return len(self._buffer)

    def __repr__(self) -> str:
        return (
            f"ReplayBuffer(capacity={self.capacity}, "
            f"stored={len(self)}, dtype={self.dtype}, device={self.device})"
        )

    @property
    def is_ready(self) -> bool:
        """True when the buffer holds at least one transition."""
        return len(self._buffer) > 0


# ---------------------------------------------------------------------------
# Double DQN Agent
# ---------------------------------------------------------------------------

class DoubleDQNAgent:
    """Double Deep Q-Network agent for discrete-action continuous environments.

    Implements the Double DQN algorithm (van Hasselt et al., 2016) to reduce
    the overestimation bias of standard DQN.  The key idea is to decouple
    *action selection* (done by ``online_net``) from *action evaluation* (done
    by ``target_net``) when computing the Bellman target.

    Architecture overview
    ---------------------
    * **online_net** -- the network being actively trained via gradient descent.
    * **target_net** -- a periodically-synced frozen copy used for stable
      bootstrap targets.  Updated via hard copy every ``target_update_freq``
      calls to :meth:`update`.

    Parameters
    ----------
    state_dim : int
        Dimensionality of the observation vector (= ``2*d + 3`` for
        ``ContinuousEnvironment``).
    n_actions : int
        Size of the discrete action space (= ``EnvConfig.n_actions``).
    hidden_dim : int
        Width of each hidden layer in both Q-networks.  Defaults to 128.
    n_layers : int
        Number of hidden layers.  Defaults to 2.
    lr : float
        Adam learning rate for ``online_net``.  Defaults to 1e-3.
    gamma : float
        Discount factor in [0, 1].  Defaults to 0.99.
    target_update_freq : int
        Number of :meth:`update` calls between hard target-network syncs.
        Defaults to 100.
    buffer_capacity : int
        Maximum replay buffer capacity.  Defaults to 50_000.
    buffer_seed : int or None
        Optional seed for reproducible replay-buffer sampling.
    dtype : torch.dtype
        Floating-point dtype for networks and buffer.  Defaults to
        ``torch.float64``.
    device : torch.device or str
        Compute device.  Defaults to CPU.

    Examples
    --------
    >>> from environment import ContinuousEnvironment, EnvConfig
    >>> cfg = EnvConfig(dim=2, n_actions=20, budget=10)
    >>> env = ContinuousEnvironment(config=cfg)
    >>> agent = DoubleDQNAgent(state_dim=env.state_dim, n_actions=env.n_actions)
    >>> state = env.reset()
    >>> action = agent.select_action(state, epsilon=0.1)
    >>> isinstance(action, int)
    True
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        hidden_dim: int = 128,
        n_layers: int = 2,
        lr: float = 1e-3,
        gamma: float = 0.99,
        target_update_freq: int = 100,
        buffer_capacity: int = 50_000,
        buffer_seed: int | None = None,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str = "cpu",
    ) -> None:
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.dtype = dtype
        self.device = torch.device(device)

        # ---- networks ----
        net_kwargs = dict(
            state_dim=state_dim,
            n_actions=n_actions,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            dtype=dtype,
            device=self.device,
        )
        self.online_net = QNetwork(**net_kwargs)
        self.target_net = QNetwork(**net_kwargs)
        # Synchronise target weights with online weights at initialisation.
        self._sync_target()
        # Target network is never directly trained.
        for p in self.target_net.parameters():
            p.requires_grad_(False)

        # ---- optimiser ----
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=lr)

        # ---- loss ----
        # Huber / SmoothL1 loss is less sensitive to outlier Q-value errors
        # than MSE and is standard in DQN implementations.
        self.loss_fn = nn.SmoothL1Loss()

        # ---- replay buffer ----
        self.replay_buffer = ReplayBuffer(
            capacity=buffer_capacity,
            device=self.device,
            dtype=dtype,
            seed=buffer_seed,
        )

        # Internal counter used to trigger target-network hard updates.
        self._update_count: int = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _sync_target(self) -> None:
        """Hard-copy all parameters from online_net into target_net."""
        self.target_net.load_state_dict(self.online_net.state_dict())

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, state: Tensor, epsilon: float = 0.0) -> int:
        """Epsilon-greedy action selection.

        With probability *epsilon* a uniformly random action index is returned
        (exploration); otherwise the greedy action ``argmax Q_online(s)`` is
        chosen (exploitation).

        Parameters
        ----------
        state : Tensor
            ``(state_dim,)`` float64 state vector from the environment.
        epsilon : float
            Exploration probability in [0, 1].  Pass 0.0 for pure greedy
            evaluation (e.g. at test time).

        Returns
        -------
        int
            Chosen action index in ``[0, n_actions)``.
        """
        if random.random() < epsilon:
            return random.randrange(self.n_actions)

        # Greedy: forward pass through online_net without gradient tracking.
        self.online_net.eval()
        with torch.no_grad():
            s = state.to(dtype=self.dtype, device=self.device)
            if s.dim() == 1:
                s = s.unsqueeze(0)          # (1, state_dim)
            q_values = self.online_net(s)   # (1, n_actions)
            action = int(q_values.argmax(dim=-1).item())
        self.online_net.train()
        return action

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def update(self, batch_size: int = 64) -> float | None:
        """Sample from the replay buffer and perform one gradient step.

        Implements the Double DQN Bellman target:

            a* = argmax_a  Q_online(s', a)          (action selection)
            Y  = r + gamma * Q_target(s', a*)       (action evaluation)
                 -- with Y = r when done = 1        (terminal masking)

        Then minimises the Huber loss between Y and Q_online(s, a) using Adam.

        Parameters
        ----------
        batch_size : int
            Mini-batch size.  The method is a no-op when the replay buffer
            contains fewer than ``batch_size`` transitions.

        Returns
        -------
        float or None
            The scalar training loss for logging, or ``None`` if the buffer
            is not yet large enough to sample a full batch.
        """
        if len(self.replay_buffer) < batch_size:
            return None

        # ---- sample ----
        batch = self.replay_buffer.sample(batch_size)
        # batch.* shapes:
        #   state, next_state : (B, state_dim)
        #   action            : (B,)  int64
        #   reward            : (B,)  float64
        #   done              : (B,)  float64  (0.0 / 1.0)

        # ---- Double DQN target ----
        with torch.no_grad():
            # Step 1 -- action selection by online_net.
            next_q_online = self.online_net(batch.next_state)   # (B, n_actions)
            best_actions = next_q_online.argmax(dim=-1)         # (B,)

            # Step 2 -- action evaluation by target_net.
            next_q_target = self.target_net(batch.next_state)   # (B, n_actions)
            # Gather Q-values of the selected actions: shape (B,)
            next_q_values = next_q_target.gather(
                dim=1, index=best_actions.unsqueeze(1)
            ).squeeze(1)

            # Step 3 -- Bellman target (terminal masking via (1 - done)).
            targets = batch.reward + self.gamma * next_q_values * (1.0 - batch.done)

        # ---- online Q-values for taken actions ----
        # Q_online(s, a) for the actual action taken: shape (B,)
        q_pred = self.online_net(batch.state).gather(
            dim=1, index=batch.action.unsqueeze(1)
        ).squeeze(1)

        # ---- Huber loss & gradient step ----
        loss = self.loss_fn(q_pred, targets)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for training stability.
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # ---- periodic hard target-network update ----
        self._update_count += 1
        if self._update_count % self.target_update_freq == 0:
            self.update_target_network()

        return float(loss.item())

    # ------------------------------------------------------------------
    # Target network management
    # ------------------------------------------------------------------

    def update_target_network(self, tau: float = 1.0) -> None:
        """Synchronise target_net with online_net.

        Supports both **hard** (tau = 1.0, the default) and **soft** (Polyak)
        updates.  The hard update is triggered automatically by :meth:`update`
        every ``target_update_freq`` gradient steps.  Call this method directly
        with ``tau < 1.0`` if you prefer a soft-update schedule.

        Parameters
        ----------
        tau : float
            Interpolation coefficient in (0, 1].  ``tau = 1.0`` performs a
            full hard copy; ``tau < 1.0`` performs the Polyak average::

                theta_target <- tau * theta_online + (1 - tau) * theta_target
        """
        if tau == 1.0:
            self._sync_target()
        else:
            with torch.no_grad():
                for p_online, p_target in zip(
                    self.online_net.parameters(),
                    self.target_net.parameters(),
                ):
                    p_target.data.copy_(
                        tau * p_online.data + (1.0 - tau) * p_target.data
                    )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def store(
        self,
        state: Tensor,
        action: int,
        reward: float,
        next_state: Tensor,
        done: bool,
    ) -> None:
        """Push one transition into the replay buffer.

        Thin wrapper around ``ReplayBuffer.push`` for ergonomic use inside a
        training loop::

            agent.store(state, action, reward, next_state, done)
            loss = agent.update(batch_size=64)

        Parameters
        ----------
        state : Tensor
            Pre-step observation from the environment.
        action : int
            Action index that was executed.
        reward : float
            Scalar reward observed.
        next_state : Tensor
            Post-step observation from the environment.
        done : bool
            Whether the episode terminated after this step.
        """
        self.replay_buffer.push(state, action, reward, next_state, done)

    def __repr__(self) -> str:
        buf_len = len(self.replay_buffer)
        return (
            f"DoubleDQNAgent("
            f"state_dim={self.state_dim}, n_actions={self.n_actions}, "
            f"gamma={self.gamma}, target_update_freq={self.target_update_freq}, "
            f"buffer={buf_len}/{self.replay_buffer.capacity}, "
            f"updates={self._update_count})"
        )
