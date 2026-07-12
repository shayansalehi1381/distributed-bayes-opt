"""Integration tests for the reinforcement learning components."""

import pytest
import torch

from src.environment import ContinuousEnvironment, EnvConfig
from src.rl_agent import DoubleDQNAgent


def test_rl_agent_environment_integration() -> None:
    """Test the interaction between DoubleDQNAgent and ContinuousEnvironment.
    
    This ensures that:
    1. The agent can take actions in the environment.
    2. The environment states and rewards are correctly formatted float64 tensors/scalars.
    3. The agent can store transitions in its replay buffer.
    4. A batch can be sampled and an update step can be executed without dtype/shape errors,
       returning a valid float loss.
    """
    # 1. Initialize environment and agent
    config = EnvConfig(dim=2, n_actions=10, budget=5, noise_std=0.0)
    env = ContinuousEnvironment(config=config)
    
    agent = DoubleDQNAgent(
        state_dim=env.state_dim,
        n_actions=env.n_actions,
        hidden_dim=32,  # small for fast testing
        n_layers=1,     # small for fast testing
        buffer_capacity=100,
        dtype=torch.float64,
        device="cpu"
    )
    
    # 2. Run a short interaction loop for 5 steps
    state = env.reset(seed=42)
    assert state.dtype == torch.float64
    assert state.shape == (env.state_dim,)
    
    for _ in range(5):
        # Select action
        action = agent.select_action(state, epsilon=1.0)  # purely random for testing
        
        # Step environment
        next_state, reward, done, info = env.step(action)
        
        # Verify types
        assert next_state.dtype == torch.float64
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        
        # Store transition
        agent.store(state, action, reward, next_state, done)
        
        state = next_state
    
    # The budget is 5, so it should be done after 5 steps
    assert done is True
    assert len(agent.replay_buffer) == 5
    
    # 3. Execute a single agent.update(batch_size=4)
    loss = agent.update(batch_size=4)
    
    # Assert loss is a float and not None
    assert loss is not None
    assert isinstance(loss, float)
    
    # Ensure backward pass ran successfully
    for param in agent.online_net.parameters():
        assert param.grad is not None
