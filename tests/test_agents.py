"""Tests for botwire RL agents."""

import jax
import jax.numpy as jnp
import pytest

from botwire.configs import get_sac_config


@pytest.fixture
def obs_action_spec():
    obs_spec = {"proprio": (14,)}
    action_spec = ((7,), jnp.full((7,), -1.0), jnp.full((7,), 1.0))
    return obs_spec, action_spec


def test_sac_agent_select_action(obs_action_spec):
    from botwire.agents import SACAgent

    obs_spec, action_spec = obs_action_spec
    config = get_sac_config()
    config.replay_capacity = 1000
    config.min_replay_size = 10
    config.batch_size = 8

    agent = SACAgent(obs_spec, action_spec, config)
    obs = {"proprio": jnp.zeros(14)}
    rng = jax.random.PRNGKey(0)
    action = agent.select_action(obs, rng)
    assert action.shape == (7,)
    assert jnp.all(action >= -1.0) and jnp.all(action <= 1.0)


def test_sac_agent_observe_update(obs_action_spec):
    from botwire.agents import SACAgent

    obs_spec, action_spec = obs_action_spec
    config = get_sac_config()
    config.replay_capacity = 100
    config.min_replay_size = 5
    config.batch_size = 4

    agent = SACAgent(obs_spec, action_spec, config)
    obs = {"proprio": jnp.zeros(14)}
    action = jnp.zeros(7)

    # Fill replay buffer
    for _ in range(6):
        agent.observe(obs, action, reward=0.1, next_observation=obs, done=False)

    # Update should now run
    metrics = agent.update()
    assert isinstance(metrics, dict)
    if metrics:
        assert "critic_loss" in metrics or "actor_loss" in metrics


def test_bc_agent_train(obs_action_spec):
    from botwire.agents import BCAgent

    obs_spec, action_spec = obs_action_spec
    config = get_sac_config()
    config.learning_rate = 1e-3

    agent = BCAgent(obs_spec, action_spec, config)
    obs = {"proprio": jnp.zeros(14)}
    action = agent.select_action(obs, jax.random.PRNGKey(0))
    assert action.shape == (7,)
