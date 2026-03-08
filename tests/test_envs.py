"""Tests for botwire environment wrappers."""

import jax
import jax.numpy as jnp
import pytest

from botwire.envs.menagerie import MENAGERIE_REGISTRY, list_robots


def test_menagerie_registry_not_empty():
    assert len(MENAGERIE_REGISTRY) > 0


def test_list_robots_all():
    robots = list_robots()
    assert len(robots) > 0
    assert "franka_fr3" in robots


def test_list_robots_by_category():
    arms = list_robots(category="arm")
    assert "franka_fr3" in arms
    assert "ur5e" in arms

    quadrupeds = list_robots(category="quadruped")
    assert "spot" in quadrupeds or "unitree_go2" in quadrupeds


@pytest.mark.sim
def test_brax_env_reset_step():
    """Requires brax to be installed."""
    pytest.importorskip("brax")

    from botwire.envs import BraxEnv

    env = BraxEnv(env_name="ant", backend="generalized")

    rng = jax.random.PRNGKey(0)
    state, obs = env.reset(rng)

    assert "proprio" in obs
    assert obs["proprio"].shape == (env.observation_spec["proprio"][0],)

    action = jnp.zeros(env.action_dim)
    state, obs, reward, done, info = env.step(state, action)

    assert reward.shape == ()
    assert done.shape == ()


def test_moma_env_reset_step():
    from botwire.envs.moma_env import MoMaEnv

    env = MoMaEnv(robot="franka_fr3", use_rgb=False)
    rng = jax.random.PRNGKey(0)
    state, obs = env.reset(rng)

    assert "proprio" in obs
    action = jnp.zeros(7)
    state, obs, reward, done, info = env.step(state, action)
    assert reward.shape == ()


def test_episode_limit_wrapper():
    from botwire.envs.moma_env import MoMaEnv
    from botwire.envs.wrappers import EpisodeLimit

    base_env = MoMaEnv(robot="franka_fr3", use_rgb=False, episode_length=1000)
    env = EpisodeLimit(base_env, max_steps=5)

    rng = jax.random.PRNGKey(0)
    state, obs = env.reset(rng)

    for _ in range(4):
        action = jnp.zeros(7)
        state, obs, reward, done, info = env.step(state, action)
        assert not bool(done), "Should not terminate before max_steps"

    # 5th step should terminate
    state, obs, reward, done, info = env.step(state, jnp.zeros(7))
    assert bool(done), "Should terminate at max_steps=5"
