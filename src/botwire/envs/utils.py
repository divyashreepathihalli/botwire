"""Utility functions for environment handling."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from botwire.common.abc import BotwireEnv
from botwire.common.types import RNGKey


def collect_rollout(
    env: BotwireEnv,
    policy_fn: Any,
    params: Any,
    rng: RNGKey,
    num_steps: int,
) -> dict[str, list]:
    """Collect a single rollout from an environment using a policy function.

    Args:
        env: Any BotwireEnv instance.
        policy_fn: Callable (params, obs, rng) → action.
        params: Policy parameters.
        rng: JAX random key.
        num_steps: Number of steps to collect.

    Returns:
        Dict with "observations", "actions", "rewards", "dones" lists.
    """
    rng, reset_rng = jax.random.split(rng)
    state, obs = env.reset(reset_rng)

    rollout: dict[str, list] = {
        "observations": [],
        "actions": [],
        "rewards": [],
        "dones": [],
    }

    for _ in range(num_steps):
        rng, act_rng = jax.random.split(rng)
        action = policy_fn(params, obs, act_rng)

        rollout["observations"].append(obs)
        rollout["actions"].append(action)

        state, obs, reward, done, _ = env.step(state, action)
        rollout["rewards"].append(float(reward))
        rollout["dones"].append(bool(done))

        if bool(done):
            rng, reset_rng = jax.random.split(rng)
            state, obs = env.reset(reset_rng)

    return rollout


def evaluate_policy(
    env: BotwireEnv,
    policy_fn: Any,
    params: Any,
    rng: RNGKey,
    num_episodes: int = 10,
) -> dict[str, float]:
    """Evaluate a policy over multiple episodes.

    Args:
        env: Any BotwireEnv instance.
        policy_fn: Callable (params, obs, rng) → action.
        params: Policy parameters.
        rng: JAX random key.
        num_episodes: Number of evaluation episodes.

    Returns:
        Dict with "mean_return", "std_return", "mean_episode_length".
    """
    returns = []
    lengths = []

    for ep in range(num_episodes):
        rng, reset_rng = jax.random.split(rng)
        state, obs = env.reset(reset_rng)
        ep_return = 0.0
        ep_length = 0

        for _ in range(10_000):  # safety limit
            rng, act_rng = jax.random.split(rng)
            action = policy_fn(params, obs, act_rng, deterministic=True)
            state, obs, reward, done, _ = env.step(state, action)
            ep_return += float(reward)
            ep_length += 1
            if bool(done):
                break

        returns.append(ep_return)
        lengths.append(ep_length)

    returns_arr = np.array(returns)
    return {
        "mean_return": float(np.mean(returns_arr)),
        "std_return": float(np.std(returns_arr)),
        "min_return": float(np.min(returns_arr)),
        "max_return": float(np.max(returns_arr)),
        "mean_episode_length": float(np.mean(lengths)),
    }
