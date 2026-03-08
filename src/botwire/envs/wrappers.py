"""Environment wrappers that add functionality without changing the core env.

All wrappers implement the BotwireEnv interface and can be composed:

    env = BraxEnv("ant")
    env = EpisodeLimit(env, max_steps=500)
    env = ObsNormalize(env)
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from botwire.common.abc import BotwireEnv
from botwire.common.types import Action, Observation, RNGKey


class AutoReset(BotwireEnv):
    """Automatically reset the environment when an episode ends.

    When ``done=True``, the state and observation returned are those of a
    fresh episode, not the terminal transition.
    """

    def __init__(self, env: BotwireEnv) -> None:
        self._env = env

    def reset(self, rng: RNGKey) -> tuple[Any, Observation]:
        return self._env.reset(rng)

    def step(
        self,
        state: Any,
        action: Action,
    ) -> tuple[Any, Observation, jax.Array, jax.Array, dict]:
        next_state, obs, reward, done, info = self._env.step(state, action)

        def do_reset(_: Any) -> tuple[Any, Observation]:
            rng = jax.random.PRNGKey(0)  # placeholder; use state RNG in practice
            return self._env.reset(rng)

        def keep(_: Any) -> tuple[Any, Observation]:
            return next_state, obs

        reset_state, reset_obs = jax.lax.cond(done, do_reset, keep, None)
        return reset_state, reset_obs, reward, done, info

    @property
    def observation_spec(self) -> dict[str, tuple[int, ...]]:
        return self._env.observation_spec

    @property
    def action_spec(self) -> tuple[tuple[int, ...], jax.Array, jax.Array]:
        return self._env.action_spec


class EpisodeLimit(BotwireEnv):
    """Force episode termination after ``max_steps`` steps.

    Wraps any BotwireEnv and adds a step counter to the state.

    Args:
        env: The environment to wrap.
        max_steps: Maximum steps per episode before forced ``done=True``.
    """

    def __init__(self, env: BotwireEnv, max_steps: int = 1000) -> None:
        self._env = env
        self.max_steps = max_steps

    def reset(self, rng: RNGKey) -> tuple[Any, Observation]:
        state, obs = self._env.reset(rng)
        return {"inner": state, "step": jnp.zeros((), dtype=jnp.int32)}, obs

    def step(
        self,
        state: Any,
        action: Action,
    ) -> tuple[Any, Observation, jax.Array, jax.Array, dict]:
        inner_state, obs, reward, done, info = self._env.step(state["inner"], action)
        step = state["step"] + 1
        done = done | (step >= self.max_steps)
        new_state = {"inner": inner_state, "step": step}
        info = dict(info)
        info["episode_step"] = step
        return new_state, obs, reward, done, info

    @property
    def observation_spec(self) -> dict[str, tuple[int, ...]]:
        return self._env.observation_spec

    @property
    def action_spec(self) -> tuple[tuple[int, ...], jax.Array, jax.Array]:
        return self._env.action_spec


class ObsNormalize(BotwireEnv):
    """Normalize observations using running mean/std statistics.

    Statistics are updated online during environment interaction and can
    be saved/loaded for consistent normalization during evaluation.

    Args:
        env: The environment to wrap.
        clip: Clip normalized values to [-clip, clip]. None = no clipping.
    """

    def __init__(self, env: BotwireEnv, clip: float | None = 10.0) -> None:
        self._env = env
        self.clip = clip
        self._stats: dict[str, dict[str, np.ndarray]] = {}

        # Initialize stats with zeros
        for key, shape in env.observation_spec.items():
            self._stats[key] = {
                "mean": np.zeros(shape, dtype=np.float32),
                "var": np.ones(shape, dtype=np.float32),
                "count": np.array(0.0),
            }

    def reset(self, rng: RNGKey) -> tuple[Any, Observation]:
        state, obs = self._env.reset(rng)
        obs = self._normalize(obs)
        return state, obs

    def step(
        self,
        state: Any,
        action: Action,
    ) -> tuple[Any, Observation, jax.Array, jax.Array, dict]:
        next_state, obs, reward, done, info = self._env.step(state, action)
        self._update_stats(obs)
        obs = self._normalize(obs)
        return next_state, obs, reward, done, info

    def _update_stats(self, obs: Observation) -> None:
        for key, val in obs.items():
            x = np.array(val)
            s = self._stats[key]
            s["count"] += 1
            delta = x - s["mean"]
            s["mean"] += delta / s["count"]
            s["var"] = (s["var"] * (s["count"] - 1) + delta * (x - s["mean"])) / s["count"]

    def _normalize(self, obs: Observation) -> Observation:
        normed: Observation = {}
        for key, val in obs.items():
            s = self._stats[key]
            std = np.sqrt(s["var"] + 1e-8)
            normalized = (np.array(val) - s["mean"]) / std
            if self.clip is not None:
                normalized = np.clip(normalized, -self.clip, self.clip)
            normed[key] = jnp.array(normalized)
        return normed

    def save_stats(self, path: str) -> None:
        """Save normalization statistics to a NumPy file."""
        np.save(path, self._stats, allow_pickle=True)

    def load_stats(self, path: str) -> None:
        """Load normalization statistics from a NumPy file."""
        self._stats = np.load(path, allow_pickle=True).item()

    @property
    def observation_spec(self) -> dict[str, tuple[int, ...]]:
        return self._env.observation_spec

    @property
    def action_spec(self) -> tuple[tuple[int, ...], jax.Array, jax.Array]:
        return self._env.action_spec
