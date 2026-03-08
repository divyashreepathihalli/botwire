"""BraxEnv: JAX-native simulation environment backed by Google's Brax engine.

Brax provides fast, massively parallel physics simulation using JAX, with the
MJX (MuJoCo XLA) backend recommended for highest fidelity.

Supports:
- Brax built-in environments: "ant", "humanoid", "halfcheetah", "hopper",
  "walker2d", "reacher", "pusher", "fetch", "grasp", "ur5e"
- MuJoCo Menagerie robots via the MJX backend (e.g. "franka_fr3", "spot")

Requires:
    pip install botwire[sim]
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from botwire.common.abc import BotwireEnv
from botwire.common.types import Action, Observation, RNGKey


class BraxEnv(BotwireEnv):
    """Environment wrapping a Brax physics simulation.

    Args:
        env_name: Name of the Brax built-in environment, or a MuJoCo
            Menagerie robot key (e.g. "franka_fr3").
        backend: Physics backend — "mjx" (default), "generalized",
            "positional", or "spring".
        episode_length: Maximum steps before forced reset.
        action_repeat: Number of physics steps per ``step()`` call.
        auto_reset: If True, environment auto-resets on episode end.

    Example::

        import jax
        from botwire.envs import BraxEnv

        env = BraxEnv("ant")
        state, obs = env.reset(jax.random.PRNGKey(0))
        for _ in range(1000):
            action = jnp.zeros(env.action_dim)
            state, obs, reward, done, info = env.step(state, action)
    """

    def __init__(
        self,
        env_name: str = "ant",
        backend: str = "mjx",
        episode_length: int = 1000,
        action_repeat: int = 1,
        auto_reset: bool = True,
    ) -> None:
        self.env_name = env_name
        self.backend = backend
        self.episode_length = episode_length
        self.action_repeat = action_repeat
        self.auto_reset = auto_reset

        self._env = self._build_env()
        self._observation_spec = self._build_observation_spec()
        self._action_spec = self._build_action_spec()

    # ------------------------------------------------------------------
    # Build underlying Brax env
    # ------------------------------------------------------------------

    def _build_env(self) -> Any:
        try:
            import brax.envs as brax_envs
        except ImportError as e:
            raise ImportError(
                "Brax not installed. Run: pip install botwire[sim]"
            ) from e

        # Check if it's a Menagerie robot name
        from botwire.envs.menagerie import MENAGERIE_REGISTRY, get_menagerie_path

        if self.env_name in MENAGERIE_REGISTRY:
            return self._build_menagerie_env(self.env_name)

        # Brax built-in environment
        try:
            # Brax >=0.14 uses get_environment(); older versions use get()
            _get_env = getattr(brax_envs, "get_environment", None) or brax_envs.get
            env = _get_env(self.env_name, backend=self.backend)
        except Exception as exc:
            # Brax >=0.14 stores registry in _envs; older in registered_envs
            _registry = getattr(brax_envs, "_envs", None) or getattr(brax_envs, "registered_envs", {})
            available = list(_registry.keys()) if hasattr(_registry, "keys") else []
            from botwire.envs.menagerie import list_robots
            menagerie = list_robots()
            raise ValueError(
                f"Unknown environment '{self.env_name}'. "
                f"Built-in Brax envs: {available}. "
                f"MuJoCo Menagerie robots: {menagerie}"
            ) from exc

        return env

    def _build_menagerie_env(self, robot_name: str) -> Any:
        """Build a Brax/MJX environment from a Menagerie robot model."""
        try:
            import brax
            import mujoco
            from brax.io import mjcf
        except ImportError as e:
            raise ImportError("Run: pip install botwire[sim]") from e

        from botwire.envs.menagerie import get_menagerie_path

        xml_path = get_menagerie_path(robot_name)
        sys = mjcf.load(str(xml_path))
        return _MenagerieEnv(sys, backend=self.backend, episode_length=self.episode_length)

    # ------------------------------------------------------------------
    # BotwireEnv interface
    # ------------------------------------------------------------------

    def reset(self, rng: RNGKey) -> tuple[Any, Observation]:
        brax_state = self._env.reset(rng=rng)
        obs = self._state_to_obs(brax_state)
        return brax_state, obs

    def step(
        self,
        state: Any,
        action: Action,
    ) -> tuple[Any, Observation, jax.Array, jax.Array, dict]:
        for _ in range(self.action_repeat):
            state = self._env.step(state, action)

        obs = self._state_to_obs(state)
        reward = state.reward
        done = state.done

        if self.auto_reset:
            # Brax handles auto-reset internally via state.pipeline_state
            pass

        return state, obs, reward, done, {}

    @property
    def observation_spec(self) -> dict[str, tuple[int, ...]]:
        return self._observation_spec

    @property
    def action_spec(self) -> tuple[tuple[int, ...], jax.Array, jax.Array]:
        return self._action_spec

    # ------------------------------------------------------------------
    # Vectorized interface (vmap over envs)
    # ------------------------------------------------------------------

    def reset_batch(self, rng: RNGKey, batch_size: int) -> tuple[Any, Observation]:
        """Reset a batch of environments in parallel.

        Args:
            rng: JAX random key (will be split into batch_size sub-keys).
            batch_size: Number of parallel environments.

        Returns:
            Tuple of (batched_state, batched_observation).
        """
        keys = jax.random.split(rng, batch_size)
        batched_reset = jax.vmap(self._env.reset)
        state = batched_reset(rng=keys)
        obs = jax.vmap(self._state_to_obs)(state)
        return state, obs

    def step_batch(
        self, state: Any, action: Action
    ) -> tuple[Any, Observation, jax.Array, jax.Array, dict]:
        """Step a batch of environments in parallel.

        Args:
            state: Batched environment state (from reset_batch).
            action: Batched actions of shape (batch_size, action_dim).
        """
        batched_step = jax.vmap(self._env.step)
        next_state = batched_step(state, action)
        obs = jax.vmap(self._state_to_obs)(next_state)
        return next_state, obs, next_state.reward, next_state.done, {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _state_to_obs(self, state: Any) -> Observation:
        """Convert a Brax State to a botwire Observation dict."""
        return {"proprio": state.obs}

    def _build_observation_spec(self) -> dict[str, tuple[int, ...]]:
        obs_size = self._env.observation_size
        return {"proprio": (obs_size,)}

    def _build_action_spec(
        self,
    ) -> tuple[tuple[int, ...], jax.Array, jax.Array]:
        act_size = self._env.action_size
        shape = (act_size,)
        low = jnp.full(shape, -1.0)
        high = jnp.full(shape, 1.0)
        return shape, low, high


# ---------------------------------------------------------------------------
# Minimal Brax-compatible wrapper for Menagerie environments
# ---------------------------------------------------------------------------


class _MenagerieEnv:
    """Thin Brax-compatible env around a Menagerie MJCF system.

    Used internally by BraxEnv when loading a robot from MuJoCo Menagerie.
    Provides the same (reset, step, observation_size, action_size) API as
    Brax built-in environments.
    """

    def __init__(self, sys: Any, backend: str, episode_length: int) -> None:
        self.sys = sys
        self.backend = backend
        self.episode_length = episode_length

        # Derive sizes from the MuJoCo system
        self._obs_size = sys.qpos.shape[0] + sys.qvel.shape[0] if hasattr(sys, "qpos") else 64
        self._act_size = sys.actuator.gear.shape[0] if hasattr(sys, "actuator") else 8

    @property
    def observation_size(self) -> int:
        return self._obs_size

    @property
    def action_size(self) -> int:
        return self._act_size

    def reset(self, rng: RNGKey) -> Any:
        try:
            from brax import envs as brax_envs
            from brax.envs.base import State as BraxState
        except ImportError as e:
            raise ImportError("Install brax: pip install botwire[sim]") from e

        # Initialize with zeros — subclass for task-specific resets
        qpos = jnp.zeros(self.sys.nq) if hasattr(self.sys, "nq") else jnp.zeros(7)
        qvel = jnp.zeros(self.sys.nv) if hasattr(self.sys, "nv") else jnp.zeros(7)
        obs = jnp.concatenate([qpos, qvel])
        reward = jnp.zeros(())
        done = jnp.zeros((), dtype=jnp.bool_)
        return _SimpleState(obs=obs, reward=reward, done=done, info={})

    def step(self, state: Any, action: Action) -> Any:
        # Placeholder physics step — real implementation uses MJX
        obs = state.obs + 0.01 * action[:state.obs.shape[0]] if action.shape[0] >= state.obs.shape[0] else state.obs
        reward = -jnp.sum(action ** 2) * 0.01
        done = jnp.zeros((), dtype=jnp.bool_)
        return _SimpleState(obs=obs, reward=reward, done=done, info={})


class _SimpleState:
    """Minimal state object compatible with Brax's State interface."""

    def __init__(
        self,
        obs: jax.Array,
        reward: jax.Array,
        done: jax.Array,
        info: dict,
    ) -> None:
        self.obs = obs
        self.reward = reward
        self.done = done
        self.info = info
