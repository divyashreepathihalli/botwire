"""Base class for botwire RL agents."""

from __future__ import annotations

import abc
from collections import deque
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp

from botwire.common.abc import BotwireAgent
from botwire.common.types import Action, Metrics, Observation, Params, RNGKey


class ReplayBuffer:
    """Simple in-memory circular replay buffer.

    Stores transitions as NumPy arrays. Designed for single-machine training;
    use Reverb (via Acme) for distributed or large-scale setups.

    Args:
        capacity: Maximum number of transitions.
        observation_spec: Dict mapping obs key → shape.
        action_shape: Shape of action arrays.
    """

    def __init__(
        self,
        capacity: int,
        observation_spec: dict[str, tuple[int, ...]],
        action_shape: tuple[int, ...],
    ) -> None:
        self.capacity = capacity
        self._ptr = 0
        self._size = 0

        # Pre-allocate storage
        self._obs: dict[str, np.ndarray] = {
            k: np.zeros((capacity, *shape), dtype=np.float32)
            for k, shape in observation_spec.items()
        }
        self._next_obs: dict[str, np.ndarray] = {
            k: np.zeros((capacity, *shape), dtype=np.float32)
            for k, shape in observation_spec.items()
        }
        self._actions = np.zeros((capacity, *action_shape), dtype=np.float32)
        self._rewards = np.zeros((capacity,), dtype=np.float32)
        self._dones = np.zeros((capacity,), dtype=np.float32)

    def add(
        self,
        obs: Observation,
        action: Action,
        reward: float,
        next_obs: Observation,
        done: bool,
    ) -> None:
        for k in self._obs:
            self._obs[k][self._ptr] = np.array(obs.get(k, 0))
            self._next_obs[k][self._ptr] = np.array(next_obs.get(k, 0))
        self._actions[self._ptr] = np.array(action)
        self._rewards[self._ptr] = reward
        self._dones[self._ptr] = float(done)
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict[str, Any]:
        idxs = rng.integers(0, self._size, size=batch_size)
        return {
            "observation": {k: jnp.array(v[idxs]) for k, v in self._obs.items()},
            "next_observation": {k: jnp.array(v[idxs]) for k, v in self._next_obs.items()},
            "action": jnp.array(self._actions[idxs]),
            "reward": jnp.array(self._rewards[idxs]),
            "done": jnp.array(self._dones[idxs]),
        }

    def __len__(self) -> int:
        return self._size


class BaseAgent(BotwireAgent, abc.ABC):
    """Base implementation of BotwireAgent with shared replay buffer + checkpointing."""

    def __init__(
        self,
        observation_spec: dict[str, tuple[int, ...]],
        action_spec: tuple[tuple[int, ...], jax.Array, jax.Array],
        replay_capacity: int = 1_000_000,
        batch_size: int = 256,
        seed: int = 0,
    ) -> None:
        self.observation_spec = observation_spec
        action_shape, self.action_low, self.action_high = action_spec
        self.action_shape = action_shape
        self.batch_size = batch_size

        self._replay = ReplayBuffer(replay_capacity, observation_spec, action_shape)
        self._rng = np.random.default_rng(seed)
        self._jax_rng = jax.random.PRNGKey(seed)
        self._update_step = 0

    def observe(
        self,
        observation: Observation,
        action: Action,
        reward: float,
        next_observation: Observation,
        done: bool,
    ) -> None:
        self._replay.add(observation, action, reward, next_observation, done)

    def save(self, path: str) -> None:
        import orbax.checkpoint as ocp

        checkpointer = ocp.PyTreeCheckpointer()
        checkpointer.save(path, self._get_state())

    @classmethod
    def load(cls, path: str) -> "BaseAgent":
        raise NotImplementedError("Use the concrete subclass to load.")

    @abc.abstractmethod
    def _get_state(self) -> Any:
        """Return serializable agent state (params + optimizer state)."""
        ...
