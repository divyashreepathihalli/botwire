"""Abstract base classes for all botwire modules.

Every concrete implementation (BraxEnv, ACTPolicy, SACAgent, …) inherits
from one of these ABCs, ensuring a uniform interface across the framework.
"""

import abc
from typing import Any, Iterator

import jax
import jax.numpy as jnp

from botwire.common.types import Action, Batch, Metrics, Observation, Params, RNGKey


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


class BotwirePolicy(abc.ABC):
    """Abstract base for all Flax-based imitation-learning policies.

    A policy maps observations to actions and exposes a differentiable
    loss function for training with Optax.

    Implementations: ACTPolicy, DiffusionPolicy, VLAPolicy.
    """

    @abc.abstractmethod
    def init(
        self,
        rng: RNGKey,
        observation_spec: dict[str, tuple[int, ...]],
    ) -> Params:
        """Initialize and return a pytree of model parameters.

        Args:
            rng: JAX random key for weight initialization.
            observation_spec: Maps sensor name to array shape (without batch dim).

        Returns:
            Nested pytree of jax.Arrays (Flax variables dict).
        """
        ...

    @abc.abstractmethod
    def select_action(
        self,
        params: Params,
        observation: Observation,
        rng: RNGKey,
        *,
        deterministic: bool = False,
    ) -> Action:
        """Sample a single action from the policy.

        Must be JIT-compatible.

        Args:
            params: Model parameters (from ``init``).
            observation: Dict of sensor arrays (no batch dim).
            rng: JAX random key used for stochastic policies.
            deterministic: If True, return the mode of the action distribution.

        Returns:
            Action array.
        """
        ...

    @abc.abstractmethod
    def loss(
        self,
        params: Params,
        batch: Batch,
        rng: RNGKey,
    ) -> tuple[jax.Array, Metrics]:
        """Compute scalar training loss and auxiliary metrics.

        Args:
            params: Model parameters.
            batch: Dict with keys "observation", "action", etc.
            rng: JAX random key.

        Returns:
            Tuple of (scalar loss, metrics dict).
        """
        ...


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class BotwireEnv(abc.ABC):
    """Abstract base for all simulation environments.

    Follows a functional (stateless) interface compatible with JAX's
    JIT compiler: ``reset`` and ``step`` return a new state rather than
    mutating an object.

    Implementations: BraxEnv, MoMaEnv.
    """

    @abc.abstractmethod
    def reset(self, rng: RNGKey) -> tuple[Any, Observation]:
        """Reset the environment and return initial state + observation.

        Args:
            rng: JAX random key used for stochastic resets.

        Returns:
            Tuple of (env_state pytree, observation dict).
        """
        ...

    @abc.abstractmethod
    def step(
        self,
        state: Any,
        action: Action,
    ) -> tuple[Any, Observation, jax.Array, jax.Array, dict]:
        """Advance the environment by one step.

        Args:
            state: Current environment state pytree.
            action: Action array.

        Returns:
            Tuple of (next_state, observation, reward, done, info).
            reward and done are scalar jax.Arrays.
        """
        ...

    @property
    @abc.abstractmethod
    def observation_spec(self) -> dict[str, tuple[int, ...]]:
        """Maps sensor name to array shape (no batch dim)."""
        ...

    @property
    @abc.abstractmethod
    def action_spec(self) -> tuple[tuple[int, ...], jax.Array, jax.Array]:
        """Returns (shape, low, high) for the action space."""
        ...

    @property
    def action_dim(self) -> int:
        """Convenience: total number of action dimensions."""
        shape, _, _ = self.action_spec
        return int(jnp.prod(jnp.array(shape)))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class BotwireDataset(abc.ABC):
    """Abstract base for all robot datasets.

    Datasets are backed by RLDS / tensorflow-datasets and expose a
    clean Python iterator interface for JAX training loops.

    Implementations: RLDSDataset (wrapping OXE or local RLDS data).
    """

    @abc.abstractmethod
    def as_iterator(
        self,
        batch_size: int,
        sequence_length: int,
        *,
        shuffle: bool = True,
        seed: int = 0,
    ) -> Iterator[Batch]:
        """Return an infinite iterator of batched trajectory segments.

        Each batch has shape (batch_size, sequence_length, ...) for all
        time-indexed fields.

        Args:
            batch_size: Number of trajectories per batch.
            sequence_length: Number of timesteps per trajectory segment.
            shuffle: Whether to shuffle episodes.
            seed: Random seed for shuffling.

        Yields:
            Batch dicts with JAX arrays.
        """
        ...

    @abc.abstractmethod
    def episode_count(self) -> int:
        """Total number of episodes in the dataset."""
        ...

    @abc.abstractmethod
    def step_count(self) -> int:
        """Total number of environment steps across all episodes."""
        ...

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"episodes={self.episode_count()}, "
            f"steps={self.step_count()})"
        )


# ---------------------------------------------------------------------------
# Agent (RL)
# ---------------------------------------------------------------------------


class BotwireAgent(abc.ABC):
    """Abstract base for RL agents (wrapping Acme JAX agents).

    Implementations: SACAgent, D4PGAgent, BCAgent.
    """

    @abc.abstractmethod
    def select_action(
        self,
        observation: Observation,
        rng: RNGKey,
    ) -> Action:
        """Select an action for the given observation (online inference).

        Args:
            observation: Current environment observation.
            rng: JAX random key.

        Returns:
            Action array.
        """
        ...

    @abc.abstractmethod
    def observe(
        self,
        observation: Observation,
        action: Action,
        reward: float,
        next_observation: Observation,
        done: bool,
    ) -> None:
        """Store a transition in the replay buffer."""
        ...

    @abc.abstractmethod
    def update(self) -> Metrics:
        """Perform one gradient update step.

        Returns:
            Dict of metric names to scalar values.
        """
        ...

    @abc.abstractmethod
    def save(self, path: str) -> None:
        """Serialize agent state (parameters + optimizer) to disk."""
        ...

    @classmethod
    @abc.abstractmethod
    def load(cls, path: str) -> "BotwireAgent":
        """Restore an agent from disk."""
        ...


# ---------------------------------------------------------------------------
# Controller (dm_robotics)
# ---------------------------------------------------------------------------


class BotwireController(abc.ABC):
    """Abstract base for low-level robot controllers.

    Wraps dm_robotics QP controllers or other control primitives.
    Controllers operate on numpy arrays (not JAX) because they interface
    with physics engines or real hardware at control frequency.

    Implementations: CartesianController, JointController.
    """

    @abc.abstractmethod
    def compute_action(
        self,
        observation: dict[str, Any],
        target: Any,
    ) -> Any:
        """Compute joint torques / velocities to achieve the target.

        Args:
            observation: Current robot state (joints, velocities, etc.).
            target: Desired end-effector pose or joint configuration.

        Returns:
            Control output (torques or velocities).
        """
        ...

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset internal controller state."""
        ...


# ---------------------------------------------------------------------------
# Hardware Interface
# ---------------------------------------------------------------------------


class RobotInterface(abc.ABC):
    """Abstract interface for physical (or simulated) robot hardware.

    Implementations: ROSBridge (ROS2), GeminiRoboticsAdapter.
    """

    @abc.abstractmethod
    def get_observation(self) -> Observation:
        """Read current sensor state from the robot."""
        ...

    @abc.abstractmethod
    def send_action(self, action: Action) -> None:
        """Send an action command to the robot."""
        ...

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset the robot to its home configuration."""
        ...

    @abc.abstractmethod
    def close(self) -> None:
        """Release hardware resources."""
        ...

    def __enter__(self) -> "RobotInterface":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
