"""Shared Flax network architectures for RL agents.

These networks serve as the actor and critic for SAC, D4PG, and BC agents.
All are compatible with Acme's JAX agent interface.
"""

from __future__ import annotations

from typing import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp


class MLP(nn.Module):
    """Multi-layer perceptron with configurable depth and activation."""

    hidden_sizes: Sequence[int] = (256, 256)
    output_size: int = 1
    activation: str = "relu"
    activate_final: bool = False

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        act_fn = {"relu": nn.relu, "tanh": jnp.tanh, "elu": nn.elu}.get(
            self.activation, nn.relu
        )
        for size in self.hidden_sizes:
            x = nn.Dense(size)(x)
            x = act_fn(x)
        x = nn.Dense(self.output_size)(x)
        if self.activate_final:
            x = act_fn(x)
        return x


class GaussianActor(nn.Module):
    """Gaussian policy network for SAC.

    Outputs mean and log_std of a diagonal Gaussian distribution.

    Args:
        hidden_sizes: MLP hidden layer sizes.
        action_dim: Dimensionality of the action space.
        log_std_min: Minimum log standard deviation (for stability).
        log_std_max: Maximum log standard deviation.
    """

    hidden_sizes: Sequence[int] = (256, 256)
    action_dim: int = 1
    log_std_min: float = -5.0
    log_std_max: float = 2.0

    @nn.compact
    def __call__(
        self,
        observation: jax.Array,
        train: bool = True,
    ) -> tuple[jax.Array, jax.Array]:
        x = observation
        for size in self.hidden_sizes:
            x = nn.Dense(size)(x)
            x = nn.relu(x)

        mean = nn.Dense(self.action_dim)(x)
        log_std = nn.Dense(self.action_dim)(x)
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(
        self,
        params: dict,
        observation: jax.Array,
        rng: jax.Array,
        *,
        deterministic: bool = False,
    ) -> tuple[jax.Array, jax.Array]:
        """Sample action and compute log probability.

        Returns:
            Tuple of (action, log_prob). Action is squashed via tanh.
        """
        mean, log_std = self.apply(params, observation)
        std = jnp.exp(log_std)

        if deterministic:
            action = mean
            log_prob = jnp.zeros(())
        else:
            noise = jax.random.normal(rng, mean.shape)
            x = mean + std * noise
            action = jnp.tanh(x)
            # Log prob with tanh squashing correction
            log_prob = (
                -0.5 * ((noise ** 2) + 2 * jnp.log(std) + jnp.log(2 * jnp.pi))
                - jnp.log(1 - action ** 2 + 1e-6)
            ).sum(axis=-1)

        return action, log_prob


class DoubleCritic(nn.Module):
    """Twin Q-network for SAC (reduces overestimation bias).

    Returns Q1 and Q2 values for a given (observation, action) pair.
    """

    hidden_sizes: Sequence[int] = (256, 256)

    @nn.compact
    def __call__(
        self,
        observation: jax.Array,
        action: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        x = jnp.concatenate([observation, action], axis=-1)
        q1 = MLP(self.hidden_sizes, output_size=1)(x).squeeze(-1)
        q2 = MLP(self.hidden_sizes, output_size=1)(x).squeeze(-1)
        return q1, q2


class DistributionalCritic(nn.Module):
    """Distributional critic for D4PG (outputs a categorical distribution over returns).

    Args:
        hidden_sizes: MLP hidden layer sizes.
        num_atoms: Number of atoms in the return distribution.
        v_min: Minimum return value.
        v_max: Maximum return value.
    """

    hidden_sizes: Sequence[int] = (256, 256)
    num_atoms: int = 51
    v_min: float = -150.0
    v_max: float = 150.0

    @nn.compact
    def __call__(
        self,
        observation: jax.Array,
        action: jax.Array,
    ) -> jax.Array:
        x = jnp.concatenate([observation, action], axis=-1)
        logits = MLP(self.hidden_sizes, output_size=self.num_atoms)(x)
        return logits  # (B, num_atoms) — apply softmax for probabilities

    @property
    def atoms(self) -> jax.Array:
        """Return value support points."""
        return jnp.linspace(self.v_min, self.v_max, self.num_atoms)

    def expected_value(self, logits: jax.Array) -> jax.Array:
        """Compute expected Q-value from logits."""
        probs = jax.nn.softmax(logits, axis=-1)
        return jnp.sum(probs * self.atoms, axis=-1)


class DeterministicActor(nn.Module):
    """Deterministic policy network for D4PG / DDPG."""

    hidden_sizes: Sequence[int] = (256, 256)
    action_dim: int = 1

    @nn.compact
    def __call__(self, observation: jax.Array) -> jax.Array:
        x = observation
        for size in self.hidden_sizes:
            x = nn.Dense(size)(x)
            x = nn.relu(x)
        return jnp.tanh(nn.Dense(self.action_dim)(x))
