"""BCAgent — Behavioral Cloning agent.

A simple imitation learning agent that trains a policy network via supervised
learning on (observation, action) pairs from a dataset.

Useful as a baseline and for bootstrapping RL from demonstrations.
"""

from __future__ import annotations

from typing import Any, Iterator

import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax

from botwire.agents.networks import GaussianActor
from botwire.common.abc import BotwireAgent
from botwire.common.types import Action, Batch, Metrics, Observation, Params, RNGKey


class BCAgent(BotwireAgent):
    """Behavioral Cloning agent using a Gaussian actor.

    Args:
        observation_spec: Dict mapping obs key → shape.
        action_spec: Tuple (shape, low, high).
        config: ConfigDict with learning_rate, hidden_sizes.
    """

    def __init__(
        self,
        observation_spec: dict[str, tuple[int, ...]],
        action_spec: tuple[tuple[int, ...], jax.Array, jax.Array],
        config: ml_collections.ConfigDict,
    ) -> None:
        self.observation_spec = observation_spec
        action_shape, _, _ = action_spec
        self.action_dim = int(np.prod(action_shape))
        self.config = config

        obs_dim = sum(int(np.prod(s)) for s in observation_spec.values())
        self._obs_dim = obs_dim

        hidden_sizes = tuple(getattr(config, "hidden_sizes", (256, 256)))
        self.actor = GaussianActor(hidden_sizes=hidden_sizes, action_dim=self.action_dim)

        rng = jax.random.PRNGKey(0)
        dummy_obs = jnp.zeros((1, obs_dim))
        self.params = self.actor.init(rng, dummy_obs)

        self.opt = optax.adam(float(config.learning_rate))
        self.opt_state = self.opt.init(self.params)

        self._update_fn = jax.jit(self._bc_update)

    def select_action(self, observation: Observation, rng: RNGKey) -> Action:
        obs_flat = self._flatten_obs(observation)
        action, _ = self.actor.sample(self.params, obs_flat, rng)
        return action[0]

    def observe(self, obs, action, reward, next_obs, done) -> None:
        pass  # BC doesn't use online data

    def update(self) -> Metrics:
        return {}  # BC is trained via train_on_dataset()

    def train_on_dataset(
        self,
        dataset_iter: Iterator[Batch],
        num_steps: int,
    ) -> list[Metrics]:
        """Train BC policy on a dataset iterator.

        Args:
            dataset_iter: Iterator of batches with "observation" and "action".
            num_steps: Number of gradient steps.

        Returns:
            List of per-step metric dicts.
        """
        history = []
        rng = jax.random.PRNGKey(42)

        for step in range(num_steps):
            batch = next(dataset_iter)
            rng, update_rng = jax.random.split(rng)
            self.params, self.opt_state, metrics = self._update_fn(
                self.params, self.opt_state, batch, update_rng
            )
            history.append({k: float(v) for k, v in metrics.items()})

        return history

    def _bc_update(self, params, opt_state, batch, rng):
        obs_flat = self._flatten_obs_batch(batch["observation"])
        target_actions = batch["action"]
        if target_actions.ndim == 3:
            # Flatten sequence dim: (B, T, D) → (B*T, D)
            B, T, D = target_actions.shape
            target_actions = target_actions.reshape(B * T, D)
            obs_flat = jnp.tile(obs_flat[:, None], (1, T, 1)).reshape(B * T, -1)

        def loss_fn(p):
            mean, log_std = self.actor.apply(p, obs_flat)
            std = jnp.exp(log_std)
            log_prob = -0.5 * ((target_actions - mean) / std) ** 2 - jnp.log(std) - 0.5 * jnp.log(2 * jnp.pi)
            return -jnp.mean(log_prob)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = self.opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, {"bc_loss": loss}

    def _flatten_obs(self, obs: Observation) -> jnp.ndarray:
        return jnp.concatenate([jnp.ravel(v) for v in obs.values()])[None]

    def _flatten_obs_batch(self, obs: dict) -> jnp.ndarray:
        return jnp.concatenate([v.reshape(v.shape[0], -1) for v in obs.values()], axis=-1)

    def save(self, path: str) -> None:
        import orbax.checkpoint as ocp

        checkpointer = ocp.PyTreeCheckpointer()
        checkpointer.save(path, {"params": self.params})

    @classmethod
    def load(cls, path: str) -> "BCAgent":
        raise NotImplementedError

    def _get_state(self) -> dict:
        return {"params": self.params}
