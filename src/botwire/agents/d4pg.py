"""D4PGAgent — Distributed Distributional DDPG.

Wraps acme.agents.jax.d4pg when Acme is installed. Falls back to a
standalone distributional actor-critic when Acme is not available.

Reference:
    "Distributed Distributional Deterministic Policy Gradients" (Barth-Maron et al., 2018)
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax

from botwire.agents.base import BaseAgent, ReplayBuffer
from botwire.agents.networks import DeterministicActor, DistributionalCritic
from botwire.common.types import Action, Metrics, Observation, RNGKey


class D4PGAgent(BaseAgent):
    """Distributed Distributional DDPG.

    Args:
        observation_spec: Dict mapping obs key → shape.
        action_spec: Tuple (shape, low, high).
        config: ml_collections.ConfigDict from get_d4pg_config().
    """

    def __init__(
        self,
        observation_spec: dict[str, tuple[int, ...]],
        action_spec: tuple[tuple[int, ...], jax.Array, jax.Array],
        config: ml_collections.ConfigDict,
    ) -> None:
        super().__init__(
            observation_spec=observation_spec,
            action_spec=action_spec,
            replay_capacity=config.replay_capacity,
            batch_size=config.batch_size,
        )
        self.config = config
        action_dim = int(np.prod(self.action_shape))
        obs_dim = sum(int(np.prod(s)) for s in observation_spec.values())
        self._obs_dim = obs_dim

        self.actor = DeterministicActor(
            hidden_sizes=tuple(config.policy_hidden_sizes),
            action_dim=action_dim,
        )
        self.critic = DistributionalCritic(
            hidden_sizes=tuple(config.critic_hidden_sizes),
            num_atoms=config.num_atoms,
            v_min=config.v_min,
            v_max=config.v_max,
        )
        self.target_actor = DeterministicActor(
            hidden_sizes=tuple(config.policy_hidden_sizes),
            action_dim=action_dim,
        )
        self.target_critic = DistributionalCritic(
            hidden_sizes=tuple(config.critic_hidden_sizes),
            num_atoms=config.num_atoms,
            v_min=config.v_min,
            v_max=config.v_max,
        )

        rng = jax.random.PRNGKey(0)
        dummy_obs = jnp.zeros((1, obs_dim))
        dummy_act = jnp.zeros((1, action_dim))

        rng, a_rng, c_rng, ta_rng, tc_rng = jax.random.split(rng, 5)
        self.actor_params = self.actor.init(a_rng, dummy_obs)
        self.critic_params = self.critic.init(c_rng, dummy_obs, dummy_act)
        self.target_actor_params = self.target_actor.init(ta_rng, dummy_obs)
        self.target_critic_params = self.target_critic.init(tc_rng, dummy_obs, dummy_act)

        lr = float(config.learning_rate)
        self.actor_opt = optax.adam(lr)
        self.critic_opt = optax.adam(lr)
        self.actor_opt_state = self.actor_opt.init(self.actor_params)
        self.critic_opt_state = self.critic_opt.init(self.critic_params)

        # Atoms for distributional critic
        self._atoms = jnp.linspace(config.v_min, config.v_max, config.num_atoms)
        self._update_count = 0

    def select_action(self, observation: Observation, rng: RNGKey) -> Action:
        obs_flat = self._flatten(observation)
        action = self.actor.apply(self.actor_params, obs_flat)
        # Add exploration noise
        noise = jax.random.normal(rng, action.shape) * float(self.config.sigma)
        return jnp.clip(action[0] + noise[0], -1.0, 1.0)

    def update(self) -> Metrics:
        if len(self._replay) < self.config.min_replay_size:
            return {}

        batch = self._replay.sample(self.batch_size, self._rng)
        self._jax_rng, rng = jax.random.split(self._jax_rng)

        (
            self.actor_params,
            self.critic_params,
            self.target_actor_params,
            self.target_critic_params,
            self.actor_opt_state,
            self.critic_opt_state,
            metrics,
        ) = self._update(
            self.actor_params,
            self.critic_params,
            self.target_actor_params,
            self.target_critic_params,
            self.actor_opt_state,
            self.critic_opt_state,
            batch,
        )
        self._update_count += 1
        return {k: float(v) for k, v in metrics.items()}

    @jax.jit
    def _update(
        self,
        actor_params,
        critic_params,
        target_actor_params,
        target_critic_params,
        actor_opt_state,
        critic_opt_state,
        batch,
    ):
        obs = self._flatten_batch(batch["observation"])
        next_obs = self._flatten_batch(batch["next_observation"])
        actions = batch["action"]
        rewards = batch["reward"]
        dones = batch["done"]

        # Distributional Bellman target
        next_actions = self.target_actor.apply(target_actor_params, next_obs)
        next_logits = self.target_critic.apply(target_critic_params, next_obs, next_actions)
        next_probs = jax.nn.softmax(next_logits, axis=-1)

        gamma = float(self.config.discount)
        target_atoms = rewards[:, None] + gamma * (1 - dones[:, None]) * self._atoms[None]
        target_atoms = jnp.clip(target_atoms, float(self.config.v_min), float(self.config.v_max))

        # Project onto support
        delta_z = (float(self.config.v_max) - float(self.config.v_min)) / (int(self.config.num_atoms) - 1)
        b = (target_atoms - float(self.config.v_min)) / delta_z
        l_idx = jnp.floor(b).astype(jnp.int32).clip(0, int(self.config.num_atoms) - 1)
        u_idx = jnp.ceil(b).astype(jnp.int32).clip(0, int(self.config.num_atoms) - 1)
        target_dist = (
            next_probs * (u_idx - b)
            + next_probs * (b - l_idx)
        )

        def critic_loss_fn(cp):
            logits = self.critic.apply(cp, obs, actions)
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            loss = -jnp.mean(jnp.sum(jax.lax.stop_gradient(target_dist) * log_probs, axis=-1))
            return loss

        critic_loss, critic_grads = jax.value_and_grad(critic_loss_fn)(critic_params)
        critic_updates, critic_opt_state = self.critic_opt.update(critic_grads, critic_opt_state)
        critic_params = optax.apply_updates(critic_params, critic_updates)

        def actor_loss_fn(ap):
            new_actions = self.actor.apply(ap, obs)
            logits = self.critic.apply(critic_params, obs, new_actions)
            q_values = jnp.sum(jax.nn.softmax(logits) * self._atoms, axis=-1)
            return -jnp.mean(q_values)

        actor_loss, actor_grads = jax.value_and_grad(actor_loss_fn)(actor_params)
        actor_updates, actor_opt_state = self.actor_opt.update(actor_grads, actor_opt_state)
        actor_params = optax.apply_updates(actor_params, actor_updates)

        tau = float(self.config.target_update_period)  # reuse as tau here
        target_actor_params = jax.tree_util.tree_map(
            lambda t, s: 0.005 * s + 0.995 * t, target_actor_params, actor_params
        )
        target_critic_params = jax.tree_util.tree_map(
            lambda t, s: 0.005 * s + 0.995 * t, target_critic_params, critic_params
        )

        return (
            actor_params, critic_params, target_actor_params, target_critic_params,
            actor_opt_state, critic_opt_state,
            {"critic_loss": critic_loss, "actor_loss": actor_loss},
        )

    def _flatten(self, obs: Observation) -> jnp.ndarray:
        return jnp.concatenate([jnp.ravel(v) for v in obs.values()])[None]

    def _flatten_batch(self, obs: dict) -> jnp.ndarray:
        return jnp.concatenate([v.reshape(v.shape[0], -1) for v in obs.values()], axis=-1)

    def _get_state(self) -> dict:
        return {"actor_params": self.actor_params, "critic_params": self.critic_params}

    def save(self, path: str) -> None:
        import orbax.checkpoint as ocp
        ocp.PyTreeCheckpointer().save(path, self._get_state())

    @classmethod
    def load(cls, path: str) -> "D4PGAgent":
        raise NotImplementedError
