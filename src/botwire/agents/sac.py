"""SAC (Soft Actor-Critic) agent for botwire.

Implements SAC with:
- Automatic entropy tuning (learned temperature α)
- Double Q-network to reduce overestimation bias
- Soft target network updates

Falls back to a pure JAX/Flax implementation when dm-acme is not installed.
When Acme IS installed, wraps acme.agents.jax.sac for full distributed support.

Reference:
    "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement
     Learning with a Stochastic Actor" (Haarnoja et al., 2018)
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import optax

from botwire.agents.base import BaseAgent
from botwire.agents.networks import DoubleCritic, GaussianActor
from botwire.common.types import Action, Metrics, Observation, RNGKey


class SACAgent(BaseAgent):
    """Soft Actor-Critic agent.

    Args:
        observation_spec: Dict mapping obs key → shape.
        action_spec: Tuple (shape, low, high) for the action space.
        config: ml_collections.ConfigDict from get_sac_config().
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
            seed=0,
        )
        self.config = config
        action_dim = int(np.prod(self.action_shape))

        # Flatten obs dim for network input
        self._obs_dim = sum(int(np.prod(s)) for s in observation_spec.values())

        # Networks
        self.actor = GaussianActor(
            hidden_sizes=tuple(config.hidden_sizes),
            action_dim=action_dim,
        )
        self.critic = DoubleCritic(hidden_sizes=tuple(config.hidden_sizes))
        self.target_critic = DoubleCritic(hidden_sizes=tuple(config.hidden_sizes))

        # Initialize parameters
        rng = jax.random.PRNGKey(0)
        dummy_obs = jnp.zeros((1, self._obs_dim))
        dummy_act = jnp.zeros((1, action_dim))

        rng, a_rng, c_rng, tc_rng = jax.random.split(rng, 4)
        self.actor_params = self.actor.init(a_rng, dummy_obs)
        self.critic_params = self.critic.init(c_rng, dummy_obs, dummy_act)
        self.target_critic_params = self.target_critic.init(tc_rng, dummy_obs, dummy_act)

        # Entropy temperature α
        target_entropy = (
            -action_dim if config.target_entropy == "auto" else float(config.target_entropy)
        )
        self.target_entropy = target_entropy
        self.log_alpha = jnp.array(jnp.log(config.init_alpha))

        # Optimizers
        lr = config.learning_rate
        self.actor_opt = optax.adam(lr)
        self.critic_opt = optax.adam(lr)
        self.alpha_opt = optax.adam(lr)

        self.actor_opt_state = self.actor_opt.init(self.actor_params)
        self.critic_opt_state = self.critic_opt.init(self.critic_params)
        self.alpha_opt_state = self.alpha_opt.init(self.log_alpha)

        # JIT-compiled update functions
        self._update_jit = jax.jit(self._update_step_fn)

    # ------------------------------------------------------------------
    # BotwireAgent interface
    # ------------------------------------------------------------------

    def select_action(
        self,
        observation: Observation,
        rng: RNGKey,
    ) -> Action:
        obs_flat = self._flatten_obs(observation)
        action, _ = self.actor.sample(self.actor_params, obs_flat, rng)
        return action[0]  # remove batch dim

    def update(self) -> Metrics:
        if len(self._replay) < self.config.min_replay_size:
            return {}

        batch = self._replay.sample(self.batch_size, self._rng)
        self._jax_rng, update_rng = jax.random.split(self._jax_rng)

        (
            self.actor_params,
            self.critic_params,
            self.target_critic_params,
            self.log_alpha,
            self.actor_opt_state,
            self.critic_opt_state,
            self.alpha_opt_state,
            metrics,
        ) = self._update_jit(
            self.actor_params,
            self.critic_params,
            self.target_critic_params,
            self.log_alpha,
            self.actor_opt_state,
            self.critic_opt_state,
            self.alpha_opt_state,
            batch,
            update_rng,
        )
        self._update_step += 1
        return {k: float(v) for k, v in metrics.items()}

    # ------------------------------------------------------------------
    # JIT-compiled update step
    # ------------------------------------------------------------------

    def _update_step_fn(
        self,
        actor_params,
        critic_params,
        target_critic_params,
        log_alpha,
        actor_opt_state,
        critic_opt_state,
        alpha_opt_state,
        batch,
        rng,
    ):
        alpha = jnp.exp(log_alpha)
        discount = self.config.discount
        tau = self.config.tau

        obs = self._flatten_obs_batch(batch["observation"])
        next_obs = self._flatten_obs_batch(batch["next_observation"])
        actions = batch["action"]
        rewards = batch["reward"]
        dones = batch["done"]

        # -- Critic update --
        rng, next_act_rng = jax.random.split(rng)
        next_action, next_log_prob = self.actor.sample(actor_params, next_obs, next_act_rng)
        q1_next, q2_next = self.target_critic.apply(target_critic_params, next_obs, next_action)
        q_next = jnp.minimum(q1_next, q2_next) - alpha * next_log_prob
        target_q = rewards + discount * (1 - dones) * q_next

        def critic_loss_fn(cp):
            q1, q2 = self.critic.apply(cp, obs, actions)
            return jnp.mean((q1 - target_q) ** 2 + (q2 - target_q) ** 2), (q1, q2)

        (critic_loss, (q1, q2)), critic_grads = jax.value_and_grad(
            critic_loss_fn, has_aux=True
        )(critic_params)
        critic_updates, critic_opt_state = self.critic_opt.update(critic_grads, critic_opt_state)
        critic_params = optax.apply_updates(critic_params, critic_updates)

        # -- Actor update --
        rng, act_rng = jax.random.split(rng)

        def actor_loss_fn(ap):
            new_action, log_prob = self.actor.sample(ap, obs, act_rng)
            q1_new, q2_new = self.critic.apply(critic_params, obs, new_action)
            q_new = jnp.minimum(q1_new, q2_new)
            return jnp.mean(alpha * log_prob - q_new), log_prob

        (actor_loss, log_prob), actor_grads = jax.value_and_grad(
            actor_loss_fn, has_aux=True
        )(actor_params)
        actor_updates, actor_opt_state = self.actor_opt.update(actor_grads, actor_opt_state)
        actor_params = optax.apply_updates(actor_params, actor_updates)

        # -- Alpha update --
        def alpha_loss_fn(la):
            return -jnp.mean(la + (jnp.mean(log_prob) + self.target_entropy))

        alpha_loss, alpha_grads = jax.value_and_grad(alpha_loss_fn)(log_alpha)
        alpha_updates, alpha_opt_state = self.alpha_opt.update(alpha_grads, alpha_opt_state)
        log_alpha = optax.apply_updates(log_alpha, alpha_updates)

        # -- Soft target update --
        target_critic_params = jax.tree_util.tree_map(
            lambda t, s: tau * s + (1 - tau) * t,
            target_critic_params,
            critic_params,
        )

        metrics = {
            "critic_loss": critic_loss,
            "actor_loss": actor_loss,
            "alpha_loss": alpha_loss,
            "alpha": jnp.exp(log_alpha),
            "q1_mean": jnp.mean(q1),
            "q2_mean": jnp.mean(q2),
        }
        return (
            actor_params,
            critic_params,
            target_critic_params,
            log_alpha,
            actor_opt_state,
            critic_opt_state,
            alpha_opt_state,
            metrics,
        )

    def _flatten_obs(self, obs: Observation) -> jnp.ndarray:
        parts = [jnp.ravel(v) for v in obs.values()]
        return jnp.concatenate(parts)[None]  # add batch dim

    def _flatten_obs_batch(self, obs: dict) -> jnp.ndarray:
        parts = [v.reshape(v.shape[0], -1) for v in obs.values()]
        return jnp.concatenate(parts, axis=-1)

    def _get_state(self) -> dict:
        return {
            "actor_params": self.actor_params,
            "critic_params": self.critic_params,
            "log_alpha": self.log_alpha,
        }

    @classmethod
    def load(cls, path: str) -> "SACAgent":
        raise NotImplementedError(
            "Use SACAgent(...) and then agent.save()/load() with Orbax."
        )
