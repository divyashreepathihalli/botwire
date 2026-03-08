"""Reinforcement Learning Trainer.

Runs the standard RL training loop:
    collect experience → store in replay buffer → update agent → evaluate

Compatible with any BotwireAgent (SACAgent, D4PGAgent) and BotwireEnv
(BraxEnv, MoMaEnv).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import ml_collections
from absl import logging

from botwire.common.abc import BotwireAgent, BotwireEnv
from botwire.common.types import Metrics, RNGKey
from botwire.envs.utils import evaluate_policy
from botwire.training.checkpointing import CheckpointManager
from botwire.training.logger import MetricsLogger


class RLTrainer:
    """Train a BotwireAgent via reinforcement learning.

    Args:
        agent: Any BotwireAgent instance (SACAgent, D4PGAgent, etc.).
        env: Any BotwireEnv instance (BraxEnv, MoMaEnv, etc.).
        config: Top-level training config (get_rl_train_config()).
    """

    def __init__(
        self,
        agent: BotwireAgent,
        env: BotwireEnv,
        config: ml_collections.ConfigDict,
    ) -> None:
        self.agent = agent
        self.env = env
        self.config = config

        self._ckpt_manager = CheckpointManager(
            directory=config.output_dir,
            max_to_keep=config.keep_checkpoints,
            save_interval_steps=config.save_every,
        )
        self._logger = MetricsLogger(
            log_dir=config.log_dir,
            use_wandb=config.use_wandb,
            wandb_project=config.wandb_project,
            use_tensorboard=config.use_tensorboard,
        )

    def train(self, rng: RNGKey | None = None) -> None:
        """Run the full RL training loop.

        Args:
            rng: JAX random key. Defaults to PRNGKey(config.seed).
        """
        config = self.config
        rng = rng if rng is not None else jax.random.PRNGKey(config.seed)

        # Initial reset
        rng, reset_rng = jax.random.split(rng)
        state, obs = self.env.reset(reset_rng)
        episode_return = 0.0
        episode_length = 0
        episode_count = 0

        logging.info("Starting RL training: %d steps", config.num_steps)

        for step in range(1, config.num_steps + 1):
            rng, act_rng = jax.random.split(rng)
            action = self.agent.select_action(obs, act_rng)

            next_state, next_obs, reward, done, info = self.env.step(state, action)
            episode_return += float(reward)
            episode_length += 1

            self.agent.observe(obs, action, float(reward), next_obs, bool(done))

            # Agent update
            update_metrics = self.agent.update()

            if bool(done):
                episode_count += 1
                if step % config.log_every == 0:
                    ep_metrics: Metrics = {
                        "episode_return": episode_return,
                        "episode_length": episode_length,
                        "num_episodes": episode_count,
                    }
                    ep_metrics.update(update_metrics)
                    self._logger.log(ep_metrics, step=step, prefix="train")

                # Reset
                rng, reset_rng = jax.random.split(rng)
                state, obs = self.env.reset(reset_rng)
                episode_return = 0.0
                episode_length = 0
            else:
                state, obs = next_state, next_obs

            # Periodic evaluation
            if step % config.eval_every == 0:
                rng, eval_rng = jax.random.split(rng)
                eval_metrics = self._evaluate(eval_rng)
                self._logger.log(eval_metrics, step=step, prefix="eval")

            # Checkpoint
            if step % config.save_every == 0:
                self.agent.save(f"{config.output_dir}/step_{step:09d}")
                logging.info("Saved agent checkpoint at step %d", step)

        self._ckpt_manager.close()
        self._logger.close()
        logging.info("RL training complete.")

    def _evaluate(self, rng: RNGKey) -> Metrics:
        """Run evaluation episodes and return metrics."""
        num_eval = int(self.config.agent.get("eval_episodes", 10))
        returns = []
        lengths = []

        for _ in range(num_eval):
            rng, reset_rng, act_rng = jax.random.split(rng, 3)
            state, obs = self.env.reset(reset_rng)
            ep_return = 0.0
            ep_len = 0
            for _ in range(10_000):
                rng, step_rng = jax.random.split(rng)
                action = self.agent.select_action(obs, step_rng)
                state, obs, reward, done, _ = self.env.step(state, action)
                ep_return += float(reward)
                ep_len += 1
                if bool(done):
                    break
            returns.append(ep_return)
            lengths.append(ep_len)

        import numpy as np

        return {
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            "mean_episode_length": float(np.mean(lengths)),
        }
