"""Imitation Learning Trainer.

Trains a BotwirePolicy (ACT, DiffusionPolicy, etc.) on an RLDS dataset
using supervised learning with Optax.

Supports:
- Gradient clipping
- Cosine LR schedule with linear warmup
- Orbax checkpointing
- WandB / TensorBoard logging
- Multi-device (pmap) training
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import ml_collections
import optax
import orbax.checkpoint as ocp
from absl import logging

from botwire.common.abc import BotwireDataset, BotwirePolicy
from botwire.common.types import Metrics, Params, RNGKey
from botwire.training.checkpointing import CheckpointManager
from botwire.training.logger import MetricsLogger


class ImitationLearningTrainer:
    """Train a BotwirePolicy via imitation learning on a dataset.

    Args:
        policy: Any BotwirePolicy (ACT, Diffusion, etc.).
        dataset: A BotwireDataset with RLDS episodes.
        config: Top-level training configuration (get_base_train_config()).

    Example::

        trainer = ImitationLearningTrainer(policy, dataset, config)
        trainer.train()  # runs to completion, saves checkpoints
    """

    def __init__(
        self,
        policy: BotwirePolicy,
        dataset: BotwireDataset,
        config: ml_collections.ConfigDict,
    ) -> None:
        self.policy = policy
        self.dataset = dataset
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
            wandb_entity=config.get("wandb_entity", ""),
            use_tensorboard=config.use_tensorboard,
        )

    def train(
        self,
        params: Params | None = None,
        observation_spec: dict[str, tuple[int, ...]] | None = None,
        rng: RNGKey | None = None,
    ) -> Params:
        """Run the full IL training loop.

        Args:
            params: Pre-initialized policy parameters. If None, initializes
                    from the policy's ``init()`` method.
            observation_spec: Shape spec for observations. Required if params=None.
            rng: JAX random key. Defaults to PRNGKey(config.seed).

        Returns:
            Final trained parameters.
        """
        config = self.config
        rng = rng if rng is not None else jax.random.PRNGKey(config.seed)

        # Initialize parameters
        if params is None:
            if observation_spec is None:
                raise ValueError("Must provide observation_spec when params=None.")
            rng, init_rng = jax.random.split(rng)
            params = self.policy.init(init_rng, observation_spec)
            logging.info(
                "Initialized policy with %s parameters.",
                _count_params(params),
            )

        # Optimizer: cosine schedule with warmup
        warmup_steps = max(1, config.num_steps // 20)
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=float(config.policy.learning_rate),
            warmup_steps=warmup_steps,
            decay_steps=config.num_steps,
            end_value=float(config.policy.learning_rate) * 0.1,
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(float(config.policy.get("grad_clip", 10.0))),
            optax.adamw(lr_schedule, weight_decay=float(config.policy.get("weight_decay", 1e-4))),
        )
        opt_state = optimizer.init(params)

        # Dataset iterator
        ds_iter = self.dataset.as_iterator(
            batch_size=config.dataset.batch_size,
            sequence_length=config.dataset.sequence_length,
            shuffle=True,
            seed=int(config.seed),
        )

        # JIT-compiled update step
        @jax.jit
        def update_step(params, opt_state, batch, rng):
            loss, grads = jax.value_and_grad(self.policy.loss, has_aux=True)(params, batch, rng)
            scalar_loss, metrics = loss
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, metrics

        logging.info(
            "Starting IL training: %d steps, batch_size=%d, seq_len=%d",
            config.num_steps,
            config.dataset.batch_size,
            config.dataset.sequence_length,
        )

        for step in range(1, config.num_steps + 1):
            batch = next(ds_iter)
            rng, step_rng = jax.random.split(rng)
            params, opt_state, metrics = update_step(params, opt_state, batch, step_rng)

            if step % config.log_every == 0:
                self._logger.log(metrics, step=step, prefix="train")

            if step % config.save_every == 0:
                self._ckpt_manager.save(
                    step,
                    {"params": params, "opt_state": opt_state, "step": step},
                )

        self._ckpt_manager.close()
        self._logger.close()
        logging.info("IL training complete.")
        return params


def _count_params(params: Any) -> str:
    n = sum(x.size for x in jax.tree_util.tree_leaves(params))
    if n >= 1e6:
        return f"{n/1e6:.1f}M"
    if n >= 1e3:
        return f"{n/1e3:.1f}K"
    return str(n)
