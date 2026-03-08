"""Base Flax module for all botwire policies."""

from __future__ import annotations

import abc
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp

from botwire.common.types import Action, Batch, Metrics, Observation, Params, RNGKey


class FlaxPolicy(nn.Module, abc.ABC):
    """Base class combining Flax Module and BotwirePolicy semantics.

    Concrete policies override ``__call__`` for forward pass and
    optionally ``select_action`` for inference-time behavior.
    """

    @abc.abstractmethod
    def __call__(
        self,
        observation: Observation,
        train: bool = True,
    ) -> Any:
        """Forward pass.  Returns raw model output (not necessarily an action)."""
        ...

    def select_action(
        self,
        params: Params,
        observation: Observation,
        rng: RNGKey,
        *,
        deterministic: bool = False,
    ) -> Action:
        """Default: call the model and take the first element as the action."""
        output = self.apply(params, observation, train=False)
        if isinstance(output, tuple):
            return output[0]
        return output

    def loss(
        self,
        params: Params,
        batch: Batch,
        rng: RNGKey,
    ) -> tuple[jax.Array, Metrics]:
        """Default MSE loss between predicted and ground-truth actions."""
        obs = batch["observation"]
        target_action = batch["action"]
        pred = self.apply(params, obs, train=True, rngs={"dropout": rng})
        if isinstance(pred, tuple):
            pred = pred[0]
        loss = jnp.mean((pred - target_action) ** 2)
        return loss, {"loss": loss}
