"""VLAPolicy: BotwirePolicy wrapper around the Gemini Robotics SDK.

Bridges Gemini Robotics-ER VLA inference with the botwire training and
evaluation infrastructure.

Unlike ACTPolicy and DiffusionPolicy (which are Flax modules trained from
scratch), VLAPolicy wraps a pre-trained foundation model — either via the
Gemini API or a local Safari SDK model server.

Fine-tuning is handled via the Gemini Robotics SDK flywheel (see the
botwire-upload CLI which uploads datasets to the Gemini Robotics platform).
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import ml_collections
import numpy as np

from botwire.common.abc import BotwirePolicy
from botwire.common.types import Action, Batch, Metrics, Observation, Params, RNGKey
from botwire.policies.vla.gemini_bridge import GeminiRoboticsAdapter


class VLAPolicy(BotwirePolicy):
    """Vision-Language-Action policy backed by Gemini Robotics-ER.

    This policy uses a pre-trained VLM for zero-shot or fine-tuned robot
    control. It accepts natural language task descriptions and visual
    observations, and outputs robot actions.

    Args:
        config: ml_collections.ConfigDict from get_vla_config().
        action_dim: Dimensionality of the robot's action space.
        task_description: Default task description (can be overridden per call).

    Example::

        from botwire.policies.vla import VLAPolicy
        from botwire.configs import get_vla_config

        config = get_vla_config()
        policy = VLAPolicy(config, action_dim=7, task_description="Pick up the red cube")

        # No training needed — uses pre-trained Gemini Robotics-ER
        params = {}   # VLA has no local Flax params
        action = policy.select_action(params, obs, rng)
    """

    def __init__(
        self,
        config: ml_collections.ConfigDict,
        action_dim: int = 7,
        task_description: str = "complete the task",
    ) -> None:
        self.config = config
        self.action_dim = action_dim
        self.task_description = task_description
        self._adapter = GeminiRoboticsAdapter(
            model_id=config.model_id,
            action_dim=action_dim,
            thinking_budget=config.thinking_budget,
            temperature=config.temperature,
        )

    # ------------------------------------------------------------------
    # BotwirePolicy interface
    # ------------------------------------------------------------------

    def init(
        self,
        rng: RNGKey,
        observation_spec: dict[str, tuple[int, ...]],
    ) -> Params:
        """VLAPolicy has no local parameters — returns an empty dict."""
        return {}

    def select_action(
        self,
        params: Params,
        observation: Observation,
        rng: RNGKey,
        *,
        deterministic: bool = True,
        task_description: str | None = None,
    ) -> Action:
        """Run VLA inference via Gemini Robotics SDK.

        Args:
            params: Unused (VLA has no local Flax params).
            observation: Dict with "image" and optionally "proprio".
            rng: JAX random key (unused for deterministic inference).
            deterministic: Ignored (temperature=0 is always used by default).
            task_description: Override the default task description.

        Returns:
            Predicted action as a JAX array of shape (action_dim,).
        """
        desc = task_description or self.task_description

        # Convert JAX arrays to NumPy for the SDK
        np_obs = {k: np.array(v) for k, v in observation.items()}
        action_np = self._adapter.predict_action(np_obs, desc)
        return jnp.array(action_np)

    def loss(
        self,
        params: Params,
        batch: Batch,
        rng: RNGKey,
    ) -> tuple[jax.Array, Metrics]:
        """VLAPolicy does not support local gradient-based fine-tuning.

        Fine-tuning is done on the Gemini Robotics platform via the
        ``botwire-upload`` CLI. This method raises NotImplementedError.
        """
        raise NotImplementedError(
            "VLAPolicy fine-tuning is handled by the Gemini Robotics platform. "
            "Use `botwire-upload` to upload your dataset for fine-tuning."
        )

    def set_task(self, task_description: str) -> None:
        """Update the default task description for subsequent calls."""
        self.task_description = task_description

    def __repr__(self) -> str:
        return (
            f"VLAPolicy(model='{self.config.model_id}', "
            f"action_dim={self.action_dim}, "
            f"task='{self.task_description}')"
        )
