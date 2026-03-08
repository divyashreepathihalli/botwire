"""Utility functions shared across botwire modules."""

from __future__ import annotations

import logging
import random
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from absl import logging as absl_logging

from botwire.common.types import Metrics, RNGKey

logger = logging.getLogger(__name__)


def seed_everything(seed: int) -> RNGKey:
    """Set random seeds for Python, NumPy, and JAX; return a JAX RNG key.

    Args:
        seed: Integer seed value.

    Returns:
        JAX RNGKey split from the given seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    key = jax.random.PRNGKey(seed)
    absl_logging.info("Random seed set to %d", seed)
    return key


def get_device(device: str = "auto") -> jax.Device:
    """Return a JAX device by name.

    Args:
        device: One of "cpu", "gpu", "tpu", or "auto" (picks GPU/TPU if available).

    Returns:
        A :class:`jax.Device` object.
    """
    if device == "auto":
        backends = ["gpu", "tpu", "cpu"]
        for backend in backends:
            try:
                devs = jax.devices(backend)
                if devs:
                    return devs[0]
            except RuntimeError:
                continue
        return jax.devices("cpu")[0]
    return jax.devices(device)[0]


def log_metrics(
    metrics: Metrics,
    step: int,
    prefix: str = "",
    *,
    use_wandb: bool = False,
    use_tb: bool = False,
    tb_writer: Any = None,
) -> None:
    """Log a metrics dict to console and optionally to WandB / TensorBoard.

    Args:
        metrics: Dict of metric name → scalar value.
        step: Current training step.
        prefix: Optional prefix prepended to each metric name.
        use_wandb: If True, log to Weights & Biases (must be initialized).
        use_tb: If True, log to TensorBoard via ``tb_writer``.
        tb_writer: A ``torch.utils.tensorboard.SummaryWriter`` (or compatible).
    """
    prefixed = {f"{prefix}/{k}" if prefix else k: v for k, v in metrics.items()}

    # Console
    parts = [f"{k}={float(v):.4f}" for k, v in prefixed.items()]
    absl_logging.info("step=%d  %s", step, "  ".join(parts))

    # WandB
    if use_wandb:
        try:
            import wandb

            wandb.log(prefixed, step=step)
        except ImportError:
            logger.warning("wandb not installed; skipping WandB logging.")

    # TensorBoard
    if use_tb and tb_writer is not None:
        for k, v in prefixed.items():
            tb_writer.add_scalar(k, float(v), global_step=step)


def tree_stack(trees: list[Any], axis: int = 0) -> Any:
    """Stack a list of pytrees along ``axis``.

    Equivalent to ``jax.tree.map(lambda *a: jnp.stack(a, axis), *trees)``.
    """
    return jax.tree.map(lambda *a: jnp.stack(a, axis=axis), *trees)


def tree_unstack(tree: Any) -> list[Any]:
    """Split the leading axis of a pytree into a list of pytrees."""
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    n = leaves[0].shape[0]
    return [
        treedef.unflatten([leaf[i] for leaf in leaves]) for i in range(n)
    ]


def count_parameters(params: Any) -> int:
    """Count the total number of trainable parameters in a pytree."""
    return sum(x.size for x in jax.tree_util.tree_leaves(params))


def format_params(params: Any) -> str:
    """Human-readable parameter count (e.g. '12.5M')."""
    n = count_parameters(params)
    if n >= 1e9:
        return f"{n/1e9:.1f}B"
    if n >= 1e6:
        return f"{n/1e6:.1f}M"
    if n >= 1e3:
        return f"{n/1e3:.1f}K"
    return str(n)
