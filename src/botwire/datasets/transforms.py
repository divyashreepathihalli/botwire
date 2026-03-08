"""Data transforms: normalization, augmentation, and JAX conversion.

All transforms operate on dicts of numpy/JAX arrays matching the RLDS
batch schema produced by RLDSDataset.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf

from botwire.common.types import Batch


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


class RunningStats:
    """Online running mean and standard deviation using Welford's algorithm."""

    def __init__(self, shape: tuple[int, ...]) -> None:
        self.n = 0
        self.mean = np.zeros(shape, dtype=np.float64)
        self.M2 = np.zeros(shape, dtype=np.float64)

    def update(self, x: np.ndarray) -> None:
        x = x.reshape(-1, *self.mean.shape)
        for xi in x:
            self.n += 1
            delta = xi - self.mean
            self.mean += delta / self.n
            delta2 = xi - self.mean
            self.M2 += delta * delta2

    @property
    def std(self) -> np.ndarray:
        if self.n < 2:
            return np.ones_like(self.mean)
        return np.sqrt(self.M2 / (self.n - 1))

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / (self.std + 1e-8)

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        return x * (self.std + 1e-8) + self.mean


def normalize_batch(
    batch: Batch,
    stats: dict[str, dict[str, jax.Array]],
) -> Batch:
    """Normalize observations and actions using precomputed statistics.

    Args:
        batch: Dict with "observation" (nested dict) and "action" keys.
        stats: Dict mapping key → {"mean": array, "std": array}.
               E.g. {"action": {"mean": ..., "std": ...},
                     "observation/proprio": {"mean": ..., "std": ...}}

    Returns:
        Normalized batch (same structure).
    """
    batch = dict(batch)

    # Normalize actions
    if "action" in stats and "action" in batch:
        s = stats["action"]
        batch["action"] = (batch["action"] - s["mean"]) / (s["std"] + 1e-8)

    # Normalize observations
    if "observation" in batch:
        obs = dict(batch["observation"])
        for key, val in obs.items():
            stat_key = f"observation/{key}"
            if stat_key in stats:
                s = stats[stat_key]
                obs[key] = (val - s["mean"]) / (s["std"] + 1e-8)
        batch["observation"] = obs

    return batch


# ---------------------------------------------------------------------------
# Image transforms
# ---------------------------------------------------------------------------


def resize_images(
    batch: Batch,
    image_keys: tuple[str, ...],
    size: tuple[int, int],
) -> Batch:
    """Resize image observations to (H, W).

    Operates on JAX arrays. Images are expected to be uint8 [0, 255] or
    float32 [0, 1].
    """
    batch = dict(batch)
    obs = dict(batch.get("observation", {}))
    for key in image_keys:
        if key in obs:
            img = obs[key]
            # Use TF resize then convert back — avoids needing PIL/cv2
            img_tf = tf.image.resize(img, size)
            obs[key] = jnp.array(img_tf.numpy())
    if obs:
        batch["observation"] = obs
    return batch


def random_crop(
    image: jax.Array,
    rng: jax.Array,
    crop_size: tuple[int, int],
) -> jax.Array:
    """Randomly crop an image to crop_size.

    Args:
        image: Array of shape (..., H, W, C).
        rng: JAX random key.
        crop_size: Target (H, W).

    Returns:
        Cropped image of shape (..., crop_H, crop_W, C).
    """
    h, w = image.shape[-3], image.shape[-2]
    crop_h, crop_w = crop_size
    rng_h, rng_w = jax.random.split(rng)
    top = jax.random.randint(rng_h, (), 0, h - crop_h + 1)
    left = jax.random.randint(rng_w, (), 0, w - crop_w + 1)
    return jax.lax.dynamic_slice(
        image,
        [0] * (image.ndim - 3) + [top, left, 0],
        list(image.shape[:-3]) + [crop_h, crop_w, image.shape[-1]],
    )


def random_brightness(
    image: jax.Array,
    rng: jax.Array,
    max_delta: float = 0.2,
) -> jax.Array:
    """Randomly adjust image brightness."""
    delta = jax.random.uniform(rng, (), minval=-max_delta, maxval=max_delta)
    return jnp.clip(image + delta, 0.0, 1.0)


def normalize_image(image: jax.Array) -> jax.Array:
    """Scale uint8 image [0, 255] → float32 [0, 1]."""
    return image.astype(jnp.float32) / 255.0


# ---------------------------------------------------------------------------
# Batch utilities
# ---------------------------------------------------------------------------


def to_jax_batch(batch: dict[str, Any]) -> Batch:
    """Recursively convert numpy arrays / TF tensors in a dict to jax.Arrays."""

    def _convert(x: Any) -> Any:
        if isinstance(x, dict):
            return {k: _convert(v) for k, v in x.items()}
        if isinstance(x, tf.Tensor):
            return jnp.array(x.numpy())
        if isinstance(x, np.ndarray):
            return jnp.array(x)
        return x

    return _convert(batch)
