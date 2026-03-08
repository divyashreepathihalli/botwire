"""BotwireDataset backed by RLDS (tensorflow-datasets) data.

RLDS (Reinforcement Learning Dataset Standard) is used by Open X-Embodiment
and is the canonical format for robot datasets in the GDM ecosystem.

Each RLDS dataset is a collection of episodes. Each episode is a dict:
    {
      "steps": tf.data.Dataset of {
          "observation": {...},
          "action":      tf.Tensor,
          "reward":      tf.Tensor,
          "is_terminal": tf.Tensor,
          "is_first":    tf.Tensor,
          "is_last":     tf.Tensor,
      }
    }

RLDSDataset flattens these into (batch_size, seq_len, ...) tensors and
converts them to JAX arrays for training.
"""

from __future__ import annotations

import functools
from typing import Any, Iterator

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from botwire.common.abc import BotwireDataset
from botwire.common.types import Batch
from botwire.datasets.transforms import normalize_batch, resize_images


class RLDSDataset(BotwireDataset):
    """Dataset backed by an RLDS-formatted tensorflow_datasets source.

    Args:
        tf_dataset: A ``tf.data.Dataset`` of RLDS episodes.
        episode_count: Total number of episodes (for metadata).
        step_count: Total number of steps (for metadata).
        image_keys: Observation keys that contain image data.
        image_size: Target (H, W) for image resizing. None = no resize.
        normalize: Whether to apply per-key normalization statistics.
        stats: Dict of {"mean": ..., "std": ...} per observation/action key.
    """

    def __init__(
        self,
        tf_dataset: tf.data.Dataset,
        episode_count: int,
        step_count: int,
        *,
        image_keys: tuple[str, ...] = ("image",),
        image_size: tuple[int, int] | None = (224, 224),
        normalize: bool = True,
        stats: dict[str, dict[str, np.ndarray]] | None = None,
    ) -> None:
        self._tf_dataset = tf_dataset
        self._episode_count = episode_count
        self._step_count = step_count
        self.image_keys = image_keys
        self.image_size = image_size
        self.normalize = normalize
        self.stats = stats or {}

    # ------------------------------------------------------------------
    # BotwireDataset interface
    # ------------------------------------------------------------------

    def episode_count(self) -> int:
        return self._episode_count

    def step_count(self) -> int:
        return self._step_count

    def as_iterator(
        self,
        batch_size: int,
        sequence_length: int,
        *,
        shuffle: bool = True,
        seed: int = 0,
    ) -> Iterator[Batch]:
        """Return an infinite iterator of JAX batches.

        Each batch is a dict:
            {
              "observation": {key: jnp.ndarray (B, T, ...)},
              "action":      jnp.ndarray (B, T, action_dim),
              "reward":      jnp.ndarray (B, T),
              "is_terminal": jnp.ndarray (B, T),
            }
        """
        ds = self._build_pipeline(batch_size, sequence_length, shuffle=shuffle, seed=seed)
        for batch in ds:
            yield _tf_to_jax(batch)

    # ------------------------------------------------------------------
    # Internal pipeline
    # ------------------------------------------------------------------

    def _build_pipeline(
        self,
        batch_size: int,
        sequence_length: int,
        shuffle: bool,
        seed: int,
    ) -> tf.data.Dataset:
        ds = self._tf_dataset

        if shuffle:
            ds = ds.shuffle(buffer_size=1000, seed=seed, reshuffle_each_iteration=True)

        # Flatten episodes → step sequences
        ds = ds.flat_map(functools.partial(_episode_to_sequences, seq_len=sequence_length))

        if shuffle:
            ds = ds.shuffle(buffer_size=10_000, seed=seed + 1)

        ds = ds.batch(batch_size, drop_remainder=True)

        # Apply image resizing
        if self.image_size is not None and self.image_keys:
            ds = ds.map(
                functools.partial(
                    _resize_image_keys,
                    image_keys=self.image_keys,
                    size=self.image_size,
                ),
                num_parallel_calls=tf.data.AUTOTUNE,
            )

        ds = ds.prefetch(tf.data.AUTOTUNE)
        ds = ds.repeat()  # infinite iterator
        return ds


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _episode_to_sequences(episode: dict[str, Any], seq_len: int) -> tf.data.Dataset:
    """Convert one RLDS episode into overlapping subsequences of length seq_len."""
    steps = episode["steps"]

    # Batch all steps in the episode into a single dict of tensors
    all_steps: dict[str, tf.Tensor] = {}
    for key in ["observation", "action", "reward", "is_terminal", "is_first", "is_last"]:
        if key == "observation":
            # observation is itself a dict
            obs_ds = steps.map(lambda s: s["observation"])  # type: ignore[index]
            all_steps["observation"] = _batch_nested_dataset(obs_ds)
        else:
            all_steps[key] = tf.stack(list(steps.map(lambda s, k=key: s[k])))  # type: ignore[index]

    # Slide a window of seq_len over the episode
    n_steps = tf.shape(all_steps["reward"])[0]
    indices = tf.range(n_steps - seq_len + 1)

    def slice_at(i: tf.Tensor) -> dict[str, tf.Tensor]:
        sliced: dict[str, tf.Tensor] = {}
        for k, v in all_steps.items():
            if k == "observation":
                sliced[k] = {ok: ov[i : i + seq_len] for ok, ov in v.items()}
            else:
                sliced[k] = v[i : i + seq_len]
        return sliced

    return tf.data.Dataset.from_tensor_slices(indices).map(slice_at)


def _batch_nested_dataset(ds: tf.data.Dataset) -> dict[str, tf.Tensor]:
    """Stack a dataset of dicts into a single dict of stacked tensors."""
    result: dict[str, list[tf.Tensor]] = {}
    for item in ds:
        for k, v in item.items():
            result.setdefault(k, []).append(v)
    return {k: tf.stack(vs) for k, vs in result.items()}


def _resize_image_keys(
    batch: dict[str, Any],
    image_keys: tuple[str, ...],
    size: tuple[int, int],
) -> dict[str, Any]:
    """Resize image tensors in a batch."""
    obs = dict(batch["observation"])
    for key in image_keys:
        if key in obs:
            obs[key] = tf.image.resize(obs[key], size)
    batch = dict(batch)
    batch["observation"] = obs
    return batch


def _tf_to_jax(batch: dict[str, Any]) -> Batch:
    """Recursively convert TF tensors in a batch to JAX arrays."""

    def convert(x: Any) -> Any:
        if isinstance(x, dict):
            return {k: convert(v) for k, v in x.items()}
        if isinstance(x, tf.Tensor):
            return jnp.array(x.numpy())
        return x

    return convert(batch)
