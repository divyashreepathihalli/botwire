"""High-level dataset loading helpers.

These functions are the primary entry points for loading datasets in botwire.

Open X-Embodiment datasets:
    Any dataset registered in the OXE collection on tensorflow_datasets can be
    loaded by name. See the full list at:
    https://github.com/google-deepmind/open_x_embodiment

Local RLDS datasets:
    Episodes recorded with EpisodeWriter (botwire.datasets.rlds_writer) are
    stored as standard RLDS TFRecord files and can be loaded with rlds_load().
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import tensorflow as tf
import tensorflow_datasets as tfds
from absl import logging

from botwire.datasets.rlds_dataset import RLDSDataset

# ---------------------------------------------------------------------------
# Open X-Embodiment dataset registry
# ---------------------------------------------------------------------------

# Subset of widely-used OXE datasets.  Full list:
# https://github.com/google-deepmind/open_x_embodiment#datasets
OXE_DATASETS: dict[str, str] = {
    # Google Robot (manipulation)
    "fractal20220817_data": "fractal20220817_data",
    "kuka": "kuka",
    # RT-2 data
    "bridge": "bridge",
    "bridge_dataset": "bridge_dataset",
    # Stanford / Berkeley
    "bc_z": "bc_z",
    "robo_net": "robo_net",
    "language_table": "language_table",
    # CMU / Columbia
    "columbia_cairlab_pusht_real": "columbia_cairlab_pusht_real",
    "stanford_kuka_multimodal_dataset_converted_externally_to_rlds": (
        "stanford_kuka_multimodal_dataset_converted_externally_to_rlds"
    ),
    # Robotics at Google
    "nyu_door_opening_surprising_effectiveness": (
        "nyu_door_opening_surprising_effectiveness"
    ),
    "viola": "viola",
    "berkeley_autolab_ur5": "berkeley_autolab_ur5",
    "toto": "toto",
}


def oxe_load(
    name: str,
    split: str = "train",
    *,
    data_dir: str | None = None,
    image_keys: tuple[str, ...] = ("image",),
    image_size: tuple[int, int] | None = (224, 224),
    normalize: bool = True,
) -> RLDSDataset:
    """Load an Open X-Embodiment dataset by name.

    The dataset is downloaded and cached automatically by tensorflow-datasets.

    Args:
        name: Dataset name (see OXE_DATASETS or any registered TFDS name).
        split: TFDS split string, e.g. "train", "train[:80%]", "test".
        data_dir: Local cache directory. Defaults to ~/tensorflow_datasets.
        image_keys: Observation keys containing RGB images.
        image_size: Resize images to (H, W). None = keep original.
        normalize: Normalize observations and actions to [-1, 1].

    Returns:
        A :class:`RLDSDataset` ready for iteration.

    Example:
        >>> ds = oxe_load("fractal20220817_data", split="train[:10%]")
        >>> print(ds)
        RLDSDataset(episodes=..., steps=...)
    """
    tfds_name = OXE_DATASETS.get(name, name)
    logging.info("Loading OXE dataset '%s' (split='%s')", tfds_name, split)

    builder = tfds.builder(tfds_name, data_dir=data_dir)
    builder.download_and_prepare()

    tf_dataset = builder.as_dataset(split=split)

    info = builder.info
    episode_count = info.splits[split.split("[")[0]].num_examples
    # Step count is approximate (not always in metadata)
    step_count = int(episode_count) * 100  # rough estimate

    return RLDSDataset(
        tf_dataset=tf_dataset,
        episode_count=episode_count,
        step_count=step_count,
        image_keys=image_keys,
        image_size=image_size,
        normalize=normalize,
    )


def rlds_load(
    path: str | Path,
    split: str = "train",
    *,
    image_keys: tuple[str, ...] = ("image",),
    image_size: tuple[int, int] | None = (224, 224),
    normalize: bool = True,
) -> RLDSDataset:
    """Load a locally stored RLDS dataset from a directory.

    The directory should contain TFRecord files written by
    :class:`botwire.datasets.EpisodeWriter`.

    Args:
        path: Directory containing RLDS TFRecord shards.
        split: Which split to load (if the directory has train/val/test subdirs).
        image_keys: Observation keys containing RGB images.
        image_size: Resize images to (H, W). None = keep original.
        normalize: Normalize observations and actions.

    Returns:
        A :class:`RLDSDataset` ready for iteration.

    Example:
        >>> ds = rlds_load("./recorded_episodes")
        >>> for batch in ds.as_iterator(batch_size=16, sequence_length=10):
        ...     print(batch["action"].shape)
    """
    path = Path(path)
    split_path = path / split if (path / split).exists() else path
    logging.info("Loading local RLDS dataset from '%s'", split_path)

    pattern = str(split_path / "*.tfrecord*")
    files = tf.io.matching_files(pattern)
    if len(files) == 0:
        raise FileNotFoundError(
            f"No TFRecord files found in '{split_path}'. "
            "Use EpisodeWriter to record episodes first."
        )

    tf_dataset = tf.data.TFRecordDataset(
        files, compression_type="GZIP", num_parallel_reads=tf.data.AUTOTUNE
    ).map(_parse_rlds_record, num_parallel_calls=tf.data.AUTOTUNE)

    # Count episodes
    episode_count = sum(1 for _ in tf_dataset)

    return RLDSDataset(
        tf_dataset=tf_dataset,
        episode_count=episode_count,
        step_count=episode_count * 100,  # will be updated by writer metadata
        image_keys=image_keys,
        image_size=image_size,
        normalize=normalize,
    )


def _parse_rlds_record(serialized: tf.Tensor) -> dict[str, Any]:
    """Parse a serialized RLDS episode TFRecord."""
    # RLDS stores episodes as serialized tf.train.SequenceExample.
    # The exact schema depends on the dataset; this is a simplified parser
    # for episodes written by EpisodeWriter.
    features = tf.io.parse_single_example(
        serialized,
        features={
            "steps/observation/image": tf.io.VarLenFeature(tf.string),
            "steps/observation/proprio": tf.io.VarLenFeature(tf.float32),
            "steps/action": tf.io.VarLenFeature(tf.float32),
            "steps/reward": tf.io.VarLenFeature(tf.float32),
            "steps/is_terminal": tf.io.VarLenFeature(tf.bool),
        },
    )
    return {
        "steps": {
            "observation": {
                "image": tf.sparse.to_dense(features["steps/observation/image"]),
                "proprio": tf.sparse.to_dense(features["steps/observation/proprio"]),
            },
            "action": tf.sparse.to_dense(features["steps/action"]),
            "reward": tf.sparse.to_dense(features["steps/reward"]),
            "is_terminal": tf.sparse.to_dense(features["steps/is_terminal"]),
        }
    }
