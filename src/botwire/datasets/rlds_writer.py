"""EpisodeWriter: record robot episodes in RLDS format.

Episodes written by this class can be loaded directly with rlds_load() and
are compatible with the Open X-Embodiment dataset schema.

Usage:
    from botwire.datasets import EpisodeWriter

    writer = EpisodeWriter("/path/to/output", shard_size=100)
    writer.open()

    for episode_idx in range(num_episodes):
        state, obs = env.reset(rng)
        while True:
            action = policy.select_action(params, obs, rng)
            next_state, next_obs, reward, done, info = env.step(state, action)

            writer.add_step(
                observation=obs,
                action=np.array(action),
                reward=float(reward),
                is_terminal=bool(done),
            )
            if done:
                writer.end_episode()
                break
            state, obs = next_state, next_obs

    writer.close()

    # Or use as a context manager:
    with EpisodeWriter("/path/to/output") as writer:
        ...
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from absl import logging


class EpisodeWriter:
    """Write robot episodes to RLDS-formatted TFRecord shards.

    Args:
        output_dir: Directory where TFRecord shards will be written.
        shard_size: Maximum number of episodes per shard file.
        compression: TFRecord compression type ("GZIP", "ZLIB", or "").
    """

    def __init__(
        self,
        output_dir: str | Path,
        shard_size: int = 100,
        compression: str = "GZIP",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.shard_size = shard_size
        self.compression = compression

        self._writer: tf.io.TFRecordWriter | None = None
        self._current_episode: list[dict[str, Any]] = []
        self._episode_count = 0
        self._step_count = 0
        self._shard_episode_count = 0
        self._shard_index = 0

    def open(self) -> None:
        """Open the output directory and start the first shard."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._open_shard()
        logging.info("EpisodeWriter opened at '%s'", self.output_dir)

    def _open_shard(self) -> None:
        shard_path = self.output_dir / f"episodes-{self._shard_index:05d}.tfrecord"
        if self.compression:
            shard_path = shard_path.with_suffix(".tfrecord.gz")
        self._writer = tf.io.TFRecordWriter(
            str(shard_path),
            options=tf.io.TFRecordOptions(compression_type=self.compression),
        )
        self._shard_episode_count = 0
        logging.debug("Opened shard '%s'", shard_path)

    def add_step(
        self,
        observation: dict[str, np.ndarray],
        action: np.ndarray,
        reward: float,
        is_terminal: bool = False,
        is_first: bool = False,
        **extras: Any,
    ) -> None:
        """Add one environment step to the current episode buffer.

        Args:
            observation: Dict of sensor arrays (e.g. {"image": ..., "proprio": ...}).
            action: Action array taken at this step.
            reward: Scalar reward received.
            is_terminal: Whether this is the final step of the episode.
            is_first: Whether this is the first step of the episode.
            **extras: Additional step metadata stored as-is.
        """
        step = {
            "observation": {k: np.asarray(v) for k, v in observation.items()},
            "action": np.asarray(action),
            "reward": float(reward),
            "is_terminal": bool(is_terminal),
            "is_first": bool(is_first),
            "is_last": bool(is_terminal),
            "timestamp": time.time(),
            **extras,
        }
        self._current_episode.append(step)

    def end_episode(
        self,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Flush the current episode buffer to disk.

        Args:
            metadata: Optional episode-level metadata dict.
        """
        if not self._current_episode:
            logging.warning("end_episode() called with no steps; skipping.")
            return

        episode = {
            "steps": self._current_episode,
            "metadata": metadata or {},
            "episode_id": self._episode_count,
        }

        serialized = _serialize_episode(episode)
        assert self._writer is not None
        self._writer.write(serialized)

        self._episode_count += 1
        self._step_count += len(self._current_episode)
        self._shard_episode_count += 1
        self._current_episode = []

        # Rotate shard if full
        if self._shard_episode_count >= self.shard_size:
            self._writer.flush()
            self._writer.close()
            self._shard_index += 1
            self._open_shard()

    def close(self) -> None:
        """Flush all pending data and close the current shard."""
        if self._current_episode:
            logging.warning(
                "EpisodeWriter closed with %d unfinished steps; "
                "call end_episode() before close().",
                len(self._current_episode),
            )
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()
            self._writer = None

        # Write dataset metadata
        meta = {
            "total_episodes": self._episode_count,
            "total_steps": self._step_count,
            "num_shards": self._shard_index + 1,
            "compression": self.compression,
        }
        meta_path = self.output_dir / "dataset_metadata.json"
        meta_path.write_text(json.dumps(meta, indent=2))
        logging.info(
            "EpisodeWriter closed: %d episodes, %d steps written to '%s'",
            self._episode_count,
            self._step_count,
            self.output_dir,
        )

    # Context manager
    def __enter__(self) -> "EpisodeWriter":
        self.open()
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    @property
    def num_episodes(self) -> int:
        return self._episode_count

    @property
    def num_steps(self) -> int:
        return self._step_count


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _bytes_feature(value: bytes) -> tf.train.Feature:
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value: list[float]) -> tf.train.Feature:
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_feature(value: list[int]) -> tf.train.Feature:
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _serialize_episode(episode: dict[str, Any]) -> bytes:
    """Serialize an episode dict to a TFRecord bytes string."""
    steps = episode["steps"]
    n_steps = len(steps)

    # Flatten steps into per-key lists
    actions = np.stack([s["action"] for s in steps])
    rewards = np.array([s["reward"] for s in steps], dtype=np.float32)
    is_terminal = np.array([s["is_terminal"] for s in steps], dtype=np.int64)

    feature: dict[str, tf.train.Feature] = {
        "episode_id": _int64_feature([episode["episode_id"]]),
        "n_steps": _int64_feature([n_steps]),
        "steps/action": _float_feature(actions.flatten().tolist()),
        "steps/action/shape": _int64_feature(list(actions.shape)),
        "steps/reward": _float_feature(rewards.tolist()),
        "steps/is_terminal": _int64_feature(is_terminal.tolist()),
    }

    # Serialize each observation key
    for obs_key in steps[0]["observation"]:
        arr = np.stack([s["observation"][obs_key] for s in steps])
        if arr.dtype == np.uint8 or obs_key == "image":
            # Store images as raw bytes for efficiency
            feature[f"steps/observation/{obs_key}"] = _bytes_feature(arr.tobytes())
            feature[f"steps/observation/{obs_key}/shape"] = _int64_feature(list(arr.shape))
            feature[f"steps/observation/{obs_key}/dtype"] = _bytes_feature(
                arr.dtype.str.encode()
            )
        else:
            feature[f"steps/observation/{obs_key}"] = _float_feature(
                arr.astype(np.float32).flatten().tolist()
            )
            feature[f"steps/observation/{obs_key}/shape"] = _int64_feature(list(arr.shape))

    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()
