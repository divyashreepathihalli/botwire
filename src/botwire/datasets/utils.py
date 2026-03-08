"""Dataset utility functions."""

from __future__ import annotations

from typing import Any, Iterator

import jax
import jax.numpy as jnp
import numpy as np

from botwire.common.types import Batch, RNGKey


def batch_episodes(
    episodes: list[dict[str, Any]],
    batch_size: int,
    sequence_length: int,
    rng: RNGKey,
) -> Iterator[Batch]:
    """Sample random batches from a list of in-memory episodes.

    Useful for small datasets that fit in RAM.

    Args:
        episodes: List of episode dicts with "observation", "action", "reward" keys.
        batch_size: Number of sequences per batch.
        sequence_length: Length of each sequence.
        rng: JAX random key.

    Yields:
        Batches of shape (batch_size, sequence_length, ...).
    """
    while True:
        rng, sample_rng = jax.random.split(rng)
        batch = _sample_batch(episodes, batch_size, sequence_length, sample_rng)
        yield batch


def _sample_batch(
    episodes: list[dict[str, Any]],
    batch_size: int,
    seq_len: int,
    rng: RNGKey,
) -> Batch:
    samples = []
    rng_ep, rng_start = jax.random.split(rng)
    ep_indices = jax.random.randint(rng_ep, (batch_size,), 0, len(episodes))

    for i, ep_idx in enumerate(np.array(ep_indices)):
        ep = episodes[ep_idx]
        ep_len = len(ep["action"])
        if ep_len <= seq_len:
            start = 0
        else:
            start = int(jax.random.randint(rng_start[i] if rng_start.ndim > 1 else rng_start,
                                            (), 0, ep_len - seq_len))
        samples.append({
            "observation": {
                k: v[start:start + seq_len]
                for k, v in ep["observation"].items()
            },
            "action": ep["action"][start:start + seq_len],
            "reward": ep["reward"][start:start + seq_len],
        })

    return {
        "observation": {
            k: jnp.stack([s["observation"][k] for s in samples])
            for k in samples[0]["observation"]
        },
        "action": jnp.stack([s["action"] for s in samples]),
        "reward": jnp.stack([s["reward"] for s in samples]),
    }


def compute_dataset_stats(
    dataset_iter: Iterator[Batch],
    num_batches: int = 100,
) -> dict[str, dict[str, jax.Array]]:
    """Estimate mean and std for each key by consuming num_batches.

    Returns a stats dict suitable for passing to normalize_batch().
    """
    accum: dict[str, list[jax.Array]] = {}

    for _ in range(num_batches):
        batch = next(dataset_iter)
        for key in ("action",):
            accum.setdefault(key, []).append(batch[key])
        if "observation" in batch:
            for obs_key, val in batch["observation"].items():
                full_key = f"observation/{obs_key}"
                if val.dtype in (jnp.float32, jnp.float64):
                    accum.setdefault(full_key, []).append(val)

    stats: dict[str, dict[str, jax.Array]] = {}
    for key, arrays in accum.items():
        concat = jnp.concatenate([a.reshape(-1, a.shape[-1]) for a in arrays], axis=0)
        stats[key] = {
            "mean": jnp.mean(concat, axis=0),
            "std": jnp.std(concat, axis=0) + 1e-8,
        }
    return stats
