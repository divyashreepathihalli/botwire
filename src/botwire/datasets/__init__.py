"""RLDS / Open X-Embodiment dataset integration for botwire.

Datasets are stored in the RLDS (Reinforcement Learning Dataset Standard)
format via tensorflow-datasets and can be loaded directly from the
Open X-Embodiment collection or from local directories.

Usage:
    from botwire.datasets import oxe_load, rlds_load

    # Load an OXE dataset from the public TFDS registry
    ds = oxe_load("fractal20220817_data", split="train[:80%]")

    # Load a locally recorded dataset
    ds = rlds_load("/path/to/episodes")

    # Iterate batches for IL training
    for batch in ds.as_iterator(batch_size=32, sequence_length=16):
        ...  # batch is a dict of jax.Arrays

    # Record new episodes
    from botwire.datasets import EpisodeWriter
    with EpisodeWriter("/path/to/episodes") as writer:
        writer.add_step(obs=..., action=..., reward=...)
        writer.end_episode()
"""

from botwire.datasets.loaders import oxe_load, rlds_load
from botwire.datasets.rlds_dataset import RLDSDataset
from botwire.datasets.rlds_writer import EpisodeWriter

__all__ = [
    "RLDSDataset",
    "EpisodeWriter",
    "oxe_load",
    "rlds_load",
]
