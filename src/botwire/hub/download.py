"""Download datasets and policy checkpoints from HuggingFace Hub."""

from __future__ import annotations

from pathlib import Path

from absl import logging


def load_dataset(
    repo_id: str,
    *,
    local_dir: str | Path | None = None,
    token: str | None = None,
    image_keys: tuple[str, ...] = ("image",),
    image_size: tuple[int, int] | None = (224, 224),
) -> "RLDSDataset":
    """Download an RLDS dataset from HuggingFace Hub and return a RLDSDataset.

    Args:
        repo_id: HuggingFace Hub dataset ID (e.g. "my-org/franka-reach-v1").
        local_dir: Directory to cache the downloaded dataset. Defaults to ~/.cache/botwire.
        token: HuggingFace API token.
        image_keys: Observation keys containing images.
        image_size: Resize images to (H, W).

    Returns:
        A :class:`~botwire.datasets.RLDSDataset` ready for iteration.

    Example::

        ds = load_dataset("my-org/franka-reach-v1")
        for batch in ds.as_iterator(batch_size=32, sequence_length=10):
            ...
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ImportError("Run: pip install huggingface-hub") from e

    from botwire.datasets import rlds_load

    if local_dir is None:
        local_dir = Path.home() / ".cache" / "botwire" / "datasets" / repo_id.replace("/", "__")

    logging.info("Downloading dataset '%s' from HuggingFace Hub...", repo_id)
    local_path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local_dir),
        token=token,
    )
    logging.info("Dataset cached at '%s'", local_path)
    return rlds_load(
        local_path,
        image_keys=image_keys,
        image_size=image_size,
    )


def load_policy(
    repo_id: str,
    policy_name: str = "botwire_policy",
    *,
    local_dir: str | Path | None = None,
    token: str | None = None,
) -> dict:
    """Download a botwire policy checkpoint from HuggingFace Hub.

    Args:
        repo_id: HuggingFace Hub model ID.
        policy_name: Name of the checkpoint subdirectory in the repo.
        local_dir: Local cache directory.
        token: HuggingFace API token.

    Returns:
        Restored state dict (with "params" and optionally "opt_state").

    Example::

        state = load_policy("my-org/franka-act-v1")
        params = state["params"]
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ImportError("Run: pip install huggingface-hub") from e

    import orbax.checkpoint as ocp

    if local_dir is None:
        local_dir = Path.home() / ".cache" / "botwire" / "models" / repo_id.replace("/", "__")

    logging.info("Downloading policy '%s' from HuggingFace Hub...", repo_id)
    local_path = snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        local_dir=str(local_dir),
        token=token,
    )

    ckpt_path = Path(local_path) / policy_name
    checkpointer = ocp.PyTreeCheckpointer()
    state = checkpointer.restore(str(ckpt_path))
    logging.info("Policy restored from '%s'", ckpt_path)
    return state
