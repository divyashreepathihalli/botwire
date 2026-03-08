"""Upload datasets and policy checkpoints to HuggingFace Hub."""

from __future__ import annotations

from pathlib import Path

from absl import logging


def push_dataset_to_hub(
    local_path: str | Path,
    repo_id: str,
    *,
    private: bool = False,
    token: str | None = None,
    commit_message: str = "Upload botwire RLDS dataset",
) -> str:
    """Upload a local RLDS dataset directory to HuggingFace Hub.

    Args:
        local_path: Path to the local RLDS dataset directory (containing TFRecord shards).
        repo_id: HuggingFace Hub repository ID (e.g. "my-org/franka-reach-v1").
        private: Create a private repository if it doesn't exist.
        token: HuggingFace API token. Uses HF_TOKEN env var or cached login if None.
        commit_message: Git commit message for the upload.

    Returns:
        URL of the uploaded dataset on HuggingFace Hub.

    Example::

        url = push_dataset_to_hub("./data", "my-org/franka-reach-v1")
        print(f"Dataset available at: {url}")
    """
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError as e:
        raise ImportError(
            "huggingface-hub not installed. Run: pip install huggingface-hub"
        ) from e

    local_path = Path(local_path)
    if not local_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: '{local_path}'")

    api = HfApi(token=token)

    # Create repo if needed
    try:
        create_repo(repo_id, repo_type="dataset", private=private, token=token, exist_ok=True)
    except Exception as exc:
        logging.warning("Could not create repo: %s", exc)

    logging.info("Uploading '%s' to '%s'...", local_path, repo_id)
    api.upload_folder(
        folder_path=str(local_path),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=commit_message,
    )

    url = f"https://huggingface.co/datasets/{repo_id}"
    logging.info("Dataset uploaded: %s", url)
    return url


def push_policy_to_hub(
    checkpoint_path: str | Path,
    repo_id: str,
    *,
    private: bool = False,
    token: str | None = None,
    policy_name: str = "botwire_policy",
    commit_message: str = "Upload botwire policy checkpoint",
) -> str:
    """Upload an Orbax policy checkpoint to HuggingFace Hub.

    Args:
        checkpoint_path: Path to the local Orbax checkpoint directory.
        repo_id: HuggingFace Hub repository ID.
        private: Create a private repository.
        token: HuggingFace API token.
        policy_name: Name prefix for the checkpoint files in the repo.
        commit_message: Git commit message for the upload.

    Returns:
        URL of the uploaded model on HuggingFace Hub.
    """
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError as e:
        raise ImportError("Run: pip install huggingface-hub") from e

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: '{checkpoint_path}'")

    api = HfApi(token=token)

    try:
        create_repo(repo_id, repo_type="model", private=private, token=token, exist_ok=True)
    except Exception as exc:
        logging.warning("Could not create repo: %s", exc)

    logging.info("Uploading checkpoint '%s' to '%s'...", checkpoint_path, repo_id)
    api.upload_folder(
        folder_path=str(checkpoint_path),
        repo_id=repo_id,
        repo_type="model",
        path_in_repo=policy_name,
        commit_message=commit_message,
    )

    url = f"https://huggingface.co/{repo_id}"
    logging.info("Policy uploaded: %s", url)
    return url
