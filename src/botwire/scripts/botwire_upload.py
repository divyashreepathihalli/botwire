"""botwire-upload: Upload a dataset or policy to HuggingFace Hub.

Usage:
    # Upload a local RLDS dataset
    botwire-upload --dataset=./data --repo=my-org/franka-reach-v1

    # Upload a trained policy checkpoint
    botwire-upload --checkpoint=./checkpoints/latest --repo=my-org/franka-act-v1
"""

from absl import app, flags, logging

FLAGS = flags.FLAGS
flags.DEFINE_string("dataset", "", "Path to local RLDS dataset directory.")
flags.DEFINE_string("checkpoint", "", "Path to Orbax checkpoint directory.")
flags.DEFINE_string("repo", "", "HuggingFace Hub repo ID (e.g. 'my-org/my-dataset').")
flags.DEFINE_boolean("private", False, "Create a private repository.")


def main(argv):
    del argv

    if not FLAGS.repo:
        raise ValueError("--repo is required.")

    if FLAGS.dataset:
        _upload_dataset(FLAGS.dataset, FLAGS.repo, private=FLAGS.private)
    elif FLAGS.checkpoint:
        _upload_checkpoint(FLAGS.checkpoint, FLAGS.repo, private=FLAGS.private)
    else:
        raise ValueError("Provide either --dataset or --checkpoint.")


def _upload_dataset(local_path: str, repo_id: str, private: bool) -> None:
    from botwire.hub import push_dataset_to_hub

    logging.info("Uploading dataset '%s' to hub repo '%s'", local_path, repo_id)
    push_dataset_to_hub(local_path, repo_id, private=private)
    logging.info("Dataset uploaded successfully.")


def _upload_checkpoint(checkpoint_path: str, repo_id: str, private: bool) -> None:
    from botwire.hub import push_policy_to_hub

    logging.info("Uploading checkpoint '%s' to hub repo '%s'", checkpoint_path, repo_id)
    push_policy_to_hub(checkpoint_path, repo_id, private=private)
    logging.info("Checkpoint uploaded successfully.")


if __name__ == "__main__":
    app.run(main)
