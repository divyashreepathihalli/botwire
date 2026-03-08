"""Example: Train ACT policy on an Open X-Embodiment dataset.

Usage:
    python examples/train_il_act.py

    # With a specific OXE dataset:
    python examples/train_il_act.py --dataset=bridge_dataset --num_steps=50000
"""

from absl import app, flags, logging
import jax

flags.DEFINE_string("dataset", "fractal20220817_data", "OXE dataset name.")
flags.DEFINE_string("split", "train[:10%]", "TFDS split (use a small split for testing).")
flags.DEFINE_integer("num_steps", 10_000, "Number of gradient steps.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_integer("action_dim", 7, "Robot action dimensionality.")
flags.DEFINE_string("output_dir", "./checkpoints/act_il", "Checkpoint directory.")
FLAGS = flags.FLAGS


def main(argv):
    del argv

    from botwire.configs import get_base_train_config, get_act_config
    from botwire.datasets import oxe_load
    from botwire.policies.act import ACTPolicy
    from botwire.training import ImitationLearningTrainer

    logging.info("=== ACT IL Training on %s ===", FLAGS.dataset)

    # Load dataset
    logging.info("Loading dataset '%s' (split='%s')...", FLAGS.dataset, FLAGS.split)
    dataset = oxe_load(FLAGS.dataset, split=FLAGS.split)
    logging.info("Dataset: %s", dataset)

    # Build policy
    act_config = get_act_config()
    policy = ACTPolicy(config=act_config, action_dim=FLAGS.action_dim)

    # Build training config
    train_config = get_base_train_config()
    train_config.seed = FLAGS.seed
    train_config.num_steps = FLAGS.num_steps
    train_config.output_dir = FLAGS.output_dir
    train_config.dataset.name = FLAGS.dataset
    train_config.dataset.split = FLAGS.split
    train_config.log_every = 100
    train_config.save_every = 5_000

    # Initialize params
    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, init_rng = jax.random.split(rng)
    obs_spec = {"proprio": (14,)}   # 7 joint positions + 7 joint velocities
    params = policy.init(init_rng, obs_spec)

    # Train
    trainer = ImitationLearningTrainer(policy, dataset, train_config)
    trained_params = trainer.train(params=params, observation_spec=obs_spec, rng=rng)
    logging.info("Training complete. Final params saved to '%s'.", FLAGS.output_dir)


if __name__ == "__main__":
    app.run(main)
