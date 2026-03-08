"""Example: Train SAC agent on Brax Ant environment.

Usage:
    python examples/train_rl_brax.py

    # Or with custom args:
    python examples/train_rl_brax.py --num_steps=500000 --env=halfcheetah
"""

from absl import app, flags, logging
import jax

flags.DEFINE_string("env", "ant", "Brax environment name.")
flags.DEFINE_integer("num_steps", 200_000, "Number of environment steps.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_string("output_dir", "./checkpoints/sac_brax", "Checkpoint directory.")
FLAGS = flags.FLAGS


def main(argv):
    del argv

    from botwire.agents import SACAgent
    from botwire.configs import get_sac_config, get_rl_train_config
    from botwire.envs import BraxEnv
    from botwire.training import RLTrainer

    logging.info("=== SAC on Brax %s ===", FLAGS.env)

    # Build environment
    env = BraxEnv(env_name=FLAGS.env, backend="mjx")
    logging.info("Environment: obs_spec=%s, action_dim=%d", env.observation_spec, env.action_dim)

    # Build agent
    sac_config = get_sac_config()
    sac_config.num_steps = FLAGS.num_steps
    agent = SACAgent(env.observation_spec, env.action_spec, sac_config)

    # Build training config
    train_config = get_rl_train_config()
    train_config.seed = FLAGS.seed
    train_config.num_steps = FLAGS.num_steps
    train_config.output_dir = FLAGS.output_dir
    train_config.env.env_name = FLAGS.env
    train_config.log_every = 1000
    train_config.save_every = 50_000

    # Train
    trainer = RLTrainer(agent, env, train_config)
    rng = jax.random.PRNGKey(FLAGS.seed)
    trainer.train(rng=rng)


if __name__ == "__main__":
    app.run(main)
