"""botwire-collect: Record robot episodes from a simulation environment.

Usage:
    # Collect 100 random episodes from Brax Ant
    botwire-collect --env=ant --episodes=100 --output=./data

    # Collect with a trained policy
    botwire-collect --env=franka_fr3/reach \\
                    --policy_checkpoint=./checkpoints/latest \\
                    --episodes=50 \\
                    --output=./franka_data
"""

from absl import app, flags, logging
import jax
import jax.numpy as jnp

FLAGS = flags.FLAGS
flags.DEFINE_string("env", "ant", "Environment name (Brax or Menagerie).")
flags.DEFINE_integer("episodes", 100, "Number of episodes to collect.")
flags.DEFINE_string("output", "./data", "Output directory for RLDS episodes.")
flags.DEFINE_string("policy_checkpoint", "", "Path to policy checkpoint (empty = random policy).")
flags.DEFINE_integer("max_steps", 1000, "Maximum steps per episode.")
flags.DEFINE_integer("seed", 0, "Random seed.")


def main(argv):
    del argv

    from botwire.datasets import EpisodeWriter
    from botwire.envs import BraxEnv

    env = BraxEnv(env_name=FLAGS.env, episode_length=FLAGS.max_steps)
    rng = jax.random.PRNGKey(FLAGS.seed)

    # Load policy or use random
    policy_fn = _load_policy(FLAGS.policy_checkpoint, env) if FLAGS.policy_checkpoint else None

    logging.info(
        "Collecting %d episodes from '%s' → '%s'",
        FLAGS.episodes,
        FLAGS.env,
        FLAGS.output,
    )

    with EpisodeWriter(FLAGS.output) as writer:
        for ep in range(FLAGS.episodes):
            rng, reset_rng = jax.random.split(rng)
            state, obs = env.reset(reset_rng)
            writer.add_step(
                observation={k: v.tolist() for k, v in obs.items()},
                action=jnp.zeros(env.action_dim).tolist(),
                reward=0.0,
                is_first=True,
            )

            for step in range(FLAGS.max_steps):
                rng, act_rng = jax.random.split(rng)
                if policy_fn is not None:
                    action = policy_fn(obs, act_rng)
                else:
                    # Random policy
                    action = jax.random.uniform(
                        act_rng, (env.action_dim,), minval=-1.0, maxval=1.0
                    )

                state, next_obs, reward, done, _ = env.step(state, action)
                writer.add_step(
                    observation={k: v.tolist() for k, v in next_obs.items()},
                    action=action.tolist(),
                    reward=float(reward),
                    is_terminal=bool(done),
                )
                obs = next_obs
                if bool(done):
                    break

            writer.end_episode()

            if (ep + 1) % 10 == 0:
                logging.info("Collected %d / %d episodes", ep + 1, FLAGS.episodes)

    logging.info("Collection complete. Wrote %d episodes to '%s'", FLAGS.episodes, FLAGS.output)


def _load_policy(checkpoint_path: str, env) -> object:
    """Load a trained policy from a checkpoint."""
    import orbax.checkpoint as ocp

    checkpointer = ocp.PyTreeCheckpointer()
    state = checkpointer.restore(checkpoint_path)
    params = state["params"]

    # Return a simple closure
    from botwire.policies.act import ACTPolicy
    from botwire.configs import get_act_config

    policy = ACTPolicy(config=get_act_config(), action_dim=env.action_dim)

    def policy_fn(obs, rng):
        return policy.select_action(params, obs, rng)

    return policy_fn


if __name__ == "__main__":
    app.run(main)
