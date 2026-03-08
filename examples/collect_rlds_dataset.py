"""Example: Collect robot episodes from a Brax environment and save as RLDS.

Usage:
    python examples/collect_rlds_dataset.py

    # Collect 50 episodes with a random policy:
    python examples/collect_rlds_dataset.py --episodes=50 --env=ant --output=./my_dataset
"""

from absl import app, flags, logging
import jax
import jax.numpy as jnp

flags.DEFINE_string("env", "ant", "Brax environment name.")
flags.DEFINE_integer("episodes", 20, "Number of episodes to collect.")
flags.DEFINE_integer("max_steps_per_episode", 500, "Max steps per episode.")
flags.DEFINE_string("output", "./collected_data", "Output directory for RLDS episodes.")
flags.DEFINE_integer("seed", 0, "Random seed.")
FLAGS = flags.FLAGS


def main(argv):
    del argv

    import numpy as np
    from botwire.datasets import EpisodeWriter
    from botwire.envs import BraxEnv

    env = BraxEnv(env_name=FLAGS.env, episode_length=FLAGS.max_steps_per_episode)
    rng = jax.random.PRNGKey(FLAGS.seed)

    logging.info(
        "Collecting %d episodes from '%s' → '%s'",
        FLAGS.episodes,
        FLAGS.env,
        FLAGS.output,
    )

    episode_returns = []

    with EpisodeWriter(FLAGS.output, shard_size=50) as writer:
        for ep in range(FLAGS.episodes):
            rng, reset_rng = jax.random.split(rng)
            state, obs = env.reset(reset_rng)

            episode_return = 0.0
            step_count = 0
            writer.add_step(
                observation={k: np.array(v) for k, v in obs.items()},
                action=np.zeros(env.action_dim),
                reward=0.0,
                is_first=True,
            )

            for _ in range(FLAGS.max_steps_per_episode):
                rng, act_rng = jax.random.split(rng)
                # Random policy for demonstration
                action = jax.random.uniform(
                    act_rng, (env.action_dim,), minval=-1.0, maxval=1.0
                )

                state, next_obs, reward, done, _ = env.step(state, action)
                episode_return += float(reward)
                step_count += 1

                writer.add_step(
                    observation={k: np.array(v) for k, v in next_obs.items()},
                    action=np.array(action),
                    reward=float(reward),
                    is_terminal=bool(done),
                )
                obs = next_obs

                if bool(done):
                    break

            writer.end_episode(metadata={"return": episode_return, "length": step_count})
            episode_returns.append(episode_return)

            if (ep + 1) % 5 == 0:
                logging.info(
                    "Episode %d/%d | return=%.2f | steps=%d",
                    ep + 1,
                    FLAGS.episodes,
                    episode_return,
                    step_count,
                )

    import numpy as np
    logging.info(
        "Collection complete: %d episodes, mean return=%.2f, saved to '%s'",
        FLAGS.episodes,
        float(np.mean(episode_returns)),
        FLAGS.output,
    )
    logging.info("Load with: from botwire.datasets import rlds_load; ds = rlds_load('%s')", FLAGS.output)


if __name__ == "__main__":
    app.run(main)
