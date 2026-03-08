"""botwire-eval: Evaluate a trained policy on an environment.

Usage:
    botwire-eval --policy_checkpoint=./checkpoints/latest \\
                 --env=ant \\
                 --episodes=20
"""

from absl import app, flags, logging
import jax

FLAGS = flags.FLAGS
flags.DEFINE_string("policy_checkpoint", "", "Path to Orbax checkpoint directory.")
flags.DEFINE_string("env", "ant", "Environment name.")
flags.DEFINE_integer("episodes", 10, "Number of evaluation episodes.")
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_boolean("render", False, "Render episodes (requires display).")


def main(argv):
    del argv

    from botwire.envs import BraxEnv
    from botwire.envs.utils import evaluate_policy

    env = BraxEnv(env_name=FLAGS.env)
    rng = jax.random.PRNGKey(FLAGS.seed)

    # Load policy
    import orbax.checkpoint as ocp
    from botwire.policies.act import ACTPolicy
    from botwire.configs import get_act_config

    checkpointer = ocp.PyTreeCheckpointer()
    state = checkpointer.restore(FLAGS.policy_checkpoint)
    params = state["params"]
    policy = ACTPolicy(config=get_act_config(), action_dim=env.action_dim)

    def policy_fn(p, obs, rng, *, deterministic=True):
        return policy.select_action(p, obs, rng, deterministic=deterministic)

    metrics = evaluate_policy(
        env,
        lambda obs, rng, **kw: policy_fn(params, obs, rng, **kw),
        params=None,  # already captured in closure
        rng=rng,
        num_episodes=FLAGS.episodes,
    )

    logging.info("=== Evaluation Results (%d episodes) ===", FLAGS.episodes)
    for k, v in metrics.items():
        logging.info("  %s: %.4f", k, v)


if __name__ == "__main__":
    app.run(main)
