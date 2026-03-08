"""botwire-train: Train a policy (IL or RL) from the command line.

Usage:
    # Imitation learning with ACT policy on an OXE dataset
    botwire-train --mode=il \\
                  --dataset.name=fractal20220817_data \\
                  --policy.name=act \\
                  --num_steps=100000

    # Reinforcement learning with SAC on Brax Ant
    botwire-train --mode=rl \\
                  --env.env_name=ant \\
                  --agent.name=sac \\
                  --num_steps=1000000

    # Override any config field:
    botwire-train --mode=il --dataset.batch_size=128 --policy.learning_rate=1e-4
"""

from absl import app, flags, logging
import jax
import ml_collections
from ml_collections import config_flags

from botwire.configs.train_config import get_base_train_config, get_rl_train_config

FLAGS = flags.FLAGS
flags.DEFINE_string("mode", "il", "Training mode: 'il' or 'rl'.")
flags.DEFINE_string("output_dir", "./checkpoints", "Output directory for checkpoints.")
flags.DEFINE_integer("seed", 42, "Random seed.")


def main(argv):
    del argv  # unused

    if FLAGS.mode == "il":
        _train_il()
    elif FLAGS.mode == "rl":
        _train_rl()
    else:
        raise ValueError(f"Unknown mode '{FLAGS.mode}'. Use 'il' or 'rl'.")


def _train_il():
    """Run imitation learning training."""
    from botwire.configs.train_config import get_base_train_config
    from botwire.datasets import oxe_load, rlds_load
    from botwire.policies.act import ACTPolicy
    from botwire.policies.diffusion import DiffusionPolicy
    from botwire.training import ImitationLearningTrainer

    config = get_base_train_config()
    config.seed = FLAGS.seed
    config.output_dir = FLAGS.output_dir

    logging.info("=== botwire-train (IL) ===")
    logging.info("Dataset: %s", config.dataset.name)
    logging.info("Policy: %s", config.policy.name)
    logging.info("Steps: %d", config.num_steps)

    # Load dataset
    if config.dataset.name == "local":
        dataset = rlds_load(config.dataset.local_path)
    else:
        dataset = oxe_load(
            config.dataset.name,
            split=config.dataset.split,
            image_size=tuple(config.dataset.image_size),
        )
    logging.info("Dataset: %s", dataset)

    # Build policy
    policy_name = config.policy.name
    if policy_name == "act":
        policy = ACTPolicy(config=config.policy, action_dim=7)
    elif policy_name == "diffusion":
        policy = DiffusionPolicy(config=config.policy, action_dim=7)
    else:
        raise ValueError(f"Unknown policy '{policy_name}'. Choose 'act' or 'diffusion'.")

    # Observation spec (will be read from dataset in future)
    obs_spec = {"proprio": (14,)}

    # Train
    trainer = ImitationLearningTrainer(policy, dataset, config)
    rng = jax.random.PRNGKey(config.seed)
    rng, init_rng = jax.random.split(rng)
    params = policy.init(init_rng, obs_spec)
    trainer.train(params=params, observation_spec=obs_spec, rng=rng)


def _train_rl():
    """Run reinforcement learning training."""
    from botwire.agents import SACAgent
    from botwire.configs.train_config import get_rl_train_config
    from botwire.envs import BraxEnv
    from botwire.training import RLTrainer

    config = get_rl_train_config()
    config.seed = FLAGS.seed
    config.output_dir = FLAGS.output_dir

    logging.info("=== botwire-train (RL) ===")
    logging.info("Environment: %s", config.env.env_name)
    logging.info("Agent: %s", config.agent.name)
    logging.info("Steps: %d", config.num_steps)

    env = BraxEnv(
        env_name=config.env.env_name,
        backend=config.env.backend,
        episode_length=config.env.episode_length,
    )

    agent_name = config.agent.name
    if agent_name == "sac":
        agent = SACAgent(env.observation_spec, env.action_spec, config.agent)
    else:
        raise ValueError(f"Unknown agent '{agent_name}'. Choose 'sac'.")

    trainer = RLTrainer(agent, env, config)
    rng = jax.random.PRNGKey(config.seed)
    trainer.train(rng=rng)


if __name__ == "__main__":
    app.run(main)
