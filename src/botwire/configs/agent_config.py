"""RL agent configurations."""

import ml_collections


def get_sac_config() -> ml_collections.ConfigDict:
    """Soft Actor-Critic (SAC) configuration for Acme JAX agent."""
    config = ml_collections.ConfigDict()

    config.name = "sac"

    # Network architecture (shared for actor + critic)
    config.hidden_sizes = (256, 256)
    config.activation = "relu"

    # SAC hyperparameters
    config.learning_rate = 3e-4
    config.discount = 0.99
    config.tau = 0.005                  # target network soft-update rate
    config.init_alpha = 1.0             # entropy temperature initial value
    config.target_entropy = "auto"      # "auto" = -action_dim

    # Replay buffer
    config.replay_capacity = 1_000_000
    config.min_replay_size = 10_000
    config.batch_size = 256
    config.num_sgd_steps_per_step = 1   # gradient steps per env step

    # Training schedule
    config.num_steps = 1_000_000
    config.eval_every = 10_000
    config.eval_episodes = 10

    return config


def get_d4pg_config() -> ml_collections.ConfigDict:
    """Distributed Distributional DDPG (D4PG) configuration."""
    config = ml_collections.ConfigDict()

    config.name = "d4pg"

    # Network architecture
    config.policy_hidden_sizes = (256, 256)
    config.critic_hidden_sizes = (256, 256)

    # D4PG hyperparameters
    config.learning_rate = 1e-4
    config.discount = 0.99
    config.target_update_period = 100
    config.sigma = 0.2                  # exploration noise

    # Distributional critic
    config.num_atoms = 51
    config.v_min = -150.0
    config.v_max = 150.0

    # N-step returns
    config.n_step = 5

    # Replay buffer
    config.replay_capacity = 1_000_000
    config.min_replay_size = 10_000
    config.batch_size = 256

    # Training schedule
    config.num_steps = 1_000_000
    config.eval_every = 10_000
    config.eval_episodes = 10

    return config
