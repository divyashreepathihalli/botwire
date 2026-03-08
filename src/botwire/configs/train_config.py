"""Top-level training configuration (composes all sub-configs)."""

import ml_collections

from botwire.configs.agent_config import get_sac_config
from botwire.configs.dataset_config import get_dataset_config
from botwire.configs.env_config import get_brax_env_config
from botwire.configs.policy_config import get_act_config


def get_base_train_config() -> ml_collections.ConfigDict:
    """Return a default training configuration for IL with ACT policy.

    Override individual fields before passing to a trainer:

        config = get_base_train_config()
        config.dataset.name = "bridge_dataset"
        config.policy.learning_rate = 5e-5
    """
    config = ml_collections.ConfigDict()

    # Training mode: "il" (imitation learning) or "rl" (reinforcement learning)
    config.mode = "il"

    # Sub-configs
    config.dataset = get_dataset_config()
    config.env = get_brax_env_config()
    config.policy = get_act_config()
    config.agent = get_sac_config()       # used only in RL mode

    # Hardware
    config.seed = 42
    config.num_devices = 1               # data-parallel devices; 0 = auto

    # Training loop
    config.num_steps = 100_000           # total gradient steps (IL) or env steps (RL)
    config.log_every = 100
    config.eval_every = 5_000
    config.save_every = 10_000
    config.keep_checkpoints = 3

    # Output directories
    config.output_dir = "./checkpoints"
    config.log_dir = "./logs"

    # Observability
    config.use_wandb = False
    config.wandb_project = "botwire"
    config.wandb_entity = ""
    config.use_tensorboard = False

    # HuggingFace Hub (set to push after training)
    config.push_to_hub = False
    config.hub_repo = ""                 # e.g. "my-org/franka-reach-act"

    return config


def get_rl_train_config() -> ml_collections.ConfigDict:
    """Return a default training configuration for RL with SAC + BraxEnv."""
    config = get_base_train_config()
    config.mode = "rl"
    config.num_steps = 1_000_000
    return config
