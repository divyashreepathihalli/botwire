"""ml_collections-based configuration system for botwire.

All training runs are driven by a ConfigDict hierarchy. This makes
configs serializable, diffable, and overridable from the command line
via absl flags.

Usage:
    from botwire.configs import get_base_train_config

    config = get_base_train_config()
    config.agent.learning_rate = 3e-4   # override
"""

from botwire.configs.agent_config import get_d4pg_config, get_sac_config
from botwire.configs.dataset_config import get_dataset_config
from botwire.configs.env_config import get_brax_env_config, get_moma_env_config
from botwire.configs.policy_config import (
    get_act_config,
    get_diffusion_config,
    get_vla_config,
)
from botwire.configs.train_config import get_base_train_config

__all__ = [
    "get_base_train_config",
    "get_dataset_config",
    "get_brax_env_config",
    "get_moma_env_config",
    "get_act_config",
    "get_diffusion_config",
    "get_vla_config",
    "get_sac_config",
    "get_d4pg_config",
]
