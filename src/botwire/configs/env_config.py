"""Environment configurations."""

import ml_collections


def get_brax_env_config() -> ml_collections.ConfigDict:
    """Return the default BraxEnv configuration."""
    config = ml_collections.ConfigDict()

    # Brax built-in environments: "ant", "humanoid", "halfcheetah", "hopper",
    #   "walker2d", "reacher", "pusher", "fetch", "grasp", "ur5e"
    # MuJoCo Menagerie keys: "franka_fr3/reach", "ur5e/reach", etc.
    config.env_name = "ant"

    # Physics backend: "mjx" (MuJoCo XLA, recommended), "generalized",
    #   "positional", "spring"
    config.backend = "mjx"

    # Number of parallel environments for vectorized rollouts (Brax supports
    # massive parallelism on GPU/TPU)
    config.num_envs = 256

    # Episode length before forced reset
    config.episode_length = 1000

    # Action repeat (frameskip)
    config.action_repeat = 1

    # Whether to auto-reset on episode end
    config.auto_reset = True

    # Observation normalization (running stats)
    config.normalize_obs = True

    return config


def get_moma_env_config() -> ml_collections.ConfigDict:
    """Return the default MoMaEnv (dm_robotics) configuration."""
    config = ml_collections.ConfigDict()

    # Robot model from MuJoCo Menagerie
    config.robot = "franka_fr3"
    config.scene = "tabletop"

    # Control timestep (seconds)
    config.control_timestep = 0.05       # 20 Hz

    # Observation modalities
    config.use_proprio = True
    config.use_rgb = True
    config.use_depth = False
    config.camera_names = ("front_camera", "wrist_camera")
    config.image_size = (84, 84)

    # Episode settings
    config.episode_length = 500
    config.auto_reset = True

    return config
