"""botwire: Full-stack robotics infrastructure on Google DeepMind's open source stack.

Provides a unified, composable framework for:
- Data collection and management (RLDS / Open X-Embodiment)
- Physics simulation (Brax + MuJoCo Menagerie + dm_robotics MoMa)
- Policy learning (imitation learning: ACT, Diffusion; RL: Acme SAC/D4PG)
- VLA deployment (Gemini Robotics SDK)
- HuggingFace Hub integration for model/dataset sharing

Quick start:
    from botwire.datasets import oxe_load
    from botwire.envs import BraxEnv
    from botwire.policies.act import ACTPolicy

Example (RL training):
    import jax
    from botwire.envs import BraxEnv
    from botwire.agents import SACAgent

    env = BraxEnv(env_name="ant")
    agent = SACAgent.from_config(env.observation_spec, env.action_spec)
    # ... training loop
"""

__version__ = "0.1.0"

from botwire.common.abc import (
    BotwireAgent,
    BotwireDataset,
    BotwireEnv,
    BotwirePolicy,
    RobotInterface,
)

__all__ = [
    "__version__",
    "BotwirePolicy",
    "BotwireEnv",
    "BotwireDataset",
    "BotwireAgent",
    "RobotInterface",
]
