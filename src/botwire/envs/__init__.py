"""Simulation environment wrappers for botwire.

Provides two concrete environment backends:

1. **BraxEnv** — JAX-native fast simulation via Google's Brax engine.
   Supports MuJoCo Menagerie robot models via the MJX backend.
   Best for RL training at massive scale (millions of steps/sec on GPU/TPU).

2. **MoMaEnv** — Manipulation-focused simulation via dm_robotics MoMa.
   Better suited for precise sim-to-real transfer on real robot tasks.

Both environments implement the :class:`BotwireEnv` interface so that
policies and training loops are backend-agnostic.

Usage:
    from botwire.envs import BraxEnv, MoMaEnv, make_env

    # Brax built-in environment
    env = BraxEnv(env_name="ant")

    # MuJoCo Menagerie robot via Brax/MJX
    env = BraxEnv(env_name="franka_fr3/reach")

    # dm_robotics MoMa environment
    env = MoMaEnv(robot="franka_fr3", scene="tabletop")

    # Functional interface (JIT-compatible)
    state, obs = env.reset(jax.random.PRNGKey(0))
    state, obs, reward, done, info = env.step(state, action)
"""

from botwire.envs.brax_env import BraxEnv
from botwire.envs.menagerie import MENAGERIE_REGISTRY, load_menagerie_model
from botwire.envs.moma_env import MoMaEnv
from botwire.envs.wrappers import AutoReset, EpisodeLimit, ObsNormalize


def make_env(name: str, backend: str = "mjx", **kwargs) -> "BraxEnv | MoMaEnv":
    """Create an environment by name.

    Dispatches to BraxEnv for names in the Brax/Menagerie registry, or
    raises an error for unknown names.

    Args:
        name: Environment name, e.g. "ant", "franka_fr3/reach".
        backend: Physics backend for BraxEnv ("mjx", "generalized", etc.).
        **kwargs: Additional keyword arguments passed to the environment.

    Returns:
        A :class:`BotwireEnv` instance.
    """
    if name in MENAGERIE_REGISTRY or "/" in name:
        return BraxEnv(env_name=name, backend=backend, **kwargs)
    return BraxEnv(env_name=name, backend=backend, **kwargs)


__all__ = [
    "BraxEnv",
    "MoMaEnv",
    "AutoReset",
    "EpisodeLimit",
    "ObsNormalize",
    "MENAGERIE_REGISTRY",
    "load_menagerie_model",
    "make_env",
]
