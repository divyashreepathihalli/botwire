"""RL agents for botwire, wrapping Acme JAX agents.

All agents implement the :class:`~botwire.common.abc.BotwireAgent` interface
for a consistent select_action / observe / update loop.

Available agents:
- **SACAgent** — Soft Actor-Critic. Sample-efficient, handles continuous actions.
- **D4PGAgent** — Distributed Distributional DDPG. Distributional critic.
- **BCAgent** — Behavioral Cloning. Simple IL baseline using Acme.

Usage:
    from botwire.agents import SACAgent
    from botwire.envs import BraxEnv
    from botwire.configs import get_sac_config

    env = BraxEnv("ant")
    config = get_sac_config()
    agent = SACAgent(
        observation_spec=env.observation_spec,
        action_spec=env.action_spec,
        config=config,
    )

    state, obs = env.reset(rng)
    while True:
        action = agent.select_action(obs, rng)
        state, obs, reward, done, info = env.step(state, action)
        agent.observe(obs, action, reward, next_obs, done)
        metrics = agent.update()
"""

from botwire.agents.bc import BCAgent
from botwire.agents.sac import SACAgent

__all__ = ["SACAgent", "BCAgent"]

try:
    from botwire.agents.d4pg import D4PGAgent

    __all__.append("D4PGAgent")
except ImportError:
    pass  # Acme not installed
