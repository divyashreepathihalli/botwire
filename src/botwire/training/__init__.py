"""Training infrastructure for botwire.

Provides training loops for both imitation learning (IL) and
reinforcement learning (RL) that work with any BotwirePolicy/BotwireAgent
and BotwireEnv.

Usage (IL):
    from botwire.training import ImitationLearningTrainer
    from botwire.datasets import oxe_load
    from botwire.policies.act import ACTPolicy
    from botwire.configs import get_base_train_config

    config = get_base_train_config()
    dataset = oxe_load("fractal20220817_data")
    policy = ACTPolicy(config.policy, action_dim=7)
    params = policy.init(rng, obs_spec)

    trainer = ImitationLearningTrainer(policy, dataset, config)
    trainer.train()

Usage (RL):
    from botwire.training import RLTrainer
    from botwire.envs import BraxEnv
    from botwire.agents import SACAgent

    env = BraxEnv("ant")
    agent = SACAgent(env.observation_spec, env.action_spec, config.agent)
    trainer = RLTrainer(agent, env, config)
    trainer.train()
"""

from botwire.training.il_trainer import ImitationLearningTrainer
from botwire.training.rl_trainer import RLTrainer

__all__ = ["ImitationLearningTrainer", "RLTrainer"]
