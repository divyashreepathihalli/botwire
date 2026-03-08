# Botwire

**Full-stack robotics infrastructure on Google DeepMind's open source stack.**

botwire is a JAX-native robotics framework modeled after HuggingFace's [LeRobot](https://github.com/huggingface/lerobot), built entirely on GDM's open source ecosystem.

```
Robot / Sim  →  collect episodes  →  RLDS dataset  →  train policy  →  deploy
                 (EpisodeWriter)    (Open X-Emb.)    (ACT / Diffusion)  (Gemini SDK)
```

## GDM Libraries Integrated

| Library | Role |
|---------|------|
| [Brax](https://github.com/google/brax) | JAX-native physics simulation (PPO/SAC built-in) |
| [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) | 70+ robot MJCF models (Franka, UR5, Spot, humanoids) |
| [dm_robotics](https://github.com/google-deepmind/dm_robotics) | QP controllers, MoMa manipulation environments |
| [Acme](https://github.com/deepmind/acme) | RL agent framework: SAC, D4PG, BC |
| [Open X-Embodiment / RLDS](https://github.com/google-deepmind/open_x_embodiment) | Unified robot datasets via tensorflow-datasets |
| [Gemini Robotics SDK](https://github.com/google-deepmind/gemini-robotics-sdk) | VLM-based deployment and fine-tuning |

## Installation

```bash
# Core (JAX + datasets + HF Hub)
pip install botwire

# With simulation (Brax + MuJoCo Menagerie)
pip install botwire[sim]

# With dm_robotics (QP controllers + MoMa)
pip install botwire[dm]

# With RL agents (Acme + Reverb)
pip install botwire[rl]

# Full (excludes VLA which needs special Trusted Tester access)
pip install botwire[all]

# Development
pip install botwire[all,dev]
```

## Quick Start

### Imitation Learning (ACT on Open X-Embodiment)

```python
import jax
from botwire.datasets import oxe_load
from botwire.policies.act import ACTPolicy
from botwire.configs import get_act_config, get_base_train_config
from botwire.training import ImitationLearningTrainer

# 1. Load an Open X-Embodiment dataset
dataset = oxe_load("fractal20220817_data", split="train[:10%]")

# 2. Build ACT policy
config = get_base_train_config()
policy = ACTPolicy(config.policy, action_dim=7)

# 3. Train
trainer = ImitationLearningTrainer(policy, dataset, config)
params = policy.init(jax.random.PRNGKey(0), {"proprio": (14,)})
trained_params = trainer.train(params=params, observation_spec={"proprio": (14,)})

# 4. Inference
obs = {"proprio": jax.numpy.zeros(14)}
action = policy.select_action(trained_params, obs, jax.random.PRNGKey(1))
```

### Reinforcement Learning (SAC on Brax)

```python
import jax
from botwire.envs import BraxEnv
from botwire.agents import SACAgent
from botwire.configs import get_sac_config, get_rl_train_config
from botwire.training import RLTrainer

env = BraxEnv(env_name="ant", backend="mjx")
agent = SACAgent(env.observation_spec, env.action_spec, get_sac_config())
trainer = RLTrainer(agent, env, get_rl_train_config())
trainer.train(rng=jax.random.PRNGKey(42))
```

### Data Collection

```python
import jax
import numpy as np
from botwire.envs import BraxEnv
from botwire.datasets import EpisodeWriter

env = BraxEnv("ant")
rng = jax.random.PRNGKey(0)

with EpisodeWriter("./my_dataset") as writer:
    for _ in range(100):
        state, obs = env.reset(rng)
        writer.add_step(observation={k: np.array(v) for k, v in obs.items()},
                        action=np.zeros(env.action_dim), reward=0.0, is_first=True)
        for _ in range(500):
            rng, act_rng = jax.random.split(rng)
            action = jax.random.uniform(act_rng, (env.action_dim,), minval=-1., maxval=1.)
            state, obs, reward, done, _ = env.step(state, action)
            writer.add_step(observation={k: np.array(v) for k, v in obs.items()},
                            action=np.array(action), reward=float(reward), is_terminal=bool(done))
            if bool(done):
                break
        writer.end_episode()
```

### VLA with Gemini Robotics SDK

```python
from botwire.policies.vla import VLAPolicy
from botwire.configs import get_vla_config
import jax

config = get_vla_config()
policy = VLAPolicy(config, action_dim=7, task_description="Pick up the red cube.")
action = policy.select_action({}, {"image": image_array, "proprio": joints}, jax.random.PRNGKey(0))
```

## CLI

```bash
# Train (IL)
botwire-train --mode=il --dataset.name=fractal20220817_data --num_steps=100000

# Train (RL)
botwire-train --mode=rl --env.env_name=ant --num_steps=1000000

# Collect dataset
botwire-collect --env=ant --episodes=100 --output=./data

# Evaluate
botwire-eval --policy_checkpoint=./checkpoints/latest --env=ant --episodes=20

# Upload to HuggingFace Hub
botwire-upload --dataset=./data --repo=my-org/ant-dataset-v1
```

## Repository Structure

```
src/botwire/
├── common/          # Abstract base classes, type aliases, utilities
├── configs/         # ml_collections ConfigDicts for all hyperparameters
├── datasets/        # RLDS dataset loading (OXE + local), EpisodeWriter
├── envs/            # BraxEnv, MoMaEnv, MuJoCo Menagerie registry, wrappers
├── policies/
│   ├── act/         # Action Chunking with Transformers (CVAE + transformer)
│   ├── diffusion/   # Diffusion Policy (DDPM/DDIM in JAX)
│   └── vla/         # VLA via Gemini Robotics SDK
├── agents/          # SACAgent, D4PGAgent, BCAgent (Acme JAX)
├── controllers/     # CartesianController (dm_robotics QP)
├── training/        # ImitationLearningTrainer, RLTrainer, checkpointing, logger
├── hub/             # HuggingFace Hub upload/download
└── scripts/         # CLI entry points
```

## Supported Robots (MuJoCo Menagerie)

| Category | Robots |
|----------|--------|
| Arms | Franka FR3, Franka Panda, UR5e, UR10e, KUKA iiwa 14 |
| Quadrupeds | Boston Dynamics Spot, Unitree Go1/Go2, ANYmal C |
| Humanoids | Unitree H1/G1, PAL Talos |
| Mobile | Hello Robot Stretch 3, Google Robot |
| Grippers | Robotiq 2F-85 |
| Drones | Bitcraze Crazyflie 2 |

```python
from botwire.envs.menagerie import list_robots
print(list_robots(category="arm"))
# ['franka_emika_panda', 'franka_fr3', 'kuka_iiwa_14', 'ur10e', 'ur5e']
```

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  Gemini Robotics-ER (VLM reasoning + planning)       │  VLAPolicy
└──────────────────────────┬──────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────┐
│  IL Policies: ACT, DiffusionPolicy (Flax/JAX)       │  policies/
│  RL Agents: SAC, D4PG, BC (Acme JAX backend)        │  agents/
└──────────────────────────┬──────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────┐
│  Simulation: Brax/MJX + MuJoCo Menagerie robots     │  envs/
│  Low-level control: dm_robotics QP controllers       │  controllers/
└──────────────────────────┬──────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────┐
│  Data: RLDS + Open X-Embodiment + EpisodeWriter      │  datasets/
└─────────────────────────────────────────────────────┘
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
