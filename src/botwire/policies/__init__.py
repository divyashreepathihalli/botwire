"""Imitation learning policy models for botwire.

All policies are implemented as Flax ``nn.Module`` subclasses and conform
to the :class:`~botwire.common.abc.BotwirePolicy` interface.

Available policies:
- **ACTPolicy** — Action Chunking with Transformers (CVAE + transformer).
  Best for manipulation tasks. Predicts ``chunk_size`` actions at once.
- **DiffusionPolicy** — Denoising Diffusion Policy (DDPM/DDIM).
  Handles multi-modal action distributions. Slower at inference.
- **VLAPolicy** — Vision-Language-Action policy backed by Gemini Robotics SDK.
  Enables natural-language task specification and open-vocabulary generalization.

Usage:
    from botwire.policies.act import ACTPolicy
    from botwire.configs import get_act_config

    config = get_act_config()
    policy = ACTPolicy(config)
    params = policy.init(rng, observation_spec)
    action = policy.select_action(params, obs, rng)
"""

from botwire.policies.act.modeling import ACTPolicy
from botwire.policies.diffusion.modeling import DiffusionPolicy

__all__ = [
    "ACTPolicy",
    "DiffusionPolicy",
]

try:
    from botwire.policies.vla.modeling import VLAPolicy

    __all__.append("VLAPolicy")
except ImportError:
    pass  # Gemini Robotics SDK not installed
