"""Imitation-learning policy configurations."""

import ml_collections


def get_act_config() -> ml_collections.ConfigDict:
    """Action Chunking with Transformers (ACT) configuration."""
    config = ml_collections.ConfigDict()

    config.name = "act"

    # Transformer architecture
    config.hidden_dim = 512
    config.num_heads = 8
    config.num_encoder_layers = 4
    config.num_decoder_layers = 7
    config.feedforward_dim = 3200
    config.dropout = 0.1

    # CVAE (conditional VAE) latent space
    config.latent_dim = 32
    config.kl_weight = 10.0             # weight of the KL divergence term

    # Temporal chunking
    config.chunk_size = 100             # actions predicted per forward pass
    config.temporal_agg = True          # temporal ensembling at inference

    # Vision encoder (ResNet-18 by default)
    config.vision_backbone = "resnet18"
    config.image_size = (224, 224)

    # Training
    config.learning_rate = 1e-5
    config.weight_decay = 1e-4
    config.grad_clip = 10.0

    return config


def get_diffusion_config() -> ml_collections.ConfigDict:
    """Diffusion Policy (DDPM) configuration."""
    config = ml_collections.ConfigDict()

    config.name = "diffusion"

    # Noise schedule
    config.num_diffusion_steps = 100    # training steps
    config.num_inference_steps = 10     # DDIM steps at inference
    config.beta_schedule = "squaredcos_cap_v2"
    config.beta_start = 0.0001
    config.beta_end = 0.02
    config.clip_sample = True

    # U-Net denoiser
    config.down_dims = (256, 512, 1024)
    config.kernel_size = 5
    config.n_groups = 8

    # Observation conditioning
    config.obs_horizon = 2              # number of past obs to condition on
    config.action_horizon = 8           # actions predicted per step
    config.prediction_horizon = 16      # denoising window

    # Training
    config.learning_rate = 1e-4
    config.weight_decay = 1e-6
    config.grad_clip = 1.0
    config.ema_decay = 0.75

    return config


def get_vla_config() -> ml_collections.ConfigDict:
    """VLA policy backed by Gemini Robotics SDK."""
    config = ml_collections.ConfigDict()

    config.name = "vla"

    # Gemini model to use
    config.model_id = "gemini-robotics-er-2.0"

    # Action chunking
    config.action_horizon = 4

    # Thinking budget (0 = fast, higher = more reasoning)
    config.thinking_budget = 0

    # Temperature for action sampling
    config.temperature = 0.0           # 0 = deterministic

    # Whether to use streaming inference
    config.stream = False

    return config
