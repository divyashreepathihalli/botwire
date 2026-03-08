"""Tests for botwire policy models."""

import jax
import jax.numpy as jnp
import pytest

from botwire.configs import get_act_config, get_diffusion_config


@pytest.mark.parametrize("action_dim,proprio_dim", [(7, 14), (6, 12)])
def test_act_policy_init_and_forward(action_dim, proprio_dim):
    from botwire.policies.act import ACTPolicy

    config = get_act_config()
    config.chunk_size = 10  # small for testing
    config.hidden_dim = 64
    config.num_encoder_layers = 1
    config.num_decoder_layers = 1
    config.num_heads = 4
    config.feedforward_dim = 128

    policy = ACTPolicy(config=config, action_dim=action_dim, proprio_dim=proprio_dim)

    rng = jax.random.PRNGKey(0)
    obs_spec = {"proprio": (proprio_dim,)}

    # Init
    params = policy.init(rng, obs_spec)
    assert params is not None

    # Forward pass
    obs = {"proprio": jnp.zeros((1, proprio_dim))}
    predicted, mu, log_var = policy.apply(params, obs, train=False)
    assert predicted.shape == (1, config.chunk_size, action_dim)
    assert mu.shape == (1, config.latent_dim)


def test_act_policy_select_action():
    from botwire.policies.act import ACTPolicy

    config = get_act_config()
    config.chunk_size = 5
    config.hidden_dim = 32
    config.num_encoder_layers = 1
    config.num_decoder_layers = 1
    config.num_heads = 2
    config.feedforward_dim = 64

    policy = ACTPolicy(config=config, action_dim=7)
    rng = jax.random.PRNGKey(0)
    params = policy.init(rng, {"proprio": (14,)})
    obs = {"proprio": jnp.zeros(14)}
    action = policy.select_action(params, obs, rng)
    assert action.shape == (7,)


def test_act_policy_loss():
    from botwire.policies.act import ACTPolicy

    config = get_act_config()
    config.chunk_size = 4
    config.hidden_dim = 32
    config.num_encoder_layers = 1
    config.num_decoder_layers = 1
    config.num_heads = 2
    config.feedforward_dim = 64

    policy = ACTPolicy(config=config, action_dim=7)
    rng = jax.random.PRNGKey(0)
    params = policy.init(rng, {"proprio": (14,)})

    batch = {
        "observation": {"proprio": jnp.zeros((4, 14))},
        "action": jnp.zeros((4, config.chunk_size, 7)),
    }
    loss, metrics = policy.loss(params, batch, rng)
    assert loss.shape == ()
    assert "loss" in metrics
    assert float(loss) >= 0


def test_diffusion_scheduler():
    from botwire.policies.diffusion.noise_scheduler import DDPMScheduler, DDIMScheduler

    scheduler = DDPMScheduler(num_train_steps=100)
    x0 = jnp.ones((2, 10, 7))
    noise = jnp.zeros_like(x0)
    t = jnp.array([0, 50])
    noisy = scheduler.add_noise(x0, noise, t)
    assert noisy.shape == x0.shape

    infer_scheduler = DDIMScheduler(num_train_steps=100, num_inference_steps=10)
    assert len(infer_scheduler.timesteps) == 10


def test_diffusion_policy_init():
    from botwire.policies.diffusion import DiffusionPolicy

    config = get_diffusion_config()
    config.down_dims = (32, 64)
    config.num_diffusion_steps = 10
    config.num_inference_steps = 3
    config.prediction_horizon = 4

    policy = DiffusionPolicy(config=config, action_dim=7, proprio_dim=14)
    rng = jax.random.PRNGKey(0)
    params = policy.init(rng, {"proprio": (14,)})
    assert params is not None
