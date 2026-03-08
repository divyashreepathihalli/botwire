"""Diffusion Policy — Denoising Diffusion for robot action prediction.

Based on: "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"
(Chi et al., 2023) — https://arxiv.org/abs/2303.04137

A 1D U-Net denoiser is conditioned on current observations and trained to
reverse a DDPM noise process over action trajectories.

Advantages over ACT:
- Handles multi-modal action distributions naturally
- No explicit latent bottleneck (CVAE)
- Better on tasks with multiple valid action modes

Trade-off:
- Slower inference (requires multiple denoising steps)
"""

from __future__ import annotations

from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections

from botwire.common.types import Action, Batch, Metrics, Observation, Params, RNGKey
from botwire.policies.base import FlaxPolicy
from botwire.policies.diffusion.noise_scheduler import DDIMScheduler, DDPMScheduler


# ---------------------------------------------------------------------------
# 1D U-Net denoiser
# ---------------------------------------------------------------------------


class ResBlock1D(nn.Module):
    """1D residual block with GroupNorm."""

    out_channels: int
    kernel_size: int = 5
    n_groups: int = 8

    @nn.compact
    def __call__(self, x: jax.Array, cond: jax.Array | None = None) -> jax.Array:
        # x: (B, T, C)
        h = nn.GroupNorm(num_groups=self.n_groups)(x)
        h = nn.relu(h)
        h = nn.Conv(self.out_channels, (self.kernel_size,), padding="SAME")(h)

        if cond is not None:
            # Add conditioning via FiLM
            scale = nn.Dense(self.out_channels)(cond)[:, None, :]
            shift = nn.Dense(self.out_channels)(cond)[:, None, :]
            h = h * (1 + scale) + shift

        h = nn.GroupNorm(num_groups=self.n_groups)(h)
        h = nn.relu(h)
        h = nn.Conv(self.out_channels, (self.kernel_size,), padding="SAME")(h)

        # Skip connection
        if x.shape[-1] != self.out_channels:
            x = nn.Conv(self.out_channels, (1,))(x)
        return x + h


class UNet1D(nn.Module):
    """1D U-Net for denoising action sequences.

    Args:
        action_dim: Dimensionality of the action space.
        down_dims: Channel sizes at each downsampling level.
        kernel_size: Conv kernel size.
        n_groups: Groups for GroupNorm.
    """

    action_dim: int = 7
    down_dims: tuple[int, ...] = (256, 512, 1024)
    kernel_size: int = 5
    n_groups: int = 8

    @nn.compact
    def __call__(
        self,
        x_t: jax.Array,       # (B, T, action_dim) — noisy actions
        t_emb: jax.Array,     # (B, d_time) — timestep embedding
        cond: jax.Array,       # (B, cond_dim) — observation conditioning
    ) -> jax.Array:
        cond_emb = jnp.concatenate([t_emb, cond], axis=-1)

        # Initial projection
        h = nn.Conv(self.down_dims[0], (self.kernel_size,), padding="SAME")(x_t)

        # Encoder (downsampling)
        skips = []
        for dim in self.down_dims:
            h = ResBlock1D(dim, self.kernel_size, self.n_groups)(h, cond_emb)
            h = ResBlock1D(dim, self.kernel_size, self.n_groups)(h, cond_emb)
            skips.append(h)
            h = nn.avg_pool(h, window_shape=(2,), strides=(2,), padding="SAME")

        # Bottleneck
        h = ResBlock1D(self.down_dims[-1], self.kernel_size, self.n_groups)(h, cond_emb)
        h = ResBlock1D(self.down_dims[-1], self.kernel_size, self.n_groups)(h, cond_emb)

        # Decoder (upsampling)
        for dim, skip in zip(reversed(self.down_dims), reversed(skips)):
            # Upsample
            B, T, C = h.shape
            h = jax.image.resize(h, (B, T * 2, C), method="nearest")
            h = jnp.concatenate([h, skip], axis=-1)
            h = ResBlock1D(dim, self.kernel_size, self.n_groups)(h, cond_emb)
            h = ResBlock1D(dim, self.kernel_size, self.n_groups)(h, cond_emb)

        # Final projection
        h = nn.GroupNorm(num_groups=self.n_groups)(h)
        h = nn.relu(h)
        return nn.Conv(self.action_dim, (1,))(h)


def sinusoidal_timestep_embedding(t: jax.Array, dim: int = 256) -> jax.Array:
    """Sinusoidal timestep embedding."""
    half = dim // 2
    freqs = jnp.exp(-jnp.log(10000) * jnp.arange(half) / (half - 1))
    args = t[:, None].astype(jnp.float32) * freqs[None]
    embedding = jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)
    return embedding


# ---------------------------------------------------------------------------
# Full Diffusion Policy
# ---------------------------------------------------------------------------


class DiffusionPolicy(FlaxPolicy):
    """Denoising Diffusion Policy.

    Args:
        config: ml_collections.ConfigDict from get_diffusion_config().
        action_dim: Dimensionality of the robot's action space.
        proprio_dim: Dimensionality of proprioceptive observations.
    """

    config: ml_collections.ConfigDict
    action_dim: int = 7
    proprio_dim: int = 14

    def setup(self) -> None:
        cfg = self.config
        self.unet = UNet1D(
            action_dim=self.action_dim,
            down_dims=tuple(cfg.down_dims),
            kernel_size=cfg.kernel_size,
            n_groups=cfg.n_groups,
        )
        self.obs_encoder = nn.Dense(256)  # encode proprio observations
        self.train_scheduler = DDPMScheduler(
            num_train_steps=cfg.num_diffusion_steps,
            schedule=cfg.beta_schedule,
            beta_start=cfg.beta_start,
            beta_end=cfg.beta_end,
            clip_sample=cfg.clip_sample,
        )
        self.infer_scheduler = DDIMScheduler(
            num_train_steps=cfg.num_diffusion_steps,
            num_inference_steps=cfg.num_inference_steps,
            schedule=cfg.beta_schedule,
            beta_start=cfg.beta_start,
            beta_end=cfg.beta_end,
            clip_sample=cfg.clip_sample,
        )

    def __call__(
        self,
        x_t: jax.Array,           # (B, T, action_dim) — noisy actions
        t: jax.Array,              # (B,) — timestep indices
        observation: Observation,
        train: bool = True,
    ) -> jax.Array:
        """Predict noise from noisy actions, timestep, and observation."""
        t_emb = sinusoidal_timestep_embedding(t, dim=256)
        proprio = observation.get("proprio", jnp.zeros((x_t.shape[0], self.proprio_dim)))
        obs_emb = nn.relu(self.obs_encoder(proprio))
        return self.unet(x_t, t_emb, obs_emb)

    def loss(
        self,
        params: Params,
        batch: Batch,
        rng: RNGKey,
    ) -> tuple[jax.Array, Metrics]:
        """DDPM training loss: MSE between predicted and actual noise."""
        obs = batch["observation"]
        actions = batch["action"]     # (B, T, action_dim)
        B, T, D = actions.shape

        rng, t_rng, noise_rng = jax.random.split(rng, 3)

        # Sample random timesteps
        t = jax.random.randint(t_rng, (B,), 0, self.config.num_diffusion_steps)
        noise = jax.random.normal(noise_rng, actions.shape)

        # Forward diffusion
        x_t = self.train_scheduler.add_noise(actions, noise, t)

        # Predict noise
        pred_noise = self.apply(
            params,
            x_t,
            t,
            obs,
            train=True,
            rngs={"dropout": rng},
        )

        loss = jnp.mean((pred_noise - noise) ** 2)
        return loss, {"loss": loss}

    def select_action(
        self,
        params: Params,
        observation: Observation,
        rng: RNGKey,
        *,
        deterministic: bool = False,
    ) -> Action:
        """Denoise from Gaussian noise to get an action chunk, return first action."""
        B = 1
        T = self.config.prediction_horizon
        D = self.action_dim

        rng, noise_rng = jax.random.split(rng)
        x_t = jax.random.normal(noise_rng, (B, T, D))

        # DDIM denoising loop
        for t_val in self.infer_scheduler.timesteps:
            t = jnp.array([t_val])
            pred_noise = self.apply(params, x_t, t, observation, train=False)
            x_t = self.infer_scheduler.step(pred_noise[0], int(t_val), x_t[0])[None]

        if self.config.clip_sample:
            x_t = jnp.clip(x_t, -1.0, 1.0)

        return x_t[0, 0]  # first batch, first timestep

    def init(
        self,
        rng: RNGKey,
        observation_spec: dict[str, tuple[int, ...]],
    ) -> Params:
        proprio_dim = observation_spec.get("proprio", (self.proprio_dim,))[0]
        dummy_obs = {"proprio": jnp.zeros((1, proprio_dim))}
        dummy_x_t = jnp.zeros((1, self.config.prediction_horizon, self.action_dim))
        dummy_t = jnp.zeros((1,), dtype=jnp.int32)
        rng, init_rng = jax.random.split(rng)
        return self.apply(
            {},
            dummy_x_t,
            dummy_t,
            dummy_obs,
            train=True,
            rngs={"params": rng},
        )
