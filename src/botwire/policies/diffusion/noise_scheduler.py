"""DDPM/DDIM noise schedulers implemented in JAX.

Reference: "Denoising Diffusion Probabilistic Models" (Ho et al., 2020)
           "Denoising Diffusion Implicit Models" (Song et al., 2021)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def make_beta_schedule(
    schedule: str = "squaredcos_cap_v2",
    num_steps: int = 100,
    beta_start: float = 0.0001,
    beta_end: float = 0.02,
) -> jax.Array:
    """Create a beta schedule for the diffusion process.

    Args:
        schedule: One of "linear", "cosine", "squaredcos_cap_v2".
        num_steps: Number of diffusion timesteps.
        beta_start: Starting beta value (for linear schedule).
        beta_end: Ending beta value (for linear schedule).

    Returns:
        Beta array of shape (num_steps,).
    """
    if schedule == "linear":
        return jnp.linspace(beta_start, beta_end, num_steps)

    if schedule in ("cosine", "squaredcos_cap_v2"):
        # Cosine schedule from Improved DDPM (Nichol & Dhariwal, 2021)
        steps = num_steps + 1
        t = jnp.linspace(0, num_steps, steps) / num_steps
        alphas_cumprod = jnp.cos((t + 0.008) / 1.008 * jnp.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return jnp.clip(betas, 0.0001, 0.9999)

    raise ValueError(f"Unknown beta schedule: '{schedule}'")


class DDPMScheduler:
    """DDPM (training) noise scheduler.

    Computes the forward noising process q(x_t | x_0) and the
    denoising loss target.
    """

    def __init__(
        self,
        num_train_steps: int = 100,
        schedule: str = "squaredcos_cap_v2",
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        clip_sample: bool = True,
    ) -> None:
        self.num_train_steps = num_train_steps
        self.clip_sample = clip_sample

        betas = make_beta_schedule(schedule, num_train_steps, beta_start, beta_end)
        alphas = 1.0 - betas
        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = jnp.cumprod(alphas)
        self.sqrt_alphas_cumprod = jnp.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = jnp.sqrt(1.0 - self.alphas_cumprod)

    def add_noise(
        self,
        x_0: jax.Array,
        noise: jax.Array,
        timesteps: jax.Array,
    ) -> jax.Array:
        """Forward diffusion: q(x_t | x_0) = sqrt(ᾱ_t) x_0 + sqrt(1 - ᾱ_t) ε."""
        sqrt_alpha = self.sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[timesteps]
        # Broadcast over batch
        while sqrt_alpha.ndim < x_0.ndim:
            sqrt_alpha = sqrt_alpha[..., None]
            sqrt_one_minus = sqrt_one_minus[..., None]
        return sqrt_alpha * x_0 + sqrt_one_minus * noise

    def get_velocity(self, x_0: jax.Array, noise: jax.Array, t: jax.Array) -> jax.Array:
        """v-parameterization velocity target."""
        alpha = self.sqrt_alphas_cumprod[t]
        sigma = self.sqrt_one_minus_alphas_cumprod[t]
        while alpha.ndim < x_0.ndim:
            alpha = alpha[..., None]
            sigma = sigma[..., None]
        return alpha * noise - sigma * x_0


class DDIMScheduler:
    """DDIM (inference) scheduler for fast sampling.

    Allows sampling with far fewer steps than training (e.g. 10 vs 100).
    """

    def __init__(
        self,
        num_train_steps: int = 100,
        num_inference_steps: int = 10,
        schedule: str = "squaredcos_cap_v2",
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        clip_sample: bool = True,
        eta: float = 0.0,  # 0 = deterministic DDIM, 1 = DDPM
    ) -> None:
        self.num_train_steps = num_train_steps
        self.num_inference_steps = num_inference_steps
        self.clip_sample = clip_sample
        self.eta = eta

        betas = make_beta_schedule(schedule, num_train_steps, beta_start, beta_end)
        alphas_cumprod = jnp.cumprod(1.0 - betas)
        self.alphas_cumprod = alphas_cumprod

        # Compute timestep schedule
        step_ratio = num_train_steps // num_inference_steps
        self.timesteps = (jnp.arange(0, num_inference_steps) * step_ratio).round().astype(
            jnp.int32
        )[::-1]

    def step(
        self,
        model_output: jax.Array,     # predicted noise ε_θ(x_t, t)
        timestep: int,
        x_t: jax.Array,
    ) -> jax.Array:
        """Compute x_{t-1} from x_t and predicted noise."""
        t = timestep
        prev_t = t - self.num_train_steps // self.num_inference_steps

        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else jnp.ones(())

        beta_prod_t = 1 - alpha_prod_t
        beta_prod_prev = 1 - alpha_prod_prev

        # Predicted x_0 from noise prediction
        pred_x0 = (x_t - jnp.sqrt(beta_prod_t) * model_output) / jnp.sqrt(alpha_prod_t)
        if self.clip_sample:
            pred_x0 = jnp.clip(pred_x0, -1.0, 1.0)

        # Direction toward x_t
        pred_epsilon = (x_t - jnp.sqrt(alpha_prod_t) * pred_x0) / jnp.sqrt(beta_prod_t)

        # DDIM update
        pred_sample_direction = jnp.sqrt(beta_prod_prev) * pred_epsilon
        prev_sample = jnp.sqrt(alpha_prod_prev) * pred_x0 + pred_sample_direction

        return prev_sample
