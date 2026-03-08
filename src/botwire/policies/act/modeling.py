"""ACT — Action Chunking with Transformers.

Based on: "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware"
(Zhao et al., 2023) — https://arxiv.org/abs/2304.13705

ACT predicts a chunk of ``chunk_size`` future actions at once using a
transformer encoder-decoder architecture conditioned on a CVAE latent.

Architecture:
    Encoder (CVAE):  [CLS | proprio | image tokens] → μ, σ (latent z)
    Decoder:  [z | image tokens | query tokens] → chunk of actions
"""

from __future__ import annotations

from functools import partial
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax

from botwire.common.types import Action, Batch, Metrics, Observation, Params, RNGKey
from botwire.policies.base import FlaxPolicy


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------


class SinusoidalPE(nn.Module):
    """Sinusoidal positional encoding."""

    max_len: int = 1000
    d_model: int = 512

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        seq_len = x.shape[-2]
        position = jnp.arange(seq_len)[:, None]
        div_term = jnp.exp(jnp.arange(0, self.d_model, 2) * (-jnp.log(10000.0) / self.d_model))
        pe = jnp.zeros((seq_len, self.d_model))
        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
        return x + pe[None]


class TransformerEncoderLayer(nn.Module):
    """Single Transformer encoder layer."""

    hidden_dim: int = 512
    num_heads: int = 8
    feedforward_dim: int = 3200
    dropout: float = 0.1

    @nn.compact
    def __call__(self, x: jax.Array, train: bool = True) -> jax.Array:
        # Self-attention
        attn = nn.MultiHeadDotProductAttention(num_heads=self.num_heads)(x, x)
        attn = nn.Dropout(rate=self.dropout)(attn, deterministic=not train)
        x = nn.LayerNorm()(x + attn)
        # FFN
        ff = nn.Dense(self.feedforward_dim)(x)
        ff = nn.relu(ff)
        ff = nn.Dropout(rate=self.dropout)(ff, deterministic=not train)
        ff = nn.Dense(self.hidden_dim)(ff)
        x = nn.LayerNorm()(x + ff)
        return x


class TransformerDecoderLayer(nn.Module):
    """Single Transformer decoder layer."""

    hidden_dim: int = 512
    num_heads: int = 8
    feedforward_dim: int = 3200
    dropout: float = 0.1

    @nn.compact
    def __call__(
        self,
        tgt: jax.Array,
        memory: jax.Array,
        train: bool = True,
    ) -> jax.Array:
        # Masked self-attention
        attn1 = nn.MultiHeadDotProductAttention(num_heads=self.num_heads)(tgt, tgt)
        attn1 = nn.Dropout(rate=self.dropout)(attn1, deterministic=not train)
        tgt = nn.LayerNorm()(tgt + attn1)
        # Cross-attention
        attn2 = nn.MultiHeadDotProductAttention(num_heads=self.num_heads)(tgt, memory)
        attn2 = nn.Dropout(rate=self.dropout)(attn2, deterministic=not train)
        tgt = nn.LayerNorm()(tgt + attn2)
        # FFN
        ff = nn.Dense(self.feedforward_dim)(tgt)
        ff = nn.relu(ff)
        ff = nn.Dropout(rate=self.dropout)(ff, deterministic=not train)
        ff = nn.Dense(self.hidden_dim)(ff)
        tgt = nn.LayerNorm()(tgt + ff)
        return tgt


class CVAEEncoder(nn.Module):
    """CVAE encoder: maps (proprio, actions) → (μ, log σ²) in latent space."""

    hidden_dim: int = 512
    latent_dim: int = 32
    num_heads: int = 8
    num_layers: int = 4
    dropout: float = 0.1

    @nn.compact
    def __call__(
        self,
        proprio: jax.Array,           # (B, proprio_dim)
        actions: jax.Array,            # (B, chunk_size, action_dim)
        train: bool = True,
    ) -> tuple[jax.Array, jax.Array]:
        B = proprio.shape[0]
        chunk_size = actions.shape[1]

        # Project inputs to hidden_dim
        proprio_tok = nn.Dense(self.hidden_dim)(proprio)[:, None, :]   # (B, 1, H)
        action_tok = nn.Dense(self.hidden_dim)(actions)                  # (B, T, H)
        cls_tok = self.param("cls_token", nn.initializers.zeros, (1, 1, self.hidden_dim))
        cls_tok = jnp.tile(cls_tok, (B, 1, 1))

        # Concatenate: [CLS | proprio | action_tokens]
        tokens = jnp.concatenate([cls_tok, proprio_tok, action_tok], axis=1)

        # Transformer encoder layers
        x = SinusoidalPE(d_model=self.hidden_dim)(tokens)
        for _ in range(self.num_layers):
            x = TransformerEncoderLayer(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                feedforward_dim=self.hidden_dim * 4,
                dropout=self.dropout,
            )(x, train=train)

        # Use CLS token output as the latent representation
        cls_out = x[:, 0]                       # (B, H)
        mu = nn.Dense(self.latent_dim)(cls_out)
        log_var = nn.Dense(self.latent_dim)(cls_out)
        return mu, log_var


class ACTDecoder(nn.Module):
    """ACT decoder: maps (z, image_features, query_tokens) → action chunk."""

    hidden_dim: int = 512
    num_heads: int = 8
    num_layers: int = 7
    feedforward_dim: int = 3200
    dropout: float = 0.1
    chunk_size: int = 100
    action_dim: int = 7

    @nn.compact
    def __call__(
        self,
        z: jax.Array,                  # (B, latent_dim)
        proprio: jax.Array,             # (B, proprio_dim)
        train: bool = True,
    ) -> jax.Array:
        B = z.shape[0]

        # Learnable query tokens (one per action in the chunk)
        query = self.param(
            "query_tokens",
            nn.initializers.normal(0.02),
            (1, self.chunk_size, self.hidden_dim),
        )
        query = jnp.tile(query, (B, 1, 1))     # (B, T, H)

        # Memory from encoder (z and proprio)
        z_proj = nn.Dense(self.hidden_dim)(z)[:, None, :]        # (B, 1, H)
        proprio_proj = nn.Dense(self.hidden_dim)(proprio)[:, None, :]  # (B, 1, H)
        memory = jnp.concatenate([z_proj, proprio_proj], axis=1)   # (B, 2, H)

        # Decoder layers
        tgt = SinusoidalPE(d_model=self.hidden_dim)(query)
        for _ in range(self.num_layers):
            tgt = TransformerDecoderLayer(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                feedforward_dim=self.feedforward_dim,
                dropout=self.dropout,
            )(tgt, memory, train=train)

        # Project to action space
        actions = nn.Dense(self.action_dim)(tgt)   # (B, T, action_dim)
        return actions


# ---------------------------------------------------------------------------
# Full ACT Policy
# ---------------------------------------------------------------------------


class ACTPolicy(FlaxPolicy):
    """Action Chunking with Transformers (ACT) policy.

    Args:
        config: ml_collections.ConfigDict from get_act_config().
        action_dim: Dimensionality of the robot's action space.
        proprio_dim: Dimensionality of proprioceptive observations.
    """

    config: ml_collections.ConfigDict
    action_dim: int = 7
    proprio_dim: int = 14

    def setup(self) -> None:
        cfg = self.config
        self.encoder = CVAEEncoder(
            hidden_dim=cfg.hidden_dim,
            latent_dim=cfg.latent_dim,
            num_heads=cfg.num_heads,
            num_layers=cfg.num_encoder_layers,
            dropout=cfg.dropout,
        )
        self.decoder = ACTDecoder(
            hidden_dim=cfg.hidden_dim,
            num_heads=cfg.num_heads,
            num_layers=cfg.num_decoder_layers,
            feedforward_dim=cfg.feedforward_dim,
            dropout=cfg.dropout,
            chunk_size=cfg.chunk_size,
            action_dim=self.action_dim,
        )

    def __call__(
        self,
        observation: Observation,
        actions: jax.Array | None = None,   # (B, T, action_dim) — needed for training
        train: bool = True,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Forward pass.

        Returns:
            Tuple of (predicted_actions, mu, log_var).
            During inference, actions=None and mu=log_var=zeros.
        """
        proprio = observation.get("proprio", jnp.zeros((1, self.proprio_dim)))

        # CVAE: encode only during training (teacher-forcing with GT actions)
        if train and actions is not None:
            mu, log_var = self.encoder(proprio, actions, train=train)
            z = mu + jnp.exp(0.5 * log_var) * jax.random.normal(
                self.make_rng("rng") if train else jax.random.PRNGKey(0),
                mu.shape,
            )
        else:
            # At inference, sample z from prior N(0, I)
            B = proprio.shape[0]
            latent_dim = self.config.latent_dim
            mu = jnp.zeros((B, latent_dim))
            log_var = jnp.zeros((B, latent_dim))
            z = mu  # deterministic inference from prior mean

        predicted_actions = self.decoder(z, proprio, train=train)
        return predicted_actions, mu, log_var

    def select_action(
        self,
        params: Params,
        observation: Observation,
        rng: RNGKey,
        *,
        deterministic: bool = False,
    ) -> Action:
        """Return the first action from a predicted chunk."""
        result = self.apply(
            params,
            observation,
            actions=None,
            train=False,
        )
        predicted_actions, _, _ = result
        return predicted_actions[0, 0]   # first batch, first timestep

    def loss(
        self,
        params: Params,
        batch: Batch,
        rng: RNGKey,
    ) -> tuple[jax.Array, Metrics]:
        """CVAE loss = reconstruction loss + KL divergence."""
        obs = batch["observation"]
        target_actions = batch["action"]   # (B, T, action_dim)

        predicted_actions, mu, log_var = self.apply(
            params,
            obs,
            actions=target_actions,
            train=True,
            rngs={"dropout": rng, "rng": rng},
        )

        # Reconstruction loss (L1 for stability)
        recon_loss = jnp.mean(jnp.abs(predicted_actions - target_actions))

        # KL divergence: -0.5 * sum(1 + log_var - mu² - exp(log_var))
        kl_loss = -0.5 * jnp.mean(1.0 + log_var - mu ** 2 - jnp.exp(log_var))

        kl_weight = self.config.get("kl_weight", 10.0)
        total_loss = recon_loss + kl_weight * kl_loss

        return total_loss, {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
        }

    def init(
        self,
        rng: RNGKey,
        observation_spec: dict[str, tuple[int, ...]],
    ) -> Params:
        """Initialize ACT parameters."""
        proprio_dim = observation_spec.get("proprio", (self.proprio_dim,))[0]
        dummy_obs = {"proprio": jnp.zeros((1, proprio_dim))}
        dummy_actions = jnp.zeros((1, self.config.chunk_size, self.action_dim))
        rng, init_rng = jax.random.split(rng)
        return self.apply(
            {},
            dummy_obs,
            actions=dummy_actions,
            train=True,
            rngs={"params": rng, "dropout": init_rng, "rng": init_rng},
            method="__call__",
        )
