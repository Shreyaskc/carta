"""Encoder variants for the C1 ablation (Perceiver vs GRU-gated vs plain MLP-VAE).

Kept separate from the model package (`src/`) so that stays unchanged.
All three use type-specific projections and identifier embeddings. The attention
variants add learned positional embeddings; the MLP concatenates ordered inputs.
These compare encoding mechanisms without claiming matched parameter counts:

  - perceiver : the MultiQueryPerceiverEncoder from src/ (imported unchanged)
  - gru       : a GRU-gated single-context encoder – one context vector refined by
                a GRU cell across L layers (cross-attention + update/reset gates;
                no self-attention or feed-forward).
  - mlp       : plain MLP-VAE over flattened tokens (no attention bottleneck) –
                the "does the Perceiver actually help?" control.

Decoder is the multi-head Decoder from src/ in all cases.
"""
from __future__ import annotations

from typing import List, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp

from src.config import ModelConfig
from src.models.vae import Decoder, MultiQueryPerceiverEncoder


class GatedAttentionEncoder(nn.Module):
    """GRU-gated single-context encoder."""
    config: ModelConfig

    @nn.compact
    def __call__(self, cont_data, one_hot_data, embed_data, training: bool
                 ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        d = self.config.feature_embedding_dim
        toks: List[jnp.ndarray] = []
        if cont_data.shape[1] > 0:
            toks.append(jnp.expand_dims(nn.Dense(d, name="continuous_tokens")(cont_data), 1))
        if one_hot_data.shape[1] > 0:
            toks.append(jnp.expand_dims(nn.Dense(d, name="one_hot_tokens")(one_hot_data), 1))
        for i, card in enumerate(self.config.embedding_cardinalities):
            emb = nn.Embed(card, d, name=f"embedding_{i}")(embed_data[:, i])
            toks.append(jnp.expand_dims(emb, 1))
        x = jnp.concatenate(toks, axis=1)
        n_tok = x.shape[1]
        x = x + self.param("positional_embeddings", nn.initializers.lecun_normal(),
                           (1, n_tok, d))

        # single learnable context vector
        ctx = self.param("init_context", nn.initializers.lecun_normal(), (1, 1, d))
        ctx = jnp.tile(ctx, (x.shape[0], 1, 1))              # (B,1,d)
        gru = nn.GRUCell(features=d, name="gru")
        for i in range(self.config.encoder_num_layers):
            attn = nn.MultiHeadDotProductAttention(
                num_heads=self.config.encoder_num_heads, name=f"cross_attention_{i}"
            )(inputs_q=ctx, inputs_kv=x)
            attn = nn.Dropout(self.config.dropout_rate, deterministic=not training)(attn)
            carry = jnp.squeeze(ctx, axis=1)                # (B,d)
            new_carry, _ = gru(carry, jnp.squeeze(attn, axis=1))
            ctx = jnp.expand_dims(new_carry, axis=1)        # (B,1,d)

        final = jnp.squeeze(ctx, axis=1)
        mean = nn.Dense(self.config.latent_dim, name="mean")(final)
        log_var = nn.Dense(self.config.latent_dim, name="log_var")(final)
        return mean, log_var


class MLPVAEEncoder(nn.Module):
    """Plain MLP-VAE over flattened tokens – no attention bottleneck (control)."""
    config: ModelConfig

    @nn.compact
    def __call__(self, cont_data, one_hot_data, embed_data, training: bool
                 ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        d = self.config.feature_embedding_dim
        toks: List[jnp.ndarray] = []
        if cont_data.shape[1] > 0:
            toks.append(nn.Dense(d, name="continuous_tokens")(cont_data))
        if one_hot_data.shape[1] > 0:
            toks.append(nn.Dense(d, name="one_hot_tokens")(one_hot_data))
        for i, card in enumerate(self.config.embedding_cardinalities):
            toks.append(nn.Embed(card, d, name=f"embedding_{i}")(embed_data[:, i]))
        h = jnp.concatenate(toks, axis=-1)                  # (B, N_tok*d)
        for j, dim in enumerate([d * 4, d * 2, d]):
            h = nn.gelu(nn.Dense(dim, name=f"mlp_{j}")(h))
            h = nn.Dropout(self.config.dropout_rate, deterministic=not training)(h)
        mean = nn.Dense(self.config.latent_dim, name="mean")(h)
        log_var = nn.Dense(self.config.latent_dim, name="log_var")(h)
        return mean, log_var


_ENCODERS = {"perceiver": MultiQueryPerceiverEncoder,
             "gru": GatedAttentionEncoder,
             "mlp": MLPVAEEncoder}


class ConfigurableVAE(nn.Module):
    """VAE with a swappable encoder; mirrors the reference design TabularVariationalAutoencoder API.

    Set ``variational=False`` for a deterministic autoencoder (z = mean, no
    reparameterisation noise); combined with kl_weight=0 this is a plain AE, and
    with an L1 penalty on the latent (added in the training loss) a sparse AE.
    """
    config: ModelConfig
    encoder_type: str = "perceiver"
    variational: bool = True

    def setup(self):
        self.encoder = _ENCODERS[self.encoder_type](config=self.config)
        self.decoder = Decoder(config=self.config)

    def __call__(self, batch, rng_key, training: bool = False):
        mean, log_var = self.encoder(batch["continuous"], batch["one_hot"],
                                     batch["embedding"], training=training)
        if self.variational:
            std = jnp.exp(0.5 * log_var)
            z = mean + std * jax.random.normal(rng_key, shape=mean.shape)
        else:
            z = mean
        return self.decoder(z, training=training), mean, log_var

    def encode(self, batch):
        mean, _ = self.encoder(batch["continuous"], batch["one_hot"],
                               batch["embedding"], training=False)
        return mean
