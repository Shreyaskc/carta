from typing import Dict, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp

from src.config import ModelConfig


class MultiQueryPerceiverEncoder(nn.Module):
    config: ModelConfig

    @nn.compact
    def __call__(self,
                 cont_data: jnp.ndarray,
                 one_hot_data: jnp.ndarray,
                 embed_data: jnp.ndarray,
                 training: bool
                 ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Forward pass for the Multi-Query Perceiver Encoder."""

        feature_tokens = []
        num_feature_tokens = 0
        if cont_data.shape[1] > 0:
            cont_tokens = nn.Dense(features=self.config.feature_embedding_dim, name='continuous_tokens')(cont_data)
            feature_tokens.append(jnp.expand_dims(cont_tokens, axis=1))
            num_feature_tokens += 1
        if one_hot_data.shape[1] > 0:
            one_hot_tokens = nn.Dense(features=self.config.feature_embedding_dim, name='one_hot_tokens')(one_hot_data)
            feature_tokens.append(jnp.expand_dims(one_hot_tokens, axis=1))
            num_feature_tokens += 1
        for i, cardinality in enumerate(self.config.embedding_cardinalities):
            embedding_layer = nn.Embed(num_embeddings=cardinality, features=self.config.feature_embedding_dim, name=f'embedding_{i}')
            embed_token = embedding_layer(embed_data[:, i])
            feature_tokens.append(jnp.expand_dims(embed_token, axis=1))
            num_feature_tokens += 1
        x = jnp.concatenate(feature_tokens, axis=1)

        pos_embed = self.param(
            'positional_embeddings',
            nn.initializers.lecun_normal(),
            (1, num_feature_tokens, self.config.feature_embedding_dim)
        )
        x += pos_embed

        latent_queries = self.param(
            'latent_queries',
            nn.initializers.lecun_normal(),
            (1, self.config.num_latent_queries, self.config.feature_embedding_dim)
        )
        latent_queries = jnp.tile(latent_queries, (x.shape[0], 1, 1))

        for i in range(self.config.encoder_num_layers):
            # Cross-Attention
            cross_attn_output = nn.MultiHeadDotProductAttention(
                num_heads=self.config.encoder_num_heads, name=f'cross_attention_{i}'
            )(inputs_q=latent_queries, inputs_kv=x)
            cross_attn_output = nn.Dropout(rate=self.config.dropout_rate, deterministic=not training)(cross_attn_output)
            latent_queries = nn.LayerNorm(name=f'ln_cross_{i}')(latent_queries + cross_attn_output)

            # Self-Attention
            self_attn_output = nn.SelfAttention(
                num_heads=self.config.encoder_num_heads, name=f'self_attention_{i}'
            )(inputs_q=latent_queries)
            self_attn_output = nn.Dropout(rate=self.config.dropout_rate, deterministic=not training)(self_attn_output)
            latent_queries = nn.LayerNorm(name=f'ln_self_{i}')(latent_queries + self_attn_output)

            # Block c: MLP
            ffn_output = nn.Dense(features=self.config.feature_embedding_dim, name=f'ffn1_{i}')(latent_queries)
            ffn_output = nn.gelu(ffn_output)
            ffn_output = nn.Dropout(rate=self.config.dropout_rate, deterministic=not training)(ffn_output)
            ffn_output = nn.Dense(features=self.config.feature_embedding_dim, name=f'ffn2_{i}')(ffn_output)
            latent_queries = nn.LayerNorm(name=f'ln_ffn_{i}')(latent_queries + ffn_output)

        final_context = jnp.mean(latent_queries, axis=1)

        mean = nn.Dense(features=self.config.latent_dim, name='mean')(final_context)
        log_var = nn.Dense(features=self.config.latent_dim, name='log_var')(final_context)

        return mean, log_var


class Decoder(nn.Module):
    config: ModelConfig

    @nn.compact
    def __call__(self, latent_vec: jnp.ndarray, training: bool) -> Tuple:
        """
        Forward pass for the Decoder. (Unchanged)
        """
        x = latent_vec
        for dim in self.config.decoder_dims:
            x = nn.Dense(features=dim)(x)
            x = nn.gelu(x)
            x = nn.Dropout(rate=self.config.dropout_rate, deterministic=not training)(x)

        continuous_recon = nn.Dense(
            features=self.config.continuous_output_dim, name='continuous_output'
        )(x)

        if self.config.one_hot_output_dim > 0:
            one_hot_recon_logits = nn.Dense(
                features=self.config.one_hot_output_dim, name='one_hot_output'
            )(x)
        else:
            one_hot_recon_logits = jnp.zeros((x.shape[0], 0))

        embedding_recon_logits = []
        for i, cardinality in enumerate(self.config.embedding_cardinalities):
            logits = nn.Dense(features=cardinality, name=f'embedding_output_{i}')(x)
            embedding_recon_logits.append(logits)

        return continuous_recon, one_hot_recon_logits, embedding_recon_logits


class TabularVariationalAutoencoder(nn.Module):
    """
    The main Variational Autoencoder (VAE) module.
    This version uses a MultiQueryPerceiverEncoder for robust feature
    interaction and a standard MLP-based decoder for reconstruction.
    """
    config: ModelConfig

    def setup(self):
        """Initializes the encoder and decoder submodules."""
        self.encoder = MultiQueryPerceiverEncoder(config=self.config)
        self.decoder = Decoder(config=self.config)

    def __call__(self, batch: Dict[str, jnp.ndarray], rng_key: jnp.ndarray, training: bool = False) -> Tuple:
        """
        Full forward pass through the VAE.
        """
        mean, log_var = self.encoder(
            cont_data=batch['continuous'],
            one_hot_data=batch['one_hot'],
            embed_data=batch['embedding'],
            training=training
        )

        std = jnp.exp(0.5 * log_var)
        epsilon = jax.random.normal(rng_key, shape=mean.shape)
        z = mean + std * epsilon

        recon = self.decoder(z, training=training)
        return recon, mean, log_var

    def encode(self, batch: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """
        Utility method to get latent mean vectors only.
        """
        mean, _ = self.encoder(
            cont_data=batch['continuous'],
            one_hot_data=batch['one_hot'],
            embed_data=batch['embedding'],
            training=False
        )
        return mean
