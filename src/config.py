# src/config.py

from dataclasses import dataclass, field, fields
import math
from typing import List
from pathlib import Path

@dataclass
class PathConfig:
    """Configuration for file paths."""
    data_path: Path = Path("./data/transactions.parquet")
    artifact_dir: Path = Path("./trained_model")

@dataclass
class FeatureConfig:
    """Dataset-specific feature roles, assigned by the experiment loader."""
    continuous_cols: List[str] = field(default_factory=list)
    one_hot_cols: List[str] = field(default_factory=list)
    embedding_cols: List[str] = field(default_factory=list)

@dataclass
class ModelConfig:
    """Hyperparameters for the Tabular Variational Autoencoder."""
    num_latent_queries: int = 8  # Number of queries in the Perceiver encoder
    feature_embedding_dim: int = 64
    encoder_num_layers: int = 4 # Increased layers for deeper processing
    encoder_num_heads: int = 4  # Increased heads for more attention patterns
    latent_dim: int = 16      # Increased latent dim for more capacity
    decoder_dims: List[int] = field(default_factory=lambda: [32, 64, 128])
    dropout_rate: float = 0.1
    embedding_cardinalities: List[int] = field(default_factory=list, init=False)
    one_hot_output_dim: int = field(default=0, init=False)
    continuous_output_dim: int = field(default=0, init=False)

@dataclass
class TrainingConfig:
    """Configuration for the training process."""
    use_powerlaw_transform: bool = True
    epochs: int = 300
    batch_size: int = 1024 * 16
    random_state: int = 42
    test_set_fraction: float = 0.15
    patience: int = 50
    learning_rate_max: float = 5e-4
    use_lr_scheduler: bool = True
    warmup_epochs: int = 10
    kl_weight_max: float = 1.0
    use_cyclical_annealing: bool = True
    kl_anneal_cycles: int = 10
    contrastive_weight: float = 0.1
    contrastive_temperature: float = 0.07
    augmentation_mask_prob: float = 0.2



@dataclass
class ProjectConfig:
    """Root configuration object."""
    paths: PathConfig = field(default_factory=PathConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    random_state: int = 42


def apply_overrides(config, overrides):
    allowed = {item.name for item in fields(config) if item.init}
    unknown = set(overrides) - allowed
    if unknown:
        raise ValueError(f"Unknown or derived configuration fields: {sorted(unknown)}")
    for name, value in overrides.items():
        setattr(config, name, value)
    return config


def validate_configs(model, training):
    for name in ("num_latent_queries", "feature_embedding_dim", "encoder_num_layers", "encoder_num_heads", "latent_dim"):
        value = getattr(model, name)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if model.feature_embedding_dim % model.encoder_num_heads:
        raise ValueError("feature_embedding_dim must be divisible by encoder_num_heads")
    if not math.isfinite(model.dropout_rate) or not 0 <= model.dropout_rate < 1:
        raise ValueError("dropout_rate must be in [0, 1)")
    if not model.decoder_dims or any(not isinstance(dim, int) or dim <= 0 for dim in model.decoder_dims):
        raise ValueError("decoder_dims must contain positive integers")
    for name in ("epochs", "batch_size", "patience", "kl_anneal_cycles"):
        value = getattr(training, name)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not isinstance(training.warmup_epochs, int) or training.warmup_epochs < 0:
        raise ValueError("warmup_epochs must be a nonnegative integer")
    if not isinstance(training.random_state, int) or not 0 <= training.random_state < 2**32:
        raise ValueError("random_state must be an integer in [0, 2**32)")
    for name in ("learning_rate_max", "contrastive_temperature"):
        value = getattr(training, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name in ("kl_weight_max", "contrastive_weight"):
        value = getattr(training, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if not 0 <= training.augmentation_mask_prob <= 1:
        raise ValueError("augmentation_mask_prob must be in [0, 1]")
    if not 0 < training.test_set_fraction < 1:
        raise ValueError("test_set_fraction must be in (0, 1)")
