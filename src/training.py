import logging
import hashlib
from typing import Any, Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct
from flax.training import train_state
from flax.core import frozen_dict
import flax.linen as nn
from tqdm.autonotebook import tqdm

from src.config import ModelConfig, TrainingConfig


@struct.dataclass
class TrainState(train_state.TrainState):
    """Custom, JAX-compatible train state for VAE training."""
    key: jax.random.PRNGKey
    model_config: ModelConfig = struct.field(pytree_node=False)
    training_config: TrainingConfig = struct.field(pytree_node=False)


def initialize_train_state(
    model_class: Callable[..., nn.Module],
    model_config: ModelConfig,
    training_config: TrainingConfig,
    key: jax.random.PRNGKey,
    train_data_size: int,
) -> TrainState:
    """Creates an initial `TrainState` for the training process."""
    model = model_class(model_config)
    init_key, training_key = jax.random.split(key)

    dummy_cont = jnp.ones([1, model_config.continuous_output_dim])
    dummy_one_hot = jnp.ones([1, model_config.one_hot_output_dim])
    dummy_embed = jnp.ones([1, len(model_config.embedding_cardinalities)], dtype=jnp.int32)
    dummy_batch = {'continuous': dummy_cont, 'one_hot': dummy_one_hot, 'embedding': dummy_embed}

    init_rngs = {'params': init_key, 'dropout': init_key}
    params = model.init(init_rngs, dummy_batch, init_key, training=True)['params']

    if training_config.use_lr_scheduler:
        steps_per_epoch = max(1, train_data_size // training_config.batch_size)
        total_steps = steps_per_epoch * training_config.epochs
        warmup_steps = steps_per_epoch * training_config.warmup_epochs

        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=training_config.learning_rate_max,
            warmup_steps=warmup_steps,
            decay_steps=total_steps,
            end_value=0.0
        )
        tx = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(learning_rate=lr_schedule)
        )
    else:
        tx = optax.adam(training_config.learning_rate_max)

    return TrainState(
        apply_fn=model.apply,
        params=params,
        tx=tx,
        opt_state=tx.init(params),
        key=training_key,
        model_config=model_config,
        training_config=training_config,
        step=0,
    )

def stable_path_id(path):
    return int.from_bytes(hashlib.blake2b(repr(path).encode(), digest_size=4).digest(), "little")


def augment_batch(batch: Dict[str, jnp.ndarray], key: jax.random.PRNGKey, config: TrainingConfig) -> Dict:
    """
    Legacy elementwise float masking, with identifier indices unchanged.
    This low-level path does not implement the manuscript's canonical feature
    recipe; use gater_core.train_and_encode for the paper runner.
    """
    def _mask(data, subkey):
        mask = jax.random.bernoulli(subkey, p=1.0 - config.augmentation_mask_prob, shape=data.shape)
        return data * mask

    key1, key2 = jax.random.split(key)

    def create_view(base_key, data_batch):
        return jax.tree_util.tree_map_with_path(
            lambda path, x: _mask(x, jax.random.fold_in(base_key, stable_path_id(path))) if x.dtype.kind == 'f' else x,
            data_batch
        )

    batch_view1 = create_view(key1, batch)
    batch_view2 = create_view(key2, batch)

    return {'view1': batch_view1, 'view2': batch_view2}

def calculate_annealed_kl_weight(config: TrainingConfig, current_step: int, total_steps: int) -> float:
    """Calculates the KL weight based on a cyclical schedule."""
    if not config.use_cyclical_annealing:
        return config.kl_weight_max

    if config.kl_anneal_cycles <= 0:
        raise ValueError("kl_anneal_cycles must be positive")
    cycle_length = jnp.maximum(1, total_steps // config.kl_anneal_cycles)
    progress_in_cycle = (current_step % cycle_length) / cycle_length
    return config.kl_weight_max * jnp.minimum(1.0, progress_in_cycle * 2)


def normalize_latents(z):
    denominator = jnp.sqrt(jnp.maximum(jnp.sum(jnp.square(z), axis=1, keepdims=True), 1e-16))
    return z / denominator


def compute_loss(
    params: frozen_dict.FrozenDict,
    state: TrainState,
    batch: Dict[str, jnp.ndarray],
    rngs: Dict[str, jax.random.PRNGKey],
    training: bool,
    use_contrastive: bool,
    kl_weight: float,
) -> Tuple[jnp.ndarray, Dict]:
    """Computes the VAE loss, now with an optional contrastive component."""

    def vae_loss_fn(data_batch, sampling_key, dropout_key):
        (recon_cont, recon_one_hot_logits, recon_embed_logits), mean, log_var = state.apply_fn(
            {'params': params}, data_batch, training=training, rng_key=sampling_key, rngs={'dropout': dropout_key}
        )
        loss_cont = optax.l2_loss(recon_cont, data_batch['continuous']).mean()
        loss_one_hot = 0.0
        if state.model_config.one_hot_output_dim > 0:
            labels = jnp.argmax(data_batch['one_hot'], axis=1)
            loss_one_hot = optax.softmax_cross_entropy_with_integer_labels(recon_one_hot_logits, labels).mean()
        loss_embed = 0.0
        if state.model_config.embedding_cardinalities:
            for i, logits in enumerate(recon_embed_logits):
                labels = data_batch['embedding'][:, i]
                loss_embed += optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()

        reconstruction_loss = loss_cont + loss_one_hot + loss_embed
        kl_div = -0.5 * jnp.sum(1 + log_var - jnp.square(mean) - jnp.exp(log_var), axis=-1).mean()

        total_vae_loss = reconstruction_loss + kl_weight * kl_div

        metrics = {'total': total_vae_loss, 'reconstruction': reconstruction_loss, 'kl': kl_div}
        return total_vae_loss, (mean, metrics)

    # --- Contrastive Loss Calculation (InfoNCE) ---
    def contrastive_loss_fn(z1, z2, temperature):
        z1_norm = normalize_latents(z1)
        z2_norm = normalize_latents(z2)

        logits = jnp.dot(z1_norm, z2_norm.T) / temperature
        labels = jnp.arange(z1.shape[0])

        loss_z1 = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
        loss_z2 = optax.softmax_cross_entropy_with_integer_labels(logits.T, labels).mean()
        return (loss_z1 + loss_z2) / 2.0

    if use_contrastive:
        (loss1, (mean1, metrics1)) = vae_loss_fn(batch['view1'], rngs['sampling1'], rngs['dropout1'])
        (loss2, (mean2, metrics2)) = vae_loss_fn(batch['view2'], rngs['sampling2'], rngs['dropout2'])

        vae_loss = (loss1 + loss2) / 2.0
        metrics = jax.tree_util.tree_map(lambda x, y: (x + y) / 2.0, metrics1, metrics2)

        con_loss = (contrastive_loss_fn(mean1, mean2, state.training_config.contrastive_temperature)
                    if state.training_config.contrastive_weight > 0 else 0.0)
        total_loss = vae_loss + state.training_config.contrastive_weight * con_loss
        metrics['contrastive'] = con_loss
        metrics['total'] = total_loss
    else:
        total_loss, (_, metrics) = vae_loss_fn(batch, rngs['sampling1'], rngs['dropout1'])
        metrics['contrastive'] = 0.0

    return total_loss, metrics


@jax.jit
def train_step(state: TrainState, batch: Dict[str, jnp.ndarray], total_train_steps: int) -> Tuple[TrainState, Dict]:
    """Performs a single training step with augmentation and contrastive loss."""
    keys = jax.random.split(state.key, 6)
    new_state_key, aug_key, dropout1_key, dropout2_key, sampling1_key, sampling2_key = keys

    augmented_batch = augment_batch(batch, aug_key, state.training_config)

    kl_weight = calculate_annealed_kl_weight(state.training_config, state.step, total_train_steps)

    rngs = {
        'dropout1': dropout1_key, 'dropout2': dropout2_key,
        'sampling1': sampling1_key, 'sampling2': sampling2_key
    }

    grad_fn = jax.value_and_grad(compute_loss, has_aux=True)
    (loss, metrics), grads = grad_fn(
        state.params, state, augmented_batch, rngs, training=True, use_contrastive=True, kl_weight=kl_weight
    )

    new_state = state.apply_gradients(grads=grads)
    new_state = new_state.replace(key=new_state_key)
    metrics['kl_weight'] = kl_weight
    return new_state, metrics


@jax.jit
def eval_step(state: TrainState, batch: Dict[str, jnp.ndarray]) -> Dict:
    """Performs an evaluation step (no augmentation, no gradients)."""
    dropout_key, sampling_key = jax.random.split(jax.random.PRNGKey(0))
    rngs = {'dropout1': dropout_key, 'sampling1': sampling_key}

    _, metrics = compute_loss(
        state.params, state, batch, rngs, training=False, use_contrastive=False, kl_weight=state.training_config.kl_weight_max
    )
    return metrics


def run_training_loop(
    train_state: TrainState,
    train_generator: Any,
    val_data: Dict[str, np.ndarray],
    train_data_size = None,
) -> Tuple[TrainState, Dict]:
    """Executes the main training loop and returns the best state and loss history."""
    if not train_data_size:
        num_train_samples = val_data['continuous'].shape[0] / train_state.training_config.test_set_fraction
    elif isinstance(train_data_size, int):
        num_train_samples = train_data_size
    elif isinstance(train_data_size, tuple):
        num_train_samples = train_data_size[0]
    batch_size = train_state.training_config.batch_size
    num_steps_per_epoch = max(1, int(num_train_samples // batch_size))
    total_train_steps = num_steps_per_epoch * train_state.training_config.epochs

    best_val_loss = float('inf')
    patience_counter = 0
    best_state = train_state

    history = {'train_loss_iter': [], 'val_loss_epoch': [], 'kl_weight_iter': []}

    for epoch in range(train_state.training_config.epochs):
        with tqdm(range(num_steps_per_epoch), desc=f"Epoch {epoch+1}", leave=False) as pbar:
            for step in pbar:
                batch_data = next(train_generator)

                batch_data = jax.tree_util.tree_map(jnp.asarray, batch_data)

                train_state, train_metrics = train_step(train_state, batch_data, total_train_steps)
# Legacy low-level loop is incomplete; use the supported gater_core wrapper.
                iter_loss = jax.device_get(train_metrics['total'])
                history['train_loss_iter'].append(iter_loss)
                history['kl_weight_iter'].append(jax.device_get(train_metrics['kl_weight']))

                pbar.set_postfix(train_loss=f"{iter_loss:.4f}", kl_w=f"{train_metrics['kl_weight']:.2f}")

        val_metrics = run_evaluation(train_state, val_data, batch_size)
        avg_val_loss = jax.device_get(val_metrics['reconstruction'])
        history['val_loss_epoch'].append(avg_val_loss)

        logging.info(
            f"Epoch {epoch+1:03d} | Val Loss: {avg_val_loss:.4f} "
            f"(Recon: {val_metrics['reconstruction']:.4f}, KL: {val_metrics['kl']:.4f})"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_state = train_state
            logging.info(f"New best validation loss: {best_val_loss:.4f}. Best in-memory state retained.")
        else:
            patience_counter += 1

        if patience_counter >= train_state.training_config.patience:
            logging.warning(f"Stopping early after {patience_counter} epochs with no improvement.")
            break

    return best_state, history

@jax.jit
def calculate_reconstruction_errors(state: TrainState, batch: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
    """Calculates per-sample reconstruction errors for anomaly detection."""
    dropout_key, sampling_key = jax.random.split(jax.random.PRNGKey(0))

    (recon_cont, recon_one_hot_logits, recon_embed_logits), _, _ = state.apply_fn(
        {'params': state.params}, batch, training=False, rng_key=sampling_key, rngs={'dropout': dropout_key}
    )

    cont_error = jnp.mean((batch['continuous'] - recon_cont)**2, axis=1)

    one_hot_error = jnp.zeros(batch['continuous'].shape[0])
    if state.model_config.one_hot_output_dim > 0:
        one_hot_labels = jnp.argmax(batch['one_hot'], axis=1)
        one_hot_error = optax.softmax_cross_entropy_with_integer_labels(recon_one_hot_logits, one_hot_labels)

    total_embed_error = jnp.zeros(batch['continuous'].shape[0])
    if state.model_config.embedding_cardinalities:
        embed_errors = []
        for i, logits in enumerate(recon_embed_logits):
            labels = batch['embedding'][:, i]
            embed_errors.append(optax.softmax_cross_entropy_with_integer_labels(logits, labels))
        total_embed_error = jnp.sum(jnp.stack(embed_errors, axis=1), axis=1)

    total_error = cont_error + one_hot_error + total_embed_error

    return {
        'total_error': total_error, 'continuous_error': cont_error,
        'one_hot_error': one_hot_error, 'embedding_error': total_embed_error
    }

def data_generator(data: Dict[str, np.ndarray], batch_size: int, random_state: int):
    """Creates a data generator that yields batches of data indefinitely."""
    rng = np.random.RandomState(random_state)
    num_samples = data['continuous'].shape[0]
    if num_samples == 0 or batch_size <= 0:
        raise ValueError("Batch generation requires positive sample count and batch size")
    indices = np.arange(num_samples)

    while True:
        perm = rng.permutation(indices)
        for i in range(0, num_samples, batch_size):
            batch_indices = perm[i : i + batch_size]
            if len(batch_indices) == batch_size or num_samples < batch_size:
                yield {key: val[batch_indices] for key, val in data.items()}

def run_evaluation(state: TrainState, data: Dict[str, np.ndarray], batch_size: int) -> Dict:
    """Runs evaluation on a dataset in batches to conserve memory."""
    num_samples = data['continuous'].shape[0]
    if num_samples == 0: return {}
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    num_batches = (num_samples + batch_size - 1) // batch_size
    all_metrics = []
    weights = []

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch_data = {key: val[start_idx:end_idx] for key, val in data.items()}
        batch_data = jax.tree_util.tree_map(jnp.asarray, batch_data)

        metrics = eval_step(state, batch_data)
        all_metrics.append(metrics)
        weights.append(len(batch_data['continuous']))

    avg_metrics = {
        key: sum(m[key] * weight for m, weight in zip(all_metrics, weights)) / num_samples
        for key in all_metrics[0]
    }
    return avg_metrics
