"""Shared train-and-encode implementation for CARTA and representation comparisons. Returns a frozen latent mean for all loaded rows in input order. Preprocessing fits pooled unlabeled features; the supervised probe split is separate."""
from __future__ import annotations

import logging
from functools import partial
from typing import Dict, Tuple

import numpy as np

import datasets
from adapter import split_roles


def _make_train_step(use_contrastive: bool, augmenter):
    import jax
    from src.training import calculate_annealed_kl_weight, compute_loss

    @jax.jit
    def step(state, batch, total_steps):
        keys = jax.random.split(state.key, 6)
        new_key, aug_key, d1, d2, s1, s2 = keys
        kl_w = calculate_annealed_kl_weight(state.training_config, state.step, total_steps)
        gfn = jax.value_and_grad(compute_loss, has_aux=True)
        if use_contrastive:
            ab = augmenter(batch, aug_key)
            rngs = {"dropout1": d1, "dropout2": d2, "sampling1": s1, "sampling2": s2}
            (loss, metrics), grads = gfn(state.params, state, ab, rngs, True, True, kl_w)
        else:
            rngs = {"dropout1": d1, "sampling1": s1}
            (loss, metrics), grads = gfn(state.params, state, batch, rngs, True, False, kl_w)
        new_state = state.apply_gradients(grads=grads).replace(key=new_key)
        metrics["kl_weight"] = kl_w
        return new_state, metrics
    return step


def _make_ae_train_step(l1_weight: float):
    """Autoencoder step: reconstruction only (+ optional L1 on the latent for a
    sparse AE). No KL, no contrastive. Recon terms mirror src.training.compute_loss
    so AE/SAE are apples-to-apples with the VAE-family embeddings."""
    import jax
    import jax.numpy as jnp
    import optax

    @jax.jit
    def step(state, batch, total_steps):
        new_key, d1, s1 = jax.random.split(state.key, 3)

        def loss_fn(params):
            (recon_cont, recon_oh, recon_emb), mean, _ = state.apply_fn(
                {"params": params}, batch, training=True, rng_key=s1,
                rngs={"dropout": d1})
            loss = optax.l2_loss(recon_cont, batch["continuous"]).mean()
            if state.model_config.one_hot_output_dim > 0:
                labels = jnp.argmax(batch["one_hot"], axis=1)
                loss += optax.softmax_cross_entropy_with_integer_labels(recon_oh, labels).mean()
            if state.model_config.embedding_cardinalities:
                for i, logits in enumerate(recon_emb):
                    loss += optax.softmax_cross_entropy_with_integer_labels(
                        logits, batch["embedding"][:, i]).mean()
            loss += l1_weight * jnp.mean(jnp.abs(mean))       # 0 for plain AE
            return loss, {"total": loss}

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        return state.apply_gradients(grads=grads).replace(key=new_key), metrics
    return step


def train_and_encode(dataset: str, *, epochs: int = 300, encoder: str = "perceiver",
                     use_contrastive: bool = True, excision_mode: str = "feature",
                     use_powerlaw: bool = True, variational: bool = True,
                     l1_weight: float = 0.0,
                     model_overrides: Dict | None = None,
                     train_overrides: Dict | None = None, card_threshold: int = 50,
                     subsample: int = 0, seed: int = 42, batch_size: int = 16384
                     ) -> Tuple[np.ndarray, "datasets.DatasetSpec"]:
    import jax
    import jax.numpy as jnp
    from sklearn.model_selection import train_test_split

    from src.config import ProjectConfig, apply_overrides, validate_configs
    from src.data.data_processing import SmartPowerlawTransformer
    from src.data.temporal_preprocessor import TabularPreprocessor
    from src.training import initialize_train_state, run_evaluation, data_generator
    from encoders import ConfigurableVAE
    from augment import make_augmenter

    cfg = ProjectConfig()
    cfg.training.epochs = epochs
    cfg.training.batch_size = batch_size
    cfg.training.random_state = seed
    cfg.training.use_powerlaw_transform = use_powerlaw
    apply_overrides(cfg.training, train_overrides or {})
    apply_overrides(cfg.model, model_overrides or {})
    validate_configs(cfg.model, cfg.training)

    batch_size = cfg.training.batch_size
    epochs = cfg.training.epochs
    seed = cfg.training.random_state
    use_powerlaw = cfg.training.use_powerlaw_transform
    df, spec = datasets.load(dataset)
    if subsample and subsample < len(df):
        df = df.sample(subsample, random_state=seed).reset_index(drop=True)
    spec = datasets.describe_population(df, spec)
    spec.encoding_protocol = {"encoder": encoder, "seed": seed, "epochs": epochs, "batch_size": batch_size,
                              "excision_mode": excision_mode, "card_threshold": card_threshold,
                              "train_overrides": train_overrides or {}, "model_overrides": model_overrides or {},
                              "representation_fit": "pooled-feature transductive", "checkpoint_selection": "held-out reconstruction",
                              "use_contrastive": use_contrastive,
                              "use_powerlaw": use_powerlaw, "variational": variational, "l1_weight": l1_weight}
    roles = split_roles(df, spec, card_threshold)
    if not roles.continuous:
        raise SystemExit("CARTA needs >=1 continuous feature.")

    cfg.features.continuous_cols = roles.continuous
    cfg.features.one_hot_cols = roles.one_hot
    cfg.features.embedding_cols = roles.embedding
    feat = df[spec.features].copy()
    if use_powerlaw:
        powerlaw = SmartPowerlawTransformer(continuous_cols=roles.continuous)
        feat = powerlaw.fit_transform(feat)
    else:
        powerlaw = None  # skip for all-numeric PCA datasets (Gaussian, no power law)

    pre = TabularPreprocessor(roles.continuous, roles.one_hot, roles.embedding, random_state=seed)
    pre.fit(feat)
    cfg.model.continuous_output_dim = len(roles.continuous)
    cfg.model.one_hot_output_dim = pre.metadata["one_hot_output_dim"]
    cfg.model.embedding_cardinalities = pre.metadata["embedding_cardinalities"]

    tr_df, va_df = train_test_split(feat, test_size=cfg.training.test_set_fraction,
                                   random_state=seed)
    tr, va = pre.transform(tr_df), pre.transform(va_df)

    from dataclasses import asdict

    # one-hot block sizes (per low-card feature) for feature-wise token excision
    one_hot_groups = ([len(c) for c in pre.transformers["one_hot"].categories_]
                      if roles.one_hot else [])
    augmenter = make_augmenter(excision_mode, cfg.training.augmentation_mask_prob,
                               one_hot_groups)

    # Guard: warmup-cosine-decay needs total_steps > warmup_steps. Short runs
    # (smoke tests, tiny subsamples) would give negative decay_steps -> fall back
    # to a constant LR. No effect on full runs.
    steps_per_epoch = max(1, tr_df.shape[0] // batch_size)
    total_steps = steps_per_epoch * epochs
    if cfg.training.use_lr_scheduler and total_steps <= steps_per_epoch * cfg.training.warmup_epochs:
        logging.warning("run too short for warmup schedule; using constant LR")
        cfg.training.use_lr_scheduler = False

    if not variational:                 # AE / sparse-AE: recon-only, no KL, no contrastive
        cfg.training.kl_weight_max = 0.0
        cfg.training.use_cyclical_annealing = False
    spec.encoding_protocol.update({"model_config": asdict(cfg.model), "training_config": asdict(cfg.training),
                                   "use_contrastive": use_contrastive and variational})
    model_class = partial(ConfigurableVAE, encoder_type=encoder, variational=variational)
    key = jax.random.PRNGKey(seed)
    state = initialize_train_state(model_class, cfg.model, cfg.training, key, tr_df.shape[0])
    gen = data_generator(tr, batch_size, seed)
    step_fn = (_make_ae_train_step(l1_weight) if not variational
               else _make_train_step(use_contrastive, augmenter))

    best_val, best_params, patience = float("inf"), state.params, 0
    for ep in range(epochs):
        for _ in range(steps_per_epoch):
            b = jax.tree_util.tree_map(jnp.asarray, next(gen))
            state, _ = step_fn(state, b, total_steps)
        vm = run_evaluation(state, va, batch_size)
        vloss = float(jax.device_get(vm["reconstruction"])) if vm else float("nan")
        if vloss < best_val:
            best_val, best_params, patience = vloss, state.params, 0
        else:
            patience += 1
        if ep % 10 == 0 or ep == epochs - 1:
            logging.info(f"[{dataset}/{encoder}] epoch {ep} val={vloss:.4f} best={best_val:.4f}")
        if patience >= cfg.training.patience:
            logging.info(f"[{dataset}/{encoder}] early stop @ epoch {ep}")
            break

    model = ConfigurableVAE(config=cfg.model, encoder_type=encoder, variational=variational)
    proc = pre.transform(feat)
    embs, bs = [], 8192
    for i in range(0, len(feat), bs):
        b = {k: jnp.asarray(v[i:i + bs]) for k, v in proc.items()}
        embs.append(np.asarray(model.apply({"params": best_params}, b, method=model.encode)))
    return np.concatenate(embs, 0), spec
