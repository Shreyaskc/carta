"""Compare CARTA against other UNSUPERVISED representations.

Every method produces a 16-dim, label-free embedding; all are scored with the
SAME probes (linear + GBDT AUPRC, label-efficiency), the SAME K-means silhouette,
and we record training wall-time as the compute-cost axis. This answers "is CARTA
a better unsupervised representation than the standard alternatives, and at what
cost?" (the supervised GBDT/CatBoost baselines stay a separate reference axis).

Methods (all label-free, 16-dim, via the same preprocessing/tokenization):
  pca16        PCA to 16 dims (sklearn)                   -- linear DR, 0 trainable params
  umap16       UMAP to 16 dims (umap-learn) on the raw matrix -- non-linear DR baseline
  ae           MLP autoencoder, reconstruction only        -- variational=False, l1=0
  sae          sparse autoencoder (recon + L1 on latent)   -- variational=False, l1=1e-3
  vanilla_vae  MLP-encoder VAE, no excision, no contrastive -- a "traditional VAE"
  scarf        SSL corruption baseline (SCARF augmentation)
  gater        ours (Perceiver + feature excision + InfoNCE)

Run on a single GPU (IEEE-CIS is small enough for CPU too):
  EPOCHS=300 SEEDS=0,1,2,3,4 DATASET=ieee_cis python run_representation_comparison.py

Env knobs: METHODS, EPOCHS, SEEDS, DATASET, SAE_L1 (sparse-AE penalty),
REUSE_CARTA=1 (reuse a saved gater_<ds>.npy instead of retraining, quality only).
Writes results/representation_comparison_<ds>.json. Runs MERGE into that file, so to
add just UMAP without recomputing the others:
  METHODS=umap16 DATASET=ieee_cis python run_representation_comparison.py
"""
import os, sys, json, time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                       # src/ and the lib modules live here
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import datasets, baselines, probe
from experiment_io import load_embedding, require_matching_protocol, evaluation_fingerprint
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

RESULTS = os.environ.get("GATER_RESULTS", os.path.join(_HERE, "results", "new-runs"))
DS = os.environ.get("DATASET", "ieee_cis")
EPOCHS = int(os.environ.get("EPOCHS", "300"))
SEEDS = tuple(int(s) for s in os.environ.get("SEEDS", "0,1,2,3,4").split(","))

METHODS = [name.strip() for name in os.environ.get("METHODS", "").split(",") if name.strip()]

def carta_protocol():
    """Intended default CARTA recipe, excluding data-derived decoder dimensions."""
    from dataclasses import asdict, fields
    from src.config import ModelConfig, TrainingConfig
    model = ModelConfig()
    training = TrainingConfig(epochs=EPOCHS)
    if EPOCHS <= training.warmup_epochs:
        training.use_lr_scheduler = False
    return {"encoder": "perceiver", "excision_mode": "feature", "epochs": EPOCHS,
            "seed": 42, "batch_size": training.batch_size, "card_threshold": 50,
            "checkpoint_selection": "held-out reconstruction",
            "use_contrastive": True, "variational": True, "use_powerlaw": True, "l1_weight": 0.0,
            "model_config": {item.name: getattr(model, item.name) for item in fields(model) if item.init},
            "training_config": asdict(training)}

def silhouette(emb):
    Z = StandardScaler().fit_transform(emb)
    best_k, best_s = None, -1.0
    for k in range(3, 8):
        lab = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(Z)
        s = silhouette_score(Z[:20000], lab[:20000], random_state=42)
        if s > best_s:
            best_k, best_s = k, s
    return best_k, round(float(best_s), 4)


def embed(method):
    """Return (embedding, train_seconds). PCA/UMAP are sklearn/umap-learn on the raw
    z-scored matrix (linear vs. non-linear DR baselines); the rest via gater_core."""
    if method == "pca16":
        df, spec = datasets.load(DS)
        X = baselines.build_raw_matrix(df, spec)
        t0 = time.time()
        E = baselines.pca_reduce(X, 16)
        return E.astype(np.float32), round(time.time() - t0, 1)

    if method == "umap16":                           # non-linear DR baseline (PCA's counterpart)
        from umap import UMAP
        df, spec = datasets.load(DS)
        X = baselines.build_raw_matrix(df, spec)
        t0 = time.time()
        # No random_state: UMAP forces n_jobs=1 when a seed is set, which both cripples
        # the compute-axis timing and is very slow on ~590K rows. We run multi-threaded
        # instead (embedding not bit-reproducible; probe seeds still control AUPRC variance).
        E = UMAP(n_components=16).fit_transform(X)
        return E.astype(np.float32), round(time.time() - t0, 1)

    from gater_core import train_and_encode
    kw = dict(dataset=DS, epochs=EPOCHS)
    NO_AUG = {"augmentation_mask_prob": 0.0}
    if method == "ae":                               # plain MLP autoencoder (recon only)
        kw.update(encoder="mlp", use_contrastive=False, variational=False,
                  l1_weight=0.0, train_overrides=NO_AUG)
    elif method == "sae":                            # sparse autoencoder (recon + L1 latent)
        kw.update(encoder="mlp", use_contrastive=False, variational=False,
                  l1_weight=float(os.environ.get("SAE_L1", "1e-3")), train_overrides=NO_AUG)
    elif method == "vanilla_vae":                    # MLP-VAE, no excision, no contrastive
        kw.update(encoder="mlp", use_contrastive=False, excision_mode="cell",
                  train_overrides=NO_AUG)
    elif method == "scarf":                          # SSL corruption baseline
        kw.update(encoder="perceiver", use_contrastive=True, excision_mode="scarf")
    elif method == "gater":                          # ours
        kw.update(encoder="perceiver", use_contrastive=True, excision_mode="feature")
    else:
        raise ValueError(f"unknown method {method!r}")
    t0 = time.time()
    E, _ = train_and_encode(**kw)
    return E.astype(np.float32), round(time.time() - t0, 1)


def main():
    if not METHODS:
        raise ValueError("Set METHODS explicitly, e.g. METHODS=pca16,ae,sae,vanilla_vae,scarf,gater")
    df, spec = datasets.load(DS)
    y = df[spec.label].to_numpy().astype(int)
    os.makedirs(RESULTS, exist_ok=True)
    population = evaluation_fingerprint(df, spec)
    out = {"dataset": DS, "seeds": list(SEEDS), "epochs": EPOCHS,
           "evaluation_fingerprint": population, "methods": {}}
    path = os.path.join(RESULTS, f"representation_comparison_{DS}.json")
    saved = os.path.join(RESULTS, f"gater_{DS}.npy")

    # MERGE into an existing file so a targeted run (e.g. METHODS=umap16) adds a
    # method without recomputing -- and thus perturbing -- the others already scored.
    if os.path.exists(path):
        with open(path) as handle:
            existing = json.load(handle)
        require_matching_protocol(existing, DS, SEEDS, EPOCHS, population)
        out["methods"] = existing.get("methods", {})

    # By default CARTA is RETRAINED here so its compute cost is measured on the same
    # hardware as the other methods (a fair compute comparison). Set REUSE_CARTA=1 to
    # reuse a saved reference embedding instead (quality metrics only, no timing).
    reuse_gater = os.environ.get("REUSE_CARTA", "0") == "1"
    for m in METHODS:
        artifact = None
        if m == "gater" and reuse_gater and os.path.exists(saved):
            E, artifact = load_embedding(saved, df, spec, os.environ.get("ALLOW_UNVERIFIED_EMBEDDINGS") == "1",
                                         expected_protocol=carta_protocol(), return_metadata=True)
            if E.shape[1] != 16:
                raise ValueError("Reused CARTA embedding must have 16 columns")
            E, secs = E.astype(np.float32), None
            print(f"[{m}] reusing {saved} (no timing)", flush=True)
        else:
            E, secs = embed(m)
        full = probe.evaluate(E, y, SEEDS)
        le = probe.label_efficiency(E, y, seeds=SEEDS)
        k, sil = silhouette(E)
        out["methods"][m] = {
            "lin_auprc": round(full["linear"]["auprc"], 4),
            "lin_auprc_std": round(full["linear"].get("auprc_std", 0.0), 4),
            "gbdt_auprc": round(full["gbdt"]["auprc"], 4),
            "silhouette": sil, "best_k": k,
            "train_seconds": secs,
            "label_efficiency": {f: round(v["auprc"], 4) for f, v in le.items()},
        }
        if artifact is not None:
            out["methods"][m]["reused_artifact"] = artifact
        with open(path, "w") as handle:
            json.dump(out, handle, indent=2, allow_nan=False)
        print(f"[{m}] lin={out['methods'][m]['lin_auprc']} gbdt={out['methods'][m]['gbdt_auprc']} "
              f"silhouette={sil} (k={k}) train_s={secs}", flush=True)

    print(f"\nwrote {path}")
    print("== REPRESENTATION COMPARISON DONE ==")


if __name__ == "__main__":
    main()
