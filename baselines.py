"""Baseline representations for the comparability study.

  * raw      : StandardScaler(numeric) + OneHotEncoder(categoricals) -> dense matrix
               (this is what a GBDT/logistic head sees with no learned embedding)
  * pca16    : PCA to 16 dims on the raw matrix -- the classical linear DR baseline,
               the fair 16-dim comparator to CARTA's 16-dim latent.

CARTA's own 16-dim frozen embedding is produced on the cluster by `gater_runner.py`
and evaluated with the SAME probe (see run_baselines.py --embeddings).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from datasets import DatasetSpec


def build_raw_matrix(df: pd.DataFrame, spec: DatasetSpec,
                     card_threshold: int = 50) -> np.ndarray:
    # Only one-hot encode LOW-cardinality categoricals (card <= threshold).
    # High-card cols (merchant_name, city, zip etc.) are excluded: one-hot would
    # cause sparse-matrix explosion (e.g. 1397+2075+3109 dims x 2M rows ~ 50 GB).
    # This is the "raw practical baseline" -- what a practitioner would actually run.
    # It also makes the paper's motivation concrete: high-card features are infeasible
    # for raw OHE; CARTA handles them via learned embeddings.
    from adapter import split_roles
    roles = split_roles(df, spec, card_threshold)
    feat_cols = roles.continuous + roles.one_hot
    parts = [("num", StandardScaler(), roles.continuous)]
    if roles.one_hot:
        parts.append(("cat", OneHotEncoder(handle_unknown="ignore",
                                          sparse_output=False), roles.one_hot))
    ct = ColumnTransformer(parts)
    X = ct.fit_transform(df[feat_cols])
    return np.asarray(X, dtype=np.float32)


def build_freq_matrix(df: pd.DataFrame, spec: DatasetSpec,
                      card_threshold: int = 50) -> np.ndarray:
    """Frequency-encoded baseline: includes high-card columns by replacing each
    category with its log-normalized frequency. This is the fairest raw baseline
    for TabFormer — it uses all features including merchant identifiers."""
    from adapter import split_roles
    roles = split_roles(df, spec, card_threshold)
    parts = [("num", StandardScaler(), roles.continuous)]
    if roles.one_hot:
        parts.append(("ohe", OneHotEncoder(handle_unknown="ignore",
                                          sparse_output=False), roles.one_hot))
    ct = ColumnTransformer(parts)
    X_base = ct.fit_transform(df[roles.continuous + roles.one_hot]).astype(np.float32)
    # frequency-encode high-card columns
    freq_cols = []
    for col in roles.embedding:
        freq = df[col].map(df[col].value_counts(normalize=True)).fillna(0)
        freq_cols.append(np.log1p(freq.values).reshape(-1, 1).astype(np.float32))
    if freq_cols:
        return np.hstack([X_base] + freq_cols)
    return X_base


def pca_reduce(X: np.ndarray, k: int = 16, seed: int = 0) -> np.ndarray:
    k = min(k, X.shape[1])
    return PCA(n_components=k, random_state=seed).fit_transform(X).astype(np.float32)
