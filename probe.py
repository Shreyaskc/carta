"""Shared evaluation protocol – identical for baselines and CARTA embeddings.

The scientific contract: whatever the representation (raw-encoded features, PCA,
or a frozen 16-dim CARTA embedding), it is evaluated the SAME way so numbers are
directly comparable:

  * fixed stratified train/test split (seeded);
  * two frozen-representation probes: a LINEAR probe (LogisticRegression) and a
    STRONG head (HistGradientBoostingClassifier);
  * primary metric AUPRC (average precision) – the right metric for rare-class
    fraud; ROC-AUC reported alongside (it hides the cost – a well-known lesson);
  * label-efficiency curve: refit the probe on a stratified 1/5/10/25/100% subset
    of TRAIN labels, evaluate on the fixed test set (this is claim C4).

All randomness is seeded; results are means over `seeds`.
"""
from __future__ import annotations

from typing import Dict, List, Sequence
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier



def _validate_labels(y):
    y = np.asarray(y)
    if y.ndim != 1 or set(np.unique(y)) != {0, 1}:
        raise ValueError("Fraud probes require one-dimensional binary labels containing both 0 and 1")
    return y


def _validate_features(X, y):
    X = np.asarray(X)
    if X.ndim != 2 or len(X) != len(y) or X.shape[1] == 0:
        raise ValueError("Features must be a nonempty 2D matrix aligned with labels")
    try:
        finite = np.isfinite(X).all()
    except TypeError as error:
        raise ValueError("Features must be numeric") from error
    if not finite:
        raise ValueError("Features must be finite")
    return X


def _validate_seeds(seeds):
    seeds = tuple(seeds)
    if not seeds or any(not isinstance(seed, (int, np.integer)) or seed < 0 or seed >= 2**32 for seed in seeds):
        raise ValueError("Provide at least one integer seed in [0, 2**32)")
    return seeds


def _fraction_key(fraction):
    rounded = f"{fraction:.2f}"
    return rounded if abs(float(rounded) - fraction) <= 1e-12 else f"{fraction:.12g}"


def split(y: np.ndarray, seed: int, test_frac: float = 0.3):
    y = _validate_labels(y)
    _validate_seeds((seed,))
    if not np.isfinite(test_frac) or not 0 < test_frac < 1:
        raise ValueError("test_frac must be finite and in (0, 1)")
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=test_frac, random_state=seed, stratify=y)
    return tr, te


def _fit_eval(clf, Xtr, ytr, Xte, yte) -> Dict[str, float]:
    clf.fit(Xtr, ytr)
    if hasattr(clf, "predict_proba"):
        p = clf.predict_proba(Xte)[:, 1]
    else:
        p = clf.decision_function(Xte)
    return {"auprc": float(average_precision_score(yte, p)),
            "auc": float(roc_auc_score(yte, p))}


def _probes(seed: int):
    return {
        "linear": LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0),
        "gbdt": HistGradientBoostingClassifier(max_depth=6, learning_rate=0.1,
                                              max_iter=200, random_state=seed),
    }


def evaluate(X: np.ndarray, y: np.ndarray, seeds: Sequence[int] = (0, 1, 2, 3, 4),
             test_frac: float = 0.3) -> Dict[str, Dict[str, float]]:
    """Full-label probe performance (mean over seeds), per probe."""
    y = _validate_labels(y)
    X = _validate_features(X, y)
    seeds = _validate_seeds(seeds)
    acc: Dict[str, List[Dict[str, float]]] = {}
    for s in seeds:
        tr, te = split(y, s, test_frac)
        for name, clf in _probes(s).items():
            acc.setdefault(name, []).append(
                _fit_eval(clf, X[tr], y[tr], X[te], y[te]))
    out = {}
    for name, runs in acc.items():
        out[name] = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
        out[name].update({f"{k}_std": float(np.std([r[k] for r in runs]))
                         for k in runs[0]})
    return out


def label_efficiency(X: np.ndarray, y: np.ndarray,
                     fracs: Sequence[float] = (0.01, 0.05, 0.10, 0.25, 1.00),
                     seeds: Sequence[int] = (0, 1, 2, 3, 4), test_frac: float = 0.3,
                     probe: str = "linear") -> Dict[str, Dict[str, float]]:
    """AUPRC/AUC vs fraction of TRAIN labels used (claim C4). Mean over seeds."""
    y = _validate_labels(y)
    X = _validate_features(X, y)
    seeds = _validate_seeds(seeds)
    fracs = tuple(fracs)
    if not fracs or any(not np.isfinite(f) or not 0 < f <= 1 for f in fracs):
        raise ValueError("Label fractions must be nonempty, finite and in (0, 1]")
    keys = [_fraction_key(f) for f in fracs]
    if len(set(keys)) != len(keys):
        raise ValueError("Label fractions must have distinct output keys")
    if probe not in ("linear", "gbdt"):
        raise ValueError("probe must be linear or gbdt")
    curve: Dict[str, List[Dict[str, float]]] = {}
    for s in seeds:
        tr, te = split(y, s, test_frac)
        ytr = y[tr]
        pos, neg = tr[ytr == 1], tr[ytr == 0]
        rng = np.random.default_rng(s)
        for f in fracs:
            n_pos = max(1, int(round(len(pos) * f)))
            n_neg = max(1, int(round(len(neg) * f)))
            sub = np.concatenate([rng.choice(pos, n_pos, replace=False),
                                  rng.choice(neg, n_neg, replace=False)])
            clf = _probes(s)[probe]
            curve.setdefault(_fraction_key(f), []).append(
                _fit_eval(clf, X[sub], y[sub], X[te], y[te]))
    return {f: {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
            for f, runs in curve.items()}
