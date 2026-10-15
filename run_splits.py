"""Re-probe frozen pooled embeddings under random and temporal/entity splits.

CPU-only probe diagnostics; these do not retrain preprocessing or the encoder.
A merchant-disjoint probe split can still expose test merchants to pooled
representation fitting and does not establish train-only generalization.

Run: DATASET=ieee_cis SPLIT=temporal python run_splits.py
     DATASET=tabformer SPLIT=merchant python run_splits.py
Env: DATASET, SPLIT, TEST_FRAC, EMB, GATER_RESULTS.
"""
import os, sys, json

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import numpy as np
import pandas as pd
import datasets, baselines, probe
from experiment_io import load_embedding
from sklearn.model_selection import GroupShuffleSplit

RESULTS = os.environ.get("GATER_RESULTS", os.path.join(_HERE, "results", "new-runs"))
DS = os.environ.get("DATASET", "ieee_cis")
SPLIT = os.environ.get("SPLIT", "temporal")
TEST_FRAC = float(os.environ.get("TEST_FRAC", "0.3"))
EMB = os.environ.get("EMB", os.path.join(RESULTS, f"gater_{DS}.npy"))


def temporal_idx(n):
    """last TEST_FRAC of rows in transaction_ts order = test (IEEE-CIS is temporal)."""
    ts = pd.read_parquet(datasets._ieee_path())["transaction_ts"].to_numpy()
    assert len(ts) == n, f"ts rows {len(ts)} != embedding rows {n}"
    order = np.argsort(ts, kind="stable")
    cut = int(round(n * (1 - TEST_FRAC)))
    return order[:cut], order[cut:]


def merchant_idx(df, y, n):
    """GroupShuffleSplit on the merchant id so test merchants are unseen in train."""
    mcol = next((c for c in df.columns if "merchant" in c.lower() and "name" in c.lower()), None)
    mcol = mcol or next((c for c in df.columns if c.lower() in ("merchant_name", "merchant")), None)
    if mcol is None:
        raise KeyError(f"no merchant column in {list(df.columns)}")
    groups = df[mcol].to_numpy()
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_FRAC, random_state=0)
    tr, te = next(gss.split(np.arange(n), y, groups))
    return tr, te


def score(X, tr, te, y):
    out = {}
    for name, clf in probe._probes(0).items():
        out[name] = probe._fit_eval(clf, X[tr], y[tr], X[te], y[te])
    return out


def main():
    df, spec = datasets.load(DS)
    y = df[spec.label].to_numpy().astype(int)
    n = len(y)

    reps = {}
    Xraw = baselines.build_raw_matrix(df, spec)
    reps["raw"] = Xraw
    reps["pca16"] = baselines.pca_reduce(Xraw, 16)
    if spec.card:
        Xfreq = baselines.build_freq_matrix(df, spec)
        if Xfreq.shape[1] > Xraw.shape[1]:
            reps["freq"] = Xfreq
    if os.path.exists(EMB):
        reps["gater16"] = load_embedding(EMB, df, spec, os.environ.get("ALLOW_UNVERIFIED_EMBEDDINGS") == "1").astype(np.float32)
        assert reps["gater16"].shape[0] == n, "embedding/label misalignment"
    else:
        print(f"WARN: embedding {EMB} not found; skipping gater16")

    # random split (seed 0), then a frozen-representation split diagnostic
    rtr, rte = probe.split(y, 0, TEST_FRAC)
    ltr, lte = temporal_idx(n) if SPLIT == "temporal" else merchant_idx(df, y, n)
    print(f"[{DS}/{SPLIT}] n={n} test_frac={TEST_FRAC} "
          f"| random test pos={int(y[rte].sum())} | {SPLIT} test pos={int(y[lte].sum())}")

    out = {"representation_fit": "pooled-feature transductive", "merchant_unseen_scope": "probe only", "dataset": DS, "split": SPLIT, "test_frac": TEST_FRAC, "reps": {}}
    for name, X in reps.items():
        rnd = score(X, rtr, rte, y)
        rob = score(X, ltr, lte, y)
        out["reps"][name] = {"random": rnd, SPLIT: rob}
        print(f"  {name:8s} random  lin={rnd['linear']['auprc']:.4f} gbdt={rnd['gbdt']['auprc']:.4f}"
              f"   {SPLIT} lin={rob['linear']['auprc']:.4f} gbdt={rob['gbdt']['auprc']:.4f}")

    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, f"splits_{DS}_{SPLIT}.json")
    json.dump(out, open(path, "w"), indent=2)
    print(f"wrote {path}\n== SPLITS DONE ==")


if __name__ == "__main__":
    main()
