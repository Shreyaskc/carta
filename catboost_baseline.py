"""CatBoost baseline with NATIVE categorical handling.

The reviewer-requested strong high-cardinality baseline. Unlike the `raw` GBDT
(which one-hot encodes only low-cardinality columns and *drops* merchant name /
city / ZIP because dense one-hot is infeasible), CatBoost ingests ALL
categoricals --- including the high-cardinality identifiers --- via its built-in
ordered target statistics, no one-hot. This is the fair test of "can a strong
off-the-shelf GBDT that natively handles high-cardinality categoricals match
CARTA's learned embeddings?"

Protocol is identical to the other baselines (`probe.py`): the SAME seeded
stratified 70/30 split and the SAME sklearn AUPRC/AUC metrics, so numbers are
directly comparable to Table 1's GBDT columns.

Usage (CPU is fine for IEEE-CIS; use --task-type GPU for TabFormer 2M):
  pip install catboost
  python catboost_baseline.py --dataset ieee_cis  --seeds 0,1,2,3,4
  python catboost_baseline.py --dataset tabformer --seeds 0,1,2,3,4 --task-type GPU

Writes results/catboost_<dataset>.json  (golden rule: never hand-edit).
"""
from __future__ import annotations

import argparse, json, os, time
from typing import Dict, List, Sequence
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
import datasets, probe


RESULTS = os.environ.get("GATER_RESULTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "new-runs"))


def build_frame(df: pd.DataFrame, spec) -> "tuple[pd.DataFrame, List[str]]":
    """Numeric as float (NaN ok --- CatBoost handles it); categoricals as strings
    (all of them, incl. high-cardinality). Returns (feature_frame, cat_feature_names)."""
    X = pd.DataFrame(index=df.index)
    for c in spec.numeric:
        X[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    for c in spec.categorical:
        X[c] = df[c].astype("string").fillna("MISSING").astype(str)
    return X, list(spec.categorical)


def _fit_eval(Xtr, ytr, Xte, yte, cat_features, seed, args) -> Dict[str, float]:
    from catboost import CatBoostClassifier, Pool
    model = CatBoostClassifier(
        iterations=args.iterations,
        depth=args.depth,
        learning_rate=args.learning_rate,
        loss_function="Logloss",
        auto_class_weights="Balanced",  # rare-class fraud; documented in the paper
        random_seed=seed,
        task_type=args.task_type,
        verbose=False,
    )
    train_pool = Pool(Xtr, ytr, cat_features=cat_features)
    model.fit(train_pool)
    p = model.predict_proba(Pool(Xte, cat_features=cat_features))[:, 1]
    return {"auprc": float(average_precision_score(yte, p)),
            "auc": float(roc_auc_score(yte, p))}


def full_label(X, y, cat_features, seeds, args) -> Dict[str, float]:
    runs = []
    for s in seeds:
        tr, te = probe.split(y, s)
        runs.append(_fit_eval(X.iloc[tr], y[tr], X.iloc[te], y[te], cat_features, s, args))
        print(f"    seed {s}: AUPRC={runs[-1]['auprc']:.4f} AUC={runs[-1]['auc']:.4f}", flush=True)
    out = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
    out.update({f"{k}_std": float(np.std([r[k] for r in runs])) for k in runs[0]})
    return out


def label_efficiency(X, y, cat_features, seeds, args,
                     fracs: Sequence[float] = (0.01, 0.05, 0.10, 0.25, 1.00)) -> Dict[str, Dict[str, float]]:
    """CatBoost trained on 1/5/10/25/100% of TRAIN (mirrors probe.label_efficiency's
    subsampling exactly). CatBoost is the classifier here, not a frozen-rep probe."""
    curve: Dict[str, List[Dict[str, float]]] = {}
    for s in seeds:
        tr, te = probe.split(y, s)
        ytr = y[tr]
        pos, neg = tr[ytr == 1], tr[ytr == 0]
        rng = np.random.default_rng(s)
        for f in fracs:
            n_pos = max(1, int(round(len(pos) * f)))
            n_neg = max(1, int(round(len(neg) * f)))
            sub = np.concatenate([rng.choice(pos, n_pos, replace=False),
                                  rng.choice(neg, n_neg, replace=False)])
            curve.setdefault(f"{f:.2f}", []).append(
                _fit_eval(X.iloc[sub], y[sub], X.iloc[te], y[te], cat_features, s, args))
    return {f: {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
            for f, runs in curve.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ieee_cis", choices=list(datasets.LOADERS))
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--iterations", type=int, default=200)  # matches GBDT probe max_iter
    ap.add_argument("--depth", type=int, default=6)        # matches GBDT probe max_depth
    ap.add_argument("--learning-rate", type=float, default=0.1)
    ap.add_argument("--task-type", default="CPU", choices=["CPU", "GPU"])
    ap.add_argument("--no-label-eff", action="store_true", help="skip label-efficiency curve")
    args = ap.parse_args()
    seeds = tuple(int(s) for s in args.seeds.split(","))

    t0 = time.time()
    df, spec = datasets.load(args.dataset)
    y = df[spec.label].to_numpy().astype(int)
    X, cat_features = build_frame(df, spec)
    print(f"[{spec.name}] rows={spec.n_rows} fraud={spec.fraud_rate:.4f} "
          f"| numeric={len(spec.numeric)} categorical={len(cat_features)} "
          f"(native, incl. high-card {list(spec.card)}) | seeds={list(seeds)}", flush=True)

    out = {"dataset": spec.name, "n_rows": spec.n_rows, "fraud_rate": spec.fraud_rate,
           "seeds": list(seeds), "model": "catboost",
           "params": {"iterations": args.iterations, "depth": args.depth,
                      "learning_rate": args.learning_rate,
                      "auto_class_weights": "Balanced"},
           "numeric": list(spec.numeric), "categorical": list(cat_features),
           "full_label": {}, "label_efficiency": {}}

    print("  full-label:", flush=True)
    out["full_label"]["catboost"] = {"gbdt": full_label(X, y, cat_features, seeds, args)}
    fl = out["full_label"]["catboost"]["gbdt"]
    print(f"  == CatBoost AUPRC={fl['auprc']:.4f} +/- {fl['auprc_std']:.4f} | "
          f"AUC={fl['auc']:.4f} ==", flush=True)

    if not args.no_label_eff:
        print("  label-efficiency:", flush=True)
        out["label_efficiency"]["catboost"] = label_efficiency(X, y, cat_features, seeds, args)

    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, f"catboost_{spec.name}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[done in {time.time()-t0:.1f}s] wrote {path}", flush=True)


if __name__ == "__main__":
    main()
