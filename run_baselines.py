"""Run the baseline bar (and optionally a CARTA embedding) through the shared probe.

Local, CPU, no GPU needed (sklearn only). Produces the reference AUPRC/label-
efficiency numbers that CARTA must match/complement.

Usage:
  python run_baselines.py --dataset ieee_cis
  python run_baselines.py --dataset ulb_creditcard
  python run_baselines.py --dataset ieee_cis --embeddings results/gater_ieee_cis.npy

All numbers -> results/baselines_<dataset>.json  (golden rule: never hand-edit).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

import baselines
import datasets
import probe
from adapter import split_roles
from experiment_io import load_embedding, dataset_fingerprint, evaluation_fingerprint

RESULTS = os.environ.get("GATER_RESULTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "new-runs"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ieee_cis", choices=list(datasets.LOADERS))
    ap.add_argument("--embeddings", default=None,
                    help="optional .npy of CARTA 16-dim embeddings (rows aligned to df)")
    ap.add_argument("--allow-unverified-embeddings", action="store_true", help="explicitly accept a legacy embedding without row metadata")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--card-threshold", type=int, default=50)
    args = ap.parse_args()
    seeds = tuple(int(s) for s in args.seeds.split(","))

    t0 = time.time()
    df, spec = datasets.load(args.dataset)
    y = df[spec.label].to_numpy().astype(int)
    roles = split_roles(df, spec, args.card_threshold)
    print(f"[{spec.name}] rows={spec.n_rows} fraud={spec.fraud_rate:.4f} "
          f"| roles: {roles.summary()} | cat_card={spec.card}")

    reps = {}
    Xraw = baselines.build_raw_matrix(df, spec, args.card_threshold)
    reps["raw"] = Xraw
    reps["pca16"] = baselines.pca_reduce(Xraw, 16)
    # frequency-encoding baseline: includes high-card cols (fairer for TabFormer)
    if spec.card:   # only datasets that have high-card columns benefit
        Xfreq = baselines.build_freq_matrix(df, spec, args.card_threshold)
        if Xfreq.shape[1] > Xraw.shape[1]:   # only add if it adds high-card cols
            reps["freq"] = Xfreq
    if args.embeddings:
        emb = load_embedding(args.embeddings, df, spec, args.allow_unverified_embeddings)
        reps[f"gater{emb.shape[1]}"] = emb.astype(np.float32)

    out = {"evaluation_fingerprint": evaluation_fingerprint(df, spec), "row_fingerprint": dataset_fingerprint(df, spec), "dataset": spec.name, "n_rows": spec.n_rows, "fraud_rate": spec.fraud_rate,
           "roles": roles.__dict__, "seeds": list(seeds),
           "full_label": {}, "label_efficiency": {}}
    for rep_name, X in reps.items():
        print(f"  -> {rep_name}: dim={X.shape[1]}")
        out["full_label"][rep_name] = probe.evaluate(X, y, seeds)
        out["label_efficiency"][rep_name] = probe.label_efficiency(X, y, seeds=seeds)
        fl = out["full_label"][rep_name]
        print(f"    AUPRC linear={fl['linear']['auprc']:.4f} "
              f"gbdt={fl['gbdt']['auprc']:.4f} | AUC gbdt={fl['gbdt']['auc']:.4f}")

    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, f"baselines_{spec.name}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[done in {time.time()-t0:.1f}s] wrote {path}")


if __name__ == "__main__":
    main()
