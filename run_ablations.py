"""Ablation orchestrator: train -> encode -> score each ablation, one command.

Cluster env (jax + flax + sklearn). For each RUNNABLE ablation it trains a CARTA
variant, produces the frozen embedding, and scores it with the SHARED probe
(probe.evaluate + label_efficiency) so every row is comparable to the baselines and
to `full`. NEEDS_CODE ablations are listed in the manifest but skipped (with the
reason) until their code variant is implemented.

Usage:
  python run_ablations.py --dataset ieee_cis --epochs 300
  python run_ablations.py --dataset ieee_cis --only full,no_contrastive,gru_encoder --epochs 300
  python run_ablations.py --dataset ieee_cis --only full --epochs 2 --subsample 20000   # smoke

Output: results/ablations_<dataset>.json  (+ per-ablation .npy in results/ablations/)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)  # src/ (model package) + flat modules live here
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


from experiment_io import save_embedding
import ablations  # noqa: E402
import datasets  # noqa: E402
import probe  # noqa: E402
from gater_core import train_and_encode  # noqa: E402

RESULTS = os.environ.get("GATER_RESULTS", os.path.join(_HERE, "results", "new-runs"))
EMB_DIR = os.path.join(RESULTS, "ablations")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ieee_cis", choices=list(datasets.LOADERS))
    ap.add_argument("--only", default="", help="comma list of ablation names (default: all RUNNABLE)")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=16384)
    ap.add_argument("--subsample", type=int, default=0)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    seeds = tuple(int(s) for s in args.seeds.split(","))
    names = args.only.split(",") if args.only else ablations.runnable_names()
    os.makedirs(EMB_DIR, exist_ok=True)

    out = {"dataset": args.dataset, "epochs": args.epochs, "seeds": list(seeds),
           "results": {}, "skipped": {}}
    path = os.path.join(RESULTS, f"ablations_{args.dataset}.json")

    def flush():
        # persist after every ablation so a crash in a long multi-hour run keeps
        # the ablations already completed (single end-of-run write would lose all)
        with open(path, "w") as f:
            json.dump(out, f, indent=2)

    # record skipped NEEDS_CODE ablations transparently
    for ab in ablations.NEEDS_CODE:
        out["skipped"][ab.name] = {"claim": ab.claim, "needs_code": ab.needs_code}

    for name in names:
        ab = ablations.ALL.get(name)
        if ab is None:
            print(f"!! unknown ablation {name!r}; skip"); continue
        if ab.needs_code:
            out["skipped"][name] = {"claim": ab.claim, "needs_code": ab.needs_code}
            print(f"-- {name}: NEEDS CODE ({ab.needs_code}); skip"); continue

        t0 = time.time()
        print(f"== ablation '{name}' [{ab.claim}] encoder={ab.encoder} "
              f"contrastive={ab.use_contrastive} excision={ab.excision_mode} "
              f"model={ab.model} train={ab.train}")
        emb, spec = train_and_encode(
            args.dataset, epochs=args.epochs, encoder=ab.encoder,
            use_contrastive=ab.use_contrastive, excision_mode=ab.excision_mode,
            model_overrides=ab.model, train_overrides=ab.train,
            subsample=args.subsample, batch_size=args.batch_size)
        npy = os.path.join(EMB_DIR, f"{name}__{spec.name}.npy")
        save_embedding(npy, emb, spec)

        df, _ = datasets.load(args.dataset)
        if args.subsample and args.subsample < len(df):
            df = df.sample(args.subsample, random_state=42).reset_index(drop=True)
        y = df[spec.label].to_numpy().astype(int)
        full = probe.evaluate(emb, y, seeds)
        le = probe.label_efficiency(emb, y, seeds=seeds)
        out["results"][name] = {"claim": ab.claim, "encoder": ab.encoder, "dim": int(emb.shape[1]),
                                "note": ab.note, "full_label": full, "label_efficiency": le,
                                "sec": round(time.time() - t0, 1)}
        print(f"  AUPRC linear={full['linear']['auprc']:.4f} gbdt={full['gbdt']['auprc']:.4f} "
              f"| AUC gbdt={full['gbdt']['auc']:.4f}  ({out['results'][name]['sec']}s)")
        flush()

    flush()
    print(f"[done] wrote {path}")


if __name__ == "__main__":
    main()
