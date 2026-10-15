"""Train CARTA self-supervised on a PUBLIC dataset and dump a frozen embedding.

Runs on a single GPU (needs jax + flax + the model package in src/).
`datasets.py` yields clean public frames and `adapter.split_roles` sets the
low/high-cardinality split. Everything else (power-law preconditioning,
tokenization, VAE, token excision + InfoNCE + cyclical KL) lives in src/,
driven via `gater_core.train_and_encode`.

Output: results/gater_<dataset>[_<encoder>].npy – one latent-mean row per input row,
in ORIGINAL df order, scored with the SAME probe:
  python run_baselines.py --dataset ieee_cis --embeddings results/gater_ieee_cis.npy

SMOKE TEST (CPU, meaningless numbers, only checks the pipeline):
  python gater_runner.py --dataset ieee_cis --epochs 2 --subsample 20000 --batch-size 2048
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)  # src/ (model package) + flat modules live here
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import datasets  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ieee_cis", choices=list(datasets.LOADERS))
    ap.add_argument("--encoder", default="perceiver", choices=["perceiver", "gru", "mlp"])
    ap.add_argument("--no-powerlaw", action="store_true",
                    help="skip SmartPowerlawTransformer (use for all-numeric PCA datasets)")
    ap.add_argument("--excision-mode", default="feature", choices=["cell", "feature"],
                    help="feature = manuscript default; cell = within-table ablation reference")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=16384)
    ap.add_argument("--card-threshold", type=int, default=50)
    ap.add_argument("--subsample", type=int, default=0, help="rows for smoke tests (0=all)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    from gater_core import train_and_encode

    emb, spec = train_and_encode(
        args.dataset, epochs=args.epochs, encoder=args.encoder,
        excision_mode=args.excision_mode, use_powerlaw=not args.no_powerlaw,
        card_threshold=args.card_threshold,
        subsample=args.subsample, seed=args.seed, batch_size=args.batch_size)

    suffix = "" if args.encoder == "perceiver" else f"_{args.encoder}"
    out = args.out or os.path.join(os.environ.get("GATER_RESULTS", os.path.join(_HERE, "results", "new-runs")), f"gater_{spec.name}{suffix}.npy")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    from experiment_io import save_embedding
    save_embedding(out, emb, spec)
    print(f"[gater:{spec.name}/{args.encoder}] saved {emb.shape} -> {out}")


if __name__ == "__main__":
    main()
