"""Cluster stability analysis runs locally, no GPU needed.

Loads the frozen CARTA embeddings (gater_ieee_cis.npy), runs K-means 10 times
with different seeds, computes pairwise ARI to measure cluster stability.

Usage:
  python cluster_stability.py
  python cluster_stability.py --emb results/gater_ieee_cis.npy --k 3 --runs 10
Outputs: results/cluster_stability.json
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
RESULTS_DIR = os.environ.get("GATER_RESULTS", os.path.join(ROOT, "results", "new-runs"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default=os.path.join(RESULTS_DIR, "gater_ieee_cis.npy"))
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--runs", type=int, default=10)
    args = ap.parse_args()

    emb = np.load(args.emb, allow_pickle=False)
    if emb.ndim != 2 or not all(emb.shape) or not np.isfinite(emb).all():
        raise ValueError("Embedding must be a nonempty finite 2D array")
    if args.runs < 2 or not 1 <= args.k <= len(emb):
        raise ValueError("Require at least two runs and 1 <= k <= the number of rows")
    emb = emb.astype(np.float32)
    Z = StandardScaler().fit_transform(emb)
    print(f"embeddings: {emb.shape}  k={args.k}  runs={args.runs}")

    labels_list = []
    for seed in range(args.runs):
        km = KMeans(n_clusters=args.k, random_state=seed, n_init=5)
        labels_list.append(km.fit_predict(Z))
        print(f"  run {seed}: done", flush=True)

    # pairwise ARI
    aris = []
    for i in range(args.runs):
        for j in range(i + 1, args.runs):
            aris.append(float(adjusted_rand_score(labels_list[i], labels_list[j])))
    mean_ari = float(np.mean(aris))
    std_ari = float(np.std(aris))
    print(f"\nPairwise ARI: mean={mean_ari:.4f}  std={std_ari:.4f}")
    print("Interpretation: ARI>0.8 = highly stable, 0.5-0.8 = moderate, <0.5 = unstable")

    out = {"k": args.k, "runs": args.runs, "n_pairs": len(aris),
           "mean_ari": mean_ari, "std_ari": std_ari, "all_aris": aris}
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "cluster_stability.json")
    with open(path, "w") as handle:
        json.dump(out, handle, indent=2, allow_nan=False)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
