"""Recompute label-efficiency AUPRC with per-seed std (for Fig 2 error bars).

Same linear-probe protocol as probe.label_efficiency, but keeps the per-seed
values so we can report a standard deviation at each label fraction. Writes
results/v4/labeleff_std.json = {dataset: {method: {frac: {auprc, auprc_std}}}}.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import datasets, baselines, probe
from experiment_io import load_embedding

RESULT = os.environ.get("GATER_RESULTS", os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "new-runs"))
FRACS = (0.01, 0.05, 0.10, 0.25, 1.00)
SEEDS = (0, 1, 2, 3, 4)


def le_with_std(X, y):
    per = {f"{f:.2f}": [] for f in FRACS}
    for s in SEEDS:
        tr, te = probe.split(y, s)
        ytr = y[tr]
        pos, neg = tr[ytr == 1], tr[ytr == 0]
        rng = np.random.default_rng(s)
        for f in FRACS:
            n_pos = max(1, int(round(len(pos) * f)))
            n_neg = max(1, int(round(len(neg) * f)))
            sub = np.concatenate([rng.choice(pos, n_pos, replace=False),
                                  rng.choice(neg, n_neg, replace=False)])
            r = probe._fit_eval(probe._probes(s)["linear"], X[sub], y[sub], X[te], y[te])
            per[f"{f:.2f}"].append(r["auprc"])
    return {f: {"auprc": float(np.mean(v)), "auprc_std": float(np.std(v))}
            for f, v in per.items()}


def main():
    os.makedirs(RESULT, exist_ok=True)
    out = {}
    for ds in ("ieee_cis", "tabformer"):
        if ds == "tabformer":
            os.environ.setdefault("TABFORMER_SAMPLE", "2000000")
            os.environ.setdefault("TABFORMER_MINFREQ", "100")
        df, spec = datasets.load(ds)
        y = df[spec.label].to_numpy().astype(int)
        raw = baselines.build_raw_matrix(df, spec)
        reps = [("raw", raw), ("pca16", baselines.pca_reduce(raw, 16)),
                ("gater16", load_embedding(os.path.join(RESULT, f"gater_{ds}.npy"), df, spec, os.environ.get("ALLOW_UNVERIFIED_EMBEDDINGS") == "1").astype(np.float32))]
        out[ds] = {}
        for name, X in reps:
            out[ds][name] = le_with_std(X, y)
            print(ds, name, {k: round(v["auprc_std"], 4) for k, v in out[ds][name].items()}, flush=True)
        with open(os.path.join(RESULT, "labeleff_std.json"), "w") as handle:
            json.dump(out, handle, indent=2, allow_nan=False)
    print("wrote labeleff_std.json")


if __name__ == "__main__":
    main()
