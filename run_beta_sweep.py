"""Compare fixed-beta and scheduled-KL configurations under feature excision. This runner produces exploratory configuration results; test-set rankings must not be used alone to choose a final recipe."""
import os, sys, json, time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import datasets, probe
from gater_core import train_and_encode
from experiment_io import evaluation_fingerprint

RESULTS = os.environ.get("GATER_RESULTS",
                         os.path.join(_HERE, "results", "new-runs"))
DS = os.environ.get("DATASET", "tabformer")
EPOCHS = int(os.environ.get("EPOCHS", "300"))
BATCH = int(os.environ.get("BATCH", "16384"))
SEED = int(os.environ.get("SEED", "0"))
PROBE_SEEDS = tuple(int(s) for s in os.environ.get("PROBE_SEEDS", "0,1,2,3,4").split(","))

# KL settings, ALL under feature-wise excision (the canonical excision mode).
# {} = default cyclical annealing; use_cyclical_annealing=False = fixed beta at the
# configured max; kl_weight_max sets the fixed beta explicitly.
ALL_CONFIGS = {
    "cyclical":      {},                                               # reference (feature + cyclical)
    "fixed_default": {"use_cyclical_annealing": False},                  # feature + no-cyclical (the missing combo)
    "beta0":         {"use_cyclical_annealing": False, "kl_weight_max": 0.0},
    "beta0p1":       {"use_cyclical_annealing": False, "kl_weight_max": 0.1},
    "beta1":         {"use_cyclical_annealing": False, "kl_weight_max": 1.0},
}
NAMES = os.environ.get("CONFIGS", ",".join(ALL_CONFIGS)).split(",")


def main():
    df, spec = datasets.load(DS)
    y = df[spec.label].to_numpy().astype(int)
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, f"beta_sweep_{DS}.json")
    protocol = {"dataset": DS, "excision": "feature", "seed": SEED,
                "probe_seeds": list(PROBE_SEEDS), "epochs": EPOCHS, "batch": BATCH,
                "evaluation_fingerprint": evaluation_fingerprint(df, spec)}
    out = dict(protocol, configs={})
    if os.path.exists(path):
        with open(path) as handle:
            existing = json.load(handle)
        if existing.get("configs") and any(existing.get(key) != value for key, value in protocol.items()):
            raise ValueError("Existing beta sweep protocol/population differs or is unverified; use a separate output directory")
        out["configs"] = existing.get("configs", {})
    out.setdefault("configs", {})

    for name in NAMES:
        cfg = ALL_CONFIGS[name]
        t0 = time.time()
        emb, encoding_spec = train_and_encode(DS, epochs=EPOCHS, encoder="perceiver",
                                  use_contrastive=True, excision_mode="feature",
                                  train_overrides=cfg, seed=SEED, batch_size=BATCH)
        full = probe.evaluate(emb, y, PROBE_SEEDS)
        le = probe.label_efficiency(emb, y, seeds=PROBE_SEEDS)
        out["configs"][name] = {
            "kl_overrides": cfg,
            "encoding_protocol": encoding_spec.encoding_protocol,
            "lin_auprc": round(full["linear"]["auprc"], 4),
            "gbdt_auprc": round(full["gbdt"]["auprc"], 4),
            "gbdt_auc": round(full["gbdt"]["auc"], 4),
            "le_onepc": round(le["0.01"]["auprc"], 4),
            "sec": round(time.time() - t0, 1),
        }
        with open(path, "w") as handle:
            json.dump(out, handle, indent=2, allow_nan=False)
        print(f"[{name}] lin={out['configs'][name]['lin_auprc']} "
              f"gbdt={out['configs'][name]['gbdt_auprc']} ({out['configs'][name]['sec']}s)", flush=True)

    # report winners (by linear probe = representation quality, and by gbdt = task ceiling)
    cfgs = out["configs"]
    best_lin = max(cfgs, key=lambda k: cfgs[k]["lin_auprc"])
    best_gbdt = max(cfgs, key=lambda k: cfgs[k]["gbdt_auprc"])
    print(f"\nwrote {path}")
    print(f"best linear: {best_lin} ({cfgs[best_lin]['lin_auprc']})   "
          f"best gbdt: {best_gbdt} ({cfgs[best_gbdt]['gbdt_auprc']})")
    print("== BETA SWEEP DONE ==")


if __name__ == "__main__":
    main()
