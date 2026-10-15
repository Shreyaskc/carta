"""Train independent CARTA encoders and aggregate frozen-probe estimates across training seeds. Probe seeds are averaged separately within each encoder run. Output records must preserve both seed populations and configuration overrides."""
import os, sys, json, time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import datasets, probe
from gater_core import train_and_encode
from experiment_io import save_embedding, validate_reference_overrides, require_baseline_compatibility, dataset_fingerprint, evaluation_fingerprint

RESULTS = os.environ.get("GATER_RESULTS",
                         os.path.join(_HERE, "results", "new-runs"))
DS = os.environ.get("DATASET", "ieee_cis")
EPOCHS = int(os.environ.get("EPOCHS", "300"))
BATCH = int(os.environ.get("BATCH", "16384"))
EXCISION = os.environ.get("EXCISION", "feature")       # canonical config
SEEDS = tuple(int(s) for s in os.environ.get("SEEDS", "0,1,2,3,4").split(","))
PROBE_SEEDS = tuple(int(s) for s in os.environ.get("PROBE_SEEDS", "0,1,2,3,4").split(","))
# Optional explicit TrainingConfig overrides as JSON. The manuscript default
# retains cyclical KL; fixed beta is a separate ablation, not a selected winner.

TRAIN_OVERRIDES = json.loads(os.environ.get("TRAIN_OVERRIDES", "{}"))

def agg(points, key):
    """mean/std across the per-training-seed point estimates for a metric."""
    vals = [p[key] for p in points]
    return float(np.mean(vals)), float(np.std(vals))


def main():
    validate_reference_overrides(TRAIN_OVERRIDES)
    df, spec = datasets.load(DS)
    y = df[spec.label].to_numpy().astype(int)
    baseline_path = os.path.join(RESULTS, f"baselines_{DS}.json")
    if os.path.exists(baseline_path):
        with open(baseline_path) as handle:
            require_baseline_compatibility(json.load(handle), df, spec, PROBE_SEEDS)
    os.makedirs(RESULTS, exist_ok=True)
    emb_dir = os.path.join(RESULTS, "reference_seeds")
    os.makedirs(emb_dir, exist_ok=True)

    lin_pts, gbdt_pts, le_pts = [], [], []          # one point estimate per training seed
    for s in SEEDS:
        t0 = time.time()
        emb, encoding_spec = train_and_encode(DS, epochs=EPOCHS, encoder="perceiver",
                                  use_contrastive=True, excision_mode=EXCISION,
                                  train_overrides=TRAIN_OVERRIDES,
                                  seed=s, batch_size=BATCH)
        save_embedding(os.path.join(emb_dir, f"gater_{DS}_seed{s}.npy"), emb, encoding_spec)
        full = probe.evaluate(emb, y, PROBE_SEEDS)    # mean over probe seeds
        le = probe.label_efficiency(emb, y, seeds=PROBE_SEEDS)
        lin_pts.append(full["linear"]); gbdt_pts.append(full["gbdt"])
        le_pts.append({f: v["auprc"] for f, v in le.items()})
        print(f"[seed {s}] lin={full['linear']['auprc']:.4f} gbdt={full['gbdt']['auprc']:.4f} "
              f"({round(time.time()-t0,1)}s)", flush=True)

    # aggregate ACROSS training seeds -> mean +/- training-seed std
    lin_m, lin_s = agg(lin_pts, "auprc"); lin_am, lin_as = agg(lin_pts, "auc")
    gb_m, gb_s = agg(gbdt_pts, "auprc"); gb_am, gb_as = agg(gbdt_pts, "auc")
    fracs = sorted(le_pts[0].keys())
    le_out = {f: {"auprc": float(np.mean([p[f] for p in le_pts])),
                 "auprc_std": float(np.std([p[f] for p in le_pts]))} for f in fracs}

    gater_row = {
        "linear": {"auprc": lin_m, "auprc_std": lin_s, "auc": lin_am, "auc_std": lin_as},
        "gbdt":   {"auprc": gb_m,  "auprc_std": gb_s,  "auc": gb_am,  "auc_std": gb_as},
        "n_train_seeds": len(SEEDS), "excision": EXCISION,
        "std_kind": "training-seed",          # explicit: NOT probe-seed noise
    }

    prov = {"runner": "run_reference_seeds.py", "config": EXCISION,
            "train_overrides": TRAIN_OVERRIDES, "train_seeds": list(SEEDS),
            "probe_seeds": list(PROBE_SEEDS), "epochs": EPOCHS, "batch": BATCH}
    # OUT_TAG set -> standalone file (excision-mode comparison, does NOT touch baselines).
    # OUT_TAG unset -> merge ONLY the gater16 row + label-efficiency into baselines_<ds>.json.
    OUT_TAG = os.environ.get("OUT_TAG", "")
    if OUT_TAG:
        path = os.path.join(RESULTS, f"refseeds_{OUT_TAG}_{DS}.json")
        base = {"full_label": {}, "label_efficiency": {}}
    else:
        path = os.path.join(RESULTS, f"baselines_{DS}.json")
        base = json.load(open(path)) if os.path.exists(path) else {"full_label": {}, "label_efficiency": {}}
    base.setdefault("dataset", DS)
    base.setdefault("seeds", list(PROBE_SEEDS))
    base.setdefault("row_fingerprint", dataset_fingerprint(df, spec))
    base.setdefault("evaluation_fingerprint", evaluation_fingerprint(df, spec))
    base.setdefault("full_label", {})["gater16"] = gater_row
    base.setdefault("label_efficiency", {})["gater16"] = le_out
    base["gater16_provenance"] = prov
    json.dump(base, open(path, "w"), indent=2)

    print(f"\n== CANONICAL {EXCISION} over {len(SEEDS)} training seeds ==")
    print(f"  linear AUPRC = {lin_m:.4f} +/- {lin_s:.4f} (training-seed std)")
    print(f"  gbdt   AUPRC = {gb_m:.4f} +/- {gb_s:.4f}   auc = {gb_am:.4f}")
    print(f"  LE@1% = {le_out.get('0.01',{}).get('auprc')}")
    print(f"  updated gater16 row in {path}")
    print("== REFERENCE SEEDS DONE ==")


if __name__ == "__main__":
    main()
