"""Train one feature-excision CARTA encoder for each dataset and evaluate each frozen embedding over probe seeds. This convenience workflow does not establish independent encoder-training-seed uncertainty."""
import os
import subprocess
import sys
from pathlib import Path

def main():
    HERE = Path(__file__).parent
    RESULTS = Path(os.environ.setdefault("GATER_RESULTS", str(HERE / "results" / "new-runs")))
    os.chdir(HERE)

    EPOCHS = os.environ.get("EPOCHS", "300")
    SEEDS = os.environ.get("SEEDS", "0,1,2,3,4")
    os.environ.setdefault("TABFORMER_SAMPLE", "2000000")
    os.environ.setdefault("TABFORMER_MINFREQ", "100")
    PY = os.environ.get("GATER_PY", sys.executable)


    def run(cmd, label=""):
        print(f"\n{'='*70}\n{label or ' '.join(cmd)}\n{'='*70}", flush=True)
        subprocess.run(cmd, check=True)


    # Confirm a GPU is visible (CPU falls back automatically).
    run([PY, "-c", "import jax; print('JAX devices:', jax.devices())"],
        "JAX device inventory; full CPU training cost is not validated")

    for DS in ("ieee_cis", "tabformer"):
        emb = str(RESULTS / f"gater_{DS}.npy")
        run([PY, "gater_runner.py", "--dataset", DS, "--epochs", EPOCHS,
             "--excision-mode", "feature", "--out", emb],
            f"[{DS}] train C2 (feature-excision) reference encoder -> {emb}")
        run([PY, "run_baselines.py", "--dataset", DS, "--seeds", SEEDS, "--embeddings", emb],
            f"[{DS}] re-score baselines with C2 embedding, 5 seeds (updates gater16 row)")

    print("\n\n== C2 REFERENCE RUN DONE ==", flush=True)
    print("Hand back these (I regenerate Table 1, label-eff fig, UMAP, entity figs):")
    print("  results/gater_ieee_cis.npy        results/gater_tabformer.npy")
    print("  results/baselines_ieee_cis.json   results/baselines_tabformer.json")



if __name__ == "__main__":
    main()
