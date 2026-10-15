"""Exploratory merchant-entity clustering of a supplied frozen CARTA embedding. Requires matching local dataset rows and embeddings; does not establish merchant-disjoint generalization."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import umap

HERE  = os.path.dirname(os.path.abspath(__file__))
ROOT  = HERE
sys.path.insert(0, ROOT)
os.environ.setdefault("TABFORMER_SAMPLE", "2000000")
os.environ.setdefault("TABFORMER_MINFREQ", "100")

import datasets
from experiment_io import load_embedding

print("Loading TabFormer data + embeddings …")
df, spec = datasets.load("tabformer")
EMB_DIR = os.environ.get("GATER_RESULTS", os.path.join(ROOT, "results", "new-runs"))
emb = load_embedding(os.path.join(EMB_DIR, "gater_tabformer.npy"), df, spec, os.environ.get("ALLOW_UNVERIFIED_EMBEDDINGS") == "1").astype(np.float32)
if len(emb) < 8:
    raise ValueError("Entity visualization requires at least eight aligned rows")
y   = df[spec.label].to_numpy().astype(int)

print(f"emb: {emb.shape}  |  fraud: {y.mean():.4f}")

rng = np.random.default_rng(42)

# ── K-means on subsample (fast), then predict full ────────────────────
print("fitting K-means …")
N_FIT  = min(200_000, len(emb))
fit_idx = rng.choice(len(emb), N_FIT, replace=False)
Z_scaled = StandardScaler().fit_transform(emb)

best_k, best_sil = 5, -1
for k in [4, 5, 6]:
    km = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=5, batch_size=10000)
    lab_fit = km.fit_predict(Z_scaled[fit_idx])
    sil = silhouette_score(Z_scaled[fit_idx[:10000]], lab_fit[:10000], random_state=42)
    print(f"  k={k}: silhouette={sil:.4f}")
    if sil > best_sil:
        best_k, best_sil = k, sil
        best_km = km

print(f"best k={best_k}")
labels = best_km.predict(Z_scaled)

# ── Per-cluster stats ────────────────────────────────────────────────
stats = []
for c in range(best_k):
    mask = labels == c
    top_mcc  = df.loc[mask, "mcc"].value_counts().head(3).index.tolist()
    top_chip = df.loc[mask, "use_chip"].value_counts().index[0]
    fraud_r  = y[mask].mean()
    size     = mask.sum()
    avg_amt  = df.loc[mask, "amount"].mean()
    stats.append(dict(cluster=c, size=size, fraud=fraud_r,
                      top_mcc=top_mcc, chip=top_chip, avg_amt=avg_amt))

# Sort by fraud rate descending
stats.sort(key=lambda d: d["fraud"], reverse=True)
print("\nCluster stats:")
for s in stats:
    print(f"  C{s['cluster']+1}: n={s['size']:,}  fraud={s['fraud']*100:.3f}%  "
          f"avg_amt=${s['avg_amt']:.0f}  chip={s['chip']}  mcc_top={s['top_mcc'][:2]}")

# ── UMAP on 30K subsample, coloured by cluster ────────────────────────
print("fitting UMAP on 30K sample …")
N_VIZ = min(30_000, len(emb))
viz_idx = rng.choice(len(emb), N_VIZ, replace=False)
reducer = umap.UMAP(n_components=2, n_neighbors=min(30, len(viz_idx) - 1), min_dist=0.15,
                   metric="cosine", random_state=42, n_jobs=1)
Z2 = reducer.fit_transform(Z_scaled[viz_idx])
lab_viz = labels[viz_idx]
y_viz   = y[viz_idx]

# ── Figure ──────────────────────────────────────────────────────────
PALETTE = ["#e41a1c","#ff7f00","#4daf4a","#377eb8","#984ea3","#a65628"][:best_k]

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
plt.rcParams.update({"font.size": 8})

# Left: UMAP coloured by cluster
ax = axes[0]
for i, s in enumerate(stats):
    c_orig = s["cluster"]
    mask_c = lab_viz == c_orig
    ax.scatter(Z2[mask_c, 0], Z2[mask_c, 1],
               c=PALETTE[i], s=2, alpha=0.25, rasterized=True, linewidths=0,
               label=f"C{i+1} ({s['size']/1000:.0f}K)")
ax.set_title("CARTA latent — merchant archetypes", fontsize=8.5, fontweight="bold")
ax.set_xticks([]); ax.set_yticks([])
ax.set_xlabel("UMAP-1", fontsize=7); ax.set_ylabel("UMAP-2", fontsize=7)
ax.legend(fontsize=6.5, markerscale=3, frameon=False,
          loc="lower right", ncol=2)

# Right: fraud rate + avg amount per cluster
ax2 = axes[1]
labels_r = [f"C{i+1}" for i in range(best_k)]
fr_vals  = [s["fraud"]*100 for s in stats]
amt_vals = [s["avg_amt"] for s in stats]
overall_fr = y.mean()*100

x = np.arange(best_k)
bars = ax2.bar(x, fr_vals, color=PALETTE, width=0.5, edgecolor="white", lw=0.5)
ax2.axhline(overall_fr, color="#555555", linestyle="--", lw=1.0,
            label=f"Overall ({overall_fr:.3f}%)")
for bar, val in zip(bars, fr_vals):
    ax2.text(bar.get_x()+bar.get_width()/2, val + max(fr_vals)*0.04,
             f"{val:.3f}%", ha="center", va="bottom", fontsize=7, fontweight="bold")
ax2.set_ylim(0, max(fr_vals)*1.4)
ax2.set_xticks(x)
ax2.set_xticklabels([f"C{i+1}\n({s['size']/1000:.0f}K)"
                    for i, s in enumerate(stats)], fontsize=7.5)
ax2.set_ylabel("Fraud rate (%)", fontsize=8)
ax2.set_title("Fraud rate by merchant archetype", fontsize=8.5, fontweight="bold")
ax2.legend(fontsize=7, frameon=False)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

# MCC lookup (common codes only)
MCC_NAME = {"5812":"Restaurants","5411":"Grocery","5541":"Gas stations",
            "5813":"Bars/liquor","5499":"Misc. food","5651":"Clothing",
            "5691":"Clothing stores","5732":"Electronics","5912":"Drug stores"}
def mcc_label(code):
    return MCC_NAME.get(str(code), f"MCC {code}")

hi = stats[0]
hi_mccs = ", ".join(mcc_label(m) for m in hi['top_mcc'][:2])
note = f"C1: {hi['chip']} — {hi_mccs}"
ax2.text(0.02, 0.97, note, transform=ax2.transAxes,
         fontsize=6.5, va="top", color=PALETTE[0], style="italic")

plt.suptitle(
    f"TabFormer: {best_k} merchant archetypes (silhouette={best_sil:.3f}, "
    f"K-means on 16-dim CARTA embeddings, no labels used)",
    fontsize=7.5, y=1.01)
plt.tight_layout()

out = os.path.join(HERE, "fig_tabformer_entity.pdf")
fig.savefig(out, bbox_inches="tight", dpi=200)
print(f"saved {out}")
