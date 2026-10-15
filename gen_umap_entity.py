"""Generate UMAP visualization and entity-profiling figures from frozen embeddings.
Run: python gen_umap_entity.py
Outputs: fig_umap.pdf, fig_entity_profiles.pdf
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import umap

HERE   = os.path.dirname(os.path.abspath(__file__))
ROOT   = HERE
sys.path.insert(0, ROOT)

import datasets
from experiment_io import load_embedding

# — load data ────────────────────────────────────────────────────────────
print("loading embeddings + labels …")
EMB_DIR = os.environ.get("GATER_RESULTS", os.path.join(ROOT, "results", "new-runs"))
emb_path = os.path.join(EMB_DIR, "gater_ieee_cis.npy")
df, spec = datasets.load("ieee_cis")
emb = load_embedding(emb_path, df, spec, os.environ.get("ALLOW_UNVERIFIED_EMBEDDINGS") == "1").astype(np.float32)
if len(emb) < 8:
    raise ValueError("Entity visualization requires at least eight aligned rows")
y = df[spec.label].to_numpy().astype(int)            # 0/1 fraud label

rng = np.random.default_rng(42)

# — UMAP projection ──────────────────────────────────────────────────────
print("fitting UMAP on 40k sample …")
N_VIZ = min(40_000, len(emb))
idx = rng.choice(len(emb), N_VIZ, replace=False)
emb_sub, y_sub = emb[idx], y[idx]

reducer = umap.UMAP(n_components=2, n_neighbors=min(30, len(idx) - 1), min_dist=0.1,
                   metric="cosine", random_state=42, n_jobs=1)
Z2 = reducer.fit_transform(emb_sub)

# — Figure 1: UMAP of the CARTA-16 latent (qualitative) ────────────────────
print("generating UMAP figure …")
C_OK  = "#aec7e8"   # non-fraud (light blue)
C_FR  = "#d62728"   # fraud (red)
ALPHA = 0.18
MS    = 2.5

fig, ax = plt.subplots(1, 1, figsize=(4.4, 3.2))
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})

mask_ok = y_sub == 0
ax.scatter(Z2[mask_ok, 0], Z2[mask_ok, 1],
           c=C_OK, s=MS, alpha=ALPHA, rasterized=True, linewidths=0)
ax.scatter(Z2[~mask_ok, 0], Z2[~mask_ok, 1],
           c=C_FR, s=MS*2, alpha=0.6, rasterized=True, linewidths=0)
ax.set_title("CARTA-16 latent (UMAP projection)", fontsize=8.5, fontweight="bold")
ax.set_xticks([]); ax.set_yticks([])
ax.set_xlabel("Component 1", fontsize=7)
ax.set_ylabel("Component 2", fontsize=7)

legend_handles = [
    mpatches.Patch(color=C_FR, label=f"Fraud (n={y_sub.sum():,})"),
    mpatches.Patch(color=C_OK, label=f"Non-fraud (n={(~y_sub.astype(bool)).sum():,})"),
]
fig.legend(handles=legend_handles, loc="lower center", ncol=2,
           bbox_to_anchor=(0.5, -0.03), frameon=False, fontsize=8)

plt.tight_layout(rect=[0, 0.06, 1, 1])
out1 = os.path.join(HERE, "fig_umap.pdf")
fig.savefig(out1, bbox_inches="tight", dpi=200)
print(f"saved {out1}")
plt.close()

# — Figure 2: entity profiling – K-means behavioral archetypes ────────────
print("running K-means entity profiling …")
# Use all embeddings (no sub-sample) for cluster quality
Z_scaled = StandardScaler().fit_transform(emb)

# Silhouette sweep over k=3..7
best_k, best_sil, best_labels = 5, -1, None
for k in range(3, 8):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    lab = km.fit_predict(Z_scaled)
    sil = silhouette_score(Z_scaled[:20000], lab[:20000], random_state=42)
    print(f"  k={k}: silhouette={sil:.4f}")
    if sil > best_sil:
        best_k, best_sil, best_labels = k, sil, lab

print(f"best k={best_k} (silhouette={best_sil:.4f})")

# Per-cluster stats for the profile chart
labels = best_labels
fraud_rate_per_cluster = []
size_per_cluster = []
mean_emb_per_cluster = []
for c in range(best_k):
    mask = labels == c
    fraud_rate_per_cluster.append(y[mask].mean())
    size_per_cluster.append(mask.sum())
    mean_emb_per_cluster.append(emb[mask].mean(axis=0))

# Sort clusters by fraud rate for interpretability
order = np.argsort(fraud_rate_per_cluster)[::-1]
fr_sorted  = [fraud_rate_per_cluster[i] for i in order]
sz_sorted  = [size_per_cluster[i] for i in order]

# — EP* macro values (printed for paper/gen_macros / manual block) ────────
print("=== ENTITY-PROFILE STATS (for EP* macros) ===")
print(f"EPkbest={best_k}  EPsil={best_sil:.4f}  EPoverall={y.mean()*100:.2f}%")
for rank, (fr, sz) in enumerate(zip(fr_sorted, sz_sorted), start=1):
    print(f"  C{rank}: fraud={fr*100:.2f}%  size={sz}")
print(f"EPfraudratio={fr_sorted[0]/fr_sorted[-1]:.2f}x  (C1 {fr_sorted[0]*100:.2f}% / "
      f"C{best_k} {fr_sorted[-1]*100:.2f}%)")
print("=== END EP STATS ===")

CLUSTER_COLORS = ["#d62728","#ff7f0e","#8c564b","#2ca02c","#1f77b4",
                 "#9467bd","#17becf"][:best_k]

fig2, axes2 = plt.subplots(1, 2, figsize=(7.0, 2.8))
plt.rcParams.update({"font.size": 8})

# Left: fraud rate per cluster (bar)
ax = axes2[0]
x = np.arange(best_k)
fr_pct = [r*100 for r in fr_sorted]
overall_pct = y.mean()*100
ymax = max(fr_pct) * 1.35

bars = ax.bar(x, fr_pct, color=CLUSTER_COLORS,
              width=0.6, edgecolor="white", linewidth=0.5)
ax.axhline(overall_pct, color="#555555", linestyle="--",
           linewidth=1.0, label=f"Overall ({overall_pct:.1f}%)")
# annotate bar values
for bar, val in zip(bars, fr_pct):
    ax.text(bar.get_x() + bar.get_width()/2, val + ymax*0.03,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")
ax.set_ylim(0, ymax)
ax.set_xticks(x)
ax.set_xticklabels([f"C{i+1}\n({sz_sorted[i]/1000:.0f}K txns)"
                    for i in range(best_k)], fontsize=7.5)
ax.set_ylabel("Fraud rate (%)", fontsize=8)
ax.set_title("Fraud rate by behavioral archetype", fontsize=8.5, fontweight="bold")
ax.legend(fontsize=7.5, frameon=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Right: cluster sizes
ax2 = axes2[1]
wedge_colors = CLUSTER_COLORS
wedges, texts, autotexts = ax2.pie(
    sz_sorted,
    labels=[f"C{i+1}\n({s/1000:.0f}K)" for i, s in enumerate(sz_sorted)],
    colors=wedge_colors,
    autopct="%1.0f%%", startangle=90,
    textprops={"fontsize": 7.5},
    wedgeprops={"linewidth": 0.5, "edgecolor": "white"})
for at in autotexts:
    at.set_fontsize(7)
ax2.set_title("Transaction volume by archetype", fontsize=8.5, fontweight="bold")

plt.suptitle(f"CARTA latent space: {best_k} behavioral archetypes  "
             f"(silhouette={best_sil:.3f}, K-means on 16-dim embeddings)",
             fontsize=8, y=1.01)
plt.tight_layout()
out2 = os.path.join(HERE, "fig_entity_profiles.pdf")
fig2.savefig(out2, bbox_inches="tight", dpi=200)
print(f"saved {out2}")
