"""Generate all paper figures from the results JSONs.
Run from the paper/ directory:  python gen_figures.py
Outputs: fig_label_efficiency.pdf, fig_architecture.pdf
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch

HERE   = os.path.dirname(os.path.abspath(__file__))
RESULT = os.environ.get("GATER_RESULTS", os.path.join(HERE, "results"))

def load(name):
    with open(os.path.join(RESULT, name)) as f:
        return json.load(f)

ieee_base = load("baselines_ieee_cis.json")
tf_base   = load("baselines_tabformer.json")

# ── label-efficiency figure [UNKNOWN: exact count of trailing decorative rule characters]
fracs   = [0.01, 0.05, 0.10, 0.25, 1.00]
xlabels = ["1%", "5%", "10%", "25%", "100%"]

def le_curve(baseline_json, rep):
    le = baseline_json["label_efficiency"][rep]
    return [le[f"{f:.2f}"]["auprc"] for f in fracs]

# per-seed std for error bars (from compute_labeleff_std.py), if available
_STD_PATH = os.path.join(RESULT, "labeleff_std.json")
_STD = json.load(open(_STD_PATH)) if os.path.exists(_STD_PATH) else None

def le_std(ds, rep):
    if not _STD:
        return None
    s = _STD[ds][rep]
    return [s[f"{f:.2f}"]["auprc_std"] for f in fracs]

ieee_raw   = le_curve(ieee_base, "raw")
ieee_pca   = le_curve(ieee_base, "pca16")
ieee_gater = le_curve(ieee_base, "gater16")
tf_raw     = le_curve(tf_base,   "raw")
tf_pca     = le_curve(tf_base,   "pca16")
tf_gater   = le_curve(tf_base,   "gater16")
ieee_err = {"Raw": le_std("ieee_cis", "raw"), "PCA-16": le_std("ieee_cis", "pca16"), "CARTA-16": le_std("ieee_cis", "gater16")}
tf_err   = {"Raw": le_std("tabformer", "raw"), "PCA-16": le_std("tabformer", "pca16"), "CARTA-16": le_std("tabformer", "gater16")}

fig, ax_le = plt.subplots(1, 1, figsize=(4.6, 3.0))
plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False})

COLORS = {"CARTA-16": "#2166ac", "Raw": "#d6604d", "PCA-16": "#878787"}
MARKS  = {"CARTA-16": "o",        "Raw": "s",      "PCA-16": "^"}

def draw(ax, raw, pca, gater, title, ymin, ymax, err=None):
    x = np.arange(len(fracs))
    for name, vals in [("CARTA-16", gater), ("Raw", raw), ("PCA-16", pca)]:
        ls = "-" if name == "CARTA-16" else "--"
        lw = 1.8 if name == "CARTA-16" else 1.2
        yerr = err.get(name) if err else None
        ax.errorbar(x, vals, yerr=yerr, color=COLORS[name], marker=MARKS[name],
                    markersize=4, linewidth=lw, linestyle=ls, label=name,
                    capsize=2, elinewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(xlabels)
    ax.set_xlabel("Fraction of training labels")
    ax.set_ylabel("AUPRC (linear probe)")
    ax.set_title(title, fontweight="bold", fontsize=9)
    ax.set_ylim(ymin, ymax)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)

draw(ax_le, ieee_raw, ieee_pca, ieee_gater,
     "IEEE-CIS (3.5% fraud)", ymin=0.07, ymax=0.24, err=ieee_err)

handles, labels = ax_le.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3,
           bbox_to_anchor=(0.5, -0.04), frameon=False, fontsize=8)

plt.tight_layout(rect=[0, 0.06, 1, 1])
fig.savefig(os.path.join(HERE, "fig_label_efficiency.pdf"),
            bbox_inches="tight", dpi=200)
print("saved fig_label_efficiency.pdf")

# ── architecture diagram [UNKNOWN: exact count of trailing decorative rule characters]
# Box geometry helpers: all coordinates in data-unit space
# Each box: (cx, cy, w, h). Arrow connects right-edge to left-edge.
def bx(cx, w): return cx - w/2   # left x
def rx(cx, w): return cx + w/2   # right x
GAP = 0.10   # whitespace between arrow tip and box edge

def draw_box(ax, cx, cy, w, h, txt, ec, dashed=False, fs=7):
    ls = "--" if dashed else "-"
    ax.add_patch(FancyBboxPatch(
        (cx-w/2, cy-h/2), w, h, boxstyle="round,pad=0.06",
        lw=1.0, linestyle=ls, edgecolor=ec,
        facecolor=ec+"1A", zorder=2))        # 1A = 10% opacity hex
    ax.text(cx, cy, txt, ha="center", va="center",
            fontsize=fs, zorder=3, multialignment="center", color="#111111")

def h_arrow(ax, x1, x2, y, col="#555555", lw=1.0):
    ax.annotate("", xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="-|>", color=col,
                                lw=lw, mutation_scale=9))

def v_arrow(ax, x, y1, y2, col, dashed=False, lw=0.9):
    ls = "--" if dashed else "-"
    ax.annotate("", xy=(x, y2), xytext=(x, y1),
                arrowprops=dict(arrowstyle="-|>", color=col, lw=lw,
                                linestyle=ls, mutation_scale=8))

fig2, ax2 = plt.subplots(figsize=(9.0, 3.5))
ax2.set_xlim(-0.2, 13.8)
ax2.set_ylim(-1.55, 2.4)
ax2.axis("off")

# Palette
CI = "#777777"   # input
CT = "#1565c0"   # tokenization (blue)
CE = "#d84315"   # excision (orange-red)
CP = "#00695c"   # perceiver (teal)
CV = "#4a148c"   # VAE (purple)
CD = "#1b5e20"   # downstream (green)
CL = "#b71c1c"   # decoder/loss (red)

# Stage centers and widths
# input→tok gap tightened (was 3.2, now 2.95); raw box narrowed (1.7→1.5)
SX = {"i": 1.0, "t": 2.95, "e": 5.1, "p": 7.3, "v": 9.45, "d": 11.75}
SW = {"i": 1.5, "t": 1.3,  "e": 1.3, "p": 1.65,"v": 1.65,  "d": 1.65}
SH_s = 0.46    # small box
SH_t = 1.72    # tall box
YR   = [1.65, 1.0, 0.35]   # three-row y positions

# Col 1: raw input
for y, lbl in zip(YR, ["Continuous\n(amt, card1…)", "Low-card\ncats (OHE)",
                       "High-card cats\n(merchant, zip)"]):
    draw_box(ax2, SX["i"], y, SW["i"], SH_s, lbl, CI, fs=7.5)
ax2.text(SX["i"], -0.28, "Raw features", ha="center", fontsize=7, color="gray")

# Col 2: tokenization
for y, lbl in zip(YR, ["Dense proj.", "Dense proj.", "Embed table\n×$n_e$"]):
    draw_box(ax2, SX["t"], y, SW["t"], SH_s, lbl, CT, fs=7.5)
# widen tokenization boxes slightly so "Low-card cats (OHE)" fits – done via fs already
ax2.text(SX["t"], -0.28, "Heterogeneous\ntokenization", ha="center", fontsize=7, color="gray")
for y in YR:
    h_arrow(ax2, rx(SX["i"], SW["i"])+GAP, bx(SX["t"], SW["t"])-GAP, y)

# Col 3: token excision (tall, dashed)
draw_box(ax2, SX["e"], 1.0, SW["e"], SH_t,
         "Token\nexcision\n(p=0.2)\n[train only]", CE, dashed=True, fs=7.5)
for y in YR:
    h_arrow(ax2, rx(SX["t"], SW["t"])+GAP, bx(SX["e"], SW["e"])-GAP, y)

# Col 4: Perceiver (tall)
draw_box(ax2, SX["p"], 1.0, SW["p"], SH_t,
         "Perceiver\nencoder\nQ=8, L=4\nO(Q·N)", CP, fs=7.5)
h_arrow(ax2, rx(SX["e"], SW["e"])+GAP, bx(SX["p"], SW["p"])-GAP, 1.0)

# Col 5: VAE (tall)
draw_box(ax2, SX["v"], 1.0, SW["v"], SH_t,
         "VAE\nbottleneck\nμ,σ²\nz∈R¹⁶", CV, fs=7.5)
h_arrow(ax2, rx(SX["p"], SW["p"])+GAP, bx(SX["v"], SW["v"])-GAP, 1.0)

# Col 6: downstream. Fraud detection is evaluated in this paper; drift
# monitoring and forecasting are marked future work (not evaluated here).
for y, lbl in zip(YR, ["Fraud detection", "Drift monitoring\n(future work)",
                       "Forecasting\n(future work)"]):
    draw_box(ax2, SX["d"], y, SW["d"], SH_s, lbl, CD, fs=7.5)
ax2.text(SX["d"], -0.28, "Downstream\n(frozen z)", ha="center", fontsize=7, color="gray")
for y in YR:
    h_arrow(ax2, rx(SX["v"], SW["v"])+GAP, bx(SX["d"], SW["d"])-GAP, y)

# Decoder training loop – placed well below main boxes for long visible arrows
dec_cy  = -0.62          # decoder box centre y
dec_h   = 0.46
dec_cx  = (SX["p"] + SX["v"]) / 2
dec_w   = rx(SX["v"], SW["v"]) - bx(SX["p"], SW["p"]) + 0.3
box_bot = 1.0 - SH_t/2   # bottom edge of Perceiver/VAE boxes ≈ 0.14
dec_top = dec_cy + dec_h/2

draw_box(ax2, dec_cx, dec_cy, dec_w, dec_h,
         "Multi-head decoder  [training only]", CL, dashed=True, fs=7.5)
ax2.text(dec_cx, dec_cy - dec_h/2 - 0.22,
         "Loss = VAE reconstruction  +  InfoNCE contrastive  +  cyclical KL",
         ha="center", fontsize=7.5, color=CL)

# Long red dashed arrows: VAE box-bottom → decoder top; decoder top → Perceiver box-bottom
v_arrow(ax2, SX["v"], box_bot - 0.04, dec_top + 0.04, CL, dashed=True)
v_arrow(ax2, SX["p"], dec_top + 0.04, box_bot - 0.04, CL, dashed=True)

# ── legend [UNKNOWN: exact count of trailing decorative rule characters]
lx, ly = 0.0, -0.95   # legend top-left anchor (data coords)
row = 0.38           # vertical gap between legend rows
# legend swatches – use plain Rectangle (no FancyBbox alpha issues)
import matplotlib.patches as mpatches
ax2.add_patch(mpatches.Rectangle(
    (lx, ly-0.18), 0.38, 0.22, lw=1.1, linestyle="-",
    edgecolor="#555555", facecolor="#eeeeee", zorder=4))
ax2.text(lx+0.50, ly-0.07, "Training + inference",
         va="center", fontsize=7.5, color="#333333")

ax2.add_patch(mpatches.Rectangle(
    (lx, ly-0.18-row), 0.38, 0.22, lw=1.2, linestyle="--",
    edgecolor="#b71c1c", facecolor="#fce8e8", zorder=4))
ax2.text(lx+0.50, ly-0.07-row, "Training only (discarded at inference)",
         va="center", fontsize=7.5, color="#b71c1c")

fig2.tight_layout(pad=0.1)
fig2.savefig(os.path.join(HERE, "fig_architecture.pdf"),
             bbox_inches="tight", dpi=250)
print("saved fig_architecture.pdf")

# ── representation quality vs. compute (unsupervised comparison) [UNKNOWN: exact count of trailing decorative rule characters]
# Quantitative view: linear-probe AUPRC (quality, y) against training wall-time
# (compute cost, x, LINEAR axis). CARTA and SCARF share the top quality tier;
# CARTA sits to SCARF's left (same quality, less compute). PCA trains in ≈0 s, so
# it is drawn as a left-pointing arrow off the left edge rather than on-scale.
rc = load("representation_comparison_ieee_cis.json")["methods"]
# umap16 is included automatically once a run adds it (else silently skipped).
RC_DISP = [("umap16", "UMAP-16"), ("ae", "AE"), ("sae", "SAE"),
           ("vanilla_vae", "VAE"), ("scarf", "SCARF"), ("gater", "CARTA-16")]
RC_COL = {"CARTA-16": "#2166ac", "SCARF": "#b2182b", "PCA-16": "#878787",
          "AE": "#878787", "SAE": "#878787", "VAE": "#878787", "UMAP-16": "#878787"}
# per-point label offsets (pts): (dx, dy, ha, va). CARTA/SCARF centred above.
RC_OFF = {"AE": (8, 1, "left", "center"), "SAE": (8, 1, "left", "center"),
          "VAE": (8, -1, "left", "center"), "UMAP-16": (-9, 1, "right", "center"),
          "CARTA-16": (0, 11, "center", "bottom"), "SCARF": (0, 11, "center", "bottom")}

fig3, ax3 = plt.subplots(figsize=(4.8, 3.0))
for key, disp in RC_DISP:
    if key not in rc or rc[key].get("train_seconds") is None:
        continue
    m = rc[key]
    x, y, yerr = m["train_seconds"], m["lin_auprc"], m.get("lin_auprc_std", 0.0)
    big = disp in ("CARTA-16", "SCARF")
    ax3.errorbar(x, y, yerr=yerr, marker="o", markersize=8 if big else 5,
                 color=RC_COL[disp], ecolor=RC_COL[disp],
                 markeredgecolor="white", markeredgewidth=0.8 if big else 0.4,
                 capsize=2, elinewidth=0.8, zorder=3 if big else 2)
    dx, dy, ha, va = RC_OFF[disp]
    ax3.annotate(disp, (x, y), textcoords="offset points", xytext=(dx, dy),
                 ha=ha, va=va, fontsize=8,
                 fontweight="bold" if big else "normal", color=RC_COL[disp])

# axis range over the trained methods, with headroom on the left for the PCA arrow
xs = [rc[k]["train_seconds"] for k, _ in RC_DISP
      if k in rc and rc[k].get("train_seconds") is not None]
xmax = max(xs) * 1.12
xmin = -xmax * 0.10                        # a little negative room for the arrow
ax3.set_xlim(xmin, xmax)
ax3.set_ylim(0.03, 0.24)          # floor low enough for UMAP-16 (~0.05, near base rate)

# PCA: trains in <1 s (off the left edge) -> left-pointing arrow at its AUPRC
pca_y = rc["pca16"]["lin_auprc"]
tip = xmin * 0.55                         # arrow head near the left edge
ax3.annotate("PCA-16\n($<$1 s)", xy=(tip, pca_y), xytext=(xmax * 0.14, pca_y),
             textcoords="data", ha="left", va="center", fontsize=8, color="#878787",
             arrowprops=dict(arrowstyle="-|>", color="#878787", lw=1.1),
             annotation_clip=False)

ax3.set_xlabel("Training wall-clock time (s)")
ax3.set_ylabel("AUPRC (linear probe)")
ax3.set_title("Representation quality vs. compute (IEEE-CIS)",
              fontweight="bold", fontsize=9)
ax3.grid(alpha=0.3, linewidth=0.5)
ax3.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
fig3.tight_layout()
fig3.savefig(os.path.join(HERE, "fig_repcomp.pdf"), bbox_inches="tight", dpi=200)
print("saved fig_repcomp.pdf")
