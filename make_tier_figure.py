"""Fig. 2 -- the five design tiers, plotted from the real layout coordinates.

Case membership comes from the SPLIT DIRECTORIES, not from the manifest's
`exported` column: that column is stale for tier D (it reads 0 for all 70 rows
while 66 D cases are on disk and in the splits). The manifest is used only for
tier labels and panel geometry.
"""
import os, glob
import numpy as np, pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOM_X, ROOM_Y = 8.80, 6.10
P_SUP, P_RET = 0.591, 0.600
SUP, RET = "#0f4c5c", "#a63a1e"
INK, MUT, LINE = "#16222b", "#657680", "#d3dbdf"

mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5,
    "axes.edgecolor": LINE, "axes.labelcolor": INK,
    "xtick.color": MUT, "ytick.color": MUT, "text.color": INK,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
})

man = pd.read_csv("exports/layout_manifest.csv").set_index("layout_id")
on_disk = set()
for s in ("train", "val", "test"):
    on_disk |= {os.path.basename(p) for p in glob.glob(f"splits_cfd_gap/{s}/*")}
# The gap-0.30 split covers 192 of 193: C015 bridges two components and is
# excluded from that SPLIT, but it is a solved, exported case and belongs in a
# figure about the DATASET. Add it back explicitly.
on_disk |= {i for i in man.index[man.gap_dropped == 1] if man.loc[i, "exported"] == 1}
df = man.loc[sorted(on_disk & set(man.index))].copy()
assert len(df) == 193, f"expected 193 dataset cases, matched {len(df)}"

TIERS = [("A",  "reference grid"),
         ("A'", "grid x translation"),
         ("B",  "Sobol, original box"),
         ("C",  "per-vent jitter"),
         ("D",  "maximin, full region")]

fig = plt.figure(figsize=(11.0, 5.6))
gs = fig.add_gridspec(2, 5, height_ratios=[1.0, 0.92], hspace=0.42, wspace=0.18,
                      left=0.045, right=0.985, top=0.90, bottom=0.085)

# ---------- row 1: ceiling plan per tier ----------
for i, (t, sub) in enumerate(TIERS):
    ax = fig.add_subplot(gs[0, i])
    sel = df[df.tier == t]
    ax.add_patch(Rectangle((0, 0), ROOM_X, ROOM_Y, fc="none", ec=INK, lw=0.9))
    for _, r in sel.iterrows():
        for k in range(3):
            ax.add_patch(Rectangle((r[f"supply{k}_x"] - P_SUP / 2, r[f"supply{k}_y"] - P_SUP / 2),
                                   P_SUP, P_SUP, fc=SUP, ec="none", alpha=0.16))
            ax.add_patch(Rectangle((r[f"return{k}_x"] - P_RET / 2, r[f"return{k}_y"] - P_RET / 2),
                                   P_RET, P_RET, fc=RET, ec="none", alpha=0.16))
    ax.set_xlim(-0.45, ROOM_X + 0.45); ax.set_ylim(-0.45, ROOM_Y + 0.45)
    ax.set_aspect("equal"); ax.set_xticks([0, 4, 8]); ax.set_yticks([0, 3, 6])
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.set_title(f"Tier {t}   n = {len(sel)}", fontsize=8.5, fontweight="bold",
                 color=INK, pad=3)
    ax.text(0.5, -0.185, sub, transform=ax.transAxes, ha="center",
            fontsize=7, color=MUT)
    if i == 0:
        ax.set_ylabel("y  (m)", fontsize=7.5)

# ---------- row 2: the parameter space tier D exists to fill ----------
ax = fig.add_subplot(gs[1, :2])
sprd = np.linspace(0.30, 3.90, 300)
ax.fill_between(sprd, 0.34 - sprd, sprd - 0.42, color=MUT, alpha=0.10, lw=0,
                label="geometrically feasible")
ax.add_patch(Rectangle((1.00, -1.00), 2.00, 2.00, fc="none", ec=SUP, lw=1.0,
                       ls=(0, (4, 2)), label="tiers A–C sampling box"))
base = df[df.tier != "D"]; dee = df[df.tier == "D"]
ax.scatter(base.spread_m, base.dx_m, s=9, c=SUP, alpha=0.65, lw=0, label="tiers A–C")
ax.scatter(dee.spread_m, dee.dx_m, s=13, c=RET, alpha=0.8, lw=0, label="tier D")
ax.set_xlabel("array spread  (m)"); ax.set_ylabel("translation  dx  (m)")
ax.set_xlim(0.1, 4.0); ax.set_ylim(-3.6, 3.6)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
ax.legend(frameon=False, fontsize=6.8, loc="upper left", handletextpad=0.5)
ax.set_title("Why tier D exists — the unused degree of freedom is dx",
             fontsize=8.5, fontweight="bold", loc="left", pad=4)

# ---------- row 2b: separation to nearest other layout ----------
ax2 = fig.add_subplot(gs[1, 2:4])
for t, c, lab in (("D", RET, "tier D"), (None, SUP, "tiers A–C")):
    v = (df[df.tier == "D"] if t else df[df.tier != "D"])["nearest_distance_m"].dropna()
    ax2.hist(v, bins=np.arange(0, 1.45, 0.06), color=c, alpha=0.62, lw=0, label=lab)
ax2.axvline(0.30, color=INK, lw=0.9, ls=(0, (3, 2)))
ax2.axvline(0.58, color=INK, lw=0.9, ls=(0, (3, 2)))
ax2.text(0.305, ax2.get_ylim()[1] * 0.93, "gap-0.30", fontsize=6.6, color=INK)
ax2.text(0.585, ax2.get_ylim()[1] * 0.93, "gap-0.58", fontsize=6.6, color=INK)
ax2.set_xlabel("distance to nearest other layout  (m)"); ax2.set_ylabel("layouts")
for sp in ("top", "right"): ax2.spines[sp].set_visible(False)
ax2.legend(frameon=False, fontsize=6.8, loc="upper right", handletextpad=0.5)
ax2.set_title("Separation is what the split thresholds cut on",
              fontsize=8.5, fontweight="bold", loc="left", pad=4)

# ---------- row 2c: legend / key ----------
axk = fig.add_subplot(gs[1, 4]); axk.axis("off")
axk.add_patch(Rectangle((0.06, 0.78), 0.13, 0.10, fc=SUP, alpha=0.5,
                        transform=axk.transAxes, clip_on=False))
axk.text(0.24, 0.80, "supply diffuser\n0.591 m square", fontsize=7, color=INK,
         transform=axk.transAxes, va="bottom")
axk.add_patch(Rectangle((0.06, 0.56), 0.13, 0.10, fc=RET, alpha=0.5,
                        transform=axk.transAxes, clip_on=False))
axk.text(0.24, 0.58, "return grille\n0.600 m square", fontsize=7, color=INK,
         transform=axk.transAxes, va="bottom")
axk.text(0.06, 0.44, "Room 8.80 x 6.10 m,\nceiling height 3.20 m.\n\n"
                     "Panels overlaid across every\nlayout in the tier; darker\n"
                     "regions are positions many\nlayouts share.",
         fontsize=7, color=MUT, transform=axk.transAxes, va="top")

fig.suptitle("Fig. 2   The five design tiers \u2014 193 layouts of six ceiling panels in one fixed room",
             fontsize=10.5, fontweight="bold", color=INK, x=0.045, ha="left", y=0.975)

for ext in ("png", "pdf"):
    fig.savefig(f"figures/fig2_tiers.{ext}", dpi=300, bbox_inches="tight",
                facecolor="white")
print("wrote figures/fig2_tiers.png and .pdf")
print(df.tier.value_counts().reindex([t for t, _ in TIERS]).to_string())
