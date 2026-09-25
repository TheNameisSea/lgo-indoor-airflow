"""Fig. 1 -- the controlled chain, drawn with REAL data at every stage.

Four step bands, mirroring the reference figure's structure (coloured banner
over grouped content), but every illustration is measured rather than sketched:
the ceiling plan is a real layout's panel coordinates, the field slices are a
real solved case, the neighbour gather is a real stratified kNN, and the split
panel shows the real nearest-neighbour distribution.
"""
import os, glob
import numpy as np, pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow, FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial import cKDTree

ROOM_X, ROOM_Y, ROOM_Z = 8.80, 6.10, 3.20
P_SUP, P_RET = 0.591, 0.600
S1, S2, S3, S4 = "#0f4c5c", "#31606b", "#9a6324", "#a63a1e"
SUP, RET, SOL, LEK = "#0f4c5c", "#a63a1e", "#7d8d96", "#4f8a5b"
INK, INK2, MUT, LINE, SUB = "#16222b", "#3c4d58", "#657680", "#d3dbdf", "#eef1f3"

mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7,
    "axes.edgecolor": LINE, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": MUT, "ytick.color": MUT,
    "axes.linewidth": .6, "xtick.major.width": .6, "ytick.major.width": .6,
    "xtick.labelsize": 6.2, "ytick.labelsize": 6.2,
})
CASE = "splits_cfd_gap/val/B005"

# ---------------------------------------------------------------- data
fl = pd.read_csv(f"{CASE}/Fluid_data.csv")
hv = pd.read_csv(f"{CASE}/New_HVAC.csv")
from scipy.cluster.hierarchy import fcluster, linkage
P = hv[["X (m)", "Y (m)", "Z (m)"]].to_numpy(); w = hv["Velocity[k] (m/s)"].to_numpy()
lab = fcluster(linkage(P, "single"), 0.20, criterion="distance")
panels = []
for i in sorted(set(lab)):
    m = lab == i
    panels.append(("supply" if w[m].mean() < 0 else "return", P[m, 0].mean(), P[m, 1].mean()))

def banner(ax, n, title, tag, colour):
    ax.set_axis_off()
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, transform=ax.transAxes,
                 boxstyle="square,pad=0", fc=colour, ec="none", clip_on=False))
    ax.text(.012, .5, f"STEP {n}", transform=ax.transAxes, va="center",
            fontsize=6.6, fontweight="bold", color="w", family="monospace")
    ax.text(.072, .5, title, transform=ax.transAxes, va="center",
            fontsize=8.6, fontweight="bold", color="w")
    ax.text(.995, .5, tag, transform=ax.transAxes, va="center", ha="right",
            fontsize=6.4, color="w", alpha=.85, family="monospace")

def textpanel(ax, head, lines, colour):
    ax.set_axis_off()
    ax.text(0, 1.0, head, transform=ax.transAxes, va="top", fontsize=6.6,
            fontweight="bold", color=colour, family="monospace")
    y = .855
    for k, v in lines:
        ax.text(0, y, k, transform=ax.transAxes, va="top", fontsize=6.6, color=INK2)
        ax.text(1, y, v, transform=ax.transAxes, va="top", ha="right",
                fontsize=6.6, color=INK, family="monospace")
        y -= .112

fig = plt.figure(figsize=(11.5, 11.0))
gs = fig.add_gridspec(9, 3, height_ratios=[.26, .12, 1, .12, 1, .12, 1, .12, 1],
                      hspace=.50, wspace=.26, left=.055, right=.975,
                      top=.972, bottom=.04)

fig.text(.055, .992, "Fig. 1   From diffuser layout to predicted field: the controlled chain",
         fontsize=11.5, fontweight="bold", color=INK, ha="left", va="top")
fig.text(.055, .9765, "One room, one variable. Each step exists to make a single claim testable — "
         "that the model learned layout→field transfer rather than memorising a nearby field it can copy.",
         fontsize=7.4, color=MUT, ha="left", va="top")

# ======================================================= STEP 1
banner(fig.add_subplot(gs[1, :]), 1, "The cause — layout is the only variable",
       "193 designed layouts", S1)

ax = fig.add_subplot(gs[2, 0])
# The three parameters have exact definitions and the earlier draft drew two of
# them wrong. Verified against exports/layout_manifest.csv for this case:
#   spread = x of the OUTER supply BEFORE translation      (1.539 m)
#   dx     = offset of the pinned middle supply from 4.46 m (0.696 m)
#   row    = y of the supply row                            (1.318 m)
# so the untranslated array is drawn as a ghost and dx is the shift between the
# ghost and the real layout. Anything else mislabels the family.
_m = pd.read_csv("exports/layout_manifest.csv").set_index("layout_id").loc["B005"]
SPREAD, ROW, DX, PIN = float(_m.spread_m), float(_m.row_m), float(_m.dx_m), 4.46
ghost_sup = [SPREAD, PIN, 8.92 - SPREAD]

ax.add_patch(Rectangle((0, 0), ROOM_X, ROOM_Y, fc="none", ec=INK, lw=1.0))
for gx_ in ghost_sup:                                   # untranslated array
    ax.add_patch(Rectangle((gx_ - P_SUP / 2, ROW - P_SUP / 2), P_SUP, P_SUP,
                 fc="none", ec=SUP, lw=.7, ls=(0, (2, 1.6)), alpha=.75))
for kind, x, y in panels:                               # the real layout
    hw = (P_SUP if kind == "supply" else P_RET) / 2
    ax.add_patch(Rectangle((x - hw, y - hw), 2 * hw, 2 * hw,
                 fc=SUP if kind == "supply" else RET, ec="none", alpha=.85))

# spread -- from the wall to the outer supply of the UNTRANSLATED array
ax.annotate("", xy=(0, -0.62), xytext=(SPREAD, -0.62),
            arrowprops=dict(arrowstyle="<->", color=INK, lw=.8))
ax.plot([0, 0], [-0.9, ROW], color=INK, lw=.5, ls=(0, (1, 2)), alpha=.6)
ax.plot([SPREAD, SPREAD], [-0.9, ROW], color=SUP, lw=.5, ls=(0, (1, 2)), alpha=.8)
ax.text(SPREAD / 2, -1.26, f"spread = {SPREAD:.2f} m", ha="center", fontsize=6.2,
        color=INK, family="monospace")

# `spread` is the OUTER vent's position from the wall, NOT the gap between
# vents -- verified across 173 layouts to 0.0000 m. The two differ by up to
# 3.63 m, and the name runs opposite to the geometry: vent spacing is
# 4.46 - spread, so a LARGER spread packs the array TIGHTER. Both are drawn,
# because a reader who assumes "spread = gap between vents" is wrong by metres.
SPACING = PIN - SPREAD
ax.annotate("", xy=(ghost_sup[0], ROW + 1.05), xytext=(ghost_sup[1], ROW + 1.05),
            arrowprops=dict(arrowstyle="<->", color=SUP, lw=.8))
for gx_ in ghost_sup[:2]:
    ax.plot([gx_, gx_], [ROW + .35, ROW + 1.05], color=SUP, lw=.5,
            ls=(0, (1, 2)), alpha=.8)
ax.text((ghost_sup[0] + ghost_sup[1]) / 2, ROW + 1.34,
        f"spacing = {SPACING:.2f} m", ha="center",
        fontsize=6.2, color=SUP, family="monospace")

# dx -- pinned middle panel, 4.46 m, shifted to its actual position
ax.annotate("", xy=(PIN, ROOM_Y + .62), xytext=(PIN + DX, ROOM_Y + .62),
            arrowprops=dict(arrowstyle="<->", color=INK, lw=.8))
ax.plot([PIN, PIN], [ROW, ROOM_Y + .9], color=SUP, lw=.5, ls=(0, (1, 2)), alpha=.8)
ax.plot([PIN + DX, PIN + DX], [ROW, ROOM_Y + .9], color=SUP, lw=.5, ls=(0, (1, 2)), alpha=.8)
ax.text(PIN + DX + .30, ROOM_Y + .62, f"dx = {DX:+.2f} m", ha="left", fontsize=6.2,
        color=INK, family="monospace", va="center")

# row -- floor to the supply row
ax.annotate("", xy=(ROOM_X + .70, 0), xytext=(ROOM_X + .70, ROW),
            arrowprops=dict(arrowstyle="<->", color=INK, lw=.8))
ax.text(ROOM_X + .95, ROW / 2, f"row = {ROW:.2f} m", fontsize=6.2, color=INK,
        family="monospace", rotation=90, va="center", ha="left")

ax.text(0, -2.10, "dashed outline = the same layout at dx = 0", fontsize=5.9, color=MUT)
ax.text(0, -2.52, "vent spacing = 4.46 − spread, so larger spread packs the array tighter",
        fontsize=5.9, color=MUT)
ax.set_xlim(-.6, ROOM_X + 2.5); ax.set_ylim(-3.05, ROOM_Y + 1.30)
ax.set_aspect("equal"); ax.set_xticks([0, 4, 8]); ax.set_yticks([0, 3, 6])
for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
ax.set_title("(a)  ceiling plan — the 3-parameter family", fontsize=7.4,
             fontweight="bold", loc="left", color=INK, pad=3)
ax.set_xlabel("x  (m)", fontsize=6.6); ax.set_ylabel("y  (m)", fontsize=6.6)

ax = fig.add_subplot(gs[2, 1])
ax.add_patch(Rectangle((0, 0), ROOM_X, ROOM_Z, fc="none", ec=INK, lw=1.0))
ax.add_patch(Rectangle((2.4, 0), 1.5, .75, fc=SOL, alpha=.28, ec="none"))
ax.add_patch(Rectangle((5.6, 0), 1.2, .75, fc=SOL, alpha=.28, ec="none"))
ax.text(3.15, 1.02, "furniture", ha="center", fontsize=5.8, color=MUT)
for kind, x, _ in panels:
    if kind == "supply":
        for dxo in (-.18, 0, .18):
            ax.arrow(x + dxo, ROOM_Z - .08, 0, -1.0, head_width=.13,
                     head_length=.16, fc=SUP, ec=SUP, lw=.7, alpha=.75)
    else:
        ax.arrow(x, ROOM_Z - 1.05, 0, .82, head_width=.15, head_length=.17,
                 fc=RET, ec=RET, lw=.8, alpha=.8)
ax.text(.06, 4.30, "supply  0.571 m/s ↓  287.7 K        return  pressure outlet",
        fontsize=6.0, color=SUP, va="center")
ax.text(.06, 3.83, "walls to 308.06 K        door-undercut leak 0.0009 m³/s",
        fontsize=6.0, color=MUT, va="center")
ax.set_xlim(-.3, ROOM_X + .3); ax.set_ylim(-.35, 4.70)
ax.set_aspect("equal"); ax.set_xticks([0, 4, 8]); ax.set_yticks([0, 3])
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.set_title("(b)  section — what drives the flow", fontsize=7.4,
             fontweight="bold", loc="left", color=INK, pad=3)
ax.set_xlabel("x  (m)", fontsize=6.6); ax.set_ylabel("z  (m)", fontsize=6.6)

textpanel(fig.add_subplot(gs[2, 2]), "HELD FIXED — BYTE-IDENTICAL", [
    ("Room envelope", "8.80 × 6.10 × 3.20 m"),
    ("Furniture + 18 wall patches", "copied verbatim"),
    ("Supply flow rate", "12.6 ACH"),
    ("Solver / mesh / relaxation", "one protocol"),
    ("Iterations, ramped from rest", "3000"),
    ("", ""),
    ("VARIED", "6 ceiling panels"),
    ("supply 3 × 0.591 m sq", "spread, row, dx"),
    ("return 3 × 0.600 m sq", "mirrored, −0.04 m"),
], S1)

# ======================================================= STEP 2
banner(fig.add_subplot(gs[3, :]), 2, "The data — CFD behind a quality gate",
       "OpenFOAM v2412 · buoyantSimpleFoam · k–ε", S2)

Y_SLICE = float(np.mean([y for k, _, y in panels if k == "supply"]))
sl = fl[(fl["Y (m)"] - Y_SLICE).abs() < .12]
from scipy.interpolate import griddata
gx, gz = np.meshgrid(np.linspace(0, ROOM_X, 420), np.linspace(0, ROOM_Z, 160))
pts2 = sl[["X (m)", "Z (m)"]].to_numpy()
def field(col):
    return griddata(pts2, sl[col].to_numpy(), (gx, gz), method="linear")
cmap_v = LinearSegmentedColormap.from_list("v", ["#f4f6f7", "#8fb8c4", S1, "#08303b"])
ax = fig.add_subplot(gs[4, 0])
h = ax.imshow(field("Velocity: Magnitude (m/s)"), origin="lower", cmap=cmap_v,
              extent=[0, ROOM_X, 0, ROOM_Z], vmin=0, vmax=1.1, aspect="equal",
              interpolation="bilinear")
ax.set_xlim(0, ROOM_X); ax.set_ylim(0, ROOM_Z); ax.set_aspect("equal")
ax.set_xticks([0, 4, 8]); ax.set_yticks([0, 3])
cb = fig.colorbar(h, ax=ax, fraction=.028, pad=.015); cb.set_label("|u|  (m/s)", fontsize=6)
cb.ax.tick_params(labelsize=5.6)
ax.set_title("(c)  solved velocity, slice at the supply row", fontsize=7.4,
             fontweight="bold", loc="left", color=INK, pad=3)
ax.set_xlabel(f"x  (m)        y = {Y_SLICE:.2f} m", fontsize=6.6)
ax.set_xlabel("x  (m)", fontsize=6.6); ax.set_ylabel("z  (m)", fontsize=6.6)

cmap_t = LinearSegmentedColormap.from_list("t", ["#1b6f8b", "#f4f6f7", "#b5431f"])
ax = fig.add_subplot(gs[4, 1])
h = ax.imshow(field("Temperature (K)"), origin="lower", cmap=cmap_t,
              extent=[0, ROOM_X, 0, ROOM_Z], vmin=290.5, vmax=295.5,
              aspect="equal", interpolation="bilinear")
ax.set_xlim(0, ROOM_X); ax.set_ylim(0, ROOM_Z); ax.set_aspect("equal")
ax.set_xticks([0, 4, 8]); ax.set_yticks([0, 3])
cb = fig.colorbar(h, ax=ax, fraction=.028, pad=.015); cb.set_label("T  (K)", fontsize=6)
cb.ax.tick_params(labelsize=5.6)
ax.set_title("(d)  temperature, same slice", fontsize=7.4, fontweight="bold",
             loc="left", color=INK, pad=3)
ax.set_xlabel("x  (m)", fontsize=6.6); ax.set_ylabel("z  (m)", fontsize=6.6)

textpanel(fig.add_subplot(gs[4, 2]), "QUALITY GATE — APPLIED AT EXPORT", [
    ("Net boundary flux", "< 10⁻³ supply"),
    ("Per-opening flux", "± 5 %"),
    ("Mean room speed", "0.05 – 0.60 m/s"),
    ("ε clip severity", "< 10⁻² of mean"),
    ("Cases rejected by the gate", "2 of 195"),
    ("", ""),
    ("BLINDED — solver OUTPUT", "not an input"),
    ("return-vent u,v,w", "46 % of cloud"),
    ("pressure everywhere; T off-supply", "zeroed"),
], S2)

# ======================================================= STEP 3
banner(fig.add_subplot(gs[5, :]), 3, "The split — separating transfer from layout lookup",
       "threshold = vent panel width", S3)

ax = fig.add_subplot(gs[6, 0])
for i, (d, lbdl) in enumerate(((0.30, "gap-0.30"), (0.58, "gap-0.58"))):
    y0 = 1.35 - i * 1.25
    ax.add_patch(Rectangle((0, y0), P_SUP, .52, fc=SUP, alpha=.32, ec=SUP, lw=.8))
    ax.add_patch(Rectangle((d, y0), P_SUP, .52, fc=RET, alpha=.32, ec=RET, lw=.8))
    ov = max(0, P_SUP - d) / P_SUP * 100
    ax.add_patch(Rectangle((d, y0), max(0, P_SUP - d), .52, fc=INK, alpha=.30, ec="none"))
    ax.text(1.32, y0 + .30, f"{lbdl}", fontsize=6.8, color=INK, family="monospace",
            fontweight="bold")
    ax.text(1.32, y0 + .08, f"{ov:.0f} % of the diffuser face still overlaps",
            fontsize=6.2, color=MUT)
ax.text(0, 2.16, "one panel, and the same panel displaced by the threshold",
        fontsize=6.4, color=MUT)
ax.set_xlim(-.1, 2.9); ax.set_ylim(-.15, 2.4); ax.set_axis_off()
ax.set_title("(e)  what the threshold means geometrically", fontsize=7.4,
             fontweight="bold", loc="left", color=INK, pad=3)

man = pd.read_csv("exports/layout_manifest.csv").set_index("layout_id")
od = set()
for s in ("train", "val", "test"):
    od |= {os.path.basename(p) for p in glob.glob(f"splits_cfd_gap/{s}/*")}
od |= {i for i in man.index[man.gap_dropped == 1] if man.loc[i, "exported"] == 1}
dm = man.loc[sorted(od & set(man.index))]
ax = fig.add_subplot(gs[6, 1])
ax.hist(dm[dm.tier != "D"].nearest_distance_m.dropna(), bins=np.arange(0, 1.3, .06),
        color=SUP, alpha=.62, lw=0, label="tiers A–C")
ax.hist(dm[dm.tier == "D"].nearest_distance_m.dropna(), bins=np.arange(0, 1.3, .06),
        color=RET, alpha=.62, lw=0, label="tier D")
for t in (.30, .58):
    ax.axvline(t, color=INK, lw=.9, ls=(0, (3, 2)))
ax.text(.305, 35, "0.30", fontsize=6.2, color=INK); ax.text(.585, 35, "0.58", fontsize=6.2, color=INK)
ax.set_xlabel("distance to nearest other layout  (m)", fontsize=6.6)
ax.set_ylabel("layouts", fontsize=6.6)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=6.2, loc="upper right")
ax.set_title("(f)  the separation the thresholds cut on", fontsize=7.4,
             fontweight="bold", loc="left", color=INK, pad=3)

ax = fig.add_subplot(gs[6, 2])
xs = np.arange(2); wdt = .36
ash = [.385, -.004]; ful = [.036, -.120]
ax.bar(xs - wdt / 2, ash, wdt, color=S3, label="ASHRAE band R²")
ax.bar(xs + wdt / 2, ful, wdt, color=MUT, label="full-field v_R²")
ax.axhline(0, color=INK, lw=.8)
for i, (a, f) in enumerate(zip(ash, ful)):
    ax.text(i - wdt / 2, a + (.02 if a > 0 else -.05), f"{a:+.3f}", ha="center",
            fontsize=6.2, family="monospace", color=INK)
    ax.text(i + wdt / 2, f + (.02 if f > 0 else -.05), f"{f:+.3f}", ha="center",
            fontsize=6.2, family="monospace", color=INK)
ax.set_xticks(xs); ax.set_xticklabels(["gap-0.30", "gap-0.58"], fontsize=6.6,
                                       family="monospace")
ax.set_ylim(-.22, .50); ax.set_ylabel("baseline score", fontsize=6.6)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=6.2, loc="upper right")
ax.set_title("(g)  nearest-layout retrieval collapses", fontsize=7.4,
             fontweight="bold", loc="left", color=INK, pad=3)

# ======================================================= STEP 4
banner(fig.add_subplot(gs[7, :]), 4, "The model — one local branch, three interchangeable hosts",
       "queried at arbitrary interior points", S4)

sup_pts = np.array([p for p, kk in zip(P, lab) if panels[kk - 1][0] == "supply"])
ret_pts = np.array([p for p, kk in zip(P, lab) if panels[kk - 1][0] == "return"])
sol_pts = np.vstack([pd.read_csv(f"{CASE}/{n}.csv")[["X (m)", "Y (m)", "Z (m)"]].to_numpy()
                     for n in ("Floor", "Ceiling", "Wall_01", "Wall_03", "Table")])
lek_pts = pd.read_csv(f"{CASE}/Leak.csv")[["X (m)", "Y (m)", "Z (m)"]].to_numpy()
q = np.array([4.40, 3.05, 1.10])
gath = []
for pts, K, c, nm in ((sup_pts, 32, SUP, "supply 32"), (ret_pts, 16, RET, "return 16"),
                      (sol_pts, 32, SOL, "solid 32"), (lek_pts, 8, LEK, "leak 8")):
    d, i = cKDTree(pts).query(q, k=K)
    gath.append((pts[np.atleast_1d(i)], c, nm, np.atleast_1d(d).max()))
ax = fig.add_subplot(gs[8, 0])
ax.add_patch(Rectangle((0, 0), ROOM_X, ROOM_Z, fc="none", ec=LINE, lw=.8))
for pts, c, nm, dmax in gath:
    ax.scatter(pts[:, 0], pts[:, 2], s=13, c=c, lw=0, alpha=.95, zorder=4,
               label=f"{nm}   ≤ {dmax:.2f} m")
    for p in pts[::4]:
        ax.plot([q[0], p[0]], [q[2], p[2]], color=c, lw=.35, alpha=.45, zorder=2)
ax.scatter([q[0]], [q[2]], s=55, marker="*", c=INK, zorder=6,
           edgecolors="w", linewidths=.5, label="query  x")
ax.set_xlim(-.3, ROOM_X + .3); ax.set_ylim(-.9, ROOM_Z + .3); ax.set_aspect("equal")
ax.set_xticks([0, 4, 8]); ax.set_yticks([0, 3])
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=5.6, loc="lower center", ncol=3,
          handletextpad=.25, columnspacing=.7, bbox_to_anchor=(.5, -.52))
ax.set_title("(h)  the real stratified gather — 88 neighbours", fontsize=7.4,
             fontweight="bold", loc="left", color=INK, pad=3)
ax.set_xlabel("x  (m)", fontsize=6.6); ax.set_ylabel("z  (m)", fontsize=6.6)

ax = fig.add_subplot(gs[8, 1]); ax.set_axis_off()
def blk(x, y, w_, h_, txt, fc, tc="w", fs=6.4):
    ax.add_patch(FancyBboxPatch((x, y), w_, h_, boxstyle="round,pad=.008,rounding_size=.02",
                 fc=fc, ec="none", transform=ax.transAxes, clip_on=False))
    ax.text(x + w_ / 2, y + h_ / 2, txt, transform=ax.transAxes, ha="center",
            va="center", fontsize=fs, color=tc, fontweight="bold")
blk(.02, .80, .44, .13, "boundary cloud\n15,000 × 13 ch", SUB, INK)
blk(.54, .80, .44, .13, "query  x", SUB, INK)
blk(.02, .58, .44, .13, "host encoder", S2)
blk(.54, .58, .44, .13, "stratified gather\n88 relative offsets", S4)
blk(.02, .36, .44, .13, "$h_{host}(x)$", S2)
blk(.54, .36, .44, .13, "$h_{local}(x)$", S4)
blk(.24, .17, .52, .11, "+   shared output head", INK)
blk(.24, .01, .52, .11, "u, v, w, p, T", SUB, INK)
for x0, y0, y1 in ((.24, .80, .71), (.76, .80, .71), (.24, .58, .49),
                   (.76, .58, .49), (.36, .36, .28), (.64, .36, .28), (.50, .17, .12)):
    ax.annotate("", xy=(x0, y1), xytext=(x0, y0), transform=ax.transAxes,
                arrowprops=dict(arrowstyle="-|>", color=MUT, lw=.8))
ax.text(.5, -.075, "zero-initialised output projection ⇒  $h_{local}\\equiv0$  at init,\n"
        "so the augmented model is BIT-IDENTICAL to its host",
        transform=ax.transAxes, ha="center", va="top", fontsize=6.3, color=INK)
ax.set_title("(i)  where the branch enters", fontsize=7.4, fontweight="bold",
             loc="left", color=INK, pad=3)

ax = fig.add_subplot(gs[8, 2])
d_sup = cKDTree(sup_pts).query(sl[["X (m)", "Y (m)", "Z (m)"]].to_numpy(), k=1)[0]
zc = np.select([d_sup < 1, d_sup < 2], [0, 1], 2)
ax.scatter(sl["X (m)"], sl["Z (m)"], c=zc, s=2.2, lw=0, rasterized=True, alpha=.9,
           vmin=0, vmax=2,
           cmap=LinearSegmentedColormap.from_list("z", [S4, S3, "#c2ced4"]))
# A dashed dark outline is unreadable over the zone scatter. Lighten the band
# itself, then stroke it twice -- a thick white halo under a dark line -- so the
# boundary reads whichever zone colour happens to sit behind it.
ax.add_patch(Rectangle((1.0, 0.10), ROOM_X - 2.0, 1.25, fc="w", alpha=.52,
                       ec="none", zorder=5))
ax.add_patch(Rectangle((1.0, 0.10), ROOM_X - 2.0, 1.25, fc="none",
                       ec="w", lw=3.4, zorder=6))
ax.add_patch(Rectangle((1.0, 0.10), ROOM_X - 2.0, 1.25, fc="none",
                       ec=INK, lw=1.5, ls=(0, (4, 2)), zorder=7))
ax.text(4.4, 1.62, "ASHRAE 55 seated band   0.10–1.35 m", ha="center",
        fontsize=6.2, color=INK, zorder=8, fontweight="bold",
        bbox=dict(boxstyle="round,pad=.28", fc="w", ec=INK, lw=.8, alpha=.95))
ax.set_xlim(0, ROOM_X); ax.set_ylim(0, ROOM_Z); ax.set_aspect("equal")
ax.set_xticks([0, 4, 8]); ax.set_yticks([0, 3])
for s in ("top", "right"): ax.spines[s].set_visible(False)
hs = [plt.Line2D([], [], marker="s", ls="", ms=4, color=c, label=l) for c, l in
      ((S4, "near_jet  < 1 m"), (S3, "transition  1–2 m"), ("#c9d3d8", "far_field  > 2 m"))]
ax.legend(handles=hs, frameon=False, fontsize=5.6, loc="lower center", ncol=3,
          handletextpad=.25, columnspacing=.7, bbox_to_anchor=(.5, -.52))
ax.set_title("(j)  the two orthogonal scoring regions", fontsize=7.4,
             fontweight="bold", loc="left", color=INK, pad=3)
ax.set_xlabel("x  (m)", fontsize=6.6); ax.set_ylabel("z  (m)", fontsize=6.6)

for ext in ("png", "pdf"):
    fig.savefig(f"figures/fig1_chain.{ext}", dpi=260, bbox_inches="tight", facecolor="white")
print("wrote figures/fig1_chain.png / .pdf")
