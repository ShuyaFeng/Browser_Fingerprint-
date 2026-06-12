"""
Figure 1: framework overview + core insight.

Top row: the pipeline (correlated fingerprint features -> entropy game ->
Shapley attribution that conserves the total -> downstream analyses).
Bottom: the core finding, the marginal sum (70 bits, over-counting overlap)
versus the conserved Shapley / joint entropy (14 bits).

Rendered to results/figures/fig0_overview.pdf (vector, fonts embedded).
"""

import os
import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.size"] = 9
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

RED = "#d62728"      # hardware / GPU
GREEN = "#2ca02c"    # fonts
BLUE = "#1f77b4"     # browser / user-agent
GRAY = "#9e9e9e"     # other
DARK = "#333333"

FIGDIR = "results/figures"
os.makedirs(FIGDIR, exist_ok=True)


def box(ax, x, y, w, h, fc="white", ec=DARK, lw=1.2):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                       linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2)
    ax.add_patch(p)


def arrow(ax, x0, y0, x1, y1, lw=1.6, color=DARK):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=13,
                        linewidth=lw, color=color, zorder=3)
    ax.add_patch(a)


fig, ax = plt.subplots(figsize=(7.1, 3.8))
ax.set_xlim(0, 10)
ax.set_ylim(0, 5.4)
ax.axis("off")

TITLE_FS = 8.5

# ---------------------------------------------------------------- top pipeline
ytop, htop = 3.05, 2.05

# (1) Input
box(ax, 0.45, ytop, 2.05, htop, fc="#f7f7f7")
ax.text(1.475, ytop + htop - 0.18, "Fingerprint", ha="center", va="top",
        fontweight="bold", fontsize=TITLE_FS)
feat_rows = [
    ("WebGL hash", RED), ("GPU renderer", RED), ("Canvas", RED),
    ("fonts", GREEN), ("user-agent", BLUE), ("timezone, ...", GRAY),
]
for i, (name, c) in enumerate(feat_rows):
    yy = ytop + htop - 0.52 - i * 0.245
    ax.add_patch(plt.Rectangle((0.62, yy - 0.07), 0.15, 0.15, color=c, zorder=3))
    ax.text(0.84, yy, name, ha="left", va="center", fontsize=7.6, color=DARK)
# correlation brace on the three GPU features
ax.annotate("", xy=(0.57, ytop + htop - 0.52), xytext=(0.57, ytop + htop - 1.02),
            arrowprops=dict(arrowstyle="-", color=RED, lw=2.4))

arrow(ax, 2.52, ytop + htop / 2, 2.92, ytop + htop / 2)

# (2) Framework
box(ax, 2.95, ytop, 2.55, htop, fc="#eef4fb")
ax.text(4.225, ytop + htop - 0.18, "Entropy game", ha="center", va="top",
        fontweight="bold", fontsize=TITLE_FS)
ax.text(4.225, ytop + htop - 0.74, r"$v(S) = H(F_S)$", ha="center", va="center",
        fontsize=10)
ax.text(4.225, ytop + htop - 1.22, r"Shapley value $\phi_i$", ha="center",
        va="center", fontsize=8.8)
ax.text(4.225, ytop + 0.27, r"$\sum_i \phi_i = H(F)$", ha="center", va="center",
        fontsize=8.6,
        bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=DARK, lw=0.8))
ax.text(5.42, ytop + 0.27, "conserved", ha="right", va="center", fontsize=6.8,
        color=GRAY, style="italic")

arrow(ax, 5.52, ytop + htop / 2, 5.92, ytop + htop / 2)

# (3) Attribution mini-bars
box(ax, 5.95, ytop, 1.75, htop, fc="#f7f7f7")
ax.text(6.825, ytop + htop - 0.18, "Attribution", ha="center", va="top",
        fontweight="bold", fontsize=TITLE_FS)
mini = [("fonts", 0.92, GREEN), ("WebGL", 0.75, RED), ("GPU", 0.63, RED),
        ("agent", 0.59, BLUE), ("...", 0.22, GRAY)]
for i, (name, val, c) in enumerate(mini):
    yy = ytop + htop - 0.6 - i * 0.26
    ax.add_patch(plt.Rectangle((6.55, yy - 0.08), val, 0.16, color=c, zorder=3))
    ax.text(6.5, yy, name, ha="right", va="center", fontsize=7.2)

arrow(ax, 7.72, ytop + htop / 2, 8.12, ytop + htop / 2)

# (4) Analyses
box(ax, 8.15, ytop, 1.75, htop, fc="#fbf3ee")
ax.text(9.025, ytop + htop - 0.18, "Analyses", ha="center", va="top",
        fontweight="bold", fontsize=TITLE_FS)
apps = ["Redundancy", "Defense audit", "Migration", "Re-identification",
        "Cross-dataset"]
for i, a in enumerate(apps):
    ax.text(8.3, ytop + htop - 0.58 - i * 0.275, "• " + a, ha="left",
            va="center", fontsize=7.0)

# ---------------------------------------------------------------- bottom insight
ax.text(0.45, 2.45, "Core finding", fontweight="bold", fontsize=9.5, ha="left")

x0 = 2.55
scale = 5.85 / 70.2  # marginal bar spans x0 .. x0+5.85

def segbar(ax, y, segs, h=0.32):
    x = x0
    for w, c in segs:
        ax.add_patch(plt.Rectangle((x, y), w * scale, h, color=c, zorder=3,
                                   ec="white", lw=0.5))
        x += w * scale
    return x

# marginal bar
marg_segs = [(8.83, RED), (7.31, RED), (6.58, RED), (3.86, RED),
             (8.03, GREEN), (6.99, BLUE), (28.6, GRAY)]
y_marg = 1.72
ax.text(0.45, y_marg + 0.16, "Marginal sum", fontsize=8.2, ha="left", va="center")
endx = segbar(ax, y_marg, marg_segs)
ax.text(endx + 0.12, y_marg + 0.16, "70.2 bits", fontsize=8.5, va="center",
        fontweight="bold")

# shapley bar
shap_segs = [(1.76, RED), (1.53, RED), (1.21, RED), (1.25, RED),
             (1.89, GREEN), (1.45, BLUE), (3.06, GRAY)]
y_shap = 0.78
ax.text(0.45, y_shap + 0.16, "Shapley (= joint)", fontsize=8.2, ha="left",
        va="center")
endx2 = segbar(ax, y_shap, shap_segs)
ax.text(endx2 + 0.12, y_shap + 0.16, "14.1 bits", fontsize=8.5, va="center",
        fontweight="bold")

# redundancy span: from shapley-end to marginal-end, drawn over the marginal bar
ax.add_patch(plt.Rectangle((endx2, y_marg), endx - endx2, 0.32,
                           facecolor="none", edgecolor=DARK, lw=0.8,
                           hatch="////", zorder=4, alpha=0.45))
ax.annotate("", xy=(endx2, y_marg + 0.62), xytext=(endx, y_marg + 0.62),
            arrowprops=dict(arrowstyle="<->", color=DARK, lw=1.0))
ax.text((endx2 + endx) / 2, y_marg + 0.70,
        "56 bits (80%) redundant, mostly the GPU cluster",
        ha="center", va="bottom", fontsize=7.2, color=DARK)

plt.tight_layout(pad=0.3)
plt.savefig(f"{FIGDIR}/fig0_overview.pdf")
plt.close()
print("saved fig0_overview.pdf")
