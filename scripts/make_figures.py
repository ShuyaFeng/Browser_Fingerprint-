"""
Generate paper figures from experiment results.

Reads results/*.csv and produces results/figures/*.pdf (vector, S&P-ready).

Figures:
  fig1_shapley_vs_marginal.pdf  — Shapley attribution vs marginal entropy
  fig2_interaction_heatmap.pdf  — pairwise Shapley interaction heatmap
  fig3_desktop_vs_mobile.pdf    — attribution split by device class
  fig4_category_breakdown.pdf   — entropy attribution grouped by feature category
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Patch
matplotlib.rcParams["pdf.fonttype"] = 42   # embed TrueType (S&P requirement)
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.size"] = 10

from scripts.figstyle import labels as feat_labels

RESULTS = "results"
FIGDIR = "results/figures"
os.makedirs(FIGDIR, exist_ok=True)

CAT_COLORS = {
    "hardware/GPU": "#d62728", "hardware/audio": "#ff7f0e",
    "hardware/screen": "#ffbb78", "hardware/CPU": "#e377c2",
    "OS/fonts": "#2ca02c", "OS": "#98df8a", "browser": "#1f77b4",
    "locale": "#9467bd", "?": "#7f7f7f",
}


def fig1_shapley_vs_marginal():
    df = pd.read_csv(f"{RESULTS}/shapley_attribution.csv")
    df = df.sort_values("shapley_bits", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    y = np.arange(len(df))
    # Marginal entropy as light background bars
    ax.barh(y, df["marginal_bits"], color="lightgray", label="Marginal entropy H(X_i)")
    # Shapley value as colored foreground bars
    colors = [CAT_COLORS.get(c, "#7f7f7f") for c in df["category"]]
    ax.barh(y, df["shapley_bits"], color=colors, label="Shapley value φ_i")

    # Error bars for CI if present
    if "ci_low" in df.columns:
        ax.errorbar(df["shapley_bits"], y,
                    xerr=[df["shapley_bits"] - df["ci_low"],
                          df["ci_high"] - df["shapley_bits"]],
                    fmt="none", ecolor="black", elinewidth=0.8, capsize=2)

    ax.set_yticks(y)
    ax.set_yticklabels(feat_labels(df["feature"]))
    ax.set_xlabel("Entropy contribution (bits)")
    # legend: gray = marginal, colored bars keyed by feature category
    cats = [c for c in CAT_COLORS if c in set(df["category"])]
    handles = [Patch(facecolor="lightgray", label="Marginal entropy $H(X_i)$")] + \
              [Patch(facecolor=CAT_COLORS[c], label=f"Shapley $\\phi_i$: {c}") for c in cats]
    ax.legend(handles=handles, loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{FIGDIR}/fig1_shapley_vs_marginal.pdf")
    plt.close()
    print("  saved fig1_shapley_vs_marginal.pdf")


def fig2_interaction_heatmap():
    df = pd.read_csv(f"{RESULTS}/pairwise_interactions.csv")
    feats = sorted(set(df["feature_i"]) | set(df["feature_j"]))
    idx = {f: i for i, f in enumerate(feats)}
    n = len(feats)
    M = np.full((n, n), np.nan)
    for _, r in df.iterrows():
        i, j = idx[r["feature_i"]], idx[r["feature_j"]]
        M[i, j] = r["interaction_bits"]
        M[j, i] = r["interaction_bits"]

    fig, ax = plt.subplots(figsize=(7, 6))
    vmax = np.nanmax(np.abs(M))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(n)); ax.set_xticklabels(feat_labels(feats), rotation=45,
                                                ha="right", rotation_mode="anchor")
    ax.set_yticks(range(n)); ax.set_yticklabels(feat_labels(feats))
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Shapley interaction (bits)\nnegative = redundant, positive = synergistic")
    # Annotate cells
    for i in range(n):
        for j in range(n):
            if not np.isnan(M[i, j]) and abs(M[i, j]) > 0.3:
                ax.text(j, i, f"{M[i,j]:.1f}", ha="center", va="center",
                        fontsize=6, color="black")
    plt.tight_layout()
    plt.savefig(f"{FIGDIR}/fig2_interaction_heatmap.pdf")
    plt.close()
    print("  saved fig2_interaction_heatmap.pdf")


def fig3_desktop_vs_mobile():
    df = pd.read_csv(f"{RESULTS}/desktop_vs_mobile.csv")
    df = df.sort_values("shapley_desktop", ascending=True)
    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(8, 5))
    h = 0.38
    ax.barh(y + h/2, df["shapley_desktop"], height=h, color="#1f77b4", label="Desktop")
    ax.barh(y - h/2, df["shapley_mobile"], height=h, color="#ff7f0e", label="Mobile")
    ax.set_yticks(y); ax.set_yticklabels(feat_labels(df["feature"]))
    ax.set_xlabel("Shapley value (bits)")
    ax.legend(); ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{FIGDIR}/fig3_desktop_vs_mobile.pdf")
    plt.close()
    print("  saved fig3_desktop_vs_mobile.pdf")


def fig4_category_breakdown():
    df = pd.read_csv(f"{RESULTS}/shapley_attribution.csv")
    by_cat = df.groupby("category").agg(
        shapley=("shapley_bits", "sum"),
        marginal=("marginal_bits", "sum"),
    ).sort_values("shapley", ascending=True)

    y = np.arange(len(by_cat))
    fig, ax = plt.subplots(figsize=(8, 5))
    h = 0.38
    ax.barh(y + h/2, by_cat["marginal"], height=h, color="lightgray", label="Sum of marginals")
    colors = [CAT_COLORS.get(c, "#7f7f7f") for c in by_cat.index]
    ax.barh(y - h/2, by_cat["shapley"], height=h, color=colors, label="Sum of Shapley")
    ax.set_yticks(y); ax.set_yticklabels(by_cat.index)
    ax.set_xlabel("Entropy (bits)")
    ax.legend(); ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{FIGDIR}/fig4_category_breakdown.pdf")
    plt.close()
    print("  saved fig4_category_breakdown.pdf")


def fig5_defense_effectiveness():
    """Defense true effectiveness vs naive marginal prediction."""
    if not os.path.exists(f"{RESULTS}/defense_effectiveness.csv"):
        print("  skip fig5 (run defense_analysis.py first)")
        return
    df = pd.read_csv(f"{RESULTS}/defense_effectiveness.csv")
    df = df.sort_values("actual_removed_bits", ascending=True)
    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    h = 0.27
    ax.barh(y + h, df["naive_marginal_prediction"], height=h,
            color="lightgray", label="Naive marginal prediction (what defenders assume)")
    ax.barh(y, df["actual_removed_bits"], height=h,
            color="#d62728", label="Actual entropy removed (true effect)")
    ax.barh(y - h, df["residual_entropy_bits"], height=h,
            color="#1f77b4", label="Residual entropy (what attackers still get)")
    ax.set_yticks(y); ax.set_yticklabels(df["defense"])
    ax.set_xlabel("Entropy (bits)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{FIGDIR}/fig5_defense_effectiveness.pdf")
    plt.close()
    print("  saved fig5_defense_effectiveness.pdf")


def fig6_point_defenses():
    """Single-point defenses are near-useless due to redundancy."""
    if not os.path.exists(f"{RESULTS}/point_defenses.csv"):
        print("  skip fig6 (run defense_analysis.py first)")
        return
    df = pd.read_csv(f"{RESULTS}/point_defenses.csv")
    df = df.sort_values("actual_removed_bits", ascending=True)
    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(9, 4))
    h = 0.4
    ax.barh(y + h/2, df["naive_marginal_prediction"], height=h,
            color="lightgray", label="Marginal entropy (naive expectation)")
    ax.barh(y - h/2, df["actual_removed_bits"], height=h,
            color="#d62728", label="Actual entropy removed")
    for i, (_, r) in enumerate(df.iterrows()):
        ax.text(r["naive_marginal_prediction"] + 0.3, i + h/2,
                f"{r['naive_overestimate_factor']:.0f}x over", va="center", fontsize=7)
    ax.set_yticks(y); ax.set_yticklabels(df["defense"])
    ax.set_xlabel("Entropy (bits)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{FIGDIR}/fig6_point_defenses.pdf")
    plt.close()
    print("  saved fig6_point_defenses.pdf")


if __name__ == "__main__":
    print("Generating figures...")
    fig1_shapley_vs_marginal()
    fig2_interaction_heatmap()
    fig3_desktop_vs_mobile()
    fig4_category_breakdown()
    fig5_defense_effectiveness()
    fig6_point_defenses()
    print(f"\nAll figures saved to {os.path.abspath(FIGDIR)}/")
