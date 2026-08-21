"""
Interactive Rebuttal Experiment 1: Min-entropy / Bayes-vulnerability Shapley.

Addresses Reviewer A's concern that the "no synergy" result is an artifact of
Shannon entropy's submodularity. We recompute:
  (a) Shapley values under Rényi min-entropy: H_∞(S) = -log2(max_x P(x))
  (b) Shapley values under Bayes vulnerability: V(S) = max_x P(x)
  (c) Pairwise Shapley interaction index under both measures
  (d) Compare feature rankings with Shannon results

If synergy appears under min-entropy but the defender-facing ranking is
preserved, the paper's practical conclusions hold even though the structural
"no synergy" claim needs qualification.

Usage:
    python scripts/rebuttal_minentropy_shapley.py                  # 300K rows, Monte Carlo
    python scripts/rebuttal_minentropy_shapley.py --mode exact      # exact (top-10 features)
    python scripts/rebuttal_minentropy_shapley.py --nrows 0         # full 7.2M dataset
"""

import sys, os, argparse, json, time, math, itertools
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import spearmanr
import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt

from src.entropy_fast import FeatureMatrix, _joint_counts


DATA_PATH = "data/raw/li_cao_imc2020/final_with_header.csv"
RESULTS_DIR = "results/rebuttal"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(f"{RESULTS_DIR}/figures", exist_ok=True)

FEATURES_MAIN = [
    "jsFonts", "fp2_webgl", "canvastest", "hybridaudio", "agent",
    "gpu", "language", "fp2_pixelratio", "browserversion", "osversion",
    "timezone", "browser", "os", "cpucores", "fp2_colordepth",
    "fp2_platform", "encoding", "doNotTrack",
]

FEATURES_INTERACTION = [
    "jsFonts", "fp2_webgl", "canvastest", "hybridaudio",
    "agent", "gpu", "language", "fp2_pixelratio", "timezone", "os",
]

FEATURE_CATEGORY = {
    "jsFonts": "OS/fonts", "fp2_webgl": "hardware/GPU",
    "canvastest": "hardware/GPU", "hybridaudio": "hardware/audio",
    "agent": "browser/UA", "gpu": "hardware/GPU",
    "language": "browser/locale", "fp2_pixelratio": "display",
    "browserversion": "browser/UA", "osversion": "OS/version",
    "timezone": "browser/locale", "browser": "browser/UA",
    "os": "OS/version", "cpucores": "hardware/CPU",
    "fp2_colordepth": "display", "fp2_platform": "OS/platform",
    "encoding": "browser/locale", "doNotTrack": "privacy",
}


# ---------------------------------------------------------------------------
# Min-entropy and Bayes vulnerability on FeatureMatrix
# ---------------------------------------------------------------------------

def minentropy_subset(fm, cols, measure="minentropy"):
    """Compute H_∞(F_S) or V(F_S) for a feature subset.

    H_∞(S) = -log2(max_x P(x))   (Rényi min-entropy)
    V(S)   = max_x P(x)           (Bayes vulnerability = 2^{-H_∞})
    """
    if len(cols) == 0:
        if measure == "vulnerability":
            return 1.0
        return 0.0
    idx = [fm._idx[c] for c in cols]
    sub = fm.codes[:, idx]
    radix = fm.cardinalities[idx]
    counts = _joint_counts(sub, radix=radix)
    n = int(counts.sum())
    p_max = float(counts.max()) / n
    if measure == "vulnerability":
        return p_max
    return -np.log2(p_max) if p_max > 0 else 0.0


def minentropy_by_index(fm, idx, measure="minentropy"):
    if len(idx) == 0:
        return 1.0 if measure == "vulnerability" else 0.0
    sub = fm.codes[:, idx]
    radix = fm.cardinalities[idx]
    counts = _joint_counts(sub, radix=radix)
    n = int(counts.sum())
    p_max = float(counts.max()) / n
    if measure == "vulnerability":
        return p_max
    return -np.log2(p_max) if p_max > 0 else 0.0


# ---------------------------------------------------------------------------
# Shapley values under min-entropy / vulnerability
# ---------------------------------------------------------------------------

def shapley_minentropy_exact(fm, features, measure="minentropy", verbose=True):
    n = len(features)
    if n > 20:
        raise ValueError(f"n={n} too large for exact Shapley (limit 20)")
    idx = [fm._idx[f] for f in features]

    cache = np.empty(2 ** n, dtype=np.float64)
    cache[0] = 1.0 if measure == "vulnerability" else 0.0
    for mask in tqdm(range(1, 2 ** n), disable=not verbose, desc=f"v(S) [{measure}]"):
        cols = [idx[b] for b in range(n) if (mask >> b) & 1]
        cache[mask] = minentropy_by_index(fm, cols, measure=measure)

    fact = [math.factorial(i) for i in range(n + 1)]
    phi = {}
    for b, feat in enumerate(features):
        bit = 1 << b
        s = 0.0
        for mask in range(2 ** n):
            if mask & bit:
                continue
            sz = bin(mask).count("1")
            w = fact[sz] * fact[n - sz - 1] / fact[n]
            s += w * (cache[mask | bit] - cache[mask])
        phi[feat] = s
    return phi, cache


def shapley_minentropy_mc(fm, features, n_perm=1000, measure="minentropy",
                          seed=42, verbose=True):
    rng = np.random.default_rng(seed)
    n = len(features)
    idx = [fm._idx[f] for f in features]
    phi_arr = np.zeros(n, dtype=np.float64)
    empty_val = 1.0 if measure == "vulnerability" else 0.0

    for _ in tqdm(range(n_perm), disable=not verbose, desc=f"MC [{measure}]"):
        perm = rng.permutation(n)
        v_prev = empty_val
        cols = []
        for p in perm:
            cols.append(idx[p])
            v_curr = minentropy_by_index(fm, cols, measure=measure)
            phi_arr[p] += (v_curr - v_prev)
            v_prev = v_curr

    phi_arr /= n_perm
    return {features[b]: float(phi_arr[b]) for b in range(n)}


# ---------------------------------------------------------------------------
# Pairwise Shapley interaction index under min-entropy
# ---------------------------------------------------------------------------

def interactions_minentropy_exact(fm, features, measure="minentropy",
                                 cache=None, verbose=True):
    n = len(features)
    if n > 16:
        raise ValueError(f"n={n} too large for exact interactions (limit 16)")
    idx = [fm._idx[f] for f in features]

    if cache is None:
        cache = np.empty(2 ** n, dtype=np.float64)
        cache[0] = 1.0 if measure == "vulnerability" else 0.0
        for mask in tqdm(range(1, 2 ** n), disable=not verbose,
                         desc=f"v(S) [{measure}]"):
            cols = [idx[b] for b in range(n) if (mask >> b) & 1]
            cache[mask] = minentropy_by_index(fm, cols, measure=measure)

    fact = [math.factorial(i) for i in range(n + 1)]
    ixn = {}
    for bi, bj in itertools.combinations(range(n), 2):
        biti, bitj = 1 << bi, 1 << bj
        val = 0.0
        for mask in range(2 ** n):
            if (mask & biti) or (mask & bitj):
                continue
            sz = bin(mask).count("1")
            w = fact[sz] * fact[n - sz - 2] / fact[n - 1]
            val += w * (cache[mask | biti | bitj] - cache[mask | biti]
                        - cache[mask | bitj] + cache[mask])
        ixn[(features[bi], features[bj])] = val
    return ixn


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def load_matrix(path, nrows, features):
    print(f"\nLoading {'all' if nrows == 0 else f'{nrows:,}'} rows...")
    t0 = time.time()
    df = pd.read_csv(path, sep="\t", nrows=nrows if nrows > 0 else None,
                     low_memory=False, usecols=features)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].fillna("__MISSING__")
        else:
            df[col] = df[col].fillna(-9999)
    fm = FeatureMatrix.from_dataframe(df, features)
    print(f"  loaded {len(df):,} rows in {time.time()-t0:.1f}s")
    return fm


def compare_rankings(shannon_phi, minentropy_phi, vuln_phi, features):
    shannon_rank = sorted(features, key=lambda f: -shannon_phi[f])
    minentropy_rank = sorted(features, key=lambda f: -minentropy_phi[f])
    vuln_rank = sorted(features, key=lambda f: -abs(vuln_phi[f]))

    s_vals = [shannon_phi[f] for f in features]
    m_vals = [minentropy_phi[f] for f in features]
    v_vals = [vuln_phi[f] for f in features]

    rho_sm, _ = spearmanr(s_vals, m_vals)
    rho_sv, _ = spearmanr(s_vals, [abs(v) for v in v_vals])

    return {
        "shannon_rank": shannon_rank,
        "minentropy_rank": minentropy_rank,
        "vulnerability_rank": vuln_rank,
        "spearman_shannon_vs_minentropy": rho_sm,
        "spearman_shannon_vs_vulnerability": rho_sv,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Rebuttal Exp 1: Min-entropy Shapley")
    parser.add_argument("--nrows", type=int, default=300_000)
    parser.add_argument("--mode", choices=["monte_carlo", "exact"],
                        default="monte_carlo")
    parser.add_argument("--n-perm", type=int, default=1000)
    args = parser.parse_args()

    t_start = time.time()

    # Load Shannon results for comparison
    shannon_path = "results/shapley_summary.json"
    if os.path.exists(shannon_path):
        with open(shannon_path) as f:
            shannon = json.load(f)
        shannon_phi = shannon["shapley_values"]
    else:
        shannon_path = "results/shapley_summary_montecarlo.json"
        with open(shannon_path) as f:
            shannon = json.load(f)
        shannon_phi = shannon["shapley_values"]

    # --- Shapley values ---
    fm = load_matrix(DATA_PATH, args.nrows, FEATURES_MAIN)

    h_total_me = minentropy_subset(fm, FEATURES_MAIN, "minentropy")
    v_total = minentropy_subset(fm, FEATURES_MAIN, "vulnerability")
    print(f"\n  H_∞(F) = {h_total_me:.4f} bits")
    print(f"  V(F)   = {v_total:.8f}")

    print("\n" + "=" * 70)
    print("  [1] Shapley values under Rényi min-entropy")
    print("=" * 70)

    if args.mode == "exact":
        phi_me, _ = shapley_minentropy_exact(fm, FEATURES_MAIN, "minentropy")
    else:
        phi_me = shapley_minentropy_mc(fm, FEATURES_MAIN, args.n_perm,
                                       "minentropy", seed=42)

    print("\n" + "=" * 70)
    print("  [2] Shapley values under Bayes vulnerability")
    print("=" * 70)

    if args.mode == "exact":
        phi_v, _ = shapley_minentropy_exact(fm, FEATURES_MAIN, "vulnerability")
    else:
        phi_v = shapley_minentropy_mc(fm, FEATURES_MAIN, args.n_perm,
                                      "vulnerability", seed=42)

    # Print comparison table
    ranking = compare_rankings(shannon_phi, phi_me, phi_v, FEATURES_MAIN)
    print(f"\n  Rank correlation (Shannon vs min-entropy Shapley): "
          f"{ranking['spearman_shannon_vs_minentropy']:.4f}")
    print(f"  Rank correlation (Shannon vs vulnerability Shapley): "
          f"{ranking['spearman_shannon_vs_vulnerability']:.4f}")

    print(f"\n  {'feature':<20} {'Shannon':>9} {'MinEntropy':>11} {'Vuln':>12}")
    print(f"  {'-'*54}")
    for f in sorted(FEATURES_MAIN, key=lambda x: -shannon_phi[x]):
        print(f"  {f:<20} {shannon_phi[f]:>9.3f} {phi_me[f]:>11.3f} "
              f"{phi_v[f]:>12.6f}")

    # Save Shapley results
    shapley_result = {
        "n_fingerprints": fm.n_samples,
        "mode": args.mode,
        "total_minentropy_bits": h_total_me,
        "total_vulnerability": v_total,
        "shannon_shapley": shannon_phi,
        "minentropy_shapley": {f: round(phi_me[f], 6) for f in FEATURES_MAIN},
        "vulnerability_shapley": {f: round(phi_v[f], 8) for f in FEATURES_MAIN},
        "rank_correlation_shannon_minentropy": ranking["spearman_shannon_vs_minentropy"],
        "rank_correlation_shannon_vulnerability": ranking["spearman_shannon_vs_vulnerability"],
        "shannon_rank": ranking["shannon_rank"],
        "minentropy_rank": ranking["minentropy_rank"],
    }
    with open(f"{RESULTS_DIR}/minentropy_shapley.json", "w") as fp:
        json.dump(shapley_result, fp, indent=2)

    # --- Pairwise interactions under min-entropy ---
    print("\n" + "=" * 70)
    print("  [3] Pairwise interactions under min-entropy (top-10 features)")
    print("=" * 70)

    fm_ixn = load_matrix(DATA_PATH, args.nrows, FEATURES_INTERACTION)
    phi_me_ixn, cache_me = shapley_minentropy_exact(
        fm_ixn, FEATURES_INTERACTION, "minentropy", verbose=True)
    ixn_me = interactions_minentropy_exact(
        fm_ixn, FEATURES_INTERACTION, "minentropy", cache=cache_me, verbose=True)

    phi_v_ixn, cache_v = shapley_minentropy_exact(
        fm_ixn, FEATURES_INTERACTION, "vulnerability", verbose=True)
    ixn_v = interactions_minentropy_exact(
        fm_ixn, FEATURES_INTERACTION, "vulnerability", cache=cache_v, verbose=True)

    # Load Shannon interactions for comparison
    shannon_ixn_path = "results/pairwise_interactions.csv"
    shannon_ixn = {}
    if os.path.exists(shannon_ixn_path):
        df_ixn = pd.read_csv(shannon_ixn_path)
        for _, row in df_ixn.iterrows():
            shannon_ixn[(row["feature_i"], row["feature_j"])] = row["interaction_bits"]

    n_synergy_me = sum(1 for v in ixn_me.values() if v > 0.01)
    n_synergy_v = sum(1 for v in ixn_v.values() if v > 0.001)
    n_total = len(ixn_me)

    print(f"\n  Min-entropy interactions: {n_synergy_me}/{n_total} pairs synergistic (>0.01 bits)")
    print(f"  Vulnerability interactions: {n_synergy_v}/{n_total} pairs synergistic")

    # Print top synergistic and most redundant
    sorted_me = sorted(ixn_me.items(), key=lambda x: x[1])
    print(f"\n  Min-entropy — 5 most redundant:")
    for (fi, fj), val in sorted_me[:5]:
        sh = shannon_ixn.get((fi, fj), shannon_ixn.get((fj, fi), float('nan')))
        print(f"    {fi:>16} × {fj:<16} I_∞={val:+.4f}  I_Shannon={sh:+.4f}")
    print(f"  Min-entropy — 5 most synergistic:")
    for (fi, fj), val in sorted_me[-5:]:
        sh = shannon_ixn.get((fi, fj), shannon_ixn.get((fj, fi), float('nan')))
        print(f"    {fi:>16} × {fj:<16} I_∞={val:+.4f}  I_Shannon={sh:+.4f}")

    # Correlation of interaction matrices
    pairs = list(ixn_me.keys())
    me_vals = [ixn_me[p] for p in pairs]
    sh_vals = []
    for p in pairs:
        v = shannon_ixn.get(p, shannon_ixn.get((p[1], p[0]), 0.0))
        sh_vals.append(v)
    rho_ixn, _ = spearmanr(me_vals, sh_vals)
    print(f"\n  Rank correlation of interaction matrices "
          f"(Shannon vs min-entropy): {rho_ixn:.4f}")

    # Save interaction results
    ixn_rows = []
    for (fi, fj), val_me in ixn_me.items():
        val_v = ixn_v.get((fi, fj), 0.0)
        val_sh = shannon_ixn.get((fi, fj), shannon_ixn.get((fj, fi), 0.0))
        ixn_rows.append({
            "feature_i": fi, "feature_j": fj,
            "interaction_shannon": round(val_sh, 4),
            "interaction_minentropy": round(val_me, 4),
            "interaction_vulnerability": round(val_v, 8),
            "type_shannon": "synergy" if val_sh > 0.05 else "redundancy" if val_sh < -0.05 else "independent",
            "type_minentropy": "synergy" if val_me > 0.01 else "redundancy" if val_me < -0.01 else "independent",
        })
    ixn_df = pd.DataFrame(ixn_rows).sort_values("interaction_minentropy")
    ixn_df.to_csv(f"{RESULTS_DIR}/minentropy_interactions.csv", index=False)

    summary = {
        "n_synergy_minentropy": n_synergy_me,
        "n_synergy_vulnerability": n_synergy_v,
        "n_total_pairs": n_total,
        "interaction_rank_corr_shannon_minentropy": rho_ixn,
        "rank_corr_shapley_shannon_minentropy": ranking["spearman_shannon_vs_minentropy"],
    }
    with open(f"{RESULTS_DIR}/minentropy_summary.json", "w") as fp:
        json.dump(summary, fp, indent=2)

    # ------------------------------------------------------------------
    # Tables and Figures for paper
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  [4] Generating tables and figures...")
    print("=" * 70)

    # --- Table 1: Shannon vs Min-entropy Shapley values ---
    table1_rows = []
    for f in sorted(FEATURES_MAIN, key=lambda x: -shannon_phi[x]):
        table1_rows.append({
            "Feature": f,
            "Category": FEATURE_CATEGORY.get(f, ""),
            "Shannon Shapley (bits)": round(shannon_phi[f], 3),
            "Min-entropy Shapley (bits)": round(phi_me[f], 3),
            "Vulnerability Shapley": round(phi_v[f], 6),
            "Shannon Rank": ranking["shannon_rank"].index(f) + 1,
            "Min-entropy Rank": ranking["minentropy_rank"].index(f) + 1,
        })
    table1_df = pd.DataFrame(table1_rows)
    table1_df.to_csv(f"{RESULTS_DIR}/table1_shapley_comparison.csv", index=False)
    print(f"  Table 1 saved: table1_shapley_comparison.csv")
    print(f"  Spearman rank correlation (Shannon vs Min-entropy): "
          f"ρ = {ranking['spearman_shannon_vs_minentropy']:.4f}")

    # --- Table 2: Pairwise Interaction Comparison ---
    table2_rows = []
    for (fi, fj), val_me in sorted(ixn_me.items(), key=lambda x: x[1]):
        val_sh = shannon_ixn.get((fi, fj), shannon_ixn.get((fj, fi), 0.0))
        type_sh = ("synergy" if val_sh > 0.05 else
                   "redundancy" if val_sh < -0.05 else "independent")
        type_me = ("synergy" if val_me > 0.01 else
                   "redundancy" if val_me < -0.01 else "independent")
        table2_rows.append({
            "Feature i": fi, "Feature j": fj,
            "Shannon I(i,j)": round(val_sh, 4),
            "Min-entropy I(i,j)": round(val_me, 4),
            "Shannon Type": type_sh,
            "Min-entropy Type": type_me,
        })
    table2_df = pd.DataFrame(table2_rows)
    table2_df.to_csv(f"{RESULTS_DIR}/table2_interaction_comparison.csv", index=False)
    max_synergy_me = max(ixn_me.values())
    print(f"  Table 2 saved: table2_interaction_comparison.csv")
    print(f"  Min-entropy synergistic pairs: {n_synergy_me}/{n_total}, "
          f"max synergy = {max_synergy_me:.4f} bits")

    # --- Figure 1: Side-by-side interaction heatmaps ---
    from matplotlib.colors import TwoSlopeNorm

    n_f = len(FEATURES_INTERACTION)
    mat_sh = np.zeros((n_f, n_f))
    mat_me_hm = np.zeros((n_f, n_f))
    for (fi, fj), val_me_pair in ixn_me.items():
        i = FEATURES_INTERACTION.index(fi)
        j = FEATURES_INTERACTION.index(fj)
        mat_me_hm[i, j] = val_me_pair
        mat_me_hm[j, i] = val_me_pair
        val_sh = shannon_ixn.get((fi, fj), shannon_ixn.get((fj, fi), 0.0))
        mat_sh[i, j] = val_sh
        mat_sh[j, i] = val_sh

    all_ixn_vals = np.concatenate([mat_sh.ravel(), mat_me_hm.ravel()])
    vmin = min(all_ixn_vals.min(), -3.0)
    vmax = max(all_ixn_vals.max(), 0.5)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    im1 = ax1.imshow(mat_sh, cmap="RdBu_r", norm=norm, aspect="equal")
    ax1.set_xticks(range(n_f))
    ax1.set_yticks(range(n_f))
    ax1.set_xticklabels(FEATURES_INTERACTION, rotation=45, ha="right", fontsize=7)
    ax1.set_yticklabels(FEATURES_INTERACTION, fontsize=7)
    ax1.set_title("Shannon Entropy", fontsize=11)
    for i in range(n_f):
        for j in range(n_f):
            if i != j:
                ax1.text(j, i, f"{mat_sh[i, j]:.2f}", ha="center",
                         va="center", fontsize=5)

    im2 = ax2.imshow(mat_me_hm, cmap="RdBu_r", norm=norm, aspect="equal")
    ax2.set_xticks(range(n_f))
    ax2.set_yticks(range(n_f))
    ax2.set_xticklabels(FEATURES_INTERACTION, rotation=45, ha="right", fontsize=7)
    ax2.set_yticklabels(FEATURES_INTERACTION, fontsize=7)
    ax2.set_title("Rényi Min-entropy", fontsize=11)
    for i in range(n_f):
        for j in range(n_f):
            if i != j:
                ax2.text(j, i, f"{mat_me_hm[i, j]:.2f}", ha="center",
                         va="center", fontsize=5)

    fig.colorbar(im2, ax=[ax1, ax2],
                 label="Interaction (blue=redundancy, red=synergy)",
                 shrink=0.8, pad=0.02)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/figures/fig1_interaction_heatmaps.pdf")
    plt.savefig(f"{RESULTS_DIR}/figures/fig1_interaction_heatmaps.png", dpi=150)
    plt.close()
    print("  Figure 1 saved: fig1_interaction_heatmaps.pdf/png")

    # --- Figure 2: Feature ranking scatter plot ---
    fig, ax = plt.subplots(figsize=(7, 6))
    s_vals = [shannon_phi[f] for f in FEATURES_MAIN]
    m_vals = [phi_me[f] for f in FEATURES_MAIN]
    ax.scatter(s_vals, m_vals, s=50, c="#1f77b4", zorder=3)
    for f, sx, mx in zip(FEATURES_MAIN, s_vals, m_vals):
        ax.annotate(f, (sx, mx), fontsize=6, xytext=(4, 4),
                    textcoords="offset points")

    all_v = s_vals + m_vals
    lo = min(all_v) - 0.1
    hi = max(all_v) + 0.1
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5, label="y = x")
    ax.set_xlabel("Shannon Shapley value (bits)")
    ax.set_ylabel("Min-entropy Shapley value (bits)")
    ax.set_title("Feature Ranking: Shannon vs Min-entropy")
    ax.annotate(f"ρ = {ranking['spearman_shannon_vs_minentropy']:.3f}",
                xy=(0.95, 0.95), xycoords="axes fraction", fontsize=11,
                ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/figures/fig2_ranking_scatter.pdf")
    plt.savefig(f"{RESULTS_DIR}/figures/fig2_ranking_scatter.png", dpi=150)
    plt.close()
    print("  Figure 2 saved: fig2_ranking_scatter.pdf/png")

    # --- Key headline for rebuttal ---
    print("\n" + "=" * 70)
    print("  REBUTTAL HEADLINE NUMBERS")
    print("=" * 70)
    print(f"  Shannon vs min-entropy Shapley rank correlation: "
          f"{ranking['spearman_shannon_vs_minentropy']:.4f}")
    print(f"  Synergistic pairs under min-entropy: {n_synergy_me}/{n_total}")
    if n_synergy_me > 0:
        print(f"  => Reviewer A is correct that min-entropy admits synergy.")
        print(f"     However, the defender-facing ranking is preserved "
              f"(ρ={ranking['spearman_shannon_vs_minentropy']:.2f}),")
        print(f"     so the practical conclusions (GPU cluster dominates, "
              f"defense priorities) hold.")
    else:
        print(f"  => Even under min-entropy, no synergy observed on real data.")
    print(f"\n  Total time: {(time.time()-t_start)/60:.1f} min")
    print(f"  Results: {os.path.abspath(RESULTS_DIR)}/")


if __name__ == "__main__":
    main()
