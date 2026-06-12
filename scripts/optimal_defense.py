"""
Optimal defense selection (reviewer Item 7).

A defender who can neutralize k features should ask which k. The field implicitly
ranks features by marginal entropy, but marginal entropy double-counts the redundant
GPU cluster, so a marginal-ranked defense spends its budget re-removing the same
shared information. We compare four selection strategies by the residual joint
entropy they leave, H(F_kept):

  marginal-top-k : neutralize the k highest marginal-entropy features (the naive
                   baseline the literature's accounting implies).
  shapley-top-k  : neutralize the k highest Shapley-value features.
  greedy         : iteratively neutralize the feature with the largest conditional
                   entropy given those still visible, which is redundancy-aware.
  optimal        : the size-k set minimizing residual entropy, by exhaustive search
                   (feasible here for small k).

We also compute a cluster-level Shapley, grouping features into hardware, OS, fonts,
browser, locale, and screen, so a defender can reason about whole redundancy clusters
rather than individual features.

Outputs:
  results/optimal_defense.csv
  results/cluster_shapley.csv
  results/figures/fig14_optimal_defense.pdf
"""

import sys, os, json, time, itertools
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt

from src.entropy_fast import FeatureMatrix
from src.shapley_fast import shapley_and_interactions_fast

DATA = "data/raw/li_cao_imc2020/final_with_header.csv"
RESULTS = "results"
NROWS = 100_000
CORR = "miller_madow"  # residual entropy of large (saturated) kept sets; relative comparison
KMAX = 6
KOPT = 2  # exhaustive optimal up to this budget (anchors that greedy is near-optimal)

FEATURES = [
    "jsFonts", "fp2_webgl", "canvastest", "hybridaudio", "agent",
    "gpu", "language", "fp2_pixelratio", "browserversion", "osversion",
    "timezone", "browser", "os", "cpucores", "fp2_colordepth",
    "fp2_platform", "encoding", "doNotTrack",
]
CLUSTERS = {
    "Hardware/GPU": ["fp2_webgl", "canvastest", "gpu", "hybridaudio",
                     "fp2_pixelratio", "cpucores", "fp2_colordepth"],
    "OS": ["os", "osversion", "fp2_platform"],
    "Fonts": ["jsFonts"],
    "Browser": ["agent", "browserversion", "browser", "encoding", "doNotTrack"],
    "Locale": ["language", "timezone"],
}


def load():
    df = pd.read_csv(DATA, sep="\t", nrows=NROWS, low_memory=False, usecols=FEATURES)
    for c in FEATURES:
        df[c] = df[c].fillna("__M__").astype(str)
    return FeatureMatrix.from_dataframe(df, FEATURES)


def residual(fm, removed):
    kept = [f for f in FEATURES if f not in removed]
    return fm.entropy_subset(kept, correction=CORR) if kept else 0.0


def greedy_select(fm):
    """Iteratively neutralize the feature that most reduces residual entropy."""
    removed, seq = [], []
    for _ in range(KMAX):
        best_f, best_res = None, None
        for f in FEATURES:
            if f in removed:
                continue
            res = residual(fm, removed + [f])
            if best_res is None or res < best_res:
                best_res, best_f = res, f
        removed.append(best_f)
        seq.append((best_f, best_res))
    return seq


def optimal_select(fm, k):
    """Exhaustive size-k removal set minimizing residual entropy."""
    best, best_res = None, None
    for combo in itertools.combinations(FEATURES, k):
        res = residual(fm, combo)
        if best_res is None or res < best_res:
            best_res, best = res, combo
    return best, best_res


def run(fm):
    h_all = fm.entropy_subset(FEATURES, correction=CORR)
    marg = fm.marginal_entropies(FEATURES, correction=CORR)
    phi, _ = shapley_and_interactions_fast(fm, FEATURES, correction=CORR, verbose=False)
    marg_rank = sorted(FEATURES, key=lambda f: -marg[f])
    shap_rank = sorted(FEATURES, key=lambda f: -phi[f])
    greedy_seq = greedy_select(fm)
    print(f"  H(all) = {h_all:.3f} bits")
    print(f"  marginal rank: {marg_rank[:6]}")
    print(f"  shapley  rank: {shap_rank[:6]}")
    print(f"  greedy order : {[f for f, _ in greedy_seq[:6]]}")

    rows = []
    for k in range(1, KMAX + 1):
        r_marg = residual(fm, marg_rank[:k])
        r_shap = residual(fm, shap_rank[:k])
        r_greedy = greedy_seq[k - 1][1]
        r_opt = optimal_select(fm, k)[1] if k <= KOPT else np.nan
        rows.append({"k": k, "residual_marginal": round(r_marg, 4),
                     "residual_shapley": round(r_shap, 4),
                     "residual_greedy": round(r_greedy, 4),
                     "residual_optimal": round(r_opt, 4) if k <= KOPT else None})
        opt_s = f"{r_opt:.3f}" if k <= KOPT else "  -  "
        print(f"  k={k}: marginal {r_marg:.3f}  shapley {r_shap:.3f}  "
              f"greedy {r_greedy:.3f}  optimal {opt_s}")
    tbl = pd.DataFrame(rows)
    tbl.to_csv(f"{RESULTS}/optimal_defense.csv", index=False)

    # headline: budget to reach a target residual
    target = 0.5 * h_all
    def budget_to(col):
        hit = tbl[tbl[col].astype(float) <= target] if tbl[col].notna().any() else None
        return int(hit["k"].iloc[0]) if hit is not None and len(hit) else None
    bm, bs = budget_to("residual_marginal"), budget_to("residual_greedy")
    print(f"\n  to halve residual entropy ({target:.2f} bits): "
          f"marginal needs k={bm}, greedy needs k={bs}")
    return tbl, phi, marg, h_all


def cluster_shapley(fm):
    print("\n  Cluster-level Shapley (each cluster a super-feature):")
    names = list(CLUSTERS.keys())
    # Build a coalition game over clusters via a synthetic FeatureMatrix proxy:
    # compute Shapley by direct enumeration over 2^|clusters| using joint entropy.
    n = len(names)
    cache = {}
    def vS(mask):
        if mask in cache:
            return cache[mask]
        feats = []
        for b in range(n):
            if (mask >> b) & 1:
                feats += CLUSTERS[names[b]]
        v = fm.entropy_subset(feats, correction=CORR) if feats else 0.0
        cache[mask] = v
        return v
    import math
    fact = [math.factorial(i) for i in range(n + 1)]
    phi = {}
    for b, nm in enumerate(names):
        bit = 1 << b
        s = 0.0
        for mask in range(2 ** n):
            if mask & bit:
                continue
            sz = bin(mask).count("1")
            w = fact[sz] * fact[n - sz - 1] / fact[n]
            s += w * (vS(mask | bit) - vS(mask))
        phi[nm] = s
    h_all = vS(2 ** n - 1)
    rows = []
    for nm in names:
        cm = sum(fm.entropy_subset([f], correction=CORR) for f in CLUSTERS[nm])
        rows.append({"cluster": nm, "marginal_sum": round(cm, 3),
                     "cluster_shapley": round(phi[nm], 3),
                     "share_pct": round(100 * phi[nm] / h_all, 1)})
    tbl = pd.DataFrame(rows).sort_values("cluster_shapley", ascending=False)
    tbl.to_csv(f"{RESULTS}/cluster_shapley.csv", index=False)
    for _, r in tbl.iterrows():
        print(f"    {r['cluster']:<14} marg {r['marginal_sum']:6.2f} -> "
              f"Shapley {r['cluster_shapley']:5.2f} ({r['share_pct']:.0f}%)")
    return tbl


def make_figure(tbl, cl_tbl, h_all):
    os.makedirs(f"{RESULTS}/figures", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.7))

    k = tbl["k"]
    ax1.plot(k, tbl["residual_marginal"], "s-", color="#9e9e9e",
             label="Marginal-ranked (naive)")
    ax1.plot(k, tbl["residual_shapley"], "^-", color="#1f77b4", label="Shapley-ranked")
    ax1.plot(k, tbl["residual_greedy"], "o-", color="#d62728", label="Greedy")
    opt = tbl["residual_optimal"].astype(float)
    ax1.plot(k, opt, "x--", color="black", label="Optimal", markersize=8)
    ax1.set_xlabel("Features neutralized ($k$)")
    ax1.set_ylabel("Residual joint entropy (bits)")
    ax1.legend(fontsize=7.5); ax1.grid(alpha=0.3)

    cl = cl_tbl.sort_values("cluster_shapley")
    y = np.arange(len(cl))
    ax2.barh(y, cl["marginal_sum"], height=0.4, color="#cccccc",
             label="Sum of marginals")
    ax2.barh(y, cl["cluster_shapley"], height=0.4, color="#d62728",
             label="Cluster Shapley")
    ax2.set_yticks(y); ax2.set_yticklabels(cl["cluster"], fontsize=8)
    ax2.set_xlabel("Bits")
    ax2.legend(fontsize=7.5); ax2.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{RESULTS}/figures/fig14_optimal_defense.pdf")
    plt.close()
    print("\n  saved fig14_optimal_defense.pdf")


if __name__ == "__main__":
    t0 = time.time()
    print("=" * 74)
    print("  Optimal defense selection (Item 7)")
    print("=" * 74)
    fm = load()
    tbl, phi, marg, h_all = run(fm)
    cl_tbl = cluster_shapley(fm)
    make_figure(tbl, cl_tbl, h_all)
    print(f"\ndone in {(time.time()-t0)/60:.1f} min")
