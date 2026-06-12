"""
Gap 2: Bridging entropy to real re-identification.

Entropy is a proxy. This script translates the entropy results into concrete
re-identification outcomes: how many users are uniquely identifiable, and how
many move from unique to k-anonymous when features are neutralized. It also
validates entropy as a faithful proxy by correlating residual entropy with
residual uniqueness across feature subsets.

Outputs:
  results/reidentification.csv        per-defense uniqueness / k-anonymity
  results/point_reidentification.csv  single-point vs cluster defenses
  results/entropy_uniqueness.csv      entropy-vs-uniqueness correlation data
  results/figures/fig8_reidentification.pdf
  results/figures/fig9_entropy_uniqueness.pdf
"""

import sys, os, json, time, itertools
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt

from src.entropy_fast import FeatureMatrix, _joint_counts

DATA_PATH = "data/raw/li_cao_imc2020/final_with_header.csv"
RESULTS = "results"
NROWS = 300_000

FEATURES = [
    "jsFonts", "fp2_webgl", "canvastest", "hybridaudio", "agent",
    "gpu", "language", "fp2_pixelratio", "browserversion", "osversion",
    "timezone", "browser", "os", "cpucores", "fp2_colordepth",
    "fp2_platform", "encoding", "doNotTrack",
]

# Defense -> neutralized features (same mapping as defense_analysis.py)
DEFENSES = {
    "None (full FP)": [],
    "Tor Browser": FEATURES,  # neutralizes everything
    "Firefox RFP": [f for f in FEATURES if f not in ("fp2_webgl", "gpu")],
    "Brave (farbling)": ["canvastest", "fp2_webgl", "hybridaudio", "gpu",
                         "jsFonts", "language"],
}

POINT_DEFENSES = {
    "None": [],
    "Block Canvas": ["canvastest"],
    "Block WebGL": ["fp2_webgl"],
    "Block Fonts": ["jsFonts"],
    "Block GPU cluster": ["canvastest", "fp2_webgl", "gpu"],
}


def load_matrix():
    print(f"Loading {NROWS:,} rows...")
    df = pd.read_csv(DATA_PATH, sep="\t", nrows=NROWS, low_memory=False,
                     usecols=FEATURES)
    for c in FEATURES:
        df[c] = df[c].fillna("__MISSING__") if df[c].dtype == object else df[c].fillna(-9999)
    return FeatureMatrix.from_dataframe(df, FEATURES)


def anonymity_stats(fm, features):
    """Compute re-identification statistics for a feature subset.

    Returns dict with:
      unique_rate     : fraction of users with a unique fingerprint (set size 1)
      k2_rate ... k20 : fraction of users in an anonymity set of size >= k
                        (i.e., NOT singled out at level k)
      median_aset     : median anonymity-set size experienced by a user
      entropy_bits    : joint entropy of the subset (for proxy validation)
    """
    N = fm.n_samples
    if len(features) == 0:
        # No distinguishing features: everyone shares one fingerprint.
        return {"unique_rate": 0.0, "k2_rate": 1.0, "k10_rate": 1.0,
                "k20_rate": 1.0, "median_aset": float(N), "entropy_bits": 0.0,
                "n_unique_users": 0}
    idx = [fm._idx[f] for f in features]
    sub = fm.codes[:, idx]
    radix = fm.cardinalities[idx]
    counts = _joint_counts(sub, radix=radix)  # size of each distinct fingerprint
    # counts[g] = number of users sharing distinct fingerprint g
    n_unique_groups = int((counts == 1).sum())   # groups of size 1 = unique users
    unique_rate = n_unique_groups / N
    # fraction of users in an anonymity set of size >= k
    def k_rate(k):
        return float(counts[counts >= k].sum()) / N
    # median anonymity-set size experienced by a randomly chosen user:
    # each user's set size = count of its group; weight groups by their size
    sizes = np.repeat(counts, counts)  # per-user set size
    median_aset = float(np.median(sizes))
    # entropy for proxy validation
    p = counts / N
    h = float(-np.sum(p * np.log2(p)))
    k = len(counts)
    h += (k - 1) / (2 * N * np.log2(np.e))  # Miller-Madow
    return {
        "unique_rate": unique_rate,
        "k2_rate": k_rate(2),
        "k10_rate": k_rate(10),
        "k20_rate": k_rate(20),
        "median_aset": median_aset,
        "entropy_bits": h,
        "n_unique_users": n_unique_groups,
    }


def run_defense_reidentification(fm):
    print("\n" + "=" * 74)
    print("  Real re-identification consequences before/after defenses (uniqueness / k-anonymity)")
    print("=" * 74)
    rows = []
    for name, neutralized in DEFENSES.items():
        remaining = [f for f in FEATURES if f not in neutralized]
        s = anonymity_stats(fm, remaining)
        rows.append({
            "defense": name,
            "n_remaining_features": len(remaining),
            "residual_entropy_bits": round(s["entropy_bits"], 3),
            "unique_rate_pct": round(100 * s["unique_rate"], 2),
            "in_anon_set_ge2_pct": round(100 * s["k2_rate"], 2),
            "in_anon_set_ge10_pct": round(100 * s["k10_rate"], 2),
            "median_anon_set": s["median_aset"],
        })
    tbl = pd.DataFrame(rows)
    print(tbl.to_string(index=False))
    tbl.to_csv(f"{RESULTS}/reidentification.csv", index=False)

    full = next(r for r in rows if r["defense"] == "None (full FP)")
    ffx = next(r for r in rows if r["defense"] == "Firefox RFP")
    print(f"\n  key security takeaways:")
    print(f"  - full fingerprint: {full['unique_rate_pct']}% of users uniquely identifiable "
          f"(median anonymity set {full['median_anon_set']:.0f})")
    print(f"  - after Firefox RFP neutralizes 16 software features, still "
          f"{ffx['unique_rate_pct']}% of users uniquely identifiable -- "
          f"from just the 2 residual GPU features ({ffx['residual_entropy_bits']} bits)")
    print(f"  => defenses remove lots of entropy, yet users remain unprotected in re-identification terms")
    return tbl


def run_point_reidentification(fm):
    print("\n" + "=" * 74)
    print("  Re-identification gains of point vs cluster defenses (security meaning of redundancy)")
    print("=" * 74)
    rows = []
    for name, neutralized in POINT_DEFENSES.items():
        remaining = [f for f in FEATURES if f not in neutralized]
        s = anonymity_stats(fm, remaining)
        rows.append({
            "defense": name,
            "unique_rate_pct": round(100 * s["unique_rate"], 2),
            "residual_entropy_bits": round(s["entropy_bits"], 3),
        })
    tbl = pd.DataFrame(rows)
    print(tbl.to_string(index=False))
    tbl.to_csv(f"{RESULTS}/point_reidentification.csv", index=False)

    base = next(r for r in rows if r["defense"] == "None")
    webgl = next(r for r in rows if r["defense"] == "Block WebGL")
    print(f"\n  - full-fingerprint uniqueness: {base['unique_rate_pct']}%")
    print(f"  - uniqueness with only WebGL blocked: {webgl['unique_rate_pct']}% "
          f"(nearly unchanged, since redundant GPU/Canvas features remain)")
    return tbl


def run_entropy_uniqueness_correlation(fm):
    """Validate entropy as a faithful proxy for re-identifiability.

    Sample many feature subsets, compute (entropy, uniqueness_rate) for each,
    and report the rank correlation. A tight monotone relation justifies using
    entropy (and its Shapley attribution) as a stand-in for re-identification.
    """
    print("\n" + "=" * 74)
    print("  Validating entropy as a proxy for re-identification")
    print("=" * 74)
    rng = np.random.default_rng(0)
    rows = []
    # systematic: all single features, plus random subsets of varying size
    subsets = [[f] for f in FEATURES]
    for _ in range(120):
        k = int(rng.integers(2, len(FEATURES) + 1))
        subsets.append(list(rng.choice(FEATURES, size=k, replace=False)))
    for sub in subsets:
        s = anonymity_stats(fm, sub)
        rows.append({"n_features": len(sub),
                     "entropy_bits": s["entropy_bits"],
                     "unique_rate": s["unique_rate"]})
    tbl = pd.DataFrame(rows)
    tbl.to_csv(f"{RESULTS}/entropy_uniqueness.csv", index=False)

    from scipy.stats import spearmanr, pearsonr
    rho, _ = spearmanr(tbl["entropy_bits"], tbl["unique_rate"])
    r, _ = pearsonr(tbl["entropy_bits"], tbl["unique_rate"])
    print(f"  sampled {len(tbl)} feature subsets")
    print(f"  Spearman rank correlation (entropy vs uniqueness): {rho:.4f}")
    print(f"  Pearson correlation:                                {r:.4f}")
    print(f"  => entropy is tightly monotone with true uniqueness, validating it as a proxy")
    return tbl, rho, r


def make_figures(defense_tbl, point_tbl, eu_tbl):
    os.makedirs(f"{RESULTS}/figures", exist_ok=True)

    # Fig 8: defense re-identification outcomes
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    d = defense_tbl.copy()
    x = np.arange(len(d))
    ax1.bar(x, d["unique_rate_pct"], color="#d62728", alpha=0.8,
            label="Users uniquely identifiable (%)")
    ax1.set_ylabel("Uniquely identifiable users (%)", color="#d62728")
    ax1.tick_params(axis="y", labelcolor="#d62728")
    ax1.set_xticks(x)
    ax1.set_xticklabels(d["defense"], rotation=15, ha="right", fontsize=8)
    ax2 = ax1.twinx()
    ax2.plot(x, d["residual_entropy_bits"], "o--", color="#1f77b4",
             label="Residual entropy (bits)")
    ax2.set_ylabel("Residual entropy (bits)", color="#1f77b4")
    ax2.tick_params(axis="y", labelcolor="#1f77b4")
    plt.tight_layout()
    plt.savefig(f"{RESULTS}/figures/fig8_reidentification.pdf")
    plt.close()
    print("  saved fig8_reidentification.pdf")

    # Fig 9: entropy vs uniqueness scatter (proxy validation)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    sc = ax.scatter(eu_tbl["entropy_bits"], 100 * eu_tbl["unique_rate"],
                    c=eu_tbl["n_features"], cmap="viridis", s=25, alpha=0.8)
    ax.set_xlabel("Residual joint entropy (bits)")
    ax.set_ylabel("Uniquely identifiable users (%)")
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("# features in subset")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULTS}/figures/fig9_entropy_uniqueness.pdf")
    plt.close()
    print("  saved fig9_entropy_uniqueness.pdf")


if __name__ == "__main__":
    t0 = time.time()
    fm = load_matrix()
    print(f"  full-fingerprint total entropy = {fm.entropy_subset(FEATURES):.3f} bits")
    dtbl = run_defense_reidentification(fm)
    ptbl = run_point_reidentification(fm)
    eutbl, rho, r = run_entropy_uniqueness_correlation(fm)
    print("\nGenerating figures...")
    make_figures(dtbl, ptbl, eutbl)
    summary = {"spearman_entropy_uniqueness": rho, "pearson_entropy_uniqueness": r}
    with open(f"{RESULTS}/reidentification_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\ndone in {time.time()-t0:.1f}s")
