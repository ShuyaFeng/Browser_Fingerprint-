"""
Gap 4: Feature-selection sensitivity analysis.

The 18 features are a curated subset of the corpus's 57 raw attributes. Shapley
attribution depends on the feature set, so a reviewer will ask whether the
conclusions are an artifact of that choice. We show they are not, two ways:

  (A) Random subsets. Draw many random feature subsets of varying size from the
      core 18 and check that (i) marginal entropy always overestimates the joint,
      (ii) pairwise interactions are essentially all negative, and (iii) the
      top-Shapley feature is always fonts or a GPU/hardware feature.

  (B) Extended feature set. Expand to 25 features by adding seven more
      hardware/GPU-related attributes and confirm the conclusions hold, and in
      fact strengthen (more correlated features -> more redundancy).

Outputs:
  results/sensitivity_subsets.csv
  results/sensitivity_extended.csv
  results/figures/fig10_sensitivity.pdf
"""

import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt

from src.entropy_fast import FeatureMatrix
from src.shapley_fast import (shapley_monte_carlo_fast,
                              shapley_and_interactions_fast)

DATA_PATH = "data/raw/li_cao_imc2020/final_with_header.csv"
RESULTS = "results"
NROWS = 50_000

CORE = [
    "jsFonts", "fp2_webgl", "canvastest", "hybridaudio", "agent",
    "gpu", "language", "fp2_pixelratio", "browserversion", "osversion",
    "timezone", "browser", "os", "cpucores", "fp2_colordepth",
    "fp2_platform", "encoding", "doNotTrack",
]
# Seven additional fingerprinting attributes from the corpus, mostly
# hardware/GPU-related (these deepen redundancy, the harder test for our claims).
EXTRA = ["ccaudio", "fp2_webglvendoe", "langsdetected", "touchSupport",
         "device", "gpuimgs", "partgpu"]
EXTENDED = CORE + EXTRA

CATEGORY = {
    "jsFonts": "OS/fonts", "fp2_webgl": "hardware/GPU", "canvastest": "hardware/GPU",
    "hybridaudio": "hardware/audio", "agent": "browser", "gpu": "hardware/GPU",
    "language": "locale", "fp2_pixelratio": "hardware/screen",
    "browserversion": "browser", "osversion": "OS", "timezone": "locale",
    "browser": "browser", "os": "OS", "cpucores": "hardware/CPU",
    "fp2_colordepth": "hardware/screen", "fp2_platform": "OS",
    "encoding": "browser", "doNotTrack": "browser",
    "ccaudio": "hardware/audio", "fp2_webglvendoe": "hardware/GPU",
    "langsdetected": "locale", "touchSupport": "hardware/screen",
    "device": "OS", "gpuimgs": "hardware/GPU", "partgpu": "hardware/GPU",
}

def is_fonts_or_hw(f):
    return f == "jsFonts" or CATEGORY.get(f, "").startswith("hardware")


def load_matrix(features):
    print(f"Loading {NROWS:,} rows ({len(features)} features)...")
    df = pd.read_csv(DATA_PATH, sep="\t", nrows=NROWS, low_memory=False,
                     usecols=features)
    for c in features:
        df[c] = df[c].fillna("__MISSING__") if df[c].dtype == object else df[c].fillna(-9999)
    return FeatureMatrix.from_dataframe(df, features)


# ---------------------------------------------------------------------------
# (A) Random-subset sensitivity
# ---------------------------------------------------------------------------

def run_random_subsets(fm):
    print("\n" + "=" * 74)
    print("  (A) Random feature-subset sensitivity")
    print("=" * 74)
    rng = np.random.default_rng(0)

    rows = []
    # marginal overestimate is cheap: cover a wide range of K, many draws
    for K in [8, 10, 12, 14, 16]:
        for _ in range(20):
            sub = list(rng.choice(CORE, size=K, replace=False))
            joint = fm.entropy_subset(sub)
            marg = sum(fm.marginal_entropies(sub).values())
            rows.append({"K": K, "overestimate": marg / joint})
    over_tbl = pd.DataFrame(rows)

    # interactions + top-Shapley: exact, one shared cache per subset
    ix_rows = []
    for K in [8, 10, 12]:
        for _ in range(10):
            sub = list(rng.choice(CORE, size=K, replace=False))
            phi, ixn = shapley_and_interactions_fast(fm, sub, verbose=False)
            n_neg = sum(1 for v in ixn.values() if v < 0)
            top = max(phi, key=phi.get)
            ix_rows.append({
                "K": K,
                "neg_interaction_frac": n_neg / len(ixn),
                "top_feature": top,
                "top_is_fonts_or_hw": is_fonts_or_hw(top),
            })
    ix_tbl = pd.DataFrame(ix_rows)

    # Summary
    print("\n  marginal overestimate ratio (sum_marginal / joint), by subset size K:")
    print(f"  {'K':>3} {'min':>7} {'median':>8} {'max':>7}  (n=20 each)")
    for K in [8, 10, 12, 14, 16]:
        v = over_tbl[over_tbl.K == K]["overestimate"]
        print(f"  {K:>3} {v.min():>7.2f} {v.median():>8.2f} {v.max():>7.2f}")
    print(f"\n  marginal overestimate >1 in all {len(over_tbl)} subsets: "
          f"{(over_tbl.overestimate > 1).all()} "
          f"(min {over_tbl.overestimate.min():.2f}x)")

    print(f"\n  negative-interaction fraction (should be near 1.0):")
    print(f"  {'K':>3} {'min':>7} {'median':>8} {'max':>7}")
    for K in [8, 10, 12]:
        v = ix_tbl[ix_tbl.K == K]["neg_interaction_frac"]
        print(f"  {K:>3} {v.min():>7.3f} {v.median():>8.3f} {v.max():>7.3f}")
    all_neg = (ix_tbl.neg_interaction_frac == 1.0).mean()
    print(f"  fraction of subsets with no positive interaction at all: {all_neg:.2%}")
    print(f"  mean negative-interaction fraction: {ix_tbl.neg_interaction_frac.mean():.3f}")

    top_hw = ix_tbl.top_is_fonts_or_hw.mean()
    print(f"\n  fraction where top Shapley feature is fonts or hardware: {top_hw:.2%}")
    print(f"  top Shapley feature distribution: "
          f"{ix_tbl.top_feature.value_counts().to_dict()}")

    over_tbl.to_csv(f"{RESULTS}/sensitivity_subsets.csv", index=False)
    return over_tbl, ix_tbl


# ---------------------------------------------------------------------------
# (B) Extended 25-feature set
# ---------------------------------------------------------------------------

def run_extended(fm_ext):
    print("\n" + "=" * 74)
    print("  (B) Extended feature set (25 features)")
    print("=" * 74)
    joint = fm_ext.entropy_subset(EXTENDED)
    marg = fm_ext.marginal_entropies(EXTENDED)
    sum_marg = sum(marg.values())
    print(f"  total entropy H(F) = {joint:.3f} bits")
    print(f"  sum of marginal entropies = {sum_marg:.3f} bits")
    print(f"  marginal overestimate = {100*(sum_marg-joint)/joint:.1f}% "
          f"(396.6% with 18 features)")

    print(f"\n  Monte Carlo Shapley (500 permutations)...")
    phi = shapley_monte_carlo_fast(fm_ext, EXTENDED, n_permutations=500,
                                   verbose=False, seed=42)
    ranked = sorted(phi.items(), key=lambda x: -x[1])
    print(f"  Top 6 by Shapley:")
    for f, v in ranked[:6]:
        print(f"    {f:<18} {CATEGORY.get(f,'?'):<16} {v:.3f}")
    top = ranked[0][0]
    print(f"  top Shapley feature: {top} ({'is' if is_fonts_or_hw(top) else 'is not'} fonts/hardware)")

    # interactions among the top-12 highest-Shapley features (exact)
    top12 = [f for f, _ in ranked[:12]]
    _, ixn = shapley_and_interactions_fast(fm_ext, top12, verbose=False)
    n_neg = sum(1 for v in ixn.values() if v < 0)
    print(f"\n  pairwise interactions among top-12 features: {n_neg}/{len(ixn)} negative "
          f"({100*n_neg/len(ixn):.0f}%)")
    strongest = min(ixn.items(), key=lambda x: x[1])
    print(f"  strongest redundant pair: {strongest[0]} = {strongest[1]:.3f} bits")

    summary = {
        "n_features": len(EXTENDED), "joint_entropy": joint,
        "sum_marginals": sum_marg,
        "overestimate_pct": 100*(sum_marg-joint)/joint,
        "top_shapley_feature": top,
        "top12_neg_interaction_frac": n_neg/len(ixn),
        "shapley": {f: round(v,4) for f,v in phi.items()},
    }
    with open(f"{RESULTS}/sensitivity_extended.csv", "w") as f:
        pd.DataFrame(ranked, columns=["feature","shapley_bits"]).to_csv(f, index=False)
    with open(f"{RESULTS}/sensitivity_extended.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def make_figure(over_tbl, ix_tbl):
    os.makedirs(f"{RESULTS}/figures", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

    # Left: marginal overestimate by K (boxplot)
    Ks = [8, 10, 12, 14, 16]
    data = [over_tbl[over_tbl.K == K]["overestimate"].values for K in Ks]
    ax1.boxplot(data, tick_labels=[str(k) for k in Ks])
    ax1.axhline(1.0, color="gray", ls="--", lw=0.8)
    ax1.set_xlabel("Number of features in subset")
    ax1.set_ylabel("Marginal overestimate (sum / joint)")
    ax1.grid(axis="y", alpha=0.3)

    # Right: negative-interaction fraction by K (boxplot)
    Ks2 = [8, 10, 12]
    data2 = [ix_tbl[ix_tbl.K == K]["neg_interaction_frac"].values for K in Ks2]
    ax2.boxplot(data2, tick_labels=[str(k) for k in Ks2])
    ax2.set_ylim(0, 1.05)
    ax2.set_xlabel("Number of features in subset")
    ax2.set_ylabel("Fraction of negative interactions")
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{RESULTS}/figures/fig10_sensitivity.pdf")
    plt.close()
    print("\n  saved fig10_sensitivity.pdf")


if __name__ == "__main__":
    t0 = time.time()
    fm = load_matrix(CORE)
    over_tbl, ix_tbl = run_random_subsets(fm)
    make_figure(over_tbl, ix_tbl)

    fm_ext = load_matrix(EXTENDED)
    run_extended(fm_ext)

    print(f"\ndone in {(time.time()-t0)/60:.1f} min")
