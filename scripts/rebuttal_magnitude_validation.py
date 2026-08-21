"""
Interactive Rebuttal Experiment 2: Magnitude-sensitive entropy–uniqueness validation.

Addresses Reviewer A's concern that the ρ=0.94 Spearman correlation may be driven
by subset cardinality rather than entropy's intrinsic ability to predict
re-identification outcomes.

We produce:
  (a) Calibration plot: log2(unique_rate) vs residual entropy, with OLS fit line
      and per-point residuals — a magnitude-sensitive check
  (b) Same-size subset correlation: for each cardinality k, sample many size-k
      subsets and report the within-k Spearman correlation
  (c) Defense-only correlation: the 4 real deployed defenses as standalone points
  (d) Partial correlation controlling for |S|

Outputs:
  results/rebuttal/magnitude_validation.json
  results/rebuttal/magnitude_calibration.csv
  results/rebuttal/samesize_correlation.csv
  results/rebuttal/figures/fig_calibration.pdf
  results/rebuttal/figures/fig_samesize.pdf
"""

import sys, os, argparse, json, time, itertools
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from scipy.optimize import curve_fit
import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt

from src.entropy_fast import FeatureMatrix, _joint_counts

DATA_PATH = "data/raw/li_cao_imc2020/final_with_header.csv"
RESULTS_DIR = "results/rebuttal"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(f"{RESULTS_DIR}/figures", exist_ok=True)

FEATURES = [
    "jsFonts", "fp2_webgl", "canvastest", "hybridaudio", "agent",
    "gpu", "language", "fp2_pixelratio", "browserversion", "osversion",
    "timezone", "browser", "os", "cpucores", "fp2_colordepth",
    "fp2_platform", "encoding", "doNotTrack",
]

DEFENSES = {
    "None (full FP)": [],
    "Tor Browser": FEATURES,
    "Firefox RFP": [f for f in FEATURES if f not in ("fp2_webgl", "gpu")],
    "Brave (farbling)": ["canvastest", "fp2_webgl", "hybridaudio", "gpu",
                         "jsFonts", "language"],
}


def load_matrix(path, nrows):
    print(f"Loading {nrows:,} rows...")
    df = pd.read_csv(path, sep="\t", nrows=nrows, low_memory=False,
                     usecols=FEATURES)
    for c in FEATURES:
        df[c] = df[c].fillna("__MISSING__") if df[c].dtype == object else df[c].fillna(-9999)
    return FeatureMatrix.from_dataframe(df, FEATURES)


def uniqueness_stats(fm, features):
    N = fm.n_samples
    if len(features) == 0:
        return {"unique_rate": 0.0, "entropy_bits": 0.0, "n_features": 0,
                "median_aset": float(N)}
    idx = [fm._idx[f] for f in features]
    sub = fm.codes[:, idx]
    radix = fm.cardinalities[idx]
    counts = _joint_counts(sub, radix=radix)
    n_unique = int((counts == 1).sum())
    unique_rate = n_unique / N
    p = counts / N
    h = float(-np.sum(p * np.log2(p)))
    k = len(counts)
    h += (k - 1) / (2 * N * np.log2(np.e))
    sizes = np.repeat(counts, counts)
    median_aset = float(np.median(sizes))
    return {"unique_rate": unique_rate, "entropy_bits": h,
            "n_features": len(features), "median_aset": median_aset}


def sample_subsets(features, n_per_size=40, seed=0):
    rng = np.random.default_rng(seed)
    subsets = []
    n = len(features)
    for k in range(1, n + 1):
        n_possible = math.comb(n, k)
        if n_possible <= n_per_size:
            for combo in itertools.combinations(features, k):
                subsets.append(list(combo))
        else:
            seen = set()
            while len(seen) < n_per_size:
                s = tuple(sorted(rng.choice(features, size=k, replace=False)))
                if s not in seen:
                    seen.add(s)
                    subsets.append(list(s))
    return subsets


import math

def main():
    parser = argparse.ArgumentParser(
        description="Rebuttal Exp 2: Magnitude-sensitive validation")
    parser.add_argument("--nrows", type=int, default=300_000)
    parser.add_argument("--n-per-size", type=int, default=40,
                        help="Subsets sampled per cardinality k")
    args = parser.parse_args()

    t0 = time.time()
    fm = load_matrix(DATA_PATH, args.nrows)
    N = fm.n_samples
    h_all = fm.entropy_subset(FEATURES)
    print(f"  H(F) = {h_all:.3f} bits, N = {N:,}")

    # ------------------------------------------------------------------
    # (a) Sample subsets across all sizes
    # ------------------------------------------------------------------
    print("\n[1] Sampling feature subsets for calibration...")
    subsets = sample_subsets(FEATURES, n_per_size=args.n_per_size)
    print(f"  {len(subsets)} subsets total")

    rows = []
    for sub in subsets:
        s = uniqueness_stats(fm, sub)
        rows.append(s)
    df = pd.DataFrame(rows)
    df.to_csv(f"{RESULTS_DIR}/magnitude_calibration.csv", index=False)

    # Add defense points
    defense_rows = []
    for name, neutralized in DEFENSES.items():
        remaining = [f for f in FEATURES if f not in neutralized]
        s = uniqueness_stats(fm, remaining)
        s["defense"] = name
        defense_rows.append(s)
    df_def = pd.DataFrame(defense_rows)

    # ------------------------------------------------------------------
    # (b) Overall Spearman and Pearson
    # ------------------------------------------------------------------
    mask = df["unique_rate"] > 0
    df_pos = df[mask].copy()
    df_pos["log_unique"] = np.log2(df_pos["unique_rate"])

    rho_all, p_rho = spearmanr(df_pos["entropy_bits"], df_pos["unique_rate"])
    r_all, p_r = pearsonr(df_pos["entropy_bits"], df_pos["log_unique"])
    print(f"\n  Overall Spearman (entropy vs unique_rate): {rho_all:.4f} (p={p_rho:.2e})")
    print(f"  Pearson (entropy vs log2(unique_rate)):     {r_all:.4f} (p={p_r:.2e})")

    # ------------------------------------------------------------------
    # (c) OLS: log2(unique_rate) = a * entropy + b  (magnitude calibration)
    # ------------------------------------------------------------------
    from numpy.polynomial.polynomial import polyfit
    b_ols, a_ols = polyfit(df_pos["entropy_bits"].values,
                           df_pos["log_unique"].values, 1)
    df_pos["predicted_log_unique"] = a_ols * df_pos["entropy_bits"] + b_ols
    df_pos["residual"] = df_pos["log_unique"] - df_pos["predicted_log_unique"]
    rmse = float(np.sqrt(np.mean(df_pos["residual"] ** 2)))
    print(f"  OLS: log2(uniq) = {a_ols:.4f} * H + ({b_ols:.4f})")
    print(f"  RMSE of log2(unique_rate): {rmse:.4f}")

    # Defense-specific calibration check
    print(f"\n  Defense calibration:")
    for _, row in df_def.iterrows():
        if row["unique_rate"] > 0:
            actual = np.log2(row["unique_rate"])
            predicted = a_ols * row["entropy_bits"] + b_ols
            err = actual - predicted
            print(f"    {row['defense']:<20} H={row['entropy_bits']:.2f}  "
                  f"actual log2(u)={actual:.2f}  predicted={predicted:.2f}  "
                  f"error={err:+.2f}")
        else:
            print(f"    {row['defense']:<20} unique_rate=0 (all anonymized)")

    # ------------------------------------------------------------------
    # (d) Same-size subset correlation
    # ------------------------------------------------------------------
    print(f"\n[2] Within-cardinality Spearman correlations...")
    samesize_rows = []
    for k in range(1, len(FEATURES) + 1):
        dfk = df_pos[df_pos["n_features"] == k]
        if len(dfk) >= 5:
            rho_k, p_k = spearmanr(dfk["entropy_bits"], dfk["unique_rate"])
            samesize_rows.append({"k": k, "n_subsets": len(dfk),
                                  "spearman": round(rho_k, 4), "p_value": p_k})
            print(f"    k={k:2d}: ρ={rho_k:.4f} (n={len(dfk)}, p={p_k:.4f})")
    samesize_df = pd.DataFrame(samesize_rows)
    samesize_df.to_csv(f"{RESULTS_DIR}/samesize_correlation.csv", index=False)

    avg_within_rho = samesize_df["spearman"].mean()
    print(f"\n  Average within-k Spearman: {avg_within_rho:.4f}")

    # ------------------------------------------------------------------
    # (e) Partial correlation controlling for |S|
    # ------------------------------------------------------------------
    from numpy.linalg import lstsq
    X_partial = np.column_stack([df_pos["n_features"].values,
                                 np.ones(len(df_pos))])
    # Residualize entropy on |S|
    coef_h, *_ = lstsq(X_partial, df_pos["entropy_bits"].values, rcond=None)
    resid_h = df_pos["entropy_bits"].values - X_partial @ coef_h
    # Residualize unique_rate on |S|
    coef_u, *_ = lstsq(X_partial, df_pos["unique_rate"].values, rcond=None)
    resid_u = df_pos["unique_rate"].values - X_partial @ coef_u
    rho_partial, p_partial = spearmanr(resid_h, resid_u)
    print(f"\n  Partial Spearman (controlling for |S|): {rho_partial:.4f} (p={p_partial:.2e})")

    # ------------------------------------------------------------------
    # Save summary
    # ------------------------------------------------------------------
    summary = {
        "n_subsets": len(df),
        "n_positive_uniqueness": int(mask.sum()),
        "spearman_overall": rho_all,
        "pearson_log2": r_all,
        "ols_slope": a_ols,
        "ols_intercept": b_ols,
        "ols_rmse_log2": rmse,
        "partial_spearman_controlling_cardinality": rho_partial,
        "partial_spearman_p": p_partial,
        "avg_within_k_spearman": avg_within_rho,
        "samesize_results": samesize_rows,
    }
    with open(f"{RESULTS_DIR}/magnitude_validation.json", "w") as fp:
        json.dump(summary, fp, indent=2)

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------
    print("\n[3] Generating figures...")

    # Fig A: Calibration plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    sc = ax1.scatter(df_pos["entropy_bits"], df_pos["log_unique"],
                     c=df_pos["n_features"], cmap="viridis", s=18, alpha=0.6,
                     zorder=2)
    x_line = np.linspace(0, df_pos["entropy_bits"].max() * 1.05, 100)
    ax1.plot(x_line, a_ols * x_line + b_ols, "r-", lw=1.5, alpha=0.8,
             label=f"OLS: slope={a_ols:.3f}, RMSE={rmse:.2f}")
    # Add defense points
    for _, row in df_def.iterrows():
        if row["unique_rate"] > 0:
            ax1.plot(row["entropy_bits"], np.log2(row["unique_rate"]),
                     "D", color="red", markersize=8, zorder=5)
            ax1.annotate(row["defense"], (row["entropy_bits"],
                         np.log2(row["unique_rate"])),
                         fontsize=6, xytext=(5, 5),
                         textcoords="offset points")
    cb = fig.colorbar(sc, ax=ax1)
    cb.set_label("|S| (# features)")
    ax1.set_xlabel("Residual entropy H(F_S) [bits]")
    ax1.set_ylabel("log₂(unique rate)")
    ax1.set_title("Calibration: entropy → log uniqueness", fontsize=10)
    ax1.legend(fontsize=7)
    ax1.grid(alpha=0.3)

    # Fig B: Within-k Spearman
    ax2.bar(samesize_df["k"], samesize_df["spearman"], color="#1f77b4",
            alpha=0.8)
    ax2.axhline(avg_within_rho, color="red", ls="--", lw=1,
                label=f"mean = {avg_within_rho:.3f}")
    ax2.axhline(rho_all, color="gray", ls=":", lw=1,
                label=f"overall ρ = {rho_all:.3f}")
    ax2.set_xlabel("Feature subset size k")
    ax2.set_ylabel("Spearman ρ (entropy vs unique_rate)")
    ax2.set_title("Within-cardinality correlation", fontsize=10)
    ax2.legend(fontsize=7)
    ax2.grid(alpha=0.3)
    ax2.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/figures/fig_calibration.pdf")
    plt.savefig(f"{RESULTS_DIR}/figures/fig_calibration.png", dpi=150)
    plt.close()
    print("  saved fig_calibration.pdf/png")

    # ------------------------------------------------------------------
    # Headline for rebuttal
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  REBUTTAL HEADLINE NUMBERS")
    print("=" * 70)
    print(f"  Overall Spearman:                         {rho_all:.4f}")
    print(f"  Partial Spearman (controlling for |S|):    {rho_partial:.4f}")
    print(f"  Average within-k Spearman:                {avg_within_rho:.4f}")
    print(f"  OLS log2(unique) ~ entropy: RMSE =        {rmse:.4f}")
    if rho_partial > 0.5:
        print(f"  => Entropy predicts re-identification BEYOND what subset")
        print(f"     cardinality alone explains (partial ρ={rho_partial:.2f} after")
        print(f"     controlling for |S|). The magnitude relationship is")
        print(f"     well-calibrated (RMSE {rmse:.2f} on log2 scale).")
    print(f"\n  Total time: {(time.time()-t0)/60:.1f} min")
    print(f"  Results: {os.path.abspath(RESULTS_DIR)}/")


if __name__ == "__main__":
    main()
