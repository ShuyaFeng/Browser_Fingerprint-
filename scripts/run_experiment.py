"""
Main experiment: Shapley entropy attribution on Li & Cao IMC 2020 dataset.

Uses the fast FeatureMatrix backend (numpy void-view + mixed-radix packing),
~10-40x faster than the pandas reference implementation.

Results saved to results/ directory.

Usage:
    python scripts/run_experiment.py                  # Monte Carlo, 300K rows (fast, ~5 min)
    python scripts/run_experiment.py --mode exact     # Exact Shapley 18 features (~40 min)
    python scripts/run_experiment.py --nrows 0        # full 7.2M dataset
"""

import sys, os, argparse, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from src.entropy_fast import FeatureMatrix
from src.shapley_fast import (
    shapley_exact_fast, shapley_monte_carlo_fast,
    shapley_interactions_fast, check_efficiency_fast, shapley_ci_fast,
)

DATA_PATH = "data/raw/li_cao_imc2020/final_with_header.csv"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# -------------------------------------------------------------------------
# Feature definitions — actual column names from the dataset
# -------------------------------------------------------------------------

ALL_FP_FEATURES = [
    "jsFonts", "fp2_webgl", "canvastest", "hybridaudio", "agent", "ccaudio",
    "gpu", "fp2_webglvendoe", "language", "fp2_pixelratio", "browserversion",
    "osversion", "langsdetected", "timezone", "touchSupport", "browser",
    "os", "cpucores", "fp2_colordepth", "fp2_platform", "httpheaders",
    "encoding", "doNotTrack", "WebGL", "cookie",
]

# 18-feature set for the main analysis (exact-feasible)
FEATURES_MAIN = [
    "jsFonts", "fp2_webgl", "canvastest", "hybridaudio", "agent",
    "gpu", "language", "fp2_pixelratio", "browserversion", "osversion",
    "timezone", "browser", "os", "cpucores", "fp2_colordepth",
    "fp2_platform", "encoding", "doNotTrack",
]

# Top-10 for pairwise interaction analysis (2^10 = 1024 subsets, fast exact)
FEATURES_INTERACTION = [
    "jsFonts", "fp2_webgl", "canvastest", "hybridaudio",
    "agent", "gpu", "language", "fp2_pixelratio", "timezone", "os",
]

# Feature -> category mapping (for defense-targeting analysis later)
FEATURE_CATEGORY = {
    "jsFonts": "OS/fonts", "fp2_webgl": "hardware/GPU", "canvastest": "hardware/GPU",
    "hybridaudio": "hardware/audio", "ccaudio": "hardware/audio", "agent": "browser",
    "gpu": "hardware/GPU", "fp2_webglvendoe": "hardware/GPU", "language": "locale",
    "fp2_pixelratio": "hardware/screen", "browserversion": "browser",
    "osversion": "OS", "langsdetected": "locale", "timezone": "locale",
    "touchSupport": "hardware/screen", "browser": "browser", "os": "OS",
    "cpucores": "hardware/CPU", "fp2_colordepth": "hardware/screen",
    "fp2_platform": "OS", "httpheaders": "browser", "encoding": "browser",
    "doNotTrack": "browser", "WebGL": "browser", "cookie": "browser",
}


def load_matrix(path: str, nrows: int, features: list) -> tuple:
    cols_needed = list(set(features + ["touchSupport"]))
    print(f"\nLoading {'all' if nrows == 0 else f'{nrows:,}'} rows...")
    t0 = time.time()
    df = pd.read_csv(path, sep="\t", nrows=nrows if nrows > 0 else None,
                     low_memory=False, usecols=cols_needed)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].fillna("__MISSING__")
        else:
            df[col] = df[col].fillna(-9999)
    print(f"  loaded {len(df):,} rows in {time.time()-t0:.1f}s")

    fm = FeatureMatrix.from_dataframe(df, [c for c in cols_needed])
    touch = df["touchSupport"].astype(str).str.contains("true", case=False).values
    return fm, touch, len(df)


def run_feature_profile(fm: FeatureMatrix, features: list) -> pd.DataFrame:
    print("\n[1] Feature profile + marginal entropies")
    rows = []
    for f in features:
        rows.append({
            "feature": f,
            "category": FEATURE_CATEGORY.get(f, "?"),
            "n_distinct": int(fm.cardinalities[fm._idx[f]]),
            "marginal_entropy_bits": round(fm.entropy_subset([f]), 4),
        })
    profile = pd.DataFrame(rows).sort_values("marginal_entropy_bits", ascending=False)
    print(profile.to_string(index=False))
    profile.to_csv(f"{RESULTS_DIR}/feature_profile.csv", index=False)
    return profile


def run_shapley(fm: FeatureMatrix, features: list, mode: str,
                n_perm: int, with_ci: bool) -> dict:
    n = len(features)
    h_total = fm.entropy_subset(features)
    me = fm.marginal_entropies(features)
    sum_marg = sum(me.values())

    print(f"\n[2] Shapley attribution ({n} features, mode={mode})")
    print(f"  N = {fm.n_samples:,} fingerprints")
    print(f"  total entropy H(F)   = {h_total:.4f} bits")
    print(f"  sum of marginals     = {sum_marg:.4f} bits")
    print(f"  overestimate         = {sum_marg - h_total:.4f} bits "
          f"({100*(sum_marg-h_total)/h_total:.1f}%)")

    t0 = time.time()
    if mode == "exact":
        print(f"\n  exact Shapley (2^{n} = {2**n:,} subsets)...")
        phi = shapley_exact_fast(fm, features, verbose=True)
    else:
        print(f"\n  Monte Carlo Shapley ({n_perm} permutations)...")
        phi = shapley_monte_carlo_fast(fm, features, n_permutations=n_perm,
                                       verbose=True, seed=42)
    print(f"  done in {time.time()-t0:.1f}s")

    eff = check_efficiency_fast(fm, features, phi, tol=0.1)
    print(f"\n  efficiency axiom: {'PASS' if eff['passed'] else 'FAIL'} "
          f"(gap={eff['gap_bits']:.5f} bits)")

    # Optional bootstrap CI
    ci = None
    if with_ci:
        print(f"\n  computing bootstrap confidence intervals (50 resamples)...")
        ci = shapley_ci_fast(fm, features, n_bootstrap=50, n_permutations=200,
                             verbose=True, seed=7)

    rows = []
    for f in features:
        row = {
            "feature": f,
            "category": FEATURE_CATEGORY.get(f, "?"),
            "shapley_bits": round(phi[f], 4),
            "marginal_bits": round(me[f], 4),
            "shapley_minus_marginal": round(phi[f] - me[f], 4),
            "shapley_pct_of_total": round(100 * phi[f] / h_total, 2),
        }
        if ci:
            row["ci_low"] = round(ci[f][1], 4)
            row["ci_high"] = round(ci[f][2], 4)
        rows.append(row)
    tbl = pd.DataFrame(rows).sort_values("shapley_bits", ascending=False)

    print(f"\n  {'feature':<20} {'category':<16} {'Shapley':>9} {'marginal':>9} {'diff':>8} {'share':>7}")
    print(f"  {'-'*72}")
    for _, r in tbl.iterrows():
        print(f"  {r['feature']:<20} {r['category']:<16} {r['shapley_bits']:>9.3f} "
              f"{r['marginal_bits']:>9.3f} {r['shapley_minus_marginal']:>+8.3f} "
              f"{r['shapley_pct_of_total']:>6.1f}%")
    print(f"  {'-'*72}")
    print(f"  {'total':<37} {h_total:>9.3f} {sum_marg:>9.3f}")

    tbl.to_csv(f"{RESULTS_DIR}/shapley_attribution.csv", index=False)
    summary = {
        "mode": mode, "n_fingerprints": fm.n_samples, "n_features": n,
        "total_entropy_bits": h_total, "sum_marginals_bits": sum_marg,
        "overestimation_bits": sum_marg - h_total,
        "overestimation_pct": 100 * (sum_marg - h_total) / h_total,
        "efficiency_gap": eff["gap_bits"],
        "shapley_values": {f: round(phi[f], 4) for f in features},
        "marginal_entropies": {f: round(me[f], 4) for f in features},
    }
    with open(f"{RESULTS_DIR}/shapley_summary.json", "w") as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False)
    return phi


def run_interactions(fm: FeatureMatrix, features: list) -> dict:
    print(f"\n[3] Pairwise Shapley interaction values (top {len(features)} features)")
    t0 = time.time()
    ixn = shapley_interactions_fast(fm, features, verbose=True)
    print(f"  done in {time.time()-t0:.1f}s")

    rows = []
    for (fi, fj), val in ixn.items():
        rows.append({
            "feature_i": fi, "feature_j": fj, "interaction_bits": round(val, 4),
            "type": "synergy" if val > 0.05 else "redundancy" if val < -0.05 else "independent",
        })
    tbl = pd.DataFrame(rows).sort_values("interaction_bits")
    print(f"\n  8 most redundant pairs (negative = correlated/duplicated):")
    print(tbl.head(8).to_string(index=False))
    print(f"\n  5 most synergistic pairs (positive = complementary):")
    print(tbl.tail(5).to_string(index=False))
    tbl.to_csv(f"{RESULTS_DIR}/pairwise_interactions.csv", index=False)
    return ixn


def run_split(fm: FeatureMatrix, touch: np.ndarray, features: list, n_perm: int):
    print(f"\n[4] Desktop vs mobile attribution comparison")
    from src.entropy_fast import FeatureMatrix as FM
    fm_desktop = FM(fm.codes[~touch], fm.feature_names, fm.cardinalities)
    fm_mobile = FM(fm.codes[touch], fm.feature_names, fm.cardinalities)
    print(f"  desktop: {fm_desktop.n_samples:,}   mobile: {fm_mobile.n_samples:,}")

    phi_d = shapley_monte_carlo_fast(fm_desktop, features, n_permutations=n_perm,
                                     verbose=False, seed=0)
    phi_m = shapley_monte_carlo_fast(fm_mobile, features, n_permutations=n_perm,
                                     verbose=False, seed=0)
    rows = []
    for f in features:
        rows.append({"feature": f, "shapley_desktop": round(phi_d[f], 4),
                     "shapley_mobile": round(phi_m[f], 4),
                     "diff": round(phi_d[f] - phi_m[f], 4)})
    tbl = pd.DataFrame(rows).sort_values("shapley_desktop", ascending=False)
    print(f"\n  {'feature':<20} {'desktop':>9} {'mobile':>9} {'diff':>8}")
    print(f"  {'-'*48}")
    for _, r in tbl.iterrows():
        print(f"  {r['feature']:<20} {r['shapley_desktop']:>9.3f} "
              f"{r['shapley_mobile']:>9.3f} {r['diff']:>+8.3f}")
    tbl.to_csv(f"{RESULTS_DIR}/desktop_vs_mobile.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nrows", type=int, default=300_000,
                        help="Number of rows to load. 0 = all 7.2M")
    parser.add_argument("--mode", choices=["monte_carlo", "exact"], default="monte_carlo")
    parser.add_argument("--n-perm", type=int, default=1000, help="Number of Monte Carlo permutations")
    parser.add_argument("--ci", action="store_true", help="Compute bootstrap confidence intervals")
    parser.add_argument("--skip-interactions", action="store_true")
    parser.add_argument("--skip-split", action="store_true")
    parser.add_argument("--outdir", default=None, help="Output directory for results (default results/)")
    args = parser.parse_args()

    if args.outdir:
        RESULTS_DIR = args.outdir
        os.makedirs(RESULTS_DIR, exist_ok=True)

    t_start = time.time()
    fm, touch, n = load_matrix(DATA_PATH, args.nrows, ALL_FP_FEATURES)

    run_feature_profile(fm, FEATURES_MAIN)
    run_shapley(fm, FEATURES_MAIN, args.mode, args.n_perm, args.ci)
    if not args.skip_interactions:
        run_interactions(fm, FEATURES_INTERACTION)
    if not args.skip_split:
        run_split(fm, touch, FEATURES_INTERACTION, max(300, args.n_perm // 2))

    print(f"\n{'='*60}")
    print(f"  all experiments done in {(time.time()-t_start)/60:.1f} min")
    print(f"  results saved to: {os.path.abspath(RESULTS_DIR)}/")
    print(f"    feature_profile.csv / shapley_attribution.csv")
    print(f"    shapley_summary.json / pairwise_interactions.csv")
    print(f"    desktop_vs_mobile.csv")
    print('='*60)
