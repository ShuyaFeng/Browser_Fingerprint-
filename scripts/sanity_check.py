"""
End-to-end sanity check pipeline.

Runs on synthetic data (always available) and on real data if present.
Verifies: estimator calibration, Shapley axioms, interaction detection.

Usage:
    python scripts/sanity_check.py               # synthetic only
    python scripts/sanity_check.py --real        # also load Li & Cao data
"""

import sys, os, argparse, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from src.entropy import calibrate_estimator, marginal_entropies, total_entropy
from src.shapley import shapley_exact, shapley_monte_carlo, check_efficiency, shapley_interactions
from src.data_loader import generate_synthetic, load_li_cao, dataset_summary, LI_CAO_BROWSER_FEATURES


PASS = "  PASS"
FAIL = "  FAIL"


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def check(condition, label, detail=""):
    status = PASS if condition else FAIL
    print(f"{status}  {label}" + (f"  [{detail}]" if detail else ""))
    return condition


# ---------------------------------------------------------------------------
# 1. Estimator calibration
# ---------------------------------------------------------------------------

def run_calibration():
    section("1. Entropy Estimator Calibration")
    all_ok = True

    for n_cat, n_samp, tol in [(8, 1_000, 0.15), (50, 10_000, 0.05), (200, 50_000, 0.02)]:
        r = calibrate_estimator(n_categories=n_cat, n_samples=n_samp, n_trials=20)
        ok = r["mean_error"] < tol
        all_ok &= ok
        check(ok,
              f"k={n_cat:3d} categories, N={n_samp:6,d} samples",
              f"mean_error={r['mean_error']:.4f} bits (threshold {tol})")
    return all_ok


# ---------------------------------------------------------------------------
# 2. Shapley axioms on synthetic data
# ---------------------------------------------------------------------------

def run_axiom_checks():
    section("2. Shapley Axiom Verification (synthetic)")

    rng = np.random.default_rng(42)
    n = 8_000

    # Independent features: known ground truth
    df_indep = pd.DataFrame({
        "f1": rng.integers(0, 2, n),    # 1 bit
        "f2": rng.integers(0, 4, n),    # 2 bits
        "f3": rng.integers(0, 8, n),    # 3 bits
        "f_const": np.zeros(n, int),    # 0 bits (dummy)
    })
    feats_indep = ["f1", "f2", "f3", "f_const"]

    t0 = time.time()
    phi = shapley_exact(df_indep, feats_indep, verbose=False)
    elapsed = time.time() - t0

    all_ok = True

    # Efficiency
    eff = check_efficiency(df_indep, feats_indep, phi, tol=0.05)
    all_ok &= check(eff["passed"], "Efficiency axiom: sum(phi) == H(F)",
                    f"gap={eff['gap_bits']:.5f} bits")

    # Dummy
    all_ok &= check(abs(phi["f_const"]) < 0.05, "Dummy axiom: constant feature -> phi ≈ 0",
                    f"phi(f_const)={phi['f_const']:.4f}")

    # Independent features match marginals
    for feat, true_h in [("f1", 1.0), ("f2", 2.0), ("f3", 3.0)]:
        all_ok &= check(abs(phi[feat] - true_h) < 0.07,
                        f"Independent feature {feat}: phi ≈ {true_h} bits",
                        f"phi={phi[feat]:.4f}")

    print(f"  (Exact Shapley on {len(feats_indep)} features took {elapsed:.2f}s)")

    # Correlated features: phi should < marginal
    f1_corr = rng.integers(0, 16, n)
    df_corr = pd.DataFrame({
        "f1": f1_corr,
        "f2": f1_corr % 4,   # determined by f1
        "f3": rng.integers(0, 8, n),
    })
    phi_c = shapley_exact(df_corr, ["f1", "f2", "f3"], verbose=False)
    me_c = marginal_entropies(df_corr, ["f1", "f2", "f3"])
    gap = (me_c["f1"] + me_c["f2"]) - (phi_c["f1"] + phi_c["f2"])
    all_ok &= check(gap > 0.5, "Correlated features: Shapley < sum of marginals",
                    f"marginals={me_c['f1']+me_c['f2']:.3f}, shapley={phi_c['f1']+phi_c['f2']:.3f}")

    return all_ok


# ---------------------------------------------------------------------------
# 3. Monte Carlo convergence
# ---------------------------------------------------------------------------

def run_mc_convergence():
    section("3. Monte Carlo Shapley Convergence")
    rng = np.random.default_rng(7)
    n = 15_000
    df = pd.DataFrame({
        "f1": rng.integers(0, 4, n),
        "f2": rng.integers(0, 8, n),
        "f3": rng.integers(0, 16, n),
    })
    feats = ["f1", "f2", "f3"]

    phi_exact = shapley_exact(df, feats, verbose=False)

    all_ok = True
    for n_perm in [200, 1000]:
        phi_mc = shapley_monte_carlo(df, feats, n_permutations=n_perm,
                                      verbose=False, seed=42)
        max_err = max(abs(phi_mc[f] - phi_exact[f]) for f in feats)
        tol = 0.15 if n_perm == 200 else 0.05
        ok = max_err < tol
        all_ok &= check(ok, f"MC n_permutations={n_perm}: max error < {tol} bits",
                        f"max_error={max_err:.4f}")
    return all_ok


# ---------------------------------------------------------------------------
# 4. Interaction detection
# ---------------------------------------------------------------------------

def run_interaction_detection():
    section("4. Pairwise Interaction Detection")
    rng = np.random.default_rng(9)
    n = 12_000
    f1 = rng.integers(0, 16, n)
    df = pd.DataFrame({
        "gpu":    f1,
        "canvas": f1 % 8,    # correlated with gpu (negative interaction)
        "screen": rng.integers(0, 50, n),  # independent
    })
    feats = ["gpu", "canvas", "screen"]
    ixn = shapley_interactions(df, feats, verbose=False)

    all_ok = True
    all_ok &= check(ixn[("gpu", "canvas")] < -0.1,
                    "GPU x Canvas: negative interaction (redundancy)",
                    f"I(gpu,canvas)={ixn[('gpu','canvas')]:.4f}")
    all_ok &= check(abs(ixn[("gpu", "screen")]) < 0.15,
                    "GPU x Screen: near-zero interaction (independent)",
                    f"I(gpu,screen)={ixn[('gpu','screen')]:.4f}")
    all_ok &= check(abs(ixn[("canvas", "screen")]) < 0.15,
                    "Canvas x Screen: near-zero interaction (independent)",
                    f"I(canvas,screen)={ixn[('canvas','screen')]:.4f}")
    return all_ok


# ---------------------------------------------------------------------------
# 5. Full synthetic fingerprint pipeline
# ---------------------------------------------------------------------------

def run_full_synthetic():
    section("5. Full Synthetic Fingerprint Pipeline")

    df, gt = generate_synthetic(n_samples=20_000, seed=0)
    feats = list(df.columns)
    print(f"  Dataset: {len(df):,} samples x {len(feats)} features")

    # Dataset summary
    summ = dataset_summary(df)
    print(f"\n  Top 5 features by distinct values:")
    print(summ.head(5).to_string(index=False))

    # Marginal vs total
    me = marginal_entropies(df, feats)
    h_total = total_entropy(df, feats)
    sum_me = sum(me.values())
    print(f"\n  Total entropy H(F):         {h_total:.3f} bits")
    print(f"  Sum of marginal entropies:  {sum_me:.3f} bits")
    print(f"  Overestimation by marginals: {sum_me - h_total:.3f} bits ({100*(sum_me-h_total)/h_total:.1f}%)")

    # Shapley (Monte Carlo — full feature set)
    print(f"\n  Running Monte Carlo Shapley (n_permutations=500)...")
    t0 = time.time()
    phi = shapley_monte_carlo(df, feats, n_permutations=500, verbose=True, seed=42)
    elapsed = time.time() - t0

    eff = check_efficiency(df, feats, phi, tol=0.1)
    all_ok = check(eff["passed"], "Efficiency axiom on full synthetic data",
                   f"gap={eff['gap_bits']:.5f} bits")

    # Rank features by Shapley value
    ranked = sorted(phi.items(), key=lambda x: x[1], reverse=True)
    print(f"\n  Shapley attribution (top 5 of {len(feats)}):")
    print(f"  {'Feature':<22} {'Shapley':>10}  {'Marginal':>10}  {'Diff':>8}")
    print(f"  {'-'*54}")
    for feat, phi_val in ranked[:5]:
        marg = me[feat]
        print(f"  {feat:<22} {phi_val:>10.3f}  {marg:>10.3f}  {phi_val-marg:>+8.3f}")

    print(f"\n  (Monte Carlo took {elapsed:.1f}s)")
    print(f"\n  FINDING: Marginals overestimate by {sum_me-h_total:.2f} bits — "
          f"Shapley correctly attributes the {h_total:.2f} bits total.")
    return all_ok


# ---------------------------------------------------------------------------
# 6. Real data check (optional)
# ---------------------------------------------------------------------------

def run_real_data(data_path: str):
    section("6. Real Data Check (Li & Cao IMC 2020)")

    try:
        print(f"  Loading 50,000 sample rows from {data_path}...")
        df = load_li_cao(data_path, nrows=50_000)
        print(f"  Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")
        print(f"  Columns: {list(df.columns)[:8]} ...")

        summ = dataset_summary(df)
        print(f"\n  Top 10 features by distinct values:")
        print(summ.head(10).to_string(index=False))

        # Pick available features with reasonable cardinality
        available = [c for c in df.columns if df[c].nunique() > 1][:15]
        if len(available) < 3:
            print("  Not enough features — check dataset schema.")
            return False

        print(f"\n  Using {len(available)} features for entropy analysis")
        me = marginal_entropies(df, available)
        h_total = total_entropy(df, available)
        sum_me = sum(me.values())
        print(f"  Total H(F):              {h_total:.3f} bits")
        print(f"  Sum of marginals:        {sum_me:.3f} bits")
        print(f"  Overestimation:          {sum_me - h_total:.3f} bits")

        print(f"\n  Running Monte Carlo Shapley (200 permutations, slow but correctness check)...")
        phi = shapley_monte_carlo(df, available, n_permutations=200, verbose=True, seed=0)
        eff = check_efficiency(df, available, phi, tol=0.15)
        ok = check(eff["passed"], "Efficiency axiom on real data",
                   f"gap={eff['gap_bits']:.4f} bits")

        ranked = sorted(phi.items(), key=lambda x: x[1], reverse=True)
        print(f"\n  Top 5 features by Shapley value:")
        print(f"  {'Feature':<28} {'Shapley':>10}  {'Marginal':>10}")
        for feat, pv in ranked[:5]:
            print(f"  {feat:<28} {pv:>10.3f}  {me[feat]:>10.3f}")

        return ok

    except FileNotFoundError as e:
        print(f"  SKIP: {e}")
        return True  # Not a failure — data just not downloaded yet


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true",
                        help="Also run checks on the Li & Cao dataset if available")
    parser.add_argument("--data", default="data/raw/li_cao_imc2020/final_with_header.csv",
                        help="Path to Li & Cao CSV")
    args = parser.parse_args()

    results = []
    results.append(("Estimator calibration",  run_calibration()))
    results.append(("Shapley axioms",          run_axiom_checks()))
    results.append(("Monte Carlo convergence", run_mc_convergence()))
    results.append(("Interaction detection",   run_interaction_detection()))
    results.append(("Full synthetic pipeline", run_full_synthetic()))

    if args.real:
        results.append(("Real data check", run_real_data(args.data)))

    section("SUMMARY")
    all_passed = True
    for name, ok in results:
        all_passed &= ok
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")

    print()
    if all_passed:
        print("  All checks passed. Code is ready for real experiments.")
        print("  Next step: bash scripts/download_data.sh")
        print("             python scripts/sanity_check.py --real")
    else:
        print("  Some checks FAILED. Fix before running real experiments.")

    sys.exit(0 if all_passed else 1)
