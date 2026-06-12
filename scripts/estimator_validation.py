"""
Estimator validation (reviewer Items 1, 2, 3).

The pairwise interaction index is a fourth-order difference of entropy
estimates. Under heavy undersampling the plug-in and Miller-Madow estimators
are downward biased, and that bias grows with coalition size, which can push
interactions negative even for independent features. We therefore (a) validate
the estimator against ground truth, (b) bound the residual bias with an
independence null, and (c) re-derive the headline numbers with a
coverage-adjusted estimator.

Estimators compared:
  miller_madow : plug-in + (K-1)/(2N). Standard, but still biased when many
                 cells are unobserved (the saturated regime).
  chao_shen    : Good-Turing coverage-adjusted Horvitz-Thompson estimator,
                 the standard remedy under undersampling.

Theory anchor: v(S)=H(F_S) is submodular, so the exact interaction index is
<= 0 for every pair. The sign is therefore never in question. The question is
whether a NEGATIVE estimate reflects real redundancy or estimator bias, and how
large the true magnitude is. These controls answer both.

Key checks:
  (A) Ground truth. Planted exact duplicate (true I = -H), 60%-correlated
      feature, and independent features (true I = 0). A correct estimator must
      drive the independent pairs to ~0 and recover the duplicate near -H.
  (B) Independence null vs N. Shuffle each column independently (true I = 0 for
      every pair) and measure the residual "bias floor" at N = 50K..1M.
  (C) Corrected headline numbers. Joint entropy, marginal-overestimate ratio
      (with 1/N extrapolation), and GPU-cluster redundancy under both
      estimators.

Outputs:
  results/estimator_validation.csv / .json
  results/figures/fig12_estimator_validation.pdf
"""

import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt

from src.entropy_fast import FeatureMatrix, extrapolate_entropy
from src.shapley_fast import shapley_and_interactions_fast

DATA = "data/raw/li_cao_imc2020/final_with_header.csv"
RESULTS = "results"
SIZES = [50_000, 200_000, 1_000_000]
POOL = 1_300_000

# 10 features incl. the GPU cluster (2^10 exact interactions fast even at 1M)
FEATURES = [
    "jsFonts", "fp2_webgl", "canvastest", "gpu", "agent",
    "hybridaudio", "language", "timezone", "browser", "os",
]
GPU_PAIR = ("fp2_webgl", "gpu")

# 18-feature set for the overestimate-ratio correction
FEATURES_MAIN = [
    "jsFonts", "fp2_webgl", "canvastest", "hybridaudio", "agent",
    "gpu", "language", "fp2_pixelratio", "browserversion", "osversion",
    "timezone", "browser", "os", "cpucores", "fp2_colordepth",
    "fp2_platform", "encoding", "doNotTrack",
]
GPU_CLUSTER = ["fp2_webgl", "canvastest", "gpu"]
ESTIMATORS = ["miller_madow", "chao_shen"]


def shuffle_columns(fm, seed):
    rng = np.random.default_rng(seed)
    codes = fm.codes.copy()
    n = codes.shape[0]
    for j in range(codes.shape[1]):
        codes[:, j] = codes[rng.permutation(n), j]
    return FeatureMatrix(codes, fm.feature_names, fm.cardinalities)


def gpu_of(ixn):
    return ixn.get(GPU_PAIR, ixn.get(GPU_PAIR[::-1]))


# --------------------------------------------------------------------------
# (A) Ground-truth recovery
# --------------------------------------------------------------------------

def run_ground_truth():
    print("\n" + "=" * 78)
    print("  (A) Ground-truth recovery: independent pairs MUST go to 0")
    print("=" * 78)
    rng = np.random.default_rng(0)
    n = 200_000
    K = 80
    df = pd.DataFrame({f"ind{i}": rng.integers(0, K, n) for i in range(4)})
    df["dup_of_ind0"] = df["ind0"].to_numpy()
    partial = df["ind1"].to_numpy(copy=True)
    mask = rng.random(n) < 0.6
    partial[~mask] = rng.integers(0, K, int((~mask).sum()))
    df["partial_ind1"] = partial
    feats = list(df.columns)
    fm = FeatureMatrix.from_dataframe(df, feats)

    h_single = np.log2(K)  # true entropy of a uniform K-valued feature
    print(f"  planted uniform feature entropy H = log2({K}) = {h_single:.2f} bits")
    print(f"  exact duplicate => true interaction = -H = {-h_single:.2f}")
    print(f"  independent     => true interaction = 0\n")
    out = {"true_minus_H": -h_single}
    for corr in ESTIMATORS:
        _, ixn = shapley_and_interactions_fast(fm, feats, correction=corr,
                                               verbose=False)
        g = lambda a, b: ixn.get((a, b), ixn.get((b, a)))
        dup = g("ind0", "dup_of_ind0")
        part = g("ind1", "partial_ind1")
        indep = np.mean([g("ind0", "ind1"), g("ind1", "ind2"), g("ind2", "ind3"),
                         g("ind0", "ind2"), g("ind0", "ind3"), g("ind2", "ind3")])
        print(f"  {corr:13s}: duplicate {dup:+.3f} (true {-h_single:+.2f}) | "
              f"partial {part:+.3f} | independent {indep:+.4f} (true 0)")
        out[corr] = {"duplicate": float(dup), "partial": float(part),
                     "independent": float(indep)}
    print(f"\n  Verdict: Miller-Madow puts independent pairs at "
          f"{out['miller_madow']['independent']:+.3f} (spurious redundancy) and "
          f"attenuates the duplicate to {out['miller_madow']['duplicate']:.2f} "
          f"(true {-h_single:.2f}).")
    print(f"           Chao-Shen recovers independent {out['chao_shen']['independent']:+.4f} "
          f"and duplicate {out['chao_shen']['duplicate']:.2f}.")
    return out


# --------------------------------------------------------------------------
# (B) Independence null vs N
# --------------------------------------------------------------------------

def run_null_sweep(pool):
    print("\n" + "=" * 78)
    print("  (B) Independence bias floor vs N (real GPU pair, shuffled = true 0)")
    print("=" * 78)
    rows = []
    for corr in ESTIMATORS:
        print(f"\n  [{corr}]")
        print(f"  {'N':>9} | {'real GPU':>9} {'floor GPU':>10} | "
              f"{'real mean':>9} {'floor mean':>10}")
        print("  " + "-" * 60)
        for N in SIZES:
            df = pool.sample(N, random_state=0).reset_index(drop=True)
            fm = FeatureMatrix.from_dataframe(df, FEATURES)
            _, ixn_real = shapley_and_interactions_fast(fm, FEATURES,
                                                        correction=corr, verbose=False)
            gpu_real = gpu_of(ixn_real)
            real_mean = float(np.mean(list(ixn_real.values())))
            fl_gpu, fl_mean = [], []
            for s in range(3):
                fm_s = shuffle_columns(fm, seed=100 + s)
                _, ixn_s = shapley_and_interactions_fast(fm_s, FEATURES,
                                                         correction=corr, verbose=False)
                fl_gpu.append(gpu_of(ixn_s))
                fl_mean.append(float(np.mean(list(ixn_s.values()))))
            fgpu, fmean = float(np.mean(fl_gpu)), float(np.mean(fl_mean))
            print(f"  {N:>9,} | {gpu_real:>+9.3f} {fgpu:>+10.3f} | "
                  f"{real_mean:>+9.3f} {fmean:>+10.3f}")
            rows.append({"estimator": corr, "N": N, "real_gpu": gpu_real,
                         "floor_gpu": fgpu, "real_mean": real_mean,
                         "floor_mean": fmean})
    tbl = pd.DataFrame(rows)
    tbl.to_csv(f"{RESULTS}/estimator_validation.csv", index=False)
    return tbl


# --------------------------------------------------------------------------
# (C) Corrected headline numbers (18 features)
# --------------------------------------------------------------------------

def run_corrected_headline(pool):
    print("\n" + "=" * 78)
    print("  (C) Corrected headline numbers (18 features, N=300K)")
    print("=" * 78)
    df = pool.sample(300_000, random_state=0).reset_index(drop=True)
    fm = FeatureMatrix.from_dataframe(df, FEATURES_MAIN)
    out = {}
    for corr in ESTIMATORS:
        h = fm.entropy_subset(FEATURES_MAIN, correction=corr)
        marg = fm.marginal_entropies(FEATURES_MAIN, correction=corr)
        sm = sum(marg.values())
        out[corr] = {"H": h, "sum_marg": sm, "ratio_pct": 100 * (sm - h) / h}
    ex = extrapolate_entropy(fm, FEATURES_MAIN, correction="chao_shen")
    h_inf = ex["H_inf"]
    sm_cs = out["chao_shen"]["sum_marg"]
    ratio_corr = 100 * (sm_cs - h_inf) / h_inf
    out["extrapolated"] = {"H_inf": h_inf, "ratio_pct": ratio_corr}
    print(f"  Miller-Madow : H(F)={out['miller_madow']['H']:.2f}  "
          f"sum_marg={out['miller_madow']['sum_marg']:.2f}  "
          f"overestimate={out['miller_madow']['ratio_pct']:.0f}%")
    print(f"  Chao-Shen    : H(F)={out['chao_shen']['H']:.2f}  "
          f"sum_marg={out['chao_shen']['sum_marg']:.2f}  "
          f"overestimate={out['chao_shen']['ratio_pct']:.0f}%")
    print(f"  Extrapolated : H_inf={h_inf:.2f}  overestimate={ratio_corr:.0f}%")

    # GPU cluster redundancy under both estimators (exact Shapley on 10-set)
    print("\n  GPU-cluster redundancy (Shapley vs marginal):")
    for corr in ESTIMATORS:
        phi, _ = shapley_and_interactions_fast(fm.__class__(
            fm.codes[:, [fm._idx[f] for f in FEATURES]], FEATURES,
            fm.cardinalities[[fm._idx[f] for f in FEATURES]]),
            FEATURES, correction=corr, verbose=False)
        marg = {f: fm.entropy_subset([f], correction=corr) for f in GPU_CLUSTER}
        gpu_marg = sum(marg.values())
        gpu_shap = sum(phi[f] for f in GPU_CLUSTER)
        red = 100 * (1 - gpu_shap / gpu_marg)
        out.setdefault("gpu_cluster", {})[corr] = {
            "marginal": gpu_marg, "shapley": gpu_shap, "redundancy_pct": red}
        print(f"    {corr:13s}: marginal {gpu_marg:.2f} -> Shapley {gpu_shap:.2f} "
              f"(redundancy {red:.0f}%)")
    return out


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------

def make_figure(gt, sweep):
    os.makedirs(f"{RESULTS}/figures", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.7))

    # Panel 1: ground-truth independent baseline, MM vs CS
    labels = ["exact\nduplicate", "60%\ncorrelated", "independent\n(true 0)"]
    mm = [gt["miller_madow"]["duplicate"], gt["miller_madow"]["partial"],
          gt["miller_madow"]["independent"]]
    cs = [gt["chao_shen"]["duplicate"], gt["chao_shen"]["partial"],
          gt["chao_shen"]["independent"]]
    x = np.arange(3)
    w = 0.38
    ax1.bar(x - w / 2, mm, w, color="#9e9e9e", label="Miller-Madow")
    ax1.bar(x + w / 2, cs, w, color="#d62728", label="Chao-Shen")
    ax1.axhline(0, color="black", lw=0.7)
    ax1.axhline(gt["true_minus_H"], color="#1f77b4", lw=0.9, ls=":",
                label="true $-H$ (duplicate)")
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel("Recovered interaction (bits)")
    ax1.legend(fontsize=7.5, loc="lower left"); ax1.grid(axis="y", alpha=0.3)

    # Panel 2: independence floor vs N, MM vs CS
    EST_NAME = {"miller_madow": "Miller-Madow", "chao_shen": "Chao-Shen"}
    for corr, color, mk in [("miller_madow", "#9e9e9e", "s"),
                            ("chao_shen", "#d62728", "o")]:
        sub = sweep[sweep["estimator"] == corr].sort_values("N")
        ax2.plot(sub["N"], -sub["floor_gpu"], mk + "--", color=color,
                 label=f"{EST_NAME[corr]} floor")
        ax2.plot(sub["N"], -sub["real_gpu"], mk + "-", color=color, alpha=0.55,
                 label=f"{EST_NAME[corr]} real GPU")
    ax2.set_xscale("log")
    ax2.set_xlabel("Sample size $N$")
    ax2.set_ylabel("Redundancy magnitude (bits)")
    ax2.legend(fontsize=7); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{RESULTS}/figures/fig12_estimator_validation.pdf")
    plt.close()
    print("\n  saved fig12_estimator_validation.pdf")


if __name__ == "__main__":
    t0 = time.time()
    print("Theory: v(S)=H(F_S) submodular => interaction index <= 0 exactly.")
    print("These controls separate real redundancy from estimator bias and")
    print("re-derive the headline numbers with a coverage-adjusted estimator.\n")
    print(f"Reading {POOL:,} rows (once)...")
    tload = time.time()
    pool = pd.read_csv(DATA, sep="\t", nrows=POOL, low_memory=False,
                       usecols=list(set(FEATURES + FEATURES_MAIN)))
    for c in pool.columns:
        pool[c] = pool[c].fillna("__M__").astype(str)
    print(f"  done in {time.time()-tload:.0f}s")

    gt = run_ground_truth()
    sweep = run_null_sweep(pool)
    headline = run_corrected_headline(pool)
    make_figure(gt, sweep)

    with open(f"{RESULTS}/estimator_validation.json", "w") as f:
        json.dump({"ground_truth": gt,
                   "null_sweep": sweep.to_dict("records"),
                   "headline": headline}, f, indent=2)
    print(f"\ndone in {(time.time()-t0)/60:.1f} min")
