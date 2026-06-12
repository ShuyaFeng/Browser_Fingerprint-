"""
Gap 1: Cross-dataset replication on a second, independent corpus.

We re-run the attribution on the FPStalker dataset (Vastel et al., S&P 2018),
collected from volunteers via a browser extension circa 2015-2017, a different
population, era, and collection method than Li and Cao (a European website,
2017-2018). We use the feature set common to both datasets and ask whether the
central structure survives: marginal overestimate, all-negative interactions, a
redundant GPU cluster, and fonts/GPU on top.

To isolate the population effect from sample size, both datasets are subsampled
to the same size and analyzed with exact Shapley over the same 14 common features.

Outputs:
  results/cross_dataset.csv
  results/figures/fig11_cross_dataset.pdf
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
from src.shapley_fast import shapley_and_interactions_fast
from scripts.figstyle import labels as feat_labels

LICAO = "data/raw/li_cao_imc2020/final_with_header.csv"
FPSTALKER = "data/raw/fpstalker/fingerprints.csv"
RESULTS = "results"
N = 15_000  # match the public FPStalker subset size

# Features present in both datasets (Li & Cao names)
COMMON = [
    "jsFonts", "fp2_webgl", "canvastest", "gpu", "fp2_webglvendoe", "agent",
    "language", "browserversion", "timezone", "browser", "os", "fp2_platform",
    "encoding", "doNotTrack",
]
GPU_CLUSTER = ["fp2_webgl", "canvastest", "gpu", "fp2_webglvendoe"]


def load_fpstalker():
    df = pd.read_csv(FPSTALKER, sep="\t", low_memory=False)
    for c in COMMON:
        df[c] = df[c].fillna("__MISSING__").astype(str)
    if len(df) > N:
        df = df.sample(N, random_state=0).reset_index(drop=True)
    return FeatureMatrix.from_dataframe(df, COMMON), len(df)


def load_licao():
    df = pd.read_csv(LICAO, sep="\t", nrows=200_000, low_memory=False,
                     usecols=COMMON)
    for c in COMMON:
        df[c] = df[c].fillna("__MISSING__").astype(str)
    df = df.sample(N, random_state=0).reset_index(drop=True)
    return FeatureMatrix.from_dataframe(df, COMMON), len(df)


def analyze(fm, name):
    joint = fm.entropy_subset(COMMON)
    marg = fm.marginal_entropies(COMMON)
    sum_marg = sum(marg.values())
    print(f"\n[{name}] exact Shapley + interactions on {len(COMMON)} features...")
    t0 = time.time()
    phi, ixn = shapley_and_interactions_fast(fm, COMMON, verbose=False)
    print(f"  done in {time.time()-t0:.0f}s")

    n_neg = sum(1 for v in ixn.values() if v < 0)
    strongest = min(ixn.items(), key=lambda x: x[1])
    ranked = sorted(phi.items(), key=lambda x: -x[1])
    gpu_marg = sum(marg[f] for f in GPU_CLUSTER)
    gpu_shap = sum(phi[f] for f in GPU_CLUSTER)

    res = {
        "dataset": name,
        "n": fm.n_samples,
        "joint_entropy": round(joint, 3),
        "sum_marginals": round(sum_marg, 3),
        "overestimate_pct": round(100 * (sum_marg - joint) / joint, 1),
        "neg_interaction_pct": round(100 * n_neg / len(ixn), 1),
        "strongest_pair": f"{strongest[0][0]}x{strongest[0][1]}",
        "strongest_pair_bits": round(strongest[1], 3),
        "top1_feature": ranked[0][0],
        "top2_feature": ranked[1][0],
        "gpu_cluster_marginal": round(gpu_marg, 3),
        "gpu_cluster_shapley": round(gpu_shap, 3),
        "gpu_cluster_redundancy_pct": round(100 * (1 - gpu_shap / gpu_marg), 1),
    }
    print(f"  joint H {joint:.2f} | sum of marginals {sum_marg:.2f} | overestimate {res['overestimate_pct']}%")
    print(f"  negative interactions {res['neg_interaction_pct']}% | strongest redundancy {res['strongest_pair']} "
          f"= {res['strongest_pair_bits']}")
    print(f"  Top Shapley: {ranked[0][0]} ({ranked[0][1]:.3f}), "
          f"{ranked[1][0]} ({ranked[1][1]:.3f})")
    print(f"  GPU cluster: sum of marginals {gpu_marg:.2f} -> Shapley {gpu_shap:.2f} "
          f"(redundancy {res['gpu_cluster_redundancy_pct']}%)")
    return res, phi, marg


def main():
    t0 = time.time()
    fm_l, n_l = load_licao()
    fm_f, n_f = load_fpstalker()
    print(f"Li&Cao: {n_l:,} | FPStalker: {n_f:,} | shared features: {len(COMMON)}")

    res_l, phi_l, marg_l = analyze(fm_l, "Li & Cao")
    res_f, phi_f, marg_f = analyze(fm_f, "FPStalker")

    tbl = pd.DataFrame([res_l, res_f])
    tbl.to_csv(f"{RESULTS}/cross_dataset.csv", index=False)

    print("\n" + "=" * 74)
    print("  Cross-dataset comparison (same 14 features, 15k samples each)")
    print("=" * 74)
    cols = ["dataset", "overestimate_pct", "neg_interaction_pct",
            "top1_feature", "strongest_pair", "gpu_cluster_redundancy_pct"]
    print(tbl[cols].to_string(index=False))

    print(f"\n  conclusions:")
    both_overest = res_l["overestimate_pct"] > 100 and res_f["overestimate_pct"] > 100
    both_neg = res_l["neg_interaction_pct"] == 100 and res_f["neg_interaction_pct"] == 100
    print(f"  - marginals overestimate heavily in both datasets: {both_overest} "
          f"({res_l['overestimate_pct']}% vs {res_f['overestimate_pct']}%)")
    print(f"  - all interactions negative in both datasets: {both_neg} "
          f"({res_l['neg_interaction_pct']}% vs {res_f['neg_interaction_pct']}%)")
    print(f"  - GPU cluster redundancy holds in both datasets: "
          f"{res_l['gpu_cluster_redundancy_pct']}% vs {res_f['gpu_cluster_redundancy_pct']}%")

    # Figure: side-by-side Shapley for the two datasets
    os.makedirs(f"{RESULTS}/figures", exist_ok=True)
    feats_sorted = sorted(COMMON, key=lambda f: -(phi_l[f] + phi_f[f]))
    y = np.arange(len(feats_sorted))
    fig, ax = plt.subplots(figsize=(8, 5))
    h = 0.38
    ax.barh(y + h/2, [phi_l[f] for f in feats_sorted], height=h,
            color="#1f77b4", label="Li & Cao (website, 2017-18)")
    ax.barh(y - h/2, [phi_f[f] for f in feats_sorted], height=h,
            color="#ff7f0e", label="FPStalker (extension, 2015-17)")
    ax.set_yticks(y)
    ax.set_yticklabels(feat_labels(feats_sorted))
    ax.invert_yaxis()
    ax.set_xlabel("Shapley value (bits)")
    ax.legend()
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULTS}/figures/fig11_cross_dataset.pdf")
    plt.close()
    print(f"\n  saved fig11_cross_dataset.pdf")
    print(f"done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
