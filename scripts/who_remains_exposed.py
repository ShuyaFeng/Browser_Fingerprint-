"""
Who remains exposed (reviewer Item 12).

Aggregate uniqueness rates hide a skewed distribution: most users sit in large
anonymity sets while a tail is fully exposed. We characterize that tail and show it
is driven by rare hardware, the same GPU cluster the attribution and durability
analyses single out. This turns the redundancy finding into a statement about which
users a defense must protect.

We compute, on the Li and Cao corpus, each user's anonymity-set size (how many
share their full fingerprint), the CDF of those sizes, and the uniqueness rate as a
function of how rare the user's GPU renderer is.

Outputs:
  results/exposure.csv
  results/figures/fig15_exposure.pdf
"""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt

from src.entropy_fast import FeatureMatrix

DATA = "data/raw/li_cao_imc2020/final_with_header.csv"
RESULTS = "results"
NROWS = 300_000

FEATURES = [
    "jsFonts", "fp2_webgl", "canvastest", "hybridaudio", "agent",
    "gpu", "language", "fp2_pixelratio", "browserversion", "osversion",
    "timezone", "browser", "os", "cpucores", "fp2_colordepth",
    "fp2_platform", "encoding", "doNotTrack",
]


def anonymity_sizes(fm):
    """Per-row anonymity-set size = number of rows with the identical fingerprint."""
    # pack all 18 columns into one key via the radix machinery, then invert
    codes = fm.codes
    # build a single composite key by hashing rows (void view), get inverse+counts
    m = np.ascontiguousarray(codes, dtype=np.int64)
    void_dt = np.dtype((np.void, m.dtype.itemsize * m.shape[1]))
    flat = m.view(void_dt).ravel()
    _, inv, counts = np.unique(flat, return_inverse=True, return_counts=True)
    return counts[inv]  # per-row anonymity-set size


def main():
    df = pd.read_csv(DATA, sep="\t", nrows=NROWS, low_memory=False, usecols=FEATURES)
    for c in FEATURES:
        df[c] = df[c].fillna("__M__").astype(str)
    fm = FeatureMatrix.from_dataframe(df, FEATURES)
    n = len(df)

    anon = anonymity_sizes(fm)
    uniq_rate = float((anon == 1).mean())
    print(f"  N = {n:,}")
    print(f"  unique users (anonymity = 1): {uniq_rate*100:.2f}%")
    for k in [1, 2, 5, 10, 20, 50]:
        print(f"  anonymity-set size <= {k:>3}: {100*(anon <= k).mean():.1f}% of users")

    # CDF data
    ks = np.unique(anon)
    cdf = np.array([(anon <= k).mean() for k in ks])

    # rare-GPU tail: bucket users by how common their GPU renderer is
    gpu_pop = df["gpu"].map(df["gpu"].value_counts()).to_numpy()
    bins = [1, 10, 100, 1000, 10**9]
    labels = ["1-9", "10-99", "100-999", "1000+"]
    buckets = np.digitize(gpu_pop, bins[1:-1])  # 0..3
    rows = []
    print("\n  uniqueness by GPU-renderer rarity (population sharing that GPU):")
    for bidx, lab in enumerate(labels):
        mask = buckets == bidx
        if mask.sum() == 0:
            continue
        ur = float((anon[mask] == 1).mean())
        share = float(mask.mean())
        rows.append({"gpu_population": lab, "user_share_pct": round(100*share, 1),
                     "uniqueness_rate_pct": round(100*ur, 1)})
        print(f"    GPU shared by {lab:>8} users: {100*share:4.1f}% of users, "
              f"uniqueness {100*ur:5.1f}%")
    exp = pd.DataFrame(rows)
    exp.to_csv(f"{RESULTS}/exposure.csv", index=False)

    # fraction of all unique users who have a rare (pop<10) GPU
    rare = gpu_pop < 10
    share_unique_rare = float((rare & (anon == 1)).sum() / max((anon == 1).sum(), 1))
    print(f"\n  among unique users, {100*share_unique_rare:.0f}% have a rare GPU "
          f"(shared by <10 users)")

    # figure
    os.makedirs(f"{RESULTS}/figures", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.6))
    ax1.plot(ks, cdf, color="#1f77b4")
    ax1.axhline(uniq_rate, color="#d62728", ls=":", lw=1,
                label=f"unique: {uniq_rate*100:.0f}%")
    ax1.set_xscale("log")
    ax1.set_xlabel("Anonymity-set size")
    ax1.set_ylabel("Fraction of users (CDF)")
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    y = np.arange(len(exp))
    ax2.barh(y, exp["uniqueness_rate_pct"], color="#d62728")
    ax2.set_yticks(y)
    ax2.set_yticklabels([f"{l}\n({s:.0f}% users)" for l, s in
                         zip(exp["gpu_population"], exp["user_share_pct"])], fontsize=7.5)
    ax2.set_xlabel("Uniqueness rate (%)")
    ax2.invert_yaxis(); ax2.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULTS}/figures/fig15_exposure.pdf")
    plt.close()
    print("\n  saved fig15_exposure.pdf")

    with open(f"{RESULTS}/exposure_summary.json", "w") as f:
        json.dump({"uniqueness_rate": uniq_rate,
                   "share_unique_with_rare_gpu": share_unique_rare,
                   "buckets": rows}, f, indent=2)


if __name__ == "__main__":
    main()
