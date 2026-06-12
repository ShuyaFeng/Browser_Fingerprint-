"""
Temporal attribution and the linking game (reviewer Items 8 and 9).

A single-visit fingerprint measures uniqueness at one instant. Tracking, the real
threat, requires re-recognizing the same browser across sessions, which depends not
only on how distinctive a feature is but on how stable it is over time. A high-entropy
feature that changes between visits, such as the user-agent, is a poor long-term
tracker. We use the FPStalker dataset~\cite{vastel2018fpstalker}, where each browser
is observed repeatedly over roughly ten months, to separate the two. To make
"across sessions" meaningful rather than counting same-day reloads, every temporal
measurement below uses consecutive visits of the same browser at least seven days
apart.

(8) Stability-weighted attribution. For each feature we measure its cross-session
    stability, the rate at which its value is unchanged between visits at least a
    week apart, and contrast it with the one-shot Shapley attribution. The user-agent
    is the single most distinctive feature per visit yet drifts constantly, while the
    GPU cluster is both distinctive and stable, so durable identity rests on the
    hardware cluster.

(9) The linking game. A cross-session linker declares two fingerprints the same
    browser when a feature group matches exactly. Its true-positive rate is the
    stability of that group and its false-positive rate is its collision rate. The
    full fingerprint is a brittle tracker because some feature almost always drifts,
    while the stable GPU core links durably. A defense reduces tracking only if it
    neutralizes that stable core.

Outputs:
  results/temporal_stability.csv
  results/linking_game.csv
  results/figures/fig13_temporal.pdf
"""

import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt

from src.entropy_fast import FeatureMatrix, _joint_counts
from src.shapley_fast import shapley_and_interactions_fast
from scripts.figstyle import label as feat_label

DATA = "data/raw/fpstalker/fingerprints.csv"
RESULTS = "results"
CORR = "chao_shen"
MIN_GAP_DAYS = 7  # consecutive visits must be at least a week apart

# browserversion is empty in FPStalker, so we exclude it. 11 features remain.
FEATURES = [
    "jsFonts", "fp2_webgl", "canvastest", "gpu", "fp2_webglvendoe",
    "agent", "timezone", "language", "os", "resolution", "fp2_platform",
]
GPU_CLUSTER = ["fp2_webgl", "canvastest", "gpu", "fp2_webglvendoe"]
CATEGORY = {
    "jsFonts": "fonts", "fp2_webgl": "GPU", "canvastest": "GPU", "gpu": "GPU",
    "fp2_webglvendoe": "GPU", "agent": "browser", "timezone": "locale",
    "language": "locale", "os": "OS", "resolution": "screen", "fp2_platform": "OS",
}
CATCOLOR = {"GPU": "#d62728", "fonts": "#2ca02c", "browser": "#1f77b4",
            "locale": "#9467bd", "OS": "#ff7f0e", "screen": "#8c564b"}

# Trackers built from feature groups, for the linking game
TRACKERS = {
    "Full fingerprint": FEATURES,
    "GPU core": GPU_CLUSTER,
    "User-agent": ["agent"],
    "Fonts": ["jsFonts"],
    "Locale": ["timezone", "language"],
}


def load():
    df = pd.read_csv(DATA, sep="\t", low_memory=False,
                     usecols=["browser_id", "creation_date"] + FEATURES)
    for c in FEATURES:
        df[c] = df[c].fillna("__M__").astype(str)
    df["creation_date"] = pd.to_datetime(df["creation_date"], errors="coerce")
    df = df.dropna(subset=["creation_date"]).sort_values(
        ["browser_id", "creation_date"]).reset_index(drop=True)
    return df


def crosssession_pairs(df):
    """Consecutive same-browser visit pairs at least MIN_GAP_DAYS apart."""
    bid = df["browser_id"].to_numpy()
    dt = df["creation_date"].to_numpy()
    same = bid[1:] == bid[:-1]
    idx = np.nonzero(same)[0]
    gap = (dt[idx + 1] - dt[idx]) / np.timedelta64(1, "D")
    keep = gap >= MIN_GAP_DAYS
    return idx[keep], idx[keep] + 1


# --------------------------------------------------------------------------
# (8) Stability and durable attribution
# --------------------------------------------------------------------------

def feature_stability(df, i_idx, j_idx):
    s = {}
    for f in FEATURES:
        v = df[f].to_numpy()
        s[f] = float((v[i_idx] == v[j_idx]).mean())
    return s


def persistent_identity(df):
    agg = df.groupby("browser_id")[FEATURES].agg(
        lambda x: x.value_counts().index[0])
    return agg.reset_index(drop=True)


def run_attribution(df, i_idx, j_idx):
    print("\n" + "=" * 74)
    print("  (8) Stability-weighted temporal attribution")
    print("=" * 74)
    print(f"  cross-session visit pairs (>= {MIN_GAP_DAYS}d apart): {len(i_idx):,}")
    stab = feature_stability(df, i_idx, j_idx)

    fm_visit = FeatureMatrix.from_dataframe(df, FEATURES)
    phi_oneshot, _ = shapley_and_interactions_fast(fm_visit, FEATURES,
                                                   correction=CORR, verbose=False)
    persist = persistent_identity(df)
    fm_dur = FeatureMatrix.from_dataframe(persist, FEATURES)
    phi_dur, _ = shapley_and_interactions_fast(fm_dur, FEATURES,
                                              correction=CORR, verbose=False)

    rows = []
    for f in FEATURES:
        rows.append({"feature": f, "category": CATEGORY[f],
                     "stability": round(stab[f], 4),
                     "shapley_oneshot": round(phi_oneshot[f], 4),
                     "shapley_durable": round(phi_dur[f], 4)})
    tbl = pd.DataFrame(rows).sort_values("shapley_oneshot", ascending=False)
    tbl.to_csv(f"{RESULTS}/temporal_stability.csv", index=False)

    print(f"  {'feature':<16}{'cat':<8}{'stability':>10}{'1-shot phi':>12}{'durable phi':>13}")
    print("  " + "-" * 59)
    for _, r in tbl.iterrows():
        print(f"  {r['feature']:<16}{r['category']:<8}{r['stability']:>10.3f}"
              f"{r['shapley_oneshot']:>12.3f}{r['shapley_durable']:>13.3f}")

    gpu_os = 100 * sum(phi_oneshot[f] for f in GPU_CLUSTER) / sum(phi_oneshot.values())
    gpu_dur = 100 * sum(phi_dur[f] for f in GPU_CLUSTER) / sum(phi_dur.values())
    print(f"\n  user-agent: one-shot Shapley {phi_oneshot['agent']:.2f} bits "
          f"(highest) but stability only {stab['agent']:.2f}")
    print(f"  GPU cluster mean stability {np.mean([stab[f] for f in GPU_CLUSTER]):.2f}")
    print(f"  GPU-cluster attribution share: one-shot {gpu_os:.0f}% -> durable {gpu_dur:.0f}%")
    return tbl, stab, {"agent_oneshot": float(phi_oneshot["agent"]),
                       "agent_stability": float(stab["agent"]),
                       "gpu_stability": float(np.mean([stab[f] for f in GPU_CLUSTER])),
                       "gpu_share_oneshot": gpu_os, "gpu_share_durable": gpu_dur}


# --------------------------------------------------------------------------
# (9) The linking game
# --------------------------------------------------------------------------

def linkability(df, feats, i_idx, j_idx, neg):
    if not feats:
        return 0.0, 0.0
    code = df[feats].astype(str).agg("\x1f".join, axis=1).to_numpy()
    tpr = float((code[i_idx] == code[j_idx]).mean())
    a, b = neg
    fpr = float((code[a] == code[b]).mean())
    return tpr, fpr


def run_linking(df, i_idx, j_idx):
    print("\n" + "=" * 74)
    print("  (9) The linking game: which feature group tracks across sessions")
    print("=" * 74)
    rng = np.random.default_rng(0)
    bid = df["browser_id"].to_numpy()
    n = len(df)
    M = 1_000_000
    a = rng.integers(0, n, M); b = rng.integers(0, n, M)
    keep = bid[a] != bid[b]
    neg = (a[keep], b[keep])
    print(f"  positive (cross-session same-browser) pairs: {len(i_idx):,} | "
          f"negative pairs: {keep.sum():,}\n")

    rows = []
    for name, feats in TRACKERS.items():
        tpr, fpr = linkability(df, feats, i_idx, j_idx, neg)
        # single-visit uniqueness of this group (one-shot identifying power)
        fm = FeatureMatrix.from_dataframe(df, feats)
        counts = _joint_counts(fm.codes, radix=fm.cardinalities)
        uniq = float((counts == 1).sum() / counts.sum())
        rows.append({"tracker": name, "n_features": len(feats),
                     "single_visit_uniqueness": round(uniq, 4),
                     "link_tpr": round(tpr, 4), "link_fpr": round(fpr, 4),
                     "link_margin": round(tpr - fpr, 4)})
        print(f"  {name:<18} uniq={uniq:6.3f}  TPR={tpr:6.3f}  FPR={fpr:7.4f}  "
              f"margin={tpr-fpr:6.3f}")
    tbl = pd.DataFrame(rows)
    tbl.to_csv(f"{RESULTS}/linking_game.csv", index=False)

    full = next(r for r in rows if r["tracker"] == "Full fingerprint")
    gpu = next(r for r in rows if r["tracker"] == "GPU core")
    ua = next(r for r in rows if r["tracker"] == "User-agent")
    print(f"\n  The full fingerprint is brittle (TPR {full['link_tpr']:.2f}): some feature"
          f" almost always drifts.")
    print(f"  The GPU core links {gpu['link_tpr']/max(full['link_tpr'],1e-9):.1f}x better"
          f" (TPR {gpu['link_tpr']:.2f}, margin {gpu['link_margin']:.2f}).")
    print(f"  The user-agent is unique per visit (uniq {ua['single_visit_uniqueness']:.2f})"
          f" but a poor tracker (TPR {ua['link_tpr']:.2f}).")
    print(f"  -> Stopping cross-session tracking requires neutralizing the GPU core,"
          f" not the high-entropy but volatile user-agent.")
    return tbl, {"full_tpr": full["link_tpr"], "gpu_tpr": gpu["link_tpr"],
                 "gpu_margin": gpu["link_margin"], "ua_tpr": ua["link_tpr"],
                 "ua_uniq": ua["single_visit_uniqueness"]}


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------

def make_figure(stab_tbl, link_tbl):
    os.makedirs(f"{RESULTS}/figures", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.8))

    # manual annotation offsets (points) to avoid label collisions
    OFFSETS = {"gpu": (3, 5), "os": (3, -8), "fp2_webglvendoe": (-12, -11),
               "timezone": (-8, 6), "language": (3, -9), "resolution": (-20, -11)}
    for _, r in stab_tbl.iterrows():
        c = CATCOLOR[r["category"]]
        ax1.scatter(r["stability"], r["shapley_oneshot"], s=48, color=c, zorder=3)
        dx, dy = OFFSETS.get(r["feature"], (3, 2))
        ax1.annotate(feat_label(r["feature"]),
                     (r["stability"], r["shapley_oneshot"]),
                     fontsize=6.3, xytext=(dx, dy), textcoords="offset points")
    ax1.set_xlabel("Cross-session stability")
    ax1.set_ylabel("One-shot Shapley value (bits)")
    ax1.grid(alpha=0.3)

    names = list(link_tbl["tracker"])
    y = np.arange(len(names))
    ax2.barh(y + 0.2, link_tbl["link_margin"], height=0.38, color="#d62728",
             label="Cross-session link margin (TPR $-$ FPR)")
    ax2.barh(y - 0.2, link_tbl["single_visit_uniqueness"], height=0.38,
             color="#9e9e9e", label="Single-visit uniqueness")
    ax2.set_yticks(y); ax2.set_yticklabels(names, fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlabel("Rate")
    ax2.legend(fontsize=7, loc="lower right")
    ax2.grid(axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{RESULTS}/figures/fig13_temporal.pdf")
    plt.close()
    print("\n  saved fig13_temporal.pdf")


if __name__ == "__main__":
    t0 = time.time()
    df = load()
    i_idx, j_idx = crosssession_pairs(df)
    print(f"FPStalker: {len(df):,} fingerprints, {df['browser_id'].nunique():,} browsers, "
          f"{df['creation_date'].min().date()} to {df['creation_date'].max().date()}")
    stab_tbl, stab, attr_stats = run_attribution(df, i_idx, j_idx)
    link_tbl, link_extra = run_linking(df, i_idx, j_idx)
    make_figure(stab_tbl, link_tbl)
    with open(f"{RESULTS}/temporal_summary.json", "w") as f:
        json.dump({"attribution": attr_stats, "linking": link_extra}, f, indent=2)
    print(f"\ndone in {(time.time()-t0)/60:.1f} min")
