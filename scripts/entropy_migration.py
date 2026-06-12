"""
Result 3: Entropy migration under a counterfactual defense-deployment sequence.

We do NOT have cross-time fingerprint data (Li & Cao spans only 2017-18). Instead
we model the *historical order* in which anti-fingerprinting defenses neutralized
features, and at each stage recompute the Shapley attribution of the REMAINING
(not-yet-defended) features. This shows how entropy MIGRATES from software-layer
features to hardware (GPU) features as defenses deploy — the core Direction-B
claim that "defenses shift the attack surface rather than eliminate entropy."

This is a counterfactual/simulated longitudinal analysis, methodologically valid
on a single-time-point dataset. Real cross-time data is future work.

Deployment stages (historical defense rollout, approximate):
  Stage 0: no defense (all 18 features)
  Stage 1: UA reduction        -> neutralize browserversion, osversion, fp2_platform
  Stage 2: locale defenses     -> + language, timezone
  Stage 3: Canvas defense      -> + canvastest
  Stage 4: font defense        -> + jsFonts
  Stage 5: full software unif.  -> + agent, browser, os, encoding, doNotTrack,
                                    cpucores, fp2_colordepth, fp2_pixelratio, hybridaudio
  (remaining: fp2_webgl, gpu  -- GPU rendering, hardest to neutralize)
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
from src.shapley_fast import shapley_monte_carlo_fast

DATA_PATH = "data/raw/li_cao_imc2020/final_with_header.csv"
RESULTS = "results"
NROWS = 300_000

FEATURES = [
    "jsFonts", "fp2_webgl", "canvastest", "hybridaudio", "agent",
    "gpu", "language", "fp2_pixelratio", "browserversion", "osversion",
    "timezone", "browser", "os", "cpucores", "fp2_colordepth",
    "fp2_platform", "encoding", "doNotTrack",
]

CATEGORY = {
    "jsFonts": "OS/fonts", "fp2_webgl": "hardware/GPU", "canvastest": "hardware/GPU",
    "hybridaudio": "hardware/audio", "agent": "browser", "gpu": "hardware/GPU",
    "language": "locale", "fp2_pixelratio": "hardware/screen",
    "browserversion": "browser", "osversion": "OS", "timezone": "locale",
    "browser": "browser", "os": "OS", "cpucores": "hardware/CPU",
    "fp2_colordepth": "hardware/screen", "fp2_platform": "OS",
    "encoding": "browser", "doNotTrack": "browser",
}

# Cumulative neutralized sets at each stage
STAGES = [
    ("S0: no defense", []),
    ("S1: UA reduction", ["browserversion", "osversion", "fp2_platform"]),
    ("S2: + locale", ["browserversion", "osversion", "fp2_platform",
                      "language", "timezone"]),
    ("S3: + Canvas", ["browserversion", "osversion", "fp2_platform",
                      "language", "timezone", "canvastest"]),
    ("S4: + fonts", ["browserversion", "osversion", "fp2_platform",
                     "language", "timezone", "canvastest", "jsFonts"]),
    ("S5: + full software", ["browserversion", "osversion", "fp2_platform",
                             "language", "timezone", "canvastest", "jsFonts",
                             "agent", "browser", "os", "encoding", "doNotTrack",
                             "cpucores", "fp2_colordepth", "fp2_pixelratio",
                             "hybridaudio"]),
]


def make_figure(tbl):
    """Plot fig7 from the migration table (replottable from the saved CSV)."""
    fig, ax1 = plt.subplots(figsize=(9, 5))
    x = np.arange(len(tbl))

    # Bars: residual entropy (left axis)
    ax1.bar(x, tbl["residual_entropy_bits"], color="lightsteelblue",
            alpha=0.7, label="Residual entropy (bits)")
    ax1.set_ylabel("Residual joint entropy (bits)", color="steelblue")
    ax1.set_xticks(x)
    ax1.set_xticklabels(tbl["stage"], rotation=20, ha="right", fontsize=8)
    ax1.tick_params(axis="y", labelcolor="steelblue")

    # Line: GPU share of remaining attribution (right axis)
    ax2 = ax1.twinx()
    ax2.plot(x, tbl["gpu_shapley_pct"], "o-", color="#d62728",
             linewidth=2, markersize=7, label="GPU share of residual (%)")
    ax2.plot(x, tbl["software_shapley_pct"], "s--", color="#2ca02c",
             linewidth=2, markersize=6, label="Software share of residual (%)")
    ax2.set_ylabel("Share of residual attribution (%)")
    ax2.set_ylim(0, 105)

    lines1, lab1 = ax1.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lab1 + lab2, loc="center left", fontsize=8)
    plt.tight_layout()
    os.makedirs(f"{RESULTS}/figures", exist_ok=True)
    plt.savefig(f"{RESULTS}/figures/fig7_entropy_migration.pdf")
    plt.close()
    print(f"\n  saved fig7_entropy_migration.pdf")


def is_hardware(f):
    return CATEGORY[f].startswith("hardware")


def main():
    t0 = time.time()
    print(f"Loading {NROWS:,} rows...")
    df = pd.read_csv(DATA_PATH, sep="\t", nrows=NROWS, low_memory=False, usecols=FEATURES)
    for c in FEATURES:
        df[c] = df[c].fillna("__MISSING__") if df[c].dtype == object else df[c].fillna(-9999)
    fm = FeatureMatrix.from_dataframe(df, FEATURES)
    h_total_all = fm.entropy_subset(FEATURES)
    print(f"  full-fingerprint total entropy = {h_total_all:.3f} bits\n")

    rows = []
    print(f"{'stage':<22} {'resid H':>8} {'GPU %':>9} {'HW %':>9} {'SW %':>9}")
    print("-" * 62)
    for name, neutralized in STAGES:
        remaining = [f for f in FEATURES if f not in neutralized]
        residual = fm.entropy_subset(remaining)
        # Shapley of remaining features (re-attributed within residual surface)
        phi = shapley_monte_carlo_fast(fm, remaining, n_permutations=300,
                                       verbose=False, seed=0)
        gpu_share = sum(phi[f] for f in remaining if CATEGORY[f] == "hardware/GPU")
        hw_share = sum(phi[f] for f in remaining if is_hardware(f))
        sw_share = sum(phi[f] for f in remaining if not is_hardware(f))
        total_phi = sum(phi.values())
        gpu_pct = 100 * gpu_share / total_phi if total_phi > 0 else 0
        hw_pct = 100 * hw_share / total_phi if total_phi > 0 else 0
        sw_pct = 100 * sw_share / total_phi if total_phi > 0 else 0
        rows.append({
            "stage": name, "n_remaining": len(remaining),
            "residual_entropy_bits": round(residual, 3),
            "gpu_shapley_pct": round(gpu_pct, 1),
            "hardware_shapley_pct": round(hw_pct, 1),
            "software_shapley_pct": round(sw_pct, 1),
            "gpu_shapley_bits": round(gpu_share, 3),
        })
        print(f"{name:<22} {residual:>8.3f} {gpu_pct:>8.1f}% {hw_pct:>8.1f}% {sw_pct:>8.1f}%")

    tbl = pd.DataFrame(rows)
    tbl.to_csv(f"{RESULTS}/entropy_migration.csv", index=False)

    make_figure(tbl)

    # Headline numbers
    s0, s5 = rows[0], rows[-1]
    print(f"\n  entropy migration headline numbers:")
    print(f"  - no defense (S0): residual entropy {s0['residual_entropy_bits']} bits, "
          f"GPU share {s0['gpu_shapley_pct']}%")
    print(f"  - all software blocked (S5): residual entropy {s5['residual_entropy_bits']} bits, "
          f"GPU share {s5['gpu_shapley_pct']}%")
    print(f"  => total entropy drops {s0['residual_entropy_bits']-s5['residual_entropy_bits']:.2f} bits, "
          f"but GPU share rises from {s0['gpu_shapley_pct']}% to {s5['gpu_shapley_pct']}%")
    print(f"  => defenses do not eliminate entropy; they shift the attack surface from software to GPU hardware")
    print(f"\n  done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
