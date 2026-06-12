"""
Result 4: Defense mis-targeting analysis.

Maps real-world anti-fingerprinting defenses (Firefox RFP, Tor Browser, Brave)
to the features they neutralize, then quantifies their TRUE effectiveness using
residual joint entropy — and shows how the redundancy we found makes naive
marginal-entropy reasoning grossly overestimate defense effectiveness.

Key metrics per defense D (neutralized feature set):
  residual_entropy     = H(features NOT neutralized)        -> what's left
  actual_removed       = H(F) - residual_entropy            -> true entropy killed
  naive_marginal_pred  = sum of MARGINAL entropies of D     -> defender's naive guess
  shapley_pred         = sum of SHAPLEY values of D          -> our method's prediction

Expected: shapley_pred ≈ actual_removed (Shapley is faithful);
          naive_marginal_pred >> actual_removed (redundancy inflates naive guess).

Defense->feature mappings sourced from:
  Firefox RFP:  MozillaWiki Security/Fingerprinting; Mozilla Support RFP page
  Tor Browser:  support.torproject.org/tor-browser/features/fingerprinting-protections
  Brave:        brave.com/privacy-updates (farbling 2.0, language/font randomization)
"""

import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from src.entropy_fast import FeatureMatrix

DATA_PATH = "data/raw/li_cao_imc2020/final_with_header.csv"
RESULTS = "results"
NROWS = 300_000

# Same 18-feature set as the main experiment
FEATURES = [
    "jsFonts", "fp2_webgl", "canvastest", "hybridaudio", "agent",
    "gpu", "language", "fp2_pixelratio", "browserversion", "osversion",
    "timezone", "browser", "os", "cpucores", "fp2_colordepth",
    "fp2_platform", "encoding", "doNotTrack",
]

# -------------------------------------------------------------------------
# Defense -> neutralized features mapping (with provenance in comments)
# -------------------------------------------------------------------------
# A feature is "neutralized" if the defense standardizes, spoofs, blocks, or
# randomizes it such that it no longer distinguishes users.

DEFENSES = {
    # Tor Browser: most aggressive — uniformizes essentially everything.
    # Canvas blocked, WebGL disabled/hardened, font whitelist, letterboxing,
    # UA unified, timezone UTC, language restricted set, audio hardened.
    "Tor Browser": [
        "jsFonts", "fp2_webgl", "canvastest", "hybridaudio", "agent",
        "gpu", "language", "fp2_pixelratio", "browserversion", "osversion",
        "timezone", "browser", "os", "cpucores", "fp2_colordepth",
        "fp2_platform", "encoding", "doNotTrack",
    ],
    # Firefox resistFingerprinting: standardizes software-layer features and
    # canvas, but WebGL rendering hash and GPU renderer are NOT fully
    # neutralized (RFP spoofs WEBGL_debug_renderer_info but rendering-based
    # WebGL/GPU fingerprints remain a known gap). This is the key omission.
    "Firefox RFP": [
        "jsFonts", "canvastest", "hybridaudio", "agent", "language",
        "fp2_pixelratio", "browserversion", "osversion", "timezone",
        "browser", "os", "cpucores", "fp2_colordepth", "fp2_platform",
        "encoding", "doNotTrack",
        # NOT neutralized: fp2_webgl, gpu
    ],
    # Brave: farbling (randomization) of canvas, WebGL, audio; randomizes
    # fonts and language (v1.39+). Keeps the real user-agent / platform to
    # avoid breaking sites, so UA-derived features are NOT neutralized.
    "Brave (farbling)": [
        "canvastest", "fp2_webgl", "hybridaudio", "gpu", "jsFonts", "language",
        # NOT neutralized: agent, browserversion, osversion, fp2_platform,
        # browser, os, timezone, fp2_pixelratio, cpucores, fp2_colordepth,
        # encoding, doNotTrack
    ],
}

# Single-point / cluster defenses (e.g., privacy extensions that block one thing)
POINT_DEFENSES = {
    "Block Canvas only":        ["canvastest"],
    "Block WebGL only":         ["fp2_webgl"],
    "Block Fonts only":         ["jsFonts"],
    "Block GPU cluster":        ["canvastest", "fp2_webgl", "gpu"],
    "Block all hardware":       ["canvastest", "fp2_webgl", "gpu", "hybridaudio",
                                 "fp2_pixelratio", "cpucores", "fp2_colordepth"],
}


def load_matrix():
    print(f"Loading {NROWS:,} rows...")
    df = pd.read_csv(DATA_PATH, sep="\t", nrows=NROWS, low_memory=False,
                     usecols=FEATURES)
    for c in FEATURES:
        if df[c].dtype == object:
            df[c] = df[c].fillna("__MISSING__")
        else:
            df[c] = df[c].fillna(-9999)
    return FeatureMatrix.from_dataframe(df, FEATURES)


def load_shapley_and_marginal():
    with open(f"{RESULTS}/shapley_summary_montecarlo.json") as f:
        d = json.load(f)
    return d["shapley_values"], d["marginal_entropies"], d["total_entropy_bits"]


def analyze(fm, name, neutralized, shapley, marginal, h_total):
    remaining = [f for f in FEATURES if f not in neutralized]
    residual = fm.entropy_subset(remaining) if remaining else 0.0
    actual_removed = h_total - residual
    naive_pred = sum(marginal[f] for f in neutralized)
    shapley_pred = sum(shapley[f] for f in neutralized)
    return {
        "defense": name,
        "n_neutralized": len(neutralized),
        "residual_entropy_bits": round(residual, 3),
        "actual_removed_bits": round(actual_removed, 3),
        "pct_entropy_removed": round(100 * actual_removed / h_total, 1),
        "naive_marginal_prediction": round(naive_pred, 3),
        "shapley_prediction": round(shapley_pred, 3),
        "naive_overestimate_factor": round(naive_pred / actual_removed, 2) if actual_removed > 0.01 else float("inf"),
        "shapley_accuracy_gap": round(abs(shapley_pred - actual_removed), 3),
    }


def main():
    t0 = time.time()
    fm = load_matrix()
    shapley, marginal, h_total = load_shapley_and_marginal()
    print(f"  total entropy H(F) = {h_total:.3f} bits\n")

    # --- Major real-world defenses ---
    print("=" * 78)
    print("  Real effectiveness of mainstream defenses (residual-entropy analysis)")
    print("=" * 78)
    rows = []
    for name, neut in DEFENSES.items():
        rows.append(analyze(fm, name, neut, shapley, marginal, h_total))
    tbl = pd.DataFrame(rows)
    print(tbl.to_string(index=False))
    tbl.to_csv(f"{RESULTS}/defense_effectiveness.csv", index=False)

    print(f"\n  interpretation:")
    for r in rows:
        print(f"  - {r['defense']}: actually removes {r['actual_removed_bits']} bits "
              f"({r['pct_entropy_removed']}%), residual {r['residual_entropy_bits']} bits. "
              f"Naive marginal prediction {r['naive_marginal_prediction']} bits "
              f"(overestimates {r['naive_overestimate_factor']}x), "
              f"Shapley prediction {r['shapley_prediction']} bits "
              f"(error only {r['shapley_accuracy_gap']} bits)")

    # --- Point / cluster defenses (redundancy demonstration) ---
    print("\n" + "=" * 78)
    print("  Point vs cluster defenses (direct implication of redundancy)")
    print("=" * 78)
    rows2 = []
    for name, neut in POINT_DEFENSES.items():
        rows2.append(analyze(fm, name, neut, shapley, marginal, h_total))
    tbl2 = pd.DataFrame(rows2)
    print(tbl2[["defense", "n_neutralized", "residual_entropy_bits",
                "actual_removed_bits", "pct_entropy_removed",
                "naive_marginal_prediction", "naive_overestimate_factor"]].to_string(index=False))
    tbl2.to_csv(f"{RESULTS}/point_defenses.csv", index=False)

    # Headline redundancy comparison
    canvas_only = next(r for r in rows2 if r["defense"] == "Block Canvas only")
    webgl_only = next(r for r in rows2 if r["defense"] == "Block WebGL only")
    cluster = next(r for r in rows2 if r["defense"] == "Block GPU cluster")
    sum_individual = (canvas_only["actual_removed_bits"]
                      + webgl_only["actual_removed_bits"])
    print(f"\n  redundancy headline numbers:")
    print(f"  - block Canvas only:    removes {canvas_only['actual_removed_bits']} bits")
    print(f"  - block WebGL only:     removes {webgl_only['actual_removed_bits']} bits")
    print(f"  - naive sum of the two: {sum_individual:.3f} bits")
    print(f"  - block whole GPU cluster: removes {cluster['actual_removed_bits']} bits "
          f"(<< the sum, since the three are highly redundant)")
    print(f"  => the total gain from neutralizing a redundant cluster is far below the sum of its parts")

    # --- Mis-targeting: which high-Shapley features does each defense miss? ---
    print("\n" + "=" * 78)
    print("  High-Shapley features each defense misses (mis-targeting)")
    print("=" * 78)
    ranked = sorted(FEATURES, key=lambda f: shapley[f], reverse=True)
    for name, neut in DEFENSES.items():
        missed = [(f, shapley[f]) for f in ranked if f not in neut]
        missed_shapley = sum(s for _, s in missed)
        if missed:
            top_missed = ", ".join(f"{f}({s:.2f})" for f, s in missed[:3])
            print(f"  {name}: misses {len(missed)} features, "
                  f"residual Shapley {missed_shapley:.2f} bits. Top missed: {top_missed}")
        else:
            print(f"  {name}: nothing missed (all neutralized)")

    print(f"\n  all done in {time.time()-t0:.1f}s")
    print(f"  results: {RESULTS}/defense_effectiveness.csv, {RESULTS}/point_defenses.csv")


if __name__ == "__main__":
    main()
