"""
Bootstrap confidence intervals for the pairwise interactions (reviewer Item 4).

Reports 95% bootstrap CIs for the strongest redundancies and checks how many of all
45 pairs have a CI strictly below zero, under the Chao-Shen estimator.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd, json
from src.entropy_fast import FeatureMatrix
from src.shapley_fast import shapley_interactions_fast

DATA = "data/raw/li_cao_imc2020/final_with_header.csv"
FI = ["jsFonts", "fp2_webgl", "canvastest", "hybridaudio", "agent",
      "gpu", "language", "fp2_pixelratio", "timezone", "os"]
B = 40

df = pd.read_csv(DATA, sep="\t", nrows=300_000, low_memory=False, usecols=FI)
for c in FI:
    df[c] = df[c].fillna("__M__").astype(str)
fm = FeatureMatrix.from_dataframe(df, FI)
ixn = shapley_interactions_fast(fm, FI, correction="chao_shen", verbose=False)

rng = np.random.default_rng(7)
boot = {p: [] for p in ixn}
t0 = time.time()
for b in range(B):
    fmb = fm.bootstrap(seed=int(rng.integers(1e9)))
    ib = shapley_interactions_fast(fmb, FI, correction="chao_shen", verbose=False)
    for p in ixn:
        boot[p].append(ib.get(p, ib.get(p[::-1])))

key = sorted(ixn.items(), key=lambda x: x[1])[:5]
print("5 strongest interactions, 95pct bootstrap CI (B=%d, %.0fs):" % (B, time.time() - t0))
out = {}
for p, v in key:
    lo, hi = np.quantile(boot[p], 0.025), np.quantile(boot[p], 0.975)
    out["%s x %s" % p] = [round(v, 3), round(float(lo), 3), round(float(hi), 3)]
    print("  %12s x %-12s I=%+.3f  CI=[%+.3f, %+.3f]" % (p[0], p[1], v, lo, hi))

n_ci_neg = sum(1 for p in ixn if np.quantile(boot[p], 0.975) < 0)
print("pairs whose 95pct CI upper bound < 0: %d/%d" % (n_ci_neg, len(ixn)))
out["n_ci_below_zero"] = [n_ci_neg, len(ixn)]
with open("results/interaction_ci.json", "w") as f:
    json.dump(out, f, indent=2)
