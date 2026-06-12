"""Regenerate all paper figures from the saved results/ tables, without
recomputing any analysis. Use after style changes to the plotting code.

Figures whose inputs are not fully persisted (fig10 sensitivity right panel,
fig11 cross-dataset per-feature values, fig14 optimal defense, fig15 exposure
CDF) require re-running their analysis scripts; this driver skips them and
says so.
"""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

RESULTS = "results"

# figs 1-6 read their own CSVs
from scripts import make_figures as mf
mf.fig1_shapley_vs_marginal()
mf.fig2_interaction_heatmap()
mf.fig3_desktop_vs_mobile()
mf.fig4_category_breakdown()
mf.fig5_defense_effectiveness()
mf.fig6_point_defenses()

# fig7: entropy migration
from scripts import entropy_migration
entropy_migration.make_figure(pd.read_csv(f"{RESULTS}/entropy_migration.csv"))

# figs 8-9: re-identification
from scripts import reidentification
reidentification.make_figures(
    pd.read_csv(f"{RESULTS}/reidentification.csv"),
    pd.read_csv(f"{RESULTS}/point_reidentification.csv"),
    pd.read_csv(f"{RESULTS}/entropy_uniqueness.csv"),
)

# fig12: estimator validation
from scripts import estimator_validation
ev = json.load(open(f"{RESULTS}/estimator_validation.json"))
estimator_validation.make_figure(ev["ground_truth"], pd.DataFrame(ev["null_sweep"]))

# fig13: temporal
from scripts import temporal_analysis
temporal_analysis.make_figure(
    pd.read_csv(f"{RESULTS}/temporal_stability.csv"),
    pd.read_csv(f"{RESULTS}/linking_game.csv"),
)

print("\nSkipped (need analysis re-run): fig10 sensitivity, fig11 cross-dataset, "
      "fig14 optimal defense, fig15 exposure")
print("Done.")
