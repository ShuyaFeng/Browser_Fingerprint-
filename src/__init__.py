from .entropy import entropy_subset, entropy_all, marginal_entropies, total_entropy, calibrate_estimator
from .shapley import shapley_exact, shapley_monte_carlo, shapley_interactions, check_efficiency
from .data_loader import generate_synthetic, load_li_cao, load_amiunique, dataset_summary

# Fast backend (numpy void-view + mixed-radix packing) — use for real experiments
from .entropy_fast import FeatureMatrix
from .shapley_fast import (
    shapley_exact_fast, shapley_monte_carlo_fast,
    shapley_interactions_fast, check_efficiency_fast, shapley_ci_fast,
)
