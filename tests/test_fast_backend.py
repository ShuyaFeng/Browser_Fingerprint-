"""
Tests verifying the fast backend (entropy_fast, shapley_fast) is numerically
identical to the pandas reference implementation (entropy, shapley).

Run: python -m pytest tests/test_fast_backend.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from src.entropy import entropy_subset as slow_entropy
from src.shapley import shapley_exact as slow_shapley
from src.shapley import shapley_interactions as slow_interactions
from src.entropy_fast import FeatureMatrix, _joint_counts, _joint_counts_void
from src.shapley_fast import (shapley_exact_fast, shapley_monte_carlo_fast,
                              shapley_interactions_fast, check_efficiency_fast)


def make_df(n=8000, seed=0):
    rng = np.random.default_rng(seed)
    f1 = rng.integers(0, 200, n)      # high cardinality
    return pd.DataFrame({
        "f1": f1,
        "f2": f1 % 8,                  # correlated with f1
        "f3": rng.integers(0, 16, n),  # independent
        "f4": rng.integers(0, 4, n),   # low cardinality
        "f5": np.zeros(n, dtype=int),  # constant (dummy)
    })


class TestEntropyEquivalence:
    def test_matches_pandas_all_subsets(self):
        df = make_df()
        feats = ["f1", "f2", "f3", "f4", "f5"]
        fm = FeatureMatrix.from_dataframe(df, feats)
        from itertools import combinations
        for r in range(1, len(feats) + 1):
            for sub in combinations(feats, r):
                h_slow = slow_entropy(df, list(sub))
                h_fast = fm.entropy_subset(list(sub))
                assert abs(h_slow - h_fast) < 1e-9, \
                    f"Mismatch on {sub}: slow={h_slow}, fast={h_fast}"

    def test_packing_matches_void(self):
        """Mixed-radix packing path must equal the void-view path."""
        rng = np.random.default_rng(1)
        mat = np.column_stack([
            rng.integers(0, 5, 5000),
            rng.integers(0, 7, 5000),
            rng.integers(0, 3, 5000),
        ])
        radix = np.array([5, 7, 3])
        c_packed = np.sort(_joint_counts(mat, radix=radix))
        c_void = np.sort(_joint_counts_void(mat))
        assert np.array_equal(c_packed, c_void)


class TestShapleyEquivalence:
    def test_exact_matches_pandas(self):
        df = make_df(n=10000)
        feats = ["f1", "f2", "f3", "f4"]
        fm = FeatureMatrix.from_dataframe(df, feats)
        phi_slow = slow_shapley(df, feats, verbose=False)
        phi_fast = shapley_exact_fast(fm, feats, verbose=False)
        for f in feats:
            assert abs(phi_slow[f] - phi_fast[f]) < 1e-9, \
                f"Shapley mismatch on {f}: slow={phi_slow[f]}, fast={phi_fast[f]}"

    def test_efficiency_axiom(self):
        df = make_df()
        feats = ["f1", "f2", "f3", "f4", "f5"]
        fm = FeatureMatrix.from_dataframe(df, feats)
        phi = shapley_exact_fast(fm, feats, verbose=False)
        eff = check_efficiency_fast(fm, feats, phi, tol=0.01)
        assert eff["passed"], f"Efficiency gap={eff['gap_bits']:.2e}"

    def test_dummy_feature(self):
        df = make_df()
        feats = ["f1", "f3", "f5"]
        fm = FeatureMatrix.from_dataframe(df, feats)
        phi = shapley_exact_fast(fm, feats, verbose=False)
        assert abs(phi["f5"]) < 0.02, f"Dummy f5 should be ~0, got {phi['f5']}"

    def test_monte_carlo_converges_to_exact(self):
        df = make_df(n=10000)
        feats = ["f1", "f2", "f3", "f4"]
        fm = FeatureMatrix.from_dataframe(df, feats)
        phi_exact = shapley_exact_fast(fm, feats, verbose=False)
        phi_mc = shapley_monte_carlo_fast(fm, feats, n_permutations=2000,
                                          verbose=False, seed=1)
        for f in feats:
            assert abs(phi_exact[f] - phi_mc[f]) < 0.1, \
                f"MC diverges on {f}: exact={phi_exact[f]}, mc={phi_mc[f]}"


class TestInteractionEquivalence:
    def test_interactions_match_pandas(self):
        df = make_df(n=8000)
        feats = ["f1", "f2", "f3", "f4"]
        fm = FeatureMatrix.from_dataframe(df, feats)
        ixn_slow = slow_interactions(df, feats, verbose=False)
        ixn_fast = shapley_interactions_fast(fm, feats, verbose=False)
        for pair in ixn_slow:
            # fast uses index order; check both key orderings
            v_fast = ixn_fast.get(pair) or ixn_fast.get((pair[1], pair[0]))
            assert abs(ixn_slow[pair] - v_fast) < 1e-9, \
                f"Interaction mismatch on {pair}"
