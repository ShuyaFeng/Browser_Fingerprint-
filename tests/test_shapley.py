"""
Tests for Shapley value computation.
Critical: verify axioms hold before running on real data.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest
from src.shapley import shapley_exact, shapley_monte_carlo, shapley_interactions, check_efficiency
from src.entropy import total_entropy


def make_independent_df(n=5000, seed=0):
    """Independent features with known marginal entropies: 1, 2, 3 bits."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "f1": rng.integers(0, 2, n),   # 1 bit
        "f2": rng.integers(0, 4, n),   # 2 bits
        "f3": rng.integers(0, 8, n),   # 3 bits
    })


def make_correlated_df(n=5000, seed=1):
    """f1 and f2 are correlated (f2 = f1 mod 4), f3 is independent."""
    rng = np.random.default_rng(seed)
    f1 = rng.integers(0, 16, n)
    f2 = f1 % 4          # perfectly determined by f1 given f1's value
    f3 = rng.integers(0, 8, n)
    return pd.DataFrame({"f1": f1, "f2": f2, "f3": f3})


class TestShapleyExact:
    def test_efficiency_axiom_independent(self):
        """sum(phi_i) must equal H(F) — the most important axiom."""
        df = make_independent_df()
        features = ["f1", "f2", "f3"]
        phi = shapley_exact(df, features, verbose=False)
        check = check_efficiency(df, features, phi, tol=0.05)
        assert check["passed"], (
            f"Efficiency axiom failed: H(F)={check['total_entropy_bits']:.4f}, "
            f"sum(phi)={check['sum_shapley_bits']:.4f}, gap={check['gap_bits']:.6f}"
        )

    def test_independent_features_equal_marginal(self):
        """For independent features, phi_i should equal marginal H(X_i)."""
        df = make_independent_df(n=20_000)
        features = ["f1", "f2", "f3"]
        phi = shapley_exact(df, features, verbose=False)
        assert abs(phi["f1"] - 1.0) < 0.05, f"f1: expected ~1.0 bit, got {phi['f1']:.4f}"
        assert abs(phi["f2"] - 2.0) < 0.05, f"f2: expected ~2.0 bits, got {phi['f2']:.4f}"
        assert abs(phi["f3"] - 3.0) < 0.05, f"f3: expected ~3.0 bits, got {phi['f3']:.4f}"

    def test_dummy_feature(self):
        """A constant feature (zero entropy) must have phi=0 (dummy axiom)."""
        rng = np.random.default_rng(2)
        df = pd.DataFrame({
            "real":  rng.integers(0, 16, 5000),
            "const": np.zeros(5000, dtype=int),  # always 0, zero entropy
        })
        phi = shapley_exact(df, ["real", "const"], verbose=False)
        assert abs(phi["const"]) < 0.05, \
            f"Dummy axiom: constant feature should have phi≈0, got {phi['const']:.4f}"

    def test_correlated_features_less_than_marginals(self):
        """When f1 and f2 are correlated, phi(f1)+phi(f2) < H(f1)+H(f2)."""
        df = make_correlated_df(n=20_000)
        features = ["f1", "f2", "f3"]
        phi = shapley_exact(df, features, verbose=False)

        from src.entropy import marginal_entropies
        me = marginal_entropies(df, features)

        sum_phi_12 = phi["f1"] + phi["f2"]
        sum_me_12 = me["f1"] + me["f2"]
        assert sum_phi_12 < sum_me_12 - 0.5, (
            f"Correlated case: phi(f1)+phi(f2)={sum_phi_12:.3f} should be "
            f"< sum of marginals={sum_me_12:.3f}"
        )

    def test_efficiency_axiom_correlated(self):
        df = make_correlated_df()
        features = ["f1", "f2", "f3"]
        phi = shapley_exact(df, features, verbose=False)
        check = check_efficiency(df, features, phi, tol=0.05)
        assert check["passed"], (
            f"Efficiency failed on correlated data: gap={check['gap_bits']:.6f}"
        )


class TestShapleyMonteCarlo:
    def test_converges_to_exact(self):
        """Monte Carlo Shapley should converge to exact within tolerance."""
        df = make_independent_df(n=10_000)
        features = ["f1", "f2", "f3"]
        phi_exact = shapley_exact(df, features, verbose=False)
        phi_mc = shapley_monte_carlo(df, features, n_permutations=2000,
                                      verbose=False, seed=99)
        for f in features:
            assert abs(phi_mc[f] - phi_exact[f]) < 0.1, (
                f"Monte Carlo diverges from exact for {f}: "
                f"exact={phi_exact[f]:.4f}, mc={phi_mc[f]:.4f}"
            )

    def test_efficiency_monte_carlo(self):
        df = make_independent_df()
        features = ["f1", "f2", "f3"]
        phi = shapley_monte_carlo(df, features, n_permutations=1000, verbose=False)
        check = check_efficiency(df, features, phi, tol=0.05)
        assert check["passed"], f"Monte Carlo efficiency: gap={check['gap_bits']:.6f}"


class TestShapleyInteractions:
    def test_independent_interactions_near_zero(self):
        """Independent features should have near-zero interaction values."""
        df = make_independent_df(n=10_000)
        features = ["f1", "f2", "f3"]
        ixn = shapley_interactions(df, features, verbose=False)
        for pair, val in ixn.items():
            assert abs(val) < 0.1, \
                f"Independent features {pair}: interaction should ≈ 0, got {val:.4f}"

    def test_correlated_interaction_negative(self):
        """Correlated f1,f2 should have negative interaction (redundancy)."""
        df = make_correlated_df(n=10_000)
        features = ["f1", "f2", "f3"]
        ixn = shapley_interactions(df, features, verbose=False)
        # f1 and f2 are correlated -> negative interaction (redundant)
        assert ixn[("f1", "f2")] < -0.1, \
            f"Correlated pair: expected negative interaction, got {ixn[('f1','f2')]:.4f}"
        # f3 interactions should be near zero
        assert abs(ixn[("f1", "f3")]) < 0.1
        assert abs(ixn[("f2", "f3")]) < 0.1
