"""
Tests for entropy estimation module.
Run: python -m pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest
from src.entropy import (
    entropy_plugin, entropy_miller_madow, entropy_subset,
    marginal_entropies, total_entropy, calibrate_estimator,
)


class TestEntropyPlugin:
    def test_uniform(self):
        counts = np.array([100, 100, 100, 100])  # 4 equal categories
        h = entropy_plugin(counts)
        assert abs(h - 2.0) < 0.001, f"Expected 2.0 bits, got {h}"

    def test_deterministic(self):
        counts = np.array([1000, 0, 0])
        h = entropy_plugin(counts)
        assert h == 0.0

    def test_binary_fair(self):
        counts = np.array([500, 500])
        h = entropy_plugin(counts)
        assert abs(h - 1.0) < 0.001

    def test_zero_counts_ignored(self):
        counts = np.array([100, 0, 100])
        h = entropy_plugin(counts)
        assert abs(h - 1.0) < 0.001


class TestMillerMadow:
    def test_correction_is_positive(self):
        counts = np.array([10, 10, 10])
        h_plugin = entropy_plugin(counts)
        h_mm = entropy_miller_madow(counts)
        assert h_mm >= h_plugin, "Miller-Madow should be >= plugin estimator"

    def test_correction_shrinks_with_large_n(self):
        small = np.array([10] * 8)
        large = np.array([10000] * 8)
        corr_small = entropy_miller_madow(small) - entropy_plugin(small)
        corr_large = entropy_miller_madow(large) - entropy_plugin(large)
        assert corr_small > corr_large, "Correction should shrink as N grows"

    def test_converges_to_true_value(self):
        # Uniform over 16 categories -> ground truth = 4.0 bits
        rng = np.random.default_rng(42)
        samples = rng.integers(0, 16, size=100_000)
        df = pd.DataFrame({"x": samples})
        h = entropy_subset(df, ["x"], correction="miller_madow")
        assert abs(h - 4.0) < 0.02, f"Expected ~4.0 bits, got {h:.4f}"


class TestEntropySubset:
    def setup_method(self):
        rng = np.random.default_rng(0)
        self.df = pd.DataFrame({
            "a": rng.integers(0, 4, 5000),    # H = 2 bits (uniform 4)
            "b": rng.integers(0, 8, 5000),    # H = 3 bits (uniform 8)
            "c": rng.integers(0, 16, 5000),   # H = 4 bits (uniform 16)
        })

    def test_empty_subset(self):
        h = entropy_subset(self.df, [])
        assert h == 0.0

    def test_single_feature(self):
        h_a = entropy_subset(self.df, ["a"])
        assert abs(h_a - 2.0) < 0.05, f"Expected ~2 bits, got {h_a}"

    def test_joint_independent(self):
        # For independent features: H(A,B) ≈ H(A) + H(B)
        h_ab = entropy_subset(self.df, ["a", "b"])
        h_a = entropy_subset(self.df, ["a"])
        h_b = entropy_subset(self.df, ["b"])
        assert abs(h_ab - (h_a + h_b)) < 0.1, \
            f"H(A,B)={h_ab:.3f} should ≈ H(A)+H(B)={h_a+h_b:.3f}"

    def test_ordering_doesnt_matter(self):
        h_ab = entropy_subset(self.df, ["a", "b"])
        h_ba = entropy_subset(self.df, ["b", "a"])
        assert abs(h_ab - h_ba) < 1e-10


class TestMarginalEntropies:
    def test_returns_all_features(self):
        df = pd.DataFrame({"x": [0, 1, 0, 1], "y": [0, 0, 1, 1]})
        me = marginal_entropies(df, ["x", "y"])
        assert set(me.keys()) == {"x", "y"}

    def test_sum_overestimates_joint_when_correlated(self):
        # Perfectly correlated: x == y always
        rng = np.random.default_rng(1)
        x = rng.integers(0, 10, 2000)
        df = pd.DataFrame({"x": x, "y": x})  # y is copy of x
        me = marginal_entropies(df, ["x", "y"])
        joint = total_entropy(df, ["x", "y"])
        sum_me = sum(me.values())
        # Sum of marginals should be > joint due to perfect correlation
        assert sum_me > joint + 1.0, \
            "Correlated features: sum of marginals should >> joint entropy"


class TestCalibration:
    def test_calibration_small_n(self):
        result = calibrate_estimator(n_categories=8, n_samples=1000, n_trials=20)
        assert result["mean_error"] < 0.15, \
            f"Mean error too large for n=1000: {result['mean_error']:.4f}"

    def test_calibration_large_n(self):
        result = calibrate_estimator(n_categories=50, n_samples=50_000, n_trials=10)
        assert result["mean_error"] < 0.02, \
            f"Mean error too large for n=50000: {result['mean_error']:.4f}"
