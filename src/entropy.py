"""
Entropy estimation for browser fingerprint attribution.

All functions operate on pandas DataFrames where rows = browser instances
and columns = fingerprint features.

Bias correction: Miller-Madow correction is applied by default.
  H_MM = H_plugin + (k - 1) / (2 * N)
where k = number of distinct values observed, N = sample size.
This corrects the systematic underestimation of entropy in finite samples.

Reference:
  Miller (1955). "Note on the bias of information estimates."
  Information Theory in Psychology, 95-100.
"""

import numpy as np
import pandas as pd
from itertools import combinations
from typing import Union, Optional


# ---------------------------------------------------------------------------
# Core entropy estimator
# ---------------------------------------------------------------------------

def entropy_plugin(counts: np.ndarray) -> float:
    """Raw plug-in (MLE) entropy in bits from a frequency array."""
    counts = counts[counts > 0]
    n = counts.sum()
    if n == 0:
        return 0.0
    p = counts / n
    return float(-np.sum(p * np.log2(p)))


def entropy_miller_madow(counts: np.ndarray) -> float:
    """Miller-Madow bias-corrected entropy in bits.

    Correction term: (k - 1) / (2 * N * ln 2)  [in nats, convert to bits]
    Equivalently in bits: (k - 1) / (2 * N * log2(e))
    """
    counts = counts[counts > 0]
    n = counts.sum()
    if n == 0:
        return 0.0
    k = len(counts)
    h_plugin = entropy_plugin(counts)
    correction = (k - 1) / (2 * n * np.log2(np.e))
    return float(h_plugin + correction)


def _joint_counts(df: pd.DataFrame, features: list) -> np.ndarray:
    """Return frequency counts for the joint distribution of given features."""
    if len(features) == 0:
        return np.array([len(df)])
    subset = df[features].copy()
    # Convert each column to string to handle mixed types uniformly
    for col in features:
        subset[col] = subset[col].astype(str)
    counts = subset.groupby(features, observed=True).size().values
    return counts


def entropy_subset(
    df: pd.DataFrame,
    features: list,
    correction: str = "miller_madow",
) -> float:
    """Compute H(F_S) — joint entropy of a feature subset S.

    Parameters
    ----------
    df : DataFrame of shape (N, n_features)
    features : list of column names forming the subset S
    correction : "miller_madow" (default) | "none"

    Returns
    -------
    float : entropy in bits
    """
    if len(features) == 0:
        return 0.0
    counts = _joint_counts(df, features)
    if correction == "miller_madow":
        return entropy_miller_madow(counts)
    return entropy_plugin(counts)


def entropy_all(df: pd.DataFrame, features: list, **kwargs) -> float:
    """Convenience: entropy of the full fingerprint H(F)."""
    return entropy_subset(df, features, **kwargs)


# ---------------------------------------------------------------------------
# Bootstrap confidence interval
# ---------------------------------------------------------------------------

def entropy_ci(
    df: pd.DataFrame,
    features: list,
    n_bootstrap: int = 200,
    ci_level: float = 0.95,
    correction: str = "miller_madow",
    rng: Optional[np.random.Generator] = None,
) -> tuple[float, float, float]:
    """Bootstrap 95% CI for H(F_S).

    Returns
    -------
    (point_estimate, lower_bound, upper_bound)
    """
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(df)
    point = entropy_subset(df, features, correction=correction)
    bootstrap_vals = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        sample = df.iloc[idx]
        bootstrap_vals.append(entropy_subset(sample, features, correction=correction))
    alpha = (1 - ci_level) / 2
    lo = float(np.quantile(bootstrap_vals, alpha))
    hi = float(np.quantile(bootstrap_vals, 1 - alpha))
    return point, lo, hi


# ---------------------------------------------------------------------------
# Per-feature marginal entropy (for comparison with Shapley)
# ---------------------------------------------------------------------------

def marginal_entropies(
    df: pd.DataFrame,
    features: list,
    correction: str = "miller_madow",
) -> dict:
    """Compute marginal entropy H(X_i) for each feature independently.

    This is what Eckersley (PETS 2010) reported. Summing these overestimates
    total entropy when features are correlated.

    Returns dict: {feature_name -> entropy_bits}
    """
    return {f: entropy_subset(df, [f], correction=correction) for f in features}


def total_entropy(
    df: pd.DataFrame,
    features: list,
    correction: str = "miller_madow",
) -> float:
    """Compute total joint entropy H(F) = H(X_1, ..., X_n)."""
    return entropy_subset(df, features, correction=correction)


def sum_of_marginals(
    df: pd.DataFrame,
    features: list,
    correction: str = "miller_madow",
) -> float:
    """Sum of marginal entropies — what naive decomposition gives."""
    return sum(marginal_entropies(df, features, correction=correction).values())


# ---------------------------------------------------------------------------
# Calibration check
# ---------------------------------------------------------------------------

def calibrate_estimator(
    n_categories: int = 10,
    n_samples: int = 5000,
    n_trials: int = 50,
    correction: str = "miller_madow",
) -> dict:
    """Verify estimator accuracy on a synthetic uniform distribution.

    Ground truth: H(Uniform(k)) = log2(k) bits.
    Returns dict with mean_error and max_error across trials.
    """
    rng = np.random.default_rng(0)
    true_h = np.log2(n_categories)
    errors = []
    for _ in range(n_trials):
        samples = rng.integers(0, n_categories, size=n_samples)
        df = pd.DataFrame({"x": samples})
        est = entropy_subset(df, ["x"], correction=correction)
        errors.append(abs(est - true_h))
    return {
        "true_entropy": true_h,
        "mean_error": float(np.mean(errors)),
        "max_error": float(np.max(errors)),
        "n_categories": n_categories,
        "n_samples": n_samples,
    }
