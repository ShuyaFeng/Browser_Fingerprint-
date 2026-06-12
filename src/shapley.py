"""
Shapley value decomposition of browser fingerprint entropy.

Fingerprinting Entropy Game
---------------------------
  Players  : features N = {X_1, ..., X_n}
  Value fn : v(S) = H(F_S)  — joint entropy of features in S (bits)
             v({}) = 0  by convention

Shapley formula for feature i:
  phi_i = sum_{S ⊆ N minus {i}} [|S|!(n-|S|-1)!/n!] * [v(S+{i}) - v(S)]

Properties guaranteed:
  Efficiency  : sum(phi_i) = H(F)   (Shapley values sum to total entropy)
  Symmetry    : identical features get equal values
  Dummy       : feature with zero marginal contribution everywhere -> phi=0
  Additivity  : linear combination of games -> linear combination of values

Two computation modes:
  exact      : enumerate all 2^n subsets  (feasible for n ≤ ~22)
  kernel     : Monte Carlo sampling of orderings  (n ≤ 200)

Reference: Shapley (1953). "A value for n-person games." Contributions to the
Theory of Games, 2, 307-317.
"""

import math
import itertools
from typing import Optional, Callable

import numpy as np
import pandas as pd
from tqdm import tqdm

from .entropy import entropy_subset


# ---------------------------------------------------------------------------
# Exact Shapley (all 2^n subsets)
# ---------------------------------------------------------------------------

def shapley_exact(
    df: pd.DataFrame,
    features: list,
    correction: str = "miller_madow",
    verbose: bool = True,
) -> dict:
    """Compute exact Shapley values for all features.

    Complexity: O(2^n * N) where n = len(features), N = len(df).
    Feasible for n ≤ 22 on a modern laptop.

    Returns
    -------
    dict: {feature -> shapley_value_bits}
    """
    n = len(features)
    if n > 25:
        raise ValueError(
            f"n={n} features is too large for exact Shapley (limit 25). "
            "Use shapley_kernel() instead."
        )

    # Pre-compute v(S) for every subset — cache keyed by frozenset
    n_subsets = 2 ** n
    if verbose:
        print(f"Pre-computing v(S) for all {n_subsets:,} subsets of {n} features...")

    cache: dict = {}
    all_subsets = list(itertools.chain.from_iterable(
        itertools.combinations(features, r) for r in range(n + 1)
    ))
    for sub in tqdm(all_subsets, disable=not verbose, desc="v(S)"):
        key = frozenset(sub)
        cache[key] = entropy_subset(df, list(sub), correction=correction)

    # Apply Shapley formula
    phi = {f: 0.0 for f in features}
    remaining = [f for f in features]

    for i, feat in enumerate(features):
        others = [f for f in features if f != feat]
        for size in range(len(others) + 1):
            for sub in itertools.combinations(others, size):
                s = frozenset(sub)
                s_with_i = s | {feat}
                weight = (
                    math.factorial(len(sub))
                    * math.factorial(n - len(sub) - 1)
                    / math.factorial(n)
                )
                marginal = cache[s_with_i] - cache[s]
                phi[feat] += weight * marginal

    return phi


# ---------------------------------------------------------------------------
# Monte Carlo Shapley (random orderings)
# ---------------------------------------------------------------------------

def shapley_monte_carlo(
    df: pd.DataFrame,
    features: list,
    n_permutations: int = 1000,
    correction: str = "miller_madow",
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """Estimate Shapley values via random permutation sampling.

    Each permutation defines a marginal contribution sequence.
    Average over permutations converges to exact Shapley values.

    Complexity: O(n_permutations * n * N).
    Suitable for any n; increase n_permutations for accuracy.

    Returns
    -------
    dict: {feature -> shapley_value_bits}
    """
    rng = np.random.default_rng(seed)
    n = len(features)
    phi = {f: 0.0 for f in features}

    for _ in tqdm(range(n_permutations), disable=not verbose, desc="permutations"):
        perm = rng.permutation(features).tolist()
        v_prev = 0.0
        current_set = []
        for feat in perm:
            current_set.append(feat)
            v_curr = entropy_subset(df, current_set, correction=correction)
            phi[feat] += (v_curr - v_prev)
            v_prev = v_curr

    for f in features:
        phi[f] /= n_permutations

    return phi


# ---------------------------------------------------------------------------
# Pairwise Shapley interaction values
# ---------------------------------------------------------------------------

def shapley_interactions(
    df: pd.DataFrame,
    features: list,
    correction: str = "miller_madow",
    verbose: bool = True,
) -> dict:
    """Compute pairwise Shapley interaction index for all feature pairs (i, j).

    Shapley interaction index (Grabisch & Roubens 1999):
      I(i,j) = sum_{S ⊆ N minus {i,j}} w(|S|,n) * [v(S+{i,j}) - v(S+{i}) - v(S+{j}) + v(S)]

    where w(s,n) = s!(n-s-2)! / (n-1)!

    Positive I(i,j) -> synergy (pair is more informative than sum of parts)
    Negative I(i,j) -> redundancy (pair is less informative due to correlation)

    Returns
    -------
    dict: {(feat_i, feat_j) -> interaction_value_bits}
    """
    n = len(features)
    if n > 20:
        raise ValueError(
            f"n={n} features is too large for exact pairwise interactions (limit 20). "
            "Consider reducing to top-k features first."
        )

    # Pre-compute v(S) for all subsets
    if verbose:
        print(f"Pre-computing v(S) for pairwise interactions ({2**n:,} subsets)...")
    cache: dict = {}
    for r in range(n + 1):
        for sub in itertools.combinations(features, r):
            cache[frozenset(sub)] = entropy_subset(df, list(sub), correction=correction)

    interactions = {}
    pairs = list(itertools.combinations(features, 2))
    for feat_i, feat_j in tqdm(pairs, disable=not verbose, desc="pairs"):
        others = [f for f in features if f not in (feat_i, feat_j)]
        val = 0.0
        for size in range(len(others) + 1):
            for sub in itertools.combinations(others, size):
                s = frozenset(sub)
                w = (
                    math.factorial(len(sub))
                    * math.factorial(n - len(sub) - 2)
                    / math.factorial(n - 1)
                )
                delta = (
                    cache[s | {feat_i, feat_j}]
                    - cache[s | {feat_i}]
                    - cache[s | {feat_j}]
                    + cache[s]
                )
                val += w * delta
        interactions[(feat_i, feat_j)] = val

    return interactions


# ---------------------------------------------------------------------------
# Bootstrap CI for Shapley values
# ---------------------------------------------------------------------------

def shapley_ci(
    df: pd.DataFrame,
    features: list,
    n_bootstrap: int = 50,
    ci_level: float = 0.95,
    method: str = "monte_carlo",
    n_permutations: int = 300,
    correction: str = "miller_madow",
    seed: int = 0,
    verbose: bool = False,
) -> dict:
    """Bootstrap confidence intervals for Shapley values.

    Returns
    -------
    dict: {feature -> (point_estimate, lower, upper)}
    """
    rng = np.random.default_rng(seed)
    n = len(df)

    # Point estimate
    if method == "exact":
        point = shapley_exact(df, features, correction=correction, verbose=verbose)
    else:
        point = shapley_monte_carlo(
            df, features, n_permutations=n_permutations,
            correction=correction, verbose=verbose,
        )

    # Bootstrap
    bootstrap_phis = {f: [] for f in features}
    for b in tqdm(range(n_bootstrap), disable=not verbose, desc="bootstrap"):
        idx = rng.integers(0, n, size=n)
        sample = df.iloc[idx]
        if method == "exact":
            phi_b = shapley_exact(sample, features, correction=correction, verbose=False)
        else:
            phi_b = shapley_monte_carlo(
                sample, features, n_permutations=n_permutations,
                correction=correction, seed=int(rng.integers(1e6)),
                verbose=False,
            )
        for f in features:
            bootstrap_phis[f].append(phi_b[f])

    alpha = (1 - ci_level) / 2
    result = {}
    for f in features:
        vals = np.array(bootstrap_phis[f])
        result[f] = (
            point[f],
            float(np.quantile(vals, alpha)),
            float(np.quantile(vals, 1 - alpha)),
        )
    return result


# ---------------------------------------------------------------------------
# Diagnostic: verify efficiency axiom
# ---------------------------------------------------------------------------

def check_efficiency(
    df: pd.DataFrame,
    features: list,
    phi: dict,
    correction: str = "miller_madow",
    tol: float = 0.01,
) -> dict:
    """Verify sum(phi_i) ≈ H(F) — the efficiency axiom.

    Returns dict with 'passed', 'total_entropy', 'sum_shapley', 'gap'.
    """
    h_total = entropy_subset(df, features, correction=correction)
    phi_sum = sum(phi[f] for f in features)
    gap = abs(h_total - phi_sum)
    return {
        "passed": gap < tol,
        "total_entropy_bits": h_total,
        "sum_shapley_bits": phi_sum,
        "gap_bits": gap,
        "tolerance": tol,
    }
