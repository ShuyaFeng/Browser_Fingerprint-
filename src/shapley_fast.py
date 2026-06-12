"""
Fast Shapley value decomposition using the FeatureMatrix backend.

Same Shapley game as src/shapley.py (v(S) = H(F_S)), but operates on the
integer-encoded FeatureMatrix with mixed-radix packing, giving 10-40x speedup.

Modes:
  shapley_exact_fast       — all 2^n subsets (feasible n ≤ ~18)
  shapley_monte_carlo_fast — random permutations (any n; primary method)
  shapley_interactions_fast — pairwise interaction index (n ≤ ~16)
"""

import math
import itertools
from typing import Optional

import numpy as np
from tqdm import tqdm

from .entropy_fast import FeatureMatrix


# ---------------------------------------------------------------------------
# Exact Shapley + interactions from a single shared subset cache
# ---------------------------------------------------------------------------

def shapley_and_interactions_fast(
    fm: FeatureMatrix,
    features: list,
    correction: str = "miller_madow",
    verbose: bool = False,
) -> tuple:
    """Compute exact Shapley values AND pairwise interactions from ONE cache.

    Building the 2^n subset-entropy cache is the bottleneck; computing both
    quantities from a single cache avoids doing it twice. Returns (phi, ixn).
    """
    n = len(features)
    if n > 20:
        raise ValueError(f"n={n} too large for exact (limit 20).")
    idx = [fm._idx[f] for f in features]

    cache = np.empty(2 ** n, dtype=np.float64)
    cache[0] = 0.0
    for mask in tqdm(range(1, 2 ** n), disable=not verbose, desc="v(S)"):
        cols = [idx[b] for b in range(n) if (mask >> b) & 1]
        cache[mask] = fm.entropy_by_index(cols, correction=correction)

    fact = [math.factorial(i) for i in range(n + 1)]

    phi = {}
    for b, feat in enumerate(features):
        bit = 1 << b
        s = 0.0
        for mask in range(2 ** n):
            if mask & bit:
                continue
            sz = bin(mask).count("1")
            w = fact[sz] * fact[n - sz - 1] / fact[n]
            s += w * (cache[mask | bit] - cache[mask])
        phi[feat] = s

    ixn = {}
    for bi, bj in itertools.combinations(range(n), 2):
        biti, bitj = 1 << bi, 1 << bj
        val = 0.0
        for mask in range(2 ** n):
            if (mask & biti) or (mask & bitj):
                continue
            sz = bin(mask).count("1")
            w = fact[sz] * fact[n - sz - 2] / fact[n - 1]
            val += w * (cache[mask | biti | bitj] - cache[mask | biti]
                        - cache[mask | bitj] + cache[mask])
        ixn[(features[bi], features[bj])] = val

    return phi, ixn


# ---------------------------------------------------------------------------
# Exact Shapley with subset caching
# ---------------------------------------------------------------------------

def shapley_exact_fast(
    fm: FeatureMatrix,
    features: list,
    correction: str = "miller_madow",
    verbose: bool = True,
) -> dict:
    """Exact Shapley values. Pre-caches v(S) for all 2^n subsets."""
    n = len(features)
    if n > 20:
        raise ValueError(f"n={n} too large for exact (limit 20). Use monte_carlo.")

    idx = [fm._idx[f] for f in features]

    # Pre-compute v(S) for every subset, keyed by bitmask over local indices
    cache = np.empty(2 ** n, dtype=np.float64)
    cache[0] = 0.0
    masks = range(1, 2 ** n)
    for mask in tqdm(masks, disable=not verbose, desc="v(S)"):
        cols = [idx[b] for b in range(n) if (mask >> b) & 1]
        cache[mask] = fm.entropy_by_index(cols, correction=correction)

    # Shapley formula using cached values
    phi = {f: 0.0 for f in features}
    factorials = [math.factorial(i) for i in range(n + 1)]
    for b, feat in enumerate(features):
        bit = 1 << b
        for mask in range(2 ** n):
            if mask & bit:
                continue
            s_size = bin(mask).count("1")
            weight = factorials[s_size] * factorials[n - s_size - 1] / factorials[n]
            phi[feat] += weight * (cache[mask | bit] - cache[mask])

    return phi


# ---------------------------------------------------------------------------
# Monte Carlo Shapley (primary method) — vectorized permutation sampling
# ---------------------------------------------------------------------------

def shapley_monte_carlo_fast(
    fm: FeatureMatrix,
    features: list,
    n_permutations: int = 1000,
    correction: str = "miller_madow",
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """Estimate Shapley values via random permutation sampling.

    For each permutation, adds features one at a time and records the
    marginal entropy gain. Averages over permutations.
    """
    rng = np.random.default_rng(seed)
    n = len(features)
    idx = [fm._idx[f] for f in features]
    phi_idx = np.zeros(n, dtype=np.float64)

    for _ in tqdm(range(n_permutations), disable=not verbose, desc="permutations"):
        perm = rng.permutation(n)
        v_prev = 0.0
        cols = []
        for p in perm:
            cols.append(idx[p])
            v_curr = fm.entropy_by_index(cols, correction=correction)
            phi_idx[p] += (v_curr - v_prev)
            v_prev = v_curr

    phi_idx /= n_permutations
    return {features[b]: float(phi_idx[b]) for b in range(n)}


# ---------------------------------------------------------------------------
# Pairwise interactions
# ---------------------------------------------------------------------------

def shapley_interactions_fast(
    fm: FeatureMatrix,
    features: list,
    correction: str = "miller_madow",
    verbose: bool = True,
) -> dict:
    """Pairwise Shapley interaction index for all feature pairs."""
    n = len(features)
    if n > 18:
        raise ValueError(f"n={n} too large for exact interactions (limit 18).")

    idx = [fm._idx[f] for f in features]

    # Cache all subset values
    cache = np.empty(2 ** n, dtype=np.float64)
    cache[0] = 0.0
    for mask in tqdm(range(1, 2 ** n), disable=not verbose, desc="v(S)"):
        cols = [idx[b] for b in range(n) if (mask >> b) & 1]
        cache[mask] = fm.entropy_by_index(cols, correction=correction)

    factorials = [math.factorial(i) for i in range(n + 1)]
    interactions = {}
    for bi, bj in tqdm(list(itertools.combinations(range(n), 2)),
                       disable=not verbose, desc="pairs"):
        biti, bitj = 1 << bi, 1 << bj
        val = 0.0
        for mask in range(2 ** n):
            if (mask & biti) or (mask & bitj):
                continue
            s_size = bin(mask).count("1")
            w = factorials[s_size] * factorials[n - s_size - 2] / factorials[n - 1]
            delta = (cache[mask | biti | bitj] - cache[mask | biti]
                     - cache[mask | bitj] + cache[mask])
            val += w * delta
        interactions[(features[bi], features[bj])] = val

    return interactions


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def check_efficiency_fast(fm: FeatureMatrix, features: list, phi: dict,
                          correction: str = "miller_madow", tol: float = 0.01) -> dict:
    h_total = fm.entropy_subset(features, correction=correction)
    phi_sum = sum(phi[f] for f in features)
    gap = abs(h_total - phi_sum)
    return {
        "passed": gap < tol,
        "total_entropy_bits": h_total,
        "sum_shapley_bits": phi_sum,
        "gap_bits": gap,
        "tolerance": tol,
    }


def shapley_ci_fast(
    fm: FeatureMatrix,
    features: list,
    n_bootstrap: int = 50,
    ci_level: float = 0.95,
    n_permutations: int = 300,
    correction: str = "miller_madow",
    seed: int = 0,
    verbose: bool = False,
) -> dict:
    """Bootstrap confidence intervals via Monte Carlo Shapley on resamples."""
    rng = np.random.default_rng(seed)
    point = shapley_monte_carlo_fast(fm, features, n_permutations=n_permutations,
                                     correction=correction, verbose=False)
    boot = {f: [] for f in features}
    for _ in tqdm(range(n_bootstrap), disable=not verbose, desc="bootstrap"):
        fm_b = fm.bootstrap(seed=int(rng.integers(1e9)))
        phi_b = shapley_monte_carlo_fast(fm_b, features, n_permutations=n_permutations,
                                         correction=correction,
                                         seed=int(rng.integers(1e9)), verbose=False)
        for f in features:
            boot[f].append(phi_b[f])
    alpha = (1 - ci_level) / 2
    return {f: (point[f], float(np.quantile(boot[f], alpha)),
               float(np.quantile(boot[f], 1 - alpha))) for f in features}
