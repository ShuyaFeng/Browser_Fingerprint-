"""
High-performance entropy estimation backend.

Replaces the pandas-groupby approach (slow on high-cardinality features)
with a numpy void-view trick that computes joint counts via sorting.

Speedup: ~40-100x over pandas groupby for high-cardinality fingerprint data.

Workflow:
  1. FeatureMatrix.from_dataframe() factorizes all columns to int32 ONCE.
  2. entropy_subset(cols) computes H(F_S) using np.unique on a void view.

All entropy values are in BITS with Miller-Madow bias correction by default.
"""

import numpy as np
import pandas as pd
from typing import Optional


_INT64_MAX = np.int64(np.iinfo(np.int64).max)


def _joint_counts_void(mat: np.ndarray) -> np.ndarray:
    """Joint frequency counts for an (N, k) int matrix.

    Uses the void-view trick: reinterpret each row as a single opaque
    scalar, then np.unique (sort-based) gives group counts in O(N log N).
    Far faster than pandas groupby for high-cardinality data.
    """
    n, k = mat.shape
    if k == 0:
        return np.array([n], dtype=np.int64)
    # Ensure C-contiguous int64 so each row is a fixed byte block
    m = np.ascontiguousarray(mat, dtype=np.int64)
    # View each row (k int64s) as one void element
    void_dt = np.dtype((np.void, m.dtype.itemsize * k))
    flat = m.view(void_dt).ravel()
    _, counts = np.unique(flat, return_counts=True)
    return counts.astype(np.int64)


def _joint_counts(mat: np.ndarray, radix: Optional[np.ndarray] = None) -> np.ndarray:
    """Joint frequency counts, auto-selecting the fastest method.

    If the product of per-column cardinalities (radix) fits in int64,
    pack all columns into a single int64 key and use 1-D np.unique
    (faster than the void view). Otherwise fall back to the void view.

    Parameters
    ----------
    mat : (N, k) int matrix of factorized codes (values in 0..card-1)
    radix : optional (k,) array of per-column cardinalities. If given,
            enables the mixed-radix packing fast path.
    """
    n, k = mat.shape
    if k == 0:
        return np.array([n], dtype=np.int64)
    if k == 1:
        _, counts = np.unique(mat[:, 0], return_counts=True)
        return counts.astype(np.int64)

    if radix is not None:
        # Check whether mixed-radix packing stays within int64
        prod = np.int64(1)
        overflow = False
        for c in radix:
            c = np.int64(max(int(c), 1))
            if prod > _INT64_MAX // c:
                overflow = True
                break
            prod *= c
        if not overflow:
            # Pack: key = sum_j code_j * (prod of cardinalities of earlier cols)
            key = np.zeros(n, dtype=np.int64)
            mult = np.int64(1)
            for j in range(k):
                key += mat[:, j].astype(np.int64) * mult
                mult *= np.int64(max(int(radix[j]), 1))
            _, counts = np.unique(key, return_counts=True)
            return counts.astype(np.int64)

    return _joint_counts_void(mat)


def _entropy_from_counts(counts: np.ndarray, correction: str = "miller_madow") -> float:
    """Entropy in bits from a vector of joint counts.

    correction:
      "plugin"      — naive maximum-likelihood plug-in (downward biased).
      "miller_madow"— plug-in + (K-1)/(2N) first-order bias correction.
      "chao_shen"   — coverage-adjusted (Good-Turing) Horvitz-Thompson
                      estimator (Chao and Shen 2003). Corrects for the mass
                      of unseen cells, so it is far more robust in the
                      undersampled/saturated regime where Miller-Madow still
                      underestimates H(F_S) for large coalitions.
      "grassberger" — Grassberger (2008) finite-sample estimator.
    """
    counts = counts[counts > 0]
    n = int(counts.sum())
    if n == 0:
        return 0.0
    if correction == "chao_shen":
        return _entropy_chao_shen(counts, n)
    if correction == "grassberger":
        return _entropy_grassberger(counts, n)
    p = counts / n
    h = float(-np.sum(p * np.log2(p)))
    if correction == "miller_madow":
        k = len(counts)
        h += (k - 1) / (2 * n * np.log2(np.e))
    return h


def _entropy_chao_shen(counts: np.ndarray, n: int) -> float:
    """Chao-Shen (2003) coverage-adjusted entropy, in bits.

    Estimates sample coverage C via the Good-Turing singleton rate, scales the
    observed probabilities by C (reserving mass for unseen cells), and applies
    a Horvitz-Thompson 1/(1-(1-p)^n) inclusion-probability correction. This is
    the standard remedy for the downward plug-in bias under heavy undersampling.
    """
    f1 = int(np.sum(counts == 1))
    if f1 == n:           # degenerate: every cell a singleton -> avoid C=0
        f1 = n - 1
    C = 1.0 - f1 / n      # estimated sample coverage
    if C <= 0:
        C = 1.0 / n
    p = counts / n
    pa = C * p            # coverage-adjusted cell probabilities
    denom = 1.0 - (1.0 - pa) ** n         # Horvitz-Thompson inclusion prob
    denom = np.where(denom <= 0, 1.0, denom)
    h_nats = float(-np.sum(pa * np.log(pa) / denom))
    return h_nats / np.log(2)


def _entropy_grassberger(counts: np.ndarray, n: int) -> float:
    """Grassberger (2008) entropy estimator, in bits."""
    from scipy.special import digamma
    # G(n_i) = psi(n_i) + 0.5*(-1)^n_i * (psi((n_i+1)/2) - psi(n_i/2))
    ni = counts.astype(np.float64)
    sign = np.where(counts % 2 == 0, 1.0, -1.0)
    G = digamma(ni) + 0.5 * sign * (digamma((ni + 1) / 2) - digamma(ni / 2))
    h_nats = np.log(n) - float(np.sum(ni * G)) / n
    return h_nats / np.log(2)


class FeatureMatrix:
    """Integer-encoded feature matrix for fast repeated entropy queries.

    Factorizes every column to a contiguous int32 code ONCE. All subsequent
    entropy_subset() calls operate on the integer matrix with no pandas
    overhead.
    """

    def __init__(self, codes: np.ndarray, feature_names: list,
                 cardinalities: Optional[np.ndarray] = None):
        self.codes = codes              # (N, n_features) int64
        self.feature_names = list(feature_names)
        self._idx = {f: i for i, f in enumerate(feature_names)}
        self.n_samples = codes.shape[0]
        if cardinalities is None:
            cardinalities = np.array(
                [int(codes[:, j].max()) + 1 if codes.shape[0] else 1
                 for j in range(codes.shape[1])], dtype=np.int64)
        self.cardinalities = cardinalities

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, features: Optional[list] = None):
        if features is None:
            features = list(df.columns)
        n = len(df)
        codes = np.empty((n, len(features)), dtype=np.int64)
        cards = np.empty(len(features), dtype=np.int64)
        for j, col in enumerate(features):
            # factorize: maps distinct values -> 0..k-1; NaN -> own bucket
            c, uniques = pd.factorize(df[col], use_na_sentinel=False)
            codes[:, j] = c
            cards[j] = len(uniques)
        return cls(codes, features, cards)

    def _col_indices(self, cols: list) -> list:
        return [self._idx[c] for c in cols]

    def entropy_subset(self, cols: list, correction: str = "miller_madow") -> float:
        """H(F_S) for the given list of feature names."""
        if len(cols) == 0:
            return 0.0
        idx = self._col_indices(cols)
        return self.entropy_by_index(idx, correction=correction)

    def entropy_by_index(self, idx: list, correction: str = "miller_madow") -> float:
        """H(F_S) using integer column indices directly (avoids name lookup)."""
        if len(idx) == 0:
            return 0.0
        sub = self.codes[:, idx]
        radix = self.cardinalities[idx]
        counts = _joint_counts(sub, radix=radix)
        return _entropy_from_counts(counts, correction=correction)

    def marginal_entropies(self, cols: Optional[list] = None,
                           correction: str = "miller_madow") -> dict:
        if cols is None:
            cols = self.feature_names
        return {c: self.entropy_subset([c], correction=correction) for c in cols}

    def subsample(self, n: int, seed: int = 0) -> "FeatureMatrix":
        """Return a new FeatureMatrix with n randomly sampled rows."""
        rng = np.random.default_rng(seed)
        if n >= self.n_samples:
            return self
        sel = rng.choice(self.n_samples, size=n, replace=False)
        return FeatureMatrix(self.codes[sel], self.feature_names, self.cardinalities)

    def bootstrap(self, seed: int) -> "FeatureMatrix":
        """Return a bootstrap resample (same size, with replacement)."""
        rng = np.random.default_rng(seed)
        sel = rng.integers(0, self.n_samples, size=self.n_samples)
        return FeatureMatrix(self.codes[sel], self.feature_names, self.cardinalities)


def extrapolate_entropy(fm: "FeatureMatrix", cols: list,
                        fractions=(0.25, 0.4, 0.55, 0.7, 0.85, 1.0),
                        n_reps: int = 3, correction: str = "chao_shen",
                        seed: int = 0) -> dict:
    """Extrapolate H(F_S) to the infinite-sample limit.

    Sub-samples the corpus at increasing fractions, estimates H at each size,
    and fits H(n) = H_inf - a/n (the leading finite-sample bias term decays
    like 1/n). The intercept H_inf is the bias-extrapolated entropy. Returns
    the per-size means plus the extrapolated asymptote and its slope.

    This corrects the marginal-overestimate ratio: the joint H(F) is the most
    undersampled quantity, so plug-in/Miller-Madow underestimate it the most,
    which inflates sum_marginals / H(F). Extrapolating H(F) upward shrinks the
    ratio to its true value.
    """
    rng = np.random.default_rng(seed)
    Ns, Hs = [], []
    for frac in fractions:
        n = int(round(frac * fm.n_samples))
        if n < 10:
            continue
        vals = []
        for r in range(n_reps):
            if n >= fm.n_samples:
                sub = fm
            else:
                sel = rng.choice(fm.n_samples, size=n, replace=False)
                sub = FeatureMatrix(fm.codes[sel], fm.feature_names,
                                    fm.cardinalities)
            vals.append(sub.entropy_subset(cols, correction=correction))
        Ns.append(n)
        Hs.append(float(np.mean(vals)))
    Ns = np.array(Ns, dtype=float)
    Hs = np.array(Hs, dtype=float)
    # Fit H = H_inf - a * (1/N)
    x = 1.0 / Ns
    A = np.vstack([np.ones_like(x), -x]).T
    coef, *_ = np.linalg.lstsq(A, Hs, rcond=None)
    h_inf, a = float(coef[0]), float(coef[1])
    return {"sizes": Ns.tolist(), "H": Hs.tolist(),
            "H_inf": h_inf, "slope_a": a,
            "H_at_full": float(Hs[-1]), "correction": correction}
