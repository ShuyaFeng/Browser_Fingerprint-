"""
Data loading and preprocessing for fingerprint datasets.

Supported datasets:
  - Li & Cao IMC 2020 (Zenodo dynamics dataset)
  - AmIUnique (Laperdrix S&P 2016) — when received from authors
  - Synthetic (generated for testing and methodology validation)
"""

import os
import numpy as np
import pandas as pd
from typing import Optional


# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------

# Full feature set from Li & Cao IMC 2020, Table 1
LI_CAO_FEATURES = [
    # HTTP Headers
    "user_agent", "browser", "os", "device",
    "accept", "encoding", "language", "timezone", "http_header_list",
    # Browser Features
    "plugins", "cookie_support", "webgl_support",
    "local_storage_support", "add_behavior_support", "open_database_support",
    # OS Features
    "language_list", "font_list", "canvas_images",
    # Hardware Features
    "gpu_vendor", "gpu_renderer", "gpu_type",
    "cpu_cores", "audio_card_info", "screen_resolution",
    "color_depth", "cpu_class", "pixel_ratio",
    # IP Features (optional — exclude for browser-only analysis)
    "ip_city", "ip_region", "ip_country",
    # Consistency Features
    "lang_consistency", "resolution_consistency",
    "os_consistency", "browser_consistency", "gpu_images",
]

# Subset excluding IP (for browser fingerprint analysis without location)
LI_CAO_BROWSER_FEATURES = [f for f in LI_CAO_FEATURES
                            if f not in ("ip_city", "ip_region", "ip_country")]

# High-entropy hardware features (hypothesis: these dominate Shapley values)
HARDWARE_FEATURES = [
    "gpu_vendor", "gpu_renderer", "gpu_type",
    "cpu_cores", "audio_card_info", "screen_resolution",
    "color_depth", "cpu_class", "pixel_ratio",
]

# AmIUnique feature set (Laperdrix S&P 2016, 17 features)
AMIUNIQUE_FEATURES = [
    "user_agent", "accept_header", "encoding", "language",
    "plugins_list", "platform", "cookies_enabled", "dnt",
    "timezone_offset", "screen_resolution", "canvas_hash",
    "webgl_hash", "fonts_list", "local_storage", "session_storage",
    "cpu_cores", "ad_block",
]


# ---------------------------------------------------------------------------
# Li & Cao loader
# ---------------------------------------------------------------------------

def load_li_cao(
    path: str,
    features: Optional[list] = None,
    nrows: Optional[int] = None,
    drop_ip: bool = True,
) -> pd.DataFrame:
    """Load the Li & Cao IMC 2020 dynamics dataset.

    Parameters
    ----------
    path : path to final_with_header.csv (tab-separated)
    features : list of column names to keep (None = all)
    nrows : load only first N rows (for development/testing)
    drop_ip : if True, drop IP-based columns

    Returns
    -------
    DataFrame with one row per dynamics record
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Run: bash scripts/download_data.sh"
        )

    df = pd.read_csv(path, sep="\t", nrows=nrows, low_memory=False)

    # Normalize column names to lowercase with underscores
    df.columns = [c.lower().replace(" ", "_").replace("-", "_") for c in df.columns]

    if drop_ip:
        ip_cols = [c for c in df.columns if c.startswith("ip_")]
        df = df.drop(columns=ip_cols, errors="ignore")

    if features is not None:
        available = [f for f in features if f in df.columns]
        missing = [f for f in features if f not in df.columns]
        if missing:
            print(f"Warning: {len(missing)} requested features not in dataset: {missing}")
        df = df[available]

    return df


def load_li_cao_sample(
    path: str,
    n: int = 50_000,
    seed: int = 42,
    **kwargs,
) -> pd.DataFrame:
    """Load a random sample of n rows (for fast iteration during development)."""
    # Read full file to get total row count efficiently
    with open(path) as f:
        header = f.readline()
        total = sum(1 for _ in f)
    rng = np.random.default_rng(seed)
    skip = sorted(rng.choice(total, size=max(0, total - n), replace=False) + 1)
    df = pd.read_csv(path, sep="\t", skiprows=skip, low_memory=False, **kwargs)
    df.columns = [c.lower().replace(" ", "_").replace("-", "_") for c in df.columns]
    return df


# ---------------------------------------------------------------------------
# AmIUnique loader
# ---------------------------------------------------------------------------

def load_amiunique(path: str, features: Optional[list] = None) -> pd.DataFrame:
    """Load AmIUnique dataset (CSV format from Laperdrix et al.)."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"AmIUnique dataset not found: {path}\n"
            "Request from pierre.laperdrix@univ-lille.fr"
        )
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.lower().replace(" ", "_").replace("-", "_") for c in df.columns]
    if features:
        df = df[[f for f in features if f in df.columns]]
    return df


# ---------------------------------------------------------------------------
# Synthetic data generator (for testing and calibration)
# ---------------------------------------------------------------------------

def generate_synthetic(
    n_samples: int = 10_000,
    feature_spec: Optional[dict] = None,
    seed: int = 0,
) -> tuple[pd.DataFrame, dict]:
    """Generate a synthetic fingerprint dataset with known ground truth.

    Parameters
    ----------
    n_samples : number of fingerprints to generate
    feature_spec : dict mapping feature_name -> n_categories
                   If None, uses a default spec with known entropy structure.
    seed : random seed

    Returns
    -------
    (df, ground_truth) where ground_truth contains:
      - marginal_entropies : {feature -> true H(X_i)}
      - total_entropy_independent : sum of marginal entropies (if features were independent)
      - feature_spec : the spec used
    """
    if feature_spec is None:
        # Default: mix of low, medium, and high entropy features
        # Includes two correlated features to test interaction detection
        feature_spec = {
            "gpu_renderer":    500,   # high entropy — ~9 bits
            "font_list":       300,   # high entropy — ~8.2 bits
            "canvas_hash":     200,   # high entropy — ~7.6 bits
            "screen_resolution": 50,  # medium — ~5.6 bits
            "language":         30,   # medium — ~4.9 bits
            "timezone":         38,   # medium — ~5.2 bits
            "browser":          10,   # low — ~3.3 bits
            "os":                8,   # low — ~3.0 bits
            "cpu_cores":         6,   # low — ~2.6 bits
            "color_depth":       4,   # very low — ~2 bits
        }

    rng = np.random.default_rng(seed)
    data = {}
    marginals = {}

    for feat, n_cat in feature_spec.items():
        # Uniform distribution -> entropy = log2(n_cat)
        data[feat] = rng.integers(0, n_cat, size=n_samples)
        marginals[feat] = float(np.log2(n_cat))

    df = pd.DataFrame(data)

    # Introduce correlation between gpu_renderer and canvas_hash:
    # In 40% of cases, canvas_hash is deterministically derived from gpu_renderer.
    # This simulates the GPU×Canvas interaction (hardware rendering consistency).
    if "gpu_renderer" in df.columns and "canvas_hash" in df.columns:
        n_cat_canvas = feature_spec["canvas_hash"]
        n_cat_gpu = feature_spec["gpu_renderer"]
        mask = rng.random(n_samples) < 0.4
        df.loc[mask, "canvas_hash"] = df.loc[mask, "gpu_renderer"] % n_cat_canvas

    ground_truth = {
        "marginal_entropies": marginals,
        "total_entropy_if_independent": sum(marginals.values()),
        "feature_spec": feature_spec,
        "n_samples": n_samples,
        "has_gpu_canvas_correlation": True,
    }

    return df, ground_truth


# ---------------------------------------------------------------------------
# Dataset statistics summary
# ---------------------------------------------------------------------------

def dataset_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Print a summary table: feature, n_distinct, n_unique, missing_pct."""
    rows = []
    for col in df.columns:
        vc = df[col].value_counts(dropna=False)
        n_distinct = len(vc)
        n_unique = int((vc == 1).sum())
        missing = int(df[col].isna().sum())
        rows.append({
            "feature": col,
            "n_distinct": n_distinct,
            "n_unique_values": n_unique,
            "pct_unique": round(100 * n_unique / len(df), 1),
            "missing_pct": round(100 * missing / len(df), 1),
        })
    return pd.DataFrame(rows).sort_values("n_distinct", ascending=False)
