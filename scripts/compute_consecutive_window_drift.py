# scripts/compute_consecutive_window_drift.py
"""
Compute input-space covariate shift between consecutive evaluation windows.

Unlike Section 7.1 (which measures drift between the training period and
each window), this script measures drift between window pairs, with focus
on the adjacent transitions W1-W2, W2-W3, W3-W4.

Input drift is measured using the same statistics used in Section 7.1:
    - Jensen-Shannon divergence on per-feature histograms
    - Kolmogorov-Smirnov statistic per feature
    - Number of features with significant KS test

Two families of comparisons are produced:

    Adjacent:       W1-W2, W2-W3, W3-W4
    Long-range:     W1-W3, W2-W4, W1-W4

Reads:
    data/raw/ieee_cis/train_transaction.csv

Writes:
    results/concept_drift/consecutive_window_drift.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp

from efd.data.preprocessing import (
    FeaturePolicy,
    NumericTransactionPreprocessor,
)
from efd.data.temporal import chronological_windows


DEFAULT_TRANSACTIONS = "data/raw/ieee_cis/train_transaction.csv"
DEFAULT_OUTPUT = "results/concept_drift/consecutive_window_drift.csv"
DEFAULT_TRAIN_FRACTION = 0.60
DEFAULT_N_FUTURE_WINDOWS = 4
DEFAULT_MAX_MISSING_RATE = 0.95
DEFAULT_ALPHA = 0.05
N_BINS = 20


# ============================================================
# Pairwise comparison pairs
# ============================================================

def get_comparison_pairs() -> list[tuple[int, int]]:
    """
    Return the list of window pairs to compare.

    Order:
        Adjacent first (3 pairs), then long-range (3 pairs).
    """
    return [
        (1, 2),
        (2, 3),
        (3, 4),
        (1, 3),
        (2, 4),
        (1, 4),
    ]


# ============================================================
# Drift metrics
# ============================================================

def per_feature_js_divergence(
    x_a: np.ndarray,
    x_b: np.ndarray,
    n_bins: int = N_BINS,
) -> float:
    """
    Compute JS divergence between two one-dimensional samples.
    Returns NaN if either sample is degenerate.
    """
    a = x_a[np.isfinite(x_a)]
    b = x_b[np.isfinite(x_b)]

    if len(a) < 2 or len(b) < 2:
        return np.nan

    combined = np.concatenate([a, b])
    lo, hi = np.percentile(combined, [1, 99])

    if lo == hi:
        return np.nan

    bins = np.linspace(lo, hi, n_bins)
    hist_a, _ = np.histogram(a, bins=bins, density=True)
    hist_b, _ = np.histogram(b, bins=bins, density=True)

    # Normalise to probabilities
    sum_a = hist_a.sum()
    sum_b = hist_b.sum()
    if sum_a <= 0 or sum_b <= 0:
        return np.nan

    hist_a = hist_a / sum_a
    hist_b = hist_b / sum_b

    return float(jensenshannon(hist_a, hist_b, base=2.0) ** 2)


def compare_windows(
    X_a: np.ndarray,
    X_b: np.ndarray,
    feature_names: list[str],
    alpha: float = DEFAULT_ALPHA,
) -> dict:
    """
    Compute drift metrics between two window feature matrices.
    """
    n_features = len(feature_names)

    js_values = []
    ks_p_values = []
    ks_statistics = []

    for i in range(n_features):
        col_a = X_a[:, i]
        col_b = X_b[:, i]

        js = per_feature_js_divergence(col_a, col_b)
        js_values.append(js)

        # KS test
        valid_a = col_a[np.isfinite(col_a)]
        valid_b = col_b[np.isfinite(col_b)]

        if len(valid_a) > 1 and len(valid_b) > 1 and np.std(valid_a) > 0 and np.std(valid_b) > 0:
            stat, p = ks_2samp(valid_a, valid_b)
        else:
            stat, p = 0.0, 1.0

        ks_statistics.append(stat)
        ks_p_values.append(p)

    js_values = np.array(js_values, dtype=float)
    ks_p_values = np.array(ks_p_values, dtype=float)
    ks_statistics = np.array(ks_statistics, dtype=float)

    finite_js = js_values[np.isfinite(js_values)]

    return {
        "n_features": n_features,
        "mean_js_divergence": float(np.mean(finite_js)) if len(finite_js) > 0 else np.nan,
        "median_js_divergence": float(np.median(finite_js)) if len(finite_js) > 0 else np.nan,
        "max_js_divergence": float(np.max(finite_js)) if len(finite_js) > 0 else np.nan,
        "n_features_nonfinite_js": int(np.sum(~np.isfinite(js_values))),
        "mean_ks_statistic": float(np.mean(ks_statistics)),
        "max_ks_statistic": float(np.max(ks_statistics)),
        "n_significant_features": int((ks_p_values < alpha).sum()),
        "fraction_significant": float((ks_p_values < alpha).mean()),
    }


# ============================================================
# Main
# ============================================================

def run(
    transactions_path: str,
    output_path: str,
    train_fraction: float,
    n_future_windows: int,
    max_missing_rate: float,
    alpha: float,
):

    print("=" * 70)
    print("CONSECUTIVE-WINDOW INPUT DRIFT")
    print("=" * 70)

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print("\n[1/3] Loading transaction dataset...")
    df = pd.read_csv(transactions_path, low_memory=False)
    print(f"  Loaded {len(df):,} rows")

    print("\n[2/3] Creating chronological windows...")
    train, future_windows = chronological_windows(
        df,
        time_col="TransactionDT",
        train_fraction=train_fraction,
        n_future_windows=n_future_windows,
    )
    print(f"  Training: {len(train):,}")
    for i, w in enumerate(future_windows, 1):
        print(f"  W{i}: {len(w):,}")

    # --------------------------------------------------------
    # Preprocess with training-only preprocessor
    # --------------------------------------------------------

    preprocessor = NumericTransactionPreprocessor(
        policy=FeaturePolicy(
            target="isFraud",
            temporal="TransactionDT",
            identifier="TransactionID",
            max_missing_rate=max_missing_rate,
        )
    )
    preprocessor.fit(train)
    feature_names = preprocessor.feature_names_
    print(f"  Features: {len(feature_names)}")

    X_windows = [
        preprocessor.transform(w).to_numpy(dtype=float)
        for w in future_windows
    ]

    # --------------------------------------------------------
    # Compute pairwise drift
    # --------------------------------------------------------

    print("\n[3/3] Computing drift between window pairs...")

    pairs = get_comparison_pairs()
    rows = []

    for (a, b) in pairs:

        metrics = compare_windows(
            X_a=X_windows[a - 1],
            X_b=X_windows[b - 1],
            feature_names=feature_names,
            alpha=alpha,
        )

        row = {
            "window_a": a,
            "window_b": b,
            "comparison": f"w{a}_w{b}",
            "is_adjacent": (b - a) == 1,
            **metrics,
        }
        rows.append(row)

        print(
            f"  W{a}→W{b}: "
            f"JS={metrics['mean_js_divergence']:.6f}, "
            f"KS_max={metrics['max_ks_statistic']:.4f}, "
            f"sig_features={metrics['n_significant_features']}"
        )

    results = pd.DataFrame(rows)

    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path_obj, index=False)

    print(f"\nSaved: {output_path_obj}")
    print("\nSummary:")
    print(results[[
        "comparison", "is_adjacent",
        "mean_js_divergence", "max_ks_statistic",
        "n_significant_features", "fraction_significant",
    ]].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description="Compute covariate shift between window pairs."
    )
    parser.add_argument("--transactions", default=DEFAULT_TRANSACTIONS)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--train-fraction", type=float, default=DEFAULT_TRAIN_FRACTION)
    parser.add_argument("--windows", type=int, default=DEFAULT_N_FUTURE_WINDOWS)
    parser.add_argument("--max-missing-rate", type=float, default=DEFAULT_MAX_MISSING_RATE)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    args = parser.parse_args()

    run(
        transactions_path=args.transactions,
        output_path=args.output,
        train_fraction=args.train_fraction,
        n_future_windows=args.windows,
        max_missing_rate=args.max_missing_rate,
        alpha=args.alpha,
    )


if __name__ == "__main__":
    main()