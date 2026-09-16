# scripts/aggregate_covariate_shift.py
"""
Aggregate feature-level drift results into per-window summary statistics.

Reads:
    results/concept_drift/feature_drift_detailed.csv

Writes:
    results/concept_drift/covariate_shift_summary.csv

The input file is produced by detect_concept_drift.py and contains one
row per (window, feature) pair, with columns:
    window, feature, js_divergence, ks_p_value

This script computes, for each window:
    - number of features with significant KS test (α = 0.05)
    - maximum KS statistic (if available)
    - mean and median Jensen-Shannon divergence across features
    - maximum Jensen-Shannon divergence
    - number of features with non-finite JS divergence
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


DEFAULT_INPUT = Path("results/concept_drift/feature_drift_detailed.csv")
DEFAULT_OUTPUT = Path("results/concept_drift/covariate_shift_summary.csv")
DEFAULT_ALPHA = 0.05


def _max_ks_statistic_for_feature(
    train_values: np.ndarray,
    window_values: np.ndarray,
) -> float:
    """Return the KS statistic between two samples."""
    stat, _ = ks_2samp(train_values, window_values)
    return float(stat)


def aggregate_covariate_shift(
    input_path: Path,
    output_path: Path,
    alpha: float = DEFAULT_ALPHA,
    transactions_path: str | None = None,
    train_fraction: float = 0.60,
    n_future_windows: int = 4,
) -> pd.DataFrame:
    """
    Aggregate feature-level drift results into per-window summaries.

    Two modes are supported:

    Mode A (default, no --transactions):
        Uses only the columns present in feature_drift_detailed.csv.
        Computes:
            - n_features_total
            - n_features_significant (ks_p_value < alpha)
            - mean_js_divergence (excluding non-finite)
            - median_js_divergence
            - max_js_divergence
            - n_features_nonfinite_js
        The max KS statistic column is not available in the CSV, so it is
        set to NaN unless --transactions is supplied.

    Mode B (--transactions supplied):
        Reloads the transactions CSV, reconstructs the training/window
        split, and recomputes the KS statistic per feature so that the
        maximum KS statistic can be reported per window.
    """

    print("=" * 70)
    print("COVARIATE SHIFT AGGREGATION")
    print("=" * 70)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Feature drift file not found: {input_path}\n"
            "Run detect_concept_drift.py first."
        )

    drift_df = pd.read_csv(input_path)
    print(f"\nLoaded: {input_path}")
    print(f"Rows: {len(drift_df):,}")

    required = {"window", "feature", "js_divergence", "ks_p_value"}
    missing = required - set(drift_df.columns)
    if missing:
        raise ValueError(
            f"Input file is missing required columns: {sorted(missing)}"
        )

    windows = sorted(drift_df["window"].unique())
    print(f"Windows: {windows}")

    # -----------------------------------------------------------------
    # Optional: recompute KS statistic per feature if transactions given
    # -----------------------------------------------------------------

    max_ks_by_window: dict[int, float] = {}

    if transactions_path is not None:

        print("\n[Optional] Recomputing KS statistics from transactions...")

        # Import here to avoid a hard dependency when not needed
        from efd.data.preprocessing import (
            FeaturePolicy,
            NumericTransactionPreprocessor,
        )
        from efd.data.temporal import chronological_windows

        df = pd.read_csv(transactions_path, low_memory=False)
        train, future_windows = chronological_windows(
            df,
            time_col="TransactionDT",
            train_fraction=train_fraction,
            n_future_windows=n_future_windows,
        )

        preprocessor = NumericTransactionPreprocessor(
            policy=FeaturePolicy(
                target="isFraud",
                temporal="TransactionDT",
                identifier="TransactionID",
                max_missing_rate=0.95,
            )
        )
        X_train = preprocessor.fit_transform(train)
        feature_names = preprocessor.feature_names_
        train_np = X_train.to_numpy(dtype=float)

        for window_idx, window in enumerate(future_windows, start=1):

            X_window = preprocessor.transform(window).to_numpy(dtype=float)

            max_stat = 0.0
            for col_idx, feature in enumerate(feature_names):

                train_col = train_np[:, col_idx]
                window_col = X_window[:, col_idx]

                # Skip constant columns in either sample
                if np.nanstd(train_col) == 0 or np.nanstd(window_col) == 0:
                    continue

                stat, _ = ks_2samp(train_col, window_col)
                if stat > max_stat:
                    max_stat = stat

            max_ks_by_window[window_idx] = float(max_stat)
            print(f"  Window {window_idx}: max KS = {max_stat:.4f}")

    # -----------------------------------------------------------------
    # Aggregate per window
    # -----------------------------------------------------------------

    print("\nAggregating per window...")

    rows = []

    for window in windows:

        subset = drift_df[drift_df["window"] == window]

        n_features = len(subset)

        n_significant = int(
            (subset["ks_p_value"] < alpha).sum()
        )

        js_values = subset["js_divergence"].to_numpy(dtype=float)
        finite_mask = np.isfinite(js_values)
        js_finite = js_values[finite_mask]
        n_nonfinite = int((~finite_mask).sum())

        if len(js_finite) > 0:
            mean_js = float(np.mean(js_finite))
            median_js = float(np.median(js_finite))
            max_js = float(np.max(js_finite))
        else:
            mean_js = np.nan
            median_js = np.nan
            max_js = np.nan

        max_ks = max_ks_by_window.get(window, np.nan)

        rows.append({
            "window": int(window),
            "n_features_total": n_features,
            "n_features_significant": n_significant,
            "fraction_features_significant": (
                n_significant / n_features if n_features > 0 else np.nan
            ),
            "max_ks_statistic": max_ks,
            "mean_js_divergence": mean_js,
            "median_js_divergence": median_js,
            "max_js_divergence": max_js,
            "n_features_nonfinite_js": n_nonfinite,
        })

    summary = pd.DataFrame(rows).sort_values("window").reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)

    print(f"\nSaved: {output_path}")
    print("\nSummary:")
    print(summary.to_string(index=False))

    return summary


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate feature-level drift results into a per-window summary."
        )
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
        help="Path to feature_drift_detailed.csv",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Path for the aggregated summary CSV",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="Significance threshold for KS p-values (default 0.05)",
    )
    parser.add_argument(
        "--transactions",
        default=None,
        help=(
            "Optional: path to train_transaction.csv. If supplied, "
            "max KS statistic is recomputed per window."
        ),
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.60,
    )
    parser.add_argument(
        "--windows",
        type=int,
        default=4,
    )

    args = parser.parse_args()

    aggregate_covariate_shift(
        input_path=Path(args.input),
        output_path=Path(args.output),
        alpha=args.alpha,
        transactions_path=args.transactions,
        train_fraction=args.train_fraction,
        n_future_windows=args.windows,
    )


if __name__ == "__main__":
    main()