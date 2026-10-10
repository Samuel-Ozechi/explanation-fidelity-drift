# scripts/run_cumulative_retraining.py
"""
Cumulative retraining experiment.

Trains a separate model at each window using ALL data available up to
and including that window:

    cumulative_w1 = train + W1
    cumulative_w2 = train + W1 + W2
    cumulative_w3 = train + W1 + W2 + W3
    cumulative_w4 = train + W1 + W2 + W3 + W4

Each cumulative model is evaluated on all four future windows for
fidelity (mean relative score change at k=5) and detection (F1, PR-AUC).

Purpose: test whether the earlier finding that retraining reduces
fidelity is attributable to the smaller training sample (59k vs 354k)
or is intrinsic to adaptation under drift.

Reads:
    data/raw/ieee_cis/train_transaction.csv

Writes:
    results/cumulative_retraining/
        cumulative_fidelity_observation.csv
        cumulative_fidelity_summary.csv
        cumulative_detection_summary.csv
        cumulative_comparison.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from efd.data.preprocessing import (
    FeaturePolicy,
    NumericTransactionPreprocessor,
)
from efd.data.temporal import chronological_windows
from efd.models.isolation_forest import IsolationForestDetector
from efd.explanations.exiffi import ExIFFI
from efd.evaluation.explanation_fidelity import evaluate_fidelity_matrix
from sklearn.metrics import f1_score, average_precision_score


DEFAULT_TRAIN_FRACTION = 0.60
DEFAULT_N_ESTIMATORS = 300
DEFAULT_RANDOM_STATE = 42
DEFAULT_SAMPLE_SIZE = 100
DEFAULT_MAX_MISSING_RATE = 0.95
DEFAULT_K_VALUES = [1, 3, 5, 10, 20]
PRIMARY_K = 5


# ============================================================
# Helpers
# ============================================================

def build_cumulative_train_set(
    train: pd.DataFrame,
    future_windows: list[pd.DataFrame],
    upto_window: int,
) -> pd.DataFrame:
    """Concatenate train + W1..W{upto_window}."""
    parts = [train] + future_windows[:upto_window]
    return pd.concat(parts, ignore_index=True)


def evaluate_detection(
    model: IsolationForestDetector,
    X_window: pd.DataFrame,
    y_window: np.ndarray,
) -> dict:
    scores = model.anomaly_score(X_window)
    predictions = model.predict_anomaly(X_window)
    return {
        "n": int(len(y_window)),
        "fraud": int(y_window.sum()),
        "f1": float(f1_score(y_window, predictions, zero_division=0)),
        "pr_auc": float(average_precision_score(y_window, scores)),
    }


# ============================================================
# Main experiment
# ============================================================

def run(
    transactions_path: str,
    output_dir: str,
    train_fraction: float,
    n_estimators: int,
    random_state: int,
    sample_size: int,
    max_missing_rate: float,
    k_values: list[int],
):

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CUMULATIVE RETRAINING EXPERIMENT")
    print("=" * 70)

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    print("\n[1/6] Loading transactions...")
    df = pd.read_csv(transactions_path, low_memory=False)
    print(f"  Loaded {len(df):,}")

    print("\n[2/6] Chronological split...")
    train, future_windows = chronological_windows(
        df,
        time_col="TransactionDT",
        train_fraction=train_fraction,
        n_future_windows=4,
    )
    print(f"  Training: {len(train):,}")
    for i, w in enumerate(future_windows, 1):
        print(f"  W{i}: {len(w):,}")

    # --------------------------------------------------------
    # Preprocessing: static schema for consistency with existing runs
    # --------------------------------------------------------

    print("\n[3/6] Fitting preprocessing on the original training period...")
    preprocessor = NumericTransactionPreprocessor(
        policy=FeaturePolicy(
            target="isFraud",
            temporal="TransactionDT",
            identifier="TransactionID",
            max_missing_rate=max_missing_rate,
        )
    )
    X_train_original = preprocessor.fit_transform(train)
    feature_names = preprocessor.feature_names_
    print(f"  Features: {len(feature_names)}")

    # Pre-transform every window once
    X_future = [preprocessor.transform(w) for w in future_windows]
    y_future = [w["isFraud"].to_numpy() for w in future_windows]

    # --------------------------------------------------------
    # Train cumulative models and evaluate
    # --------------------------------------------------------

    print("\n[4/6] Training cumulative models...")

    cumulative_models: dict[int, IsolationForestDetector] = {}
    cumulative_sizes: dict[int, int] = {}

    for upto in range(1, 5):

        combined = build_cumulative_train_set(train, future_windows, upto)
        X_combined = preprocessor.transform(combined)

        model = IsolationForestDetector(
            n_estimators=n_estimators,
            max_samples="auto",
            contamination="auto",
            random_state=random_state + upto,
            n_jobs=-1,
        )
        model.fit(X_combined)

        cumulative_models[upto] = model
        cumulative_sizes[upto] = len(combined)

        print(f"  cumulative_w{upto}: trained on {len(combined):,} observations")

    # --------------------------------------------------------
    # Evaluate detection and fidelity
    # --------------------------------------------------------

    print("\n[5/6] Evaluating...")

    fidelity_rows = []
    detection_rows = []

    # Baseline median for fidelity perturbation (from original training period)
    baseline_median = np.median(
        X_train_original.to_numpy(dtype=float), axis=0
    )

    for upto in range(1, 5):

        model = cumulative_models[upto]
        model_name = f"cumulative_w{upto}"
        explainer = ExIFFI(model=model.model, feature_names=feature_names)

        for window_idx, (X_window, y_window) in enumerate(
            zip(X_future, y_future), 1
        ):

            # ---- Detection -------------------------------------
            det = evaluate_detection(model, X_window, y_window)
            detection_rows.append({
                "model": model_name,
                "train_size": cumulative_sizes[upto],
                "window": window_idx,
                **det,
            })

            # ---- Fidelity --------------------------------------
            # Match the earlier sampling protocol: 100 obs, seed = 42 + window
            n = min(sample_size, len(X_window))
            rng = np.random.RandomState(random_state + window_idx)
            indices = rng.choice(len(X_window), size=n, replace=False)

            X_sample = X_window.iloc[indices].to_numpy(dtype=float)
            explanations = explainer.explain_batch(X_sample)

            results = evaluate_fidelity_matrix(
                model=model.model,
                X=X_sample,
                explanations=explanations,
                baseline=baseline_median,
                k_values=k_values,
            )

            for r in results:
                fidelity_rows.append({
                    "model": model_name,
                    "window": window_idx,
                    "sample_index": r.sample_index,
                    "top_k": r.top_k,
                    "top_k_mass": r.top_k_mass,
                    "relative_score_change": r.relative_score_change,
                })

    fidelity_df = pd.DataFrame(fidelity_rows)
    detection_df = pd.DataFrame(detection_rows)

    # --------------------------------------------------------
    # Summaries
    # --------------------------------------------------------

    print("\n[6/6] Building summaries...")

    fidelity_summary = (
        fidelity_df
        .groupby(["model", "window", "top_k"], as_index=False)
        .agg(
            n_observations=("sample_index", "nunique"),
            mean_top_k_mass=("top_k_mass", "mean"),
            mean_relative_score_change=("relative_score_change", "mean"),
            median_relative_score_change=("relative_score_change", "median"),
            std_relative_score_change=("relative_score_change", "std"),
        )
    )

    fidelity_df.to_csv(
        output_path / "cumulative_fidelity_observation.csv", index=False
    )
    fidelity_summary.to_csv(
        output_path / "cumulative_fidelity_summary.csv", index=False
    )
    detection_df.to_csv(
        output_path / "cumulative_detection_summary.csv", index=False
    )

    # --------------------------------------------------------
    # Comparison table: cumulative vs static vs retrained
    # --------------------------------------------------------

    # Load static and retrained fidelity results from the existing grid
    existing_cell = Path(
        "results/retrained/full_grid/full_grid_cell_summary.csv"
    )

    comparison_rows = []

    if existing_cell.exists():

        existing = pd.read_csv(existing_cell)
        existing = existing[existing["top_k"] == PRIMARY_K]

        for upto in range(1, 5):

            for window_idx in range(1, 5):

                cum = fidelity_summary[
                    (fidelity_summary["model"] == f"cumulative_w{upto}")
                    & (fidelity_summary["window"] == window_idx)
                    & (fidelity_summary["top_k"] == PRIMARY_K)
                ]
                static = existing[
                    (existing["model"] == "static")
                    & (existing["window"] == window_idx)
                ]
                retrained = existing[
                    (existing["model"] == f"retrained_w{window_idx}")
                    & (existing["window"] == window_idx)
                ]

                comparison_rows.append({
                    "cumulative_model": f"cumulative_w{upto}",
                    "window": window_idx,
                    "cumulative_fidelity": (
                        float(cum.iloc[0]["mean_relative_score_change"])
                        if not cum.empty else np.nan
                    ),
                    "static_fidelity": (
                        float(static.iloc[0]["mean_relative_score_change"])
                        if not static.empty else np.nan
                    ),
                    "retrained_own_window_fidelity": (
                        float(retrained.iloc[0]["mean_relative_score_change"])
                        if not retrained.empty else np.nan
                    ),
                })

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_df.to_csv(
        output_path / "cumulative_comparison.csv", index=False
    )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FIDELITY SUMMARY (k=5)")
    print("=" * 70)
    k5 = fidelity_summary[fidelity_summary["top_k"] == PRIMARY_K]
    pivot = k5.pivot(
        index="window", columns="model", values="mean_relative_score_change"
    )
    print(pivot.to_string())

    print("\n" + "=" * 70)
    print("DETECTION SUMMARY")
    print("=" * 70)
    det_pivot = detection_df.pivot(
        index="window", columns="model", values="f1"
    )
    print("\nF1:")
    print(det_pivot.to_string())
    det_pivot_pr = detection_df.pivot(
        index="window", columns="model", values="pr_auc"
    )
    print("\nPR-AUC:")
    print(det_pivot_pr.to_string())

    if not comparison_df.empty:
        print("\n" + "=" * 70)
        print("COMPARISON: cumulative vs static vs retrained (own window)")
        print("=" * 70)
        print(comparison_df.to_string(index=False))

    print(f"\nSaved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Cumulative retraining experiment."
    )
    parser.add_argument("--transactions", required=True)
    parser.add_argument("--output-dir", default="results/cumulative_retraining")
    parser.add_argument("--train-fraction", type=float, default=DEFAULT_TRAIN_FRACTION)
    parser.add_argument("--estimators", type=int, default=DEFAULT_N_ESTIMATORS)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--max-missing-rate", type=float, default=DEFAULT_MAX_MISSING_RATE)
    parser.add_argument(
        "--k-values", type=int, nargs="+", default=DEFAULT_K_VALUES
    )
    args = parser.parse_args()

    run(
        transactions_path=args.transactions,
        output_dir=args.output_dir,
        train_fraction=args.train_fraction,
        n_estimators=args.estimators,
        random_state=args.random_state,
        sample_size=args.sample_size,
        max_missing_rate=args.max_missing_rate,
        k_values=args.k_values,
    )


if __name__ == "__main__":
    main()