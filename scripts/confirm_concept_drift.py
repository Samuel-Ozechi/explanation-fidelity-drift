# scripts/confirm_concept_drift.py
"""
Rigorous concept drift confirmation for the IEEE-CIS dataset.
Tests for a statistically significant drop in predictive performance
(P(Y|X) change) using a Random Forest model and permutation testing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import false_discovery_control

from efd.data.preprocessing import (
    FeaturePolicy,
    NumericTransactionPreprocessor,
)
from efd.data.temporal import chronological_windows


def test_performance_drop(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_window: np.ndarray,
    y_window: np.ndarray,
    n_permutations: int = 100,
    random_state: int = 42,
) -> tuple[float, float]:
    """
    Test for a significant performance drop using a permutation test.
    Returns the observed performance drop and its p-value.
    """
    rng = np.random.RandomState(random_state)
    n_train = len(X_train)

    # 1. Baseline performance on training data (cross-validation)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_window_scaled = scaler.transform(X_window)

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=random_state,
        n_jobs=-1,
    )

    cv_scores = cross_val_score(
        clf, X_train_scaled, y_train, cv=3, scoring="roc_auc"
    )
    train_performance = cv_scores.mean()

    # 2. Performance on the future window
    clf.fit(X_train_scaled, y_train)
    window_scores = clf.predict_proba(X_window_scaled)[:, 1]
    window_performance = roc_auc_score(y_window, window_scores)

    observed_drop = train_performance - window_performance

    # 3. Permutation test for significance
    X_combined = np.vstack([X_train, X_window])
    y_combined = np.concatenate([y_train, y_window])

    null_drops = []
    for i in range(n_permutations):
        perm_idx = rng.permutation(len(y_combined))
        y_perm = y_combined[perm_idx]

        y_train_perm = y_perm[:n_train]
        y_window_perm = y_perm[n_train:]

        clf_perm = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=random_state + i,
            n_jobs=-1,
        )

        cv_scores_perm = cross_val_score(
            clf_perm, X_train_scaled, y_train_perm, cv=3, scoring="roc_auc"
        )
        train_perm = cv_scores_perm.mean()

        clf_perm.fit(X_train_scaled, y_train_perm)
        window_scores_perm = clf_perm.predict_proba(X_window_scaled)[:, 1]
        window_perm = roc_auc_score(y_window_perm, window_scores_perm)

        null_drops.append(train_perm - window_perm)

    p_value = (
        np.sum(np.array(null_drops) >= observed_drop) + 1
    ) / (n_permutations + 1)

    return float(observed_drop), float(p_value)


def run_drift_confirmation(
    transactions_path: str,
    output_dir: str,
    train_fraction: float = 0.60,
    n_future_windows: int = 4,
    n_permutations: int = 100,
    random_state: int = 42,
) -> None:
    """Run the full concept drift confirmation analysis."""
    print("=" * 70)
    print("CONCEPT DRIFT CONFIRMATION - IEEE-CIS DATASET")
    print("=" * 70)

    # 1. Load and prepare data
    print("\n[1/4] Loading and preprocessing data...")
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
    y_train = train["isFraud"].to_numpy()
    X_train_np = X_train.to_numpy(dtype=float)

    feature_names = preprocessor.feature_names_
    print(f"Features: {len(feature_names)}")
    print(f"Training observations: {len(train):,}")

    # 2. Test for performance drop on each window
    print("\n[2/4] Testing for performance drop (P(Y|X) change)...")
    results = []

    for i, window in enumerate(future_windows, 1):
        print(f"\nWindow {i}")

        X_window = preprocessor.transform(window)
        y_window = window["isFraud"].to_numpy()
        X_window_np = X_window.to_numpy(dtype=float)

        observed_drop, p_value = test_performance_drop(
            X_train=X_train_np,
            y_train=y_train,
            X_window=X_window_np,
            y_window=y_window,
            n_permutations=n_permutations,
            random_state=random_state + i,
        )

        results.append(
            {
                "window": i,
                "n_observations": len(window),
                "fraud_rate": window["isFraud"].mean(),
                "performance_drop": observed_drop,
                "p_value": p_value,
            }
        )

        print(f"  Fraud rate: {window['isFraud'].mean():.4%}")
        print(f"  Performance drop: {observed_drop:.4f}")
        print(f"  p-value: {p_value:.6f}")

    # 3. Apply FDR correction
    print("\n[3/4] Applying Benjamini-Hochberg FDR correction...")
    p_values = [r["p_value"] for r in results]
    adjusted_p_values = false_discovery_control(p_values, method="bh")

    for result, adj_p in zip(results, adjusted_p_values):
        result["fdr_adjusted_p_value"] = adj_p
        result["drift_significant"] = adj_p < 0.05

    # 4. Save results
    print("\n[4/4] Saving results...")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(results)
    results_path = output_path / "concept_drift_confirmation_results.csv"
    results_df.to_csv(results_path, index=False)

    print("\n" + "=" * 70)
    print("CONCEPT DRIFT CONFIRMATION SUMMARY")
    print("=" * 70)
    print(results_df.to_string(index=False))
    print(f"\nResults saved to: {results_path}")

    # Conclusion
    significant_count = results_df["drift_significant"].sum()
    if significant_count == 0:
        print(
            "\nCONCLUSION: No statistically significant concept drift "
            "(P(Y|X) change) detected after FDR correction."
        )
    else:
        print(
            f"\nCONCLUSION: Significant concept drift detected in "
            f"{significant_count} window(s) after FDR correction."
        )


def main():
    parser = argparse.ArgumentParser(
        description="Confirm concept drift in IEEE-CIS dataset"
    )
    parser.add_argument(
        "--transactions",
        required=True,
        help="Path to train_transaction.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="results/concept_drift",
        help="Output directory for results",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.60,
        help="Fraction of data for training",
    )
    parser.add_argument(
        "--windows",
        type=int,
        default=4,
        help="Number of future windows",
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=100,
        help="Number of permutation iterations",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    run_drift_confirmation(
        transactions_path=args.transactions,
        output_dir=args.output_dir,
        train_fraction=args.train_fraction,
        n_future_windows=args.windows,
        n_permutations=args.permutations,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()