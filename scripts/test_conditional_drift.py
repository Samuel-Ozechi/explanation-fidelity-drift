# scripts/test_conditional_drift.py
"""
Conditional drift test between the training period and each future window.

Tests whether P(Y|X) is statistically distinguishable between the
training period and each future window, after accounting for the
covariate shift already established in Section 5.1.

Method:
    Classifier two-sample test on the joint (X, Y).

    For each window W:
        - Fit a classifier to separate training (label 0) from W (label 1)
          using X alone.     -> accuracy_X
        - Fit the same classifier using [X, Y] together.
                               -> accuracy_XY
        - The improvement (accuracy_XY - accuracy_X) is the conditional-
          drift signal. Under the null of no conditional drift, the joint
          classifier should not outperform the marginal classifier beyond
          what is explained by X alone.
        - Permutation test: shuffle Y within the pooled dataset and
          recompute the improvement, providing a null distribution.

Outputs:
    results/conditional_drift/conditional_drift_results.csv
    results/conditional_drift/conditional_drift_summary.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from scipy.stats import false_discovery_control

from efd.data.preprocessing import (
    FeaturePolicy,
    NumericTransactionPreprocessor,
)
from efd.data.temporal import chronological_windows


DEFAULT_TRANSACTIONS = "data/raw/ieee_cis/train_transaction.csv"
DEFAULT_OUTPUT_DIR = "results/conditional_drift"
DEFAULT_TRAIN_FRACTION = 0.60
DEFAULT_MAX_MISSING_RATE = 0.95
DEFAULT_N_ESTIMATORS = 200
DEFAULT_MAX_DEPTH = 10
DEFAULT_RANDOM_STATE = 42
DEFAULT_N_PERMUTATIONS = 200
DEFAULT_SUBSAMPLE = 4000   # cap per side to keep the CV tractable


# ============================================================
# Classifier two-sample test
# ============================================================

def classifier_two_sample_accuracy(
    X_group: np.ndarray,
    y_group: np.ndarray,
    model_factory,
    n_splits: int = 5,
    random_state: int = 42,
) -> float:
    """
    Return the cross-validated accuracy of a classifier trained to
    distinguish the two groups described by y_group (0 vs 1), using
    X_group as the feature matrix.
    """
    cv = StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )
    scores = cross_val_score(
        model_factory(),
        X_group,
        y_group,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
    )
    return float(scores.mean())


def evaluate_window(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_window: np.ndarray,
    y_window: np.ndarray,
    model_factory,
    random_state: int,
    n_permutations: int,
    subsample: int | None,
):
    """
    Run the classifier two-sample test on one training/window pair.
    """
    rng = np.random.default_rng(random_state)

    # --------------------------------------------------------
    # Optional subsampling for tractability
    # --------------------------------------------------------
    if subsample is not None:
        n_train = min(subsample, len(X_train))
        n_window = min(subsample, len(X_window))

        idx_train = rng.choice(len(X_train), size=n_train, replace=False)
        idx_window = rng.choice(len(X_window), size=n_window, replace=False)

        X_train = X_train[idx_train]
        y_train = y_train[idx_train]
        X_window = X_window[idx_window]
        y_window = y_window[idx_window]

    # --------------------------------------------------------
    # Pool the two samples
    # --------------------------------------------------------
    X_pool = np.vstack([X_train, X_window])
    y_pool = np.concatenate([y_train, y_window])

    group = np.concatenate([
        np.zeros(len(X_train), dtype=int),
        np.ones(len(X_window), dtype=int),
    ])

    # --------------------------------------------------------
    # Accuracy using X alone
    # --------------------------------------------------------
    acc_X = classifier_two_sample_accuracy(
        X_group=X_pool,
        y_group=group,
        model_factory=model_factory,
        random_state=random_state,
    )

    # --------------------------------------------------------
    # Accuracy using X and Y together
    # --------------------------------------------------------
    XY_pool = np.column_stack([X_pool, y_pool])
    acc_XY = classifier_two_sample_accuracy(
        X_group=XY_pool,
        y_group=group,
        model_factory=model_factory,
        random_state=random_state,
    )

    observed_improvement = acc_XY - acc_X

    # --------------------------------------------------------
    # Permutation test: shuffle Y within the pooled data
    # --------------------------------------------------------
    null_improvements = np.empty(n_permutations, dtype=float)

    for i in range(n_permutations):

        y_perm = rng.permutation(y_pool)
        XY_perm = np.column_stack([X_pool, y_perm])

        acc_X_perm = classifier_two_sample_accuracy(
            X_group=X_pool,
            y_group=group,
            model_factory=model_factory,
            random_state=random_state,
        )
        acc_XY_perm = classifier_two_sample_accuracy(
            X_group=XY_perm,
            y_group=group,
            model_factory=model_factory,
            random_state=random_state,
        )
        null_improvements[i] = acc_XY_perm - acc_X_perm

    p_value = (
        np.sum(null_improvements >= observed_improvement) + 1
    ) / (n_permutations + 1)

    return {
        "accuracy_X": acc_X,
        "accuracy_XY": acc_XY,
        "observed_improvement": observed_improvement,
        "null_mean": float(np.mean(null_improvements)),
        "null_std": float(np.std(null_improvements)),
        "p_value": float(p_value),
        "n_train": int(len(X_train)),
        "n_window": int(len(X_window)),
        "n_features": int(X_pool.shape[1]),
        "n_permutations": int(n_permutations),
    }


# ============================================================
# Main
# ============================================================

def run(
    transactions_path: str,
    output_dir: str,
    train_fraction: float,
    max_missing_rate: float,
    n_estimators: int,
    max_depth: int,
    random_state: int,
    n_permutations: int,
    subsample: int | None,
):

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CONDITIONAL DRIFT TEST (CLASSIFIER TWO-SAMPLE ON JOINT X,Y)")
    print("=" * 70)

    # --------------------------------------------------------
    # Load and split
    # --------------------------------------------------------

    print("\n[1/4] Loading and splitting...")
    df = pd.read_csv(transactions_path, low_memory=False)
    train, future_windows = chronological_windows(
        df,
        time_col="TransactionDT",
        train_fraction=train_fraction,
        n_future_windows=4,
    )

    # --------------------------------------------------------
    # Preprocessing (training-only schema)
    # --------------------------------------------------------

    print("\n[2/4] Fitting training-only preprocessing...")
    preprocessor = NumericTransactionPreprocessor(
        policy=FeaturePolicy(
            target="isFraud",
            temporal="TransactionDT",
            identifier="TransactionID",
            max_missing_rate=max_missing_rate,
        )
    )
    X_train = preprocessor.fit_transform(train).to_numpy(dtype=float)
    y_train = train["isFraud"].to_numpy(dtype=int)

    X_windows = [
        preprocessor.transform(w).to_numpy(dtype=float)
        for w in future_windows
    ]
    y_windows = [w["isFraud"].to_numpy(dtype=int) for w in future_windows]

    print(f"  Features: {len(preprocessor.feature_names_)}")
    print(f"  Training observations: {len(X_train):,}")

    # --------------------------------------------------------
    # Model factory
    # --------------------------------------------------------

    def model_factory():
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )

    # --------------------------------------------------------
    # Run test per window
    # --------------------------------------------------------

    print(f"\n[3/4] Running conditional drift test "
          f"({n_permutations} permutations per window)...")

    results = []
    for window_idx, (X_w, y_w) in enumerate(zip(X_windows, y_windows), 1):

        print(f"\n  Window {window_idx}")
        result = evaluate_window(
            X_train=X_train,
            y_train=y_train,
            X_window=X_w,
            y_window=y_w,
            model_factory=model_factory,
            random_state=random_state + window_idx,
            n_permutations=n_permutations,
            subsample=subsample,
        )
        result["window"] = window_idx
        results.append(result)

        print(f"    accuracy(X):   {result['accuracy_X']:.4f}")
        print(f"    accuracy(X,Y): {result['accuracy_XY']:.4f}")
        print(f"    improvement:   {result['observed_improvement']:.4f}")
        print(f"    p-value:       {result['p_value']:.6f}")

    results_df = pd.DataFrame(results)[[
        "window", "n_train", "n_window", "n_features",
        "accuracy_X", "accuracy_XY", "observed_improvement",
        "null_mean", "null_std", "p_value", "n_permutations",
    ]]

    # --------------------------------------------------------
    # BH-FDR correction
    # --------------------------------------------------------

    adjusted = false_discovery_control(
        results_df["p_value"].to_numpy(), method="bh"
    )
    results_df["fdr_adjusted_p_value"] = adjusted
    results_df["conditional_drift_significant"] = adjusted < 0.05

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    results_path = output_path / "conditional_drift_results.csv"
    results_df.to_csv(results_path, index=False)

    print("\n[4/4] Summary")
    print(results_df.to_string(index=False))

    print(f"\nSaved: {results_path}")

    return results_df


def main():
    parser = argparse.ArgumentParser(
        description="Conditional drift test via classifier two-sample on (X, Y)."
    )
    parser.add_argument("--transactions", default=DEFAULT_TRANSACTIONS)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-fraction", type=float, default=DEFAULT_TRAIN_FRACTION)
    parser.add_argument("--max-missing-rate", type=float, default=DEFAULT_MAX_MISSING_RATE)
    parser.add_argument("--n-estimators", type=int, default=DEFAULT_N_ESTIMATORS)
    parser.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--permutations", type=int, default=DEFAULT_N_PERMUTATIONS)
    parser.add_argument(
        "--subsample",
        type=int,
        default=DEFAULT_SUBSAMPLE,
        help="Cap per-side sample size (default 4000). Set 0 for no cap.",
    )
    args = parser.parse_args()

    subsample = args.subsample if args.subsample > 0 else None

    run(
        transactions_path=args.transactions,
        output_dir=args.output_dir,
        train_fraction=args.train_fraction,
        max_missing_rate=args.max_missing_rate,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=args.random_state,
        n_permutations=args.permutations,
        subsample=subsample,
    )


if __name__ == "__main__":
    main()