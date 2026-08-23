from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from efd.data.preprocessing import (
    FeaturePolicy,
    NumericTransactionPreprocessor,
)

from efd.data.temporal import (
    chronological_windows,
)

from efd.models.isolation_forest import (
    IsolationForestDetector,
)

from efd.explanations.exiffi import (
    ExIFFI,
)

from efd.evaluation.explanation_fidelity import (
    evaluate_fidelity_matrix,
    evaluate_random_fidelity_matrix,
)


def run_experiment(
    transactions_path: str,
    train_fraction: float = 0.60,
    n_estimators: int = 300,
    random_state: int = 42,
    sample_size: int = 100,
):

    print("=" * 70)
    print("BASELINE 1 — EXIFFI VS RANDOM FIDELITY CONTROL")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Load data
    # ---------------------------------------------------------

    print("\n[1/8] Loading transaction dataset...")

    df = pd.read_csv(
        transactions_path,
        low_memory=False,
    )

    print(
        f"Loaded {len(df):,} transactions."
    )

    # ---------------------------------------------------------
    # 2. Chronological split
    # ---------------------------------------------------------

    print(
        "\n[2/8] Creating chronological split..."
    )

    train, future_windows = chronological_windows(
        df,
        time_col="TransactionDT",
        train_fraction=train_fraction,
        n_future_windows=1,
    )

    # ---------------------------------------------------------
    # 3. Preprocessing
    # ---------------------------------------------------------

    print(
        "\n[3/8] Fitting preprocessing..."
    )

    preprocessor = NumericTransactionPreprocessor(
        policy=FeaturePolicy(
            target="isFraud",
            temporal="TransactionDT",
            identifier="TransactionID",
            max_missing_rate=0.95,
        )
    )

    x_train = (
        preprocessor
        .fit_transform(train)
    )

    feature_names = (
        preprocessor.feature_names_
    )

    print(
        f"Features: {len(feature_names)}"
    )

    # ---------------------------------------------------------
    # 4. Train model
    # ---------------------------------------------------------

    print(
        "\n[4/8] Training Isolation Forest..."
    )

    detector = IsolationForestDetector(
        n_estimators=n_estimators,
        max_samples="auto",
        contamination="auto",
        random_state=random_state,
        n_jobs=-1,
    )

    detector.fit(x_train)

    # ---------------------------------------------------------
    # 5. Generate explanations
    # ---------------------------------------------------------

    print(
        "\n[5/8] Generating ExIFFI explanations..."
    )

    explainer = ExIFFI(
        model=detector.model,
        feature_names=feature_names,
    )

    window = future_windows[0]

    X_window = (
        preprocessor
        .transform(window)
    )

    n = min(
        sample_size,
        len(X_window),
    )

    X_sample = X_window.iloc[:n].to_numpy(
        dtype=float,
    )

    explanations = [
        explainer.explain(
            X_sample[i],
            sample_index=i,
        )
        for i in range(n)
    ]

    print(
        f"Generated {n} explanations."
    )

    # ---------------------------------------------------------
    # 6. Baseline
    # ---------------------------------------------------------

    print(
        "\n[6/8] Calculating training baseline..."
    )

    baseline = np.median(
        x_train,
        axis=0,
    )

    k_values = [
        1,
        3,
        5,
        10,
        20,
    ]

    # ---------------------------------------------------------
    # 7. Evaluate both conditions
    # ---------------------------------------------------------

    print(
        "\n[7/8] Evaluating ExIFFI and random control..."
    )

    exiffi = evaluate_fidelity_matrix(
        model=detector.model,
        X=X_sample,
        explanations=explanations,
        baseline=baseline,
        k_values=k_values,
    )

    exiffi_df = pd.DataFrame(
        [
            {
                "sample_index": r.sample_index,
                "top_k": r.top_k,
                "score_change": r.score_change,
                "relative_score_change": (
                    r.relative_score_change
                ),
            }
            for r in exiffi
        ]
    )

    exiffi_df["condition"] = "exiffi"

    random_df = evaluate_random_fidelity_matrix(
        model=detector.model,
        X=X_sample,
        explanations=explanations,
        baseline=baseline,
        k_values=k_values,
        random_state=random_state,
    )

    random_df["condition"] = "random"

    comparison = pd.concat(
        [
            exiffi_df,
            random_df[
                [
                    "sample_index",
                    "top_k",
                    "score_change",
                    "relative_score_change",
                    "condition",
                ]
            ],
        ],
        ignore_index=True,
    )

    # ---------------------------------------------------------
    # 8. Save results
    # ---------------------------------------------------------

    print(
        "\n[8/8] Saving results..."
    )

    output_dir = Path(
        "results/tables"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / "static_if_exiffi_vs_random_fidelity.csv"
    )

    comparison.to_csv(
        output_path,
        index=False,
    )

    summary = (
        comparison
        .groupby(
            [
                "condition",
                "top_k",
            ]
        )
        .agg(
            n_observations=(
                "sample_index",
                "count",
            ),
            mean_score_change=(
                "score_change",
                "mean",
            ),
            median_score_change=(
                "score_change",
                "median",
            ),
            mean_relative_score_change=(
                "relative_score_change",
                "mean",
            ),
            median_relative_score_change=(
                "relative_score_change",
                "median",
            ),
        )
        .reset_index()
    )

    summary_path = (
        output_dir
        / "static_if_exiffi_vs_random_fidelity_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print(
        f"\nResults saved to:"
        f"\n{output_path}"
    )

    print(
        f"\nSummary saved to:"
        f"\n{summary_path}"
    )

    print("\nFidelity comparison:")
    print()

    print(
        summary.to_string(
            index=False
        )
    )

    return comparison, summary


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--transactions",
        required=True,
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
    )

    args = parser.parse_args()

    run_experiment(
        transactions_path=args.transactions,
        sample_size=args.sample_size,
    )


if __name__ == "__main__":
    main()
