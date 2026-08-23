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
)


def run_experiment(
    transactions_path: str,
    train_fraction: float = 0.60,
    n_future_windows: int = 1,
    n_estimators: int = 300,
    random_state: int = 42,
    reference_date: str = "2017-12-01",
    sample_size: int = 100,
):

    print("=" * 70)
    print("BASELINE 1 — EXIFFI LOCAL FIDELITY")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Load data
    # ---------------------------------------------------------

    print("\n[1/7] Loading transaction dataset...")

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
        "\n[2/7] Creating chronological split..."
    )

    train, future_windows = chronological_windows(
        df,
        time_col="TransactionDT",
        train_fraction=train_fraction,
        n_future_windows=n_future_windows,
    )

    # ---------------------------------------------------------
    # 3. Preprocessing
    # ---------------------------------------------------------

    print(
        "\n[3/7] Fitting preprocessing..."
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
    # 4. Train Isolation Forest
    # ---------------------------------------------------------

    print(
        "\n[4/7] Training Isolation Forest..."
    )

    detector = IsolationForestDetector(
        n_estimators=n_estimators,
        max_samples="auto",
        contamination="auto",
        random_state=random_state,
        n_jobs=-1,
    )

    detector.fit(
        x_train
    )

    # ---------------------------------------------------------
    # 5. Build ExIFFI explanations
    # ---------------------------------------------------------

    print(
        "\n[5/7] Generating ExIFFI explanations..."
    )

    explain = ExIFFI(
        model=detector.model,
        feature_names=feature_names,
    )

    # Use first future window.
    window = future_windows[0]

    X_window = (
        preprocessor
        .transform(window)
    )

    n = min(
        sample_size,
        len(X_window),
    )

    X_sample = X_window.iloc[:n].to_numpy(dtype=float)

    explanations = [
        explain.explain(
            X_sample[i],
            sample_index=i,
        )
        for i in range(n)
    ]

    print(
        f"Generated explanations for {n} observations."
    )

    # ---------------------------------------------------------
    # 6. Define baseline
    # ---------------------------------------------------------

    print(
        "\n[6/7] Evaluating feature perturbation fidelity..."
    )

    # Median feature values from training data.
    #
    # IMPORTANT:
    # baseline is calculated from TRAIN ONLY.
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

    results = evaluate_fidelity_matrix(
        model=detector.model,
        X=X_sample,
        explanations=explanations,
        baseline=baseline,
        k_values=k_values,
    )

    result_rows = [
        {
            "sample_index": r.sample_index,
            "baseline_score": r.baseline_score,
            "top_k": r.top_k,
            "top_k_mass": r.top_k_mass,
            "perturbed_score": r.perturbed_score,
            "score_change": r.score_change,
            "relative_score_change": (
                r.relative_score_change
            ),
        }
        for r in results
    ]

    result_df = pd.DataFrame(
        result_rows
    )

    # ---------------------------------------------------------
    # 7. Save
    # ---------------------------------------------------------

    print(
        "\n[7/7] Saving fidelity results..."
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
        / "static_if_exiffi_w1_fidelity.csv"
    )

    result_df.to_csv(
        output_path,
        index=False,
    )

    summary = (
        result_df
        .groupby("top_k")
        .agg(
            n_observations=(
                "sample_index",
                "count",
            ),
            mean_top_k_mass=(
                "top_k_mass",
                "mean",
            ),
            median_top_k_mass=(
                "top_k_mass",
                "median",
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
        / "static_if_exiffi_w1_fidelity_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    print(
        f"\nResults saved to:\n{output_path}"
    )

    print(
        f"\nSummary saved to:\n{summary_path}"
    )

    print(
        "\nFidelity summary:"
    )

    print(
        summary.to_string(
            index=False
        )
    )

    return result_df, summary


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