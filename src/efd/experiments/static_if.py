from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from efd.data.preprocessing import (
    FeaturePolicy,
    NumericTransactionPreprocessor,
)

from efd.data.temporal import (
    chronological_windows,
    transaction_date,
)

from efd.evaluation.detection import (
    evaluate_detection,
)

from efd.models.isolation_forest import (
    IsolationForestDetector,
)


def run_experiment(
    transactions_path: str,
    output_path: str,
    train_fraction: float = 0.60,
    n_future_windows: int = 4,
    n_estimators: int = 300,
    contamination: str | float = "auto",
    random_state: int = 42,
    reference_date: str = "2017-12-01",
):

    print("=" * 70)
    print("BASELINE 1 — STATIC ISOLATION FOREST")
    print("=" * 70)

    # 1. Load labelled IEEE-CIS transactions
    print("\n[1/6] Loading transaction dataset...")

    df = pd.read_csv(
        transactions_path,
        low_memory=False,
    )

    print(
        f"Loaded {len(df):,} transactions "
        f"and {len(df.columns):,} columns."
    )


    # 2. Chronological split
    print("\n[2/6] Creating chronological split...")

    train, future_windows = chronological_windows(
        df,
        time_col="TransactionDT",
        train_fraction=train_fraction,
        n_future_windows=n_future_windows,
    )

    print(
        f"Training observations: "
        f"{len(train):,}"
    )

    print(
        f"Future observations: "
        f"{sum(len(w) for w in future_windows):,}"
    )

    print(
        f"Future windows: "
        f"{len(future_windows)}"
    )

    print(
        "\nTraining temporal range:"
    )

    print(
        f"  {transaction_date(train['TransactionDT'].min(), reference_date)}"
        f" → "
        f"{transaction_date(train['TransactionDT'].max(), reference_date)}"
    )


    # 3. Fit preprocessing on TRAIN ONLY
    print(
        "\n[3/6] Fitting preprocessing "
        "on training period only..."
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
        preprocessor
        .feature_names_
    )

    print(
        f"Selected features: "
        f"{len(feature_names)}"
    )

    print(
        f"Removed >95% missing: "
        f"{len(preprocessor.removed_high_missing_)}"
    )

    print(
        f"Removed constants: "
        f"{len(preprocessor.removed_constant_)}"
    )


    # 4. Train STATIC Isolation Forest
    print(
        "\n[4/6] Training static Isolation Forest..."
    )

    if contamination == "auto":
        contamination_value = "auto"
    else:
        contamination_value = float(
            contamination
        )

    detector = IsolationForestDetector(
        n_estimators=n_estimators,
        max_samples="auto",
        contamination=contamination_value,
        random_state=random_state,
        n_jobs=-1,
    )

    detector.fit(
        x_train
    )

    print(
        f"Isolation Forest trained with "
        f"{n_estimators} trees."
    )


    # 5. Evaluate future windows
    print(
        "\n[5/6] Evaluating future windows..."
    )

    results = []

    for window_number, window in enumerate(
        future_windows,
        start=1,
    ):

        x_window = (
            preprocessor
            .transform(window)
        )

        scores = (
            detector
            .anomaly_score(x_window)
        )

        predictions = (
            detector
            .predict_anomaly(x_window)
        )

        metrics = evaluate_detection(
            y_true=window["isFraud"],
            anomaly_scores=scores,
            predictions=predictions,
        )

        metrics.update(
            {
                "window": window_number,

                "date_start": transaction_date(
                    window["TransactionDT"].min(),
                    reference_date,
                ),

                "date_end": transaction_date(
                    window["TransactionDT"].max(),
                    reference_date,
                ),

                "n_features": len(
                    feature_names
                ),

                "n_estimators": (
                    n_estimators
                ),

                "contamination": str(
                    contamination_value
                ),

                "random_state": (
                    random_state
                ),

                "model": (
                    "static_isolation_forest"
                ),
            }
        )

        results.append(
            metrics
        )

        print(
            f"\nWindow {window_number}"
        )

        print(
            f"  N:        "
            f"{metrics['n']:,}"
        )

        print(
            f"  Fraud:    "
            f"{metrics['fraud']:,}"
        )

        print(
            f"  Fraud rate:"
            f" {metrics['fraud_rate']:.4%}"
        )

        print(
            f"  F1:       "
            f"{metrics['f1']:.6f}"
        )

        print(
            f"  PR-AUC:   "
            f"{metrics['pr_auc']:.6f}"
        )

    # 6. Save results
    print(
        "\n[6/6] Saving results..."
    )

    result_df = pd.DataFrame(
        results
    )

    output = Path(
        output_path
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_df.to_csv(
        output,
        index=False,
    )

    # Save exact feature list.
    feature_path = (
        output.parent
        / f"{output.stem}_features.txt"
    )

    feature_path.write_text(
        "\n".join(feature_names),
        encoding="utf-8",
    )

    # Save preprocessing audit.
    audit_path = (
        output.parent
        / f"{output.stem}_preprocessing.txt"
    )

    audit_lines = [
        "BASELINE 1 PREPROCESSING",
        "",
        f"Selected features: {len(feature_names)}",
        (
            "Maximum missing rate: "
            "0.95"
        ),
        (
            "Removed high-missing features: "
            f"{len(preprocessor.removed_high_missing_)}"
        ),
        (
            "Removed constants: "
            f"{len(preprocessor.removed_constant_)}"
        ),
        "",
        "High-missing features:",
        *preprocessor.removed_high_missing_,
        "",
        "Constant features:",
        *preprocessor.removed_constant_,
    ]

    audit_path.write_text(
        "\n".join(audit_lines),
        encoding="utf-8",
    )

    print(
        f"\nResults saved to:"
        f"\n{output}"
    )

    print(
        f"\nFeatures saved to:"
        f"\n{feature_path}"
    )

    print(
        f"\nPreprocessing audit saved to:"
        f"\n{audit_path}"
    )

    print("\n")
    print(result_df.to_string(index=False))

    return result_df


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Baseline 1: "
            "Static Isolation Forest "
            "on IEEE-CIS"
        )
    )

    parser.add_argument(
        "--transactions",
        required=True,
    )

    parser.add_argument(
        "--output",
        default=(
            "results/tables/"
            "static_if_v1.csv"
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

    parser.add_argument(
        "--trees",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--contamination",
        default="auto",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--reference-date",
        default="2017-12-01",
        help=(
            "Calendar date corresponding to TransactionDT=0 "
            "(default: 2017-12-01)."
        ),
    )

    args = parser.parse_args()

    run_experiment(
        transactions_path=(
            args.transactions
        ),
        output_path=(
            args.output
        ),
        train_fraction=(
            args.train_fraction
        ),
        n_future_windows=(
            args.windows
        ),
        n_estimators=(
            args.trees
        ),
        contamination=(
            args.contamination
        ),
        random_state=(
            args.random_state
        ),
        reference_date=args.reference_date,
    )


if __name__ == "__main__":
    main()
