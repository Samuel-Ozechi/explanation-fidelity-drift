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

from efd.models.isolation_forest import (
    IsolationForestDetector,
)

from efd.explanations.exiffi import (
    ExIFFI,
)


OUTPUT_DIR = Path(
    "results/explanations"
)


def run_experiment(
    transactions_path: str,
    train_fraction: float = 0.60,
    n_future_windows: int = 4,
    n_estimators: int = 300,
    random_state: int = 42,
    reference_date: str = "2017-12-01",
    sample_size: int = 100,
):
    print("=" * 70)
    print("BASELINE 1 — TEMPORAL EXIFFI EXPLANATIONS")
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
        f"Loaded {len(df):,} transactions "
        f"and {len(df.columns):,} columns."
    )

    # ---------------------------------------------------------
    # 2. Chronological split
    # ---------------------------------------------------------

    print(
        "\n[2/7] Creating chronological windows..."
    )

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
        f"Future windows: "
        f"{len(future_windows)}"
    )

    # ---------------------------------------------------------
    # 3. Fit preprocessing on TRAIN ONLY
    # ---------------------------------------------------------

    print(
        "\n[3/7] Fitting preprocessing "
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
        preprocessor.feature_names_
    )

    print(
        f"Selected features: "
        f"{len(feature_names)}"
    )

    # ---------------------------------------------------------
    # 4. Train ONE static Isolation Forest
    # ---------------------------------------------------------

    print(
        "\n[4/7] Training static Isolation Forest..."
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

    print(
        f"Isolation Forest trained with "
        f"{n_estimators} trees."
    )

    # ---------------------------------------------------------
    # 5. Build ExIFFI explainer
    # ---------------------------------------------------------

    print(
        "\n[5/7] Initializing ExIFFI..."
    )

    explainer = ExIFFI(
        model=detector.model,
        feature_names=feature_names,
    )

    # ---------------------------------------------------------
    # 6. Generate explanations for ALL windows
    # ---------------------------------------------------------

    print(
        "\n[6/7] Generating temporal explanations..."
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    generated_files = []

    for window_number, window in enumerate(
        future_windows,
        start=1,
    ):

        print(
            f"\nWindow {window_number}"
        )

        print(
            f"  Observations: "
            f"{len(window):,}"
        )

        print(
            f"  Date range: "
            f"{transaction_date(window['TransactionDT'].min(), reference_date)}"
            f" → "
            f"{transaction_date(window['TransactionDT'].max(), reference_date)}"
        )

        # Transform using TRAIN-FITTED preprocessor.
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

        explanations = []

        for i in range(n):

            explanation = explainer.explain(
                X_sample[i],
                sample_index=i,
            )

            explanations.append(
                explanation
            )

        # -----------------------------------------------------
        # Build wide explanation matrix
        # -----------------------------------------------------

        rows = []

        for i, explanation in enumerate(
            explanations
        ):

            row = {
                "sample_index": i,
                "isFraud": int(
                    window.iloc[i]["isFraud"]
                ),
                "anomaly_score": float(
                    -explanation.anomaly_score
                ),
            }

            normalized = (
                explanation.normalized
            )

            for feature, importance in zip(
                feature_names,
                normalized,
            ):

                row[
                    f"importance__{feature}"
                ] = float(importance)

            # Top-k features.
            top_features = (
                explanation.top_features(
                    feature_names,
                    k=20,
                )
            )

            for rank, (
                feature,
                importance,
            ) in enumerate(
                top_features,
                start=1,
            ):

                row[
                    f"top_{rank}_feature"
                ] = feature

                row[
                    f"top_{rank}_importance"
                ] = float(importance)

            rows.append(row)

        result_df = pd.DataFrame(
            rows
        )

        output_path = (
            OUTPUT_DIR
            / (
                "static_if_exiffi_"
                f"w{window_number}_sample.csv"
            )
        )

        result_df.to_csv(
            output_path,
            index=False,
        )

        generated_files.append(
            output_path
        )

        print(
            f"  Generated explanations: {n}"
        )

        print(
            f"  Saved: {output_path}"
        )

    # ---------------------------------------------------------
    # 7. Save experiment metadata
    # ---------------------------------------------------------

    print(
        "\n[7/7] Saving temporal experiment metadata..."
    )

    metadata = [
        "STATIC ISOLATION FOREST — TEMPORAL EXIFFI",
        "",
        f"Train fraction: {train_fraction}",
        f"Number of trees: {n_estimators}",
        f"Random state: {random_state}",
        f"Number of future windows: {len(future_windows)}",
        f"Samples per window: {sample_size}",
        f"Number of features: {len(feature_names)}",
        "",
        "IMPORTANT:",
        "The Isolation Forest was trained once on the",
        "training period and was not retrained between",
        "future windows.",
        "",
        "The preprocessing pipeline was fitted once on",
        "the training period and reused for all future",
        "windows.",
        "",
        "Generated files:",
        *[
            str(path)
            for path in generated_files
        ],
    ]

    metadata_path = (
        OUTPUT_DIR
        / "static_if_exiffi_temporal_metadata.txt"
    )

    metadata_path.write_text(
        "\n".join(metadata),
        encoding="utf-8",
    )

    print(
        f"\nMetadata saved to:"
        f"\n{metadata_path}"
    )

    print("\nDone.")

    return generated_files


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Generate ExIFFI explanations "
            "across temporal future windows."
        )
    )

    parser.add_argument(
        "--transactions",
        required=True,
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
        "--random-state",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--reference-date",
        default="2017-12-01",
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
    )

    args = parser.parse_args()

    run_experiment(
        transactions_path=args.transactions,
        train_fraction=args.train_fraction,
        n_future_windows=args.windows,
        n_estimators=args.trees,
        random_state=args.random_state,
        reference_date=args.reference_date,
        sample_size=args.sample_size,
    )


if __name__ == "__main__":
    main()
