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
    transaction_date,
)

from efd.models.isolation_forest import (
    IsolationForestDetector,
)

from efd.explanations.exiffi import (
    ExIFFI,
)


def run_experiment(
    transactions_path: str,
    output_dir: str,
    train_fraction: float = 0.60,
    n_future_windows: int = 4,
    n_estimators: int = 300,
    random_state: int = 42,
    explanation_samples: int = 100,
    reference_date: str = "2017-12-01",
):
    print("=" * 70)
    print("BASELINE 1 + EXIFFI")
    print("=" * 70)

   
    # 1. Load data

    print("\n[1/8] Loading transaction dataset...")

    df = pd.read_csv(
        transactions_path,
        low_memory=False,
    )

    print(
        f"Loaded {len(df):,} transactions "
        f"and {len(df.columns):,} columns."
    )

    # 2. Chronological split

    print("\n[2/8] Creating chronological split...")

    train, future_windows = chronological_windows(
        df,
        time_col="TransactionDT",
        train_fraction=train_fraction,
        n_future_windows=n_future_windows,
    )

    print(
        f"Training observations: {len(train):,}"
    )

    print(
        f"Future observations: "
        f"{sum(len(w) for w in future_windows):,}"
    )

    print(
        f"Future windows: {len(future_windows)}"
    )

    print("\nTraining temporal range:")

    print(
        f"  {transaction_date(train['TransactionDT'].min(), reference_date)}"
        f" → "
        f"{transaction_date(train['TransactionDT'].max(), reference_date)}"
    )

    # 3. Preprocessing

    print(
        "\n[3/8] Fitting preprocessing "
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

    x_train = preprocessor.fit_transform(train)

    feature_names = preprocessor.feature_names_

    print(
        f"Selected features: {len(feature_names)}"
    )

    print(
        f"Removed >95% missing: "
        f"{len(preprocessor.removed_high_missing_)}"
    )

    print(
        f"Removed constants: "
        f"{len(preprocessor.removed_constant_)}"
    )

    # 4. Train static Isolation Forest
    print(
        "\n[4/8] Training static Isolation Forest..."
    )

    detector = IsolationForestDetector(
        n_estimators=n_estimators,
        max_samples="auto",
        contamination="auto",
        random_state=random_state,
        n_jobs=-1,
    )

    detector.fit(x_train)

    print(
        f"Isolation Forest trained with "
        f"{n_estimators} trees."
    )

    
    # 5. Create ExIFFI explainer
    print(
        "\n[5/8] Initialising ExIFFI..."
    )

    explainer = ExIFFI(
        model=detector.model,
        feature_names=feature_names,
    )

    print(
        f"ExIFFI ready for "
        f"{len(feature_names)} features."
    )

    # 6. Explain Window 1 sample
    print(
        "\n[6/8] Generating Window 1 explanations..."
    )

    window = future_windows[0]

    x_window = preprocessor.transform(window)

    n_explain = min(
        explanation_samples,
        len(window),
    )

    x_sample = x_window[:n_explain]

    y_sample = (
        window["isFraud"]
        .iloc[:n_explain]
        .to_numpy()
    )

    scores = detector.anomaly_score(
        x_sample
    )

    explanations = explainer.explain_batch(
        x_sample
    )

    print(
        f"Explained {len(explanations):,} "
        f"observations."
    )

    # 7. Construct explanation output
    print(
        "\n[7/8] Building explanation dataset..."
    )

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    for i, explanation in enumerate(
        explanations
    ):

        row = {
            "sample_index": i,
            "isFraud": int(y_sample[i]),
            "anomaly_score": float(scores[i]),
        }

        importance = (
            explanation.normalized
        )

        # Preserve the complete explanation vector.
        for feature_index, feature_name in enumerate(
            feature_names
        ):
            row[
                f"importance__{feature_name}"
            ] = float(
                importance[feature_index]
            )

        # Top-10 features for convenient inspection.
        top_indices = np.argsort(
            -importance
        )[:10]

        for rank, feature_index in enumerate(
            top_indices,
            start=1,
        ):

            row[
                f"top_{rank}_feature"
            ] = feature_names[
                feature_index
            ]

            row[
                f"top_{rank}_importance"
            ] = float(
                importance[feature_index]
            )

        rows.append(row)

    explanation_df = pd.DataFrame(
        rows
    )

    explanation_path = (
        output_dir
        / "static_if_exiffi_w1_sample.csv"
    )

    explanation_df.to_csv(
        explanation_path,
        index=False,
    )

    # Save compact top-feature table

    top_rows = []

    for i, explanation in enumerate(
        explanations
    ):

        top_features = (
            explanation.top_features(
                feature_names,
                k=10,
            )
        )

        for rank, (
            feature,
            importance,
        ) in enumerate(
            top_features,
            start=1,
        ):

            top_rows.append(
                {
                    "sample_index": i,
                    "isFraud": int(
                        y_sample[i]
                    ),
                    "anomaly_score": float(
                        scores[i]
                    ),
                    "rank": rank,
                    "feature": feature,
                    "importance": importance,
                }
            )

    top_df = pd.DataFrame(
        top_rows
    )

    top_path = (
        output_dir
        / "static_if_exiffi_w1_top_features.csv"
    )

    top_df.to_csv(
        top_path,
        index=False,
    )

    # Save metadata
    metadata = [
        "BASELINE 1 + EXIFFI",
        "",
        f"Training observations: {len(train)}",
        f"Future windows: {len(future_windows)}",
        f"Window explained: 1",
        f"Observations explained: {n_explain}",
        f"Features: {len(feature_names)}",
        f"Trees: {n_estimators}",
        f"Random state: {random_state}",
        "",
        "ExIFFI:",
        "Standard Isolation Forest axis-aligned splits",
        "Node feature direction = selected feature axis",
        "Path weight = parent_node_samples / child_node_samples",
        "",
        "Outputs:",
        str(explanation_path),
        str(top_path),
    ]

    metadata_path = (
        output_dir
        / "static_if_exiffi_w1_metadata.txt"
    )

    metadata_path.write_text(
        "\n".join(metadata),
        encoding="utf-8",
    )


    # 8. Summary
    print(
        "\n[8/8] Experiment complete."
    )

    print(
        f"\nFull explanations:"
        f"\n{explanation_path}"
    )

    print(
        f"\nTop features:"
        f"\n{top_path}"
    )

    print(
        f"\nMetadata:"
        f"\n{metadata_path}"
    )

    print("\nFirst observations:")

    print(
        top_df.head(30).to_string(
            index=False
        )
    )

    return explanation_df


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Baseline 1 + ExIFFI "
            "on IEEE-CIS"
        )
    )

    parser.add_argument(
        "--transactions",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "results/explanations"
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
        "--samples",
        type=int,
        default=100,
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

    args = parser.parse_args()

    run_experiment(
        transactions_path=args.transactions,
        output_dir=args.output_dir,
        train_fraction=args.train_fraction,
        n_future_windows=args.windows,
        n_estimators=args.trees,
        explanation_samples=args.samples,
        random_state=args.random_state,
        reference_date=args.reference_date,
    )


if __name__ == "__main__":
    main()