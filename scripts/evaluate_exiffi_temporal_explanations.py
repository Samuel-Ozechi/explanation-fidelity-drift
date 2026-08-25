from __future__ import annotations

from pathlib import Path

import pandas as pd

from efd.evaluation.explanation import (
    evaluate_explanation_matrix,
)


INPUT_DIR = Path("results/explanations")

OUTPUT_DIR = Path("results/tables")

OUTPUT = (
    OUTPUT_DIR
    / "static_if_exiffi_temporal_explanation_metrics.csv"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "static_if_exiffi_temporal_explanation_summary.csv"
)

CLASS_SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "static_if_exiffi_temporal_explanation_class_summary.csv"
)


def _window_input_path(window: int) -> Path:
    return (
        INPUT_DIR
        / f"static_if_exiffi_w{window}_sample.csv"
    )


def _extract_feature_names(
    df: pd.DataFrame,
) -> list[str]:

    importance_cols = [
        column
        for column in df.columns
        if column.startswith("importance__")
    ]

    if not importance_cols:
        raise ValueError(
            "No importance columns found."
        )

    return [
        column.replace(
            "importance__",
            "",
            1,
        )
        for column in importance_cols
    ]


def build_summary(
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build overall temporal explanation summary.

    The summary preserves fraud and non-fraud class counts
    for every temporal window.
    """

    required_columns = {
        "window",
        "isFraud",
        "top_1_mass",
        "top_5_mass",
        "top_10_mass",
        "entropy",
        "effective_features",
        "nonzero_features",
    }

    missing = (
        required_columns
        - set(metrics.columns)
    )

    if missing:
        raise ValueError(
            "Missing required columns for temporal "
            "summary: "
            f"{sorted(missing)}"
        )

    rows = []

    for window, group in metrics.groupby(
        "window"
    ):

        n_observations = len(group)

        n_fraud = int(
            (
                group["isFraud"] == 1
            ).sum()
        )

        n_nonfraud = int(
            (
                group["isFraud"] == 0
            ).sum()
        )

        fraud_rate = (
            n_fraud / n_observations
            if n_observations > 0
            else 0.0
        )

        if "n_features" in group.columns:
            n_features = int(
                group["n_features"].iloc[0]
            )
        else:
            n_features = None

        rows.append(
            {
                "window": window,

                "n_features": n_features,

                "n_observations": (
                    n_observations
                ),

                "n_fraud": n_fraud,

                "n_nonfraud": n_nonfraud,

                "fraud_rate": fraud_rate,

                "mean_top_1_mass": (
                    group["top_1_mass"].mean()
                ),

                "mean_top_5_mass": (
                    group["top_5_mass"].mean()
                ),

                "mean_top_10_mass": (
                    group["top_10_mass"].mean()
                ),

                "mean_entropy": (
                    group["entropy"].mean()
                ),

                "mean_effective_features": (
                    group[
                        "effective_features"
                    ].mean()
                ),

                "mean_nonzero_features": (
                    group[
                        "nonzero_features"
                    ].mean()
                ),

                "median_top_1_mass": (
                    group["top_1_mass"].median()
                ),

                "median_top_5_mass": (
                    group["top_5_mass"].median()
                ),

                "median_top_10_mass": (
                    group["top_10_mass"].median()
                ),

                "median_entropy": (
                    group["entropy"].median()
                ),

                "median_effective_features": (
                    group[
                        "effective_features"
                    ].median()
                ),

                "median_nonzero_features": (
                    group[
                        "nonzero_features"
                    ].median()
                ),

                "n_samples": float(
                    n_observations
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("window")
        .reset_index(drop=True)
    )


def build_class_summary(
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build class-stratified temporal explanation metrics.

    Produces one row for each:

        window × class

    where class is either:

        fraud
        nonfraud
    """

    required_columns = {
        "window",
        "isFraud",
        "top_1_mass",
        "top_5_mass",
        "top_10_mass",
        "entropy",
        "effective_features",
        "nonzero_features",
    }

    missing = (
        required_columns
        - set(metrics.columns)
    )

    if missing:
        raise ValueError(
            "Missing required columns for class "
            "summary: "
            f"{sorted(missing)}"
        )

    grouped = (
        metrics
        .groupby(
            ["window", "isFraud"],
            dropna=False,
        )
        .agg(
            n_observations=(
                "isFraud",
                "count",
            ),

            mean_top_1_mass=(
                "top_1_mass",
                "mean",
            ),

            mean_top_5_mass=(
                "top_5_mass",
                "mean",
            ),

            mean_top_10_mass=(
                "top_10_mass",
                "mean",
            ),

            mean_entropy=(
                "entropy",
                "mean",
            ),

            mean_effective_features=(
                "effective_features",
                "mean",
            ),

            mean_nonzero_features=(
                "nonzero_features",
                "mean",
            ),

            median_top_1_mass=(
                "top_1_mass",
                "median",
            ),

            median_top_5_mass=(
                "top_5_mass",
                "median",
            ),

            median_top_10_mass=(
                "top_10_mass",
                "median",
            ),

            median_entropy=(
                "entropy",
                "median",
            ),

            median_effective_features=(
                "effective_features",
                "median",
            ),

            median_nonzero_features=(
                "nonzero_features",
                "median",
            ),
        )
        .reset_index()
    )

    grouped["class"] = grouped[
        "isFraud"
    ].map(
        {
            0: "nonfraud",
            1: "fraud",
        }
    )

    grouped = grouped[
        [
            "window",
            "isFraud",
            "class",
            "n_observations",

            "mean_top_1_mass",
            "mean_top_5_mass",
            "mean_top_10_mass",

            "mean_entropy",
            "mean_effective_features",
            "mean_nonzero_features",

            "median_top_1_mass",
            "median_top_5_mass",
            "median_top_10_mass",

            "median_entropy",
            "median_effective_features",
            "median_nonzero_features",
        ]
    ]

    return (
        grouped
        .sort_values(
            [
                "window",
                "isFraud",
            ]
        )
        .reset_index(drop=True)
    )


def evaluate_window(
    window: int,
) -> pd.DataFrame:

    input_path = _window_input_path(
        window
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Explanation file not found: "
            f"{input_path}"
        )

    print()
    print("=" * 70)
    print(f"Window {window}")
    print("=" * 70)

    df = pd.read_csv(
        input_path
    )

    if "isFraud" not in df.columns:
        raise ValueError(
            f"'isFraud' column missing from "
            f"{input_path}"
        )

    if "anomaly_score" not in df.columns:
        raise ValueError(
            f"'anomaly_score' column missing from "
            f"{input_path}"
        )

    feature_names = _extract_feature_names(
        df
    )

    importance_cols = [
        f"importance__{feature}"
        for feature in feature_names
    ]

    importance = (
        df[importance_cols]
        .to_numpy(dtype=float)
    )

    print(
        f"  Observations: {len(df):,}"
    )

    print(
        f"  Features: {len(feature_names):,}"
    )

    print(
        f"  Fraud: "
        f"{int(df['isFraud'].sum()):,}"
    )

    metrics = evaluate_explanation_matrix(
        importance=importance,
        feature_names=feature_names,
        window=window,
    )

    # Preserve observation-level information
    # from the explanation output.
    metrics.insert(
        2,
        "isFraud",
        df["isFraud"].to_numpy(),
    )

    metrics.insert(
        3,
        "anomaly_score",
        df["anomaly_score"].to_numpy(),
    )

    # Preserve feature count at observation level
    # so temporal summaries can validate the
    # preprocessing schema.
    metrics["n_features"] = (
        len(feature_names)
    )

    # Preserve sample count.
    metrics["n_samples"] = (
        len(df)
    )

    print(
        "  Evaluation complete."
    )

    return metrics


def main():

    print("=" * 70)
    print(
        "BASELINE 1 — TEMPORAL EXIFFI "
        "EXPLANATION EVALUATION"
    )
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # 1. Evaluate all temporal windows
    # ---------------------------------------------------------

    window_metrics = []

    for window in range(1, 5):

        metrics = evaluate_window(
            window
        )

        window_metrics.append(
            metrics
        )

    # ---------------------------------------------------------
    # 2. Combine observation-level metrics
    # ---------------------------------------------------------

    print()
    print(
        "Combining temporal results..."
    )

    combined_metrics = pd.concat(
        window_metrics,
        ignore_index=True,
    )

    combined_metrics.to_csv(
        OUTPUT,
        index=False,
    )

    print()
    print(
        "Observation-level metrics saved to:"
    )

    print(
        OUTPUT
    )

    # ---------------------------------------------------------
    # 3. Overall temporal summary
    # ---------------------------------------------------------

    print()
    print(
        "Creating temporal summary..."
    )

    summary_df = build_summary(
        combined_metrics
    )

    summary_df.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    print()
    print(
        "Temporal summary saved to:"
    )

    print(
        SUMMARY_OUTPUT
    )

    print()
    print(
        "Temporal explanation summary:"
    )

    print()

    print(
        summary_df.to_string(
            index=False
        )
    )

    # ---------------------------------------------------------
    # 4. Class-stratified temporal summary
    # ---------------------------------------------------------

    print()
    print(
        "Creating class-stratified "
        "temporal summary..."
    )

    class_summary_df = (
        build_class_summary(
            combined_metrics
        )
    )

    class_summary_df.to_csv(
        CLASS_SUMMARY_OUTPUT,
        index=False,
    )

    print()
    print(
        "Class-stratified temporal summary saved to:"
    )

    print(
        CLASS_SUMMARY_OUTPUT
    )

    print()
    print(
        "Class-stratified temporal summary:"
    )

    print()

    print(
        class_summary_df.to_string(
            index=False
        )
    )

    # ---------------------------------------------------------
    # 5. Validation
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "TEMPORAL EVALUATION VALIDATION"
    )
    print("=" * 70)

    print(
        f"Windows evaluated: "
        f"{combined_metrics['window'].nunique()}"
    )

    print(
        f"Total observations: "
        f"{len(combined_metrics):,}"
    )

    print(
        "Feature counts:"
    )

    print(
        summary_df[
            [
                "window",
                "n_features",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "Class counts:"
    )

    print(
        summary_df[
            [
                "window",
                "n_fraud",
                "n_nonfraud",
                "fraud_rate",
            ]
        ].to_string(
            index=False
        )
    )

    # ---------------------------------------------------------
    # 6. Final validation checks
    # ---------------------------------------------------------

    expected_windows = {
        1,
        2,
        3,
        4,
    }

    actual_windows = set(
        combined_metrics[
            "window"
        ].unique()
    )

    if actual_windows != expected_windows:
        raise ValueError(
            "Unexpected temporal windows. "
            f"Expected {expected_windows}, "
            f"got {actual_windows}"
        )

    feature_counts = (
        summary_df[
            "n_features"
        ]
        .dropna()
        .astype(int)
        .unique()
    )

    if len(feature_counts) != 1:
        raise ValueError(
            "Feature count changed across "
            "temporal windows: "
            f"{feature_counts.tolist()}"
        )

    for window in expected_windows:

        row = summary_df.loc[
            summary_df["window"] == window
        ]

        if row.empty:
            raise ValueError(
                f"Missing summary for window "
                f"{window}"
            )

        n_observations = int(
            row["n_observations"].iloc[0]
        )

        n_fraud = int(
            row["n_fraud"].iloc[0]
        )

        n_nonfraud = int(
            row["n_nonfraud"].iloc[0]
        )

        if (
            n_fraud
            + n_nonfraud
            != n_observations
        ):
            raise ValueError(
                f"Class counts do not sum to "
                f"observations for window "
                f"{window}."
            )

    print()
    print(
        "Temporal ExIFFI evaluation complete."
    )


if __name__ == "__main__":
    main()