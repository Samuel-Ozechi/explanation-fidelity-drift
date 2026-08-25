from __future__ import annotations

from pathlib import Path

import pandas as pd

from efd.evaluation.explanation import (
    evaluate_explanation_matrix,
    summarize_explanation_metrics,
)


INPUT_DIR = Path(
    "results/explanations"
)

OUTPUT = Path(
    "results/tables/"
    "static_if_exiffi_temporal_explanation_metrics.csv"
)

SUMMARY_OUTPUT = Path(
    "results/tables/"
    "static_if_exiffi_temporal_explanation_summary.csv"
)

N_WINDOWS = 4


def get_importance_columns(
    df: pd.DataFrame,
) -> list[str]:
    """Return ExIFFI importance columns."""

    columns = [
        column
        for column in df.columns
        if column.startswith("importance__")
    ]

    if not columns:
        raise ValueError(
            "No importance__ columns found."
        )

    return columns


def evaluate_window(
    window: int,
) -> pd.DataFrame:

    input_path = (
        INPUT_DIR
        / f"static_if_exiffi_w{window}_sample.csv"
    )

    print()
    print("=" * 70)
    print(f"Window {window}")
    print("=" * 70)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing explanation file:\n{input_path}"
        )

    df = pd.read_csv(
        input_path
    )

    importance_cols = get_importance_columns(
        df
    )

    feature_names = [
        column.replace(
            "importance__",
            "",
            1,
        )
        for column in importance_cols
    ]

    importance = (
        df[importance_cols]
        .to_numpy(dtype=float)
    )

    print(
        f"  Observations: {len(df)}"
    )

    print(
        f"  Features: {len(feature_names)}"
    )

    if "isFraud" in df.columns:
        print(
            f"  Fraud: "
            f"{int(df['isFraud'].sum())}"
        )

    metrics = evaluate_explanation_matrix(
        importance=importance,
        feature_names=feature_names,
        window=window,
    )

    # ---------------------------------------------------------
    # Preserve source observation information
    # ---------------------------------------------------------

    if "isFraud" in df.columns:
        metrics.insert(
            2,
            "isFraud",
            df["isFraud"].to_numpy(),
        )

    if "anomaly_score" in df.columns:
        insert_position = (
            3
            if "isFraud" in metrics.columns
            else 2
        )

        metrics.insert(
            insert_position,
            "anomaly_score",
            df["anomaly_score"].to_numpy(),
        )

    # Explicitly record feature count.
    #
    # This is useful for temporal comparisons and prevents
    # the summary from having to infer it later.
    metrics["n_features"] = len(
        feature_names
    )

    print(
        "  Evaluation complete."
    )

    return metrics


def build_summary(
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build one row per temporal window.

    Feature count is obtained from the metrics dataframe
    when available, otherwise from the number of
    importance columns.
    """

    rows = []

    for window, group in metrics.groupby(
        "window",
        sort=True,
    ):

        summary = summarize_explanation_metrics(
            group
        )

        row = (
            summary
            .to_dict()
        )

        row["window"] = int(
            window
        )

        row["n_observations"] = int(
            len(group)
        )

        # -----------------------------------------------------
        # FIX:
        # Do not assume n_features exists.
        # Derive it safely.
        # -----------------------------------------------------

        if "n_features" in group.columns:

            row["n_features"] = int(
                group["n_features"].iloc[0]
            )

        else:

            importance_cols = [
                column
                for column in group.columns
                if column.startswith(
                    "importance__"
                )
            ]

            row["n_features"] = int(
                len(importance_cols)
            )

        if "isFraud" in group.columns:

            row["n_fraud"] = int(
                group["isFraud"].sum()
            )

            row["fraud_rate"] = float(
                group["isFraud"].mean()
            )

        rows.append(
            row
        )

    summary_df = pd.DataFrame(
        rows
    )

    # Put the important experiment dimensions first.
    preferred_columns = [
        "window",
        "n_features",
        "n_observations",
        "n_fraud",
        "fraud_rate",
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

    existing_preferred = [
        column
        for column in preferred_columns
        if column in summary_df.columns
    ]

    remaining = [
        column
        for column in summary_df.columns
        if column not in existing_preferred
    ]

    summary_df = summary_df[
        existing_preferred + remaining
    ]

    return summary_df


def main():

    print("=" * 70)
    print(
        "BASELINE 1 — TEMPORAL EXIFFI "
        "EXPLANATION EVALUATION"
    )
    print("=" * 70)

    all_metrics = []

    # ---------------------------------------------------------
    # 1. Evaluate every temporal window
    # ---------------------------------------------------------

    for window in range(
        1,
        N_WINDOWS + 1,
    ):

        metrics = evaluate_window(
            window
        )

        all_metrics.append(
            metrics
        )

    # ---------------------------------------------------------
    # 2. Combine observation-level results
    # ---------------------------------------------------------

    print()
    print(
        "Combining temporal results..."
    )

    metrics_df = pd.concat(
        all_metrics,
        ignore_index=True,
    )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    metrics_df.to_csv(
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
    # 3. Build temporal summary
    # ---------------------------------------------------------

    print()
    print(
        "Creating temporal summary..."
    )

    summary_df = build_summary(
        metrics_df
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

    # ---------------------------------------------------------
    # 4. Display summary
    # ---------------------------------------------------------

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
    # 5. Basic validation
    # ---------------------------------------------------------

    print()
    print(
        "=" * 70
    )
    print(
        "TEMPORAL EVALUATION VALIDATION"
    )
    print(
        "=" * 70
    )

    print(
        f"Windows evaluated: "
        f"{metrics_df['window'].nunique()}"
    )

    print(
        f"Total observations: "
        f"{len(metrics_df):,}"
    )

    if "n_features" in metrics_df.columns:

        print(
            "Feature counts:"
        )

        print(
            metrics_df[
                [
                    "window",
                    "n_features",
                ]
            ]
            .drop_duplicates()
            .sort_values("window")
            .to_string(index=False)
        )

    print()
    print(
        "Temporal ExIFFI evaluation complete."
    )


if __name__ == "__main__":
    main()