from pathlib import Path

import pandas as pd

from efd.evaluation.explanation import (
    evaluate_explanation_matrix,
    summarize_explanation_metrics,
)


EXPLANATION_DIR = Path(
    "results/explanations"
)

OUTPUT_DIR = Path(
    "results/tables"
)

N_WINDOWS = 4


def load_explanation_file(
    window: int,
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """
    Load one ExIFFI explanation file and extract
    the importance matrix.
    """

    input_path = (
        EXPLANATION_DIR
        / f"static_if_exiffi_w{window}_sample.csv"
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Explanation file not found:\n"
            f"{input_path}"
        )

    df = pd.read_csv(
        input_path
    )

    importance_cols = [
        column
        for column in df.columns
        if column.startswith("importance__")
    ]

    if not importance_cols:
        raise ValueError(
            f"No importance columns found in:\n"
            f"{input_path}"
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

    return (
        df,
        feature_names,
        importance,
    )


def evaluate_window(
    window: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate ExIFFI explanations for one temporal window.

    Returns
    -------
    metrics:
        Observation-level explanation metrics.

    summary_df:
        Window-level summary metrics.
    """

    print(
        f"\n{'-' * 70}"
    )

    print(
        f"WINDOW {window}"
    )

    print(
        f"{'-' * 70}"
    )

    print(
        "\nLoading explanation output..."
    )

    df, feature_names, importance = (
        load_explanation_file(
            window
        )
    )

    print(
        f"Observations: {len(df):,}"
    )

    print(
        f"Features: {len(feature_names):,}"
    )

    print(
        f"Importance matrix: "
        f"{importance.shape}"
    )

    print(
        "\nEvaluating explanations..."
    )

    metrics = evaluate_explanation_matrix(
        importance=importance,
        feature_names=feature_names,
        window=window,
    )

    # Preserve the original model information.
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

    output_path = (
        OUTPUT_DIR
        / f"static_if_exiffi_w{window}_"
          "explanation_metrics.csv"
    )

    metrics.to_csv(
        output_path,
        index=False,
    )

    print(
        f"\nObservation-level metrics saved to:"
        f"\n{output_path}"
    )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    summary = summarize_explanation_metrics(
        metrics
    )

    summary_df = (
        summary
        .to_frame()
        .T
    )

    summary_df.insert(
        0,
        "window",
        window,
    )

    summary_df.insert(
        1,
        "n_features",
        len(feature_names),
    )

    summary_df.insert(
        2,
        "n_observations",
        len(df),
    )

    summary_path = (
        OUTPUT_DIR
        / f"static_if_exiffi_w{window}_"
          "explanation_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    print(
        f"\nSummary saved to:"
        f"\n{summary_path}"
    )

    print(
        "\nWindow summary:"
    )

    print(
        summary_df.to_string(
            index=False
        )
    )

    return (
        metrics,
        summary_df,
    )


def main():

    print("=" * 70)
    print(
        "BASELINE 1 — EXIFFI "
        "TEMPORAL EXPLANATION EVALUATION"
    )
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_metrics = []
    all_summaries = []

    for window in range(
        1,
        N_WINDOWS + 1,
    ):

        metrics, summary = (
            evaluate_window(
                window
            )
        )

        all_metrics.append(
            metrics
        )

        all_summaries.append(
            summary
        )

    # ---------------------------------------------------------
    # Combined observation-level results
    # ---------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "CREATING COMBINED TEMPORAL RESULTS"
    )

    print(
        "=" * 70
    )

    combined_metrics = pd.concat(
        all_metrics,
        ignore_index=True,
    )

    combined_metrics_path = (
        OUTPUT_DIR
        / "static_if_exiffi_explanation_metrics.csv"
    )

    combined_metrics.to_csv(
        combined_metrics_path,
        index=False,
    )

    print(
        f"\nCombined observation-level metrics saved to:"
        f"\n{combined_metrics_path}"
    )

    # ---------------------------------------------------------
    # Combined summary
    # ---------------------------------------------------------

    combined_summary = pd.concat(
        all_summaries,
        ignore_index=True,
    )

    combined_summary_path = (
        OUTPUT_DIR
        / "static_if_exiffi_explanation_summary.csv"
    )

    combined_summary.to_csv(
        combined_summary_path,
        index=False,
    )

    print(
        f"\nCombined temporal summary saved to:"
        f"\n{combined_summary_path}"
    )

    # ---------------------------------------------------------
    # Display temporal comparison
    # ---------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "TEMPORAL EXPLANATION SUMMARY"
    )

    print(
        "=" * 70
    )

    print()

    print(
        combined_summary.to_string(
            index=False
        )
    )

    print(
        "\nDone."
    )


if __name__ == "__main__":
    main()