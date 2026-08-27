from __future__ import annotations

from pathlib import Path

import pandas as pd

from efd.evaluation.explanation_drift import (
    explanation_drift_summary,
    feature_drift_table,
    rank_correlation,
    top_k_jaccard,
)


INPUT_DIR = Path(
    "results/explanations"
)

OUTPUT_DIR = Path(
    "results/tables"
)

WINDOWS = [
    1,
    2,
    3,
    4,
]

TOP_K_VALUES = [
    1,
    5,
    10,
    20,
]


def load_window(
    window: int,
):

    path = (
        INPUT_DIR
        / f"static_if_exiffi_w{window}_sample.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing explanation file: {path}"
        )

    df = pd.read_csv(
        path
    )

    importance_columns = [
        column
        for column in df.columns
        if column.startswith(
            "importance__"
        )
    ]

    if not importance_columns:
        raise ValueError(
            f"No importance columns found in {path}"
        )

    importance = (
        df[importance_columns]
        .to_numpy(
            dtype=float
        )
    )

    feature_names = [
        column.replace(
            "importance__",
            "",
            1,
        )
        for column in importance_columns
    ]

    return (
        df,
        importance,
        feature_names,
    )


def main():

    print("=" * 70)
    print(
        "BASELINE 1 — "
        "TEMPORAL EXIFFI EXPLANATION DRIFT"
    )
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\n[1/5] Loading temporal explanations..."
    )

    windows = {}

    for window in WINDOWS:

        (
            df,
            importance,
            feature_names,
        ) = load_window(
            window
        )

        windows[window] = (
            df,
            importance,
            feature_names,
        )

        print(
            f"Window {window}: "
            f"{len(df):,} observations, "
            f"{importance.shape[1]:,} features"
        )

    print(
        "\n[2/5] Calculating window-to-window drift..."
    )

    summaries = []

    feature_tables = []

    stability_rows = []

    class_rows = []

    for window_a, window_b in zip(
        WINDOWS[:-1],
        WINDOWS[1:],
    ):

        (
            df_a,
            X_a,
            features_a,
        ) = windows[window_a]

        (
            df_b,
            X_b,
            features_b,
        ) = windows[window_b]

        comparison = (
            f"w{window_a}_w{window_b}"
        )

        print(
            f"\n  {comparison}"
        )

        if features_a != features_b:
            raise ValueError(
                "Feature schema mismatch "
                f"between W{window_a} and W{window_b}."
            )

        summary = (
            explanation_drift_summary(
                X_a,
                X_b,
                f"w{window_a}",
                f"w{window_b}",
            )
        )

        summary[
            "window_a"
        ] = window_a

        summary[
            "window_b"
        ] = window_b

        summary[
            "n_features"
        ] = X_a.shape[1]

        summary[
            "n_observations_a"
        ] = len(X_a)

        summary[
            "n_observations_b"
        ] = len(X_b)

        summaries.append(
            summary
        )

        feature_table = (
            feature_drift_table(
                X_a,
                X_b,
                features_a,
                f"w{window_a}",
                f"w{window_b}",
            )
        )

        feature_table.insert(
            0,
            "comparison",
            comparison,
        )

        feature_tables.append(
            feature_table
        )

        print(
            f"    JS divergence: "
            f"{summary['js_divergence']:.6f}"
        )

        print(
            f"    Cosine distance: "
            f"{summary['cosine_distance']:.6f}"
        )

        print(
            f"    Spearman correlation: "
            f"{summary['spearman_rank_correlation']:.6f}"
        )

        for k in TOP_K_VALUES:

            stability_rows.append(
                {
                    "comparison":
                        comparison,

                    "window_a":
                        window_a,

                    "window_b":
                        window_b,

                    "top_k":
                        k,

                    "jaccard":
                        top_k_jaccard(
                            X_a,
                            X_b,
                            k,
                        ),

                    "spearman_rank_correlation":
                        rank_correlation(
                            X_a,
                            X_b,
                        ),
                }
            )

        # -----------------------------------------------------
        # Class-specific drift
        # -----------------------------------------------------

        for fraud_value, class_name in [
            (1, "fraud"),
            (0, "nonfraud"),
        ]:

            if (
                "isFraud" not in df_a.columns
                or "isFraud" not in df_b.columns
            ):
                continue

            mask_a = (
                df_a["isFraud"]
                .to_numpy()
                == fraud_value
            )

            mask_b = (
                df_b["isFraud"]
                .to_numpy()
                == fraud_value
            )

            if not mask_a.any():
                continue

            if not mask_b.any():
                continue

            class_summary = (
                explanation_drift_summary(
                    X_a[mask_a],
                    X_b[mask_b],
                    f"w{window_a}_{class_name}",
                    f"w{window_b}_{class_name}",
                )
            )

            class_summary[
                "window_a"
            ] = window_a

            class_summary[
                "window_b"
            ] = window_b

            class_summary[
                "class"
            ] = class_name

            class_summary[
                "n_observations_a"
            ] = int(mask_a.sum())

            class_summary[
                "n_observations_b"
            ] = int(mask_b.sum())

            class_rows.append(
                class_summary
            )

    print(
        "\n[3/5] Saving overall drift results..."
    )

    summary_df = pd.DataFrame(
        summaries
    )

    summary_path = (
        OUTPUT_DIR
        / "static_if_exiffi_temporal_drift_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    print(
        f"Saved:\n{summary_path}"
    )

    print(
        "\n[4/5] Saving feature and Top-K drift..."
    )

    feature_drift_df = pd.concat(
        feature_tables,
        ignore_index=True,
    )

    feature_path = (
        OUTPUT_DIR
        / "static_if_exiffi_feature_drift.csv"
    )

    feature_drift_df.to_csv(
        feature_path,
        index=False,
    )

    stability_df = pd.DataFrame(
        stability_rows
    )

    stability_path = (
        OUTPUT_DIR
        / "static_if_exiffi_top_k_stability.csv"
    )

    stability_df.to_csv(
        stability_path,
        index=False,
    )

    class_df = pd.DataFrame(
        class_rows
    )

    class_path = (
        OUTPUT_DIR
        / "static_if_exiffi_temporal_class_drift.csv"
    )

    class_df.to_csv(
        class_path,
        index=False,
    )

    print(
        f"Saved:\n{feature_path}"
    )

    print(
        f"Saved:\n{stability_path}"
    )

    print(
        f"Saved:\n{class_path}"
    )

    print(
        "\n[5/5] Temporal drift summary:"
    )

    print()

    print(
        summary_df.to_string(
            index=False
        )
    )

    print(
        "\nTop-K stability:"
    )

    print()

    print(
        stability_df.to_string(
            index=False
        )
    )

    print(
        "\nClass-specific drift:"
    )

    print()

    print(
        class_df.to_string(
            index=False
        )
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "TEMPORAL EXIFFI "
        "DRIFT ANALYSIS COMPLETE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()