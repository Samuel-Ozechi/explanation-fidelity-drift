from __future__ import annotations

from pathlib import Path

import pandas as pd

from efd.evaluation.explanation_drift import (
    evaluate_pairwise_drift,
    feature_level_drift,
)


INPUT_DIR = Path(
    "results/explanations"
)

OUTPUT_DIR = Path(
    "results/tables"
)


WINDOWS = [1, 2, 3, 4]


def load_explanation(
    window: int,
) -> tuple[pd.DataFrame, list[str]]:

    path = (
        INPUT_DIR
        / f"static_if_exiffi_w{window}_sample.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Explanation file not found: {path}"
        )

    df = pd.read_csv(path)

    importance_cols = [
        column
        for column in df.columns
        if column.startswith("importance__")
    ]

    if not importance_cols:
        raise ValueError(
            f"No importance columns found in {path}"
        )

    feature_names = [
        column.replace(
            "importance__",
            "",
            1,
        )
        for column in importance_cols
    ]

    return df, feature_names


def main():

    print("=" * 70)
    print(
        "BASELINE 1 — TEMPORAL EXIFFI "
        "EXPLANATION DRIFT"
    )
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    windows = {}

    print("\n[1/5] Loading temporal explanations...")

    for window in WINDOWS:

        df, feature_names = (
            load_explanation(window)
        )

        importance_cols = [
            f"importance__{feature}"
            for feature in feature_names
        ]

        importance = (
            df[importance_cols]
            .to_numpy(dtype=float)
        )

        windows[window] = {
            "df": df,
            "features": feature_names,
            "importance": importance,
        }

        print(
            f"Window {window}: "
            f"{len(df):,} observations, "
            f"{len(feature_names):,} features"
        )

    # ---------------------------------------------------------
    # Pairwise temporal drift
    # ---------------------------------------------------------

    print(
        "\n[2/5] Evaluating pairwise explanation drift..."
    )

    pairwise_rows = []

    comparisons = [
        (1, 2),
        (2, 3),
        (3, 4),
        (1, 3),
        (2, 4),
        (1, 4),
    ]

    for window_a, window_b in comparisons:

        result = evaluate_pairwise_drift(
            importance_a=windows[window_a][
                "importance"
            ],
            importance_b=windows[window_b][
                "importance"
            ],
            window_a=window_a,
            window_b=window_b,
        )

        pairwise_rows.append(
            {
                "window_a": result.window_a,
                "window_b": result.window_b,
                "js_divergence": result.js_divergence,
                "total_variation": result.total_variation,
                "cosine_distance": result.cosine_distance,
                "cosine_similarity": result.cosine_similarity,
                "spearman_correlation": result.spearman_correlation,
                "top_1_overlap": result.top_1_overlap,
                "top_5_overlap": result.top_5_overlap,
                "top_10_overlap": result.top_10_overlap,
                "top_20_overlap": result.top_20_overlap,
                "top_1_jaccard": result.top_1_jaccard,
                "top_5_jaccard": result.top_5_jaccard,
                "top_10_jaccard": result.top_10_jaccard,
                "top_20_jaccard": result.top_20_jaccard,
            }
        )

        print(
            f"Window {window_a} → {window_b}: "
            f"JS={result.js_divergence:.6f}, "
            f"TV={result.total_variation:.6f}, "
            f"Cosine={result.cosine_similarity:.6f}, "
            f"Top-10 Jaccard="
            f"{result.top_10_jaccard:.4f}"
        )

    pairwise_df = pd.DataFrame(
        pairwise_rows
    )

    pairwise_path = (
        OUTPUT_DIR
        / "static_if_exiffi_temporal_drift.csv"
    )

    pairwise_df.to_csv(
        pairwise_path,
        index=False,
    )

    # ---------------------------------------------------------
    # Feature-level drift
    # ---------------------------------------------------------

    print(
        "\n[3/5] Evaluating feature-level drift..."
    )

    feature_rows = []

    for window_a, window_b in comparisons:

        drift = feature_level_drift(
            importance_a=windows[window_a][
                "importance"
            ],
            importance_b=windows[window_b][
                "importance"
            ],
            feature_names=windows[window_a][
                "features"
            ],
            window_a=window_a,
            window_b=window_b,
        )

        feature_rows.append(drift)

    feature_df = pd.concat(
        feature_rows,
        ignore_index=True,
    )

    feature_path = (
        OUTPUT_DIR
        / "static_if_exiffi_temporal_feature_drift.csv"
    )

    feature_df.to_csv(
        feature_path,
        index=False,
    )

    # ---------------------------------------------------------
    # Top feature stability
    # ---------------------------------------------------------

    print(
        "\n[4/5] Creating top-feature stability..."
    )

    stability_rows = []

    for _, row in pairwise_df.iterrows():

        stability_rows.append(
            {
                "window_a": int(
                    row["window_a"]
                ),
                "window_b": int(
                    row["window_b"]
                ),
                "top_1_overlap": row[
                    "top_1_overlap"
                ],
                "top_5_overlap": row[
                    "top_5_overlap"
                ],
                "top_10_overlap": row[
                    "top_10_overlap"
                ],
                "top_20_overlap": row[
                    "top_20_overlap"
                ],
                "top_1_jaccard": row[
                    "top_1_jaccard"
                ],
                "top_5_jaccard": row[
                    "top_5_jaccard"
                ],
                "top_10_jaccard": row[
                    "top_10_jaccard"
                ],
                "top_20_jaccard": row[
                    "top_20_jaccard"
                ],
            }
        )

    stability_df = pd.DataFrame(
        stability_rows
    )

    stability_path = (
        OUTPUT_DIR
        / "static_if_exiffi_temporal_topk_stability.csv"
    )

    stability_df.to_csv(
        stability_path,
        index=False,
    )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    print(
        "\n[5/5] Creating drift summary..."
    )

    summary_rows = []

    for window in WINDOWS:

        df = windows[window]["df"]
        importance = windows[window]["importance"]

        mean_importance = importance.mean(
            axis=0
        )

        total = mean_importance.sum()

        if total > 0:
            distribution = (
                mean_importance / total
            )
        else:
            distribution = mean_importance

        top_1 = pd.Series(
            distribution
        ).nlargest(1).sum()

        top_5 = pd.Series(
            distribution
        ).nlargest(5).sum()

        top_10 = pd.Series(
            distribution
        ).nlargest(10).sum()

        entropy = -(
            distribution[
                distribution > 0
            ]
            * __import__("numpy").log(
                distribution[
                    distribution > 0
                ]
            )
        ).sum()

        effective_features = (
            __import__("numpy").exp(entropy)
        )

        summary_rows.append(
            {
                "window": window,
                "n_observations": len(df),
                "n_features": importance.shape[1],
                "n_fraud": int(
                    df["isFraud"].sum()
                ),
                "fraud_rate": float(
                    df["isFraud"].mean()
                ),
                "top_1_mass": top_1,
                "top_5_mass": top_5,
                "top_10_mass": top_10,
                "entropy": entropy,
                "effective_features": effective_features,
            }
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_path = (
        OUTPUT_DIR
        / "static_if_exiffi_temporal_drift_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    print("\nResults saved:")
    print(pairwise_path)
    print(feature_path)
    print(stability_path)
    print(summary_path)

    print(
        "\nPairwise temporal explanation drift:"
    )

    print(
        pairwise_df.to_string(
            index=False
        )
    )

    print(
        "\nTemporal explanation concentration:"
    )

    print(
        summary_df.to_string(
            index=False
        )
    )

    print(
        "\nTemporal ExIFFI explanation drift "
        "analysis complete."
    )


if __name__ == "__main__":
    main()