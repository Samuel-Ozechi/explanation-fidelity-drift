from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


INPUT = Path(
    "results/tables/"
    "static_if_exiffi_vs_random_fidelity.csv"
)

OUTPUT = Path(
    "results/tables/"
    "static_if_exiffi_vs_random_significance.csv"
)

DETAIL_OUTPUT = Path(
    "results/tables/"
    "static_if_exiffi_vs_random_paired_differences.csv"
)


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute paired rank-based effect size.

    For paired observations, this is calculated from
    the signed differences.

    Interpretation:

        > 0  ExIFFI tends to produce larger fidelity
        < 0  Random tends to produce larger fidelity
         0  no systematic difference
    """

    differences = x - y

    differences = differences[
        differences != 0
    ]

    if len(differences) == 0:
        return 0.0

    n_positive = np.sum(
        differences > 0
    )

    n_negative = np.sum(
        differences < 0
    )

    return float(
        (n_positive - n_negative)
        / len(differences)
    )


def bootstrap_mean_difference(
    differences: np.ndarray,
    n_bootstrap: int = 10_000,
    random_state: int = 42,
) -> tuple[float, float]:

    rng = np.random.default_rng(
        random_state
    )

    n = len(differences)

    bootstrap_means = np.empty(
        n_bootstrap,
        dtype=float,
    )

    for i in range(n_bootstrap):

        sample = rng.choice(
            differences,
            size=n,
            replace=True,
        )

        bootstrap_means[i] = (
            np.mean(sample)
        )

    lower = float(
        np.percentile(
            bootstrap_means,
            2.5,
        )
    )

    upper = float(
        np.percentile(
            bootstrap_means,
            97.5,
        )
    )

    return lower, upper


def main():

    print("=" * 70)
    print(
        "EXIFFI VS RANDOM — "
        "PAIRED STATISTICAL SIGNIFICANCE"
    )
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Load results
    # ---------------------------------------------------------

    print(
        "\n[1/5] Loading fidelity comparison..."
    )

    df = pd.read_csv(
        INPUT
    )

    required_columns = {
        "condition",
        "sample_index",
        "top_k",
        "mean_score_change",
    }

    # The actual input is observation-level, so
    # verify the required fields available in it.
    required_observation_columns = {
        "condition",
        "sample_index",
        "top_k",
        "score_change",
        "relative_score_change",
    }

    missing = (
        required_observation_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Input file is missing required "
            f"columns: {sorted(missing)}"
        )

    print(
        f"Rows: {len(df):,}"
    )

    print(
        "Conditions:",
        sorted(df["condition"].unique()),
    )

    print(
        "Top-k values:",
        sorted(df["top_k"].unique()),
    )

    # ---------------------------------------------------------
    # 2. Construct paired observations
    # ---------------------------------------------------------

    print(
        "\n[2/5] Constructing paired comparisons..."
    )

    results = []
    detail_rows = []

    for top_k in sorted(
        df["top_k"].unique()
    ):

        subset = df[
            df["top_k"] == top_k
        ].copy()

        exiffi = subset[
            subset["condition"] == "exiffi"
        ][
            [
                "sample_index",
                "score_change",
                "relative_score_change",
            ]
        ].rename(
            columns={
                "score_change":
                    "exiffi_score_change",
                "relative_score_change":
                    "exiffi_relative_score_change",
            }
        )

        random = subset[
            subset["condition"] == "random"
        ][
            [
                "sample_index",
                "score_change",
                "relative_score_change",
            ]
        ].rename(
            columns={
                "score_change":
                    "random_score_change",
                "relative_score_change":
                    "random_relative_score_change",
            }
        )

        paired = exiffi.merge(
            random,
            on="sample_index",
            how="inner",
        )

        if paired.empty:
            raise ValueError(
                f"No paired observations for "
                f"top_k={top_k}"
            )

        paired[
            "score_change_difference"
        ] = (
            paired["exiffi_score_change"]
            - paired["random_score_change"]
        )

        paired[
            "relative_score_change_difference"
        ] = (
            paired[
                "exiffi_relative_score_change"
            ]
            - paired[
                "random_relative_score_change"
            ]
        )

        paired["top_k"] = top_k

        detail_rows.append(
            paired
        )

        # -----------------------------------------------------
        # Statistical test: raw score change
        # -----------------------------------------------------

        exiffi_score = (
            paired[
                "exiffi_score_change"
            ].to_numpy()
        )

        random_score = (
            paired[
                "random_score_change"
            ].to_numpy()
        )

        score_difference = (
            exiffi_score
            - random_score
        )

        # Wilcoxon signed-rank test.
        #
        # Alternative hypothesis:
        #
        # ExIFFI > Random
        score_test = wilcoxon(
            exiffi_score,
            random_score,
            alternative="greater",
            zero_method="wilcox",
        )

        score_ci_lower, score_ci_upper = (
            bootstrap_mean_difference(
                score_difference
            )
        )

        # -----------------------------------------------------
        # Statistical test: relative score change
        # -----------------------------------------------------

        exiffi_relative = (
            paired[
                "exiffi_relative_score_change"
            ].to_numpy()
        )

        random_relative = (
            paired[
                "random_relative_score_change"
            ].to_numpy()
        )

        relative_difference = (
            exiffi_relative
            - random_relative
        )

        relative_test = wilcoxon(
            exiffi_relative,
            random_relative,
            alternative="greater",
            zero_method="wilcox",
        )

        relative_ci_lower, relative_ci_upper = (
            bootstrap_mean_difference(
                relative_difference
            )
        )

        # -----------------------------------------------------
        # Effect sizes
        # -----------------------------------------------------

        score_effect = cliffs_delta(
            exiffi_score,
            random_score,
        )

        relative_effect = cliffs_delta(
            exiffi_relative,
            random_relative,
        )

        results.append(
            {
                "top_k": top_k,
                "n_observations": len(paired),

                # Raw score change
                "mean_exiffi_score_change":
                    np.mean(exiffi_score),

                "mean_random_score_change":
                    np.mean(random_score),

                "mean_score_difference":
                    np.mean(score_difference),

                "median_score_difference":
                    np.median(score_difference),

                "score_difference_ci_lower":
                    score_ci_lower,

                "score_difference_ci_upper":
                    score_ci_upper,

                "wilcoxon_score_statistic":
                    float(score_test.statistic),

                "wilcoxon_score_pvalue":
                    float(score_test.pvalue),

                "score_effect_size":
                    score_effect,

                # Relative score change
                "mean_exiffi_relative_change":
                    np.mean(exiffi_relative),

                "mean_random_relative_change":
                    np.mean(random_relative),

                "mean_relative_difference":
                    np.mean(relative_difference),

                "median_relative_difference":
                    np.median(relative_difference),

                "relative_difference_ci_lower":
                    relative_ci_lower,

                "relative_difference_ci_upper":
                    relative_ci_upper,

                "wilcoxon_relative_statistic":
                    float(
                        relative_test.statistic
                    ),

                "wilcoxon_relative_pvalue":
                    float(
                        relative_test.pvalue
                    ),

                "relative_effect_size":
                    relative_effect,

                # Directional evidence
                "n_exiffi_better":
                    int(
                        np.sum(
                            score_difference > 0
                        )
                    ),

                "n_random_better":
                    int(
                        np.sum(
                            score_difference < 0
                        )
                    ),

                "n_equal":
                    int(
                        np.sum(
                            score_difference == 0
                        )
                    ),
            }
        )

    # ---------------------------------------------------------
    # 3. Save paired differences
    # ---------------------------------------------------------

    print(
        "\n[3/5] Saving paired differences..."
    )

    detail_df = pd.concat(
        detail_rows,
        ignore_index=True,
    )

    DETAIL_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    detail_df.to_csv(
        DETAIL_OUTPUT,
        index=False,
    )

    print(
        f"Saved to:\n{DETAIL_OUTPUT}"
    )

    # ---------------------------------------------------------
    # 4. Save statistical summary
    # ---------------------------------------------------------

    print(
        "\n[4/5] Saving statistical summary..."
    )

    result_df = pd.DataFrame(
        results
    )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_df.to_csv(
        OUTPUT,
        index=False,
    )

    print(
        f"Saved to:\n{OUTPUT}"
    )

    # ---------------------------------------------------------
    # 5. Print results
    # ---------------------------------------------------------

    print(
        "\n[5/5] Statistical results:"
    )

    display_columns = [
        "top_k",
        "n_observations",
        "mean_score_difference",
        "score_difference_ci_lower",
        "score_difference_ci_upper",
        "wilcoxon_score_pvalue",
        "score_effect_size",
        "mean_relative_difference",
        "relative_difference_ci_lower",
        "relative_difference_ci_upper",
        "wilcoxon_relative_pvalue",
        "relative_effect_size",
        "n_exiffi_better",
        "n_random_better",
        "n_equal",
    ]

    print()

    print(
        result_df[
            display_columns
        ].to_string(
            index=False
        )
    )

    print(
        "\nDone."
    )


if __name__ == "__main__":
    main()