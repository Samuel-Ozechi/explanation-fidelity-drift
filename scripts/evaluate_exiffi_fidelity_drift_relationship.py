# scripts/evaluate_exiffi_fidelity_drift_relationship.py

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


# ============================================================
# Configuration
# ============================================================

DEFAULT_RESULTS_DIR = Path("results/tables")

FIDELITY_FILE = (
    "static_if_exiffi_fidelity_metric_sufficiency_summary.csv"
)

DRIFT_FILE = (
    "static_if_exiffi_temporal_drift_summary.csv"
)

CLASS_DRIFT_FILE = (
    "static_if_exiffi_temporal_class_drift_significance.csv"
)

FIDELITY_OBSERVATION_FILE = (
    "static_if_exiffi_fidelity_metric_sufficiency_observation.csv"
)

DEFAULT_BOOTSTRAP = 2000
DEFAULT_RANDOM_STATE = 42
DEFAULT_ALPHA = 0.05

DRIFT_METRICS = [
    "js_divergence",
    "cosine_distance",
    "l1_distance",
    "l2_distance",
    "spearman_rank_correlation",
]

FIDELITY_METRIC = (
    "exiffi_mean_relative_score_change"
)


# ============================================================
# Utility
# ============================================================


def safe_spearman(
    x: np.ndarray,
    y: np.ndarray,
) -> float:

    result = spearmanr(x, y)

    value = result.statistic

    if np.isnan(value):
        return 0.0

    return float(value)


def safe_pearson(
    x: np.ndarray,
    y: np.ndarray,
) -> float:

    if (
        np.std(x) == 0
        or np.std(y) == 0
    ):
        return 0.0

    result = pearsonr(x, y)

    value = result.statistic

    if np.isnan(value):
        return 0.0

    return float(value)


# ============================================================
# Bootstrap correlation
# ============================================================


def bootstrap_correlation(
    x: np.ndarray,
    y: np.ndarray,
    method: str,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> Tuple[
    float,
    float,
    float,
    float,
]:

    observed = (
        safe_spearman(x, y)
        if method == "spearman"
        else safe_pearson(x, y)
    )

    n = len(x)

    values = []

    for _ in range(n_bootstrap):

        indices = rng.integers(
            0,
            n,
            size=n,
        )

        xb = x[indices]
        yb = y[indices]

        if method == "spearman":
            value = safe_spearman(
                xb,
                yb,
            )
        else:
            value = safe_pearson(
                xb,
                yb,
            )

        values.append(value)

    values = np.asarray(values)

    lower = np.percentile(
        values,
        2.5,
    )

    upper = np.percentile(
        values,
        97.5,
    )

    # Two-sided bootstrap probability
    # that correlation crosses zero.
    if observed >= 0:

        p_value = (
            2.0
            * np.mean(
                values <= 0
            )
        )

    else:

        p_value = (
            2.0
            * np.mean(
                values >= 0
            )
        )

    p_value = min(
        max(float(p_value), 0.0),
        1.0,
    )

    return (
        float(observed),
        float(lower),
        float(upper),
        p_value,
    )


# ============================================================
# Load fidelity results
# ============================================================


def load_fidelity_summary(
    results_dir: Path,
) -> pd.DataFrame:

    path = (
        results_dir
        / FIDELITY_FILE
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Missing fidelity summary:\n{path}\n\n"
            "Run E4.2-T04 first."
        )

    df = pd.read_csv(path)

    required = [
        "window",
        "top_k",
        "mean_top_k_mass",
        FIDELITY_METRIC,
    ]

    missing = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing:

        raise ValueError(
            "Fidelity summary is missing "
            f"required columns: {missing}"
        )

    return df


# ============================================================
# Load temporal drift
# ============================================================


def load_drift_summary(
    results_dir: Path,
) -> pd.DataFrame:

    path = (
        results_dir
        / DRIFT_FILE
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Missing temporal drift summary:\n{path}\n\n"
            "Run the temporal ExIFFI drift analysis first."
        )

    df = pd.read_csv(path)

    required = [
        "comparison",
        "window_a",
        "window_b",
        *DRIFT_METRICS,
    ]

    missing = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing:

        raise ValueError(
            "Temporal drift summary is missing "
            f"required columns: {missing}"
        )

    return df


# ============================================================
# Load class drift
# ============================================================


def load_class_drift(
    results_dir: Path,
) -> pd.DataFrame:

    path = (
        results_dir
        / CLASS_DRIFT_FILE
    )

    if not path.exists():

        print(
            "\nWarning: class-specific drift "
            "file was not found."
        )

        return pd.DataFrame()

    df = pd.read_csv(path)

    return df


# ============================================================
# Build temporal fidelity values
# ============================================================


def build_temporal_fidelity(
    fidelity_df: pd.DataFrame,
) -> pd.DataFrame:

    # E4.2-T04 currently evaluates Window 1.
    #
    # Therefore we need to explicitly preserve the
    # evaluated window rather than silently pretending
    # that fidelity was measured for every window.

    fidelity = (
        fidelity_df[
            [
                "window",
                "top_k",
                "mean_top_k_mass",
                FIDELITY_METRIC,
            ]
        ]
        .copy()
    )

    fidelity = fidelity.rename(
        columns={
            FIDELITY_METRIC:
                "fidelity_score_change"
        }
    )

    return fidelity


# ============================================================
# Determine fidelity/drift pairing
# ============================================================


def build_pairwise_dataset(
    fidelity_df: pd.DataFrame,
    drift_df: pd.DataFrame,
) -> pd.DataFrame:

    fidelity = build_temporal_fidelity(
        fidelity_df
    )

    rows = []

    for _, drift_row in drift_df.iterrows():

        window_a = int(
            drift_row["window_a"]
        )

        window_b = int(
            drift_row["window_b"]
        )

        comparison = drift_row[
            "comparison"
        ]

        # Fidelity is currently available
        # for Window 1 only from E4.2-T04.
        #
        # For a temporal comparison, the most
        # conservative representation is to associate
        # the drift comparison with the fidelity measured
        # at its starting window.
        #
        # If both windows are available in a future
        # extension, the code below can use the mean
        # fidelity across the two windows.

        candidates = fidelity[
            fidelity["window"].isin(
                [window_a, window_b]
            )
        ]

        if candidates.empty:
            continue

        # Prefer fidelity at the start of the
        # temporal comparison.
        start_values = candidates[
            candidates["window"]
            == window_a
        ]

        if not start_values.empty:

            selected = start_values

        else:

            selected = candidates

        for _, fidelity_row in selected.iterrows():

            row = {
                "comparison": comparison,
                "window_a": window_a,
                "window_b": window_b,
                "top_k": int(
                    fidelity_row["top_k"]
                ),
                "mean_top_k_mass":
                    float(
                        fidelity_row[
                            "mean_top_k_mass"
                        ]
                    ),
                "fidelity_score_change":
                    float(
                        fidelity_row[
                            "fidelity_score_change"
                        ]
                    ),
            }

            for metric in DRIFT_METRICS:

                row[metric] = float(
                    drift_row[metric]
                )

            rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# Fidelity change by k
# ============================================================


def calculate_fidelity_k_relationship(
    fidelity_df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for window, group in (
        fidelity_df.groupby("window")
    ):

        group = (
            group
            .sort_values("top_k")
            .reset_index(drop=True)
        )

        if len(group) < 2:
            continue

        k = group[
            "top_k"
        ].to_numpy(dtype=float)

        fidelity = group[
            "fidelity_score_change"
        ].to_numpy(dtype=float)

        mass = group[
            "mean_top_k_mass"
        ].to_numpy(dtype=float)

        spearman_k = safe_spearman(
            k,
            fidelity,
        )

        spearman_mass = safe_spearman(
            mass,
            fidelity,
        )

        monotonic = bool(
            np.all(
                np.diff(fidelity)
                >= -1e-12
            )
        )

        rows.append(
            {
                "window": int(window),
                "n_k_values": len(group),
                "fidelity_monotonic_with_k":
                    monotonic,
                "fidelity_k_spearman":
                    spearman_k,
                "fidelity_mass_spearman":
                    spearman_mass,
                "minimum_fidelity":
                    float(
                        np.min(fidelity)
                    ),
                "maximum_fidelity":
                    float(
                        np.max(fidelity)
                    ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# Correlation analysis
# ============================================================


def evaluate_correlations(
    pairwise_df: pd.DataFrame,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    correlation_rows = []
    bootstrap_rows = []

    for top_k, group in (
        pairwise_df.groupby("top_k")
    ):

        if len(group) < 3:
            continue

        fidelity = group[
            "fidelity_score_change"
        ].to_numpy(dtype=float)

        for metric in DRIFT_METRICS:

            drift = group[
                metric
            ].to_numpy(dtype=float)

            pearson = safe_pearson(
                fidelity,
                drift,
            )

            spearman = safe_spearman(
                fidelity,
                drift,
            )

            correlation_rows.append(
                {
                    "top_k": int(top_k),
                    "drift_metric": metric,
                    "n_observations":
                        len(group),
                    "pearson_correlation":
                        pearson,
                    "spearman_correlation":
                        spearman,
                }
            )

            # Bootstrap Spearman.
            (
                observed,
                lower,
                upper,
                p_value,
            ) = bootstrap_correlation(
                x=fidelity,
                y=drift,
                method="spearman",
                n_bootstrap=n_bootstrap,
                rng=rng,
            )

            bootstrap_rows.append(
                {
                    "top_k": int(top_k),
                    "drift_metric": metric,
                    "method": "spearman",
                    "observed_correlation":
                        observed,
                    "bootstrap_ci_lower":
                        lower,
                    "bootstrap_ci_upper":
                        upper,
                    "bootstrap_pvalue":
                        p_value,
                    "n_observations":
                        len(group),
                    "n_bootstrap":
                        n_bootstrap,
                }
            )

            # Bootstrap Pearson.
            (
                observed,
                lower,
                upper,
                p_value,
            ) = bootstrap_correlation(
                x=fidelity,
                y=drift,
                method="pearson",
                n_bootstrap=n_bootstrap,
                rng=rng,
            )

            bootstrap_rows.append(
                {
                    "top_k": int(top_k),
                    "drift_metric": metric,
                    "method": "pearson",
                    "observed_correlation":
                        observed,
                    "bootstrap_ci_lower":
                        lower,
                    "bootstrap_ci_upper":
                        upper,
                    "bootstrap_pvalue":
                        p_value,
                    "n_observations":
                        len(group),
                    "n_bootstrap":
                        n_bootstrap,
                }
            )

    return (
        pd.DataFrame(correlation_rows),
        pd.DataFrame(bootstrap_rows),
    )


# ============================================================
# Decision logic
# ============================================================


def build_decision(
    pairwise_df: pd.DataFrame,
    bootstrap_df: pd.DataFrame,
    fidelity_k_df: pd.DataFrame,
) -> pd.DataFrame:

    if pairwise_df.empty:

        return pd.DataFrame(
            [
                {
                    "decision":
                        "INSUFFICIENT_TEMPORAL_FIDELITY_DATA",
                    "evidence":
                        "Fidelity has not yet been evaluated "
                        "for enough temporal windows to establish "
                        "a fidelity-drift relationship.",
                }
            ]
        )

    n_comparisons = (
        pairwise_df[
            [
                "comparison",
                "top_k",
            ]
        ]
        .drop_duplicates()
        .shape[0]
    )

    # A genuine temporal relationship requires
    # more than one independent temporal comparison.
    #
    # E4.2-T04 currently gives Window 1 fidelity only,
    # so this condition will normally fail at this stage.
    temporal_comparisons = (
        pairwise_df[
            "comparison"
        ].nunique()
    )

    if temporal_comparisons < 2:

        return pd.DataFrame(
            [
                {
                    "decision":
                        "TEMPORAL_FIDELITY_EXTENSION_REQUIRED",
                    "temporal_comparisons":
                        temporal_comparisons,
                    "n_pairwise_records":
                        len(pairwise_df),
                    "evidence":
                        (
                            "E4.2-T04 has validated the fidelity "
                            "metric, but fidelity has not yet been "
                            "measured independently across enough "
                            "temporal windows to test the "
                            "fidelity-drift relationship."
                        ),
                    "recommendation":
                        (
                            "Extend the validated fidelity "
                            "evaluation to Windows 2–4 before "
                            "drawing a temporal fidelity-drift "
                            "conclusion."
                        ),
                }
            ]
        )

    significant = 0
    total = 0

    for _, row in bootstrap_df.iterrows():

        if row[
            "method"
        ] != "spearman":

            continue

        total += 1

        if (
            row["bootstrap_ci_lower"] > 0
            or
            row["bootstrap_ci_upper"] < 0
        ):
            significant += 1

    monotonic = False

    if not fidelity_k_df.empty:

        monotonic = bool(
            fidelity_k_df[
                "fidelity_monotonic_with_k"
            ].all()
        )

    if (
        total > 0
        and significant > 0
        and monotonic
    ):

        decision = (
            "FIDELITY_DRIFT_RELATIONSHIP_SUPPORTED"
        )

        evidence = (
            "The validated fidelity metric demonstrates "
            "a statistically supported association with "
            "temporal explanation drift while preserving "
            "monotonic fidelity behaviour across tested "
            "Top-K values."
        )

    else:

        decision = (
            "FIDELITY_DRIFT_RELATIONSHIP_NOT_ESTABLISHED"
        )

        evidence = (
            "The available temporal evidence is insufficient "
            "to establish a statistically supported relationship "
            "between explanation fidelity and temporal drift."
        )

    return pd.DataFrame(
        [
            {
                "decision": decision,
                "temporal_comparisons":
                    temporal_comparisons,
                "n_pairwise_records":
                    len(pairwise_df),
                "significant_bootstrap_associations":
                    significant,
                "tested_bootstrap_associations":
                    total,
                "fidelity_monotonic_with_k":
                    monotonic,
                "evidence": evidence,
            }
        ]
    )


# ============================================================
# Main experiment
# ============================================================


def run_experiment(
    results_dir: Path = DEFAULT_RESULTS_DIR,
    n_bootstrap: int = DEFAULT_BOOTSTRAP,
    random_state: int = DEFAULT_RANDOM_STATE,
):

    print("=" * 70)
    print(
        "E4.2-T05 — EXIFFI FIDELITY–DRIFT RELATIONSHIP"
    )
    print("=" * 70)

    print(
        f"\nBootstrap samples: {n_bootstrap}"
    )

    print(
        f"Random state: {random_state}"
    )

    print(
        "\nPrimary fidelity metric:"
    )

    print(
        f"  {FIDELITY_METRIC}"
    )

    rng = np.random.default_rng(
        random_state
    )

    # ---------------------------------------------------------
    # 1. Load fidelity
    # ---------------------------------------------------------

    print(
        "\n[1/7] Loading validated fidelity results..."
    )

    fidelity_df = load_fidelity_summary(
        results_dir
    )

    print(
        f"Fidelity records: "
        f"{len(fidelity_df):,}"
    )

    print(
        "Fidelity windows:",
        sorted(
            fidelity_df[
                "window"
            ].unique()
            .tolist()
        ),
    )

    # ---------------------------------------------------------
    # 2. Load drift
    # ---------------------------------------------------------

    print(
        "\n[2/7] Loading temporal drift results..."
    )

    drift_df = load_drift_summary(
        results_dir
    )

    print(
        f"Temporal drift comparisons: "
        f"{len(drift_df):,}"
    )

    print(
        "Comparisons:",
        drift_df[
            "comparison"
        ].tolist(),
    )

    # ---------------------------------------------------------
    # 3. Build pairwise data
    # ---------------------------------------------------------

    print(
        "\n[3/7] Constructing fidelity–drift dataset..."
    )

    pairwise_df = build_pairwise_dataset(
        fidelity_df=fidelity_df,
        drift_df=drift_df,
    )

    if pairwise_df.empty:

        print(
            "No valid fidelity–drift pairs "
            "could be constructed."
        )

    else:

        print(
            f"Pairwise records: "
            f"{len(pairwise_df):,}"
        )

        print(
            "Top-k values:",
            sorted(
                pairwise_df[
                    "top_k"
                ].unique()
                .tolist()
            ),
        )

        print(
            "Temporal comparisons:",
            pairwise_df[
                "comparison"
            ].nunique(),
        )

    # ---------------------------------------------------------
    # 4. Correlations
    # ---------------------------------------------------------

    print(
        "\n[4/7] Evaluating fidelity–drift associations..."
    )

    (
        correlation_df,
        bootstrap_df,
    ) = evaluate_correlations(
        pairwise_df=pairwise_df,
        n_bootstrap=n_bootstrap,
        rng=rng,
    )

    # ---------------------------------------------------------
    # 5. Fidelity behaviour
    # ---------------------------------------------------------

    print(
        "\n[5/7] Evaluating fidelity behaviour..."
    )

    fidelity_k_df = (
        calculate_fidelity_k_relationship(
            fidelity_df
        )
    )

    # ---------------------------------------------------------
    # 6. Decision
    # ---------------------------------------------------------

    print(
        "\n[6/7] Assessing temporal evidence..."
    )

    decision_df = build_decision(
        pairwise_df=pairwise_df,
        bootstrap_df=bootstrap_df,
        fidelity_k_df=fidelity_k_df,
    )

    # ---------------------------------------------------------
    # 7. Save
    # ---------------------------------------------------------

    print(
        "\n[7/7] Saving fidelity–drift results..."
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pairwise_path = (
        results_dir
        / "static_if_exiffi_fidelity_drift_relationship.csv"
    )

    correlation_path = (
        results_dir
        / "static_if_exiffi_fidelity_drift_correlation.csv"
    )

    bootstrap_path = (
        results_dir
        / "static_if_exiffi_fidelity_drift_bootstrap.csv"
    )

    fidelity_k_path = (
        results_dir
        / "static_if_exiffi_fidelity_k_relationship.csv"
    )

    decision_path = (
        results_dir
        / "static_if_exiffi_fidelity_drift_decision.csv"
    )

    pairwise_df.to_csv(
        pairwise_path,
        index=False,
    )

    correlation_df.to_csv(
        correlation_path,
        index=False,
    )

    bootstrap_df.to_csv(
        bootstrap_path,
        index=False,
    )

    fidelity_k_df.to_csv(
        fidelity_k_path,
        index=False,
    )

    decision_df.to_csv(
        decision_path,
        index=False,
    )

    print(
        "\nResults saved:"
    )

    print(
        f"  Fidelity–drift relationship:\n"
        f"    {pairwise_path}"
    )

    print(
        f"  Correlations:\n"
        f"    {correlation_path}"
    )

    print(
        f"  Bootstrap:\n"
        f"    {bootstrap_path}"
    )

    print(
        f"  Fidelity–K relationship:\n"
        f"    {fidelity_k_path}"
    )

    print(
        f"  Decision:\n"
        f"    {decision_path}"
    )

    # ---------------------------------------------------------
    # Display
    # ---------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "E4.2-T05 — FIDELITY–DRIFT SUMMARY"
    )

    print(
        "=" * 70
    )

    if not correlation_df.empty:

        print(
            "\nCorrelation analysis:"
        )

        print(
            correlation_df.to_string(
                index=False
            )
        )

    else:

        print(
            "\nCorrelation analysis:"
        )

        print(
            "Not estimable with the current "
            "temporal fidelity coverage."
        )

    print(
        "\nFidelity behaviour:"
    )

    if fidelity_k_df.empty:

        print(
            "Not available."
        )

    else:

        print(
            fidelity_k_df.to_string(
                index=False
            )
        )

    print(
        "\nDecision:"
    )

    print(
        decision_df.to_string(
            index=False
        )
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "E4.2-T05 COMPLETE"
    )

    print(
        "=" * 70
    )

    return (
        pairwise_df,
        correlation_df,
        bootstrap_df,
        fidelity_k_df,
        decision_df,
    )


# ============================================================
# CLI
# ============================================================


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the relationship between "
            "ExIFFI explanation fidelity and "
            "temporal explanation drift."
        )
    )

    parser.add_argument(
        "--results-dir",
        default=str(
            DEFAULT_RESULTS_DIR
        ),
        help=(
            "Directory containing previously "
            "generated experimental results."
        ),
    )

    parser.add_argument(
        "--bootstrap",
        type=int,
        default=DEFAULT_BOOTSTRAP,
        help=(
            "Number of bootstrap samples."
        ),
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help=(
            "Random seed."
        ),
    )

    args = parser.parse_args()

    run_experiment(
        results_dir=Path(
            args.results_dir
        ),
        n_bootstrap=args.bootstrap,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()