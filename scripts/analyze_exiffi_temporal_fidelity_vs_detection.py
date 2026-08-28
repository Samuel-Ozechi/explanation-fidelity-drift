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

DEFAULT_OUTPUT_DIR = Path("results/tables")

DEFAULT_FIDELITY_FILE = (
    DEFAULT_OUTPUT_DIR
    / "static_if_exiffi_temporal_fidelity_trajectory.csv"
)

DEFAULT_CLASS_FIDELITY_FILE = (
    DEFAULT_OUTPUT_DIR
    / "static_if_exiffi_temporal_fidelity_class_trajectory.csv"
)

DEFAULT_DETECTION_FILE = (
    DEFAULT_OUTPUT_DIR
    / "static_if_temporal_model_performance.csv"
)

DEFAULT_ALPHA = 0.05

WINDOWS = [1, 2, 3, 4]

TOP_K_VALUES = [1, 3, 5, 10, 20]


# ============================================================
# Detection trajectory
# ============================================================

# These are the values already established by the baseline
# experiment and documented in the research plan.
#
# They are used ONLY as a fallback when a detection CSV is not
# supplied.
#
# Once the canonical baseline CSV exists, pass it explicitly
# with --detection-file.

DEFAULT_DETECTION_RESULTS = pd.DataFrame(
    {
        "window": [1, 2, 3, 4],
        "f1": [
            0.269652,
            0.238556,
            0.219187,
            0.203904,
        ],
        "pr_auc": [
            0.214765,
            0.213656,
            0.139977,
            0.114344,
        ],
        "fraud_rate": [
            0.043181,
            0.034900,
            0.031344,
            0.037474,
        ],
    }
)


# ============================================================
# Utility
# ============================================================


def safe_relative_change(
    current: float,
    baseline: float,
) -> float:

    if abs(baseline) < 1e-12:
        return np.nan

    return (
        current - baseline
    ) / abs(baseline)


def safe_percentage_change(
    current: float,
    baseline: float,
) -> float:

    value = safe_relative_change(
        current,
        baseline,
    )

    if np.isnan(value):
        return np.nan

    return value * 100.0


def minmax_normalize(
    values: pd.Series,
    higher_is_worse: bool,
) -> pd.Series:

    values = values.astype(float)

    minimum = values.min()
    maximum = values.max()

    if np.isclose(
        minimum,
        maximum,
    ):
        result = pd.Series(
            np.zeros(
                len(values)
            ),
            index=values.index,
        )
    else:
        result = (
            values - minimum
        ) / (
            maximum - minimum
        )

    if higher_is_worse:
        return result

    return 1.0 - result


# ============================================================
# Load fidelity trajectory
# ============================================================


def load_fidelity_trajectory(
    path: Path,
) -> pd.DataFrame:

    if not path.exists():

        raise FileNotFoundError(
            "\n"
            "Temporal fidelity trajectory "
            "was not found:\n"
            f"{path}\n\n"
            "Run the temporal fidelity "
            "evaluation first."
        )

    df = pd.read_csv(path)

    required = {
        "window",
        "top_k",
        "mean_relative_score_change",
        "window_to_window_change",
        "change_from_window_1",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            "Temporal fidelity trajectory "
            "is missing required columns: "
            f"{sorted(missing)}"
        )

    df["window"] = (
        df["window"]
        .astype(int)
    )

    df["top_k"] = (
        df["top_k"]
        .astype(int)
    )

    return (
        df
        .sort_values(
            [
                "top_k",
                "window",
            ]
        )
        .reset_index(drop=True)
    )


# ============================================================
# Load class-specific fidelity
# ============================================================


def load_class_fidelity(
    path: Path,
) -> pd.DataFrame | None:

    if not path.exists():
        return None

    df = pd.read_csv(path)

    required = {
        "window",
        "isFraud",
        "class",
        "top_k",
        "mean_relative_score_change",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            "Class fidelity trajectory "
            "is missing required columns: "
            f"{sorted(missing)}"
        )

    df["window"] = (
        df["window"]
        .astype(int)
    )

    df["top_k"] = (
        df["top_k"]
        .astype(int)
    )

    df["isFraud"] = (
        df["isFraud"]
        .astype(int)
    )

    return (
        df
        .sort_values(
            [
                "class",
                "top_k",
                "window",
            ]
        )
        .reset_index(drop=True)
    )


# ============================================================
# Detection metrics
# ============================================================


def load_detection_results(
    path: Path | None,
) -> pd.DataFrame:

    if path is None:

        print(
            "\nNo detection trajectory CSV "
            "supplied."
        )

        print(
            "Using the documented baseline "
            "detection trajectory."
        )

        return (
            DEFAULT_DETECTION_RESULTS
            .copy()
        )

    if not path.exists():

        raise FileNotFoundError(
            "Detection results file not found:\n"
            f"{path}"
        )

    df = pd.read_csv(path)

    # --------------------------------------------------------
    # Flexible column recognition.
    # --------------------------------------------------------

    aliases = {
        "window": [
            "window",
            "Window",
        ],
        "f1": [
            "f1",
            "F1",
            "f1_score",
            "F1_score",
        ],
        "pr_auc": [
            "pr_auc",
            "PR_AUC",
            "pr-auc",
            "PR-AUC",
            "average_precision",
        ],
        "fraud_rate": [
            "fraud_rate",
            "Fraud rate",
            "fraudRate",
        ],
    }

    rename_map = {}

    for canonical, candidates in aliases.items():

        for candidate in candidates:

            if candidate in df.columns:

                rename_map[
                    candidate
                ] = canonical

                break

    df = df.rename(
        columns=rename_map
    )

    required = {
        "window",
        "f1",
        "pr_auc",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            "Detection results are missing "
            "required columns: "
            f"{sorted(missing)}"
        )

    df["window"] = (
        df["window"]
        .astype(int)
    )

    return (
        df
        .sort_values("window")
        .reset_index(drop=True)
    )


# ============================================================
# Detection trajectory analysis
# ============================================================


def calculate_detection_trajectory(
    detection_df: pd.DataFrame,
) -> pd.DataFrame:

    df = (
        detection_df
        .copy()
        .sort_values("window")
        .reset_index(drop=True)
    )

    baseline_f1 = float(
        df.iloc[0]["f1"]
    )

    baseline_pr_auc = float(
        df.iloc[0]["pr_auc"]
    )

    previous_f1 = None
    previous_pr_auc = None

    rows = []

    for _, row in df.iterrows():

        window = int(
            row["window"]
        )

        f1 = float(
            row["f1"]
        )

        pr_auc = float(
            row["pr_auc"]
        )

        if previous_f1 is None:

            f1_window_change = 0.0
            pr_auc_window_change = 0.0

        else:

            f1_window_change = (
                f1
                - previous_f1
            )

            pr_auc_window_change = (
                pr_auc
                - previous_pr_auc
            )

        rows.append(
            {
                "window": window,

                "f1": f1,

                "pr_auc": pr_auc,

                "f1_change_from_window_1": (
                    f1
                    - baseline_f1
                ),

                "pr_auc_change_from_window_1": (
                    pr_auc
                    - baseline_pr_auc
                ),

                "f1_relative_change_pct": (
                    safe_percentage_change(
                        f1,
                        baseline_f1,
                    )
                ),

                "pr_auc_relative_change_pct": (
                    safe_percentage_change(
                        pr_auc,
                        baseline_pr_auc,
                    )
                ),

                "f1_window_to_window_change": (
                    f1_window_change
                ),

                "pr_auc_window_to_window_change": (
                    pr_auc_window_change
                ),

                "f1_relative_degradation_pct": (
                    -safe_percentage_change(
                        f1,
                        baseline_f1,
                    )
                ),

                "pr_auc_relative_degradation_pct": (
                    -safe_percentage_change(
                        pr_auc,
                        baseline_pr_auc,
                    )
                ),
            }
        )

        previous_f1 = f1
        previous_pr_auc = pr_auc

    return pd.DataFrame(rows)


# ============================================================
# Fidelity trajectory analysis
# ============================================================


def calculate_fidelity_trajectory(
    fidelity_df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for top_k, group in fidelity_df.groupby(
        "top_k"
    ):

        group = (
            group
            .sort_values("window")
            .reset_index(drop=True)
        )

        baseline = float(
            group.iloc[0][
                "mean_relative_score_change"
            ]
        )

        previous = None

        for _, row in group.iterrows():

            window = int(
                row["window"]
            )

            fidelity = float(
                row[
                    "mean_relative_score_change"
                ]
            )

            if previous is None:

                window_change = 0.0

            else:

                window_change = (
                    fidelity
                    - previous
                )

            rows.append(
                {
                    "window": window,

                    "top_k": int(
                        top_k
                    ),

                    "fidelity": fidelity,

                    "fidelity_change_from_window_1": (
                        fidelity
                        - baseline
                    ),

                    "fidelity_relative_change_pct": (
                        safe_percentage_change(
                            fidelity,
                            baseline,
                        )
                    ),

                    "fidelity_window_to_window_change": (
                        window_change
                    ),
                }
            )

            previous = fidelity

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "top_k",
                "window",
            ]
        )
        .reset_index(drop=True)
    )


# ============================================================
# Merge detection + fidelity
# ============================================================


def build_integrated_trajectory(
    fidelity_trajectory: pd.DataFrame,
    detection_trajectory: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for top_k, fidelity_group in (
        fidelity_trajectory.groupby(
            "top_k"
        )
    ):

        merged = fidelity_group.merge(
            detection_trajectory,
            on="window",
            how="left",
            validate="one_to_one",
        )

        merged[
            "top_k"
        ] = int(top_k)

        rows.append(
            merged
        )

    if not rows:

        return pd.DataFrame()

    return (
        pd.concat(
            rows,
            ignore_index=True,
        )
        .sort_values(
            [
                "top_k",
                "window",
            ]
        )
        .reset_index(drop=True)
    )


# ============================================================
# Dissociation metrics
# ============================================================


def calculate_dissociation(
    integrated_df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for top_k, group in integrated_df.groupby(
        "top_k"
    ):

        group = (
            group
            .sort_values("window")
            .reset_index(drop=True)
        )

        for _, row in group.iterrows():

            # Detection degradation is positive
            # when F1/PR-AUC decrease.
            f1_degradation = float(
                row[
                    "f1_relative_degradation_pct"
                ]
            )

            pr_auc_degradation = float(
                row[
                    "pr_auc_relative_degradation_pct"
                ]
            )

            # Fidelity loss is represented as
            # negative fidelity change from
            # Window 1.
            fidelity_change = float(
                row[
                    "fidelity_relative_change_pct"
                ]
            )

            fidelity_degradation = (
                -fidelity_change
            )

            # ------------------------------------------------
            # Dissociation:
            #
            # positive:
            # detection deteriorates more than fidelity
            #
            # negative:
            # fidelity deteriorates more than detection
            # ------------------------------------------------

            dissociation_f1 = (
                fidelity_degradation
                - f1_degradation
            )

            dissociation_pr_auc = (
                fidelity_degradation
                - pr_auc_degradation
            )

            rows.append(
                {
                    "window": int(
                        row["window"]
                    ),

                    "top_k": int(
                        top_k
                    ),

                    "fidelity_degradation_pct": (
                        fidelity_degradation
                    ),

                    "f1_degradation_pct": (
                        f1_degradation
                    ),

                    "pr_auc_degradation_pct": (
                        pr_auc_degradation
                    ),

                    "fidelity_minus_f1_degradation": (
                        dissociation_f1
                    ),

                    "fidelity_minus_pr_auc_degradation": (
                        dissociation_pr_auc
                    ),

                    "absolute_f1_fidelity_dissociation": (
                        abs(
                            dissociation_f1
                        )
                    ),

                    "absolute_pr_auc_fidelity_dissociation": (
                        abs(
                            dissociation_pr_auc
                        )
                    ),
                }
            )

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "top_k",
                "window",
            ]
        )
        .reset_index(drop=True)
    )


# ============================================================
# Fidelity Drift Index
# ============================================================


def calculate_fidelity_drift_index(
    fidelity_trajectory: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for top_k, group in fidelity_trajectory.groupby(
        "top_k"
    ):

        group = (
            group
            .sort_values("window")
            .reset_index(drop=True)
        )

        baseline = float(
            group.iloc[0][
                "fidelity"
            ]
        )

        for _, row in group.iterrows():

            current = float(
                row["fidelity"]
            )

            # ------------------------------------------------
            # The fidelity measure is score retention.
            #
            # Lower retention means greater
            # explanation-fidelity degradation.
            # ------------------------------------------------

            if abs(baseline) < 1e-12:

                relative_degradation = np.nan

            else:

                relative_degradation = (
                    (
                        baseline
                        - current
                    )
                    / abs(baseline)
                )

            rows.append(
                {
                    "window": int(
                        row["window"]
                    ),

                    "top_k": int(
                        top_k
                    ),

                    "fidelity_drift_index": (
                        relative_degradation
                    ),

                    "fidelity_drift_index_pct": (
                        relative_degradation
                        * 100.0
                        if not np.isnan(
                            relative_degradation
                        )
                        else np.nan
                    ),
                }
            )

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "top_k",
                "window",
            ]
        )
        .reset_index(drop=True)
    )


# ============================================================
# Temporal slopes
# ============================================================


def calculate_temporal_slopes(
    integrated_df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for top_k, group in integrated_df.groupby(
        "top_k"
    ):

        group = (
            group
            .sort_values("window")
            .reset_index(drop=True)
        )

        x = group[
            "window"
        ].to_numpy(
            dtype=float
        )

        y_fidelity = group[
            "fidelity"
        ].to_numpy(
            dtype=float
        )

        y_f1 = group[
            "f1"
        ].to_numpy(
            dtype=float
        )

        y_pr_auc = group[
            "pr_auc"
        ].to_numpy(
            dtype=float
        )

        fidelity_slope = np.polyfit(
            x,
            y_fidelity,
            1,
        )[0]

        f1_slope = np.polyfit(
            x,
            y_f1,
            1,
        )[0]

        pr_auc_slope = np.polyfit(
            x,
            y_pr_auc,
            1,
        )[0]

        rows.append(
            {
                "top_k": int(
                    top_k
                ),

                "fidelity_temporal_slope": (
                    fidelity_slope
                ),

                "f1_temporal_slope": (
                    f1_slope
                ),

                "pr_auc_temporal_slope": (
                    pr_auc_slope
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# Correlation analysis
# ============================================================


def calculate_correlations(
    integrated_df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for top_k, group in integrated_df.groupby(
        "top_k"
    ):

        group = (
            group
            .sort_values("window")
            .reset_index(drop=True)
        )

        fidelity = group[
            "fidelity"
        ].to_numpy(
            dtype=float
        )

        f1 = group[
            "f1"
        ].to_numpy(
            dtype=float
        )

        pr_auc = group[
            "pr_auc"
        ].to_numpy(
            dtype=float
        )

        # ----------------------------------------------------
        # With only four temporal windows, correlation
        # coefficients are descriptive rather than
        # definitive inferential evidence.
        # ----------------------------------------------------

        if len(fidelity) >= 3:

            pearson_f1, pearson_f1_p = (
                pearsonr(
                    fidelity,
                    f1,
                )
            )

            spearman_f1, spearman_f1_p = (
                spearmanr(
                    fidelity,
                    f1,
                )
            )

            pearson_pr, pearson_pr_p = (
                pearsonr(
                    fidelity,
                    pr_auc,
                )
            )

            spearman_pr, spearman_pr_p = (
                spearmanr(
                    fidelity,
                    pr_auc,
                )
            )

        else:

            pearson_f1 = np.nan
            pearson_f1_p = np.nan
            spearman_f1 = np.nan
            spearman_f1_p = np.nan

            pearson_pr = np.nan
            pearson_pr_p = np.nan
            spearman_pr = np.nan
            spearman_pr_p = np.nan

        rows.append(
            {
                "top_k": int(
                    top_k
                ),

                "pearson_fidelity_f1": (
                    pearson_f1
                ),

                "pearson_fidelity_f1_pvalue": (
                    pearson_f1_p
                ),

                "spearman_fidelity_f1": (
                    spearman_f1
                ),

                "spearman_fidelity_f1_pvalue": (
                    spearman_f1_p
                ),

                "pearson_fidelity_pr_auc": (
                    pearson_pr
                ),

                "pearson_fidelity_pr_auc_pvalue": (
                    pearson_pr_p
                ),

                "spearman_fidelity_pr_auc": (
                    spearman_pr
                ),

                "spearman_fidelity_pr_auc_pvalue": (
                    spearman_pr_p
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# Summary
# ============================================================


def build_summary(
    integrated_df: pd.DataFrame,
    drift_index_df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for top_k, group in integrated_df.groupby(
        "top_k"
    ):

        group = (
            group
            .sort_values("window")
            .reset_index(drop=True)
        )

        drift_group = (
            drift_index_df[
                drift_index_df[
                    "top_k"
                ]
                == top_k
            ]
            .sort_values("window")
            .reset_index(drop=True)
        )

        final = group.iloc[-1]
        final_drift = drift_group.iloc[-1]

        rows.append(
            {
                "top_k": int(
                    top_k
                ),

                "window_1_fidelity": (
                    group.iloc[0][
                        "fidelity"
                    ]
                ),

                "window_4_fidelity": (
                    final[
                        "fidelity"
                    ]
                ),

                "absolute_fidelity_change": (
                    final[
                        "fidelity"
                    ]
                    - group.iloc[0][
                        "fidelity"
                    ]
                ),

                "relative_fidelity_change_pct": (
                    final[
                        "fidelity_relative_change_pct"
                    ]
                ),

                "fidelity_drift_index_pct": (
                    final_drift[
                        "fidelity_drift_index_pct"
                    ]
                ),

                "window_1_f1": (
                    group.iloc[0][
                        "f1"
                    ]
                ),

                "window_4_f1": (
                    final[
                        "f1"
                    ]
                ),

                "f1_relative_degradation_pct": (
                    final[
                        "f1_relative_degradation_pct"
                    ]
                ),

                "window_1_pr_auc": (
                    group.iloc[0][
                        "pr_auc"
                    ]
                ),

                "window_4_pr_auc": (
                    final[
                        "pr_auc"
                    ]
                ),

                "pr_auc_relative_degradation_pct": (
                    final[
                        "pr_auc_relative_degradation_pct"
                    ]
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# Research interpretation
# ============================================================


def generate_research_interpretation(
    summary_df: pd.DataFrame,
    correlation_df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for _, row in summary_df.iterrows():

        top_k = int(
            row["top_k"]
        )

        fidelity_change = float(
            row[
                "relative_fidelity_change_pct"
            ]
        )

        f1_degradation = float(
            row[
                "f1_relative_degradation_pct"
            ]
        )

        pr_auc_degradation = float(
            row[
                "pr_auc_relative_degradation_pct"
            ]
        )

        corr = correlation_df[
            correlation_df[
                "top_k"
            ]
            == top_k
        ]

        if len(corr) == 1:

            corr = corr.iloc[0]

            spearman_f1 = float(
                corr[
                    "spearman_fidelity_f1"
                ]
            )

            spearman_pr_auc = float(
                corr[
                    "spearman_fidelity_pr_auc"
                ]
            )

        else:

            spearman_f1 = np.nan
            spearman_pr_auc = np.nan

        if fidelity_change < 0:

            fidelity_direction = (
                "fidelity decreased"
            )

        elif fidelity_change > 0:

            fidelity_direction = (
                "fidelity increased"
            )

        else:

            fidelity_direction = (
                "fidelity remained unchanged"
            )

        if (
            f1_degradation > 0
            and pr_auc_degradation > 0
        ):

            detection_direction = (
                "both F1 and PR-AUC degraded"
            )

        elif (
            f1_degradation > 0
            or pr_auc_degradation > 0
        ):

            detection_direction = (
                "detection performance partially degraded"
            )

        else:

            detection_direction = (
                "detection performance did not degrade"
            )

        rows.append(
            {
                "top_k": top_k,

                "window_1_to_window_4": (
                    f"{fidelity_direction}; "
                    f"{detection_direction}"
                ),

                "fidelity_change_pct": (
                    fidelity_change
                ),

                "f1_degradation_pct": (
                    f1_degradation
                ),

                "pr_auc_degradation_pct": (
                    pr_auc_degradation
                ),

                "spearman_fidelity_f1": (
                    spearman_f1
                ),

                "spearman_fidelity_pr_auc": (
                    spearman_pr_auc
                ),

                "interpretation": (
                    "Descriptive temporal evidence only. "
                    "With four windows, this result should "
                    "not be treated as definitive inferential "
                    "evidence. It is used to identify the "
                    "temporal relationship that must be "
                    "tested in the robustness/statistical "
                    "experiments."
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# Research manifest
# ============================================================


def build_result_manifest(
    output_dir: Path,
    generated_files: List[Path],
) -> pd.DataFrame:

    rows = []

    for path in generated_files:

        rows.append(
            {
                "artifact": path.name,

                "path": str(
                    path
                ),

                "category": (
                    "E5.3 temporal "
                    "fidelity-vs-detection"
                ),

                "status": "generated",

                "purpose": (
                    "Research evidence artifact "
                    "for temporal explanation-fidelity "
                    "analysis."
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# Main experiment
# ============================================================


def run_experiment(
    fidelity_file: Path = DEFAULT_FIDELITY_FILE,
    detection_file: Path | None = None,
    class_fidelity_file: Path = DEFAULT_CLASS_FIDELITY_FILE,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
):

    print("=" * 70)

    print(
        "BASELINE 1 — TEMPORAL "
        "FIDELITY VS DETECTION ANALYSIS"
    )

    print("=" * 70)

    print(
        "\nThis analysis compares:"
    )

    print(
        "  • ExIFFI decision-level fidelity"
    )

    print(
        "  • F1 trajectory"
    )

    print(
        "  • PR-AUC trajectory"
    )

    print(
        "  • quantitative fidelity drift"
    )

    print(
        "  • detection/fidelity dissociation"
    )

    # ---------------------------------------------------------
    # 1. Load fidelity
    # ---------------------------------------------------------

    print(
        "\n[1/8] Loading fidelity trajectory..."
    )

    fidelity_df = load_fidelity_trajectory(
        fidelity_file
    )

    print(
        f"Loaded {len(fidelity_df):,} "
        "fidelity trajectory rows."
    )

    print(
        "Top-k values:",
        sorted(
            fidelity_df[
                "top_k"
            ]
            .unique()
            .tolist()
        ),
    )

    print(
        "Windows:",
        sorted(
            fidelity_df[
                "window"
            ]
            .unique()
            .tolist()
        ),
    )

    # ---------------------------------------------------------
    # 2. Load detection
    # ---------------------------------------------------------

    print(
        "\n[2/8] Loading detection trajectory..."
    )

    detection_df = (
        load_detection_results(
            detection_file
        )
    )

    print(
        detection_df.to_string(
            index=False
        )
    )

    # ---------------------------------------------------------
    # 3. Calculate trajectories
    # ---------------------------------------------------------

    print(
        "\n[3/8] Calculating temporal trajectories..."
    )

    fidelity_trajectory = (
        calculate_fidelity_trajectory(
            fidelity_df
        )
    )

    detection_trajectory = (
        calculate_detection_trajectory(
            detection_df
        )
    )

    integrated_df = (
        build_integrated_trajectory(
            fidelity_trajectory,
            detection_trajectory,
        )
    )

    # ---------------------------------------------------------
    # 4. Dissociation
    # ---------------------------------------------------------

    print(
        "\n[4/8] Calculating "
        "detection/fidelity dissociation..."
    )

    dissociation_df = (
        calculate_dissociation(
            integrated_df
        )
    )

    # ---------------------------------------------------------
    # 5. Fidelity drift index
    # ---------------------------------------------------------

    print(
        "\n[5/8] Calculating "
        "fidelity-drift indicators..."
    )

    drift_index_df = (
        calculate_fidelity_drift_index(
            fidelity_trajectory
        )
    )

    slope_df = (
        calculate_temporal_slopes(
            integrated_df
        )
    )

    correlation_df = (
        calculate_correlations(
            integrated_df
        )
    )

    summary_df = (
        build_summary(
            integrated_df,
            drift_index_df,
        )
    )

    interpretation_df = (
        generate_research_interpretation(
            summary_df,
            correlation_df,
        )
    )

    # ---------------------------------------------------------
    # 6. Class-specific analysis
    # ---------------------------------------------------------

    print(
        "\n[6/8] Loading class-specific fidelity..."
    )

    class_fidelity_df = (
        load_class_fidelity(
            class_fidelity_file
        )
    )

    class_dissociation_df = (
        pd.DataFrame()
    )

    if class_fidelity_df is not None:

        class_fidelity_trajectory = (
            calculate_fidelity_trajectory(
                class_fidelity_df
                .rename(
                    columns={
                        "mean_relative_score_change":
                            "mean_relative_score_change"
                    }
                )
            )
        )

        # Keep class-specific information
        # directly in a dedicated table rather
        # than forcing it into the overall
        # trajectory.

        class_rows = []

        for (
            is_fraud,
            class_name,
            top_k,
        ), group in class_fidelity_df.groupby(
            [
                "isFraud",
                "class",
                "top_k",
            ]
        ):

            group = (
                group
                .sort_values("window")
                .reset_index(drop=True)
            )

            baseline = float(
                group.iloc[0][
                    "mean_relative_score_change"
                ]
            )

            for _, row in group.iterrows():

                current = float(
                    row[
                        "mean_relative_score_change"
                    ]
                )

                class_rows.append(
                    {
                        "window": int(
                            row["window"]
                        ),

                        "isFraud": int(
                            is_fraud
                        ),

                        "class": class_name,

                        "top_k": int(
                            top_k
                        ),

                        "fidelity": current,

                        "fidelity_change_from_window_1": (
                            current
                            - baseline
                        ),

                        "fidelity_relative_change_pct": (
                            safe_percentage_change(
                                current,
                                baseline,
                            )
                        ),
                    }
                )

        class_dissociation_df = (
            pd.DataFrame(
                class_rows
            )
        )

    # ---------------------------------------------------------
    # 7. Save
    # ---------------------------------------------------------

    print(
        "\n[7/8] Saving research artifacts..."
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    integrated_path = (
        output_dir
        / "static_if_exiffi_temporal_fidelity_detection_trajectory.csv"
    )

    detection_path = (
        output_dir
        / "static_if_temporal_detection_trajectory.csv"
    )

    dissociation_path = (
        output_dir
        / "static_if_exiffi_fidelity_detection_dissociation.csv"
    )

    drift_index_path = (
        output_dir
        / "static_if_exiffi_fidelity_drift_index.csv"
    )

    slopes_path = (
        output_dir
        / "static_if_exiffi_fidelity_detection_slopes.csv"
    )

    correlations_path = (
        output_dir
        / "static_if_exiffi_fidelity_detection_correlations.csv"
    )

    summary_path = (
        output_dir
        / "static_if_exiffi_fidelity_detection_summary.csv"
    )

    interpretation_path = (
        output_dir
        / "static_if_exiffi_fidelity_detection_interpretation.csv"
    )

    integrated_df.to_csv(
        integrated_path,
        index=False,
    )

    detection_trajectory.to_csv(
        detection_path,
        index=False,
    )

    dissociation_df.to_csv(
        dissociation_path,
        index=False,
    )

    drift_index_df.to_csv(
        drift_index_path,
        index=False,
    )

    slope_df.to_csv(
        slopes_path,
        index=False,
    )

    correlation_df.to_csv(
        correlations_path,
        index=False,
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    interpretation_df.to_csv(
        interpretation_path,
        index=False,
    )

    generated_files = [
        integrated_path,
        detection_path,
        dissociation_path,
        drift_index_path,
        slopes_path,
        correlations_path,
        summary_path,
        interpretation_path,
    ]

    if (
        not class_dissociation_df.empty
    ):

        class_path = (
            output_dir
            / "static_if_exiffi_class_fidelity_temporal_analysis.csv"
        )

        class_dissociation_df.to_csv(
            class_path,
            index=False,
        )

        generated_files.append(
            class_path
        )

    manifest_df = (
        build_result_manifest(
            output_dir,
            generated_files,
        )
    )

    manifest_path = (
        output_dir
        / "static_if_exiffi_fidelity_detection_result_manifest.csv"
    )

    manifest_df.to_csv(
        manifest_path,
        index=False,
    )

    generated_files.append(
        manifest_path
    )

    # ---------------------------------------------------------
    # 8. Console summary
    # ---------------------------------------------------------

    print(
        "\n[8/8] Analysis summary"
    )

    print(
        "\nIntegrated trajectory:"
    )

    print(
        integrated_df[
            [
                "window",
                "top_k",
                "fidelity",
                "f1",
                "pr_auc",
                "fidelity_relative_change_pct",
                "f1_relative_degradation_pct",
                "pr_auc_relative_degradation_pct",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nFidelity-drift indicators:"
    )

    print(
        drift_index_df.to_string(
            index=False
        )
    )

    print(
        "\nDetection/fidelity dissociation:"
    )

    print(
        dissociation_df.to_string(
            index=False
        )
    )

    print(
        "\nTemporal slopes:"
    )

    print(
        slope_df.to_string(
            index=False
        )
    )

    print(
        "\nCorrelations:"
    )

    print(
        correlation_df.to_string(
            index=False
        )
    )

    print(
        "\nResearch summary:"
    )

    print(
        summary_df.to_string(
            index=False
        )
    )

    print(
        "\nArtifacts saved:"
    )

    for path in generated_files:

        print(
            f"  {path}"
        )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "TEMPORAL FIDELITY VS DETECTION "
        "ANALYSIS COMPLETE"
    )

    print(
        "=" * 70
    )

    return (
        integrated_df,
        dissociation_df,
        drift_index_df,
        slope_df,
        correlation_df,
        summary_df,
        interpretation_df,
    )


# ============================================================
# CLI
# ============================================================


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Compare temporal ExIFFI "
            "explanation fidelity against "
            "Isolation Forest detection "
            "performance."
        )
    )

    parser.add_argument(
        "--fidelity-file",
        default=str(
            DEFAULT_FIDELITY_FILE
        ),
        help=(
            "Temporal ExIFFI fidelity "
            "trajectory CSV."
        ),
    )

    parser.add_argument(
        "--detection-file",
        default=None,
        help=(
            "Optional detection trajectory "
            "CSV containing window, F1 and "
            "PR-AUC. If omitted, the documented "
            "baseline trajectory is used."
        ),
    )

    parser.add_argument(
        "--class-fidelity-file",
        default=str(
            DEFAULT_CLASS_FIDELITY_FILE
        ),
        help=(
            "Class-specific temporal fidelity "
            "trajectory CSV."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=str(
            DEFAULT_OUTPUT_DIR
        ),
        help=(
            "Directory for research outputs."
        ),
    )

    args = parser.parse_args()

    run_experiment(
        fidelity_file=Path(
            args.fidelity_file
        ),
        detection_file=(
            Path(
                args.detection_file
            )
            if args.detection_file
            else None
        ),
        class_fidelity_file=Path(
            args.class_fidelity_file
        ),
        output_dir=Path(
            args.output_dir
        ),
    )


if __name__ == "__main__":
    main()