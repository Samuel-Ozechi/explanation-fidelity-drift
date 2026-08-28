from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import IsolationForest

from efd.data.preprocessing import (
    FeaturePolicy,
    NumericTransactionPreprocessor,
)
from efd.data.temporal import (
    chronological_windows,
)


# ============================================================
# E4.2-T04
#
# EXIFFI FIDELITY METRIC SUFFICIENCY
# ============================================================
#
# Research question:
#
# Is mean relative model-score change sufficient as the
# primary local explanation-fidelity metric?
#
# The experiment compares:
#
#   1. ExIFFI-selected feature perturbation
#   2. Random-feature perturbation controls
#
# across:
#
#   k = {1, 3, 5, 10, 20}
#
# The analysis evaluates:
#
#   - absolute score change
#   - relative score change
#   - ExIFFI vs random effect size
#   - bootstrap confidence intervals
#   - bootstrap p-values
#   - top-k mass
#   - monotonicity across k
#   - relationship between explanation mass and
#     model-score change
#
# IMPORTANT:
#
# The explanation CSV stores the model score from the original
# explanation-generation run.
#
# We do NOT require the newly reconstructed Isolation Forest
# to reproduce that score exactly.
#
# This is intentional because sklearn/runtime/model
# reconstruction can introduce differences even when the
# experiment configuration is identical.
#
# The actual fidelity experiment uses the CURRENT reconstructed
# model consistently for:
#
#       original score
#       ExIFFI perturbation score
#       random-control perturbation score
#
# This keeps the intervention experiment internally consistent.
#
# ============================================================


# ============================================================
# Configuration
# ============================================================

DEFAULT_EXPLANATION_DIR = Path(
    "results/explanations"
)

DEFAULT_OUTPUT_DIR = Path(
    "results/tables"
)

DEFAULT_TRAIN_FRACTION = 0.60

DEFAULT_N_ESTIMATORS = 300

DEFAULT_RANDOM_STATE = 42

DEFAULT_SAMPLE_SIZE = 100

DEFAULT_RANDOM_REPEATS = 50

DEFAULT_BOOTSTRAP = 2000

DEFAULT_BOOTSTRAP_SEED = 42

DEFAULT_REFERENCE_DATE = "2017-12-01"

DEFAULT_MAX_MISSING_RATE = 0.95

DEFAULT_TOLERANCE = 1e-8

DEFAULT_TOP_K_VALUES = [
    1,
    3,
    5,
    10,
    20,
]

DEFAULT_WINDOWS = [1]


# ============================================================
# Utility
# ============================================================


def ensure_numeric_array(
    X,
) -> np.ndarray:

    if isinstance(X, pd.DataFrame):
        return X.to_numpy(
            dtype=float
        )

    return np.asarray(
        X,
        dtype=float,
    )


def model_score_samples(
    model,
    X,
    feature_names: List[str],
) -> np.ndarray:
    """
    Compute sklearn Isolation Forest score_samples
    while preserving feature names.

    sklearn score_samples convention:

        larger = more normal

    Research anomaly score convention:

        larger = more anomalous

    For fidelity, however, we work with the raw model
    score consistently because relative score change
    is based on differences from the same baseline.
    """

    X_array = ensure_numeric_array(X)

    if X_array.ndim == 1:
        X_array = X_array.reshape(
            1,
            -1,
        )

    X_frame = pd.DataFrame(
        X_array,
        columns=feature_names,
    )

    return np.asarray(
        model.score_samples(
            X_frame
        ),
        dtype=float,
    )


# ============================================================
# Explanation loading
# ============================================================


def load_explanation_file(
    window: int,
    explanation_dir: Path,
) -> pd.DataFrame:

    path = (
        explanation_dir
        / f"static_if_exiffi_w{window}_sample.csv"
    )

    if not path.exists():

        raise FileNotFoundError(
            "\nMissing explanation file:\n"
            f"{path}\n\n"
            "Expected files such as:\n"
            "results/explanations/"
            "static_if_exiffi_w1_sample.csv"
        )

    df = pd.read_csv(
        path
    )

    required = [
        "sample_index",
        "isFraud",
        "anomaly_score",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Window {window} explanation file "
            "is missing required columns: "
            f"{missing}"
        )

    importance_cols = [
        column
        for column in df.columns
        if column.startswith(
            "importance__"
        )
    ]

    if not importance_cols:

        raise ValueError(
            f"Window {window} explanation file "
            "contains no importance__ columns."
        )

    if len(df) == 0:

        raise ValueError(
            f"Window {window} explanation file "
            "contains zero observations."
        )

    print(
        f"Window {window}: "
        f"{len(df):,} observations, "
        f"{len(importance_cols):,} features"
    )

    return df


# ============================================================
# Feature extraction
# ============================================================


def extract_explanation_matrix(
    df: pd.DataFrame,
) -> Tuple[
    np.ndarray,
    List[str],
]:

    importance_cols = [
        column
        for column in df.columns
        if column.startswith(
            "importance__"
        )
    ]

    feature_names = [
        column.replace(
            "importance__",
            "",
            1,
        )
        for column in importance_cols
    ]

    matrix = (
        df[importance_cols]
        .to_numpy(
            dtype=float
        )
    )

    return (
        matrix,
        feature_names,
    )


# ============================================================
# Sampling
# ============================================================


def stratified_sample_indices(
    y: pd.Series,
    sample_size: int,
    random_state: int,
) -> np.ndarray:
    """
    Reproduce the sampling procedure used by
    generate_exiffi_temporal_explanations.py.

    This is important because the explanation CSV stores
    sample_index relative to the sampled matrix, not the
    original future-window row number.
    """

    if sample_size <= 0:

        raise ValueError(
            "sample_size must be > 0."
        )

    if sample_size > len(y):

        raise ValueError(
            "sample_size cannot exceed "
            "the window size."
        )

    values = y.to_numpy()

    classes = np.unique(
        values
    )

    if len(classes) < 2:

        raise ValueError(
            "Stratified sampling requires "
            "at least two classes."
        )

    rng = np.random.default_rng(
        random_state
    )

    n_classes = len(classes)

    base_per_class = (
        sample_size // n_classes
    )

    remainder = (
        sample_size % n_classes
    )

    selected = []

    for class_position, class_value in enumerate(
        classes
    ):

        class_indices = np.flatnonzero(
            values == class_value
        )

        target_n = (
            base_per_class
            + (
                1
                if class_position < remainder
                else 0
            )
        )

        target_n = min(
            target_n,
            len(class_indices),
        )

        selected_class = rng.choice(
            class_indices,
            size=target_n,
            replace=False,
        )

        selected.extend(
            selected_class.tolist()
        )

    selected_set = set(
        selected
    )

    remaining_n = (
        sample_size
        - len(selected)
    )

    if remaining_n > 0:

        available = np.asarray(
            [
                index
                for index in range(
                    len(values)
                )
                if index not in selected_set
            ],
            dtype=int,
        )

        additional = rng.choice(
            available,
            size=remaining_n,
            replace=False,
        )

        selected.extend(
            additional.tolist()
        )

    selected = np.asarray(
        selected,
        dtype=int,
    )

    rng.shuffle(
        selected
    )

    return selected


# ============================================================
# Sample reconstruction
# ============================================================


def reconstruct_sample(
    window: pd.DataFrame,
    X_window: pd.DataFrame,
    explanation_df: pd.DataFrame,
    sample_strategy: str,
    sampling_seed: int,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Reconstruct the exact 100-observation sample used by
    the explanation-generation pipeline.

    Returns
    -------
    X_sample
        Reconstructed feature matrix.

    y_sample
        Fraud labels.

    sample_positions
        Positional indices into the original future window.
    """

    n_explanations = len(
        explanation_df
    )

    if sample_strategy != "stratified":

        raise ValueError(
            "This implementation expects the "
            "existing temporal explanation artifacts "
            "to have been generated using "
            "sample_strategy='stratified'."
        )

    sample_positions = (
        stratified_sample_indices(
            y=window["isFraud"],
            sample_size=n_explanations,
            random_state=sampling_seed,
        )
    )

    if isinstance(
        X_window,
        pd.DataFrame,
    ):

        X_sample = (
            X_window
            .iloc[sample_positions]
            .to_numpy(
                dtype=float
            )
        )

    else:

        X_array = ensure_numeric_array(
            X_window
        )

        X_sample = (
            X_array[
                sample_positions
            ]
        )

    y_sample = (
        window["isFraud"]
        .iloc[sample_positions]
        .to_numpy(
            dtype=int
        )
    )

    # --------------------------------------------------------
    # Validate labels against explanation artifact.
    # --------------------------------------------------------

    stored_labels = (
        explanation_df[
            "isFraud"
        ]
        .to_numpy(
            dtype=int
        )
    )

    if not np.array_equal(
        y_sample,
        stored_labels,
    ):

        raise ValueError(
            "Fraud labels reconstructed from the "
            "temporal sampling procedure do not "
            "match the labels stored in the "
            "explanation file.\n\n"
            "This means the explanation artifact "
            "was generated with different sampling "
            "metadata."
        )

    # --------------------------------------------------------
    # Validate sample indices.
    #
    # The stored sample_index should normally be
    # 0..n-1 because the explanation generator assigns
    # indices after constructing X_sample.
    # --------------------------------------------------------

    stored_indices = (
        explanation_df[
            "sample_index"
        ]
        .to_numpy(
            dtype=int
        )
    )

    expected_indices = np.arange(
        n_explanations,
        dtype=int,
    )

    if not np.array_equal(
        stored_indices,
        expected_indices,
    ):

        raise ValueError(
            "Explanation sample_index values are "
            "not sequential 0..n-1.\n"
            "The current reconstruction logic relies "
            "on the sample_index convention used by "
            "generate_exiffi_temporal_explanations.py."
        )

    return (
        X_sample,
        y_sample,
        sample_positions,
    )


# ============================================================
# Score diagnostics
# ============================================================


def validate_original_scores(
    model,
    X_sample: np.ndarray,
    explanation_df: pd.DataFrame,
    feature_names: List[str],
) -> Dict[str, float]:
    """
    Compare stored explanation scores against scores from
    the reconstructed model.

    IMPORTANT:

    A mismatch is a diagnostic, NOT a fatal error.

    The actual fidelity experiment always uses scores
    generated from the current reconstructed model.
    """

    current_scores = model_score_samples(
        model=model,
        X=X_sample,
        feature_names=feature_names,
    )

    current_anomaly_scores = (
        -current_scores
    )

    stored_anomaly_scores = (
        explanation_df[
            "anomaly_score"
        ]
        .to_numpy(
            dtype=float
        )
    )

    differences = np.abs(
        current_anomaly_scores
        - stored_anomaly_scores
    )

    max_difference = float(
        np.max(
            differences
        )
    )

    mean_difference = float(
        np.mean(
            differences
        )
    )

    median_difference = float(
        np.median(
            differences
        )
    )

    exact_match = bool(
        np.allclose(
            current_anomaly_scores,
            stored_anomaly_scores,
            rtol=DEFAULT_TOLERANCE,
            atol=DEFAULT_TOLERANCE,
        )
    )

    if (
        np.std(
            current_anomaly_scores
        ) > 0
        and np.std(
            stored_anomaly_scores
        ) > 0
    ):

        correlation = float(
            np.corrcoef(
                current_anomaly_scores,
                stored_anomaly_scores,
            )[0, 1]
        )

    else:

        correlation = np.nan

    print(
        "\n  Original-score reproducibility diagnostic:"
    )

    print(
        f"    Exact match: "
        f"{exact_match}"
    )

    print(
        f"    Maximum absolute difference: "
        f"{max_difference:.12g}"
    )

    print(
        f"    Mean absolute difference: "
        f"{mean_difference:.12g}"
    )

    print(
        f"    Median absolute difference: "
        f"{median_difference:.12g}"
    )

    if np.isfinite(
        correlation
    ):

        print(
            f"    Score correlation: "
            f"{correlation:.6f}"
        )

    if not exact_match:

        print(
            "\n  WARNING:"
        )

        print(
            "    Stored explanation scores do not "
            "exactly match the reconstructed model."
        )

        print(
            "    This is retained as a reproducibility "
            "diagnostic and does not terminate E4.2-T04."
        )

        print(
            "    Fidelity will use the reconstructed "
            "model consistently."
        )

    return {
        "scores_match": int(
            exact_match
        ),
        "maximum_absolute_difference": (
            max_difference
        ),
        "mean_absolute_difference": (
            mean_difference
        ),
        "median_absolute_difference": (
            median_difference
        ),
        "score_correlation": (
            correlation
        ),
    }


# ============================================================
# Top-k extraction
# ============================================================


def top_k_indices(
    importance: np.ndarray,
    k: int,
) -> np.ndarray:

    importance = np.asarray(
        importance,
        dtype=float,
    )

    if importance.ndim != 1:

        raise ValueError(
            "importance must be one-dimensional."
        )

    k = min(
        k,
        len(importance),
    )

    return np.argsort(
        -importance
    )[:k]


def top_k_mass(
    importance: np.ndarray,
    k: int,
) -> float:

    total = float(
        np.sum(
            np.abs(
                importance
            )
        )
    )

    if total <= 0:

        return 0.0

    normalized = (
        np.abs(
            importance
        )
        / total
    )

    indices = top_k_indices(
        normalized,
        k,
    )

    return float(
        normalized[
            indices
        ].sum()
    )


# ============================================================
# Perturbation
# ============================================================


def perturb(
    x: np.ndarray,
    indices: np.ndarray,
    baseline: np.ndarray,
) -> np.ndarray:

    result = np.asarray(
        x,
        dtype=float,
    ).copy()

    result[
        indices
    ] = baseline[
        indices
    ]

    return result


# ============================================================
# ExIFFI fidelity
# ============================================================


def evaluate_exiffi_fidelity(
    model,
    X_sample: np.ndarray,
    explanation_matrix: np.ndarray,
    baseline: np.ndarray,
    feature_names: List[str],
    k_values: List[int],
) -> pd.DataFrame:
    """
    Evaluate ExIFFI-selected perturbations.

    The baseline score is the current reconstructed
    model score for the unperturbed observation.

    The perturbed score is generated by replacing
    the explanation's top-k features with training
    median values.
    """

    rows = []

    baseline_scores = model_score_samples(
        model=model,
        X=X_sample,
        feature_names=feature_names,
    )

    for observation_index in range(
        len(X_sample)
    ):

        x = X_sample[
            observation_index
        ]

        importance = (
            explanation_matrix[
                observation_index
            ]
        )

        base_score = float(
            baseline_scores[
                observation_index
            ]
        )

        for k in k_values:

            indices = top_k_indices(
                importance,
                k,
            )

            mass = top_k_mass(
                importance,
                k,
            )

            x_perturbed = perturb(
                x=x,
                indices=indices,
                baseline=baseline,
            )

            perturbed_score = float(
                model_score_samples(
                    model=model,
                    X=x_perturbed,
                    feature_names=feature_names,
                )[0]
            )

            score_change = (
                perturbed_score
                - base_score
            )

            relative_score_change = (
                abs(score_change)
                / max(
                    abs(base_score),
                    1e-12,
                )
            )

            rows.append(
                {
                    "sample_index": (
                        observation_index
                    ),
                    "top_k": k,
                    "condition": "exiffi",
                    "baseline_score": (
                        base_score
                    ),
                    "perturbed_score": (
                        perturbed_score
                    ),
                    "score_change": (
                        score_change
                    ),
                    "absolute_score_change": (
                        abs(score_change)
                    ),
                    "relative_score_change": (
                        relative_score_change
                    ),
                    "top_k_mass": mass,
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Random control
# ============================================================


def evaluate_random_fidelity(
    model,
    X_sample: np.ndarray,
    baseline: np.ndarray,
    feature_names: List[str],
    k_values: List[int],
    random_repeats: int,
    random_state: int,
) -> pd.DataFrame:
    """
    Evaluate random-feature perturbation controls.

    Every random control uses exactly the same intervention
    as the ExIFFI condition.

    Only feature selection differs.
    """

    rng = np.random.default_rng(
        random_state
    )

    n_observations = len(
        X_sample
    )

    n_features = (
        X_sample.shape[1]
    )

    baseline_scores = model_score_samples(
        model=model,
        X=X_sample,
        feature_names=feature_names,
    )

    rows = []

    for observation_index in range(
        n_observations
    ):

        x = X_sample[
            observation_index
        ]

        base_score = float(
            baseline_scores[
                observation_index
            ]
        )

        for k in k_values:

            for repeat in range(
                random_repeats
            ):

                indices = rng.choice(
                    n_features,
                    size=k,
                    replace=False,
                )

                x_perturbed = perturb(
                    x=x,
                    indices=indices,
                    baseline=baseline,
                )

                perturbed_score = float(
                    model_score_samples(
                        model=model,
                        X=x_perturbed,
                        feature_names=feature_names,
                    )[0]
                )

                score_change = (
                    perturbed_score
                    - base_score
                )

                relative_score_change = (
                    abs(score_change)
                    / max(
                        abs(base_score),
                        1e-12,
                    )
                )

                rows.append(
                    {
                        "sample_index": (
                            observation_index
                        ),
                        "top_k": k,
                        "condition": "random",
                        "repeat": repeat,
                        "baseline_score": (
                            base_score
                        ),
                        "perturbed_score": (
                            perturbed_score
                        ),
                        "score_change": (
                            score_change
                        ),
                        "absolute_score_change": (
                            abs(score_change)
                        ),
                        "relative_score_change": (
                            relative_score_change
                        ),
                    }
                )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Random aggregation
# ============================================================


def aggregate_random_controls(
    random_df: pd.DataFrame,
) -> pd.DataFrame:

    return (
        random_df
        .groupby(
            [
                "sample_index",
                "top_k",
            ],
            as_index=False,
        )
        .agg(
            random_mean_score_change=(
                "score_change",
                "mean",
            ),
            random_mean_absolute_score_change=(
                "absolute_score_change",
                "mean",
            ),
            random_mean_relative_score_change=(
                "relative_score_change",
                "mean",
            ),
            random_median_relative_score_change=(
                "relative_score_change",
                "median",
            ),
            random_std_relative_score_change=(
                "relative_score_change",
                "std",
            ),
        )
    )


# ============================================================
# Paired observation-level results
# ============================================================


def build_observation_level_results(
    exiffi_df: pd.DataFrame,
    random_aggregate: pd.DataFrame,
    labels: np.ndarray,
) -> pd.DataFrame:

    merged = exiffi_df.merge(
        random_aggregate,
        on=[
            "sample_index",
            "top_k",
        ],
        how="left",
        validate="one_to_one",
    )

    label_map = {
        index: int(
            labels[index]
        )
        for index in range(
            len(labels)
        )
    }

    merged[
        "isFraud"
    ] = merged[
        "sample_index"
    ].map(
        label_map
    )

    merged[
        "exiffi_minus_random_relative"
    ] = (
        merged[
            "relative_score_change"
        ]
        - merged[
            "random_mean_relative_score_change"
        ]
    )

    merged[
        "relative_effect_ratio"
    ] = (
        merged[
            "relative_score_change"
        ]
        / np.maximum(
            merged[
                "random_mean_relative_score_change"
            ],
            1e-12,
        )
    )

    merged[
        "exiffi_minus_random_absolute"
    ] = (
        merged[
            "absolute_score_change"
        ]
        - merged[
            "random_mean_absolute_score_change"
        ]
    )

    return merged


# ============================================================
# Bootstrap
# ============================================================


def bootstrap_mean_difference(
    values: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> Tuple[
    float,
    float,
    float,
]:
    """
    Bootstrap the mean paired difference.

    Returns:

        observed_mean
        CI lower
        CI upper
    """

    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:

        return (
            np.nan,
            np.nan,
            np.nan,
        )

    observed = float(
        np.mean(values)
    )

    indices = rng.integers(
        0,
        len(values),
        size=(
            n_bootstrap,
            len(values),
        ),
    )

    bootstrap_means = (
        values[
            indices
        ]
        .mean(
            axis=1
        )
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

    return (
        observed,
        lower,
        upper,
    )


def bootstrap_two_sided_pvalue(
    values: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> float:
    """
    Bootstrap sign-based two-sided p-value for a paired
    mean difference.

    Null hypothesis:

        mean difference = 0
    """

    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:

        return np.nan

    observed = float(
        np.mean(values)
    )

    centered = (
        values
        - observed
    )

    indices = rng.integers(
        0,
        len(centered),
        size=(
            n_bootstrap,
            len(centered),
        ),
    )

    bootstrap_null_means = (
        centered[
            indices
        ]
        .mean(
            axis=1
        )
    )

    p_value = (
        np.sum(
            np.abs(
                bootstrap_null_means
            )
            >= abs(observed)
        )
        + 1
    ) / (
        n_bootstrap
        + 1
    )

    return float(
        p_value
    )


# ============================================================
# Bootstrap summary
# ============================================================


def build_bootstrap_summary(
    observation_df: pd.DataFrame,
    k_values: List[int],
    n_bootstrap: int,
    random_state: int,
) -> pd.DataFrame:

    rows = []

    for k in k_values:

        subset = observation_df[
            observation_df[
                "top_k"
            ]
            == k
        ]

        relative_difference = (
            subset[
                "exiffi_minus_random_relative"
            ]
            .to_numpy(
                dtype=float
            )
        )

        absolute_difference = (
            subset[
                "exiffi_minus_random_absolute"
            ]
            .to_numpy(
                dtype=float
            )
        )

        rng_relative = (
            np.random.default_rng(
                random_state
                + k
            )
        )

        (
            observed_relative,
            relative_lower,
            relative_upper,
        ) = bootstrap_mean_difference(
            values=relative_difference,
            n_bootstrap=n_bootstrap,
            rng=rng_relative,
        )

        relative_p = (
            bootstrap_two_sided_pvalue(
                values=relative_difference,
                n_bootstrap=n_bootstrap,
                rng=np.random.default_rng(
                    random_state
                    + 1000
                    + k
                ),
            )
        )

        (
            observed_absolute,
            absolute_lower,
            absolute_upper,
        ) = bootstrap_mean_difference(
            values=absolute_difference,
            n_bootstrap=n_bootstrap,
            rng=np.random.default_rng(
                random_state
                + 2000
                + k
            ),
        )

        absolute_p = (
            bootstrap_two_sided_pvalue(
                values=absolute_difference,
                n_bootstrap=n_bootstrap,
                rng=np.random.default_rng(
                    random_state
                    + 3000
                    + k
                ),
            )
        )

        exiffi_relative_mean = float(
            subset[
                "relative_score_change"
            ].mean()
        )

        random_relative_mean = float(
            subset[
                "random_mean_relative_score_change"
            ].mean()
        )

        exiffi_absolute_mean = float(
            subset[
                "absolute_score_change"
            ].mean()
        )

        random_absolute_mean = float(
            subset[
                "random_mean_absolute_score_change"
            ].mean()
        )

        ratio = (
            exiffi_relative_mean
            / max(
                random_relative_mean,
                1e-12,
            )
        )

        rows.append(
            {
                "top_k": k,
                "n_observations": len(
                    subset
                ),
                "exiffi_mean_relative_score_change": (
                    exiffi_relative_mean
                ),
                "random_mean_relative_score_change": (
                    random_relative_mean
                ),
                "mean_relative_difference": (
                    observed_relative
                ),
                "bootstrap_ci_lower_relative": (
                    relative_lower
                ),
                "bootstrap_ci_upper_relative": (
                    relative_upper
                ),
                "bootstrap_pvalue_relative": (
                    relative_p
                ),
                "exiffi_mean_absolute_score_change": (
                    exiffi_absolute_mean
                ),
                "random_mean_absolute_score_change": (
                    random_absolute_mean
                ),
                "mean_absolute_difference": (
                    observed_absolute
                ),
                "bootstrap_ci_lower_absolute": (
                    absolute_lower
                ),
                "bootstrap_ci_upper_absolute": (
                    absolute_upper
                ),
                "bootstrap_pvalue_absolute": (
                    absolute_p
                ),
                "exiffi_to_random_relative_ratio": (
                    ratio
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Monotonicity
# ============================================================


def evaluate_monotonicity(
    summary_df: pd.DataFrame,
) -> Dict[str, float]:

    ordered = (
        summary_df
        .sort_values(
            "top_k"
        )
    )

    k = ordered[
        "top_k"
    ].to_numpy(
        dtype=float
    )

    exiffi = ordered[
        "exiffi_mean_relative_score_change"
    ].to_numpy(
        dtype=float
    )

    random = ordered[
        "random_mean_relative_score_change"
    ].to_numpy(
        dtype=float
    )

    if len(k) < 2:

        return {
            "exiffi_monotonic": np.nan,
            "random_monotonic": np.nan,
            "exiffi_k_spearman": np.nan,
            "random_k_spearman": np.nan,
        }

    exiffi_monotonic = bool(
        np.all(
            np.diff(
                exiffi
            )
            >= -1e-12
        )
    )

    random_monotonic = bool(
        np.all(
            np.diff(
                random
            )
            >= -1e-12
        )
    )

    exiffi_corr = spearmanr(
        k,
        exiffi,
    ).statistic

    random_corr = spearmanr(
        k,
        random,
    ).statistic

    return {
        "exiffi_monotonic": int(
            exiffi_monotonic
        ),
        "random_monotonic": int(
            random_monotonic
        ),
        "exiffi_k_spearman": float(
            exiffi_corr
        ),
        "random_k_spearman": float(
            random_corr
        ),
    }


# ============================================================
# Explanation mass relationship
# ============================================================


def evaluate_mass_relationship(
    observation_df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for k, group in (
        observation_df
        .groupby(
            "top_k"
        )
    ):

        mass = group[
            "top_k_mass"
        ].to_numpy(
            dtype=float
        )

        fidelity = group[
            "relative_score_change"
        ].to_numpy(
            dtype=float
        )

        valid = (
            np.isfinite(mass)
            & np.isfinite(fidelity)
        )

        if valid.sum() >= 3:

            result = spearmanr(
                mass[valid],
                fidelity[valid],
            )

            correlation = (
                result.statistic
            )

            p_value = (
                result.pvalue
            )

        else:

            correlation = np.nan
            p_value = np.nan

        rows.append(
            {
                "top_k": k,
                "spearman_top_k_mass_vs_relative_score_change": (
                    correlation
                ),
                "p_value": p_value,
                "n_observations": int(
                    valid.sum()
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Metric sufficiency decision
# ============================================================


def assess_metric_sufficiency(
    summary_df: pd.DataFrame,
    monotonicity: Dict[str, float],
) -> Tuple[
    str,
    str,
]:
    """
    Conservative research decision.

    The primary metric is considered sufficient only when:

        1. ExIFFI beats random at the majority of k values.
        2. The paired relative difference is positive.
        3. The bootstrap CI excludes zero for at least one
           meaningful k and preferably multiple k values.
        4. ExIFFI relative score change increases monotonically
           across k.

    This is intentionally conservative.

    The decision is a research interpretation aid, not an
    automatic scientific conclusion.
    """

    positive_count = int(
        (
            summary_df[
                "mean_relative_difference"
            ]
            > 0
        ).sum()
    )

    significant_count = int(
        (
            summary_df[
                "bootstrap_ci_lower_relative"
            ]
            > 0
        ).sum()
    )

    total = len(
        summary_df
    )

    monotonic = (
        monotonicity[
            "exiffi_monotonic"
        ]
        == 1
    )

    if (
        total > 0
        and positive_count
        >= int(
            np.ceil(
                total * 0.60
            )
        )
        and significant_count >= 2
        and monotonic
    ):

        decision = (
            "PRIMARY_METRIC_SUPPORTED"
        )

        rationale = (
            "Mean relative score change is supported "
            "as the primary fidelity metric because "
            "ExIFFI-selected interventions exceed the "
            "random control across most tested k values, "
            "the paired bootstrap evidence supports a "
            "positive ExIFFI advantage at multiple k "
            "values, and ExIFFI fidelity increases "
            "monotonically with k."
        )

    else:

        decision = (
            "PRIMARY_METRIC_NOT_YET_SUFFICIENT"
        )

        rationale = (
            "The current experiment does not provide "
            "sufficient evidence to declare mean relative "
            "score change alone as a complete primary "
            "fidelity metric. The metric should therefore "
            "be retained together with complementary "
            "measures such as absolute score change, "
            "top-k mass, and random-control comparisons."
        )

    return (
        decision,
        rationale,
    )


# ============================================================
# Diagnostics
# ============================================================


def build_score_diagnostic_row(
    window: int,
    diagnostics: Dict[str, float],
) -> Dict:

    return {
        "window": window,
        **diagnostics,
    }


# ============================================================
# Main experiment
# ============================================================


def run_experiment(
    transactions_path: str,
    explanation_dir: Path = DEFAULT_EXPLANATION_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    train_fraction: float = DEFAULT_TRAIN_FRACTION,
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    random_state: int = DEFAULT_RANDOM_STATE,
    sample_strategy: str = "stratified",
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    random_repeats: int = DEFAULT_RANDOM_REPEATS,
    n_bootstrap: int = DEFAULT_BOOTSTRAP,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    top_k_values: List[int] | None = None,
    windows: List[int] | None = None,
):

    if top_k_values is None:

        top_k_values = list(
            DEFAULT_TOP_K_VALUES
        )

    if windows is None:

        windows = list(
            DEFAULT_WINDOWS
        )

    print("=" * 70)

    print(
        "E4.2-T04 — EXIFFI FIDELITY METRIC SUFFICIENCY"
    )

    print("=" * 70)

    print(
        f"\nTraining fraction: "
        f"{train_fraction}"
    )

    print(
        f"Isolation Forest trees: "
        f"{n_estimators}"
    )

    print(
        f"Random seed: "
        f"{random_state}"
    )

    print(
        f"Top-k values: "
        f"{top_k_values}"
    )

    print(
        f"Random control repeats: "
        f"{random_repeats}"
    )

    print(
        f"Bootstrap samples: "
        f"{n_bootstrap}"
    )

    print(
        f"Windows evaluated: "
        f"{windows}"
    )

    # --------------------------------------------------------
    # Validate parameters
    # --------------------------------------------------------

    if random_repeats < 1:

        raise ValueError(
            "random_repeats must be >= 1."
        )

    if n_bootstrap < 100:

        raise ValueError(
            "bootstrap must be >= 100."
        )

    if any(
        k <= 0
        for k in top_k_values
    ):

        raise ValueError(
            "All top-k values must be > 0."
        )

    # --------------------------------------------------------
    # 1. Load explanations
    # --------------------------------------------------------

    print(
        "\n[1/8] Loading explanations..."
    )

    explanation_files = {}

    explanation_matrices = {}

    explanation_feature_names = {}

    for window in windows:

        df = load_explanation_file(
            window=window,
            explanation_dir=explanation_dir,
        )

        matrix, feature_names = (
            extract_explanation_matrix(
                df
            )
        )

        explanation_files[
            window
        ] = df

        explanation_matrices[
            window
        ] = matrix

        explanation_feature_names[
            window
        ] = feature_names

    reference_features = (
        explanation_feature_names[
            windows[0]
        ]
    )

    for window in windows[1:]:

        if (
            explanation_feature_names[
                window
            ]
            != reference_features
        ):

            raise ValueError(
                "Explanation feature sets differ "
                f"between Window {windows[0]} "
                f"and Window {window}."
            )

    # --------------------------------------------------------
    # 2. Load transaction dataset
    # --------------------------------------------------------

    print(
        "\n[2/8] Loading transaction dataset..."
    )

    transactions = pd.read_csv(
        transactions_path,
        low_memory=False,
    )

    print(
        f"Loaded "
        f"{len(transactions):,} "
        "transactions."
    )

    required_columns = [
        "TransactionDT",
        "TransactionID",
        "isFraud",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in transactions.columns
    ]

    if missing_columns:

        raise ValueError(
            "Transaction dataset is missing "
            f"required columns: {missing_columns}"
        )

    # --------------------------------------------------------
    # 3. Chronological windows
    # --------------------------------------------------------

    print(
        "\n[3/8] Creating chronological windows..."
    )

    train, future_windows = chronological_windows(
        transactions,
        time_col="TransactionDT",
        train_fraction=train_fraction,
        n_future_windows=4,
    )

    print(
        f"Training observations: "
        f"{len(train):,}"
    )

    for window_number, future in enumerate(
        future_windows,
        start=1,
    ):

        print(
            f"Window {window_number}: "
            f"{len(future):,} observations"
        )

    if max(windows) > len(
        future_windows
    ):

        raise ValueError(
            "Requested window exceeds "
            "available future windows."
        )

    # --------------------------------------------------------
    # 4. Training-only preprocessing
    # --------------------------------------------------------

    print(
        "\n[4/8] Fitting training-only preprocessing..."
    )

    preprocessor = NumericTransactionPreprocessor(
        policy=FeaturePolicy(
            target="isFraud",
            temporal="TransactionDT",
            identifier="TransactionID",
            max_missing_rate=DEFAULT_MAX_MISSING_RATE,
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

    if feature_names != reference_features:

        raise ValueError(
            "The reconstructed preprocessing feature "
            "set does not match the feature set stored "
            "in the explanation artifacts."
        )

    # --------------------------------------------------------
    # 5. Train Isolation Forest
    # --------------------------------------------------------

    print(
        "\n[5/8] Training static Isolation Forest..."
    )

    detector = IsolationForest(
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

    # --------------------------------------------------------
    # Output containers
    # --------------------------------------------------------

    all_observation_results = []

    all_random_results = []

    score_diagnostic_rows = []

    # --------------------------------------------------------
    # 6. Evaluate windows
    # --------------------------------------------------------

    print(
        "\n[6/8] Evaluating fidelity..."
    )

    for window in windows:

        print()
        print(
            f"Window {window}"
        )

        explanation_df = (
            explanation_files[
                window
            ]
        )

        explanation_matrix = (
            explanation_matrices[
                window
            ]
        )

        future = (
            future_windows[
                window - 1
            ]
        )

        X_window = (
            preprocessor
            .transform(
                future
            )
        )

        # ----------------------------------------------------
        # Reproduce sampling seed.
        #
        # The explanation-generation pipeline used:
        #
        # random_state + window_number
        #
        # therefore Window 1 = 43.
        # ----------------------------------------------------

        sampling_seed = (
            random_state
            + window
        )

        print(
            f"  Sample reconstruction..."
        )

        print(
            f"  Sampling strategy: "
            f"{sample_strategy}"
        )

        print(
            f"  Sampling seed: "
            f"{sampling_seed}"
        )

        (
            X_sample,
            y_sample,
            sample_positions,
        ) = reconstruct_sample(
            window=future,
            X_window=X_window,
            explanation_df=explanation_df,
            sample_strategy=sample_strategy,
            sampling_seed=sampling_seed,
        )

        print(
            "  Sample reconstruction validated."
        )

        print(
            f"  Observations evaluated: "
            f"{len(X_sample)}"
        )

        print(
            f"  Features: "
            f"{X_sample.shape[1]}"
        )

        print(
            f"  Fraud: "
            f"{int(y_sample.sum())}"
        )

        print(
            f"  Non-fraud: "
            f"{int(len(y_sample) - y_sample.sum())}"
        )

        # ----------------------------------------------------
        # Score diagnostic
        # ----------------------------------------------------

        diagnostics = (
            validate_original_scores(
                model=detector,
                X_sample=X_sample,
                explanation_df=explanation_df,
                feature_names=feature_names,
            )
        )

        diagnostics[
            "window"
        ] = window

        score_diagnostic_rows.append(
            diagnostics
        )

        # ----------------------------------------------------
        # Training median baseline
        # ----------------------------------------------------

        baseline = np.median(
            ensure_numeric_array(
                x_train
            ),
            axis=0,
        )

        # ----------------------------------------------------
        # ExIFFI
        # ----------------------------------------------------

        print(
            "  Evaluating ExIFFI interventions..."
        )

        exiffi_results = (
            evaluate_exiffi_fidelity(
                model=detector,
                X_sample=X_sample,
                explanation_matrix=explanation_matrix,
                baseline=baseline,
                feature_names=feature_names,
                k_values=top_k_values,
            )
        )

        exiffi_results[
            "window"
        ] = window

        exiffi_results[
            "isFraud"
        ] = [
            int(
                y_sample[
                    index
                ]
            )
            for index in (
                exiffi_results[
                    "sample_index"
                ]
            )
        ]

        all_observation_results.append(
            exiffi_results
        )

        # ----------------------------------------------------
        # Random controls
        # ----------------------------------------------------

        print(
            "  Evaluating random-feature controls..."
        )

        random_results = (
            evaluate_random_fidelity(
                model=detector,
                X_sample=X_sample,
                baseline=baseline,
                feature_names=feature_names,
                k_values=top_k_values,
                random_repeats=random_repeats,
                random_state=(
                    random_state
                    + 10000
                    + window
                ),
            )
        )

        random_results[
            "window"
        ] = window

        random_results[
            "isFraud"
        ] = [
            int(
                y_sample[
                    index
                ]
            )
            for index in (
                random_results[
                    "sample_index"
                ]
            )
        ]

        all_random_results.append(
            random_results
        )

        print(
            "  Evaluation complete."
        )

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    exiffi_df = pd.concat(
        all_observation_results,
        ignore_index=True,
    )

    random_df = pd.concat(
        all_random_results,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Random aggregate
    # --------------------------------------------------------

    random_aggregate = (
        random_df
        .groupby(
            [
                "window",
                "sample_index",
                "top_k",
            ],
            as_index=False,
        )
        .agg(
            random_mean_score_change=(
                "score_change",
                "mean",
            ),
            random_mean_absolute_score_change=(
                "absolute_score_change",
                "mean",
            ),
            random_mean_relative_score_change=(
                "relative_score_change",
                "mean",
            ),
            random_median_relative_score_change=(
                "relative_score_change",
                "median",
            ),
            random_std_relative_score_change=(
                "relative_score_change",
                "std",
            ),
        )
    )

    # --------------------------------------------------------
    # Observation-level paired results
    # --------------------------------------------------------

    observation_df = exiffi_df.merge(
        random_aggregate,
        on=[
            "window",
            "sample_index",
            "top_k",
        ],
        how="left",
        validate="one_to_one",
    )

    observation_df[
        "exiffi_minus_random_relative"
    ] = (
        observation_df[
            "relative_score_change"
        ]
        - observation_df[
            "random_mean_relative_score_change"
        ]
    )

    observation_df[
        "exiffi_minus_random_absolute"
    ] = (
        observation_df[
            "absolute_score_change"
        ]
        - observation_df[
            "random_mean_absolute_score_change"
        ]
    )

    observation_df[
        "relative_effect_ratio"
    ] = (
        observation_df[
            "relative_score_change"
        ]
        / np.maximum(
            observation_df[
                "random_mean_relative_score_change"
            ],
            1e-12,
        )
    )

    # --------------------------------------------------------
    # 7. Statistical analysis
    # --------------------------------------------------------

    print(
        "\n[7/8] Assessing fidelity metric sufficiency..."
    )

    summary_rows = []

    bootstrap_rows = []

    mass_rows = []

    monotonicity_rows = []

    for window in windows:

        window_observations = (
            observation_df[
                observation_df[
                    "window"
                ]
                == window
            ]
        )

        for k in top_k_values:

            subset = (
                window_observations[
                    window_observations[
                        "top_k"
                    ]
                    == k
                ]
            )

            relative_difference = (
                subset[
                    "exiffi_minus_random_relative"
                ]
                .to_numpy(
                    dtype=float
                )
            )

            absolute_difference = (
                subset[
                    "exiffi_minus_random_absolute"
                ]
                .to_numpy(
                    dtype=float
                )
            )

            relative_rng = (
                np.random.default_rng(
                    bootstrap_seed
                    + window * 100
                    + k
                )
            )

            (
                relative_mean,
                relative_lower,
                relative_upper,
            ) = bootstrap_mean_difference(
                values=relative_difference,
                n_bootstrap=n_bootstrap,
                rng=relative_rng,
            )

            relative_p = (
                bootstrap_two_sided_pvalue(
                    values=relative_difference,
                    n_bootstrap=n_bootstrap,
                    rng=np.random.default_rng(
                        bootstrap_seed
                        + 10000
                        + window * 100
                        + k
                    ),
                )
            )

            (
                absolute_mean,
                absolute_lower,
                absolute_upper,
            ) = bootstrap_mean_difference(
                values=absolute_difference,
                n_bootstrap=n_bootstrap,
                rng=np.random.default_rng(
                    bootstrap_seed
                    + 20000
                    + window * 100
                    + k
                ),
            )

            absolute_p = (
                bootstrap_two_sided_pvalue(
                    values=absolute_difference,
                    n_bootstrap=n_bootstrap,
                    rng=np.random.default_rng(
                        bootstrap_seed
                        + 30000
                        + window * 100
                        + k
                    ),
                )
            )

            exiffi_relative_mean = float(
                subset[
                    "relative_score_change"
                ].mean()
            )

            random_relative_mean = float(
                subset[
                    "random_mean_relative_score_change"
                ].mean()
            )

            exiffi_absolute_mean = float(
                subset[
                    "absolute_score_change"
                ].abs().mean()
            )

            random_absolute_mean = float(
                subset[
                    "random_mean_absolute_score_change"
                ].abs().mean()
            )

            top_mass_mean = float(
                subset[
                    "top_k_mass"
                ].mean()
            )

            ratio = (
                exiffi_relative_mean
                / max(
                    random_relative_mean,
                    1e-12,
                )
            )

            significant = bool(
                relative_lower > 0
            )

            summary_rows.append(
                {
                    "window": window,
                    "top_k": k,
                    "n_observations": len(
                        subset
                    ),
                    "mean_top_k_mass": (
                        top_mass_mean
                    ),
                    "exiffi_mean_relative_score_change": (
                        exiffi_relative_mean
                    ),
                    "random_mean_relative_score_change": (
                        random_relative_mean
                    ),
                    "mean_relative_difference": (
                        relative_mean
                    ),
                    "bootstrap_ci_lower_relative": (
                        relative_lower
                    ),
                    "bootstrap_ci_upper_relative": (
                        relative_upper
                    ),
                    "bootstrap_pvalue_relative": (
                        relative_p
                    ),
                    "relative_difference_ci_excludes_zero": (
                        significant
                    ),
                    "exiffi_mean_absolute_score_change": (
                        exiffi_absolute_mean
                    ),
                    "random_mean_absolute_score_change": (
                        random_absolute_mean
                    ),
                    "mean_absolute_difference": (
                        absolute_mean
                    ),
                    "bootstrap_ci_lower_absolute": (
                        absolute_lower
                    ),
                    "bootstrap_ci_upper_absolute": (
                        absolute_upper
                    ),
                    "bootstrap_pvalue_absolute": (
                        absolute_p
                    ),
                    "exiffi_to_random_relative_ratio": (
                        ratio
                    ),
                }
            )

            bootstrap_rows.append(
                {
                    "window": window,
                    "top_k": k,
                    "metric": (
                        "relative_score_change"
                    ),
                    "observed_difference": (
                        relative_mean
                    ),
                    "ci_lower": (
                        relative_lower
                    ),
                    "ci_upper": (
                        relative_upper
                    ),
                    "bootstrap_pvalue": (
                        relative_p
                    ),
                    "n_bootstrap": (
                        n_bootstrap
                    ),
                }
            )

            bootstrap_rows.append(
                {
                    "window": window,
                    "top_k": k,
                    "metric": (
                        "absolute_score_change"
                    ),
                    "observed_difference": (
                        absolute_mean
                    ),
                    "ci_lower": (
                        absolute_lower
                    ),
                    "ci_upper": (
                        absolute_upper
                    ),
                    "bootstrap_pvalue": (
                        absolute_p
                    ),
                    "n_bootstrap": (
                        n_bootstrap
                    ),
                }
            )

        # ----------------------------------------------------
        # Window monotonicity
        # ----------------------------------------------------

        window_summary = pd.DataFrame(
            [
                row
                for row in summary_rows
                if row["window"] == window
            ]
        ).sort_values(
            "top_k"
        )

        k_array = (
            window_summary[
                "top_k"
            ]
            .to_numpy(
                dtype=float
            )
        )

        relative_array = (
            window_summary[
                "exiffi_mean_relative_score_change"
            ]
            .to_numpy(
                dtype=float
            )
        )

        if len(k_array) >= 2:

            monotonic = int(
                np.all(
                    np.diff(
                        relative_array
                    )
                    >= -1e-12
                )
            )

            corr = spearmanr(
                k_array,
                relative_array,
            )

            k_spearman = float(
                corr.statistic
            )

        else:

            monotonic = np.nan
            k_spearman = np.nan

        monotonicity_rows.append(
            {
                "window": window,
                "exiffi_monotonic_with_k": (
                    monotonic
                ),
                "exiffi_k_spearman": (
                    k_spearman
                ),
            }
        )

        # ----------------------------------------------------
        # Top-k mass relationship
        # ----------------------------------------------------

        for k in top_k_values:

            subset = (
                window_observations[
                    window_observations[
                        "top_k"
                    ]
                    == k
                ]
            )

            mass = subset[
                "top_k_mass"
            ].to_numpy(
                dtype=float
            )

            fidelity = subset[
                "relative_score_change"
            ].to_numpy(
                dtype=float
            )

            valid = (
                np.isfinite(
                    mass
                )
                & np.isfinite(
                    fidelity
                )
            )

            if valid.sum() >= 3:

                correlation_result = (
                    spearmanr(
                        mass[valid],
                        fidelity[valid],
                    )
                )

                correlation = float(
                    correlation_result.statistic
                )

                p_value = float(
                    correlation_result.pvalue
                )

            else:

                correlation = np.nan
                p_value = np.nan

            mass_rows.append(
                {
                    "window": window,
                    "top_k": k,
                    "spearman_top_k_mass_vs_relative_score_change": (
                        correlation
                    ),
                    "p_value": p_value,
                    "n_observations": int(
                        valid.sum()
                    ),
                }
            )

    summary_df = pd.DataFrame(
        summary_rows
    )

    bootstrap_df = pd.DataFrame(
        bootstrap_rows
    )

    monotonicity_df = pd.DataFrame(
        monotonicity_rows
    )

    mass_df = pd.DataFrame(
        mass_rows
    )

    # --------------------------------------------------------
    # Overall sufficiency decision
    # --------------------------------------------------------

    positive_count = int(
        (
            summary_df[
                "mean_relative_difference"
            ]
            > 0
        ).sum()
    )

    significant_count = int(
        (
            summary_df[
                "relative_difference_ci_excludes_zero"
            ]
        ).sum()
    )

    total_tests = len(
        summary_df
    )

    majority_positive = (
        positive_count
        >= int(
            np.ceil(
                total_tests
                * 0.60
            )
        )
    )

    multiple_significant = (
        significant_count >= 2
    )

    monotonic_all = bool(
        (
            monotonicity_df[
                "exiffi_monotonic_with_k"
            ]
            == 1
        ).all()
    )

    if (
        majority_positive
        and multiple_significant
        and monotonic_all
    ):

        decision = (
            "PRIMARY_METRIC_SUPPORTED"
        )

        rationale = (
            "Mean relative score change is supported "
            "as the primary fidelity metric under the "
            "predefined E4.2-T04 criteria: ExIFFI "
            "interventions exceed random controls across "
            "most tested conditions, the paired bootstrap "
            "confidence interval excludes zero for at "
            "least two tested k values, and the ExIFFI "
            "fidelity response is monotonic with k."
        )

    else:

        decision = (
            "PRIMARY_METRIC_NOT_YET_SUFFICIENT"
        )

        rationale = (
            "The current results do not satisfy all "
            "predefined sufficiency criteria. Mean "
            "relative score change should therefore be "
            "retained as a primary candidate metric but "
            "reported alongside complementary fidelity "
            "measures rather than treated as sufficient "
            "in isolation."
        )

    decision_df = pd.DataFrame(
        [
            {
                "decision": decision,
                "rationale": rationale,
                "positive_exiffi_advantage_tests": (
                    positive_count
                ),
                "total_tests": (
                    total_tests
                ),
                "significant_bootstrap_tests": (
                    significant_count
                ),
                "majority_positive": (
                    majority_positive
                ),
                "multiple_significant": (
                    multiple_significant
                ),
                "monotonic_all_windows": (
                    monotonic_all
                ),
                "random_control_repeats": (
                    random_repeats
                ),
                "bootstrap_samples": (
                    n_bootstrap
                ),
            }
        ]
    )

    # --------------------------------------------------------
    # 8. Save
    # --------------------------------------------------------

    print(
        "\n[8/8] Saving metric-sufficiency results..."
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    observation_path = (
        output_dir
        / "static_if_exiffi_fidelity_metric_sufficiency_observation.csv"
    )

    random_path = (
        output_dir
        / "static_if_exiffi_fidelity_metric_sufficiency_random_control.csv"
    )

    summary_path = (
        output_dir
        / "static_if_exiffi_fidelity_metric_sufficiency_summary.csv"
    )

    bootstrap_path = (
        output_dir
        / "static_if_exiffi_fidelity_metric_sufficiency_bootstrap.csv"
    )

    mass_path = (
        output_dir
        / "static_if_exiffi_fidelity_metric_sufficiency_mass_relationship.csv"
    )

    monotonicity_path = (
        output_dir
        / "static_if_exiffi_fidelity_metric_sufficiency_monotonicity.csv"
    )

    diagnostics_path = (
        output_dir
        / "static_if_exiffi_fidelity_score_reproducibility_diagnostics.csv"
    )

    decision_path = (
        output_dir
        / "static_if_exiffi_fidelity_metric_sufficiency_decision.csv"
    )

    observation_df.to_csv(
        observation_path,
        index=False,
    )

    random_df.to_csv(
        random_path,
        index=False,
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    bootstrap_df.to_csv(
        bootstrap_path,
        index=False,
    )

    mass_df.to_csv(
        mass_path,
        index=False,
    )

    monotonicity_df.to_csv(
        monotonicity_path,
        index=False,
    )

    diagnostics_df = pd.DataFrame(
        score_diagnostic_rows
    )

    diagnostics_df.to_csv(
        diagnostics_path,
        index=False,
    )

    decision_df.to_csv(
        decision_path,
        index=False,
    )

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print()
    print(
        "Results saved:"
    )

    print(
        f"  Observation-level:\n"
        f"    {observation_path}"
    )

    print(
        f"  Random controls:\n"
        f"    {random_path}"
    )

    print(
        f"  Summary:\n"
        f"    {summary_path}"
    )

    print(
        f"  Bootstrap:\n"
        f"    {bootstrap_path}"
    )

    print(
        f"  Mass relationship:\n"
        f"    {mass_path}"
    )

    print(
        f"  Monotonicity:\n"
        f"    {monotonicity_path}"
    )

    print(
        f"  Score diagnostics:\n"
        f"    {diagnostics_path}"
    )

    print(
        f"  Decision:\n"
        f"    {decision_path}"
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "E4.2-T04 — METRIC SUFFICIENCY SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        "\nFidelity comparison:"
    )

    display_columns = [
        "window",
        "top_k",
        "mean_top_k_mass",
        "exiffi_mean_relative_score_change",
        "random_mean_relative_score_change",
        "mean_relative_difference",
        "bootstrap_ci_lower_relative",
        "bootstrap_ci_upper_relative",
        "bootstrap_pvalue_relative",
        "exiffi_to_random_relative_ratio",
    ]

    print(
        summary_df[
            display_columns
        ].to_string(
            index=False
        )
    )

    print(
        "\nMonotonicity:"
    )

    print(
        monotonicity_df.to_string(
            index=False
        )
    )

    print(
        "\nScore reproducibility diagnostics:"
    )

    print(
        diagnostics_df.to_string(
            index=False
        )
    )

    print(
        "\nMetric sufficiency decision:"
    )

    print(
        f"  {decision}"
    )

    print(
        f"\n  {rationale}"
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "E4.2-T04 COMPLETE"
    )

    print(
        "=" * 70
    )

    return {
        "observation": observation_df,
        "random": random_df,
        "summary": summary_df,
        "bootstrap": bootstrap_df,
        "mass_relationship": mass_df,
        "monotonicity": monotonicity_df,
        "score_diagnostics": diagnostics_df,
        "decision": decision_df,
    }


# ============================================================
# CLI
# ============================================================


def parse_top_k(
    value: str,
) -> List[int]:

    values = []

    for item in value.split(","):

        item = item.strip()

        if not item:
            continue

        values.append(
            int(item)
        )

    if not values:

        raise argparse.ArgumentTypeError(
            "At least one top-k value is required."
        )

    return values


def parse_windows(
    value: str,
) -> List[int]:

    values = []

    for item in value.split(","):

        item = item.strip()

        if not item:
            continue

        values.append(
            int(item)
        )

    if not values:

        raise argparse.ArgumentTypeError(
            "At least one window is required."
        )

    return values


def main():

    parser = argparse.ArgumentParser(
        description=(
            "E4.2-T04 — Evaluate whether "
            "relative model-score change is "
            "sufficient as the primary "
            "ExIFFI fidelity metric."
        )
    )

    parser.add_argument(
        "--transactions",
        required=True,
        help=(
            "Path to IEEE-CIS "
            "train_transaction.csv"
        ),
    )

    parser.add_argument(
        "--explanation-dir",
        default=str(
            DEFAULT_EXPLANATION_DIR
        ),
        help=(
            "Directory containing temporal "
            "ExIFFI explanation CSV files."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=str(
            DEFAULT_OUTPUT_DIR
        ),
        help=(
            "Directory for E4.2-T04 results."
        ),
    )

    parser.add_argument(
        "--train-fraction",
        type=float,
        default=DEFAULT_TRAIN_FRACTION,
    )

    parser.add_argument(
        "--estimators",
        type=int,
        default=DEFAULT_N_ESTIMATORS,
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
    )

    parser.add_argument(
        "--random-repeats",
        type=int,
        default=DEFAULT_RANDOM_REPEATS,
    )

    parser.add_argument(
        "--bootstrap",
        type=int,
        default=DEFAULT_BOOTSTRAP,
    )

    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=DEFAULT_BOOTSTRAP_SEED,
    )

    parser.add_argument(
        "--top-k",
        type=parse_top_k,
        default=list(
            DEFAULT_TOP_K_VALUES
        ),
        help=(
            "Comma-separated top-k values. "
            "Example: 1,3,5,10,20"
        ),
    )

    parser.add_argument(
        "--windows",
        type=parse_windows,
        default=list(
            DEFAULT_WINDOWS
        ),
        help=(
            "Comma-separated temporal windows. "
            "Example: 1 or 1,2,3,4"
        ),
    )

    args = parser.parse_args()

    run_experiment(
        transactions_path=args.transactions,
        explanation_dir=Path(
            args.explanation_dir
        ),
        output_dir=Path(
            args.output_dir
        ),
        train_fraction=args.train_fraction,
        n_estimators=args.estimators,
        random_state=args.random_state,
        sample_size=args.sample_size,
        random_repeats=args.random_repeats,
        n_bootstrap=args.bootstrap,
        bootstrap_seed=args.bootstrap_seed,
        top_k_values=args.top_k,
        windows=args.windows,
    )


if __name__ == "__main__":
    main()