from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from efd.data.preprocessing import (
    FeaturePolicy,
    NumericTransactionPreprocessor,
)

from efd.data.temporal import (
    chronological_windows,
)

from efd.models.isolation_forest import (
    IsolationForestDetector,
)

from efd.evaluation.explanation_fidelity import (
    evaluate_local_fidelity,
    score_samples_with_feature_names,
)


# ============================================================
# Configuration
# ============================================================

DEFAULT_TRAIN_FRACTION = 0.60
DEFAULT_N_FUTURE_WINDOWS = 4
DEFAULT_N_ESTIMATORS = 300
DEFAULT_RANDOM_STATE = 42
DEFAULT_SAMPLE_SIZE = 100
DEFAULT_SAMPLE_STRATEGY = "stratified"

DEFAULT_INPUT_EXPLANATION_DIR = Path(
    "results/explanations"
)

DEFAULT_OUTPUT_DIR = Path(
    "results/tables"
)

DEFAULT_K_VALUES = [
    1,
    3,
    5,
    10,
    20,
]

EXPLANATION_FILENAME_TEMPLATE = (
    "static_if_exiffi_w{window}_sample.csv"
)


# ============================================================
# Explanation adapter
# ============================================================


class ExplanationAdapter:
    """
    Minimal adapter that allows an existing ExIFFI
    explanation vector loaded from CSV to be evaluated
    by the existing fidelity implementation.

    The project already stores the normalized ExIFFI
    explanation as:

        importance__<feature>

    Therefore there is no need to regenerate explanations
    for this ticket.
    """

    def __init__(
        self,
        sample_index: int,
        normalized: np.ndarray,
        transaction_id=None,
    ):
        self.sample_index = int(
            sample_index
        )
        self.transaction_id = transaction_id

        self.normalized = np.asarray(
            normalized,
            dtype=float,
        )


# ============================================================
# Loading explanation files
# ============================================================


def load_explanation_file(
    window: int,
    explanation_dir: Path,
) -> pd.DataFrame:
    """
    Load the already-generated temporal ExIFFI
    explanation matrix for one future window.
    """

    path = (
        explanation_dir
        / EXPLANATION_FILENAME_TEMPLATE.format(
            window=window
        )
    )

    if not path.exists():
        raise FileNotFoundError(
            "\nMissing temporal explanation file:\n"
            f"  {path}\n\n"
            "Run generate_exiffi_temporal_explanations.py "
            "before running this fidelity experiment."
        )

    df = pd.read_csv(
        path
    )

    # The temporal explanation files currently produced by
    # generate_exiffi_temporal_explanations.py contain a LOCAL
    # sample_index (0..n_sampled-1), not TransactionID.  Therefore
    # TransactionID must not be required here.  The exact sampled
    # observations are reconstructed deterministically from the
    # temporal window using the same sampling strategy and seed.
    required_columns = {
        "sample_index",
        "isFraud",
        "anomaly_score",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"Window {window} explanation file "
            f"is missing required columns: "
            f"{sorted(missing)}"
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
            f"No ExIFFI importance columns found "
            f"in {path}"
        )

    print(
        f"Window {window}: "
        f"{len(df):,} explanation rows, "
        f"{len(importance_columns):,} features"
    )

    return df


def extract_importance_matrix(
    explanation_df: pd.DataFrame,
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract normalized ExIFFI importance vectors from
    the wide-format explanation CSV.
    """

    importance_columns = [
        column
        for column in explanation_df.columns
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
        for column in importance_columns
    ]

    importance_matrix = (
        explanation_df[
            importance_columns
        ]
        .to_numpy(
            dtype=float
        )
    )

    return (
        importance_matrix,
        feature_names,
    )


# ============================================================
# Validation
# ============================================================


def validate_explanation_matrix(
    explanation_df: pd.DataFrame,
    importance_matrix: np.ndarray,
    feature_names: List[str],
    expected_feature_names: List[str],
    window: int,
) -> None:
    """
    Validate the explanation matrix before fidelity
    evaluation.
    """

    if (
        feature_names
        != expected_feature_names
    ):
        raise ValueError(
            "Feature schema mismatch in "
            f"Window {window}.\n"
            "The explanation feature order does not "
            "match the preprocessing/model feature order."
        )

    if importance_matrix.ndim != 2:
        raise ValueError(
            f"Window {window}: importance matrix "
            "must be two-dimensional."
        )

    if importance_matrix.shape[1] != len(
        expected_feature_names
    ):
        raise ValueError(
            f"Window {window}: explanation matrix "
            "contains {importance_matrix.shape[1]} "
            "features but the model expects "
            f"{len(expected_feature_names)}."
        )

    if importance_matrix.shape[0] != len(
        explanation_df
    ):
        raise ValueError(
            f"Window {window}: explanation row count "
            "does not match importance matrix."
        )

    if not np.isfinite(
        importance_matrix
    ).all():
        raise ValueError(
            f"Window {window}: explanation matrix "
            "contains NaN or infinite values."
        )

    if (
        importance_matrix < -1e-12
    ).any():
        raise ValueError(
            f"Window {window}: ExIFFI normalized "
            "importance contains negative values."
        )

    sample_indices = explanation_df[
        "sample_index"
    ].to_numpy(dtype=int)

    if len(np.unique(sample_indices)) != len(sample_indices):
        raise ValueError(
            f"Window {window}: duplicate sample_index values found."
        )



# ============================================================
# Sample reconstruction
# ============================================================


def stratified_sample_indices(
    y: pd.Series,
    sample_size: int,
    random_state: int,
) -> np.ndarray:
    """
    Reproduce the stratified sampling used by
    generate_exiffi_temporal_explanations.py.

    IMPORTANT:
    The generator samples positional rows from the chronological
    window, then shuffles those selected positions.  The explanation
    CSV stores only the resulting local sample_index (0..n-1), so the
    original positions must be reconstructed using this exact procedure.
    """

    if sample_size <= 0:
        raise ValueError("sample_size must be greater than zero.")

    if sample_size > len(y):
        raise ValueError(
            "sample_size cannot exceed the number of observations "
            "in the temporal window."
        )

    rng = np.random.default_rng(random_state)
    y_values = y.to_numpy()
    classes = np.unique(y_values)

    if len(classes) < 2:
        raise ValueError(
            "Stratified sampling requires at least two classes."
        )

    n_classes = len(classes)
    base_per_class = sample_size // n_classes
    remainder = sample_size % n_classes

    selected_indices: list[int] = []

    for class_position, class_value in enumerate(classes):
        class_indices = np.flatnonzero(
            y_values == class_value
        )

        target_n = base_per_class
        if class_position < remainder:
            target_n += 1

        if target_n > len(class_indices):
            target_n = len(class_indices)

        if target_n > 0:
            selected = rng.choice(
                class_indices,
                size=target_n,
                replace=False,
            )
            selected_indices.extend(
                selected.tolist()
            )

    selected_set = set(selected_indices)
    remaining_n = sample_size - len(selected_indices)

    if remaining_n > 0:
        available = np.asarray(
            [
                i
                for i in range(len(y_values))
                if i not in selected_set
            ],
            dtype=int,
        )

        if remaining_n > len(available):
            raise ValueError(
                "Unable to reproduce the requested stratified sample."
            )

        additional = rng.choice(
            available,
            size=remaining_n,
            replace=False,
        )
        selected_indices.extend(
            additional.tolist()
        )

    selected_indices = np.asarray(
        selected_indices,
        dtype=int,
    )

    # This shuffle is part of the generator's sampling algorithm and
    # MUST be retained for exact reconstruction.
    rng.shuffle(selected_indices)

    return selected_indices


def random_sample_indices(
    n_observations: int,
    sample_size: int,
    random_state: int,
) -> np.ndarray:
    """Reproduce the generator's random sampling strategy."""

    if sample_size <= 0:
        raise ValueError("sample_size must be greater than zero.")

    if sample_size > n_observations:
        raise ValueError(
            "sample_size cannot exceed the number of observations "
            "in the temporal window."
        )

    rng = np.random.default_rng(random_state)

    return rng.choice(
        n_observations,
        size=sample_size,
        replace=False,
    )


def sequential_sample_indices(
    n_observations: int,
    sample_size: int,
) -> np.ndarray:
    """Reproduce the generator's sequential sampling strategy."""

    if sample_size <= 0:
        raise ValueError("sample_size must be greater than zero.")

    if sample_size > n_observations:
        raise ValueError(
            "sample_size cannot exceed the number of observations "
            "in the temporal window."
        )

    return np.arange(
        sample_size,
        dtype=int,
    )


def reconstruct_sample_positions(
    window: pd.DataFrame,
    sample_size: int,
    sample_strategy: str,
    random_state: int,
    window_number: int,
) -> np.ndarray:
    """
    Reproduce the exact positional sample generated for a temporal
    window.

    The explanation generator uses random_state + window_number for
    each window.  This function intentionally mirrors that behaviour.
    """

    seed = random_state + window_number

    if sample_strategy == "stratified":
        return stratified_sample_indices(
            y=window["isFraud"],
            sample_size=sample_size,
            random_state=seed,
        )

    if sample_strategy == "random":
        return random_sample_indices(
            n_observations=len(window),
            sample_size=sample_size,
            random_state=seed,
        )

    if sample_strategy == "sequential":
        return sequential_sample_indices(
            n_observations=len(window),
            sample_size=sample_size,
        )

    raise ValueError(
        f"Unsupported sample strategy: {sample_strategy}. "
        "Expected one of: sequential, random, stratified."
    )


def reconstruct_window_sample(
    window: pd.DataFrame,
    transformed_window: pd.DataFrame,
    explanation_df: pd.DataFrame,
    window_number: int,
    sample_strategy: str,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reconstruct the exact sample represented by a temporal ExIFFI file.

    The currently generated explanation files contain:

        sample_index
        isFraud
        anomaly_score
        importance__<feature>

    Here ``sample_index`` is NOT an index into the full temporal window.
    The generator creates ExIFFI explanations with sample_index=i while
    iterating over the already-selected sample.  Consequently, the exact
    source rows must be reproduced from the same sampling configuration.

    TransactionID, when present in the source dataset, is returned for
    auditability, but it is not required in the explanation CSV.
    """

    if "isFraud" not in window.columns:
        raise ValueError(
            "Transaction dataset does not contain 'isFraud'."
        )

    n_explanations = len(explanation_df)

    if n_explanations == 0:
        raise ValueError(
            f"Window {window_number}: explanation file contains no rows."
        )

    sample_indices = explanation_df[
        "sample_index"
    ].to_numpy(dtype=int)

    # The generator assigns sample_index=i in the final sampled order.
    expected_local_indices = np.arange(
        n_explanations,
        dtype=int,
    )

    if not np.array_equal(
        sample_indices,
        expected_local_indices,
    ):
        raise ValueError(
            f"Window {window_number}: sample_index does not represent "
            "the expected local explanation order. Expected 0.."
            f"{n_explanations - 1}, but received "
            f"{sample_indices[:20].tolist()}..."
        )

    if n_explanations > len(window):
        raise ValueError(
            f"Window {window_number}: explanation rows "
            f"({n_explanations}) exceed temporal-window rows "
            f"({len(window)})."
        )

    # Reproduce the exact source-row positions used by the explanation
    # generator.  This is the critical alignment fix.
    source_positions = reconstruct_sample_positions(
        window=window,
        sample_size=n_explanations,
        sample_strategy=sample_strategy,
        random_state=random_state,
        window_number=window_number,
    )

    X_sample = (
        transformed_window
        .iloc[source_positions]
        .to_numpy(dtype=float)
    )

    y_sample = (
        window["isFraud"]
        .iloc[source_positions]
        .to_numpy(dtype=int)
    )

    csv_labels = explanation_df[
        "isFraud"
    ].to_numpy(dtype=int)

    if not np.array_equal(
        y_sample,
        csv_labels,
    ):
        mismatch_positions = np.where(
            y_sample != csv_labels
        )[0]

        details = []
        for position in mismatch_positions[:10]:
            detail = {
                "local_sample_index": int(position),
                "source_window_position": int(
                    source_positions[position]
                ),
                "reconstructed_isFraud": int(
                    y_sample[position]
                ),
                "explanation_isFraud": int(
                    csv_labels[position]
                ),
            }

            if "TransactionID" in window.columns:
                detail["TransactionID"] = window[
                    "TransactionID"
                ].iloc[source_positions[position]]

            details.append(detail)

        raise ValueError(
            f"Window {window_number}: fraud labels reconstructed from "
            "the temporal window do not match the labels stored in the "
            "explanation file.\n"
            f"Sampling strategy used for reconstruction: {sample_strategy}\n"
            f"Random state: {random_state + window_number}\n"
            f"Mismatch count: {len(mismatch_positions)}\n"
            f"First mismatches: {details}\n\n"
            "This normally means the explanation CSV was generated with "
            "a different sampling strategy, sample seed, or version of "
            "the temporal window construction code."
        )

    if "TransactionID" in window.columns:
        transaction_ids = (
            window["TransactionID"]
            .iloc[source_positions]
            .to_numpy()
        )
    else:
        # Keep the output schema stable even for datasets without an ID.
        transaction_ids = np.arange(
            len(source_positions),
            dtype=int,
        )

    print(
        f"  Reconstructed sample using {sample_strategy} sampling "
        f"with seed {random_state + window_number}."
    )

    return (
        X_sample,
        y_sample,
        transaction_ids,
    )


# ============================================================
# Score consistency
# ============================================================


def validate_original_scores(
    model,
    X_sample: np.ndarray,
    explanation_df: pd.DataFrame,
    tolerance: float = 1e-8,
) -> None:
    """
    Verify that the model's current score agrees with
    the anomaly score stored when the explanations were
    generated.

    The project convention is:

        sklearn score_samples:
            larger = more normal

        research anomaly_score:
            -score_samples
    """

    model_scores = score_samples_with_feature_names(
        model,
        X_sample,
    )

    expected_anomaly_scores = (
        -model_scores
    )

    stored_anomaly_scores = (
        explanation_df[
            "anomaly_score"
        ]
        .to_numpy(
            dtype=float
        )
    )

    if not np.allclose(
        expected_anomaly_scores,
        stored_anomaly_scores,
        rtol=tolerance,
        atol=tolerance,
    ):
        maximum_difference = float(
            np.max(
                np.abs(
                    expected_anomaly_scores
                    - stored_anomaly_scores
                )
            )
        )

        raise ValueError(
            "Stored ExIFFI anomaly scores do not "
            "match the reconstructed Isolation Forest "
            "scores.\n"
            f"Maximum absolute difference: "
            f"{maximum_difference:.12g}"
        )


# ============================================================
# Fidelity evaluation
# ============================================================


def evaluate_window_fidelity(
    window: int,
    model,
    X_sample: np.ndarray,
    y_sample: np.ndarray,
    transaction_ids: np.ndarray,
    explanation_df: pd.DataFrame,
    importance_matrix: np.ndarray,
    baseline: np.ndarray,
    feature_names: List[str],
    k_values: List[int],
) -> pd.DataFrame:
    """
    Evaluate perturbation fidelity for one temporal window.
    """

    explanations = [
        ExplanationAdapter(
            sample_index=int(
                explanation_df.iloc[i][
                    "sample_index"
                ]
            ),
            normalized=importance_matrix[i],
            transaction_id=transaction_ids[i],
        )
        for i in range(
            len(explanation_df)
        )
    ]

    rows = []

    for i, explanation in enumerate(
        explanations
    ):

        x = X_sample[i]

        for k in k_values:

            if k > X_sample.shape[1]:
                raise ValueError(
                    f"k={k} exceeds the number of "
                    f"features ({X_sample.shape[1]})."
                )

            result = evaluate_local_fidelity(
                model=model,
                x=x,
                explanation=explanation,
                baseline=baseline,
                k=k,
            )

            rows.append(
                {
                    "window": window,
                    "TransactionID": transaction_ids[i],
                    "sample_index": (
                        explanation.sample_index
                    ),
                    "isFraud": int(
                        y_sample[i]
                    ),
                    "class": (
                        "fraud"
                        if int(y_sample[i]) == 1
                        else "nonfraud"
                    ),
                    "top_k": int(k),
                    "top_k_mass": (
                        result.top_k_mass
                    ),
                    "baseline_score": (
                        result.baseline_score
                    ),
                    "perturbed_score": (
                        result.perturbed_score
                    ),
                    "score_change": (
                        result.score_change
                    ),
                    "absolute_score_change": abs(
                        result.score_change
                    ),
                    "relative_score_change": (
                        result.relative_score_change
                    ),
                    "n_features": (
                        len(feature_names)
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Summary helpers
# ============================================================


def build_summary(
    fidelity_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build overall temporal fidelity summary.
    """

    summary = (
        fidelity_df
        .groupby(
            [
                "window",
                "top_k",
            ],
            as_index=False,
        )
        .agg(
            n_observations=(
                "sample_index",
                "nunique",
            ),
            mean_top_k_mass=(
                "top_k_mass",
                "mean",
            ),
            median_top_k_mass=(
                "top_k_mass",
                "median",
            ),
            mean_score_change=(
                "score_change",
                "mean",
            ),
            median_score_change=(
                "score_change",
                "median",
            ),
            mean_absolute_score_change=(
                "absolute_score_change",
                "mean",
            ),
            median_absolute_score_change=(
                "absolute_score_change",
                "median",
            ),
            mean_relative_score_change=(
                "relative_score_change",
                "mean",
            ),
            median_relative_score_change=(
                "relative_score_change",
                "median",
            ),
        )
        .sort_values(
            [
                "window",
                "top_k",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return summary


def build_class_summary(
    fidelity_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build fraud/non-fraud temporal fidelity summary.
    """

    summary = (
        fidelity_df
        .groupby(
            [
                "window",
                "isFraud",
                "class",
                "top_k",
            ],
            as_index=False,
        )
        .agg(
            n_observations=(
                "sample_index",
                "nunique",
            ),
            mean_top_k_mass=(
                "top_k_mass",
                "mean",
            ),
            median_top_k_mass=(
                "top_k_mass",
                "median",
            ),
            mean_score_change=(
                "score_change",
                "mean",
            ),
            median_score_change=(
                "score_change",
                "median",
            ),
            mean_absolute_score_change=(
                "absolute_score_change",
                "mean",
            ),
            median_absolute_score_change=(
                "absolute_score_change",
                "median",
            ),
            mean_relative_score_change=(
                "relative_score_change",
                "mean",
            ),
            median_relative_score_change=(
                "relative_score_change",
                "median",
            ),
        )
        .sort_values(
            [
                "window",
                "isFraud",
                "top_k",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return summary


# ============================================================
# Temporal fidelity trajectory
# ============================================================


def build_fidelity_trajectory(
    summary_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate temporal changes in the primary fidelity
    measure.

    The primary fidelity measure remains:

        mean_relative_score_change

    Additional quantities:

        change_from_previous_window
        absolute_change_from_previous_window
        change_from_window_1
        absolute_change_from_window_1
    """

    rows = []

    for top_k, group in summary_df.groupby(
        "top_k"
    ):

        group = (
            group
            .sort_values("window")
            .reset_index(drop=True)
        )

        baseline_value = float(
            group.iloc[0][
                "mean_relative_score_change"
            ]
        )

        previous_value = None

        for _, row in group.iterrows():

            current_value = float(
                row[
                    "mean_relative_score_change"
                ]
            )

            if previous_value is None:

                change_previous = 0.0
                absolute_change_previous = 0.0

            else:

                change_previous = (
                    current_value
                    - previous_value
                )

                absolute_change_previous = abs(
                    change_previous
                )

            change_window_1 = (
                current_value
                - baseline_value
            )

            rows.append(
                {
                    "window": int(
                        row["window"]
                    ),
                    "top_k": int(top_k),
                    "mean_relative_score_change": (
                        current_value
                    ),
                    "window_to_window_change": (
                        change_previous
                    ),
                    "absolute_window_to_window_change": (
                        absolute_change_previous
                    ),
                    "change_from_window_1": (
                        change_window_1
                    ),
                    "absolute_change_from_window_1": (
                        abs(change_window_1)
                    ),
                }
            )

            previous_value = (
                current_value
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
# Class trajectory
# ============================================================


def build_class_trajectory(
    class_summary_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build class-specific fidelity trajectories.
    """

    rows = []

    for (
        fraud_value,
        class_name,
        top_k,
    ), group in class_summary_df.groupby(
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

        baseline_value = float(
            group.iloc[0][
                "mean_relative_score_change"
            ]
        )

        previous_value = None

        for _, row in group.iterrows():

            current_value = float(
                row[
                    "mean_relative_score_change"
                ]
            )

            if previous_value is None:

                change_previous = 0.0

            else:

                change_previous = (
                    current_value
                    - previous_value
                )

            rows.append(
                {
                    "window": int(
                        row["window"]
                    ),
                    "isFraud": int(
                        fraud_value
                    ),
                    "class": class_name,
                    "top_k": int(top_k),
                    "mean_relative_score_change": (
                        current_value
                    ),
                    "window_to_window_change": (
                        change_previous
                    ),
                    "change_from_window_1": (
                        current_value
                        - baseline_value
                    ),
                }
            )

            previous_value = (
                current_value
            )

    return (
        pd.DataFrame(rows)
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
# Main experiment
# ============================================================


def run_experiment(
    transactions_path: str,
    explanation_dir: Path = DEFAULT_INPUT_EXPLANATION_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    train_fraction: float = DEFAULT_TRAIN_FRACTION,
    n_future_windows: int = DEFAULT_N_FUTURE_WINDOWS,
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    random_state: int = DEFAULT_RANDOM_STATE,
    k_values: List[int] | None = None,
    sample_strategy: str = DEFAULT_SAMPLE_STRATEGY,
):
    """
    Evaluate ExIFFI perturbation fidelity across all
    chronological future windows.

    The Isolation Forest and preprocessing are fitted
    once on the training period.

    Existing temporal ExIFFI explanation files are then
    loaded and evaluated against that same static model.
    """

    if k_values is None:
        k_values = list(
            DEFAULT_K_VALUES
        )

    print("=" * 70)
    print(
        "BASELINE 1 — TEMPORAL EXIFFI "
        "EXPLANATION FIDELITY"
    )
    print("=" * 70)

    print(
        f"\nTraining fraction: "
        f"{train_fraction}"
    )

    print(
        f"Future windows: "
        f"{n_future_windows}"
    )

    print(
        f"Isolation Forest trees: "
        f"{n_estimators}"
    )

    print(
        f"Random state: "
        f"{random_state}"
    )

    print(
        f"Top-k values: "
        f"{k_values}"
    )

    print(
        f"Sample strategy: "
        f"{sample_strategy}"
    )

    # ---------------------------------------------------------
    # 1. Load temporal explanation matrices
    # ---------------------------------------------------------

    print(
        "\n[1/8] Loading temporal explanations..."
    )

    explanation_windows: Dict[
        int,
        pd.DataFrame,
    ] = {}

    importance_matrices: Dict[
        int,
        np.ndarray,
    ] = {}

    explanation_feature_names: Dict[
        int,
        List[str],
    ] = {}

    for window_number in range(
        1,
        n_future_windows + 1,
    ):

        explanation_df = (
            load_explanation_file(
                window=window_number,
                explanation_dir=explanation_dir,
            )
        )

        (
            importance_matrix,
            feature_names,
        ) = extract_importance_matrix(
            explanation_df
        )

        explanation_windows[
            window_number
        ] = explanation_df

        importance_matrices[
            window_number
        ] = importance_matrix

        explanation_feature_names[
            window_number
        ] = feature_names

    # ---------------------------------------------------------
    # 2. Load data
    # ---------------------------------------------------------

    print(
        "\n[2/8] Loading transaction dataset..."
    )

    transactions = pd.read_csv(
        transactions_path,
        low_memory=False,
    )

    print(
        f"Loaded {len(transactions):,} "
        "transactions."
    )

    # ---------------------------------------------------------
    # 3. Chronological split
    # ---------------------------------------------------------

    print(
        "\n[3/8] Creating chronological windows..."
    )

    train, future_windows = chronological_windows(
        transactions,
        time_col="TransactionDT",
        train_fraction=train_fraction,
        n_future_windows=n_future_windows,
    )

    print(
        f"Training observations: "
        f"{len(train):,}"
    )

    for i, window in enumerate(
        future_windows,
        start=1,
    ):

        print(
            f"Window {i}: "
            f"{len(window):,} observations"
        )

    # ---------------------------------------------------------
    # 4. Fit training-only preprocessing
    # ---------------------------------------------------------

    print(
        "\n[4/8] Fitting training-only preprocessing..."
    )

    preprocessor = NumericTransactionPreprocessor(
        policy=FeaturePolicy(
            target="isFraud",
            temporal="TransactionDT",
            identifier="TransactionID",
            max_missing_rate=0.95,
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

    # ---------------------------------------------------------
    # Validate all explanation feature schemas.
    # ---------------------------------------------------------

    for window_number in range(
        1,
        n_future_windows + 1,
    ):

        validate_explanation_matrix(
            explanation_df=(
                explanation_windows[
                    window_number
                ]
            ),
            importance_matrix=(
                importance_matrices[
                    window_number
                ]
            ),
            feature_names=(
                explanation_feature_names[
                    window_number
                ]
            ),
            expected_feature_names=feature_names,
            window=window_number,
        )

    # ---------------------------------------------------------
    # 5. Train static Isolation Forest
    # ---------------------------------------------------------

    print(
        "\n[5/8] Training static Isolation Forest..."
    )

    detector = IsolationForestDetector(
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

    # ---------------------------------------------------------
    # 6. Training-only median baseline
    # ---------------------------------------------------------

    print(
        "\n[6/8] Computing training-only "
        "median intervention baseline..."
    )

    baseline = np.median(
        x_train.to_numpy(
            dtype=float
        ),
        axis=0,
    )

    if baseline.shape != (
        len(feature_names),
    ):
        raise ValueError(
            "Training baseline does not contain "
            "one value per selected feature."
        )

    print(
        f"Baseline features: "
        f"{len(baseline)}"
    )

    # ---------------------------------------------------------
    # 7. Temporal fidelity evaluation
    # ---------------------------------------------------------

    print(
        "\n[7/8] Evaluating temporal fidelity..."
    )

    all_results = []

    for window_number, window in enumerate(
        future_windows,
        start=1,
    ):

        print()
        print(
            "-" * 70
        )

        print(
            f"Window {window_number}"
        )

        explanation_df = (
            explanation_windows[
                window_number
            ]
        )

        importance_matrix = (
            importance_matrices[
                window_number
            ]
        )

        # -----------------------------------------------------
        # Transform using SAME training-fitted preprocessor.
        # -----------------------------------------------------

        X_window = (
            preprocessor
            .transform(window)
        )

        # -----------------------------------------------------
        # Reconstruct the exact sample represented by the
        # explanation CSV.
        # -----------------------------------------------------

        (
            X_sample,
            y_sample,
            transaction_ids,
        ) = reconstruct_window_sample(
            window=window,
            transformed_window=X_window,
            explanation_df=explanation_df,
            window_number=window_number,
            sample_strategy=sample_strategy,
            random_state=random_state,
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

        # -----------------------------------------------------
        # Ensure the model scores match the scores that were
        # present when ExIFFI explanations were generated.
        # -----------------------------------------------------

        validate_original_scores(
            model=detector.model,
            X_sample=X_sample,
            explanation_df=explanation_df,
        )

        print(
            "  Original model scores: validated"
        )

        # -----------------------------------------------------
        # Evaluate fidelity.
        # -----------------------------------------------------

        window_results = (
            evaluate_window_fidelity(
                window=window_number,
                model=detector.model,
                X_sample=X_sample,
                y_sample=y_sample,
                transaction_ids=transaction_ids,
                explanation_df=explanation_df,
                importance_matrix=importance_matrix,
                baseline=baseline,
                feature_names=feature_names,
                k_values=k_values,
            )
        )

        all_results.append(
            window_results
        )

        # -----------------------------------------------------
        # Print quick window summary.
        # -----------------------------------------------------

        window_summary = (
            window_results
            .groupby("top_k")
            .agg(
                mean_top_k_mass=(
                    "top_k_mass",
                    "mean",
                ),
                mean_relative_score_change=(
                    "relative_score_change",
                    "mean",
                ),
            )
            .reset_index()
        )

        print(
            "\n  Fidelity:"
        )

        print(
            window_summary.to_string(
                index=False
            )
        )

    fidelity_df = pd.concat(
        all_results,
        ignore_index=True,
    )

    # ---------------------------------------------------------
    # 8. Summaries and saving
    # ---------------------------------------------------------

    print(
        "\n[8/8] Building temporal summaries "
        "and saving results..."
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Observation-level output.
    # ---------------------------------------------------------

    fidelity_path = (
        output_dir
        / "static_if_exiffi_temporal_fidelity.csv"
    )

    fidelity_df.to_csv(
        fidelity_path,
        index=False,
    )

    # ---------------------------------------------------------
    # Overall summary.
    # ---------------------------------------------------------

    summary_df = build_summary(
        fidelity_df
    )

    summary_path = (
        output_dir
        / "static_if_exiffi_temporal_fidelity_summary.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    # ---------------------------------------------------------
    # Class summary.
    # ---------------------------------------------------------

    class_summary_df = (
        build_class_summary(
            fidelity_df
        )
    )

    class_summary_path = (
        output_dir
        / "static_if_exiffi_temporal_fidelity_class_summary.csv"
    )

    class_summary_df.to_csv(
        class_summary_path,
        index=False,
    )

    # ---------------------------------------------------------
    # Overall fidelity trajectory.
    # ---------------------------------------------------------

    trajectory_df = (
        build_fidelity_trajectory(
            summary_df
        )
    )

    trajectory_path = (
        output_dir
        / "static_if_exiffi_temporal_fidelity_trajectory.csv"
    )

    trajectory_df.to_csv(
        trajectory_path,
        index=False,
    )

    # ---------------------------------------------------------
    # Class fidelity trajectory.
    # ---------------------------------------------------------

    class_trajectory_df = (
        build_class_trajectory(
            class_summary_df
        )
    )

    class_trajectory_path = (
        output_dir
        / "static_if_exiffi_temporal_fidelity_class_trajectory.csv"
    )

    class_trajectory_df.to_csv(
        class_trajectory_path,
        index=False,
    )

    # ---------------------------------------------------------
    # Console output.
    # ---------------------------------------------------------

    print(
        "\nObservation-level fidelity saved to:"
    )

    print(
        fidelity_path
    )

    print(
        "\nTemporal fidelity summary saved to:"
    )

    print(
        summary_path
    )

    print(
        "\nClass-stratified fidelity summary saved to:"
    )

    print(
        class_summary_path
    )

    print(
        "\nFidelity trajectory saved to:"
    )

    print(
        trajectory_path
    )

    print(
        "\nClass fidelity trajectory saved to:"
    )

    print(
        class_trajectory_path
    )

    # ---------------------------------------------------------
    # Display summary.
    # ---------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "TEMPORAL EXIFFI FIDELITY SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        summary_df.to_string(
            index=False
        )
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "CLASS-STRATIFIED TEMPORAL FIDELITY"
    )

    print(
        "=" * 70
    )

    print(
        class_summary_df.to_string(
            index=False
        )
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "TEMPORAL FIDELITY TRAJECTORY"
    )

    print(
        "=" * 70
    )

    print(
        trajectory_df.to_string(
            index=False
        )
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "TEMPORAL EXIFFI FIDELITY "
        "EVALUATION COMPLETE"
    )

    print(
        "=" * 70
    )

    return (
        fidelity_df,
        summary_df,
        class_summary_df,
        trajectory_df,
        class_trajectory_df,
    )


# ============================================================
# CLI
# ============================================================


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate temporal ExIFFI "
            "perturbation fidelity."
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
            DEFAULT_INPUT_EXPLANATION_DIR
        ),
        help=(
            "Directory containing the "
            "temporal ExIFFI explanation CSVs."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=str(
            DEFAULT_OUTPUT_DIR
        ),
        help=(
            "Directory where fidelity "
            "results will be written."
        ),
    )

    parser.add_argument(
        "--train-fraction",
        type=float,
        default=DEFAULT_TRAIN_FRACTION,
    )

    parser.add_argument(
        "--future-windows",
        type=int,
        default=DEFAULT_N_FUTURE_WINDOWS,
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
        "--k-values",
        type=int,
        nargs="+",
        default=DEFAULT_K_VALUES,
        help=(
            "Top-k values to evaluate. "
            "Default: 1 3 5 10 20"
        ),
    )

    parser.add_argument(
        "--sample-strategy",
        choices=["sequential", "random", "stratified"],
        default=DEFAULT_SAMPLE_STRATEGY,
        help=(
            "Sampling strategy used when the temporal ExIFFI "
            "explanations were generated. The current explanation "
            "files were generated with 'stratified'."
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
        n_future_windows=args.future_windows,
        n_estimators=args.estimators,
        random_state=args.random_state,
        k_values=args.k_values,
        sample_strategy=args.sample_strategy,
    )


if __name__ == "__main__":
    main()