from __future__ import annotations

import argparse
from pathlib import Path

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

from efd.explanations.exiffi import (
    ExIFFI,
)


def stratified_sample_indices(
    y: pd.Series,
    sample_size: int,
    random_state: int,
) -> np.ndarray:
    """
    Generate stratified sample indices.

    The current experiment uses binary fraud labels and attempts
    to sample equally from the two classes.

    If one class contains fewer observations than the requested
    allocation, the remaining slots are filled from the other
    class.

    Parameters
    ----------
    y:
        Fraud labels aligned to the window observations.

    sample_size:
        Number of observations to sample.

    random_state:
        Random seed.

    Returns
    -------
    np.ndarray
        Positional indices into the original window.
    """

    if sample_size <= 0:
        raise ValueError(
            "sample_size must be greater than zero."
        )

    if sample_size > len(y):
        raise ValueError(
            "sample_size cannot exceed the number "
            "of observations in the window."
        )

    rng = np.random.default_rng(
        random_state
    )

    y_values = y.to_numpy()

    classes = np.unique(y_values)

    if len(classes) < 2:
        raise ValueError(
            "Stratified sampling requires at least "
            "two classes in the window."
        )

    # ---------------------------------------------------------
    # Binary stratified sampling.
    #
    # We intentionally allocate approximately half the sample
    # to each class.
    # ---------------------------------------------------------

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

        selected = rng.choice(
            class_indices,
            size=target_n,
            replace=False,
        )

        selected_indices.extend(
            selected.tolist()
        )

    # ---------------------------------------------------------
    # If one class did not have enough observations, fill the
    # remaining slots from observations not already selected.
    # ---------------------------------------------------------

    selected_set = set(
        selected_indices
    )

    remaining_n = (
        sample_size
        - len(selected_indices)
    )

    if remaining_n > 0:

        available = np.array(
            [
                i
                for i in range(len(y_values))
                if i not in selected_set
            ],
            dtype=int,
        )

        if remaining_n > len(available):
            raise ValueError(
                "Unable to construct requested "
                "stratified sample."
            )

        additional = rng.choice(
            available,
            size=remaining_n,
            replace=False,
        )

        selected_indices.extend(
            additional.tolist()
        )

    # Shuffle final sample so that the resulting observations
    # are not grouped by class.
    selected_indices = np.asarray(
        selected_indices,
        dtype=int,
    )

    rng.shuffle(
        selected_indices
    )

    return selected_indices


def random_sample_indices(
    n_observations: int,
    sample_size: int,
    random_state: int,
) -> np.ndarray:
    """
    Generate a random sample of positional indices.
    """

    if sample_size <= 0:
        raise ValueError(
            "sample_size must be greater than zero."
        )

    if sample_size > n_observations:
        raise ValueError(
            "sample_size cannot exceed the number "
            "of observations in the window."
        )

    rng = np.random.default_rng(
        random_state
    )

    return rng.choice(
        n_observations,
        size=sample_size,
        replace=False,
    )


def sequential_sample_indices(
    n_observations: int,
    sample_size: int,
) -> np.ndarray:
    """
    Generate a sequential sample from the beginning
    of the window.
    """

    if sample_size <= 0:
        raise ValueError(
            "sample_size must be greater than zero."
        )

    if sample_size > n_observations:
        raise ValueError(
            "sample_size cannot exceed the number "
            "of observations in the window."
        )

    return np.arange(
        sample_size,
        dtype=int,
    )


def select_sample_indices(
    window: pd.DataFrame,
    sample_size: int,
    strategy: str,
    random_state: int,
) -> np.ndarray:
    """
    Select sample indices according to the requested strategy.
    """

    if strategy == "sequential":

        return sequential_sample_indices(
            n_observations=len(window),
            sample_size=sample_size,
        )

    if strategy == "random":

        return random_sample_indices(
            n_observations=len(window),
            sample_size=sample_size,
            random_state=random_state,
        )

    if strategy == "stratified":

        if "isFraud" not in window.columns:
            raise ValueError(
                "Stratified sampling requires "
                "'isFraud' in the window."
            )

        return stratified_sample_indices(
            y=window["isFraud"],
            sample_size=sample_size,
            random_state=random_state,
        )

    raise ValueError(
        f"Unknown sampling strategy: {strategy}"
    )


def build_explanation_dataframe(
    explanations,
    X_sample: np.ndarray,
    y_sample: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Convert ExIFFI explanations into the wide-format CSV
    used by the downstream explanation evaluation pipeline.

    The resulting dataframe contains:

        sample_index
        isFraud
        anomaly_score
        top_1_feature
        top_1_importance
        ...
        importance__<feature>

    The full normalized ExIFFI importance vector is retained
    for every observation.
    """

    rows = []

    for explanation, fraud_label in zip(
        explanations,
        y_sample,
    ):

        normalized = explanation.normalized

        row = {
            "sample_index": explanation.sample_index,
            "isFraud": int(fraud_label),
            "anomaly_score": float(
                -explanation.anomaly_score
            ),
        }

        # -----------------------------------------------------
        # Full importance vector.
        # -----------------------------------------------------

        for feature_name, importance in zip(
            feature_names,
            normalized,
        ):

            row[
                f"importance__{feature_name}"
            ] = float(importance)

        # -----------------------------------------------------
        # Top feature information.
        # -----------------------------------------------------

        top_features = explanation.top_features(
            feature_names=feature_names,
            k=10,
        )

        for rank in range(1, 11):

            if rank <= len(top_features):

                feature, importance = (
                    top_features[rank - 1]
                )

                row[
                    f"top_{rank}_feature"
                ] = feature

                row[
                    f"top_{rank}_importance"
                ] = float(importance)

            else:

                row[
                    f"top_{rank}_feature"
                ] = None

                row[
                    f"top_{rank}_importance"
                ] = 0.0

        rows.append(row)

    return pd.DataFrame(rows)


def run_experiment(
    transactions_path: str,
    train_fraction: float = 0.60,
    n_future_windows: int = 4,
    n_estimators: int = 300,
    random_state: int = 42,
    reference_date: str = "2017-12-01",
    sample_size: int = 100,
    sample_strategy: str = "sequential",
):
    """
    Generate ExIFFI explanations across chronological
    future windows using a static Isolation Forest.

    The model and preprocessing are fitted ONLY on the
    training period.

    Explanations are then generated independently for
    each future window.

    Parameters
    ----------
    transactions_path:
        Path to IEEE-CIS train_transaction.csv.

    train_fraction:
        Fraction of observations used for training.

    n_future_windows:
        Number of future chronological windows.

    n_estimators:
        Number of Isolation Forest trees.

    random_state:
        Global random seed.

    reference_date:
        Reference date used by temporal utilities if required.

    sample_size:
        Number of observations sampled from each future window.

    sample_strategy:
        One of:

            sequential
            random
            stratified
    """

    print("=" * 70)
    print("BASELINE 1 — TEMPORAL EXIFFI EXPLANATIONS")
    print("=" * 70)

    print()
    print(
        f"Sampling strategy: {sample_strategy}"
    )

    print(
        f"Sample size per window: {sample_size}"
    )

    # ---------------------------------------------------------
    # 1. Load data
    # ---------------------------------------------------------

    print(
        "\n[1/7] Loading transaction dataset..."
    )

    df = pd.read_csv(
        transactions_path,
        low_memory=False,
    )

    print(
        f"Loaded {len(df):,} transactions "
        f"and {len(df.columns)} columns."
    )

    # ---------------------------------------------------------
    # 2. Chronological split
    # ---------------------------------------------------------

    print(
        "\n[2/7] Creating chronological windows..."
    )

    train, future_windows = chronological_windows(
        df,
        time_col="TransactionDT",
        train_fraction=train_fraction,
        n_future_windows=n_future_windows,
    )

    print(
        f"Training observations: "
        f"{len(train):,}"
    )

    print(
        f"Future windows: "
        f"{len(future_windows)}"
    )

    # ---------------------------------------------------------
    # 3. Preprocessing
    # ---------------------------------------------------------

    print(
        "\n[3/7] Fitting preprocessing "
        "on training period only..."
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
    # 4. Train static Isolation Forest
    # ---------------------------------------------------------

    print(
        "\n[4/7] Training static Isolation Forest..."
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
    # 5. Initialize ExIFFI
    # ---------------------------------------------------------

    print(
        "\n[5/7] Initializing ExIFFI..."
    )

    explain = ExIFFI(
        model=detector.model,
        feature_names=feature_names,
    )

    # ---------------------------------------------------------
    # 6. Generate temporal explanations
    # ---------------------------------------------------------

    print(
        "\n[6/7] Generating temporal explanations..."
    )

    output_dir = Path(
        "results/explanations"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata_rows = []

    for window_number, window in enumerate(
        future_windows,
        start=1,
    ):

        print()
        print(
            f"Window {window_number}"
        )

        print(
            f"  Observations: "
            f"{len(window):,}"
        )

        # -----------------------------------------------------
        # Date information.
        # -----------------------------------------------------

        if "TransactionDT" in window.columns:

            transaction_dates = pd.to_datetime(
                window["TransactionDT"],
                unit="s",
                origin=reference_date,
                errors="coerce",
            )

            date_start = transaction_dates.min()

            date_end = transaction_dates.max()

            print(
                f"  Date range: "
                f"{date_start.date() if pd.notna(date_start) else 'N/A'} "
                f"→ "
                f"{date_end.date() if pd.notna(date_end) else 'N/A'}"
            )

        else:

            date_start = None
            date_end = None

        # -----------------------------------------------------
        # Fraud statistics.
        # -----------------------------------------------------

        if "isFraud" in window.columns:

            fraud_count = int(
                window["isFraud"]
                .sum()
            )

            nonfraud_count = (
                len(window)
                - fraud_count
            )

            print(
                f"  Fraud observations: "
                f"{fraud_count:,}"
            )

            print(
                f"  Non-fraud observations: "
                f"{nonfraud_count:,}"
            )

        else:

            fraud_count = None
            nonfraud_count = None

        # -----------------------------------------------------
        # Transform future window using the SAME preprocessor
        # fitted on training data.
        # -----------------------------------------------------

        X_window = (
            preprocessor
            .transform(window)
        )

        # -----------------------------------------------------
        # Select sample indices.
        #
        # IMPORTANT:
        # These indices refer to the positional rows of the
        # original future window and therefore remain aligned
        # with the fraud labels.
        # -----------------------------------------------------

        sample_indices = select_sample_indices(
            window=window,
            sample_size=min(
                sample_size,
                len(window),
            ),
            strategy=sample_strategy,
            random_state=(
                random_state
                + window_number
            ),
        )

        # -----------------------------------------------------
        # Extract sampled feature matrix.
        #
        # This is the critical fix for the stratified sampling
        # bug. If X_window is a DataFrame, convert it to a
        # NumPy array before using positional indexing.
        # -----------------------------------------------------

        if isinstance(
            X_window,
            pd.DataFrame,
        ):

            X_sample = (
                X_window
                .iloc[sample_indices]
                .to_numpy(
                    dtype=float
                )
            )

        else:

            X_window_array = np.asarray(
                X_window,
                dtype=float,
            )

            X_sample = (
                X_window_array[
                    sample_indices
                ]
            )

        # -----------------------------------------------------
        # Extract labels using THE SAME positional indices.
        # -----------------------------------------------------

        y_sample = (
            window["isFraud"]
            .iloc[sample_indices]
            .to_numpy(
                dtype=int
            )
        )

        n = len(X_sample)

        print(
            f"  Sampled observations: "
            f"{n}"
        )

        if sample_strategy == "stratified":

            sampled_fraud = int(
                y_sample.sum()
            )

            sampled_nonfraud = (
                n
                - sampled_fraud
            )

            print(
                f"  Sampled fraud: "
                f"{sampled_fraud}"
            )

            print(
                f"  Sampled non-fraud: "
                f"{sampled_nonfraud}"
            )

        # -----------------------------------------------------
        # Sanity checks.
        # -----------------------------------------------------

        if X_sample.ndim != 2:

            raise ValueError(
                "X_sample must be a "
                "two-dimensional matrix."
            )

        if X_sample.shape[1] != len(
            feature_names
        ):

            raise ValueError(
                "Sample feature count does not "
                "match ExIFFI feature count."
            )

        if len(X_sample) != len(
            y_sample
        ):

            raise ValueError(
                "Sampled feature matrix and "
                "labels are not aligned."
            )

        # -----------------------------------------------------
        # Generate explanations.
        # -----------------------------------------------------

        explanations = []

        for i in range(n):

            explanation = explain.explain(
                X_sample[i],
                sample_index=i,
            )

            explanations.append(
                explanation
            )

        print(
            f"  Generated explanations: "
            f"{len(explanations)}"
        )

        # -----------------------------------------------------
        # Build output dataframe.
        # -----------------------------------------------------

        explanation_df = (
            build_explanation_dataframe(
                explanations=explanations,
                X_sample=X_sample,
                y_sample=y_sample,
                feature_names=feature_names,
            )
        )

        output_path = (
            output_dir
            / (
                "static_if_exiffi_"
                f"w{window_number}_sample.csv"
            )
        )

        explanation_df.to_csv(
            output_path,
            index=False,
        )

        print(
            f"  Saved: {output_path}"
        )

        # -----------------------------------------------------
        # Metadata.
        # -----------------------------------------------------

        metadata_rows.append(
            {
                "window": window_number,
                "n_observations": len(window),
                "n_sampled": n,
                "n_fraud": fraud_count,
                "n_nonfraud": nonfraud_count,
                "sampled_fraud": int(
                    y_sample.sum()
                ),
                "sampled_nonfraud": int(
                    len(y_sample)
                    - y_sample.sum()
                ),
                "sample_strategy": sample_strategy,
                "date_start": (
                    date_start
                    if date_start is not None
                    else ""
                ),
                "date_end": (
                    date_end
                    if date_end is not None
                    else ""
                ),
                "n_features": len(
                    feature_names
                ),
                "n_estimators": n_estimators,
                "random_state": random_state,
            }
        )

    # ---------------------------------------------------------
    # 7. Save experiment metadata
    # ---------------------------------------------------------

    print(
        "\n[7/7] Saving temporal experiment metadata..."
    )

    metadata_df = pd.DataFrame(
        metadata_rows
    )

    metadata_path = (
        output_dir
        / "static_if_exiffi_temporal_metadata.csv"
    )

    metadata_df.to_csv(
        metadata_path,
        index=False,
    )

    print(
        f"\nMetadata saved to:"
        f"\n{metadata_path}"
    )

    # Also preserve a human-readable metadata file.
    txt_path = (
        output_dir
        / "static_if_exiffi_temporal_metadata.txt"
    )

    with txt_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "BASELINE 1 — TEMPORAL EXIFFI "
            "EXPLANATIONS\n"
        )

        f.write(
            "=" * 70
            + "\n\n"
        )

        f.write(
            f"Transactions: "
            f"{transactions_path}\n"
        )

        f.write(
            f"Training fraction: "
            f"{train_fraction}\n"
        )

        f.write(
            f"Future windows: "
            f"{n_future_windows}\n"
        )

        f.write(
            f"Sample size per window: "
            f"{sample_size}\n"
        )

        f.write(
            f"Sampling strategy: "
            f"{sample_strategy}\n"
        )

        f.write(
            f"Random state: "
            f"{random_state}\n"
        )

        f.write(
            f"Isolation Forest trees: "
            f"{n_estimators}\n"
        )

        f.write(
            f"Features: "
            f"{len(feature_names)}\n\n"
        )

        f.write(
            metadata_df.to_string(
                index=False
            )
        )

        f.write("\n")

    print(
        f"Human-readable metadata saved to:"
        f"\n{txt_path}"
    )

    print(
        "\nTemporal ExIFFI generation complete."
    )

    return metadata_df


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Generate temporal ExIFFI explanations "
            "for future chronological windows."
        )
    )

    parser.add_argument(
        "--transactions",
        required=True,
        help=(
            "Path to "
            "train_transaction.csv"
        ),
    )

    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.60,
        help=(
            "Fraction of observations used "
            "for training. Default: 0.60"
        ),
    )

    parser.add_argument(
        "--future-windows",
        type=int,
        default=4,
        help=(
            "Number of future windows. "
            "Default: 4"
        ),
    )

    parser.add_argument(
        "--n-estimators",
        type=int,
        default=300,
        help=(
            "Number of Isolation Forest trees. "
            "Default: 300"
        ),
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help=(
            "Number of observations sampled "
            "from each future window. "
            "Default: 100"
        ),
    )

    parser.add_argument(
        "--sample-strategy",
        choices=[
            "sequential",
            "random",
            "stratified",
        ],
        default="sequential",
        help=(
            "Sampling strategy. "
            "Default: sequential"
        ),
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help=(
            "Random seed. "
            "Default: 42"
        ),
    )

    parser.add_argument(
        "--reference-date",
        default="2017-12-01",
        help=(
            "Reference date for TransactionDT. "
            "Default: 2017-12-01"
        ),
    )

    args = parser.parse_args()

    run_experiment(
        transactions_path=args.transactions,
        train_fraction=args.train_fraction,
        n_future_windows=args.future_windows,
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        reference_date=args.reference_date,
        sample_size=args.sample_size,
        sample_strategy=args.sample_strategy,
    )


if __name__ == "__main__":
    main()