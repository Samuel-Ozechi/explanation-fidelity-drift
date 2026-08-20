from __future__ import annotations

from pathlib import Path

import pandas as pd

from efd.data.preprocessing import (
    FeaturePolicy,
    NumericTransactionPreprocessor,
)
from efd.data.temporal import chronological_windows
from efd.models.isolation_forest import IsolationForestDetector


def load_baseline_1_artifacts(
    transactions_path: str | Path,
    train_fraction: float = 0.60,
    n_future_windows: int = 4,
    n_estimators: int = 300,
    contamination: str | float = "auto",
    random_state: int = 42,
) -> tuple[IsolationForestDetector, pd.DataFrame, list[str]]:
    """
    Load the exact Baseline 1 model and feature matrix.

    This reuses the same chronological split, preprocessing, and
    Isolation Forest fitting path as the Baseline 1 implementation.
    """

    df = pd.read_csv(
        transactions_path,
        low_memory=False,
    )

    train, _ = chronological_windows(
        df,
        time_col="TransactionDT",
        train_fraction=train_fraction,
        n_future_windows=n_future_windows,
    )

    preprocessor = NumericTransactionPreprocessor(
        policy=FeaturePolicy(
            target="isFraud",
            temporal="TransactionDT",
            identifier="TransactionID",
            max_missing_rate=0.95,
        )
    )

    X = preprocessor.fit_transform(train)
    feature_names = preprocessor.feature_names_

    if contamination == "auto":
        contamination_value: str | float = "auto"
    else:
        contamination_value = float(contamination)

    model = IsolationForestDetector(
        n_estimators=n_estimators,
        max_samples="auto",
        contamination=contamination_value,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X)

    return model, X, feature_names
