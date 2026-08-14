from __future__ import annotations

import numpy as np
import pandas as pd


def chronological_windows(
    frame: pd.DataFrame,
    time_col: str = "TransactionDT",
    train_fraction: float = 0.60,
    n_future_windows: int = 4,
) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    """
    Chronologically split the dataset.

    Parameters
    ----------
    frame:
        Complete labelled transaction dataframe.

    time_col:
        Temporal ordering variable.

    train_fraction:
        Fraction of earliest observations used for training.

    n_future_windows:
        Number of chronological evaluation windows.

    Returns
    -------
    train:
        Earliest chronological training observations.

    future_windows:
        Sequential future evaluation windows.
    """

    if not 0 < train_fraction < 1:
        raise ValueError(
            "train_fraction must be between 0 and 1."
        )

    if n_future_windows < 1:
        raise ValueError(
            "n_future_windows must be >= 1."
        )

    if time_col not in frame.columns:
        raise KeyError(
            f"Temporal column '{time_col}' not found."
        )

    # Stable sort is important for reproducibility if equal timestamps occur.
    ordered = (
        frame
        .sort_values(
            time_col,
            kind="mergesort"
        )
        .reset_index(drop=True)
    )

    split_idx = int(
        len(ordered) * train_fraction
    )

    train = ordered.iloc[:split_idx].copy()

    future = ordered.iloc[split_idx:].copy()

    # Equal-count windows.
    raw_windows = np.array_split(
        future,
        n_future_windows
    )

    future_windows = [
        window.copy()
        for window in raw_windows
        if len(window) > 0
    ]

    return train, future_windows