from __future__ import annotations

import numpy as np


def rank_features(
    contributions: np.ndarray,
) -> np.ndarray:
    """
    Rank features by absolute contribution.
    """

    contributions = np.asarray(
        contributions
    )

    return np.argsort(
        -np.abs(contributions)
    )


def top_k_features(
    contributions: np.ndarray,
    k: int,
) -> np.ndarray:

    ranking = rank_features(
        contributions
    )

    return ranking[:k]


def mask_features(
    x: np.ndarray,
    feature_indices: np.ndarray,
    baseline: float = 0.0,
) -> np.ndarray:
    """
    Remove/neutralize selected features.

    The returned vector is a copy.
    """

    perturbed = np.asarray(
        x,
        dtype=float,
    ).copy()

    perturbed[
        feature_indices
    ] = baseline

    return perturbed