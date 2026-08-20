from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    """
    Normalize each row so that its values sum to 1.
    """

    values = np.asarray(values, dtype=float)

    if values.ndim != 2:
        raise ValueError("values must be two-dimensional.")

    totals = values.sum(axis=1, keepdims=True)

    return np.divide(
        values,
        totals,
        out=np.zeros_like(values),
        where=totals > 0,
    )


def top_k_mass(
    importance: np.ndarray,
    k: int,
) -> np.ndarray:
    """
    Return the fraction of total explanation importance
    contained in the top-k features for each observation.
    """

    importance = _normalize_rows(importance)

    if k <= 0:
        raise ValueError("k must be positive.")

    k = min(k, importance.shape[1])

    sorted_values = np.sort(
        importance,
        axis=1,
    )[:, ::-1]

    return sorted_values[:, :k].sum(axis=1)


def explanation_entropy(
    importance: np.ndarray,
) -> np.ndarray:
    """
    Shannon entropy of each explanation.

    Importance values are normalized before computing entropy.
    Zero-valued features contribute zero.
    """

    importance = _normalize_rows(importance)

    safe = np.where(
        importance > 0,
        importance,
        1.0,
    )

    return -np.sum(
        np.where(
            importance > 0,
            importance * np.log(safe),
            0.0,
        ),
        axis=1,
    )


def effective_features(
    importance: np.ndarray,
) -> np.ndarray:
    """
    Effective number of features represented by an explanation.

    Defined as:

        exp(H)

    where H is Shannon entropy.
    """

    return np.exp(
        explanation_entropy(importance)
    )


def nonzero_feature_count(
    importance: np.ndarray,
    tolerance: float = 0.0,
) -> np.ndarray:
    """
    Number of features with importance greater than tolerance.
    """

    importance = np.asarray(
        importance,
        dtype=float,
    )

    return np.sum(
        importance > tolerance,
        axis=1,
    )


def cosine_similarity(
    a: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    """
    Row-wise cosine similarity between two explanation matrices.
    """

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if a.shape != b.shape:
        raise ValueError(
            "a and b must have identical shapes."
        )

    numerator = np.sum(
        a * b,
        axis=1,
    )

    denominator = (
        np.linalg.norm(a, axis=1)
        * np.linalg.norm(b, axis=1)
    )

    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0,
    )


def top_k_feature_set(
    importance: np.ndarray,
    feature_names: Iterable[str],
    k: int,
) -> list[set[str]]:
    """
    Return the top-k feature set for every observation.
    """

    importance = np.asarray(
        importance,
        dtype=float,
    )

    feature_names = list(feature_names)

    if importance.shape[1] != len(feature_names):
        raise ValueError(
            "Number of features does not match "
            "feature_names."
        )

    k = min(k, importance.shape[1])

    result = []

    for row in importance:

        indices = np.argsort(
            -row
        )[:k]

        result.append(
            {
                feature_names[i]
                for i in indices
            }
        )

    return result


def top_k_overlap(
    importance_a: np.ndarray,
    importance_b: np.ndarray,
    feature_names: Iterable[str],
    k: int = 10,
) -> np.ndarray:
    """
    Jaccard overlap between the top-k features of
    two explanations.

        |A ∩ B|
        -------
        |A ∪ B|
    """

    sets_a = top_k_feature_set(
        importance_a,
        feature_names,
        k,
    )

    sets_b = top_k_feature_set(
        importance_b,
        feature_names,
        k,
    )

    overlaps = []

    for a, b in zip(sets_a, sets_b):

        union = a | b

        if not union:
            overlaps.append(1.0)
            continue

        overlaps.append(
            len(a & b) / len(union)
        )

    return np.asarray(
        overlaps,
        dtype=float,
    )


def evaluate_explanation_matrix(
    importance: np.ndarray,
    feature_names: Iterable[str],
    window: int | None = None,
) -> pd.DataFrame:
    """
    Compute descriptive explanation metrics for a batch
    of local explanations.

    This function intentionally does not claim that these
    metrics measure causal or decision fidelity.

    They characterize explanation concentration,
    sparsity, and structure.
    """

    importance = np.asarray(
        importance,
        dtype=float,
    )

    if importance.ndim != 2:
        raise ValueError(
            "importance must be two-dimensional."
        )

    feature_names = list(feature_names)

    if importance.shape[1] != len(feature_names):
        raise ValueError(
            "Number of features does not match "
            "feature_names."
        )

    normalized = _normalize_rows(
        importance
    )

    result = pd.DataFrame(
        {
            "sample_index": np.arange(
                len(normalized)
            ),
            "top_1_mass": top_k_mass(
                normalized,
                1,
            ),
            "top_5_mass": top_k_mass(
                normalized,
                5,
            ),
            "top_10_mass": top_k_mass(
                normalized,
                10,
            ),
            "entropy": explanation_entropy(
                normalized
            ),
            "effective_features": effective_features(
                normalized
            ),
            "nonzero_features": nonzero_feature_count(
                normalized
            ),
        }
    )

    if window is not None:
        result.insert(
            0,
            "window",
            window,
        )

    return result


def summarize_explanation_metrics(
    metrics: pd.DataFrame,
) -> pd.Series:
    """
    Produce mean/median summary statistics for a batch
    of explanation metrics.
    """

    required = [
        "top_1_mass",
        "top_5_mass",
        "top_10_mass",
        "entropy",
        "effective_features",
        "nonzero_features",
    ]

    missing = [
        column
        for column in required
        if column not in metrics.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    return pd.Series(
        {
            "n_samples": len(metrics),

            "mean_top_1_mass":
                metrics["top_1_mass"].mean(),

            "mean_top_5_mass":
                metrics["top_5_mass"].mean(),

            "mean_top_10_mass":
                metrics["top_10_mass"].mean(),

            "mean_entropy":
                metrics["entropy"].mean(),

            "mean_effective_features":
                metrics["effective_features"].mean(),

            "mean_nonzero_features":
                metrics["nonzero_features"].mean(),

            "median_top_1_mass":
                metrics["top_1_mass"].median(),

            "median_top_5_mass":
                metrics["top_5_mass"].median(),

            "median_top_10_mass":
                metrics["top_10_mass"].median(),

            "median_entropy":
                metrics["entropy"].median(),

            "median_effective_features":
                metrics["effective_features"].median(),

            "median_nonzero_features":
                metrics["nonzero_features"].median(),
        }
    )