from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr


EPS = 1e-12


@dataclass
class ExplanationDriftResult:
    """
    Pairwise drift between two temporal explanation windows.
    """

    window_a: int
    window_b: int

    js_divergence: float
    total_variation: float
    cosine_distance: float
    cosine_similarity: float
    spearman_correlation: float

    top_1_overlap: float
    top_5_overlap: float
    top_10_overlap: float
    top_20_overlap: float

    top_1_jaccard: float
    top_5_jaccard: float
    top_10_jaccard: float
    top_20_jaccard: float


def _normalize_distribution(
    values: np.ndarray,
) -> np.ndarray:

    values = np.asarray(
        values,
        dtype=float,
    )

    values = np.clip(
        values,
        0.0,
        None,
    )

    total = values.sum()

    if total <= EPS:
        return np.zeros_like(values)

    return values / total


def _mean_explanation_distribution(
    importance: np.ndarray,
) -> np.ndarray:

    importance = np.asarray(
        importance,
        dtype=float,
    )

    if importance.ndim != 2:
        raise ValueError(
            "importance must be a 2D matrix."
        )

    mean_importance = np.mean(
        importance,
        axis=0,
    )

    return _normalize_distribution(
        mean_importance
    )


def _js_divergence(
    p: np.ndarray,
    q: np.ndarray,
) -> float:

    p = _normalize_distribution(p)
    q = _normalize_distribution(q)

    midpoint = 0.5 * (p + q)

    def kl_divergence(
        a: np.ndarray,
        b: np.ndarray,
    ) -> float:

        mask = a > EPS

        return float(
            np.sum(
                a[mask]
                * np.log(
                    a[mask]
                    / np.maximum(
                        b[mask],
                        EPS,
                    )
                )
            )
        )

    return float(
        0.5
        * kl_divergence(p, midpoint)
        + 0.5
        * kl_divergence(q, midpoint)
    )


def _total_variation(
    p: np.ndarray,
    q: np.ndarray,
) -> float:

    p = _normalize_distribution(p)
    q = _normalize_distribution(q)

    return float(
        0.5 * np.sum(np.abs(p - q))
    )


def _cosine_similarity(
    p: np.ndarray,
    q: np.ndarray,
) -> float:

    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    p_norm = np.linalg.norm(p)
    q_norm = np.linalg.norm(q)

    if p_norm <= EPS or q_norm <= EPS:
        return 0.0

    return float(
        np.dot(p, q)
        / (p_norm * q_norm)
    )


def _spearman_correlation(
    p: np.ndarray,
    q: np.ndarray,
) -> float:

    correlation, _ = spearmanr(
        p,
        q,
    )

    if np.isnan(correlation):
        return 0.0

    return float(correlation)


def _top_k_indices(
    distribution: np.ndarray,
    k: int,
) -> set[int]:

    k = min(
        k,
        len(distribution),
    )

    indices = np.argsort(
        -distribution
    )[:k]

    return set(
        int(index)
        for index in indices
    )


def _jaccard(
    a: set[int],
    b: set[int],
) -> float:

    union = a | b

    if not union:
        return 1.0

    return float(
        len(a & b) / len(union)
    )


def _overlap_coefficient(
    a: set[int],
    b: set[int],
) -> float:

    denominator = min(
        len(a),
        len(b),
    )

    if denominator == 0:
        return 0.0

    return float(
        len(a & b) / denominator
    )


def evaluate_pairwise_drift(
    importance_a: np.ndarray,
    importance_b: np.ndarray,
    window_a: int,
    window_b: int,
) -> ExplanationDriftResult:
    """
    Evaluate explanation distribution drift between
    two temporal windows.

    Parameters
    ----------
    importance_a:
        Explanation matrix for window A.
        Shape: (n_observations, n_features)

    importance_b:
        Explanation matrix for window B.
        Shape: (n_observations, n_features)

    window_a:
        Identifier for the first temporal window.

    window_b:
        Identifier for the second temporal window.
    """

    importance_a = np.asarray(
        importance_a,
        dtype=float,
    )

    importance_b = np.asarray(
        importance_b,
        dtype=float,
    )

    if importance_a.ndim != 2:
        raise ValueError(
            "importance_a must be two-dimensional."
        )

    if importance_b.ndim != 2:
        raise ValueError(
            "importance_b must be two-dimensional."
        )

    if (
        importance_a.shape[1]
        != importance_b.shape[1]
    ):
        raise ValueError(
            "Both explanation matrices must "
            "have the same number of features."
        )

    distribution_a = (
        _mean_explanation_distribution(
            importance_a
        )
    )

    distribution_b = (
        _mean_explanation_distribution(
            importance_b
        )
    )

    top_sets_a = {
        k: _top_k_indices(
            distribution_a,
            k,
        )
        for k in (1, 5, 10, 20)
    }

    top_sets_b = {
        k: _top_k_indices(
            distribution_b,
            k,
        )
        for k in (1, 5, 10, 20)
    }

    return ExplanationDriftResult(
        window_a=window_a,
        window_b=window_b,
        js_divergence=_js_divergence(
            distribution_a,
            distribution_b,
        ),
        total_variation=_total_variation(
            distribution_a,
            distribution_b,
        ),
        cosine_distance=float(
            cosine(
                distribution_a,
                distribution_b,
            )
        ),
        cosine_similarity=_cosine_similarity(
            distribution_a,
            distribution_b,
        ),
        spearman_correlation=_spearman_correlation(
            distribution_a,
            distribution_b,
        ),
        top_1_overlap=_overlap_coefficient(
            top_sets_a[1],
            top_sets_b[1],
        ),
        top_5_overlap=_overlap_coefficient(
            top_sets_a[5],
            top_sets_b[5],
        ),
        top_10_overlap=_overlap_coefficient(
            top_sets_a[10],
            top_sets_b[10],
        ),
        top_20_overlap=_overlap_coefficient(
            top_sets_a[20],
            top_sets_b[20],
        ),
        top_1_jaccard=_jaccard(
            top_sets_a[1],
            top_sets_b[1],
        ),
        top_5_jaccard=_jaccard(
            top_sets_a[5],
            top_sets_b[5],
        ),
        top_10_jaccard=_jaccard(
            top_sets_a[10],
            top_sets_b[10],
        ),
        top_20_jaccard=_jaccard(
            top_sets_a[20],
            top_sets_b[20],
        ),
    )


def feature_level_drift(
    importance_a: np.ndarray,
    importance_b: np.ndarray,
    feature_names: Iterable[str],
    window_a: int,
    window_b: int,
) -> pd.DataFrame:
    """
    Calculate feature-level explanation changes between
    two temporal windows.
    """

    importance_a = np.asarray(
        importance_a,
        dtype=float,
    )

    importance_b = np.asarray(
        importance_b,
        dtype=float,
    )

    if importance_a.shape[1] != importance_b.shape[1]:
        raise ValueError(
            "Explanation matrices must have "
            "the same feature dimension."
        )

    feature_names = list(feature_names)

    if len(feature_names) != importance_a.shape[1]:
        raise ValueError(
            "feature_names length does not match "
            "the explanation matrix."
        )

    mean_a = np.mean(
        importance_a,
        axis=0,
    )

    mean_b = np.mean(
        importance_b,
        axis=0,
    )

    absolute_change = np.abs(
        mean_b - mean_a
    )

    denominator = np.maximum(
        np.abs(mean_a),
        EPS,
    )

    relative_change = (
        (mean_b - mean_a)
        / denominator
    )

    result = pd.DataFrame(
        {
            "window_a": window_a,
            "window_b": window_b,
            "feature": feature_names,
            "mean_importance_a": mean_a,
            "mean_importance_b": mean_b,
            "absolute_change": absolute_change,
            "relative_change": relative_change,
        }
    )

    return result.sort_values(
        "absolute_change",
        ascending=False,
    ).reset_index(
        drop=True
    )