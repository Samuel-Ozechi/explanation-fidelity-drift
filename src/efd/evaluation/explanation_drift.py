from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr


def _validate_matrix(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        raise ValueError(
            "Importance matrix must be 2-dimensional."
        )

    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError(
            "Importance matrix must not be empty."
        )

    if not np.isfinite(X).all():
        raise ValueError(
            "Importance matrix contains non-finite values."
        )

    return X


def _mean_distribution(
    X: np.ndarray,
) -> np.ndarray:

    X = _validate_matrix(X)

    # ExIFFI importance values are treated as
    # non-negative contribution magnitudes for
    # distributional comparison.
    X = np.clip(X, 0.0, None)

    distribution = X.mean(axis=0)

    total = distribution.sum()

    if total <= 0:
        return np.full(
            X.shape[1],
            1.0 / X.shape[1],
        )

    return distribution / total


def cosine_distance(
    X_a: np.ndarray,
    X_b: np.ndarray,
) -> float:

    a = _mean_distribution(X_a)
    b = _mean_distribution(X_b)

    if a.shape != b.shape:
        raise ValueError(
            "Matrices must have the same number of features."
        )

    denominator = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )

    if denominator == 0:
        return 0.0

    similarity = (
        np.dot(a, b)
        / denominator
    )

    if np.isclose(
        similarity,
        1.0,
    ):
        return 0.0

    return float(
        1.0 - similarity
    )


def l1_distance(
    X_a: np.ndarray,
    X_b: np.ndarray,
) -> float:

    a = _mean_distribution(X_a)
    b = _mean_distribution(X_b)

    if a.shape != b.shape:
        raise ValueError(
            "Matrices must have the same number of features."
        )

    return float(
        np.abs(a - b).sum()
    )


def l2_distance(
    X_a: np.ndarray,
    X_b: np.ndarray,
) -> float:

    a = _mean_distribution(X_a)
    b = _mean_distribution(X_b)

    if a.shape != b.shape:
        raise ValueError(
            "Matrices must have the same number of features."
        )

    return float(
        np.linalg.norm(a - b)
    )


def js_divergence(
    X_a: np.ndarray,
    X_b: np.ndarray,
) -> float:

    a = _mean_distribution(X_a)
    b = _mean_distribution(X_b)

    if a.shape != b.shape:
        raise ValueError(
            "Matrices must have the same number of features."
        )

    return float(
        jensenshannon(
            a,
            b,
            base=2.0,
        )
        ** 2
    )


def rank_correlation(
    X_a: np.ndarray,
    X_b: np.ndarray,
) -> float:

    a = _mean_distribution(X_a)
    b = _mean_distribution(X_b)

    if a.shape != b.shape:
        raise ValueError(
            "Matrices must have the same number of features."
        )

    result = spearmanr(
        a,
        b,
    )

    statistic = result.statistic

    if not np.isfinite(statistic):
        return 0.0

    return float(statistic)


def top_k_features(
    X: np.ndarray,
    k: int,
) -> set[int]:

    X = _validate_matrix(X)

    if k <= 0:
        raise ValueError(
            "k must be positive."
        )

    k = min(
        k,
        X.shape[1],
    )

    mean_importance = X.mean(
        axis=0
    )

    indices = np.argsort(
        -mean_importance
    )[:k]

    return set(
        indices.tolist()
    )


def top_k_jaccard(
    X_a: np.ndarray,
    X_b: np.ndarray,
    k: int,
) -> float:

    features_a = top_k_features(
        X_a,
        k,
    )

    features_b = top_k_features(
        X_b,
        k,
    )

    union = (
        features_a
        | features_b
    )

    if not union:
        return 1.0

    intersection = (
        features_a
        & features_b
    )

    return float(
        len(intersection)
        / len(union)
    )


def feature_drift_table(
    X_a: np.ndarray,
    X_b: np.ndarray,
    feature_names: list[str],
    label_a: str,
    label_b: str,
) -> pd.DataFrame:

    X_a = _validate_matrix(X_a)
    X_b = _validate_matrix(X_b)

    if X_a.shape[1] != X_b.shape[1]:
        raise ValueError(
            "Matrices must have the same number of features."
        )

    if len(feature_names) != X_a.shape[1]:
        raise ValueError(
            "feature_names length does not match matrix width."
        )

    distribution_a = _mean_distribution(
        X_a
    )

    distribution_b = _mean_distribution(
        X_b
    )

    rank_a = (
        pd.Series(distribution_a)
        .rank(
            method="average",
            ascending=False,
        )
        .to_numpy()
    )

    rank_b = (
        pd.Series(distribution_b)
        .rank(
            method="average",
            ascending=False,
        )
        .to_numpy()
    )

    result = pd.DataFrame(
        {
            "feature": feature_names,

            f"mean_importance_{label_a}":
                distribution_a,

            f"mean_importance_{label_b}":
                distribution_b,

            "importance_change":
                distribution_b
                - distribution_a,

            f"rank_{label_a}":
                rank_a.astype(int),

            f"rank_{label_b}":
                rank_b.astype(int),

            "rank_change":
                rank_b - rank_a,

            "absolute_importance_change":
                np.abs(
                    distribution_b
                    - distribution_a
                ),
        }
    )

    return (
        result
        .sort_values(
            "absolute_importance_change",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def explanation_drift_summary(
    X_a: np.ndarray,
    X_b: np.ndarray,
    label_a: str,
    label_b: str,
) -> dict:

    return {
        "comparison":
            f"{label_a}_{label_b}",

        "cosine_distance":
            cosine_distance(
                X_a,
                X_b,
            ),

        "js_divergence":
            js_divergence(
                X_a,
                X_b,
            ),

        "l1_distance":
            l1_distance(
                X_a,
                X_b,
            ),

        "l2_distance":
            l2_distance(
                X_a,
                X_b,
            ),

        "spearman_rank_correlation":
            rank_correlation(
                X_a,
                X_b,
            ),
    }