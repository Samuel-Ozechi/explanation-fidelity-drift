import numpy as np

from efd.evaluation.explanation_drift import (
    cosine_distance,
    explanation_drift_summary,
    feature_drift_table,
    js_divergence,
    l1_distance,
    l2_distance,
    rank_correlation,
    top_k_jaccard,
)


def test_identical_matrices_have_zero_drift():

    X = np.array(
        [
            [0.1, 0.2, 0.7],
            [0.2, 0.2, 0.6],
        ]
    )

    assert cosine_distance(
        X,
        X,
    ) == 0.0

    assert l1_distance(
        X,
        X,
    ) == 0.0

    assert l2_distance(
        X,
        X,
    ) == 0.0

    assert js_divergence(
        X,
        X,
    ) == 0.0

    assert top_k_jaccard(
        X,
        X,
        2,
    ) == 1.0

    assert rank_correlation(
        X,
        X,
    ) == 1.0


def test_changed_explanations_have_positive_drift():

    X_a = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.7, 0.2, 0.1],
        ]
    )

    X_b = np.array(
        [
            [0.1, 0.8, 0.1],
            [0.1, 0.7, 0.2],
        ]
    )

    assert js_divergence(
        X_a,
        X_b,
    ) > 0

    assert l1_distance(
        X_a,
        X_b,
    ) > 0

    assert l2_distance(
        X_a,
        X_b,
    ) > 0


def test_feature_drift_table():

    X_a = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.7, 0.2, 0.1],
        ]
    )

    X_b = np.array(
        [
            [0.1, 0.8, 0.1],
            [0.1, 0.7, 0.2],
        ]
    )

    result = feature_drift_table(
        X_a,
        X_b,
        ["a", "b", "c"],
        "w1",
        "w2",
    )

    assert len(result) == 3

    assert (
        "importance_change"
        in result.columns
    )

    assert (
        "rank_change"
        in result.columns
    )


def test_summary_contains_expected_metrics():

    X_a = np.array(
        [[0.8, 0.1, 0.1]]
    )

    X_b = np.array(
        [[0.1, 0.8, 0.1]]
    )

    result = explanation_drift_summary(
        X_a,
        X_b,
        "w1",
        "w2",
    )

    assert (
        result["comparison"]
        == "w1_w2"
    )

    assert (
        result["js_divergence"]
        > 0
    )

    assert (
        result["l1_distance"]
        > 0
    )

    assert (
        result["l2_distance"]
        > 0
    )