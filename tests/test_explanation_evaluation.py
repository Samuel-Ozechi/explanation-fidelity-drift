import numpy as np

from efd.evaluation.explanation import (
    effective_features,
    explanation_entropy,
    evaluate_explanation_matrix,
    nonzero_feature_count,
    top_k_mass,
    top_k_overlap,
)


def test_top_k_mass():

    importance = np.array(
        [
            [0.5, 0.3, 0.2],
            [0.1, 0.2, 0.7],
        ]
    )

    result = top_k_mass(
        importance,
        1,
    )

    assert np.allclose(
        result,
        [0.5, 0.7],
    )


def test_entropy_is_nonnegative():

    importance = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
        ]
    )

    result = explanation_entropy(
        importance
    )

    assert np.all(result >= 0)

    assert np.isclose(
        result[0],
        0.0,
    )


def test_effective_features():

    importance = np.array(
        [
            [1.0, 0.0],
            [0.5, 0.5],
        ]
    )

    result = effective_features(
        importance
    )

    assert np.isclose(
        result[0],
        1.0,
    )

    assert np.isclose(
        result[1],
        2.0,
    )


def test_nonzero_feature_count():

    importance = np.array(
        [
            [0.5, 0.5, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )

    result = nonzero_feature_count(
        importance
    )

    assert np.array_equal(
        result,
        [2, 1],
    )


def test_top_k_overlap():

    feature_names = [
        "A",
        "B",
        "C",
        "D",
    ]

    a = np.array(
        [
            [0.4, 0.3, 0.2, 0.1],
        ]
    )

    b = np.array(
        [
            [0.5, 0.3, 0.1, 0.1],
        ]
    )

    result = top_k_overlap(
        a,
        b,
        feature_names,
        k=2,
    )

    assert np.isclose(
        result[0],
        1.0,
    )


def test_evaluate_explanation_matrix():

    feature_names = [
        "A",
        "B",
        "C",
    ]

    importance = np.array(
        [
            [0.5, 0.3, 0.2],
            [1.0, 0.0, 0.0],
        ]
    )

    result = evaluate_explanation_matrix(
        importance,
        feature_names,
        window=1,
    )

    assert len(result) == 2

    assert "window" in result.columns
    assert "top_1_mass" in result.columns
    assert "top_5_mass" in result.columns
    assert "entropy" in result.columns
    assert "effective_features" in result.columns

    assert np.isclose(
        result.iloc[1]["top_1_mass"],
        1.0,
    )