import numpy as np

from efd.evaluation.explanation_drift import (
    evaluate_pairwise_drift,
    feature_level_drift,
)


def test_identical_explanations_have_zero_js_drift():

    importance = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.6, 0.3, 0.1],
        ]
    )

    result = evaluate_pairwise_drift(
        importance,
        importance.copy(),
        window_a=1,
        window_b=2,
    )

    assert np.isclose(
        result.js_divergence,
        0.0,
    )

    assert np.isclose(
        result.total_variation,
        0.0,
    )

    assert np.isclose(
        result.cosine_similarity,
        1.0,
    )


def test_feature_drift_detects_change():

    a = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.7, 0.2, 0.1],
        ]
    )

    b = np.array(
        [
            [0.1, 0.8, 0.1],
            [0.2, 0.7, 0.1],
        ]
    )

    result = feature_level_drift(
        a,
        b,
        ["A", "B", "C"],
        window_a=1,
        window_b=2,
    )

    assert result.iloc[0]["feature"] in {
        "A",
        "B",
    }

    assert (
        result.iloc[0]["absolute_change"]
        > 0
    )


def test_pairwise_drift_requires_same_feature_count():

    a = np.ones((5, 3))
    b = np.ones((5, 4))

    try:

        evaluate_pairwise_drift(
            a,
            b,
            window_a=1,
            window_b=2,
        )

    except ValueError as exc:

        assert (
            "same number of features"
            in str(exc)
        )

    else:

        raise AssertionError(
            "Expected ValueError"
        )