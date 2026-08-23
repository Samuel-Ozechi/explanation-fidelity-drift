import numpy as np

from efd.evaluation.explanation_fidelity import (
    perturb_features,
)


def test_perturb_features_changes_only_selected_features():

    x = np.array(
        [1.0, 2.0, 3.0, 4.0]
    )

    baseline = np.array(
        [10.0, 20.0, 30.0, 40.0]
    )

    indices = np.array(
        [1, 3]
    )

    result = perturb_features(
        x=x,
        feature_indices=indices,
        baseline=baseline,
    )

    expected = np.array(
        [1.0, 20.0, 3.0, 40.0]
    )

    assert np.allclose(
        result,
        expected,
    )


def test_perturb_features_does_not_modify_original():

    x = np.array(
        [1.0, 2.0, 3.0]
    )

    baseline = np.array(
        [10.0, 20.0, 30.0]
    )

    perturb_features(
        x=x,
        feature_indices=np.array([0]),
        baseline=baseline,
    )

    assert np.allclose(
        x,
        [1.0, 2.0, 3.0],
    )