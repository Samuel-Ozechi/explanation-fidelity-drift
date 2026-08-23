import numpy as np

from efd.evaluation.explanation_fidelity import (
    evaluate_random_fidelity_matrix,
)


def test_random_fidelity_returns_expected_rows():

    from sklearn.ensemble import IsolationForest

    rng = np.random.default_rng(42)

    X_train = rng.normal(
        size=(100, 5)
    )

    X = rng.normal(
        size=(3, 5)
    )

    model = IsolationForest(
        n_estimators=10,
        random_state=42,
    )

    model.fit(X_train)

    class Explanation:

        def __init__(self, index):
            self.sample_index = index

    explanations = [
        Explanation(i)
        for i in range(len(X))
    ]

    baseline = np.median(
        X_train,
        axis=0,
    )

    results = evaluate_random_fidelity_matrix(
        model=model,
        X=X,
        explanations=explanations,
        baseline=baseline,
        k_values=[1, 2],
        random_state=42,
    )

    assert len(results) == 6

    assert set(
        results["top_k"]
    ) == {1, 2}

    assert (
        results["score_change"]
        .notna()
        .all()
    )