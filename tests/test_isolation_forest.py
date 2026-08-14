import numpy as np

from efd.models.isolation_forest import (
    IsolationForestDetector,
)


def test_isolation_forest_score_direction():

    rng = np.random.RandomState(42)

    normal = rng.normal(
        loc=0,
        scale=1,
        size=(100, 3),
    )

    anomalies = rng.normal(
        loc=10,
        scale=1,
        size=(5, 3),
    )

    x = np.vstack(
        [
            normal,
            anomalies,
        ]
    )

    detector = (
        IsolationForestDetector(
            n_estimators=50,
            random_state=42,
        )
    )

    detector.fit(x)

    scores = (
        detector
        .anomaly_score(x)
    )

    assert scores.shape == (105,)

    # On average, the deliberately
    # distant observations should have
    # higher anomaly scores.
    assert (
        scores[-5:].mean()
        > scores[:100].mean()
    )