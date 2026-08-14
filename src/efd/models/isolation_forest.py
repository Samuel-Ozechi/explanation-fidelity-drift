from __future__ import annotations

import numpy as np

from sklearn.ensemble import IsolationForest


class IsolationForestDetector:
    """
    Isolation Forest wrapper.

    Important score convention:

        sklearn.score_samples()
            larger = more normal

        anomaly_score()
            larger = more anomalous

    Therefore:

        anomaly_score(x)
            = -score_samples(x)
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_samples: str | int = "auto",
        contamination: str | float = "auto",
        random_state: int = 42,
        n_jobs: int = -1,
    ):

        self.model = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    def fit(
        self,
        x,
    ) -> "IsolationForestDetector":

        self.model.fit(x)

        return self

    def anomaly_score(
        self,
        x,
    ) -> np.ndarray:

        return -self.model.score_samples(x)

    def predict_anomaly(
        self,
        x,
    ) -> np.ndarray:

        sklearn_prediction = (
            self.model.predict(x)
        )

        # sklearn:
        #
        # +1 = normal
        # -1 = anomaly
        #
        # Research convention:
        #
        # 0 = normal
        # 1 = anomaly

        return (
            sklearn_prediction == -1
        ).astype(int)