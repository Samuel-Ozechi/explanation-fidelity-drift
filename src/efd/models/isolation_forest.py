from __future__ import annotations

import numpy as np
import pandas as pd

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
        self.feature_names_: list[str] | None = None

    def _as_model_frame(self, x):

        if self.feature_names_ is None:
            return x

        if isinstance(x, pd.DataFrame):
            return x

        x_array = np.asarray(x)

        if x_array.ndim == 1:
            x_array = x_array.reshape(1, -1)

        return pd.DataFrame(
            x_array,
            columns=self.feature_names_,
        )

    def fit(
        self,
        x,
    ) -> "IsolationForestDetector":

        if hasattr(x, "columns"):
            self.feature_names_ = list(x.columns)

        self.model.fit(x)

        return self

    def anomaly_score(
        self,
        x,
    ) -> np.ndarray:

        x_model = self._as_model_frame(x)

        return -self.model.score_samples(x_model)

    def predict_anomaly(
        self,
        x,
    ) -> np.ndarray:

        x_model = self._as_model_frame(x)

        sklearn_prediction = (
            self.model.predict(x_model)
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
