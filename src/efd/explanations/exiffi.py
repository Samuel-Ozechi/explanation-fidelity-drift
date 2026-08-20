from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


@dataclass
class ExIFFIExplanation:
    """
    Local ExIFFI explanation for one observation.

    contributions[j] contains the raw ExIFFI importance
    assigned to feature j.

    normalized[j] converts the raw values into a
    probability-like feature distribution for inspection.
    """

    sample_index: int
    anomaly_score: float
    contributions: np.ndarray

    @property
    def normalized(self) -> np.ndarray:
        total = np.sum(self.contributions)

        if total <= 0:
            return np.zeros_like(self.contributions)

        return self.contributions / total

    def top_features(
        self,
        feature_names: list[str],
        k: int = 10,
    ) -> list[tuple[str, float]]:

        importance = self.normalized

        indices = np.argsort(-importance)[:k]

        return [
            (
                feature_names[i],
                float(importance[i]),
            )
            for i in indices
        ]


class ExIFFI:
    """
    ExIFFI implementation specialized to the standard
    axis-aligned Isolation Forest.

    For a standard Isolation Forest node:

        f(x) = x_j - tau

    and therefore the split-direction vector is:

        v = e_j

    Hence the node-level feature attribution is assigned
    entirely to feature j.

    The path weighting follows the ExIFFI formulation:

        lambda_k(x) = |X_k| / |X_{k+1}|

    for standard Isolation Forest because ||v|| = 1.
    """

    def __init__(
        self,
        model: IsolationForest,
        feature_names: list[str],
    ) -> None:

        if not hasattr(model, "estimators_"):
            raise ValueError(
                "IsolationForest must be fitted before "
                "creating ExIFFI."
            )

        if len(feature_names) != model.n_features_in_:
            raise ValueError(
                "feature_names length does not match "
                "model.n_features_in_."
            )

        self.model = model
        self.feature_names = feature_names
        self.n_features = model.n_features_in_

    def _as_model_frame(
        self,
        x: np.ndarray,
    ) -> pd.DataFrame | np.ndarray:

        if isinstance(x, pd.DataFrame):
            return x

        x_array = np.asarray(x, dtype=float)

        if x_array.ndim == 1:
            x_array = x_array.reshape(1, -1)

        if hasattr(self.model, "feature_names_in_"):
            return pd.DataFrame(
                x_array,
                columns=list(self.model.feature_names_in_),
            )

        return x_array

    def _get_path_nodes(
        self,
        estimator,
        x: np.ndarray,
    ) -> np.ndarray:

        indicator = estimator.decision_path(x)

        start = indicator.indptr[0]
        end = indicator.indptr[1]

        return indicator.indices[start:end]

    def _node_information(
        self,
        tree,
        node_id: int,
        x: np.ndarray,
    ) -> tuple[int | None, float]:

        feature = tree.feature[node_id]

        # Leaf node.
        if feature < 0:
            return None, 0.0

        left = tree.children_left[node_id]
        right = tree.children_right[node_id]

        n_parent = tree.n_node_samples[node_id]

        threshold = tree.threshold[node_id]

        if x[feature] <= threshold:

            child = left

        else:

            child = right

        n_child = tree.n_node_samples[child]

        if n_child <= 0:
            return feature, 0.0

        # ExIFFI node/path weight.
        #
        # Standard IF uses an axis-aligned split,
        # therefore ||v|| = 1.
        weight = (
            n_parent / n_child
        )

        return feature, float(weight)

    def explain(
        self,
        x: np.ndarray,
        sample_index: int = 0,
    ) -> ExIFFIExplanation:

        x = np.asarray(
            x,
            dtype=float,
        )

        if x.ndim != 1:
            raise ValueError(
                "x must be one-dimensional."
            )

        if len(x) != self.n_features:
            raise ValueError(
                "Observation dimensionality does not "
                "match the fitted model."
            )

        x_2d = x.reshape(1, -1)
        x_model = self._as_model_frame(x_2d)

        raw_importance = np.zeros(
            self.n_features,
            dtype=float,
        )

        for estimator in self.model.estimators_:

            tree = estimator.tree_

            path_nodes = self._get_path_nodes(
                estimator,
                x_2d,
            )

            for node_id in path_nodes:

                feature, weight = (
                    self._node_information(
                        tree,
                        int(node_id),
                        x,
                    )
                )

                if feature is None:
                    continue

                raw_importance[feature] += weight

        # Preserve sklearn's score_samples convention.
        #
        # More negative values correspond to more
        # anomalous observations.
        anomaly_score = float(
            self.model.score_samples(x_model)[0]
        )

        return ExIFFIExplanation(
            sample_index=sample_index,
            anomaly_score=anomaly_score,
            contributions=raw_importance,
        )

    def explain_batch(
        self,
        X: np.ndarray,
    ) -> list[ExIFFIExplanation]:

        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError(
                "X must be two-dimensional."
            )

        return [
            self.explain(
                X[i],
                sample_index=i,
            )
            for i in range(len(X))
        ]

    def path_features(
        self,
        x: np.ndarray,
    ) -> dict:
        """
        Return the actual Isolation Forest decision-path
        features for one observation.

        Used for implementation validation and research
        diagnostics.
        """

        x = np.asarray(x, dtype=float)

        if x.ndim != 1:
            raise ValueError("x must be one-dimensional.")

        x_2d = x.reshape(1, -1)

        result = {}

        for tree_index, estimator in enumerate(
            self.model.estimators_
        ):

            tree = estimator.tree_

            path_nodes = self._get_path_nodes(
                estimator,
                x_2d,
            )

            features = []

            for node_id in path_nodes:

                feature = tree.feature[
                    int(node_id)
                ]

                # Ignore leaf.
                if feature < 0:
                    continue

                features.append(
                    int(feature)
                )

            result[tree_index] = features

        return result
