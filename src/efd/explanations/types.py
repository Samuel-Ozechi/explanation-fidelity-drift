from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LocalExplanation:
    sample_index: int
    anomaly_score: float
    feature_names: list[str]
    contributions: np.ndarray

    @property
    def ranking(self) -> np.ndarray:
        return np.argsort(-np.abs(self.contributions))

    def top_k(self, k: int = 10) -> list[tuple[str, float]]:
        indices = self.ranking[:k]

        return [
            (
                self.feature_names[i],
                float(self.contributions[i]),
            )
            for i in indices
        ]