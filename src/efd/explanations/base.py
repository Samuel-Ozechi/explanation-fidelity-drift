from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from efd.explanations.types import (
    LocalExplanation,
)


class ExplanationMethod(ABC):
    """
    Common interface for model explanation methods.
    """

    @abstractmethod
    def explain(
        self,
        x: np.ndarray,
        sample_index: int = 0,
    ) -> LocalExplanation:
        """
        Explain one model decision.
        """
        raise NotImplementedError

    @abstractmethod
    def explain_batch(
        self,
        x: np.ndarray,
    ) -> list[LocalExplanation]:
        """
        Explain multiple model decisions.
        """
        raise NotImplementedError