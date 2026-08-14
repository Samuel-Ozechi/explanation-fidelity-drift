from __future__ import annotations

import numpy as np

from sklearn.metrics import (
    average_precision_score,
    f1_score,
)


def evaluate_detection(
    y_true,
    anomaly_scores,
    predictions,
) -> dict:

    y_true = np.asarray(
        y_true
    )

    anomaly_scores = np.asarray(
        anomaly_scores
    )

    predictions = np.asarray(
        predictions
    )

    if len(y_true) == 0:
        raise ValueError(
            "Cannot evaluate an empty window."
        )

    result = {
        "n": int(
            len(y_true)
        ),

        "fraud": int(
            y_true.sum()
        ),

        "fraud_rate": float(
            y_true.mean()
        ),

        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),

        "pr_auc": float(
            average_precision_score(
                y_true,
                anomaly_scores,
            )
        ),
    }

    return result