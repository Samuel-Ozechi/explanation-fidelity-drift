import pandas as pd
import numpy as np

from efd.experiments.exiffi_static_if import (
    load_baseline_1_artifacts,
)
from efd.explanations.exiffi import ExIFFI


model, X, feature_names = load_baseline_1_artifacts(
    transactions_path=(
        "data/raw/ieee_cis/train_transaction.csv"
    ),
    train_fraction=0.60,
    n_future_windows=4,
    n_estimators=300,
    contamination="auto",
    random_state=42,
)


explainer = ExIFFI(
    model=model,
    feature_names=feature_names,
)

X_sample = X[:100]

explanations = explainer.explain_batch(
    X_sample
)

rows = []

for explanation in explanations:

    importance = explanation.normalized

    ranking = np.argsort(
        -importance
    )[:10]

    row = {
        "sample_index":
            explanation.sample_index,

        "anomaly_score":
            explanation.anomaly_score,
    }

    for rank, feature_idx in enumerate(
        ranking,
        start=1,
    ):

        row[f"rank_{rank}_feature"] = (
            feature_names[feature_idx]
        )

        row[f"rank_{rank}_importance"] = (
            importance[feature_idx]
        )

    rows.append(row)


output = pd.DataFrame(rows)

output.to_csv(
    "results/tables/exiffi_smoke.csv",
    index=False,
)

print(output.head())
