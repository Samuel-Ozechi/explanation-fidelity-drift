import numpy as np
import pandas as pd

from scripts.evaluate_exiffi_temporal_explanations import (
    build_class_summary,
    build_summary,
)


def make_metrics():

    return pd.DataFrame(
        {
            "window": [1, 1, 2, 2],
            "isFraud": [0, 1, 0, 1],
            "n_features": [4, 4, 4, 4],
            "top_1_mass": [
                0.4,
                0.5,
                0.3,
                0.6,
            ],
            "top_5_mass": [
                0.7,
                0.8,
                0.6,
                0.9,
            ],
            "top_10_mass": [
                0.9,
                0.95,
                0.85,
                0.98,
            ],
            "entropy": [
                1.0,
                1.2,
                1.1,
                1.3,
            ],
            "effective_features": [
                2.0,
                2.5,
                2.2,
                2.8,
            ],
            "nonzero_features": [
                4,
                4,
                4,
                4,
            ],
        }
    )


def test_build_summary_has_one_row_per_window():

    metrics = make_metrics()

    summary = build_summary(metrics)

    assert len(summary) == 2

    assert summary["window"].tolist() == [
        1,
        2,
    ]


def test_build_summary_preserves_class_counts():

    metrics = make_metrics()

    summary = build_summary(metrics)

    assert summary.loc[
        summary["window"] == 1,
        "n_fraud",
    ].iloc[0] == 1

    assert summary.loc[
        summary["window"] == 1,
        "n_nonfraud",
    ].iloc[0] == 1


def test_build_summary_calculates_top_k_mass():

    metrics = make_metrics()

    summary = build_summary(metrics)

    value = summary.loc[
        summary["window"] == 1,
        "mean_top_1_mass",
    ].iloc[0]

    assert np.isclose(
        value,
        0.45,
    )


def test_class_summary_contains_fraud_and_nonfraud():

    metrics = make_metrics()

    summary = build_class_summary(metrics)

    assert len(summary) == 4

    assert set(
        summary["class"]
    ) == {
        "fraud",
        "nonfraud",
    }