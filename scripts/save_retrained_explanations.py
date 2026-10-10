# scripts/save_retrained_explanations.py
"""
Persist retrained model explanation matrices to disk.

The original run_retrained_if.py computed explanations in memory but
only saved fidelity results. This script reproduces the same models
and windows, generates the explanations, and writes each model-window
explanation matrix to CSV in the same format as the static model's
explanations:

    results/retrained/retrained_explanations_{model}_w{window}.csv

Each file contains one row per sampled observation, with columns:
    sample_index
    isFraud
    anomaly_score
    importance__<feature>  (one column per feature)

Reproduces the exact sample used in run_retrained_if.py (random_state
+ window_index).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from efd.data.preprocessing import (
    FeaturePolicy,
    NumericTransactionPreprocessor,
)
from efd.data.temporal import chronological_windows
from efd.models.isolation_forest import IsolationForestDetector
from efd.explanations.exiffi import ExIFFI


DEFAULT_TRAIN_FRACTION = 0.60
DEFAULT_N_ESTIMATORS = 300
DEFAULT_RANDOM_STATE = 42
DEFAULT_SAMPLE_SIZE = 100
DEFAULT_MAX_MISSING_RATE = 0.95


def build_models_and_windows(
    transactions_path: str,
    train_fraction: float,
    n_estimators: int,
    random_state: int,
    max_missing_rate: float,
):
    """Reproduce the exact setup used in run_retrained_if.py."""

    print("[1/4] Loading transactions...")
    df = pd.read_csv(transactions_path, low_memory=False)
    print(f"  Loaded {len(df):,} rows")

    print("[2/4] Creating chronological windows...")
    train, future_windows = chronological_windows(
        df,
        time_col="TransactionDT",
        train_fraction=train_fraction,
        n_future_windows=4,
    )
    print(f"  Training: {len(train):,}")
    for i, w in enumerate(future_windows, 1):
        print(f"  W{i}: {len(w):,}")

    print("[3/4] Fitting preprocessing on training period...")
    preprocessor = NumericTransactionPreprocessor(
        policy=FeaturePolicy(
            target="isFraud",
            temporal="TransactionDT",
            identifier="TransactionID",
            max_missing_rate=max_missing_rate,
        )
    )
    X_train = preprocessor.fit_transform(train)
    feature_names = preprocessor.feature_names_
    print(f"  Features: {len(feature_names)}")

    print("[4/4] Training static + 4 retrained models...")

    models: Dict[str, IsolationForestDetector] = {}

    static = IsolationForestDetector(
        n_estimators=n_estimators,
        max_samples="auto",
        contamination="auto",
        random_state=random_state,
        n_jobs=-1,
    )
    static.fit(X_train)
    models["static"] = static

    for i, window in enumerate(future_windows, 1):
        X_window = preprocessor.transform(window)
        model = IsolationForestDetector(
            n_estimators=n_estimators,
            max_samples="auto",
            contamination="auto",
            random_state=random_state + i,
            n_jobs=-1,
        )
        model.fit(X_window)
        models[f"retrained_w{i}"] = model

    return models, future_windows, preprocessor, feature_names


def save_explanations_for_pair(
    model: IsolationForestDetector,
    X_window: pd.DataFrame,
    y_window: np.ndarray,
    feature_names: List[str],
    sample_size: int,
    seed: int,
    output_path: Path,
):
    """Sample observations, generate explanations, write to CSV."""

    n = min(sample_size, len(X_window))
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(X_window), size=n, replace=False)

    X_sample = X_window.iloc[indices].to_numpy(dtype=float)
    y_sample = y_window[indices]

    explainer = ExIFFI(
        model=model.model,
        feature_names=feature_names,
    )

    explanations = explainer.explain_batch(X_sample)

    rows = []
    for i, exp in enumerate(explanations):
        normalized = exp.normalized
        row = {
            "sample_index": i,
            "isFraud": int(y_sample[i]),
            "anomaly_score": float(-exp.anomaly_score),
        }
        for feature_name, value in zip(feature_names, normalized):
            row[f"importance__{feature_name}"] = float(value)
        rows.append(row)

    pd.DataFrame(rows).to_csv(output_path, index=False)


def run(
    transactions_path: str,
    output_dir: str,
    train_fraction: float,
    n_estimators: int,
    random_state: int,
    sample_size: int,
    max_missing_rate: float,
):

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SAVING RETRAINED EXPLANATION MATRICES")
    print("=" * 70)

    models, future_windows, preprocessor, feature_names = (
        build_models_and_windows(
            transactions_path=transactions_path,
            train_fraction=train_fraction,
            n_estimators=n_estimators,
            random_state=random_state,
            max_missing_rate=max_missing_rate,
        )
    )

    # Transform windows once
    X_windows = [preprocessor.transform(w) for w in future_windows]
    y_windows = [w["isFraud"].to_numpy() for w in future_windows]

    print("\nSaving explanations...")

    for model_name, model in models.items():
        for window_idx, (X_window, y_window) in enumerate(
            zip(X_windows, y_windows), 1
        ):

            seed = random_state + window_idx

            output_file = output_path / (
                f"retrained_explanations_{model_name}_w{window_idx}.csv"
            )

            save_explanations_for_pair(
                model=model,
                X_window=X_window,
                y_window=y_window,
                feature_names=feature_names,
                sample_size=sample_size,
                seed=seed,
                output_path=output_file,
            )

            print(f"  Saved: {output_file.name}")

    print("\nDone.")


def main():
    parser = argparse.ArgumentParser(
        description="Persist retrained model explanation matrices to disk."
    )
    parser.add_argument("--transactions", required=True)
    parser.add_argument("--output-dir", default="results/retrained")
    parser.add_argument("--train-fraction", type=float, default=DEFAULT_TRAIN_FRACTION)
    parser.add_argument("--estimators", type=int, default=DEFAULT_N_ESTIMATORS)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--max-missing-rate", type=float, default=DEFAULT_MAX_MISSING_RATE)
    args = parser.parse_args()

    run(
        transactions_path=args.transactions,
        output_dir=args.output_dir,
        train_fraction=args.train_fraction,
        n_estimators=args.estimators,
        random_state=args.random_state,
        sample_size=args.sample_size,
        max_missing_rate=args.max_missing_rate,
    )


if __name__ == "__main__":
    main()