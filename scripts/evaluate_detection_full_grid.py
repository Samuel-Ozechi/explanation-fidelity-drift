# scripts/evaluate_detection_full_grid.py
"""
Detection performance across the full 20-pair grid.

Mirrors analyze_full_grid_fidelity.py but for detection metrics:
  - 5 models (static + 4 retrained) × 4 windows = 20 cells
  - F1, PR-AUC, fraud rate, recall, precision per cell
  - Cross-model comparison at each window
  - Diagonal vs off-diagonal analysis
  - Static vs retrained statistical comparison
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

from efd.data.preprocessing import (
    FeaturePolicy,
    NumericTransactionPreprocessor,
)
from efd.data.temporal import chronological_windows
from efd.models.isolation_forest import IsolationForestDetector


# ============================================================
# Configuration
# ============================================================

MODELS = ["static", "retrained_w1", "retrained_w2", "retrained_w3", "retrained_w4"]
WINDOWS = [1, 2, 3, 4]

DEFAULT_TRAIN_FRACTION = 0.60
DEFAULT_N_ESTIMATORS = 300
DEFAULT_RANDOM_STATE = 42
DEFAULT_MAX_MISSING_RATE = 0.95


# ============================================================
# Model and data preparation
# ============================================================

def prepare_models_and_windows(
    transactions_path: str,
    train_fraction: float,
    n_estimators: int,
    random_state: int,
    max_missing_rate: float,
) -> tuple[
    Dict[str, IsolationForestDetector],
    List[pd.DataFrame],
    NumericTransactionPreprocessor,
]:
    """
    Reproduce the exact training setup from run_retrained_if.py:
      - preprocessing fitted on training period
      - one static model on training period
      - one retrained model per window

    Returns:
        models dict, list of transformed future windows, fitted preprocessor
    """

    print("\n[1/4] Loading transaction dataset...")
    df = pd.read_csv(transactions_path, low_memory=False)
    print(f"Loaded {len(df):,} transactions")

    print("\n[2/4] Creating chronological windows...")
    train, future_windows = chronological_windows(
        df,
        time_col="TransactionDT",
        train_fraction=train_fraction,
        n_future_windows=4,
    )
    print(f"Training: {len(train):,}")
    for i, w in enumerate(future_windows, 1):
        print(f"W{i}: {len(w):,} (fraud: {w['isFraud'].mean():.4%})")

    print("\n[3/4] Fitting preprocessing on training period...")
    preprocessor = NumericTransactionPreprocessor(
        policy=FeaturePolicy(
            target="isFraud",
            temporal="TransactionDT",
            identifier="TransactionID",
            max_missing_rate=max_missing_rate,
        )
    )
    X_train = preprocessor.fit_transform(train)
    print(f"Features: {len(preprocessor.feature_names_)}")

    print("\n[4/4] Training 5 Isolation Forest models...")

    models: Dict[str, IsolationForestDetector] = {}

    # Static model
    static = IsolationForestDetector(
        n_estimators=n_estimators,
        max_samples="auto",
        contamination="auto",
        random_state=random_state,
        n_jobs=-1,
    )
    static.fit(X_train)
    models["static"] = static
    print("  static trained on training period")

    # Retrained models
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
        print(f"  retrained_w{i} trained on window {i}")

    return models, future_windows, preprocessor


# ============================================================
# Detection metrics
# ============================================================

def compute_detection_metrics(
    model: IsolationForestDetector,
    X_window: pd.DataFrame,
    y_window: np.ndarray,
) -> Dict[str, float]:
    """
    Compute detection metrics for one model on one window.

    Uses the model's own anomaly_score (higher = more anomalous) and
    its predict_anomaly (0/1) output. Thresholds are model-internal
    (sklearn contamination='auto' => 0.5 quantile), so all models
    use consistent thresholds.
    """

    scores = model.anomaly_score(X_window)
    predictions = model.predict_anomaly(X_window)

    metrics = {
        "n": int(len(y_window)),
        "fraud": int(y_window.sum()),
        "fraud_rate": float(y_window.mean()),
        "predicted_anomalies": int(predictions.sum()),
        "predicted_anomaly_rate": float(predictions.mean()),
        "f1": float(f1_score(y_window, predictions, zero_division=0)),
        "precision": float(precision_score(y_window, predictions, zero_division=0)),
        "recall": float(recall_score(y_window, predictions, zero_division=0)),
        "pr_auc": float(average_precision_score(y_window, scores)),
    }

    return metrics


# ============================================================
# Full grid evaluation
# ============================================================

def evaluate_full_grid(
    models: Dict[str, IsolationForestDetector],
    future_windows: List[pd.DataFrame],
    preprocessor: NumericTransactionPreprocessor,
) -> pd.DataFrame:
    """
    Evaluate every model on every window.
    Produces the 5 × 4 = 20-cell detection grid.
    """

    rows = []

    for model_name in MODELS:

        model = models[model_name]

        for window_idx, window in enumerate(future_windows, 1):

            X_window = preprocessor.transform(window)
            y_window = window["isFraud"].to_numpy()

            metrics = compute_detection_metrics(model, X_window, y_window)
            metrics["model"] = model_name
            metrics["window"] = window_idx

            rows.append(metrics)

    return pd.DataFrame(rows)[
        ["model", "window", "n", "fraud", "fraud_rate",
         "predicted_anomalies", "predicted_anomaly_rate",
         "f1", "precision", "recall", "pr_auc"]
    ].sort_values(["model", "window"]).reset_index(drop=True)


# ============================================================
# Analyses
# ============================================================

def build_cross_model_table(grid_df: pd.DataFrame) -> pd.DataFrame:
    """
    Wide table: rows = window, columns = each model's F1 and PR-AUC.
    """

    rows = []

    for window, group in grid_df.groupby("window"):

        row = {"window": int(window)}

        for model in MODELS:
            match = group[group["model"] == model]
            if not match.empty:
                row[f"f1_{model}"] = float(match.iloc[0]["f1"])
                row[f"pr_auc_{model}"] = float(match.iloc[0]["pr_auc"])

        # Best model by F1 and by PR-AUC
        f1_cols = [f"f1_{m}" for m in MODELS if f"f1_{m}" in row]
        pr_cols = [f"pr_auc_{m}" for m in MODELS if f"pr_auc_{m}" in row]

        if f1_cols:
            row["best_f1_model"] = max(f1_cols, key=lambda c: row[c]).replace("f1_", "")
            row["worst_f1_model"] = min(f1_cols, key=lambda c: row[c]).replace("f1_", "")
        if pr_cols:
            row["best_pr_auc_model"] = max(pr_cols, key=lambda c: row[c]).replace("pr_auc_", "")
            row["worst_pr_auc_model"] = min(pr_cols, key=lambda c: row[c]).replace("pr_auc_", "")

        rows.append(row)

    return pd.DataFrame(rows).sort_values("window").reset_index(drop=True)


def build_diagonal_analysis(grid_df: pd.DataFrame) -> pd.DataFrame:
    """
    Diagonal (retrained_wN on window N) vs off-diagonal vs static.
    """

    rows = []

    for window in WINDOWS:

        subset = grid_df[grid_df["window"] == window]

        diagonal_model = f"retrained_w{window}"
        diagonal = subset[subset["model"] == diagonal_model]
        off_diag = subset[
            (subset["model"] != "static")
            & (subset["model"] != diagonal_model)
        ]
        static = subset[subset["model"] == "static"]

        rows.append({
            "window": window,
            "diagonal_model": diagonal_model,
            "diagonal_f1": (
                float(diagonal.iloc[0]["f1"]) if not diagonal.empty else np.nan
            ),
            "off_diagonal_mean_f1": float(off_diag["f1"].mean()),
            "static_f1": (
                float(static.iloc[0]["f1"]) if not static.empty else np.nan
            ),
            "diagonal_f1_minus_static": (
                float(diagonal.iloc[0]["f1"]) - float(static.iloc[0]["f1"])
                if (not diagonal.empty and not static.empty) else np.nan
            ),
            "diagonal_pr_auc": (
                float(diagonal.iloc[0]["pr_auc"]) if not diagonal.empty else np.nan
            ),
            "off_diagonal_mean_pr_auc": float(off_diag["pr_auc"].mean()),
            "static_pr_auc": (
                float(static.iloc[0]["pr_auc"]) if not static.empty else np.nan
            ),
            "diagonal_pr_auc_minus_static": (
                float(diagonal.iloc[0]["pr_auc"]) - float(static.iloc[0]["pr_auc"])
                if (not diagonal.empty and not static.empty) else np.nan
            ),
        })

    return pd.DataFrame(rows).sort_values("window").reset_index(drop=True)


def build_model_trajectories(grid_df: pd.DataFrame) -> pd.DataFrame:
    """
    Each model's F1 and PR-AUC across W1-W4.
    """

    rows = []

    for model, group in grid_df.groupby("model"):
        group = group.sort_values("window").reset_index(drop=True)

        baseline_f1 = float(group.iloc[0]["f1"])
        baseline_pr = float(group.iloc[0]["pr_auc"])

        for _, row in group.iterrows():

            rows.append({
                "model": model,
                "window": int(row["window"]),
                "f1": float(row["f1"]),
                "pr_auc": float(row["pr_auc"]),
                "f1_change_from_window_1": float(row["f1"]) - baseline_f1,
                "pr_auc_change_from_window_1": float(row["pr_auc"]) - baseline_pr,
                "f1_relative_pct": (
                    ((float(row["f1"]) - baseline_f1) / baseline_f1) * 100.0
                    if abs(baseline_f1) > 1e-12 else np.nan
                ),
                "pr_auc_relative_pct": (
                    ((float(row["pr_auc"]) - baseline_pr) / baseline_pr) * 100.0
                    if abs(baseline_pr) > 1e-12 else np.nan
                ),
            })

    return pd.DataFrame(rows).sort_values(["model", "window"]).reset_index(drop=True)


# ============================================================
# Main
# ============================================================

def run_analysis(
    transactions_path: str,
    output_dir: str,
    train_fraction: float = DEFAULT_TRAIN_FRACTION,
    n_estimators: int = DEFAULT_N_ESTIMATORS,
    random_state: int = DEFAULT_RANDOM_STATE,
    max_missing_rate: float = DEFAULT_MAX_MISSING_RATE,
) -> Dict[str, pd.DataFrame]:

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FULL 20-PAIR GRID DETECTION ANALYSIS")
    print("=" * 70)

    # 1. Reproduce models and windows
    models, future_windows, preprocessor = prepare_models_and_windows(
        transactions_path=transactions_path,
        train_fraction=train_fraction,
        n_estimators=n_estimators,
        random_state=random_state,
        max_missing_rate=max_missing_rate,
    )

    # 2. Evaluate the full grid
    print("\nEvaluating full 5 × 4 grid...")
    grid_df = evaluate_full_grid(models, future_windows, preprocessor)

    grid_path = output_path / "full_grid_detection.csv"
    grid_df.to_csv(grid_path, index=False)
    print(f"  Saved: {grid_path}  ({len(grid_df)} cells)")

    # 3. Cross-model table
    print("\nBuilding cross-model table...")
    cross_model = build_cross_model_table(grid_df)
    cross_path = output_path / "full_grid_detection_cross_model.csv"
    cross_model.to_csv(cross_path, index=False)
    print(f"  Saved: {cross_path}")

    # 4. Diagonal analysis
    print("\nBuilding diagonal analysis...")
    diagonal_df = build_diagonal_analysis(grid_df)
    diagonal_path = output_path / "full_grid_detection_diagonal.csv"
    diagonal_df.to_csv(diagonal_path, index=False)
    print(f"  Saved: {diagonal_path}")

    # 5. Model trajectories
    print("\nBuilding model trajectories...")
    trajectory_df = build_model_trajectories(grid_df)
    trajectory_path = output_path / "full_grid_detection_trajectories.csv"
    trajectory_df.to_csv(trajectory_path, index=False)
    print(f"  Saved: {trajectory_path}")

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("DETECTION F1 — CROSS-MODEL AT EACH WINDOW")
    print("=" * 70)

    f1_cols = ["window"] + [f"f1_{m}" for m in MODELS]
    print()
    print(cross_model[f1_cols].to_string(index=False))

    print("\n" + "=" * 70)
    print("DETECTION PR-AUC — CROSS-MODEL AT EACH WINDOW")
    print("=" * 70)

    pr_cols = ["window"] + [f"pr_auc_{m}" for m in MODELS]
    print()
    print(cross_model[pr_cols].to_string(index=False))

    print("\n" + "=" * 70)
    print("DIAGONAL VS STATIC")
    print("=" * 70)

    print()
    print(diagonal_df[[
        "window", "diagonal_model",
        "diagonal_f1", "static_f1", "diagonal_f1_minus_static",
        "diagonal_pr_auc", "static_pr_auc", "diagonal_pr_auc_minus_static",
    ]].to_string(index=False))

    print("\n" + "=" * 70)
    print("MODEL TRAJECTORIES")
    print("=" * 70)

    print()
    print(trajectory_df.to_string(index=False))

    print(f"\nAll detection grid results saved to: {output_path}")

    return {
        "grid": grid_df,
        "cross_model": cross_model,
        "diagonal": diagonal_df,
        "trajectories": trajectory_df,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Full 20-pair grid detection evaluation"
    )
    parser.add_argument("--transactions", required=True)
    parser.add_argument("--output-dir", default="results/retrained/full_grid")
    parser.add_argument("--train-fraction", type=float, default=DEFAULT_TRAIN_FRACTION)
    parser.add_argument("--estimators", type=int, default=DEFAULT_N_ESTIMATORS)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--max-missing-rate", type=float, default=DEFAULT_MAX_MISSING_RATE)
    args = parser.parse_args()

    run_analysis(
        transactions_path=args.transactions,
        output_dir=args.output_dir,
        train_fraction=args.train_fraction,
        n_estimators=args.estimators,
        random_state=args.random_state,
        max_missing_rate=args.max_missing_rate,
    )


if __name__ == "__main__":
    main()