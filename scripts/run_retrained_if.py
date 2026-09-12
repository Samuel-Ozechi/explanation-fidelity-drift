# scripts/run_retrained_if.py
"""
Experiment 2: Retrained Isolation Forest models
Train separate models on each window's data
Compare explanation behavior and fidelity vs. static model
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from efd.data.preprocessing import (
    FeaturePolicy,
    NumericTransactionPreprocessor,
)
from efd.data.temporal import chronological_windows
from efd.models.isolation_forest import IsolationForestDetector
from efd.explanations.exiffi import ExIFFI
from efd.evaluation.explanation import evaluate_explanation_matrix
from efd.evaluation.explanation_fidelity import evaluate_fidelity_matrix


def run_retrained_experiment(
    transactions_path: str,
    output_dir: str,
    train_fraction: float = 0.60,
    n_future_windows: int = 4,
    n_estimators: int = 300,
    random_state: int = 42,
    sample_size: int = 100,
    k_values: List[int] = [1, 3, 5, 10, 20],
):
    """
    Experiment 2: Train separate models for each window.
    Compare static vs. retrained model explanations and fidelity.
    """
    
    print("=" * 70)
    print("EXPERIMENT 2: RETRAINED ISOLATION FOREST MODELS")
    print("=" * 70)
    
    # -------------------------------------------------------------
    # 1. Load data
    # -------------------------------------------------------------
    
    print("\n[1/7] Loading transaction dataset...")
    df = pd.read_csv(transactions_path, low_memory=False)
    print(f"Loaded {len(df):,} transactions")
    
    # -------------------------------------------------------------
    # 2. Chronological split
    # -------------------------------------------------------------
    
    print("\n[2/7] Creating chronological windows...")
    train, future_windows = chronological_windows(
        df,
        time_col="TransactionDT",
        train_fraction=train_fraction,
        n_future_windows=n_future_windows,
    )
    
    print(f"Training: {len(train):,} observations")
    for i, w in enumerate(future_windows, 1):
        print(f"W{i}: {len(w):,} observations (fraud: {w['isFraud'].mean():.4%})")
    
    # -------------------------------------------------------------
    # 3. Preprocess all windows with training-only preprocessing
    # -------------------------------------------------------------
    
    print("\n[3/7] Fitting preprocessing on training period...")
    preprocessor = NumericTransactionPreprocessor(
        policy=FeaturePolicy(
            target="isFraud",
            temporal="TransactionDT",
            identifier="TransactionID",
            max_missing_rate=0.95,
        )
    )
    
    X_train = preprocessor.fit_transform(train)
    feature_names = preprocessor.feature_names_
    y_train = train["isFraud"].to_numpy()
    
    print(f"Features: {len(feature_names)}")
    
    # Transform all windows
    X_windows = []
    y_windows = []
    for window in future_windows:
        X_windows.append(preprocessor.transform(window))
        y_windows.append(window["isFraud"].to_numpy())
    
    # -------------------------------------------------------------
    # 4. Train retrained models for each window
    # -------------------------------------------------------------
    
    print("\n[4/7] Training retrained models...")
    
    models = []
    model_versions = ["static"]  # Start with static model
    
    # Static model
    static_model = IsolationForestDetector(
        n_estimators=n_estimators,
        max_samples="auto",
        contamination="auto",
        random_state=random_state,
        n_jobs=-1,
    )
    static_model.fit(X_train)
    models.append(static_model)
    
    # Retrained models
    for i, (X_window, y_window) in enumerate(zip(X_windows, y_windows), 1):
        print(f"  Training model for Window {i}...")
        model = IsolationForestDetector(
            n_estimators=n_estimators,
            max_samples="auto",
            contamination="auto",
            random_state=random_state + i,
            n_jobs=-1,
        )
        model.fit(X_window)
        models.append(model)
        model_versions.append(f"retrained_w{i}")
    
    # -------------------------------------------------------------
    # 5. Generate explanations for each model-window pair
    # -------------------------------------------------------------
    
    print("\n[5/7] Generating explanations...")
    
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    for model_idx, (model, model_name) in enumerate(zip(models, model_versions)):
        print(f"\n  Model: {model_name}")
        
        explainer = ExIFFI(
            model=model.model,
            feature_names=feature_names,
        )
        
        for window_idx, (X_window, y_window) in enumerate(zip(X_windows, y_windows), 1):
            print(f"    Window {window_idx}...")
            
            # Sample observations
            n = min(sample_size, len(X_window))
            rng = np.random.RandomState(random_state + window_idx)
            indices = rng.choice(len(X_window), size=n, replace=False)
            
            X_sample = X_window.iloc[indices].to_numpy(dtype=float)
            y_sample = y_window[indices]
            
            # Generate explanations
            explanations = explainer.explain_batch(X_sample)
            
            # Extract importance matrix
            importance = np.array([exp.contributions for exp in explanations])
            normalized = np.array([exp.normalized for exp in explanations])
            
            # Compute explanation metrics
            metrics = evaluate_explanation_matrix(
                importance=normalized,
                feature_names=feature_names,
                window=window_idx,
            )
            
            # Add metadata
            metrics["model"] = model_name
            metrics["isFraud"] = y_sample
            
            all_results.append(metrics)
            
            # Compute fidelity
            baseline = np.median(X_train.to_numpy(dtype=float), axis=0)
            
            fidelity_results = evaluate_fidelity_matrix(
                model=model.model,
                X=X_sample,
                explanations=explanations,
                baseline=baseline,
                k_values=k_values,
            )
            
            # Save fidelity results
            fidelity_df = pd.DataFrame([
                {
                    "model": model_name,
                    "window": window_idx,
                    "sample_index": r.sample_index,
                    "top_k": r.top_k,
                    "top_k_mass": r.top_k_mass,
                    "score_change": r.score_change,
                    "relative_score_change": r.relative_score_change,
                    "isFraud": y_sample[r.sample_index],
                }
                for r in fidelity_results
            ])
            
            fidelity_path = output_dir_path / f"retrained_fidelity_{model_name}_w{window_idx}.csv"
            fidelity_df.to_csv(fidelity_path, index=False)
    
    # -------------------------------------------------------------
    # 6. Combine and save results
    # -------------------------------------------------------------
    
    print("\n[6/7] Combining results...")
    
    combined_metrics = pd.concat(all_results, ignore_index=True)
    metrics_path = output_dir_path / "retrained_explanation_metrics.csv"
    combined_metrics.to_csv(metrics_path, index=False)
    
    # Summary by model and window
    summary = combined_metrics.groupby(["model", "window"]).agg({
        "top_1_mass": "mean",
        "top_5_mass": "mean",
        "top_10_mass": "mean",
        "entropy": "mean",
        "effective_features": "mean",
        "nonzero_features": "mean",
    }).reset_index()
    
    summary_path = output_dir_path / "retrained_explanation_summary.csv"
    summary.to_csv(summary_path, index=False)
    
    # -------------------------------------------------------------
    # 7. Print summary
    # -------------------------------------------------------------
    
    print("\n[7/7] Summary:")
    print("\nExplanation metrics by model and window:")
    print(summary.to_string(index=False))
    
    print(f"\nResults saved to: {output_dir_path}")
    
    return combined_metrics, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transactions", required=True)
    parser.add_argument("--output-dir", default="results/retrained")
    parser.add_argument("--train-fraction", type=float, default=0.60)
    parser.add_argument("--windows", type=int, default=4)
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--sample-size", type=int, default=100)
    
    args = parser.parse_args()
    
    run_retrained_experiment(
        transactions_path=args.transactions,
        output_dir=args.output_dir,
        train_fraction=args.train_fraction,
        n_future_windows=args.windows,
        n_estimators=args.trees,
        random_state=args.random_state,
        sample_size=args.sample_size,
    )


if __name__ == "__main__":
    main()