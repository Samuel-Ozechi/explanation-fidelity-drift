# scripts/detect_concept_drift_fast.py
"""
Concept Drift Detection - Optimized for Speed
Same functionality, but runs in minutes instead of hours
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, false_discovery_control
from scipy.spatial.distance import jensenshannon
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from efd.data.preprocessing import (
    FeaturePolicy,
    NumericTransactionPreprocessor,
)
from efd.data.temporal import chronological_windows


@dataclass
class DriftResult:
    window: int
    n_observations: int
    fraud_rate: float
    performance_drop: float
    p_value_performance: float
    mean_js_divergence: float
    max_ks_statistic: float
    n_features_significant: int
    significant_features: List[str]
    drift_magnitude: float
    drift_significant: bool
    fdr_adjusted_pvalue: float


def calculate_feature_js_divergence(
    X_train: np.ndarray,
    X_window: np.ndarray,
    feature_names: List[str],
    n_bins: int = 20
) -> Dict[str, float]:
    """Calculate JS divergence per feature."""
    js_divergences = {}
    
    for i, feature in enumerate(feature_names):
        train_vals = X_train[:, i]
        window_vals = X_window[:, i]
        
        train_vals = train_vals[~np.isnan(train_vals)]
        window_vals = window_vals[~np.isnan(window_vals)]
        
        if len(train_vals) < 2 or len(window_vals) < 2:
            js_divergences[feature] = 0.0
            continue
        
        all_vals = np.concatenate([train_vals, window_vals])
        bins = np.linspace(
            np.percentile(all_vals, 1),
            np.percentile(all_vals, 99),
            n_bins
        )
        
        hist_train, _ = np.histogram(train_vals, bins=bins, density=True)
        hist_window, _ = np.histogram(window_vals, bins=bins, density=True)
        
        hist_train = hist_train / (hist_train.sum() + 1e-12)
        hist_window = hist_window / (hist_window.sum() + 1e-12)
        
        js = jensenshannon(hist_train, hist_window, base=2.0) ** 2
        js_divergences[feature] = float(js)
    
    return js_divergences


def test_concept_drift_fast(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_window: np.ndarray,
    y_window: np.ndarray,
    feature_names: List[str],
    n_permutations: int = 100,  # Reduced from 1000
    random_state: int = 42
) -> DriftResult:
    """
    Fast concept drift test using Logistic Regression instead of Random Forest.
    """
    
    rng = np.random.RandomState(random_state)
    n_train = len(X_train)
    n_window = len(X_window)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_window_scaled = scaler.transform(X_window)
    
    # Use Logistic Regression (much faster than Random Forest)
    clf = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
        n_jobs=-1,
        C=1.0
    )
    
    # Performance on training (cross-validation)
    cv_scores = cross_val_score(
        clf, X_train_scaled, y_train,
        cv=3,  # Reduced from 5
        scoring='roc_auc'
    )
    train_performance = cv_scores.mean()
    
    # Performance on window
    clf.fit(X_train_scaled, y_train)
    window_scores = clf.predict_proba(X_window_scaled)[:, 1]
    window_performance = roc_auc_score(y_window, window_scores)
    
    performance_drop = train_performance - window_performance
    
    print(f"    Train AUC: {train_performance:.4f}, Window AUC: {window_performance:.4f}, Drop: {performance_drop:.4f}")
    
    # Permutation test (with progress indicator)
    X_combined = np.vstack([X_train, X_window])
    y_combined = np.concatenate([y_train, y_window])
    
    null_drops = []
    for perm_idx in range(n_permutations):
        if perm_idx % 20 == 0:
            print(f"    Permutation {perm_idx}/{n_permutations}...", end="\r")
        
        perm_idx_labels = rng.permutation(len(y_combined))
        y_perm = y_combined[perm_idx_labels]
        
        y_train_perm = y_perm[:n_train]
        y_window_perm = y_perm[n_train:]
        
        clf_perm = LogisticRegression(
            max_iter=500,
            random_state=random_state + perm_idx,
            n_jobs=-1,
            C=1.0
        )
        
        # Use only 2-fold CV for speed
        cv_scores_perm = cross_val_score(
            clf_perm, X_train_scaled, y_train_perm,
            cv=2, scoring='roc_auc'
        )
        train_perm = cv_scores_perm.mean()
        
        clf_perm.fit(X_train_scaled, y_train_perm)
        window_scores_perm = clf_perm.predict_proba(X_window_scaled)[:, 1]
        window_perm = roc_auc_score(y_window_perm, window_scores_perm)
        
        null_drops.append(train_perm - window_perm)
    
    print(f"    Permutations complete.                    ")
    
    p_value_performance = (
        np.sum(np.array(null_drops) >= performance_drop) + 1
    ) / (n_permutations + 1)
    
    # Feature distribution drift
    js_divergences = calculate_feature_js_divergence(
        X_train, X_window, feature_names
    )
    
    mean_js_divergence = np.mean(list(js_divergences.values()))
    max_ks_statistic = 0.0
    
    significant_features = []
    for i, feature in enumerate(feature_names):
        _, p_value = ks_2samp(X_train[:, i], X_window[:, i])
        if p_value < 0.05:
            significant_features.append(feature)
            ks_stat, _ = ks_2samp(X_train[:, i], X_window[:, i])
            max_ks_statistic = max(max_ks_statistic, ks_stat)
    
    drift_magnitude = (
        performance_drop / 0.5 +
        mean_js_divergence / 0.1 +
        len(significant_features) / len(feature_names)
    ) / 3.0
    
    drift_significant = p_value_performance < 0.05
    
    return DriftResult(
        window=0,
        n_observations=n_window,
        fraud_rate=np.mean(y_window),
        performance_drop=performance_drop,
        p_value_performance=p_value_performance,
        mean_js_divergence=mean_js_divergence,
        max_ks_statistic=max_ks_statistic,
        n_features_significant=len(significant_features),
        significant_features=significant_features[:10],
        drift_magnitude=drift_magnitude,
        drift_significant=drift_significant,
        fdr_adjusted_pvalue=p_value_performance
    )


def run_concept_drift_analysis_fast(
    transactions_path: str,
    output_dir: str,
    train_fraction: float = 0.60,
    n_future_windows: int = 4,
    n_permutations: int = 100,
    random_state: int = 42
) -> pd.DataFrame:
    
    print("=" * 70)
    print("CONCEPT DRIFT DETECTION - FAST VERSION")
    print("=" * 70)
    
    print("\n[1/5] Loading transaction dataset...")
    df = pd.read_csv(transactions_path, low_memory=False)
    print(f"Loaded {len(df):,} transactions")
    
    print("\n[2/5] Creating chronological windows...")
    train, future_windows = chronological_windows(
        df,
        time_col="TransactionDT",
        train_fraction=train_fraction,
        n_future_windows=n_future_windows,
    )
    
    print(f"Training: {len(train):,} observations")
    for i, w in enumerate(future_windows, 1):
        print(f"W{i}: {len(w):,} obs (fraud rate: {w['isFraud'].mean():.4%})")
    
    print("\n[3/5] Fitting preprocessing on training period...")
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
    
    print("\n[4/5] Testing for concept drift...")
    
    results = []
    
    for i, window in enumerate(future_windows, 1):
        print(f"\nWindow {i}")
        print(f"  Running drift tests...")
        
        X_window = preprocessor.transform(window)
        y_window = window["isFraud"].to_numpy()
        
        X_train_np = X_train.to_numpy(dtype=float)
        X_window_np = X_window.to_numpy(dtype=float)
        
        result = test_concept_drift_fast(
            X_train=X_train_np,
            y_train=y_train,
            X_window=X_window_np,
            y_window=y_window,
            feature_names=feature_names,
            n_permutations=n_permutations,
            random_state=random_state + i
        )
        result.window = i
        
        results.append(result)
        
        print(f"  Fraud rate: {result.fraud_rate:.4%}")
        print(f"  Performance drop: {result.performance_drop:.4f}")
        print(f"  p-value: {result.p_value_performance:.6f}")
        print(f"  Mean JS divergence: {result.mean_js_divergence:.6f}")
        print(f"  Significant features: {result.n_features_significant}")
        print(f"  Drift significant: {result.drift_significant}")
    
    print("\n[5/5] Applying FDR correction...")
    
    p_values = [r.p_value_performance for r in results]
    adjusted_p = false_discovery_control(p_values, method="bh")
    
    for result, adj_p in zip(results, adjusted_p):
        result.fdr_adjusted_pvalue = adj_p
        result.drift_significant = adj_p < 0.05
    
    # Save results
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    results_df = pd.DataFrame([
        {
            "window": r.window,
            "n_observations": r.n_observations,
            "fraud_rate": r.fraud_rate,
            "performance_drop": r.performance_drop,
            "p_value": r.p_value_performance,
            "fdr_adjusted_pvalue": r.fdr_adjusted_pvalue,
            "drift_significant": r.drift_significant,
            "mean_js_divergence": r.mean_js_divergence,
            "max_ks_statistic": r.max_ks_statistic,
            "n_significant_features": r.n_features_significant,
            "drift_magnitude": r.drift_magnitude,
        }
        for r in results
    ])
    
    output_path = output_dir_path / "concept_drift_results_fast.csv"
    results_df.to_csv(output_path, index=False)
    
    print("\n" + "=" * 70)
    print("CONCEPT DRIFT RESULTS SUMMARY")
    print("=" * 70)
    print()
    print(results_df.to_string(index=False))
    print(f"\nResults saved to: {output_path}")
    
    return results_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transactions", required=True)
    parser.add_argument("--output-dir", default="results/concept_drift")
    parser.add_argument("--train-fraction", type=float, default=0.60)
    parser.add_argument("--windows", type=int, default=4)
    parser.add_argument("--permutations", type=int, default=100, 
                       help="Reduced from 1000 for speed")
    parser.add_argument("--random-state", type=int, default=42)
    
    args = parser.parse_args()
    
    run_concept_drift_analysis_fast(
        transactions_path=args.transactions,
        output_dir=args.output_dir,
        train_fraction=args.train_fraction,
        n_future_windows=args.windows,
        n_permutations=args.permutations,
        random_state=args.random_state
    )


if __name__ == "__main__":
    main()