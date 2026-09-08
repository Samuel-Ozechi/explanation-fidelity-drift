# scripts/detect_concept_drift.py
"""
Concept Drift Detection for IEEE-CIS Transaction Dataset

This script tests for statistically significant concept drift:
- Change in P(Y|X) between training period and each future window
- Quantifies drift magnitude using multiple metrics
- Applies BH-FDR correction for multiple comparisons
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.stats import (
    ks_2samp,
    chi2_contingency,
    mannwhitneyu,
    false_discovery_control
)
from scipy.spatial.distance import jensenshannon
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from efd.data.preprocessing import (
    FeaturePolicy,
    NumericTransactionPreprocessor,
)
from efd.data.temporal import chronological_windows


@dataclass
class DriftResult:
    """Container for concept drift test results."""
    window: int
    n_observations: int
    fraud_rate: float
    
    # P(Y|X) change indicators
    performance_drop: float  # Change in predictive performance
    p_value_performance: float
    
    # Feature distribution drift
    mean_js_divergence: float
    max_ks_statistic: float
    n_features_significant: int
    significant_features: List[str]
    
    # Overall drift
    drift_magnitude: float  # Combined measure
    drift_significant: bool
    fdr_adjusted_pvalue: float


def calculate_feature_js_divergence(
    X_train: np.ndarray,
    X_window: np.ndarray,
    feature_names: List[str],
    n_bins: int = 20
) -> Dict[str, float]:
    """
    Calculate Jensen-Shannon divergence for each feature between
    training and window distributions.
    """
    js_divergences = {}
    
    for i, feature in enumerate(feature_names):
        # Get feature values
        train_vals = X_train[:, i]
        window_vals = X_window[:, i]
        
        # Remove NaN
        train_vals = train_vals[~np.isnan(train_vals)]
        window_vals = window_vals[~np.isnan(window_vals)]
        
        if len(train_vals) < 2 or len(window_vals) < 2:
            js_divergences[feature] = 0.0
            continue
        
        # Create histograms
        all_vals = np.concatenate([train_vals, window_vals])
        bins = np.linspace(
            np.percentile(all_vals, 1),
            np.percentile(all_vals, 99),
            n_bins
        )
        
        hist_train, _ = np.histogram(train_vals, bins=bins, density=True)
        hist_window, _ = np.histogram(window_vals, bins=bins, density=True)
        
        # Normalize to probabilities
        hist_train = hist_train / (hist_train.sum() + 1e-12)
        hist_window = hist_window / (hist_window.sum() + 1e-12)
        
        # Calculate JS divergence
        js = jensenshannon(hist_train, hist_window, base=2.0) ** 2
        js_divergences[feature] = float(js)
    
    return js_divergences


def test_concept_drift(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_window: np.ndarray,
    y_window: np.ndarray,
    feature_names: List[str],
    n_permutations: int = 1000,
    random_state: int = 42
) -> DriftResult:
    """
    Test for concept drift using multiple approaches:
    1. Predictive performance drop (P(Y|X) change)
    2. Feature distribution drift (JS divergence)
    3. Permutation tests for significance
    """
    
    rng = np.random.RandomState(random_state)
    n_train = len(X_train)
    n_window = len(X_window)
    
    # -----------------------------------------------------------------
    # 1. Predictive Performance Test (P(Y|X) change)
    # -----------------------------------------------------------------
    
    # Train a simple model on training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_window_scaled = scaler.transform(X_window)
    
    # Use Random Forest for robustness
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=random_state,
        n_jobs=-1
    )
    
    # Performance on training (cross-validation)
    cv_scores = cross_val_score(
        clf, X_train_scaled, y_train,
        cv=5, scoring='roc_auc'
    )
    train_performance = cv_scores.mean()
    
    # Performance on window
    clf.fit(X_train_scaled, y_train)
    window_scores = clf.predict_proba(X_window_scaled)[:, 1]
    
    # Calculate ROC-AUC on window
    from sklearn.metrics import roc_auc_score
    window_performance = roc_auc_score(y_window, window_scores)
    
    performance_drop = train_performance - window_performance
    
    # -----------------------------------------------------------------
    # 2. Permutation Test for Performance Drop Significance
    # -----------------------------------------------------------------
    
    # Combine data
    X_combined = np.vstack([X_train, X_window])
    y_combined = np.concatenate([y_train, y_window])
    
    null_drops = []
    for _ in range(n_permutations):
        # Permute labels
        perm_idx = rng.permutation(len(y_combined))
        y_perm = y_combined[perm_idx]
        
        # Split back
        y_train_perm = y_perm[:n_train]
        y_window_perm = y_perm[n_train:]
        
        # Train model on permuted data
        clf_perm = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=random_state,
            n_jobs=-1
        )
        
        # Performance on permuted training
        cv_scores_perm = cross_val_score(
            clf_perm, X_train_scaled, y_train_perm,
            cv=3, scoring='roc_auc'
        )
        train_perm = cv_scores_perm.mean()
        
        # Performance on permuted window
        clf_perm.fit(X_train_scaled, y_train_perm)
        window_scores_perm = clf_perm.predict_proba(X_window_scaled)[:, 1]
        window_perm = roc_auc_score(y_window_perm, window_scores_perm)
        
        null_drops.append(train_perm - window_perm)
    
    p_value_performance = (
        np.sum(np.array(null_drops) >= performance_drop) + 1
    ) / (n_permutations + 1)
    
    # -----------------------------------------------------------------
    # 3. Feature Distribution Drift
    # -----------------------------------------------------------------
    
    js_divergences = calculate_feature_js_divergence(
        X_train, X_window, feature_names
    )
    
    mean_js_divergence = np.mean(list(js_divergences.values()))
    max_ks_statistic = 0.0
    
    # Count features with significant KS test
    significant_features = []
    for i, feature in enumerate(feature_names):
        _, p_value = ks_2samp(X_train[:, i], X_window[:, i])
        if p_value < 0.05:
            significant_features.append(feature)
            # Track max KS statistic
            ks_stat, _ = ks_2samp(X_train[:, i], X_window[:, i])
            max_ks_statistic = max(max_ks_statistic, ks_stat)
    
    # -----------------------------------------------------------------
    # 4. Overall Drift Magnitude
    # -----------------------------------------------------------------
    
    # Combined measure: average of normalized metrics
    drift_magnitude = (
        performance_drop / 0.5 +  # Normalize by max expected drop
        mean_js_divergence / 0.1 +  # Normalize JS
        len(significant_features) / len(feature_names)
    ) / 3.0
    
    # -----------------------------------------------------------------
    # 5. Significance via FDR
    # -----------------------------------------------------------------
    
    # We'll do FDR correction across all windows later
    # For now, use permutation test p-value
    drift_significant = p_value_performance < 0.05
    
    return DriftResult(
        window=0,  # Will set later
        n_observations=n_window,
        fraud_rate=np.mean(y_window),
        performance_drop=performance_drop,
        p_value_performance=p_value_performance,
        mean_js_divergence=mean_js_divergence,
        max_ks_statistic=max_ks_statistic,
        n_features_significant=len(significant_features),
        significant_features=significant_features[:10],  # Top 10
        drift_magnitude=drift_magnitude,
        drift_significant=drift_significant,
        fdr_adjusted_pvalue=p_value_performance  # Will update after FDR
    )


def run_concept_drift_analysis(
    transactions_path: str,
    output_dir: str,
    train_fraction: float = 0.60,
    n_future_windows: int = 4,
    n_permutations: int = 1000,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Run complete concept drift analysis on IEEE-CIS dataset.
    """
    
    print("=" * 70)
    print("CONCEPT DRIFT DETECTION - IEEE-CIS")
    print("=" * 70)
    
    # -----------------------------------------------------------------
    # 1. Load Data
    # -----------------------------------------------------------------
    
    print("\n[1/5] Loading transaction dataset...")
    df = pd.read_csv(transactions_path, low_memory=False)
    print(f"Loaded {len(df):,} transactions")
    
    # -----------------------------------------------------------------
    # 2. Chronological Split
    # -----------------------------------------------------------------
    
    print("\n[2/5] Creating chronological windows...")
    train, future_windows = chronological_windows(
        df,
        time_col="TransactionDT",
        train_fraction=train_fraction,
        n_future_windows=n_future_windows,
    )
    
    print(f"Training: {len(train):,} observations")
    for i, w in enumerate(future_windows, 1):
        print(f"W{i}: {len(w):,} observations (fraud rate: {w['isFraud'].mean():.4%})")
    
    # -----------------------------------------------------------------
    # 3. Preprocessing (Training-Only)
    # -----------------------------------------------------------------
    
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
    
    # -----------------------------------------------------------------
    # 4. Detect Drift for Each Window
    # -----------------------------------------------------------------
    
    print("\n[4/5] Testing for concept drift...")
    
    results = []
    
    for i, window in enumerate(future_windows, 1):
        print(f"\nWindow {i}")
        
        X_window = preprocessor.transform(window)
        y_window = window["isFraud"].to_numpy()
        
        # Convert to numpy for faster processing
        X_train_np = X_train.to_numpy(dtype=float)
        X_window_np = X_window.to_numpy(dtype=float)
        
        result = test_concept_drift(
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
    
    # -----------------------------------------------------------------
    # 5. FDR Correction
    # -----------------------------------------------------------------
    
    print("\n[5/5] Applying Benjamini-Hochberg FDR correction...")
    
    p_values = [r.p_value_performance for r in results]
    adjusted_p = false_discovery_control(p_values, method="bh")
    
    for result, adj_p in zip(results, adjusted_p):
        result.fdr_adjusted_pvalue = adj_p
        result.drift_significant = adj_p < 0.05
    
    # -----------------------------------------------------------------
    # 6. Save Results
    # -----------------------------------------------------------------
    
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Create DataFrame
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
    
    output_path = output_dir_path / "concept_drift_results.csv"
    results_df.to_csv(output_path, index=False)
    
    # Save detailed feature-level drift for each window
    feature_drift_path = output_dir_path / "feature_drift_detailed.csv"
    feature_rows = []
    
    for i, window in enumerate(future_windows, 1):
        X_window = preprocessor.transform(window)
        X_train_np = X_train.to_numpy(dtype=float)
        X_window_np = X_window.to_numpy(dtype=float)
        
        js_divs = calculate_feature_js_divergence(
            X_train_np, X_window_np, feature_names
        )
        
        for feature, js in js_divs.items():
            _, p_val = ks_2samp(X_train_np[:, feature_names.index(feature)], 
                               X_window_np[:, feature_names.index(feature)])
            feature_rows.append({
                "window": i,
                "feature": feature,
                "js_divergence": js,
                "ks_p_value": p_val,
            })
    
    feature_df = pd.DataFrame(feature_rows)
    feature_df.to_csv(feature_drift_path, index=False)
    
    # -----------------------------------------------------------------
    # 7. Summary
    # -----------------------------------------------------------------
    
    print("\n" + "=" * 70)
    print("CONCEPT DRIFT RESULTS SUMMARY")
    print("=" * 70)
    print()
    print(results_df.to_string(index=False))
    
    print(f"\nResults saved to: {output_path}")
    print(f"Feature-level results saved to: {feature_drift_path}")
    
    return results_df


def main():
    parser = argparse.ArgumentParser(
        description="Detect concept drift in IEEE-CIS transaction dataset"
    )
    parser.add_argument(
        "--transactions",
        required=True,
        help="Path to train_transaction.csv"
    )
    parser.add_argument(
        "--output-dir",
        default="results/concept_drift",
        help="Output directory for results"
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.60,
        help="Fraction of data for training"
    )
    parser.add_argument(
        "--windows",
        type=int,
        default=4,
        help="Number of future windows"
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=1000,
        help="Number of permutation iterations"
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    run_concept_drift_analysis(
        transactions_path=args.transactions,
        output_dir=args.output_dir,
        train_fraction=args.train_fraction,
        n_future_windows=args.windows,
        n_permutations=args.permutations,
        random_state=args.random_state
    )


if __name__ == "__main__":
    main()