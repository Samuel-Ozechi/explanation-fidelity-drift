# scripts/aggregate_all_results.py
"""
Aggregate all experiment results for the paper
Combines: concept drift, detection, explanation, fidelity
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import numpy as np


def aggregate_results(
    concept_drift_dir: str,
    static_results_dir: str,
    retrained_results_dir: str,
    output_dir: str
) -> None:
    """Aggregate all results into a single paper-ready CSV."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("AGGREGATING ALL RESULTS FOR PAPER")
    print("=" * 70)
    
    results = {}
    
    # 1. Concept drift results
    print("\n[1/5] Loading concept drift results...")
    drift_path = Path(concept_drift_dir) / "concept_drift_results.csv"
    if drift_path.exists():
        drift_df = pd.read_csv(drift_path)
        results['concept_drift'] = drift_df
        print(f"  Loaded: {drift_path}")
    else:
        print("  WARNING: Concept drift results not found")
    
    # 2. Static model detection results
    print("\n[2/5] Loading static model detection results...")
    static_detection_path = Path(static_results_dir) / "static_if_v1.csv"
    if static_detection_path.exists():
        static_detection = pd.read_csv(static_detection_path)
        results['static_detection'] = static_detection
        print(f"  Loaded: {static_detection_path}")
    else:
        print("  WARNING: Static detection results not found")
    
    # 3. Static model explanation results
    print("\n[3/5] Loading static model explanation results...")
    static_explanation_path = Path(static_results_dir) / "static_if_exiffi_explanation_summary.csv"
    if static_explanation_path.exists():
        static_explanation = pd.read_csv(static_explanation_path)
        results['static_explanation'] = static_explanation
        print(f"  Loaded: {static_explanation_path}")
    else:
        print("  WARNING: Static explanation results not found")
    
    # 4. Retrained model results
    print("\n[4/5] Loading retrained model results...")
    retrained_summary_path = Path(retrained_results_dir) / "analysis/retrained_fidelity_summary.csv"
    if retrained_summary_path.exists():
        retrained_summary = pd.read_csv(retrained_summary_path)
        results['retrained_fidelity'] = retrained_summary
        print(f"  Loaded: {retrained_summary_path}")
    else:
        print("  WARNING: Retrained fidelity summary not found")
    
    # 5. Static vs retrained comparison
    print("\n[5/5] Loading static vs retrained comparison...")
    comparison_path = Path(retrained_results_dir) / "analysis/retrained_static_comparison.csv"
    if comparison_path.exists():
        comparison = pd.read_csv(comparison_path)
        results['static_vs_retrained'] = comparison
        print(f"  Loaded: {comparison_path}")
    else:
        print("  WARNING: Static vs retrained comparison not found")
    
    # Save all results
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    
    for name, df in results.items():
        path = output_path / f"paper_{name}.csv"
        df.to_csv(path, index=False)
        print(f"  {path}")
    
    # Create combined summary table for paper
    print("\nCreating combined summary table...")
    
    combined_rows = []
    
    # Static model detection
    if 'static_detection' in results:
        for _, row in results['static_detection'].iterrows():
            combined_rows.append({
                'experiment': 'static_detection',
                'window': row.get('window', 0),
                'metric': 'f1',
                'value': row.get('f1', np.nan)
            })
            combined_rows.append({
                'experiment': 'static_detection',
                'window': row.get('window', 0),
                'metric': 'pr_auc',
                'value': row.get('pr_auc', np.nan)
            })
    
    # Static model explanation
    if 'static_explanation' in results:
        for _, row in results['static_explanation'].iterrows():
            combined_rows.append({
                'experiment': 'static_explanation',
                'window': row.get('window', 0),
                'metric': 'top_1_mass',
                'value': row.get('mean_top_1_mass', np.nan)
            })
            combined_rows.append({
                'experiment': 'static_explanation',
                'window': row.get('window', 0),
                'metric': 'entropy',
                'value': row.get('mean_entropy', np.nan)
            })
    
    # Concept drift
    if 'concept_drift' in results:
        for _, row in results['concept_drift'].iterrows():
            combined_rows.append({
                'experiment': 'concept_drift',
                'window': row.get('window', 0),
                'metric': 'performance_drop',
                'value': row.get('performance_drop', np.nan)
            })
            combined_rows.append({
                'experiment': 'concept_drift',
                'window': row.get('window', 0),
                'metric': 'drift_significant',
                'value': row.get('drift_significant', False)
            })
    
    combined_df = pd.DataFrame(combined_rows)
    combined_path = output_path / "paper_combined_summary.csv"
    combined_df.to_csv(combined_path, index=False)
    print(f"  {combined_path}")
    
    print("\n" + "=" * 70)
    print("AGGREGATION COMPLETE")
    print("=" * 70)
    
    # Print summary
    print("\nResults summary:")
    for name, df in results.items():
        print(f"  {name}: {len(df)} rows")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate all results for paper"
    )
    parser.add_argument(
        "--concept-drift-dir",
        default="results/concept_drift",
        help="Directory with concept drift results"
    )
    parser.add_argument(
        "--static-results-dir",
        default="results/tables",
        help="Directory with static model results"
    )
    parser.add_argument(
        "--retrained-results-dir",
        default="results/retrained",
        help="Directory with retrained model results"
    )
    parser.add_argument(
        "--output-dir",
        default="results/paper_ready",
        help="Output directory for aggregated results"
    )
    
    args = parser.parse_args()
    
    aggregate_results(
        concept_drift_dir=args.concept_drift_dir,
        static_results_dir=args.static_results_dir,
        retrained_results_dir=args.retrained_results_dir,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()