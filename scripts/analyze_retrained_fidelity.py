# scripts/analyze_retrained_fidelity.py
"""
Analyze fidelity results from retrained models experiment
Saves comprehensive summary tables for the paper
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def load_all_fidelity_results(
    results_dir: Path
) -> pd.DataFrame:
    """Load all fidelity CSV files from retrained experiment."""
    
    fidelity_files = list(results_dir.glob('retrained_fidelity_*.csv'))
    
    if not fidelity_files:
        raise FileNotFoundError(
            f"No fidelity files found in {results_dir}\n"
            "Run run_retrained_if.py first."
        )
    
    all_fidelity = []
    
    for f in fidelity_files:
        df = pd.read_csv(f)
        
        # Parse filename: retrained_fidelity_{model}_w{window}.csv
        parts = f.stem.split('_')
        if len(parts) >= 4:
            model = parts[2]
            window = int(parts[3].replace('w', ''))
        else:
            # Fallback: try to extract from filename
            model = 'unknown'
            window = 0
        
        df['model'] = model
        df['window'] = window
        all_fidelity.append(df)
    
    return pd.concat(all_fidelity, ignore_index=True)


def summarize_fidelity_by_model_window(
    fidelity_df: pd.DataFrame
) -> pd.DataFrame:
    """Create summary table by model and window."""
    
    summary = fidelity_df.groupby(['model', 'window', 'top_k']).agg({
        'relative_score_change': ['mean', 'median', 'std', 'min', 'max'],
        'score_change': ['mean', 'median', 'std'],
        'top_k_mass': ['mean', 'median'],
        'isFraud': 'count'
    }).reset_index()
    
    summary.columns = [
        'model', 'window', 'top_k',
        'mean_relative', 'median_relative', 'std_relative', 
        'min_relative', 'max_relative',
        'mean_score', 'median_score', 'std_score',
        'mean_top_k_mass', 'median_top_k_mass',
        'n_observations'
    ]
    
    return summary


def summarize_fidelity_by_class(
    fidelity_df: pd.DataFrame
) -> pd.DataFrame:
    """Create fraud vs non-fraud fidelity summary."""
    
    summary = fidelity_df.groupby(['model', 'window', 'top_k', 'isFraud']).agg({
        'relative_score_change': ['mean', 'median', 'std'],
        'score_change': ['mean', 'median'],
    }).reset_index()
    
    summary.columns = [
        'model', 'window', 'top_k', 'isFraud',
        'mean_relative', 'median_relative', 'std_relative',
        'mean_score', 'median_score'
    ]
    
    summary['class'] = summary['isFraud'].map({0: 'nonfraud', 1: 'fraud'})
    
    return summary


def compare_static_vs_retrained(
    fidelity_df: pd.DataFrame
) -> pd.DataFrame:
    """Compare static vs retrained models at each window."""
    
    rows = []
    
    for window in fidelity_df['window'].unique():
        for k in fidelity_df['top_k'].unique():
            
            # Get static model results
            static = fidelity_df[
                (fidelity_df['model'] == 'static') &
                (fidelity_df['window'] == window) &
                (fidelity_df['top_k'] == k)
            ]
            
            if static.empty:
                continue
            
            static_mean = static['relative_score_change'].mean()
            
            # Get retrained model results
            # Support both naming conventions: 'retrained' or 'retrained_w{window}'
            retrained = fidelity_df[
                (fidelity_df['model'].isin(['retrained', f'retrained_w{window}'])) &
                (fidelity_df['window'] == window) &
                (fidelity_df['top_k'] == k)
            ]
            
            if retrained.empty:
                continue
            
            retrained_mean = retrained['relative_score_change'].mean()
            
            # Statistical test
            try:
                stat, pval = wilcoxon(
                    static['relative_score_change'].values,
                    retrained['relative_score_change'].values,
                    alternative='two-sided'
                )
            except:
                stat, pval = np.nan, np.nan
            
            rows.append({
                'window': window,
                'top_k': k,
                'static_fidelity': static_mean,
                'retrained_fidelity': retrained_mean,
                'difference': retrained_mean - static_mean,
                'percent_change': ((retrained_mean - static_mean) / static_mean) * 100,
                'wilcoxon_statistic': stat,
                'wilcoxon_pvalue': pval,
                'n_static': len(static),
                'n_retrained': len(retrained)
            })
    
    return pd.DataFrame(rows)


def create_temporal_trajectory(
    fidelity_df: pd.DataFrame
) -> pd.DataFrame:
    """Create temporal trajectory for each model."""
    
    trajectories = []
    
    for model in fidelity_df['model'].unique():
        for k in fidelity_df['top_k'].unique():
            
            subset = fidelity_df[
                (fidelity_df['model'] == model) &
                (fidelity_df['top_k'] == k)
            ].sort_values('window')
            
            if len(subset) < 2:
                continue
            
            # Calculate changes from window 1
            baseline = subset[subset['window'] == 1]['relative_score_change'].mean()
            
            for window in subset['window'].unique():
                window_data = subset[subset['window'] == window]
                mean_fidelity = window_data['relative_score_change'].mean()
                
                trajectories.append({
                    'model': model,
                    'window': window,
                    'top_k': k,
                    'mean_fidelity': mean_fidelity,
                    'change_from_window1': mean_fidelity - baseline,
                    'percent_change_from_window1': ((mean_fidelity - baseline) / baseline) * 100,
                    'n_observations': len(window_data)
                })
    
    return pd.DataFrame(trajectories)


def run_analysis(
    results_dir: str,
    output_dir: str
) -> Dict[str, pd.DataFrame]:
    """Run complete fidelity analysis."""
    
    results_path = Path(results_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("RETRAINED FIDELITY ANALYSIS")
    print("=" * 70)
    
    # 1. Load data
    print("\n[1/5] Loading fidelity results...")
    fidelity_df = load_all_fidelity_results(results_path)
    print(f"Loaded {len(fidelity_df):,} fidelity records")
    print(f"Models: {fidelity_df['model'].unique().tolist()}")
    print(f"Windows: {fidelity_df['window'].unique().tolist()}")
    print(f"Top-k values: {fidelity_df['top_k'].unique().tolist()}")
    
    # 2. Summary by model and window
    print("\n[2/5] Creating model-window summary...")
    summary_df = summarize_fidelity_by_model_window(fidelity_df)
    summary_path = output_path / "retrained_fidelity_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"  Saved: {summary_path}")
    
    # 3. Class-specific summary
    print("\n[3/5] Creating class-specific summary...")
    class_df = summarize_fidelity_by_class(fidelity_df)
    class_path = output_path / "retrained_fidelity_class_summary.csv"
    class_df.to_csv(class_path, index=False)
    print(f"  Saved: {class_path}")
    
    # 4. Static vs retrained comparison
    print("\n[4/5] Comparing static vs retrained...")
    comparison_df = compare_static_vs_retrained(fidelity_df)
    comparison_path = output_path / "retrained_static_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"  Saved: {comparison_path}")
    
    # 5. Temporal trajectories
    print("\n[5/5] Creating temporal trajectories...")
    trajectory_df = create_temporal_trajectory(fidelity_df)
    trajectory_path = output_path / "retrained_fidelity_trajectories.csv"
    trajectory_df.to_csv(trajectory_path, index=False)
    print(f"  Saved: {trajectory_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("FIDELITY ANALYSIS SUMMARY")
    print("=" * 70)
    
    print("\nStatic vs Retrained Comparison (k=5):")
    if 'top_k' in comparison_df.columns and not comparison_df.empty:
        k5_comparison = comparison_df[comparison_df['top_k'] == 5]
        if not k5_comparison.empty:
            print(k5_comparison.to_string(index=False))
    else:
        print('  No comparison results available for printing.')
    
    print("\nTemporal Trajectories (k=5):")
    if 'top_k' in trajectory_df.columns and not trajectory_df.empty:
        k5_traj = trajectory_df[trajectory_df['top_k'] == 5]
        if not k5_traj.empty:
            print(k5_traj.to_string(index=False))
    else:
        print('  No trajectory results available for printing.')
    
    print(f"\nAll results saved to: {output_path}")
    
    return {
        'summary': summary_df,
        'class_summary': class_df,
        'comparison': comparison_df,
        'trajectories': trajectory_df
    }


def main():
    parser = argparse.ArgumentParser(
        description="Analyze retrained model fidelity results"
    )
    parser.add_argument(
        "--results-dir",
        default="results/retrained",
        help="Directory containing retrained fidelity CSVs"
    )
    parser.add_argument(
        "--output-dir",
        default="results/retrained/analysis",
        help="Output directory for analysis results"
    )
    
    args = parser.parse_args()
    
    run_analysis(
        results_dir=args.results_dir,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()