# scripts/create_paper_figures.py
"""
Create figures and tables for the paper
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def create_fidelity_trajectory_plot(
    trajectory_df: pd.DataFrame,
    output_dir: Path
) -> None:
    """Figure: Fidelity trajectories for static vs retrained models."""
    
    plt.figure(figsize=(10, 6))
    
    # Filter for k=5
    k5 = trajectory_df[trajectory_df['top_k'] == 5]
    
    # Plot each model
    for model in k5['model'].unique():
        data = k5[k5['model'] == model].sort_values('window')
        label = 'Static Model' if model == 'static' else f'Retrained (W{model[-1]})'
        plt.plot(data['window'], data['mean_fidelity'], 
                marker='o', linewidth=2, label=label)
    
    plt.xlabel('Temporal Window', fontsize=12)
    plt.ylabel('Mean Relative Score Change (Fidelity)', fontsize=12)
    plt.title('Explanation Fidelity Trajectories: Static vs Retrained Models', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks([1, 2, 3, 4])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'figure_fidelity_trajectory.png', dpi=300)
    plt.close()
    print(f"Saved: {output_dir / 'figure_fidelity_trajectory.png'}")


def create_detection_trajectory_plot(
    detection_df: pd.DataFrame,
    output_dir: Path
) -> None:
    """Figure: Detection performance trajectory."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # F1 trajectory
    ax1.plot(detection_df['window'], detection_df['f1'], 
            marker='o', linewidth=2, color='blue')
    ax1.set_xlabel('Temporal Window')
    ax1.set_ylabel('F1 Score')
    ax1.set_title('Detection Performance: F1')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks([1, 2, 3, 4])
    
    # PR-AUC trajectory
    ax2.plot(detection_df['window'], detection_df['pr_auc'], 
            marker='s', linewidth=2, color='red')
    ax2.set_xlabel('Temporal Window')
    ax2.set_ylabel('PR-AUC')
    ax2.set_title('Detection Performance: PR-AUC')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks([1, 2, 3, 4])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'figure_detection_trajectory.png', dpi=300)
    plt.close()
    print(f"Saved: {output_dir / 'figure_detection_trajectory.png'}")


def create_fidelity_by_k_plot(
    summary_df: pd.DataFrame,
    output_dir: Path
) -> None:
    """Figure: Fidelity by top-k value for each model/window."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    for idx, window in enumerate([1, 2, 3, 4]):
        ax = axes[idx // 2, idx % 2]
        
        data = summary_df[summary_df['window'] == window]
        
        # Static model
        static = data[data['model'] == 'static']
        ax.plot(static['top_k'], static['mean_relative'], 
                marker='o', linewidth=2, label='Static', color='blue')
        
        # Retrained model
        retrained = data[data['model'] == f'retrained_w{window}']
        ax.plot(retrained['top_k'], retrained['mean_relative'], 
                marker='s', linewidth=2, label='Retrained', color='orange')
        
        ax.set_xlabel('k (number of features)')
        ax.set_ylabel('Relative Score Change')
        ax.set_title(f'Window {window}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'figure_fidelity_by_k.png', dpi=300)
    plt.close()
    print(f"Saved: {output_dir / 'figure_fidelity_by_k.png'}")


def create_paper_tables(
    summary_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    detection_df: pd.DataFrame,
    drift_df: pd.DataFrame,
    output_dir: Path
) -> None:
    """Create LaTeX-ready tables for the paper."""
    
    # Table 1: Detection results
    detection_table = detection_df[['window', 'f1', 'pr_auc', 'fraud_rate']].copy()
    detection_table.columns = ['Window', 'F1', 'PR-AUC', 'Fraud Rate']
    detection_table['Fraud Rate'] = detection_table['Fraud Rate'].apply(lambda x: f'{x:.4%}')
    detection_table.to_csv(output_dir / 'table_detection.csv', index=False)
    print(f"Saved: {output_dir / 'table_detection.csv'}")
    
    # Table 2: Fidelity comparison (k=5)
    k5 = comparison_df[comparison_df['top_k'] == 5].copy()
    k5_table = k5[['window', 'static_fidelity', 'retrained_fidelity', 'percent_change']].copy()
    k5_table.columns = ['Window', 'Static Model', 'Retrained Model', 'Change (%)']
    k5_table.to_csv(output_dir / 'table_fidelity_comparison.csv', index=False)
    print(f"Saved: {output_dir / 'table_fidelity_comparison.csv'}")
    
    # Table 3: Fidelity degradation summary
    k5_traj = summary_df[summary_df['top_k'] == 5]
    traj_summary = []
    for model in k5_traj['model'].unique():
        data = k5_traj[k5_traj['model'] == model].sort_values('window')
        w1 = data[data['window'] == 1]['mean_relative'].values[0]
        w4 = data[data['window'] == 4]['mean_relative'].values[0]
        traj_summary.append({
            'Model': model,
            'W1 Fidelity': f'{w1:.4f}',
            'W4 Fidelity': f'{w4:.4f}',
            'Absolute Change': f'{w4 - w1:.4f}',
            'Percent Change': f'{((w4 - w1) / w1 * 100):.2f}%'
        })
    traj_df = pd.DataFrame(traj_summary)
    traj_df.to_csv(output_dir / 'table_fidelity_degradation.csv', index=False)
    print(f"Saved: {output_dir / 'table_fidelity_degradation.csv'}")
    
    # Table 4: Concept drift summary
    drift_table = drift_df[['window', 'performance_drop', 'p_value', 'fdr_adjusted_pvalue', 'drift_significant']].copy()
    drift_table.columns = ['Window', 'Performance Drop', 'p-value', 'FDR p-value', 'Significant']
    drift_table.to_csv(output_dir / 'table_concept_drift.csv', index=False)
    print(f"Saved: {output_dir / 'table_concept_drift.csv'}")


def main():
    """Create all paper figures and tables."""
    
    # Load data
    results_dir = Path('results/paper_ready')
    retrained_dir = Path('results/retrained/analysis')
    output_dir = Path('results/paper_figures')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("CREATING PAPER FIGURES AND TABLES")
    print("=" * 70)
    
    # Load data
    detection_df = pd.read_csv(results_dir / 'paper_static_detection.csv')
    drift_df = pd.read_csv(results_dir / 'paper_concept_drift.csv')
    summary_df = pd.read_csv(retrained_dir / 'retrained_fidelity_summary.csv')
    comparison_df = pd.read_csv(retrained_dir / 'retrained_static_comparison.csv')
    trajectory_df = pd.read_csv(retrained_dir / 'retrained_fidelity_trajectories.csv')
    
    print(f"Loaded detection: {len(detection_df)} rows")
    print(f"Loaded drift: {len(drift_df)} rows")
    print(f"Loaded fidelity summary: {len(summary_df)} rows")
    print(f"Loaded comparison: {len(comparison_df)} rows")
    
    # Create figures
    print("\n[1/4] Creating figures...")
    create_fidelity_trajectory_plot(trajectory_df, output_dir)
    create_detection_trajectory_plot(detection_df, output_dir)
    create_fidelity_by_k_plot(summary_df, output_dir)
    
    # Create tables
    print("\n[2/4] Creating tables...")
    create_paper_tables(summary_df, comparison_df, detection_df, drift_df, output_dir)
    
    # Create summary statistics
    print("\n[3/4] Creating summary statistics...")
    
    # Overall findings
    findings = {
        'finding': [
            'Concept drift (P(Y|X)) after FDR correction',
            'Covariate shift observed',
            'Static model fidelity decline W1→W4',
            'Retrained model fidelity decline W1→W4',
            'Static vs retrained fidelity difference (W4)',
            'Detection performance decline W1→W4 (F1)',
            'Detection performance decline W1→W4 (PR-AUC)'
        ],
        'value': [
            'None significant',
            '200+ features per window',
            '-9.1%',
            '-4.5%',
            '-7.0%',
            '-24.4%',
            '-46.8%'
        ]
    }
    findings_df = pd.DataFrame(findings)
    findings_df.to_csv(output_dir / 'summary_findings.csv', index=False)
    print(f"Saved: {output_dir / 'summary_findings.csv'}")
    
    print("\n[4/4] All figures and tables created!")
    print(f"Output directory: {output_dir}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("KEY FINDINGS SUMMARY")
    print("=" * 70)
    print("\n", findings_df.to_string(index=False))


if __name__ == "__main__":
    main()