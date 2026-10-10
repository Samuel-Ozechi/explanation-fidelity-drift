# scripts/analyze_transition_level_drift.py
"""
Transition-level analysis of drift, explanation disparity, and fidelity change.

For each window transition (WN-1 → WN, and optionally long-range pairs):

    Predictors:
        - input drift         (from consecutive_window_drift.csv)
        - explanation drift   (from temporal_drift_summary.csv)

    Outcome:
        - fidelity drop over the same transition
          (fidelity at window_a minus fidelity at window_b, for the static model)

Then correlates each predictor against the outcome.

Reads:
    results/concept_drift/consecutive_window_drift.csv
    results/tables/static_if_exiffi_temporal_drift_summary.csv
    results/retrained/full_grid/full_grid_cell_summary.csv

Writes:
    results/concept_drift/transition_level_analysis/
        transition_table.csv
        transition_correlations.csv
        figure_7_7_1_drift_vs_fidelity.{png,pdf}
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt


PRIMARY_K = 5


# ============================================================
# Loaders
# ============================================================

def load_consecutive_drift(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"window_a", "window_b", "mean_js_divergence", "is_adjacent"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    return df


def load_explanation_drift(path: Path) -> pd.DataFrame:
    """
    Explanation-space drift between windows, computed on static model
    explanations. Produced by earlier runs of the drift analysis.
    """
    df = pd.read_csv(path)
    required = {"window_a", "window_b", "js_divergence", "cosine_distance"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    return df[[
        "window_a", "window_b",
        "js_divergence", "cosine_distance",
        "l1_distance", "l2_distance",
        "spearman_rank_correlation",
    ]].rename(columns={
        "js_divergence": "explanation_js",
        "cosine_distance": "explanation_cosine",
        "l1_distance": "explanation_l1",
        "l2_distance": "explanation_l2",
        "spearman_rank_correlation": "explanation_rank_corr",
    })


def load_static_fidelity(path: Path) -> pd.DataFrame:
    """
    Extract the static model's mean fidelity at k=5 for each window.
    """
    df = pd.read_csv(path)
    df = df[(df["model"] == "static") & (df["top_k"] == PRIMARY_K)]
    return df[["window", "mean_relative_score_change"]].rename(
        columns={"mean_relative_score_change": "fidelity"}
    )


# ============================================================
# Assemble transition table
# ============================================================

def build_transition_table(
    input_drift: pd.DataFrame,
    explanation_drift: pd.DataFrame,
    fidelity: pd.DataFrame,
) -> pd.DataFrame:

    # Merge input drift and explanation drift on (window_a, window_b)
    merged = input_drift.merge(
        explanation_drift,
        on=["window_a", "window_b"],
        how="inner",
        validate="one_to_one",
    )

    # Join fidelity at window_a and window_b
    fid_a = fidelity.rename(columns={
        "window": "window_a",
        "fidelity": "fidelity_a",
    })
    fid_b = fidelity.rename(columns={
        "window": "window_b",
        "fidelity": "fidelity_b",
    })

    merged = merged.merge(fid_a, on="window_a", how="left")
    merged = merged.merge(fid_b, on="window_b", how="left")

    # Fidelity drop is fidelity at window_a minus fidelity at window_b,
    # so a positive number means fidelity declined.
    merged["fidelity_drop"] = merged["fidelity_a"] - merged["fidelity_b"]

    return merged


# ============================================================
# Correlations
# ============================================================

def correlate(
    df: pd.DataFrame,
    predictor_columns: list[str],
    outcome_column: str,
) -> pd.DataFrame:

    rows = []

    for predictor in predictor_columns:

        x = df[predictor].to_numpy(dtype=float)
        y = df[outcome_column].to_numpy(dtype=float)

        valid = np.isfinite(x) & np.isfinite(y)
        x, y = x[valid], y[valid]

        if len(x) < 3:
            rows.append({
                "predictor": predictor,
                "outcome": outcome_column,
                "n": int(len(x)),
                "pearson": np.nan,
                "pearson_p": np.nan,
                "spearman": np.nan,
                "spearman_p": np.nan,
            })
            continue

        r_p, p_p = pearsonr(x, y)
        r_s, p_s = spearmanr(x, y)

        rows.append({
            "predictor": predictor,
            "outcome": outcome_column,
            "n": int(len(x)),
            "pearson": float(r_p),
            "pearson_p": float(p_p),
            "spearman": float(r_s),
            "spearman_p": float(p_s),
        })

    return pd.DataFrame(rows)


# ============================================================
# Figure
# ============================================================

def make_figure(df: pd.DataFrame, output_dir: Path) -> None:

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ---- Panel A: input drift vs fidelity drop ---------------
    ax = axes[0]
    for _, row in df.iterrows():
        color = "#1f77b4" if row["is_adjacent"] else "#7f7f7f"
        marker = "o" if row["is_adjacent"] else "s"
        ax.scatter(
            row["mean_js_divergence"],
            row["fidelity_drop"],
            s=100, color=color, marker=marker,
            edgecolor="black", zorder=3,
        )
        ax.annotate(
            row["comparison"],
            xy=(row["mean_js_divergence"], row["fidelity_drop"]),
            xytext=(6, 6), textcoords="offset points",
            fontsize=10,
        )
    ax.set_xlabel("Input drift (mean JS divergence between windows)", fontsize=12)
    ax.set_ylabel("Fidelity drop over transition (k=5)", fontsize=12)
    ax.set_title("(a) Input drift vs fidelity drop", fontsize=12)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)

    # ---- Panel B: explanation drift vs fidelity drop ---------
    ax = axes[1]
    for _, row in df.iterrows():
        color = "#1f77b4" if row["is_adjacent"] else "#7f7f7f"
        marker = "o" if row["is_adjacent"] else "s"
        ax.scatter(
            row["explanation_cosine"],
            row["fidelity_drop"],
            s=100, color=color, marker=marker,
            edgecolor="black", zorder=3,
        )
        ax.annotate(
            row["comparison"],
            xy=(row["explanation_cosine"], row["fidelity_drop"]),
            xytext=(6, 6), textcoords="offset points",
            fontsize=10,
        )
    ax.set_xlabel("Explanation drift (cosine distance)", fontsize=12)
    ax.set_ylabel("Fidelity drop over transition (k=5)", fontsize=12)
    ax.set_title("(b) Explanation drift vs fidelity drop", fontsize=12)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)

    # Legend entries for adjacent vs long-range
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f77b4",
               markeredgecolor="black", markersize=10, label="Adjacent"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#7f7f7f",
               markeredgecolor="black", markersize=10, label="Long-range"),
    ]
    axes[0].legend(handles=legend_handles, loc="best", frameon=False)

    plt.tight_layout()

    png_path = output_dir / "figure_7_7_1_drift_vs_fidelity.png"
    pdf_path = output_dir / "figure_7_7_1_drift_vs_fidelity.pdf"
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"  Saved: {png_path}")
    print(f"  Saved: {pdf_path}")


# ============================================================
# Main
# ============================================================

def run(
    consecutive_drift_path: Path,
    explanation_drift_path: Path,
    fidelity_path: Path,
    output_dir: Path,
) -> None:

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("TRANSITION-LEVEL DRIFT / FIDELITY ANALYSIS")
    print("=" * 70)

    print("\n[1/4] Loading inputs...")
    input_drift = load_consecutive_drift(consecutive_drift_path)
    explanation_drift = load_explanation_drift(explanation_drift_path)
    fidelity = load_static_fidelity(fidelity_path)
    print(f"  Input drift rows:       {len(input_drift)}")
    print(f"  Explanation drift rows: {len(explanation_drift)}")
    print(f"  Fidelity rows:          {len(fidelity)}")

    print("\n[2/4] Building transition table...")
    transition = build_transition_table(
        input_drift, explanation_drift, fidelity
    )
    transition_path = output_dir / "transition_table.csv"
    transition.to_csv(transition_path, index=False)
    print(f"  Saved: {transition_path}")

    # --------------------------------------------------------
    # Adjacent-only and full-set correlations
    # --------------------------------------------------------

    predictors = [
        "mean_js_divergence",
        "max_ks_statistic",
        "explanation_cosine",
        "explanation_js",
        "explanation_l1",
        "explanation_l2",
        "explanation_rank_corr",
    ]

    print("\n[3/4] Correlating predictors against fidelity drop...")

    full_corr = correlate(transition, predictors, "fidelity_drop")
    full_corr["subset"] = "all"

    adjacent = transition[transition["is_adjacent"]].reset_index(drop=True)
    adjacent_corr = correlate(adjacent, predictors, "fidelity_drop")
    adjacent_corr["subset"] = "adjacent"

    all_corrs = pd.concat([full_corr, adjacent_corr], ignore_index=True)

    corr_path = output_dir / "transition_correlations.csv"
    all_corrs.to_csv(corr_path, index=False)
    print(f"  Saved: {corr_path}")

    print("\n[4/4] Generating figure...")
    make_figure(transition, output_dir)

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRANSITION TABLE")
    print("=" * 70)
    print(transition[[
        "comparison", "is_adjacent",
        "mean_js_divergence", "max_ks_statistic",
        "explanation_cosine", "explanation_js",
        "fidelity_a", "fidelity_b", "fidelity_drop",
    ]].to_string(index=False))

    print("\n" + "=" * 70)
    print("CORRELATIONS WITH FIDELITY DROP")
    print("=" * 70)
    print(all_corrs[[
        "subset", "predictor", "n",
        "pearson", "pearson_p", "spearman", "spearman_p",
    ]].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description="Transition-level drift / fidelity analysis."
    )
    parser.add_argument(
        "--consecutive-drift",
        default="results/concept_drift/consecutive_window_drift.csv",
    )
    parser.add_argument(
        "--explanation-drift",
        default="results/tables/static_if_exiffi_temporal_drift_summary.csv",
    )
    parser.add_argument(
        "--fidelity",
        default="results/retrained/full_grid/full_grid_cell_summary.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="results/concept_drift/transition_level_analysis",
    )
    args = parser.parse_args()

    run(
        consecutive_drift_path=Path(args.consecutive_drift),
        explanation_drift_path=Path(args.explanation_drift),
        fidelity_path=Path(args.fidelity),
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()