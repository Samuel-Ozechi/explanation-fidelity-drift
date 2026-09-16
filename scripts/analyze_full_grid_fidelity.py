# scripts/analyze_full_grid_fidelity.py
"""
Full 20-pair grid fidelity analysis for the retrained experiment.

Produces comparisons across the complete 5 models × 4 windows grid:
  - Each model (static + 4 retrained) evaluated on each of the 4 windows
  - Fidelity trajectories per model
  - Fidelity trajectories per window
  - Cross-model comparison at each window
  - Diagonal comparison (each retrained model on its own window)
  - Off-diagonal comparison (each retrained model on other windows)
  - Full results tables saved for the paper
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


# ============================================================
# Configuration
# ============================================================

MODELS = ["static", "retrained_w1", "retrained_w2", "retrained_w3", "retrained_w4"]
WINDOWS = [1, 2, 3, 4]
TOP_K_VALUES = [1, 3, 5, 10, 20]
PRIMARY_K = 5


# ============================================================
# Loading
# ============================================================

def load_full_grid(
    results_dir: Path,
) -> pd.DataFrame:
    """
    Load all 20 model-window fidelity files into a single DataFrame.

    Filename pattern:
        retrained_fidelity_{model}_w{window}.csv
    """

    rows = []

    for model in MODELS:
        for window in WINDOWS:

            filename = f"retrained_fidelity_{model}_w{window}.csv"
            path = results_dir / filename

            if not path.exists():
                raise FileNotFoundError(
                    f"Missing grid file: {path}\n"
                    "Run run_retrained_if.py first."
                )

            df = pd.read_csv(path)
            df["model"] = model
            df["window"] = window
            rows.append(df)

    combined = pd.concat(rows, ignore_index=True)

    print(
        f"Loaded {len(combined):,} fidelity records "
        f"across {len(MODELS)} models × {len(WINDOWS)} windows."
    )

    return combined


# ============================================================
# Cell-level summary (model × window × k)
# ============================================================

def build_cell_summary(
    grid_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    One row per (model, window, top_k) cell.

    This is the fundamental result table of the paper:
    5 × 4 × 5 = 100 cells.
    """

    summary = (
        grid_df
        .groupby(["model", "window", "top_k"], as_index=False)
        .agg(
            n_observations=("sample_index", "nunique"),
            mean_top_k_mass=("top_k_mass", "mean"),
            median_top_k_mass=("top_k_mass", "median"),
            mean_relative_score_change=("relative_score_change", "mean"),
            median_relative_score_change=("relative_score_change", "median"),
            std_relative_score_change=("relative_score_change", "std"),
            mean_score_change=("score_change", "mean"),
            median_score_change=("score_change", "median"),
        )
    )

    return summary.sort_values(["model", "window", "top_k"]).reset_index(drop=True)


# ============================================================
# Model trajectories (each model across all windows)
# ============================================================

def build_model_trajectories(
    cell_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each model and k, compute the fidelity trajectory across W1-W4.
    """

    rows = []

    for (model, k), group in cell_summary.groupby(["model", "top_k"]):

        group = group.sort_values("window").reset_index(drop=True)

        baseline = float(group.iloc[0]["mean_relative_score_change"])
        previous = None

        for _, row in group.iterrows():

            current = float(row["mean_relative_score_change"])

            if previous is None:
                delta_prev = 0.0
            else:
                delta_prev = current - previous

            rows.append({
                "model": model,
                "top_k": int(k),
                "window": int(row["window"]),
                "mean_relative_score_change": current,
                "change_from_previous_window": delta_prev,
                "change_from_window_1": current - baseline,
                "percent_change_from_window_1": (
                    ((current - baseline) / baseline) * 100.0
                    if abs(baseline) > 1e-12 else np.nan
                ),
            })

            previous = current

    return pd.DataFrame(rows)


# ============================================================
# Window trajectories (each window across all models)
# ============================================================

def build_window_trajectories(
    cell_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each window and k, compare all 5 models trained on different data.
    Answers: 'At window W, which training distribution produces the most
    faithful explanations for that window?'
    """

    rows = []

    for (window, k), group in cell_summary.groupby(["window", "top_k"]):

        group = group.sort_values("mean_relative_score_change", ascending=False)

        # best model for this window/k
        best_row = group.iloc[0]

        # the retrained model for this window
        own_model = f"retrained_w{window}"
        own_row = group[group["model"] == own_model]

        static_row = group[group["model"] == "static"]

        rows.append({
            "window": int(window),
            "top_k": int(k),
            "best_model": best_row["model"],
            "best_fidelity": float(best_row["mean_relative_score_change"]),
            "static_fidelity": (
                float(static_row.iloc[0]["mean_relative_score_change"])
                if not static_row.empty else np.nan
            ),
            "own_retrained_fidelity": (
                float(own_row.iloc[0]["mean_relative_score_change"])
                if not own_row.empty else np.nan
            ),
            "own_retrained_minus_static": (
                float(own_row.iloc[0]["mean_relative_score_change"])
                - float(static_row.iloc[0]["mean_relative_score_change"])
                if (not own_row.empty and not static_row.empty) else np.nan
            ),
        })

    return pd.DataFrame(rows).sort_values(["window", "top_k"]).reset_index(drop=True)


# ============================================================
# Diagonal vs off-diagonal
# ============================================================

def build_diagonal_analysis(
    cell_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare the diagonal (retrained_wN evaluated on wN) vs
    off-diagonal (retrained_wM evaluated on wN, M != N).

    Answers: 'Does a model trained on the same window it is being
    evaluated on produce more faithful explanations than a model
    trained on a different window?'
    """

    rows = []

    for window in WINDOWS:
        for k in TOP_K_VALUES:

            subset = cell_summary[
                (cell_summary["window"] == window)
                & (cell_summary["top_k"] == k)
            ]

            diagonal_model = f"retrained_w{window}"
            diagonal = subset[subset["model"] == diagonal_model]
            off_diag = subset[
                (subset["model"] != "static")
                & (subset["model"] != diagonal_model)
            ]
            static = subset[subset["model"] == "static"]

            rows.append({
                "window": window,
                "top_k": k,
                "diagonal_fidelity": (
                    float(diagonal.iloc[0]["mean_relative_score_change"])
                    if not diagonal.empty else np.nan
                ),
                "off_diagonal_mean_fidelity": (
                    float(off_diag["mean_relative_score_change"].mean())
                    if not off_diag.empty else np.nan
                ),
                "static_fidelity": (
                    float(static.iloc[0]["mean_relative_score_change"])
                    if not static.empty else np.nan
                ),
                "diagonal_minus_off_diagonal": (
                    float(diagonal.iloc[0]["mean_relative_score_change"])
                    - float(off_diag["mean_relative_score_change"].mean())
                    if (not diagonal.empty and not off_diag.empty) else np.nan
                ),
                "diagonal_minus_static": (
                    float(diagonal.iloc[0]["mean_relative_score_change"])
                    - float(static.iloc[0]["mean_relative_score_change"])
                    if (not diagonal.empty and not static.empty) else np.nan
                ),
            })

    return pd.DataFrame(rows).sort_values(["window", "top_k"]).reset_index(drop=True)


# ============================================================
# Full cross-model comparison at each window
# ============================================================

def build_cross_model_table(
    cell_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Wide table: rows = window × k, columns = each model's fidelity.
    """

    rows = []

    for (window, k), group in cell_summary.groupby(["window", "top_k"]):

        row = {"window": int(window), "top_k": int(k)}

        for model in MODELS:
            match = group[group["model"] == model]
            row[f"fidelity_{model}"] = (
                float(match.iloc[0]["mean_relative_score_change"])
                if not match.empty else np.nan
            )

        # Rank models by fidelity for this cell
        ranked = sorted(
            [(m, row[f"fidelity_{m}"]) for m in MODELS if not np.isnan(row[f"fidelity_{m}"])],
            key=lambda x: -x[1],
        )
        row["best_model"] = ranked[0][0] if ranked else None
        row["worst_model"] = ranked[-1][0] if ranked else None

        rows.append(row)

    return pd.DataFrame(rows).sort_values(["window", "top_k"]).reset_index(drop=True)


# ============================================================
# Statistical comparison: static vs each retrained model
# ============================================================

def build_statistical_comparison(
    grid_df: pd.DataFrame,
    k: int = PRIMARY_K,
) -> pd.DataFrame:
    """
    For each window and each retrained model, test whether the
    retrained model's fidelity differs from the static model's.

    Uses paired Wilcoxon signed-rank test on observations matched
    by sample_index within the same window.
    """

    rows = []

    for window in WINDOWS:

        for retrained_model in [f"retrained_w{w}" for w in WINDOWS]:

            static = grid_df[
                (grid_df["model"] == "static")
                & (grid_df["window"] == window)
                & (grid_df["top_k"] == k)
            ][["sample_index", "relative_score_change"]].rename(
                columns={"relative_score_change": "static_fidelity"}
            )

            retrained = grid_df[
                (grid_df["model"] == retrained_model)
                & (grid_df["window"] == window)
                & (grid_df["top_k"] == k)
            ][["sample_index", "relative_score_change"]].rename(
                columns={"relative_score_change": "retrained_fidelity"}
            )

            paired = static.merge(retrained, on="sample_index", how="inner")

            if paired.empty:
                continue

            diff = paired["retrained_fidelity"] - paired["static_fidelity"]

            # Wilcoxon two-sided
            try:
                stat, pval = wilcoxon(
                    paired["retrained_fidelity"],
                    paired["static_fidelity"],
                    alternative="two-sided",
                    zero_method="wilcox",
                )
            except ValueError:
                stat, pval = np.nan, np.nan

            rows.append({
                "window": window,
                "retrained_model": retrained_model,
                "top_k": k,
                "n_paired": len(paired),
                "static_mean_fidelity": float(paired["static_fidelity"].mean()),
                "retrained_mean_fidelity": float(paired["retrained_fidelity"].mean()),
                "mean_difference": float(diff.mean()),
                "median_difference": float(diff.median()),
                "percent_change": (
                    (diff.mean() / paired["static_fidelity"].mean()) * 100.0
                    if abs(paired["static_fidelity"].mean()) > 1e-12 else np.nan
                ),
                "wilcoxon_statistic": float(stat) if not np.isnan(stat) else np.nan,
                "wilcoxon_pvalue": float(pval) if not np.isnan(pval) else np.nan,
            })

    return pd.DataFrame(rows).sort_values(["window", "retrained_model"]).reset_index(drop=True)


# ============================================================
# Main
# ============================================================

def run_analysis(
    results_dir: str,
    output_dir: str,
    primary_k: int = PRIMARY_K,
) -> Dict[str, pd.DataFrame]:

    results_path = Path(results_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FULL 20-PAIR GRID FIDELITY ANALYSIS")
    print("=" * 70)

    # 1. Load
    print("\n[1/7] Loading full grid...")
    grid_df = load_full_grid(results_path)

    # 2. Cell summary
    print("\n[2/7] Building cell summary (model × window × k)...")
    cell_summary = build_cell_summary(grid_df)
    cell_path = output_path / "full_grid_cell_summary.csv"
    cell_summary.to_csv(cell_path, index=False)
    print(f"  Saved: {cell_path}  ({len(cell_summary)} cells)")

    # 3. Model trajectories
    print("\n[3/7] Building model trajectories...")
    model_traj = build_model_trajectories(cell_summary)
    model_traj_path = output_path / "full_grid_model_trajectories.csv"
    model_traj.to_csv(model_traj_path, index=False)
    print(f"  Saved: {model_traj_path}")

    # 4. Window trajectories
    print("\n[4/7] Building window trajectories...")
    window_traj = build_window_trajectories(cell_summary)
    window_traj_path = output_path / "full_grid_window_trajectories.csv"
    window_traj.to_csv(window_traj_path, index=False)
    print(f"  Saved: {window_traj_path}")

    # 5. Diagonal analysis
    print("\n[5/7] Building diagonal analysis...")
    diagonal_df = build_diagonal_analysis(cell_summary)
    diagonal_path = output_path / "full_grid_diagonal_analysis.csv"
    diagonal_df.to_csv(diagonal_path, index=False)
    print(f"  Saved: {diagonal_path}")

    # 6. Cross-model table
    print("\n[6/7] Building cross-model table...")
    cross_model = build_cross_model_table(cell_summary)
    cross_model_path = output_path / "full_grid_cross_model.csv"
    cross_model.to_csv(cross_model_path, index=False)
    print(f"  Saved: {cross_model_path}")

    # 7. Statistical comparison (static vs each retrained)
    print(f"\n[7/7] Statistical comparison at k={primary_k}...")
    stats_df = build_statistical_comparison(grid_df, k=primary_k)
    stats_path = output_path / "full_grid_statistical_comparison.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"  Saved: {stats_path}")

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print(f"CROSS-MODEL FIDELITY AT k={primary_k}")
    print("=" * 70)

    display_cols = ["window", "top_k"] + [f"fidelity_{m}" for m in MODELS]
    k_filtered = cross_model[cross_model["top_k"] == primary_k]
    print()
    print(k_filtered[display_cols].to_string(index=False))

    print("\n" + "=" * 70)
    print(f"DIAGONAL VS OFF-DIAGONAL AT k={primary_k}")
    print("=" * 70)

    diag_filtered = diagonal_df[diagonal_df["top_k"] == primary_k]
    print()
    print(diag_filtered.to_string(index=False))

    print("\n" + "=" * 70)
    print(f"STATIC VS EACH RETRAINED MODEL AT k={primary_k}")
    print("=" * 70)

    print()
    print(stats_df[[
        "window", "retrained_model",
        "static_mean_fidelity", "retrained_mean_fidelity",
        "mean_difference", "percent_change", "wilcoxon_pvalue",
    ]].to_string(index=False))

    print(f"\nAll results saved to: {output_path}")

    return {
        "cell_summary": cell_summary,
        "model_trajectories": model_traj,
        "window_trajectories": window_traj,
        "diagonal_analysis": diagonal_df,
        "cross_model": cross_model,
        "statistical_comparison": stats_df,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Full 20-pair grid fidelity analysis"
    )
    parser.add_argument(
        "--results-dir",
        default="results/retrained",
        help="Directory containing the 20 fidelity CSVs"
    )
    parser.add_argument(
        "--output-dir",
        default="results/retrained/full_grid",
        help="Output directory for grid analysis"
    )
    parser.add_argument(
        "--primary-k",
        type=int,
        default=PRIMARY_K,
        help="Primary k value for headline comparisons"
    )
    args = parser.parse_args()

    run_analysis(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        primary_k=args.primary_k,
    )


if __name__ == "__main__":
    main()