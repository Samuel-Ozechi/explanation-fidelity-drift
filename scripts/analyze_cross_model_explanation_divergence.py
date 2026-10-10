# scripts/analyze_cross_model_explanation_divergence.py
"""
Cross-model explanation divergence and fidelity gap analysis.

Computes pairwise explanation divergence between the static model and
each retrained model at the same window, and relates:

    - explanation divergence (cosine, JS, L1, L2, Spearman rank)
    - fidelity gap (static fidelity - retrained fidelity)
    - input drift (raw feature-space JS divergence from Section 7.1)

Design:
    - Only within-window comparisons are reported.
    - Four windows x four retrained models = 16 static-vs-retrained pairs.
    - Diagonal pairs (retrained_wN at WN) are the primary comparison.
    - Off-diagonal pairs are reported as robustness.

Reads:
    results/explanations/static_if_exiffi_w{N}_sample.csv          (static)
    results/retrained/retrained_fidelity_{model}_w{window}.csv     (retrained)
        -- note: explanations for retrained models are also needed;
           see below for the expected filename convention.
    results/retrained/full_grid/full_grid_cell_summary.csv
    results/concept_drift/covariate_shift_summary.csv

Writes:
    results/retrained/cross_model_divergence/
        pairwise_divergence.csv
        per_window_summary.csv
        correlations.csv
        figure_7_7_1_divergence_vs_fidelity.{png,pdf}
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt


MODELS = ["static", "retrained_w1", "retrained_w2", "retrained_w3", "retrained_w4"]
WINDOWS = [1, 2, 3, 4]
PRIMARY_K = 5


# ============================================================
# Loading explanation matrices
# ============================================================

def load_explanation_matrix(
    path: Path,
) -> Tuple[np.ndarray, List[str]]:
    """
    Load an explanation CSV and return the importance matrix
    plus the feature names, ordered as they appear in the file.
    """
    if not path.exists():
        raise FileNotFoundError(f"Explanation file not found: {path}")

    df = pd.read_csv(path)

    importance_cols = [c for c in df.columns if c.startswith("importance__")]
    if not importance_cols:
        raise ValueError(f"No importance__ columns found in {path}")

    feature_names = [c.replace("importance__", "", 1) for c in importance_cols]

    matrix = df[importance_cols].to_numpy(dtype=float)

    return matrix, feature_names


def find_explanation_path(
    explanations_dir: Path,
    retrained_dir: Path,
    model: str,
    window: int,
) -> Path:
    """
    Locate the explanation CSV for a given model-window pair.

    Convention:
        static       -> results/explanations/static_if_exiffi_w{N}_sample.csv
        retrained_wk -> <retrained_dir>/retrained_explanations_{model}_w{window}.csv

    If a retrained explanation file does not exist yet, the script will
    raise. Run the retrained explanation generation first if needed.
    """
    if model == "static":
        return explanations_dir / f"static_if_exiffi_w{window}_sample.csv"
    return retrained_dir / f"retrained_explanations_{model}_w{window}.csv"


# ============================================================
# Divergence metrics on full normalised explanation vectors
# ============================================================

def _normalise_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.clip(matrix, 0.0, None)
    totals = matrix.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    return matrix / totals


def _mean_distribution(matrix: np.ndarray) -> np.ndarray:
    normed = _normalise_rows(matrix)
    dist = normed.mean(axis=0)
    total = dist.sum()
    if total <= 0:
        return np.full(matrix.shape[1], 1.0 / matrix.shape[1])
    return dist / total


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(1.0 - np.dot(a, b) / denom)


def js_divergence(a: np.ndarray, b: np.ndarray) -> float:
    return float(jensenshannon(a, b, base=2.0) ** 2)


def l1_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).sum())


def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def rank_correlation(a: np.ndarray, b: np.ndarray) -> float:
    result = spearmanr(a, b)
    stat = result.statistic
    if not np.isfinite(stat):
        return 0.0
    return float(stat)


def compute_divergence(
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
) -> Dict[str, float]:
    """
    Compute divergence between two explanation matrices using
    the mean normalised explanation vector of each.
    """
    mean_a = _mean_distribution(matrix_a)
    mean_b = _mean_distribution(matrix_b)

    return {
        "cosine_distance": cosine_distance(mean_a, mean_b),
        "js_divergence": js_divergence(mean_a, mean_b),
        "l1_distance": l1_distance(mean_a, mean_b),
        "l2_distance": l2_distance(mean_a, mean_b),
        "spearman_rank_correlation": rank_correlation(mean_a, mean_b),
    }


# ============================================================
# Fidelity gap loading
# ============================================================

def load_fidelity_gap(
    cell_summary_path: Path,
) -> pd.DataFrame:
    """
    Load the cell summary and compute, per window, the fidelity gap
    between static and each retrained model at k=PRIMARY_K.
    """
    if not cell_summary_path.exists():
        raise FileNotFoundError(
            f"Cell summary not found: {cell_summary_path}"
        )

    df = pd.read_csv(cell_summary_path)
    df = df[df["top_k"] == PRIMARY_K]

    rows = []

    for window in WINDOWS:
        static = df[(df["model"] == "static") & (df["window"] == window)]
        if static.empty:
            continue
        static_fid = float(static.iloc[0]["mean_relative_score_change"])

        for retrained_model in [f"retrained_w{i}" for i in WINDOWS]:
            r = df[(df["model"] == retrained_model) & (df["window"] == window)]
            if r.empty:
                continue
            retrained_fid = float(r.iloc[0]["mean_relative_score_change"])
            rows.append({
                "window": window,
                "retrained_model": retrained_model,
                "static_fidelity": static_fid,
                "retrained_fidelity": retrained_fid,
                "fidelity_gap": static_fid - retrained_fid,
            })

    return pd.DataFrame(rows)


# ============================================================
# Covariate shift loading
# ============================================================

def load_input_drift(
    covariate_shift_path: Path,
) -> pd.DataFrame:
    """
    Load the per-window covariate shift summary produced in Section 7.1.
    Only the mean JS divergence is used as the input-drift metric.
    """
    if not covariate_shift_path.exists():
        raise FileNotFoundError(
            f"Covariate shift summary not found: {covariate_shift_path}"
        )

    df = pd.read_csv(covariate_shift_path)
    return df[["window", "mean_js_divergence"]].rename(
        columns={"mean_js_divergence": "input_drift_js"}
    )


# ============================================================
# Main analysis
# ============================================================

def run_analysis(
    explanations_dir: Path,
    retrained_dir: Path,
    cell_summary_path: Path,
    covariate_shift_path: Path,
    output_dir: Path,
) -> None:

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CROSS-MODEL EXPLANATION DIVERGENCE ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # Load all explanation matrices
    # --------------------------------------------------------

    print("\n[1/5] Loading explanation matrices...")

    matrices: Dict[Tuple[str, int], np.ndarray] = {}
    feature_names_ref: List[str] = []

    for model in MODELS:
        for window in WINDOWS:
            path = find_explanation_path(
                explanations_dir, retrained_dir, model, window
            )
            matrix, features = load_explanation_matrix(path)

            if not feature_names_ref:
                feature_names_ref = features
            elif features != feature_names_ref:
                raise ValueError(
                    f"Feature schema mismatch: {path} does not match reference."
                )

            matrices[(model, window)] = matrix
            print(f"  Loaded {model} @ W{window}: {matrix.shape}")

    # --------------------------------------------------------
    # Compute pairwise divergence, static vs each retrained, within window
    # --------------------------------------------------------

    print("\n[2/5] Computing pairwise explanation divergence...")

    pairwise_rows = []

    for window in WINDOWS:
        static_matrix = matrices[("static", window)]

        for retrained_model in [f"retrained_w{i}" for i in WINDOWS]:
            retrained_matrix = matrices[(retrained_model, window)]

            div = compute_divergence(static_matrix, retrained_matrix)

            pairwise_rows.append({
                "window": window,
                "retrained_model": retrained_model,
                "is_diagonal": retrained_model == f"retrained_w{window}",
                **div,
            })

    pairwise = pd.DataFrame(pairwise_rows)

    pairwise_path = output_dir / "pairwise_divergence.csv"
    pairwise.to_csv(pairwise_path, index=False)
    print(f"  Saved: {pairwise_path}")

    # --------------------------------------------------------
    # Join fidelity gap and input drift
    # --------------------------------------------------------

    print("\n[3/5] Joining fidelity gap and input drift...")

    fidelity_gap = load_fidelity_gap(cell_summary_path)
    input_drift = load_input_drift(covariate_shift_path)

    enriched = pairwise.merge(
        fidelity_gap, on=["window", "retrained_model"], how="left"
    ).merge(input_drift, on="window", how="left")

    enriched_path = output_dir / "pairwise_divergence_with_gap.csv"
    enriched.to_csv(enriched_path, index=False)
    print(f"  Saved: {enriched_path}")

    # --------------------------------------------------------
    # Per-window summary (diagonal only)
    # --------------------------------------------------------

    print("\n[4/5] Building per-window summary...")

    diagonal = enriched[enriched["is_diagonal"]].copy()

    per_window = diagonal[[
        "window", "retrained_model",
        "cosine_distance", "js_divergence", "l1_distance", "l2_distance",
        "spearman_rank_correlation",
        "fidelity_gap", "input_drift_js",
    ]].sort_values("window").reset_index(drop=True)

    per_window_path = output_dir / "per_window_summary.csv"
    per_window.to_csv(per_window_path, index=False)
    print(f"  Saved: {per_window_path}")

    # --------------------------------------------------------
    # Correlations
    # --------------------------------------------------------

    print("\n[5/5] Computing correlations...")

    correlation_rows = []

    # (a) explanation divergence vs fidelity gap
    for metric in [
        "cosine_distance", "js_divergence",
        "l1_distance", "l2_distance", "spearman_rank_correlation",
    ]:
        x = enriched[metric].to_numpy(dtype=float)
        y = enriched["fidelity_gap"].to_numpy(dtype=float)

        valid = np.isfinite(x) & np.isfinite(y)
        x, y = x[valid], y[valid]

        if len(x) >= 3:
            r_p, p_p = pearsonr(x, y)
            r_s, p_s = spearmanr(x, y)
        else:
            r_p = r_s = p_p = p_s = np.nan

        correlation_rows.append({
            "predictor": metric,
            "outcome": "fidelity_gap",
            "n": int(len(x)),
            "pearson": float(r_p) if np.isfinite(r_p) else np.nan,
            "pearson_p": float(p_p) if np.isfinite(p_p) else np.nan,
            "spearman": float(r_s) if np.isfinite(r_s) else np.nan,
            "spearman_p": float(p_s) if np.isfinite(p_s) else np.nan,
        })

    # (b) input drift vs fidelity gap (at window level)
    x = per_window["input_drift_js"].to_numpy(dtype=float)
    y = per_window["fidelity_gap"].to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() >= 3:
        r_p, p_p = pearsonr(x[valid], y[valid])
        r_s, p_s = spearmanr(x[valid], y[valid])
    else:
        r_p = r_s = p_p = p_s = np.nan

    correlation_rows.append({
        "predictor": "input_drift_js",
        "outcome": "fidelity_gap",
        "n": int(valid.sum()),
        "pearson": float(r_p) if np.isfinite(r_p) else np.nan,
        "pearson_p": float(p_p) if np.isfinite(p_p) else np.nan,
        "spearman": float(r_s) if np.isfinite(r_s) else np.nan,
        "spearman_p": float(p_s) if np.isfinite(p_s) else np.nan,
    })

    # (c) input drift vs explanation divergence (at window level)
    x = per_window["input_drift_js"].to_numpy(dtype=float)
    y = per_window["cosine_distance"].to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() >= 3:
        r_p, p_p = pearsonr(x[valid], y[valid])
        r_s, p_s = spearmanr(x[valid], y[valid])
    else:
        r_p = r_s = p_p = p_s = np.nan

    correlation_rows.append({
        "predictor": "input_drift_js",
        "outcome": "cosine_distance",
        "n": int(valid.sum()),
        "pearson": float(r_p) if np.isfinite(r_p) else np.nan,
        "pearson_p": float(p_p) if np.isfinite(p_p) else np.nan,
        "spearman": float(r_s) if np.isfinite(r_s) else np.nan,
        "spearman_p": float(p_s) if np.isfinite(p_s) else np.nan,
    })

    correlation_df = pd.DataFrame(correlation_rows)
    correlation_path = output_dir / "correlations.csv"
    correlation_df.to_csv(correlation_path, index=False)
    print(f"  Saved: {correlation_path}")

    # --------------------------------------------------------
    # Figure: divergence vs fidelity gap
    # --------------------------------------------------------

    print("\nBuilding figure...")

    fig, ax = plt.subplots(figsize=(7.5, 5))

    colors = {1: "#1f77b4", 2: "#2ca02c", 3: "#ff7f0e", 4: "#d62728"}

    for _, row in per_window.iterrows():
        ax.scatter(
            row["cosine_distance"],
            row["fidelity_gap"],
            s=90,
            color=colors[int(row["window"])],
            label=f"W{int(row['window'])}",
            edgecolor="black",
            zorder=3,
        )
        ax.annotate(
            f"W{int(row['window'])}",
            xy=(row["cosine_distance"], row["fidelity_gap"]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=10,
        )

    ax.set_xlabel("Cosine distance (static vs retrained, same window)", fontsize=12)
    ax.set_ylabel("Fidelity gap (static − retrained, k=5)", fontsize=12)
    ax.set_title("Explanation divergence vs fidelity gap", fontsize=12)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()

    png_path = output_dir / "figure_7_7_1_divergence_vs_fidelity.png"
    pdf_path = output_dir / "figure_7_7_1_divergence_vs_fidelity.pdf"

    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"  Saved: {png_path}")
    print(f"  Saved: {pdf_path}")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    print("\nPer-window summary (diagonal pairs):")
    print(per_window.to_string(index=False))

    print("\nCorrelations:")
    print(correlation_df.to_string(index=False))


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Cross-model explanation divergence and fidelity gap analysis."
    )
    parser.add_argument(
        "--explanations-dir",
        default="results/explanations",
    )
    parser.add_argument(
        "--retrained-dir",
        default="results/retrained",
        help="Directory containing retrained model explanations and summaries",
    )
    parser.add_argument(
        "--cell-summary",
        default="results/retrained/full_grid/full_grid_cell_summary.csv",
    )
    parser.add_argument(
        "--covariate-shift",
        default="results/concept_drift/covariate_shift_summary.csv",
    )
    parser.add_argument(
        "--output-dir",
        default="results/retrained/cross_model_divergence",
    )
    args = parser.parse_args()

    run_analysis(
        explanations_dir=Path(args.explanations_dir),
        retrained_dir=Path(args.retrained_dir),
        cell_summary_path=Path(args.cell_summary),
        covariate_shift_path=Path(args.covariate_shift),
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()