# scripts/create_figure_7_4_1_explanation_structure.py
"""
Figure 7.4.1 — Explanation structure trajectories across future windows.

Three-panel plot:
    Panel (a):  Top-5 mass vs window
    Panel (b):  Entropy vs window
    Panel (c):  Effective feature count vs window

Each panel contains five lines, one per model:
    - static (dashed, black)
    - retrained_w1 (solid)
    - retrained_w2 (solid)
    - retrained_w3 (solid)
    - retrained_w4 (solid)

Inputs:
    results/retrained/full_grid/full_grid_cell_summary.csv
    results/retrained/retrained_explanation_summary.csv

The cell summary provides top-k mass per (model, window, top_k).
The structure summary provides entropy and effective_features per
(model, window). They are merged on (model, window) before plotting.

Writes:
    results/figures/figure_7_4_1_explanation_structure.png
    results/figures/figure_7_4_1_explanation_structure.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_CELL_SUMMARY = Path(
    "results/retrained/full_grid/full_grid_cell_summary.csv"
)
DEFAULT_STRUCTURE_SUMMARY = Path(
    "results/retrained/retrained_explanation_summary.csv"
)
DEFAULT_OUTPUT_DIR = Path("results/figures")
FIGURE_NAME = "figure_7_4_1_explanation_structure"

MODEL_STYLE = {
    "static":       {"color": "#000000", "linestyle": "--", "marker": "o", "label": "Static"},
    "retrained_w1": {"color": "#1f77b4", "linestyle": "-",  "marker": "s", "label": "Retrained W1"},
    "retrained_w2": {"color": "#2ca02c", "linestyle": "-",  "marker": "^", "label": "Retrained W2"},
    "retrained_w3": {"color": "#ff7f0e", "linestyle": "-",  "marker": "D", "label": "Retrained W3"},
    "retrained_w4": {"color": "#d62728", "linestyle": "-",  "marker": "v", "label": "Retrained W4"},
}

MODELS = list(MODEL_STYLE.keys())
PRIMARY_K = 5


def load_and_merge(
    cell_summary_path: Path,
    structure_summary_path: Path,
) -> pd.DataFrame:
    """
    Merge top-k mass (per model, window, k) with entropy and
    effective_features (per model, window) from the structure summary.
    """
    if not cell_summary_path.exists():
        raise FileNotFoundError(
            f"Cell summary not found: {cell_summary_path}\n"
            "Run analyze_full_grid_fidelity.py first."
        )
    if not structure_summary_path.exists():
        raise FileNotFoundError(
            f"Structure summary not found: {structure_summary_path}\n"
            "Run run_retrained_if.py first."
        )

    cell = pd.read_csv(cell_summary_path)
    struct = pd.read_csv(structure_summary_path)

    # The structure summary uses "window" for window index.
    # Keep only the columns we need for the merge.
    struct_cols = ["model", "window", "entropy", "effective_features"]
    missing = [c for c in struct_cols if c not in struct.columns]
    if missing:
        raise ValueError(
            f"Structure summary missing required columns: {missing}"
        )

    merged = cell.merge(
        struct[struct_cols],
        on=["model", "window"],
        how="left",
        validate="many_to_one",
    )

    return merged


def create_figure(
    cell_summary_path: Path,
    structure_summary_path: Path,
    output_dir: Path,
    dpi: int = 300,
) -> None:

    df = load_and_merge(cell_summary_path, structure_summary_path)

    k5 = df[df["top_k"] == PRIMARY_K].copy()
    windows = sorted(k5["window"].unique())

    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------
    # Figure with three panels
    # ---------------------------------------------------------------

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    panels = [
        ("mean_top_k_mass", "Top-5 mass", "(a) Explanation concentration"),
        ("entropy", "Entropy", "(b) Explanation entropy"),
        ("effective_features", "Effective feature count", "(c) Explanation diffuseness"),
    ]

    for ax, (col, ylabel, title) in zip(axes, panels):

        for model in MODELS:
            subset = k5[k5["model"] == model].sort_values("window")
            if subset.empty or col not in subset.columns:
                continue
            style = MODEL_STYLE[model]
            ax.plot(
                subset["window"],
                subset[col],
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markersize=7,
                linewidth=2.0,
                label=style["label"],
            )

        ax.set_xlabel("Future window", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=12)
        ax.set_xticks(windows)
        ax.set_xticklabels([f"W{w}" for w in windows])
        ax.grid(alpha=0.3)
        ax.set_axisbelow(True)

    # ---- Shared legend -----------------------------------------
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, -0.02),
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    png_path = output_dir / f"{FIGURE_NAME}.png"
    pdf_path = output_dir / f"{FIGURE_NAME}.pdf"

    plt.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")

    # ---------------------------------------------------------------
    # Console summary
    # ---------------------------------------------------------------

    print("\nTop-5 mass by model and window:")
    print(k5.pivot(index="window", columns="model", values="mean_top_k_mass").to_string())

    print("\nEntropy by model and window (k=5):")
    print(k5.pivot(index="window", columns="model", values="entropy").to_string())

    print("\nEffective features by model and window (k=5):")
    print(k5.pivot(index="window", columns="model", values="effective_features").to_string())


def main():
    parser = argparse.ArgumentParser(
        description="Create Figure 7.4.1 (explanation structure trajectories)."
    )
    parser.add_argument(
        "--cell-summary",
        default=str(DEFAULT_CELL_SUMMARY),
        help="Path to full_grid_cell_summary.csv",
    )
    parser.add_argument(
        "--structure-summary",
        default=str(DEFAULT_STRUCTURE_SUMMARY),
        help="Path to retrained_explanation_summary.csv",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    create_figure(
        cell_summary_path=Path(args.cell_summary),
        structure_summary_path=Path(args.structure_summary),
        output_dir=Path(args.output_dir),
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()