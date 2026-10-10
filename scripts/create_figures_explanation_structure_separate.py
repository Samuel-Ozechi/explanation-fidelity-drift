# scripts/create_figures_explanation_structure_separate.py
"""
Three separate explanation structure figures:
    Figure A — Explanation concentration (top-5 mass)
    Figure B — Explanation entropy
    Figure C — Explanation diffuseness (effective feature count)

Each figure is produced as its own PNG and PDF.

Reads:
    results/retrained/full_grid/full_grid_cell_summary.csv
    results/retrained/retrained_explanation_summary.csv

Writes:
    results/figures/figure_explanation_concentration.png / .pdf
    results/figures/figure_explanation_entropy.png / .pdf
    results/figures/figure_explanation_diffuseness.png / .pdf
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

PRIMARY_K = 5

MODEL_STYLE = {
    "static":       {"color": "#000000", "linestyle": "--", "marker": "o", "label": "Static"},
    "retrained_w1": {"color": "#1f77b4", "linestyle": "-",  "marker": "s", "label": "Retrained W1"},
    "retrained_w2": {"color": "#2ca02c", "linestyle": "-",  "marker": "^", "label": "Retrained W2"},
    "retrained_w3": {"color": "#ff7f0e", "linestyle": "-",  "marker": "D", "label": "Retrained W3"},
    "retrained_w4": {"color": "#d62728", "linestyle": "-",  "marker": "v", "label": "Retrained W4"},
}

MODELS = list(MODEL_STYLE.keys())


# ============================================================
# Data loading and merge
# ============================================================

def load_and_merge(
    cell_summary_path: Path,
    structure_summary_path: Path,
) -> pd.DataFrame:

    if not cell_summary_path.exists():
        raise FileNotFoundError(
            f"Cell summary not found: {cell_summary_path}"
        )
    if not structure_summary_path.exists():
        raise FileNotFoundError(
            f"Structure summary not found: {structure_summary_path}"
        )

    cell = pd.read_csv(cell_summary_path)
    struct = pd.read_csv(structure_summary_path)

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


# ============================================================
# Panel drawing
# ============================================================

def _draw_panel(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    out_png: Path,
    out_pdf: Path,
    dpi: int,
):

    k5 = df[df["top_k"] == PRIMARY_K].copy()
    windows = sorted(k5["window"].unique())

    fig, ax = plt.subplots(figsize=(8, 5))

    for model in MODELS:
        subset = k5[k5["model"] == model].sort_values("window")
        if subset.empty or metric not in subset.columns:
            continue
        style = MODEL_STYLE[model]
        ax.plot(
            subset["window"],
            subset[metric],
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=8,
            linewidth=2.2,
            label=style["label"],
        )

    ax.set_xlabel("Future window", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.set_xticks(windows)
    ax.set_xticklabels([f"W{w}" for w in windows])
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=10, loc="best")

    plt.tight_layout()

    plt.savefig(out_png, dpi=dpi, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


# ============================================================
# Figure creation
# ============================================================

def create_figures(
    cell_summary_path: Path,
    structure_summary_path: Path,
    output_dir: Path,
    dpi: int = 300,
):

    df = load_and_merge(cell_summary_path, structure_summary_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Figure A: Explanation concentration ---------------------
    _draw_panel(
        df=df,
        metric="mean_top_k_mass",
        ylabel="Top-5 mass",
        title="Explanation concentration across future windows",
        out_png=output_dir / "figure_explanation_concentration.png",
        out_pdf=output_dir / "figure_explanation_concentration.pdf",
        dpi=dpi,
    )

    # ---- Figure B: Explanation entropy ---------------------------
    _draw_panel(
        df=df,
        metric="entropy",
        ylabel="Entropy",
        title="Explanation entropy across future windows",
        out_png=output_dir / "figure_explanation_entropy.png",
        out_pdf=output_dir / "figure_explanation_entropy.pdf",
        dpi=dpi,
    )

    # ---- Figure C: Explanation diffuseness -----------------------
    _draw_panel(
        df=df,
        metric="effective_features",
        ylabel="Effective feature count",
        title="Explanation diffuseness across future windows",
        out_png=output_dir / "figure_explanation_diffuseness.png",
        out_pdf=output_dir / "figure_explanation_diffuseness.pdf",
        dpi=dpi,
    )

    # Console summary
    k5 = df[df["top_k"] == PRIMARY_K]
    print("\nTop-5 mass:")
    print(k5.pivot(index="window", columns="model",
                   values="mean_top_k_mass").to_string())
    print("\nEntropy:")
    print(k5.pivot(index="window", columns="model",
                   values="entropy").to_string())
    print("\nEffective features:")
    print(k5.pivot(index="window", columns="model",
                   values="effective_features").to_string())


def main():
    parser = argparse.ArgumentParser(
        description="Create separate explanation structure figures."
    )
    parser.add_argument("--cell-summary", default=str(DEFAULT_CELL_SUMMARY))
    parser.add_argument("--structure-summary", default=str(DEFAULT_STRUCTURE_SUMMARY))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    create_figures(
        cell_summary_path=Path(args.cell_summary),
        structure_summary_path=Path(args.structure_summary),
        output_dir=Path(args.output_dir),
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()