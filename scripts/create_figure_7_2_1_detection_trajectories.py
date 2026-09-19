# scripts/create_figure_7_2_1_detection_trajectories.py
"""
Figure 7.2.1 — Detection trajectories across future windows.

Two-panel plot:
    Left panel:  F1 vs window
    Right panel: PR-AUC vs window

Each panel contains five lines, one per model:
    - static (dashed, black)
    - retrained_w1 (solid)
    - retrained_w2 (solid)
    - retrained_w3 (solid)
    - retrained_w4 (solid)

Reads:
    results/retrained/full_grid/full_grid_detection_trajectories.csv

Writes:
    results/figures/figure_7_2_1_detection_trajectories.png
    results/figures/figure_7_2_1_detection_trajectories.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_INPUT = Path(
    "results/retrained/full_grid/full_grid_detection_trajectories.csv"
)
DEFAULT_OUTPUT_DIR = Path("results/figures")
FIGURE_NAME = "figure_7_2_1_detection_trajectories"

# Consistent colour and style mapping across the paper
MODEL_STYLE = {
    "static":       {"color": "#000000", "linestyle": "--", "marker": "o", "label": "Static"},
    "retrained_w1": {"color": "#1f77b4", "linestyle": "-",  "marker": "s", "label": "Retrained W1"},
    "retrained_w2": {"color": "#2ca02c", "linestyle": "-",  "marker": "^", "label": "Retrained W2"},
    "retrained_w3": {"color": "#ff7f0e", "linestyle": "-",  "marker": "D", "label": "Retrained W3"},
    "retrained_w4": {"color": "#d62728", "linestyle": "-",  "marker": "v", "label": "Retrained W4"},
}


def create_figure(
    input_path: Path,
    output_dir: Path,
    dpi: int = 300,
) -> None:
    """Create the two-panel detection trajectory figure."""

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}\n"
            "Run evaluate_detection_full_grid.py first."
        )

    df = pd.read_csv(input_path)

    required = {"model", "window", "f1", "pr_auc"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Input file is missing required columns: {sorted(missing)}"
        )

    models = list(MODEL_STYLE.keys())
    windows = sorted(df["window"].unique())

    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------
    # Figure with two panels
    # ---------------------------------------------------------------

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    # ---- Panel A: F1 -------------------------------------------
    ax = axes[0]

    for model in models:
        subset = df[df["model"] == model].sort_values("window")
        if subset.empty:
            continue
        style = MODEL_STYLE[model]
        ax.plot(
            subset["window"],
            subset["f1"],
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=7,
            linewidth=2.0,
            label=style["label"],
        )

    ax.set_xlabel("Future window", fontsize=12)
    ax.set_ylabel("F1", fontsize=12)
    ax.set_title("(a) Detection F1 across future windows", fontsize=12)
    ax.set_xticks(windows)
    ax.set_xticklabels([f"W{w}" for w in windows])
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)

    # ---- Panel B: PR-AUC ---------------------------------------
    ax = axes[1]

    for model in models:
        subset = df[df["model"] == model].sort_values("window")
        if subset.empty:
            continue
        style = MODEL_STYLE[model]
        ax.plot(
            subset["window"],
            subset["pr_auc"],
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=7,
            linewidth=2.0,
            label=style["label"],
        )

    ax.set_xlabel("Future window", fontsize=12)
    ax.set_ylabel("PR-AUC", fontsize=12)
    ax.set_title("(b) Detection PR-AUC across future windows", fontsize=12)
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

    print("\nDetection F1 trajectory:")
    pivot_f1 = df.pivot(index="window", columns="model", values="f1")
    print(pivot_f1.to_string())

    print("\nDetection PR-AUC trajectory:")
    pivot_pr = df.pivot(index="window", columns="model", values="pr_auc")
    print(pivot_pr.to_string())


def main():
    parser = argparse.ArgumentParser(
        description="Create Figure 7.2.1 (detection trajectories)."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    create_figure(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()