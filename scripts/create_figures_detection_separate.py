# scripts/create_figures_detection_separate.py
"""
Two separate detection figures:
    Figure A — Detection F1 across future windows
    Figure B — Detection PR-AUC across future windows

Each figure is produced as its own PNG and PDF.

Reads:
    results/retrained/full_grid/full_grid_detection_trajectories.csv

Writes:
    results/figures/figure_detection_f1.png
    results/figures/figure_detection_f1.pdf
    results/figures/figure_detection_pr_auc.png
    results/figures/figure_detection_pr_auc.pdf
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

MODEL_STYLE = {
    "static":       {"color": "#000000", "linestyle": "--", "marker": "o", "label": "Static"},
    "retrained_w1": {"color": "#1f77b4", "linestyle": "-",  "marker": "s", "label": "Retrained W1"},
    "retrained_w2": {"color": "#2ca02c", "linestyle": "-",  "marker": "^", "label": "Retrained W2"},
    "retrained_w3": {"color": "#ff7f0e", "linestyle": "-",  "marker": "D", "label": "Retrained W3"},
    "retrained_w4": {"color": "#d62728", "linestyle": "-",  "marker": "v", "label": "Retrained W4"},
}

MODELS = list(MODEL_STYLE.keys())


def _draw_panel(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    out_png: Path,
    out_pdf: Path,
    dpi: int,
):
    windows = sorted(df["window"].unique())

    fig, ax = plt.subplots(figsize=(8, 5))

    for model in MODELS:
        subset = df[df["model"] == model].sort_values("window")
        if subset.empty:
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


def create_figures(
    input_path: Path,
    output_dir: Path,
    dpi: int = 300,
) -> None:

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

    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Figure 1: Detection F1 ----------------------------------
    _draw_panel(
        df=df,
        metric="f1",
        ylabel="F1",
        title="Detection F1 across future windows",
        out_png=output_dir / "figure_detection_f1.png",
        out_pdf=output_dir / "figure_detection_f1.pdf",
        dpi=dpi,
    )

    # ---- Figure 2: Detection PR-AUC ------------------------------
    _draw_panel(
        df=df,
        metric="pr_auc",
        ylabel="PR-AUC",
        title="Detection PR-AUC across future windows",
        out_png=output_dir / "figure_detection_pr_auc.png",
        out_pdf=output_dir / "figure_detection_pr_auc.pdf",
        dpi=dpi,
    )

    # Console summary
    print("\nF1 values:")
    print(df.pivot(index="window", columns="model", values="f1").to_string())
    print("\nPR-AUC values:")
    print(df.pivot(index="window", columns="model", values="pr_auc").to_string())


def main():
    parser = argparse.ArgumentParser(
        description="Create separate F1 and PR-AUC detection figures."
    )
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    create_figures(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()