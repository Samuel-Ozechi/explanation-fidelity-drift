# scripts/create_figure_7_6_1_fidelity_trajectories.py
"""
Figure 7.6.1 — Fidelity trajectories across future windows at k=5.

Single-panel plot:
    x-axis: future window (W1-W4)
    y-axis: mean relative score change
    Five lines, one per model:
        - static       (dashed, black)
        - retrained_w1 (solid)
        - retrained_w2 (solid)
        - retrained_w3 (solid)
        - retrained_w4 (solid)

Reads:
    Inline table (from Table 7.6.1).
    Optionally reads from a CSV if --input is supplied.

Writes:
    results/figures/figure_7_6_1_fidelity_trajectories.png
    results/figures/figure_7_6_1_fidelity_trajectories.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_OUTPUT_DIR = Path("results/figures")
FIGURE_NAME = "figure_7_6_1_fidelity_trajectories"


# Trajectories at k=5, from Table 7.6.1
DEFAULT_DATA = pd.DataFrame({
    "window": [1, 2, 3, 4],
    "static":       [0.0349, 0.0337, 0.0327, 0.0317],
    "retrained_w1": [0.0328, 0.0308, 0.0306, 0.0293],
    "retrained_w2": [0.0335, 0.0292, 0.0308, 0.0304],
    "retrained_w3": [0.0314, 0.0286, 0.0295, 0.0303],
    "retrained_w4": [0.0306, 0.0296, 0.0290, 0.0289],
})


MODEL_STYLE = {
    "static":       {"color": "#000000", "linestyle": "--", "marker": "o", "label": "Static"},
    "retrained_w1": {"color": "#1f77b4", "linestyle": "-",  "marker": "s", "label": "Retrained W1"},
    "retrained_w2": {"color": "#2ca02c", "linestyle": "-",  "marker": "^", "label": "Retrained W2"},
    "retrained_w3": {"color": "#ff7f0e", "linestyle": "-",  "marker": "D", "label": "Retrained W3"},
    "retrained_w4": {"color": "#d62728", "linestyle": "-",  "marker": "v", "label": "Retrained W4"},
}

MODELS = list(MODEL_STYLE.keys())


def create_figure(
    data: pd.DataFrame,
    output_dir: Path,
    dpi: int = 300,
) -> None:

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    for model in MODELS:
        if model not in data.columns:
            continue
        style = MODEL_STYLE[model]
        ax.plot(
            data["window"],
            data[model],
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=8,
            linewidth=2.2,
            label=style["label"],
        )

    ax.set_xlabel("Future window", fontsize=12)
    ax.set_ylabel("Mean relative score change (k=5)", fontsize=12)
    ax.set_title("Fidelity trajectories across future windows", fontsize=12)
    ax.set_xticks(data["window"])
    ax.set_xticklabels([f"W{w}" for w in data["window"]])
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=10, loc="best")

    plt.tight_layout()

    png_path = output_dir / f"{FIGURE_NAME}.png"
    pdf_path = output_dir / f"{FIGURE_NAME}.pdf"

    plt.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create Figure 7.6.1 (fidelity trajectories at k=5)."
    )
    parser.add_argument(
        "--input", default=None,
        help="Optional CSV with columns: window, static, retrained_w1..w4"
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    data = pd.read_csv(args.input) if args.input else DEFAULT_DATA

    create_figure(
        data=data,
        output_dir=Path(args.output_dir),
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()