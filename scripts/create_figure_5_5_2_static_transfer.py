# scripts/create_figure_5_5_2_static_transfer.py
"""
Figure 5.5.2 — Static explanation transfer versus current-model baseline.

Three lines across four windows:
    - Baseline        : current model's own explanation vs itself
    - Static transfer : static explanation vs current model
    - Random control  : random top-k features vs current model

Reads:
    results/static_transfer_fidelity/static_transfer_fidelity_comparison.csv
    (or reconstructs the plot from the summary CSV if needed)

Writes:
    results/figures/figure_5_5_2_static_transfer.png
    results/figures/figure_5_5_2_static_transfer.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_COMPARISON = Path(
    "results/static_transfer_fidelity/static_transfer_fidelity_comparison.csv"
)
DEFAULT_OUTPUT_DIR = Path("results/figures")
FIGURE_NAME = "figure_5_5_2_static_transfer"

CONDITION_STYLE = {
    "baseline": {
        "color": "#2ca02c",
        "linestyle": "-",
        "marker": "o",
        "label": "Current model (baseline)",
    },
    "static_transfer": {
        "color": "#1f77b4",
        "linestyle": "-",
        "marker": "s",
        "label": "Static transfer",
    },
    "random_control": {
        "color": "#d62728",
        "linestyle": "--",
        "marker": "^",
        "label": "Random control",
    },
}

CONDITION_ORDER = ["baseline", "static_transfer", "random_control"]


def create_figure(
    comparison_path: Path,
    output_dir: Path,
    dpi: int = 300,
) -> None:

    if not comparison_path.exists():
        raise FileNotFoundError(
            f"Comparison file not found: {comparison_path}\n"
            "Run evaluate_static_explanation_transfer_fidelity.py first."
        )

    df = pd.read_csv(comparison_path)

    required = {"window", *CONDITION_ORDER}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Comparison file is missing required columns: {sorted(missing)}"
        )

    df = df.sort_values("window").reset_index(drop=True)
    windows = df["window"].tolist()

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    for condition in CONDITION_ORDER:
        style = CONDITION_STYLE[condition]
        ax.plot(
            df["window"],
            df[condition],
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=9,
            linewidth=2.4,
            label=style["label"],
        )

    # Annotate the transfer gap at each window
    for _, row in df.iterrows():
        gap = row["baseline"] - row["static_transfer"]
        ax.annotate(
            f"Δ = {gap:.4f}",
            xy=(row["window"], (row["baseline"] + row["static_transfer"]) / 2),
            xytext=(8, 0),
            textcoords="offset points",
            fontsize=9,
            color="#444444",
            va="center",
        )

    ax.set_xlabel("Future window", fontsize=12)
    ax.set_ylabel("Mean relative score change (k=5)", fontsize=12)
    ax.set_title(
        "Static explanation transfer vs current-model baseline",
        fontsize=12,
    )
    ax.set_xticks(windows)
    ax.set_xticklabels([f"W{w}" for w in windows])
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=11, loc="center right")

    plt.tight_layout()

    png_path = output_dir / f"{FIGURE_NAME}.png"
    pdf_path = output_dir / f"{FIGURE_NAME}.pdf"

    plt.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")

    # Console summary
    print("\nFidelity comparison at k=5:")
    print(df.to_string(index=False))

    print("\nTransfer gap to baseline:")
    for _, row in df.iterrows():
        gap = row["baseline"] - row["static_transfer"]
        ratio = row["static_transfer"] / row["baseline"]
        print(
            f"  W{int(row['window'])}: gap = {gap:.4f}, "
            f"transfer/baseline = {ratio:.2f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Create Figure 5.5.2 (static transfer vs baseline)."
    )
    parser.add_argument(
        "--input",
        default=str(DEFAULT_COMPARISON),
        help="Path to the comparison CSV",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
    )
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    create_figure(
        comparison_path=Path(args.input),
        output_dir=Path(args.output_dir),
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()