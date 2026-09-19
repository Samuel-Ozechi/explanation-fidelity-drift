# scripts/create_figure_7_5_1a_fidelity_response_curves.py
"""
Figure 7.5.1a — Fidelity response curves for ExIFFI and random control.

Single-panel plot:
    x-axis: k (intervention size), log scale
    y-axis: mean relative score change
    Two curves:
        - ExIFFI  (solid line)
        - Random  (dashed line)

Reads:
    Inline table (aggregated across windows).
    Optionally reads from a CSV if --input is supplied.

Writes:
    results/figures/figure_7_5_1a_fidelity_response_curves.png
    results/figures/figure_7_5_1a_fidelity_response_curves.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_OUTPUT_DIR = Path("results/figures")
FIGURE_NAME = "figure_7_5_1a_fidelity_response_curves"


# Aggregated values across W1-W4 (from Table 7.5.1)
DEFAULT_DATA = pd.DataFrame({
    "k": [1, 3, 5, 10, 20],
    "exiffi_mean": [0.01071, 0.02674, 0.03970, 0.06370, 0.09464],
    "random_mean": [0.00045, 0.00123, 0.00200, 0.00394, 0.00773],
})


def create_figure(
    data: pd.DataFrame,
    output_dir: Path,
    dpi: int = 300,
) -> None:

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.5, 5))

    ax.plot(
        data["k"],
        data["exiffi_mean"],
        marker="o",
        markersize=9,
        linewidth=2.4,
        color="#1f77b4",
        label="ExIFFI",
    )

    ax.plot(
        data["k"],
        data["random_mean"],
        marker="s",
        markersize=8,
        linewidth=2.0,
        linestyle="--",
        color="#d62728",
        label="Random control",
    )

    for _, row in data.iterrows():
        ax.annotate(
            f"{row['exiffi_mean']:.4f}",
            xy=(row["k"], row["exiffi_mean"]),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color="#1f77b4",
        )
        ax.annotate(
            f"{row['random_mean']:.4f}",
            xy=(row["k"], row["random_mean"]),
            xytext=(0, -14),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color="#d62728",
        )

    ax.set_xscale("log")
    ax.set_xticks(data["k"])
    ax.set_xticklabels([str(k) for k in data["k"]])
    ax.set_xlabel("k (number of features perturbed)", fontsize=12)
    ax.set_ylabel("Mean relative score change", fontsize=12)
    ax.set_title("Fidelity response: ExIFFI vs random control", fontsize=12)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=11)

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
        description="Create Figure 7.5.1a (fidelity response curves)."
    )
    parser.add_argument("--input", default=None,
                        help="Optional CSV with columns: k, exiffi_mean, random_mean")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    if args.input:
        data = pd.read_csv(args.input)
    else:
        data = DEFAULT_DATA

    create_figure(
        data=data,
        output_dir=Path(args.output_dir),
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()