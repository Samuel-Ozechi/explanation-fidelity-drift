# scripts/create_figure_7_1_2_fraud_rate.py
"""
Figure 7.1.2 — Fraud rate across future windows.

Reads:
    results/concept_drift/concept_drift_results.csv

Writes:
    results/figures/figure_7_1_2_fraud_rate.png
    results/figures/figure_7_1_2_fraud_rate.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_INPUT = Path("results/concept_drift/concept_drift_results.csv")
DEFAULT_OUTPUT_DIR = Path("results/figures")
FIGURE_NAME = "figure_7_1_2_fraud_rate"


def create_figure(
    input_path: Path,
    output_dir: Path,
    dpi: int = 300,
) -> None:
    """Create the line plot of fraud rate across windows."""

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}\n"
            "Run confirm_concept_drift.py first."
        )

    df = pd.read_csv(input_path)

    required = {"window", "fraud_rate"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Input file is missing required columns: {sorted(missing)}"
        )

    df = df.sort_values("window").reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------
    # Figure
    # ---------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(
        df["window"],
        df["fraud_rate"] * 100,
        marker="o",
        markersize=9,
        linewidth=2.2,
        color="#4C72B0",
        label="Fraud rate",
    )

    for _, row in df.iterrows():
        ax.annotate(
            f"{row['fraud_rate'] * 100:.2f}%",
            xy=(row["window"], row["fraud_rate"] * 100),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color="#2A4A7C",
        )

    ax.set_xlabel("Future window", fontsize=12)
    ax.set_ylabel("Fraud rate (%)", fontsize=12)
    ax.set_title(
        "Fraud rate across future windows",
        fontsize=12,
    )
    ax.set_xticks(df["window"])
    ax.set_xticklabels([f"W{w}" for w in df["window"]])
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()

    png_path = output_dir / f"{FIGURE_NAME}.png"
    pdf_path = output_dir / f"{FIGURE_NAME}.pdf"

    plt.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")

    # Console summary
    print("\nFraud rate summary:")
    for _, row in df.iterrows():
        print(f"  W{int(row['window'])}: {row['fraud_rate'] * 100:.4f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Create Figure 7.1.2 (fraud rate across windows)."
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