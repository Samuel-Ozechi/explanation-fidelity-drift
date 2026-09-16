# scripts/create_figure_7_1_1_js_distribution.py
"""
Figure 7.1.1 — Distribution of per-feature Jensen-Shannon divergence
between the training period and each future window.

Reads:
    results/concept_drift/feature_drift_detailed.csv

Writes:
    results/figures/figure_7_1_1_js_distribution.png
    results/figures/figure_7_1_1_js_distribution.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_INPUT = Path("results/concept_drift/feature_drift_detailed.csv")
DEFAULT_OUTPUT_DIR = Path("results/figures")
FIGURE_NAME = "figure_7_1_1_js_distribution"


def create_figure(
    input_path: Path,
    output_dir: Path,
    dpi: int = 300,
) -> None:
    """Create the violin plot of JS divergence per window."""

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}\n"
            "Run detect_concept_drift.py first."
        )

    df = pd.read_csv(input_path)

    required = {"window", "js_divergence"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Input file is missing required columns: {sorted(missing)}"
        )

    # Drop non-finite JS values — these correspond to degenerate features
    # whose divergence is not well-defined. They are reported separately
    # in Table 7.1.2.
    df = df[np.isfinite(df["js_divergence"])].copy()

    windows = sorted(df["window"].unique())
    data = [df[df["window"] == w]["js_divergence"].to_numpy() for w in windows]

    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------
    # Figure
    # ---------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(8, 5))

    parts = ax.violinplot(
        data,
        positions=windows,
        showmeans=False,
        showmedians=True,
        showextrema=False,
        widths=0.7,
    )

    # Style violins
    for body in parts["bodies"]:
        body.set_facecolor("#4C72B0")
        body.set_alpha(0.6)
        body.set_edgecolor("#2A4A7C")

    if "cmedians" in parts:
        parts["cmedians"].set_color("#2A4A7C")
        parts["cmedians"].set_linewidth(1.6)

    # Overlay individual feature points with jitter for visibility
    rng = np.random.default_rng(42)
    for w, values in zip(windows, data):
        jitter = rng.uniform(-0.08, 0.08, size=len(values))
        ax.scatter(
            np.full(len(values), w) + jitter,
            values,
            s=3,
            alpha=0.15,
            color="#2A4A7C",
            zorder=1,
        )

    ax.set_xlabel("Future window", fontsize=12)
    ax.set_ylabel("Jensen-Shannon divergence", fontsize=12)
    ax.set_title(
        "Per-feature JS divergence between training period and each window",
        fontsize=12,
    )
    ax.set_xticks(windows)
    ax.set_xticklabels([f"W{w}" for w in windows])
    ax.grid(axis="y", alpha=0.3)
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
    print("\nJS divergence summary (finite values only):")
    for w, values in zip(windows, data):
        print(
            f"  W{w}: n={len(values)}, "
            f"mean={np.mean(values):.6f}, "
            f"median={np.median(values):.6f}, "
            f"max={np.max(values):.6f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Create Figure 7.1.1 (JS divergence distribution)."
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