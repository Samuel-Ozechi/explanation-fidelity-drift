from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from efd.experiments.static_if import run_experiment


def main():
    parser = argparse.ArgumentParser(
        description="Run Baseline 1 static Isolation Forest."
    )

    parser.add_argument(
        "--transactions",
        required=True,
        help="Path to the IEEE-CIS transactions CSV.",
    )

    parser.add_argument(
        "--output",
        default="results/tables/static_if_v1.csv",
        help="Output CSV path for the evaluation table.",
    )

    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.60,
    )

    parser.add_argument(
        "--windows",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--trees",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--contamination",
        default="auto",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--reference-date",
        default="2017-12-01",
        help=(
            "Calendar date corresponding to TransactionDT=0."
        ),
    )

    args = parser.parse_args()

    run_experiment(
        transactions_path=args.transactions,
        output_path=args.output,
        train_fraction=args.train_fraction,
        n_future_windows=args.windows,
        n_estimators=args.trees,
        contamination=args.contamination,
        random_state=args.random_state,
        reference_date=args.reference_date,
    )


if __name__ == "__main__":
    main()
