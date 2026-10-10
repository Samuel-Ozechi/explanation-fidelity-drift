# scripts/evaluate_static_explanation_transfer_fidelity.py
"""
Static explanation transfer fidelity.

Corrected fidelity protocol:
    For each window W, we take the ExIFFI explanations produced by the
    STATIC model on the observations in W, and we evaluate those
    explanations against the CURRENT model's decisions.

    Reference model: cumulative_wN — trained on train + W1 + ... + WN.

Conditions per window:
    - baseline         : cumulative_wN's own explanation vs cumulative_wN
    - static_transfer  : static model's explanation vs cumulative_wN
    - random_control   : random top-k features vs cumulative_wN

Procedure (identical across conditions):
    Replace the top-k features identified by the explanation with
    training-period medians, keep all remaining features, re-score
    with cumulative_wN, and measure the relative change.

Outputs:
    results/static_transfer_fidelity/
        static_transfer_fidelity_observation.csv
        static_transfer_fidelity_summary.csv
        static_transfer_fidelity_comparison.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from efd.data.preprocessing import (
    FeaturePolicy,
    NumericTransactionPreprocessor,
)
from efd.data.temporal import chronological_windows
from efd.models.isolation_forest import IsolationForestDetector
from efd.explanations.exiffi import ExIFFI
from efd.evaluation.explanation_fidelity import evaluate_fidelity_matrix


DEFAULT_TRAIN_FRACTION = 0.60
DEFAULT_N_ESTIMATORS = 300
DEFAULT_RANDOM_STATE = 42
DEFAULT_SAMPLE_SIZE = 100
DEFAULT_MAX_MISSING_RATE = 0.95
DEFAULT_K_VALUES = [1, 3, 5, 10, 20]
PRIMARY_K = 5


# ============================================================
# Helpers
# ============================================================

class ExplanationAdapter:
    """Minimal interface for evaluate_local_fidelity."""
    def __init__(self, sample_index: int, normalized: np.ndarray):
        self.sample_index = int(sample_index)
        self.normalized = np.asarray(normalized, dtype=float)


def stratified_sample_indices(
    labels: pd.Series,
    sample_size: int,
    random_state: int,
) -> np.ndarray:
    """Reproduce the sampling protocol used for static explanations."""
    values = labels.to_numpy()
    classes = np.unique(values)
    if len(classes) < 2:
        raise ValueError("Stratified sampling requires at least two classes.")

    rng = np.random.default_rng(random_state)
    base_per_class, remainder = divmod(sample_size, len(classes))
    selected = []

    for class_position, class_value in enumerate(classes):
        class_indices = np.flatnonzero(values == class_value)
        target_n = base_per_class + (class_position < remainder)
        selected.extend(
            rng.choice(class_indices, size=min(target_n, len(class_indices)), replace=False)
        )

    selected_set = set(selected)
    remaining_n = sample_size - len(selected)
    if remaining_n:
        available = np.array(
            [index for index in range(len(values)) if index not in selected_set],
            dtype=int,
        )
        selected.extend(rng.choice(available, size=remaining_n, replace=False))

    selected = np.asarray(selected, dtype=int)
    rng.shuffle(selected)
    return selected


def reconstruct_static_sample(
    window: pd.DataFrame,
    transformed_window: pd.DataFrame,
    static_explanation_df: pd.DataFrame,
    window_number: int,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Reconstruct the exact 100 observations used to generate the static
    explanations on window W, and return:
        X_sample : (n, d)
        y_sample : (n,)
        static_importance : (n, d)  -- static explanations
        static_sample_index : (n,)  -- for pairing
    """
    n = len(static_explanation_df)

    importance_cols = [
        c for c in static_explanation_df.columns
        if c.startswith("importance__")
    ]
    feature_names_from_file = [
        c.replace("importance__", "", 1) for c in importance_cols
    ]

    importance = static_explanation_df[importance_cols].to_numpy(dtype=float)
    y_sample = static_explanation_df["isFraud"].to_numpy(dtype=int)
    sample_index = static_explanation_df["sample_index"].to_numpy(dtype=int)

    # Reproduce the stratified sampling used during explanation generation.
    indices = stratified_sample_indices(
        labels=window["isFraud"],
        sample_size=n,
        random_state=random_state + window_number,
    )

    X_sample = transformed_window.iloc[indices].to_numpy(dtype=float)
    y_from_window = window["isFraud"].iloc[indices].to_numpy(dtype=int)

    # Sanity: labels must match
    if not np.array_equal(y_sample, y_from_window):
        raise ValueError(
            f"Window {window_number}: reconstructed labels do not match "
            "static explanation file. Sampling protocol mismatch."
        )

    return X_sample, y_sample, importance, sample_index


def build_explanation_adapter_list(
    importance: np.ndarray,
    sample_index: np.ndarray,
) -> List[ExplanationAdapter]:
    return [
        ExplanationAdapter(sample_index=int(sample_index[i]),
                           normalized=importance[i])
        for i in range(len(importance))
    ]


def evaluate_condition(
    model,
    X_sample: np.ndarray,
    explanations,
    baseline: np.ndarray,
    k_values: List[int],
    condition: str,
    window: int,
) -> pd.DataFrame:
    """
    Run the fidelity intervention for one condition and return
    observation-level rows.
    """
    results = evaluate_fidelity_matrix(
        model=model,
        X=X_sample,
        explanations=explanations,
        baseline=baseline,
        k_values=k_values,
    )
    rows = [
        {
            "condition": condition,
            "window": window,
            "sample_index": r.sample_index,
            "top_k": r.top_k,
            "top_k_mass": r.top_k_mass,
            "relative_score_change": r.relative_score_change,
            "score_change": r.score_change,
        }
        for r in results
    ]
    return pd.DataFrame(rows)


# ============================================================
# Main
# ============================================================

def run(
    transactions_path: str,
    static_explanations_dir: str,
    output_dir: str,
    train_fraction: float,
    n_estimators: int,
    random_state: int,
    sample_size: int,
    max_missing_rate: float,
    k_values: List[int],
    random_control_repeats: int = 50,
):

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("STATIC EXPLANATION TRANSFER FIDELITY")
    print("=" * 70)

    # --------------------------------------------------------
    # Data and split
    # --------------------------------------------------------

    print("\n[1/6] Loading transactions...")
    df = pd.read_csv(transactions_path, low_memory=False)
    print(f"  Loaded {len(df):,}")

    print("\n[2/6] Chronological split...")
    train, future_windows = chronological_windows(
        df,
        time_col="TransactionDT",
        train_fraction=train_fraction,
        n_future_windows=4,
    )
    for i, w in enumerate(future_windows, 1):
        print(f"  W{i}: {len(w):,}")

    # --------------------------------------------------------
    # Preprocessing (training-only schema)
    # --------------------------------------------------------

    print("\n[3/6] Fitting preprocessing on training period...")
    preprocessor = NumericTransactionPreprocessor(
        policy=FeaturePolicy(
            target="isFraud",
            temporal="TransactionDT",
            identifier="TransactionID",
            max_missing_rate=max_missing_rate,
        )
    )
    X_train = preprocessor.fit_transform(train)
    feature_names = preprocessor.feature_names_
    print(f"  Features: {len(feature_names)}")

    X_future = [preprocessor.transform(w) for w in future_windows]
    y_future = [w["isFraud"].to_numpy() for w in future_windows]

    baseline_median = np.median(
        X_train.to_numpy(dtype=float), axis=0
    )

    # --------------------------------------------------------
    # Train cumulative models per window
    # --------------------------------------------------------

    print("\n[4/6] Training cumulative models...")
    cumulative_models: Dict[int, IsolationForestDetector] = {}

    for upto in range(1, 5):
        combined = pd.concat(
            [train] + future_windows[:upto], ignore_index=True
        )
        X_combined = preprocessor.transform(combined)
        model = IsolationForestDetector(
            n_estimators=n_estimators,
            max_samples="auto",
            contamination="auto",
            random_state=random_state + upto,
            n_jobs=-1,
        )
        model.fit(X_combined)
        cumulative_models[upto] = model
        print(f"  cumulative_w{upto}: trained on {len(combined):,} observations")

    # --------------------------------------------------------
    # Load static explanations and evaluate transfer
    # --------------------------------------------------------

    print("\n[5/6] Evaluating static-transferred explanations "
          "against current models...")

    static_dir = Path(static_explanations_dir)
    all_rows = []

    for window_number in range(1, 5):

        print(f"\n  Window {window_number}")

        static_file = static_dir / f"static_if_exiffi_w{window_number}_sample.csv"
        if not static_file.exists():
            raise FileNotFoundError(f"Missing static explanations: {static_file}")

        static_df = pd.read_csv(static_file)

        X_sample, y_sample, static_importance, static_idx = reconstruct_static_sample(
            window=future_windows[window_number - 1],
            transformed_window=X_future[window_number - 1],
            static_explanation_df=static_df,
            window_number=window_number,
            random_state=random_state,
        )
        print(f"    Observations: {len(X_sample)}")
        print(f"    Static explanations matched to observations.")

        current_model = cumulative_models[window_number]

        # ------------------------------------------------
        # Condition A: baseline — current model's own explanation
        # ------------------------------------------------
        current_explainer = ExIFFI(
            model=current_model.model, feature_names=feature_names
        )
        current_explanations = current_explainer.explain_batch(X_sample)

        df_baseline = evaluate_condition(
            model=current_model.model,
            X_sample=X_sample,
            explanations=current_explanations,
            baseline=baseline_median,
            k_values=k_values,
            condition="baseline",
            window=window_number,
        )

        # ------------------------------------------------
        # Condition B: static_transfer — static explanation vs current model
        # ------------------------------------------------
        static_adapters = build_explanation_adapter_list(
            static_importance, static_idx
        )
        df_static = evaluate_condition(
            model=current_model.model,
            X_sample=X_sample,
            explanations=static_adapters,
            baseline=baseline_median,
            k_values=k_values,
            condition="static_transfer",
            window=window_number,
        )

        # ------------------------------------------------
        # Condition C: random control — random top-k vs current model
        # ------------------------------------------------
        rng = np.random.default_rng(random_state + window_number)
        n_features = X_sample.shape[1]
        random_rows = []

        for i in range(len(X_sample)):
            x = X_sample[i]
            base_score = float(
                current_model.model.score_samples(
                    pd.DataFrame([x], columns=feature_names)
                )[0]
            )
            for k in k_values:
                for rep in range(random_control_repeats):
                    idx = rng.choice(n_features, size=k, replace=False)
                    x_p = x.copy()
                    x_p[idx] = baseline_median[idx]
                    pert_score = float(
                        current_model.model.score_samples(
                            pd.DataFrame([x_p], columns=feature_names)
                        )[0]
                    )
                    change = pert_score - base_score
                    rel = abs(change) / max(abs(base_score), 1e-12)
                    random_rows.append({
                        "condition": "random_control",
                        "window": window_number,
                        "sample_index": i,
                        "top_k": k,
                        "top_k_mass": np.nan,
                        "relative_score_change": rel,
                        "score_change": change,
                    })

        df_random = pd.DataFrame(random_rows)

        all_rows.extend([df_baseline, df_static, df_random])

    observation_df = pd.concat(all_rows, ignore_index=True)

    # --------------------------------------------------------
    # Save and summarise
    # --------------------------------------------------------

    print("\n[6/6] Summarising...")

    observation_df.to_csv(
        output_path / "static_transfer_fidelity_observation.csv", index=False
    )

    summary = (
        observation_df
        .groupby(["condition", "window", "top_k"], as_index=False)
        .agg(
            n_observations=("sample_index", "nunique"),
            mean_relative_score_change=("relative_score_change", "mean"),
            median_relative_score_change=("relative_score_change", "median"),
            std_relative_score_change=("relative_score_change", "std"),
            mean_top_k_mass=("top_k_mass", "mean"),
        )
    )
    summary.to_csv(
        output_path / "static_transfer_fidelity_summary.csv", index=False
    )

    # Wide comparison at primary k
    k5 = summary[summary["top_k"] == PRIMARY_K].pivot(
        index="window", columns="condition",
        values="mean_relative_score_change"
    ).reset_index()
    k5.to_csv(
        output_path / "static_transfer_fidelity_comparison.csv", index=False
    )

    print("\nComparison at k=5:")
    print(k5.to_string(index=False))
    print(f"\nSaved to: {output_path}")

    return observation_df, summary, k5


def main():
    parser = argparse.ArgumentParser(
        description="Static explanation transfer fidelity vs current models."
    )
    parser.add_argument("--transactions", required=True)
    parser.add_argument(
        "--static-explanations-dir",
        default="results/explanations",
    )
    parser.add_argument(
        "--output-dir",
        default="results/static_transfer_fidelity",
    )
    parser.add_argument("--train-fraction", type=float, default=DEFAULT_TRAIN_FRACTION)
    parser.add_argument("--estimators", type=int, default=DEFAULT_N_ESTIMATORS)
    parser.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--max-missing-rate", type=float, default=DEFAULT_MAX_MISSING_RATE)
    parser.add_argument("--k-values", type=int, nargs="+", default=DEFAULT_K_VALUES)
    parser.add_argument("--random-repeats", type=int, default=50)
    args = parser.parse_args()

    run(
        transactions_path=args.transactions,
        static_explanations_dir=args.static_explanations_dir,
        output_dir=args.output_dir,
        train_fraction=args.train_fraction,
        n_estimators=args.estimators,
        random_state=args.random_state,
        sample_size=args.sample_size,
        max_missing_rate=args.max_missing_rate,
        k_values=args.k_values,
        random_control_repeats=args.random_repeats,
    )


if __name__ == "__main__":
    main()