from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr


# ============================================================
# Configuration
# ============================================================

DEFAULT_INPUT_DIR = Path("results/explanations")
DEFAULT_OUTPUT_DIR = Path("results/tables")

WINDOWS = [1, 2, 3, 4]

TOP_K_VALUES = [1, 5, 10, 20]

DEFAULT_N_BOOTSTRAP = 1000
DEFAULT_N_PERMUTATIONS = 1000
DEFAULT_RANDOM_STATE = 42


# ============================================================
# Loading
# ============================================================


def load_window(
    window: int,
    input_dir: Path,
) -> pd.DataFrame:

    path = (
        input_dir
        / f"static_if_exiffi_w{window}_sample.csv"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Missing temporal explanation file:\n{path}"
        )

    df = pd.read_csv(path)

    importance_cols = [
        c
        for c in df.columns
        if c.startswith("importance__")
    ]

    if not importance_cols:
        raise ValueError(
            f"No importance columns found in {path}"
        )

    print(
        f"Window {window}: "
        f"{len(df):,} observations, "
        f"{len(importance_cols):,} features"
    )

    return df


def extract_importance_matrix(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, List[str]]:

    importance_cols = [
        c
        for c in df.columns
        if c.startswith("importance__")
    ]

    feature_names = [
        c.replace(
            "importance__",
            "",
            1,
        )
        for c in importance_cols
    ]

    X = (
        df[importance_cols]
        .to_numpy(dtype=float)
    )

    return X, feature_names


# ============================================================
# Normalisation
# ============================================================


def normalize_explanations(
    X: np.ndarray,
) -> np.ndarray:

    X = np.abs(X)

    row_sums = X.sum(
        axis=1,
        keepdims=True,
    )

    row_sums[
        row_sums == 0
    ] = 1.0

    return X / row_sums


# ============================================================
# Distribution-level metrics
# ============================================================


def mean_explanation_distribution(
    X: np.ndarray,
) -> np.ndarray:

    X_norm = normalize_explanations(X)

    distribution = X_norm.mean(
        axis=0
    )

    total = distribution.sum()

    if total > 0:
        distribution = (
            distribution / total
        )

    return distribution


def js_divergence(
    p: np.ndarray,
    q: np.ndarray,
) -> float:

    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    p = p / max(p.sum(), 1e-12)
    q = q / max(q.sum(), 1e-12)

    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0

        return np.sum(
            a[mask]
            * np.log2(
                a[mask]
                / np.maximum(
                    b[mask],
                    1e-12,
                )
            )
        )

    return float(
        0.5 * kl(p, m)
        + 0.5 * kl(q, m)
    )


def cosine_distance(
    p: np.ndarray,
    q: np.ndarray,
) -> float:

    if (
        np.linalg.norm(p) == 0
        or np.linalg.norm(q) == 0
    ):
        return 0.0

    return float(
        cosine(p, q)
    )


def l1_distance(
    p: np.ndarray,
    q: np.ndarray,
) -> float:

    return float(
        np.sum(
            np.abs(p - q)
        )
    )


def l2_distance(
    p: np.ndarray,
    q: np.ndarray,
) -> float:

    return float(
        np.linalg.norm(
            p - q
        )
    )


def rank_correlation(
    p: np.ndarray,
    q: np.ndarray,
) -> float:

    result = spearmanr(
        p,
        q,
    )

    value = result.statistic

    if np.isnan(value):
        return 0.0

    return float(value)


# ============================================================
# Top-K
# ============================================================


def top_k_features(
    distribution: np.ndarray,
    k: int,
) -> set:

    k = min(
        k,
        len(distribution),
    )

    indices = np.argsort(
        distribution
    )[::-1][:k]

    return set(indices.tolist())


def jaccard_similarity(
    a: set,
    b: set,
) -> float:

    union = a | b

    if not union:
        return 1.0

    return len(a & b) / len(union)


# ============================================================
# Core drift calculation
# ============================================================


def calculate_drift(
    X_a: np.ndarray,
    X_b: np.ndarray,
) -> Dict[str, float]:

    p = mean_explanation_distribution(
        X_a
    )

    q = mean_explanation_distribution(
        X_b
    )

    return {
        "js_divergence": js_divergence(
            p,
            q,
        ),
        "cosine_distance": cosine_distance(
            p,
            q,
        ),
        "l1_distance": l1_distance(
            p,
            q,
        ),
        "l2_distance": l2_distance(
            p,
            q,
        ),
        "spearman_rank_correlation":
            rank_correlation(
                p,
                q,
            ),
    }


# ============================================================
# Bootstrap
# ============================================================


def bootstrap_drift(
    X_a: np.ndarray,
    X_b: np.ndarray,
    metric: str,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> Tuple[float, float, float]:

    n_a = len(X_a)
    n_b = len(X_b)

    values = []

    for _ in range(n_bootstrap):

        idx_a = rng.integers(
            0,
            n_a,
            size=n_a,
        )

        idx_b = rng.integers(
            0,
            n_b,
            size=n_b,
        )

        result = calculate_drift(
            X_a[idx_a],
            X_b[idx_b],
        )

        values.append(
            result[metric]
        )

    values = np.asarray(values)

    lower = np.percentile(
        values,
        2.5,
    )

    upper = np.percentile(
        values,
        97.5,
    )

    observed = calculate_drift(
        X_a,
        X_b,
    )[metric]

    return (
        float(observed),
        float(lower),
        float(upper),
    )


# ============================================================
# Permutation testing
# ============================================================


def permutation_test(
    X_a: np.ndarray,
    X_b: np.ndarray,
    metric: str,
    n_permutations: int,
    rng: np.random.Generator,
) -> Tuple[float, float]:

    observed = calculate_drift(
        X_a,
        X_b,
    )[metric]

    combined = np.vstack(
        [
            X_a,
            X_b,
        ]
    )

    n_a = len(X_a)

    null_values = []

    for _ in range(n_permutations):

        indices = rng.permutation(
            len(combined)
        )

        perm_a = combined[
            indices[:n_a]
        ]

        perm_b = combined[
            indices[n_a:]
        ]

        value = calculate_drift(
            perm_a,
            perm_b,
        )[metric]

        null_values.append(
            value
        )

    null_values = np.asarray(
        null_values
    )

    # For distances, larger values indicate
    # greater drift.
    #
    # For correlation, smaller values indicate
    # greater drift.
    if metric == "spearman_rank_correlation":

        observed_drift = 1.0 - observed

        null_drift = (
            1.0 - null_values
        )

    else:

        observed_drift = observed
        null_drift = null_values

    p_value = (
        np.sum(
            null_drift
            >= observed_drift
        )
        + 1
    ) / (
        len(null_drift)
        + 1
    )

    return (
        float(observed),
        float(p_value),
    )


# ============================================================
# One comparison
# ============================================================


def evaluate_comparison(
    X_a: np.ndarray,
    X_b: np.ndarray,
    window_a: int,
    window_b: int,
    condition: str,
    n_bootstrap: int,
    n_permutations: int,
    rng: np.random.Generator,
) -> List[Dict]:

    metrics = [
        "js_divergence",
        "cosine_distance",
        "l1_distance",
        "l2_distance",
        "spearman_rank_correlation",
    ]

    rows = []

    for metric in metrics:

        observed, ci_lower, ci_upper = (
            bootstrap_drift(
                X_a=X_a,
                X_b=X_b,
                metric=metric,
                n_bootstrap=n_bootstrap,
                rng=rng,
            )
        )

        _, p_value = permutation_test(
            X_a=X_a,
            X_b=X_b,
            metric=metric,
            n_permutations=n_permutations,
            rng=rng,
        )

        rows.append(
            {
                "comparison": (
                    f"w{window_a}_w{window_b}"
                ),
                "window_a": window_a,
                "window_b": window_b,
                "condition": condition,
                "metric": metric,
                "observed_value": observed,
                "bootstrap_ci_lower": ci_lower,
                "bootstrap_ci_upper": ci_upper,
                "permutation_pvalue": p_value,
                "n_observations_a": len(X_a),
                "n_observations_b": len(X_b),
                "n_features": X_a.shape[1],
                "n_bootstrap": n_bootstrap,
                "n_permutations": n_permutations,
            }
        )

    return rows


# ============================================================
# Top-K significance
# ============================================================


def evaluate_top_k(
    X_a: np.ndarray,
    X_b: np.ndarray,
    window_a: int,
    window_b: int,
    n_permutations: int,
    rng: np.random.Generator,
) -> List[Dict]:

    p = mean_explanation_distribution(
        X_a
    )

    q = mean_explanation_distribution(
        X_b
    )

    rows = []

    for k in TOP_K_VALUES:

        observed = jaccard_similarity(
            top_k_features(p, k),
            top_k_features(q, k),
        )

        combined = np.vstack(
            [
                X_a,
                X_b,
            ]
        )

        n_a = len(X_a)

        null_values = []

        for _ in range(
            n_permutations
        ):

            indices = rng.permutation(
                len(combined)
            )

            perm_a = combined[
                indices[:n_a]
            ]

            perm_b = combined[
                indices[n_a:]
            ]

            p_perm = (
                mean_explanation_distribution(
                    perm_a
                )
            )

            q_perm = (
                mean_explanation_distribution(
                    perm_b
                )
            )

            value = jaccard_similarity(
                top_k_features(
                    p_perm,
                    k,
                ),
                top_k_features(
                    q_perm,
                    k,
                ),
            )

            null_values.append(
                value
            )

        null_values = np.asarray(
            null_values
        )

        # Lower Jaccard means greater instability.
        observed_instability = (
            1.0 - observed
        )

        null_instability = (
            1.0 - null_values
        )

        p_value = (
            np.sum(
                null_instability
                >= observed_instability
            )
            + 1
        ) / (
            len(null_instability)
            + 1
        )

        rows.append(
            {
                "comparison": (
                    f"w{window_a}_w{window_b}"
                ),
                "window_a": window_a,
                "window_b": window_b,
                "top_k": k,
                "observed_jaccard": observed,
                "observed_instability": (
                    observed_instability
                ),
                "permutation_pvalue": (
                    float(p_value)
                ),
                "n_observations_a": len(X_a),
                "n_observations_b": len(X_b),
                "n_features": X_a.shape[1],
                "n_permutations": (
                    n_permutations
                ),
            }
        )

    return rows


# ============================================================
# Class-specific evaluation
# ============================================================


def evaluate_class_comparisons(
    windows: Dict[int, pd.DataFrame],
    matrices: Dict[int, np.ndarray],
    n_bootstrap: int,
    n_permutations: int,
    rng: np.random.Generator,
) -> List[Dict]:

    rows = []

    for window_a, window_b in [
        (1, 2),
        (2, 3),
        (3, 4),
    ]:

        for fraud_value, class_name in [
            (1, "fraud"),
            (0, "nonfraud"),
        ]:

            df_a = windows[
                window_a
            ]

            df_b = windows[
                window_b
            ]

            mask_a = (
                df_a["isFraud"]
                .to_numpy()
                == fraud_value
            )

            mask_b = (
                df_b["isFraud"]
                .to_numpy()
                == fraud_value
            )

            X_a = matrices[
                window_a
            ][mask_a]

            X_b = matrices[
                window_b
            ][mask_b]

            if (
                len(X_a) == 0
                or len(X_b) == 0
            ):
                continue

            class_rows = evaluate_comparison(
                X_a=X_a,
                X_b=X_b,
                window_a=window_a,
                window_b=window_b,
                condition=class_name,
                n_bootstrap=n_bootstrap,
                n_permutations=n_permutations,
                rng=rng,
            )

            rows.extend(
                class_rows
            )

    return rows


# ============================================================
# Main experiment
# ============================================================


def run_experiment(
    input_dir: Path = DEFAULT_INPUT_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    n_permutations: int = DEFAULT_N_PERMUTATIONS,
    random_state: int = DEFAULT_RANDOM_STATE,
):

    print("=" * 70)
    print(
        "BASELINE 1 — TEMPORAL EXIFFI DRIFT "
        "STATISTICAL VALIDATION"
    )
    print("=" * 70)

    print(
        f"\nBootstrap samples: {n_bootstrap}"
    )

    print(
        f"Permutation samples: "
        f"{n_permutations}"
    )

    print(
        f"Random state: {random_state}"
    )

    rng = np.random.default_rng(
        random_state
    )

    # ---------------------------------------------------------
    # 1. Load
    # ---------------------------------------------------------

    print(
        "\n[1/6] Loading temporal explanations..."
    )

    windows = {}
    matrices = {}
    feature_names = {}

    for window in WINDOWS:

        df = load_window(
            window,
            input_dir,
        )

        X, names = (
            extract_importance_matrix(
                df
            )
        )

        windows[window] = df
        matrices[window] = X
        feature_names[window] = names

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    reference_features = (
        feature_names[1]
    )

    for window in WINDOWS[1:]:

        if (
            feature_names[window]
            != reference_features
        ):
            raise ValueError(
                "Feature sets differ across "
                f"windows 1 and {window}."
            )

    # ---------------------------------------------------------
    # 2. Overall drift
    # ---------------------------------------------------------

    print(
        "\n[2/6] Evaluating overall temporal drift..."
    )

    overall_rows = []

    for window_a, window_b in [
        (1, 2),
        (2, 3),
        (3, 4),
    ]:

        print(
            f"\n  Window {window_a} → "
            f"Window {window_b}"
        )

        rows = evaluate_comparison(
            X_a=matrices[window_a],
            X_b=matrices[window_b],
            window_a=window_a,
            window_b=window_b,
            condition="overall",
            n_bootstrap=n_bootstrap,
            n_permutations=n_permutations,
            rng=rng,
        )

        overall_rows.extend(
            rows
        )

        for row in rows:

            print(
                f"    {row['metric']}: "
                f"{row['observed_value']:.6f} "
                f"["
                f"{row['bootstrap_ci_lower']:.6f}, "
                f"{row['bootstrap_ci_upper']:.6f}"
                f"] "
                f"p="
                f"{row['permutation_pvalue']:.6g}"
            )

    overall_df = pd.DataFrame(
        overall_rows
    )

    # ---------------------------------------------------------
    # 3. Top-K
    # ---------------------------------------------------------

    print(
        "\n[3/6] Evaluating Top-K stability..."
    )

    top_k_rows = []

    for window_a, window_b in [
        (1, 2),
        (2, 3),
        (3, 4),
    ]:

        rows = evaluate_top_k(
            X_a=matrices[window_a],
            X_b=matrices[window_b],
            window_a=window_a,
            window_b=window_b,
            n_permutations=n_permutations,
            rng=rng,
        )

        top_k_rows.extend(
            rows
        )

    top_k_df = pd.DataFrame(
        top_k_rows
    )

    # ---------------------------------------------------------
    # 4. Class-specific
    # ---------------------------------------------------------

    print(
        "\n[4/6] Evaluating class-specific drift..."
    )

    class_rows = evaluate_class_comparisons(
        windows=windows,
        matrices=matrices,
        n_bootstrap=n_bootstrap,
        n_permutations=n_permutations,
        rng=rng,
    )

    class_df = pd.DataFrame(
        class_rows
    )

    # ---------------------------------------------------------
    # 5. Save
    # ---------------------------------------------------------

    print(
        "\n[5/6] Saving statistical results..."
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    overall_path = (
        output_dir
        / "static_if_exiffi_temporal_drift_significance.csv"
    )

    top_k_path = (
        output_dir
        / "static_if_exiffi_temporal_top_k_significance.csv"
    )

    class_path = (
        output_dir
        / "static_if_exiffi_temporal_class_drift_significance.csv"
    )

    overall_df.to_csv(
        overall_path,
        index=False,
    )

    top_k_df.to_csv(
        top_k_path,
        index=False,
    )

    class_df.to_csv(
        class_path,
        index=False,
    )

    print(
        f"Overall results:\n{overall_path}"
    )

    print(
        f"Top-K results:\n{top_k_path}"
    )

    print(
        f"Class-specific results:\n{class_path}"
    )

    # ---------------------------------------------------------
    # 6. Summary
    # ---------------------------------------------------------

    print(
        "\n[6/6] Statistical validation summary"
    )

    print(
        "\nOverall drift:"
    )

    display_columns = [
        "comparison",
        "metric",
        "observed_value",
        "bootstrap_ci_lower",
        "bootstrap_ci_upper",
        "permutation_pvalue",
    ]

    print(
        overall_df[
            display_columns
        ].to_string(
            index=False
        )
    )

    print(
        "\nTop-K stability:"
    )

    print(
        top_k_df.to_string(
            index=False
        )
    )

    print(
        "\nClass-specific drift:"
    )

    print(
        class_df[
            [
                "comparison",
                "condition",
                "metric",
                "observed_value",
                "bootstrap_ci_lower",
                "bootstrap_ci_upper",
                "permutation_pvalue",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "TEMPORAL EXIFFI DRIFT STATISTICAL "
        "VALIDATION COMPLETE"
    )

    print(
        "=" * 70
    )

    return (
        overall_df,
        top_k_df,
        class_df,
    )


# ============================================================
# CLI
# ============================================================


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Statistically validate temporal "
            "ExIFFI explanation drift."
        )
    )

    parser.add_argument(
        "--input-dir",
        default=str(
            DEFAULT_INPUT_DIR
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=str(
            DEFAULT_OUTPUT_DIR
        ),
    )

    parser.add_argument(
        "--bootstrap",
        type=int,
        default=DEFAULT_N_BOOTSTRAP,
    )

    parser.add_argument(
        "--permutations",
        type=int,
        default=DEFAULT_N_PERMUTATIONS,
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
    )

    args = parser.parse_args()

    run_experiment(
        input_dir=Path(
            args.input_dir
        ),
        output_dir=Path(
            args.output_dir
        ),
        n_bootstrap=args.bootstrap,
        n_permutations=args.permutations,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()