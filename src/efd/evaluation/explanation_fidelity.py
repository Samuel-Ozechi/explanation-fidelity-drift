from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class FidelityResult:
    """
    Fidelity evaluation for one observation.

    The score measures how much the model output changes
    when features identified by the explanation are perturbed.
    """

    sample_index: int

    baseline_score: float

    top_k: int

    top_k_mass: float

    perturbed_score: float

    score_change: float

    relative_score_change: float


def perturb_features(
    x: np.ndarray,
    feature_indices: np.ndarray,
    baseline: np.ndarray,
) -> np.ndarray:
    """
    Replace selected features with baseline values.

    Parameters
    ----------
    x:
        One observation.

    feature_indices:
        Features to perturb.

    baseline:
        Baseline feature values.

    Returns
    -------
    Perturbed observation.
    """

    x_perturbed = np.asarray(
        x,
        dtype=float,
    ).copy()

    x_perturbed[
        feature_indices
    ] = baseline[
        feature_indices
    ]

    return x_perturbed


def evaluate_local_fidelity(
    model,
    x: np.ndarray,
    explanation,
    baseline: np.ndarray,
    k: int,
) -> FidelityResult:
    """
    Evaluate whether the top-k ExIFFI features actually
    influence the model anomaly score.

    The original observation is scored first.

    Then the top-k explanation features are replaced
    with baseline values.

    Fidelity is measured as the resulting change
    in model score.
    """

    x = np.asarray(
        x,
        dtype=float,
    )

    baseline = np.asarray(
        baseline,
        dtype=float,
    )

    normalized = explanation.normalized

    ranked_indices = np.argsort(
        -normalized
    )

    top_indices = ranked_indices[:k]

    top_k_mass = float(
        normalized[top_indices].sum()
    )

    baseline_score = float(
        model.score_samples(
            x.reshape(1, -1)
        )[0]
    )

    x_perturbed = perturb_features(
        x=x,
        feature_indices=top_indices,
        baseline=baseline,
    )

    perturbed_score = float(
        model.score_samples(
            x_perturbed.reshape(1, -1)
        )[0]
    )

    score_change = (
        perturbed_score
        - baseline_score
    )

    denominator = max(
        abs(baseline_score),
        1e-12,
    )

    relative_score_change = (
        abs(score_change)
        / denominator
    )

    return FidelityResult(
        sample_index=explanation.sample_index,
        baseline_score=baseline_score,
        top_k=k,
        top_k_mass=top_k_mass,
        perturbed_score=perturbed_score,
        score_change=score_change,
        relative_score_change=relative_score_change,
    )


def evaluate_fidelity_matrix(
    model,
    X: np.ndarray,
    explanations,
    baseline: np.ndarray,
    k_values: list[int],
) -> list[FidelityResult]:
    """
    Evaluate local fidelity across observations and
    multiple explanation sizes.
    """

    X = np.asarray(
        X,
        dtype=float,
    )

    results = []

    for explanation in explanations:

        sample_index = (
            explanation.sample_index
        )

        x = X[sample_index]

        for k in k_values:

            result = evaluate_local_fidelity(
                model=model,
                x=x,
                explanation=explanation,
                baseline=baseline,
                k=k,
            )

            results.append(result)

    return results

def evaluate_random_fidelity_matrix(
    model,
    X: np.ndarray,
    explanations,
    baseline: np.ndarray,
    k_values: list[int],
    random_state: int = 42,
):
    """
    Evaluate model-score change when randomly selected
    features are replaced by the training baseline.

    This provides a control condition for ExIFFI local
    fidelity.

    The perturbation procedure is identical to ExIFFI
    fidelity; only the feature-selection mechanism differs.
    """

    X = np.asarray(
        X,
        dtype=float,
    )

    baseline = np.asarray(
        baseline,
        dtype=float,
    )

    if X.ndim != 2:
        raise ValueError("X must be two-dimensional.")

    if baseline.shape != (X.shape[1],):
        raise ValueError(
            "baseline must contain one value per feature."
        )

    rng = np.random.default_rng(random_state)

    n_features = X.shape[1]

    results = []

    for explanation, x in zip(
        explanations,
        X,
    ):

        baseline_score = float(
            model.score_samples(
                x.reshape(1, -1)
            )[0]
        )

        for k in k_values:

            if not 0 < k <= n_features:
                raise ValueError(
                    f"k must be between 1 and the number "
                    f"of features ({n_features})."
                )

            random_features = rng.choice(
                n_features,
                size=k,
                replace=False,
            )

            perturbed = x.copy()

            perturbed[
                random_features
            ] = baseline[
                random_features
            ]

            perturbed_score = float(
                model.score_samples(
                    perturbed.reshape(1, -1)
                )[0]
            )

            score_change = (
                perturbed_score
                - baseline_score
            )

            denominator = max(
                abs(baseline_score),
                1e-12,
            )

            relative_score_change = (
                abs(score_change)
                / denominator
            )

            results.append(
                {
                    "sample_index": (
                        explanation.sample_index
                    ),
                    "top_k": k,
                    "baseline_score": (
                        baseline_score
                    ),
                    "perturbed_score": (
                        perturbed_score
                    ),
                    "score_change": (
                        score_change
                    ),
                    "relative_score_change": (
                        relative_score_change
                    ),
                }
            )

    return pd.DataFrame(results)
