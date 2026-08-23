from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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