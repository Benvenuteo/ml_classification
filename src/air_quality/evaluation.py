"""Metrics, training-only threshold selection and paired uncertainty estimates."""

from __future__ import annotations

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def aggregate_elevated_probability(probabilities: np.ndarray) -> np.ndarray:
    """Combine Poor and Hazardous probabilities in the canonical class order."""
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("Expected probabilities for Good, Moderate, Poor and Hazardous.")
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("Probabilities must be finite values between zero and one.")
    if not np.allclose(values.sum(axis=1), 1):
        raise ValueError("Each probability row must sum to one.")
    return values[:, 2:].sum(axis=1)


def metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    task: str,
) -> dict:
    """Compute explicit classwise and probability metrics for a frozen prediction."""
    labels = list(range(4 if task == "multiclass" else 2))
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "log_loss": float(log_loss(y_true, probabilities, labels=labels)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
    }
    if task == "multiclass":
        result["quadratic_kappa"] = float(
            cohen_kappa_score(
                y_true,
                y_pred,
                labels=labels,
                weights="quadratic",
            )
        )
        result["ordinal_mae"] = float(np.abs(np.asarray(y_true) - y_pred).mean())
        result["severe_error_rate"] = float((np.abs(np.asarray(y_true) - y_pred) >= 2).mean())
    else:
        positive = np.asarray(probabilities)[:, 1]
        result.update(
            {
                "positive_f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "positive_f2": float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
                "positive_precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "positive_recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "average_precision": float(average_precision_score(y_true, positive)),
                "roc_auc": float(roc_auc_score(y_true, positive)),
                "brier": float(brier_score_loss(y_true, positive)),
            }
        )
    return result


def choose_threshold(y_true: np.ndarray, positive_probabilities: np.ndarray) -> dict:
    """Choose a binary threshold on nested out-of-fold training predictions only.

    F2 gives recall greater weight by a declared modelling preference, without
    implying a validated health cost. Ties use the threshold closest to 0.5.
    """
    y_true = np.asarray(y_true)
    values = np.asarray(positive_probabilities)
    if len(y_true) != len(values) or set(np.unique(y_true)) != {0, 1}:
        raise ValueError("Threshold selection requires aligned binary labels with both classes.")
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("Invalid probabilities for threshold selection.")
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2)
    scores = np.array(
        [fbeta_score(y_true, values >= cutoff, beta=2, zero_division=0) for cutoff in grid]
    )
    candidates = np.flatnonzero(np.isclose(scores, scores.max(), rtol=0, atol=1e-12))
    chosen = candidates[np.argmin(np.abs(grid[candidates] - 0.5))]
    return {
        "threshold": float(grid[chosen]),
        "objective": "training_nested_oof_f2",
        "training_oof_f2": float(scores[chosen]),
        "selection_score_is_optimistic": True,
    }


def _confusion_metrics(y_true: np.ndarray, y_pred: np.ndarray, classes: int) -> np.ndarray:
    """Compute bootstrap metrics efficiently from a single confusion matrix."""
    matrix = np.bincount(classes * y_true + y_pred, minlength=classes**2).reshape(classes, classes)
    actual = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    correct = np.diag(matrix)
    f1 = np.divide(
        2 * correct, actual + predicted, out=np.zeros(classes), where=(actual + predicted) != 0
    )
    recall = np.divide(correct, actual, out=np.zeros(classes), where=actual != 0)
    return np.array([correct.sum() / matrix.sum(), f1.mean(), recall.mean()])


def bootstrap_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    iterations: int,
    seed: int,
    strata: np.ndarray | None = None,
) -> dict:
    """Return percentile intervals conditional on the fixed test class composition.

    The bootstrap resamples test records within the original four-class strata.
    These intervals do not include retraining or domain-shift uncertainty.
    """
    y_true, y_pred = np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)
    strata = y_true if strata is None else np.asarray(strata)
    groups = [np.flatnonzero(strata == label) for label in np.unique(strata)]
    rng = np.random.default_rng(seed)
    classes = len(np.unique(y_true))
    samples = np.empty((iterations, 3))
    for iteration in range(iterations):
        indices = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
        samples[iteration] = _confusion_metrics(y_true[indices], y_pred[indices], classes)
    bounds = np.quantile(samples, [0.025, 0.975], axis=0)
    return {
        name: {"low": float(bounds[0, index]), "high": float(bounds[1, index])}
        for index, name in enumerate(("accuracy", "macro_f1", "balanced_accuracy"))
    }


def paired_binary_comparison(
    y_true: np.ndarray,
    direct: np.ndarray,
    aggregated: np.ndarray,
    *,
    strata: np.ndarray,
    iterations: int,
    seed: int,
) -> dict:
    """Compare two frozen classifiers on the same binary task and test records."""
    y_true = np.asarray(y_true, dtype=int)
    groups = [np.flatnonzero(strata == label) for label in np.unique(strata)]
    rng = np.random.default_rng(seed)
    differences = np.empty(iterations)
    for iteration in range(iterations):
        indices = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups])
        differences[iteration] = (
            _confusion_metrics(y_true[indices], direct[indices], 2)[1]
            - _confusion_metrics(y_true[indices], aggregated[indices], 2)[1]
        )
    direct_only = int(((direct == y_true) & (aggregated != y_true)).sum())
    aggregated_only = int(((direct != y_true) & (aggregated == y_true)).sum())
    discordant = direct_only + aggregated_only
    return {
        "macro_f1_delta_direct_minus_aggregated": float(
            _confusion_metrics(y_true, direct, 2)[1] - _confusion_metrics(y_true, aggregated, 2)[1]
        ),
        "paired_95ci": [float(value) for value in np.quantile(differences, [0.025, 0.975])],
        "mcnemar_exact_p": (
            float(binomtest(direct_only, discordant, p=0.5).pvalue) if discordant else 1.0
        ),
        "direct_only_correct": direct_only,
        "aggregated_only_correct": aggregated_only,
        "interpretation": "McNemar tests paired correctness, not equality of macro-F1.",
    }
