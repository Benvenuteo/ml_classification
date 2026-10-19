"""Probability grouping, decision thresholds and paired uncertainty calculations."""

import numpy as np
import pytest

from air_quality.evaluation import (
    aggregate_elevated_probability,
    bootstrap_intervals,
    choose_threshold,
    paired_binary_comparison,
)


def test_aggregation_uses_probabilities_instead_of_argmax_label():
    probabilities = np.array([[0.10, 0.20, 0.35, 0.35], [0.26, 0.25, 0.25, 0.24]])
    np.testing.assert_allclose(aggregate_elevated_probability(probabilities), [0.70, 0.49])
    with pytest.raises(ValueError, match="sum"):
        aggregate_elevated_probability(probabilities * 0.8)
    with pytest.raises(ValueError, match="Expected"):
        aggregate_elevated_probability(np.ones((2, 2)))


def test_threshold_selection_is_deterministic_and_handles_ties():
    target = np.array([0, 0, 1, 1])
    values = np.array([0.1, 0.2, 0.8, 0.9])
    result = choose_threshold(target, values)
    assert result["threshold"] == 0.5
    assert result["training_oof_f2"] == 1
    assert result == choose_threshold(target, values)
    with pytest.raises(ValueError, match="both classes"):
        choose_threshold(np.zeros(4), values)


def test_perfect_bootstrap_is_exact_and_reproducible():
    target = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    intervals = bootstrap_intervals(target, target, iterations=100, seed=42)
    for interval in intervals.values():
        assert interval == {"low": 1.0, "high": 1.0}


def test_paired_comparison_keeps_identical_predictions_equal():
    target = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    strata = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    result = paired_binary_comparison(
        target, target, target, strata=strata, iterations=100, seed=42
    )
    assert result["paired_95ci"] == [0.0, 0.0]
    assert result["mcnemar_exact_p"] == 1
    assert result["macro_f1_delta_direct_minus_aggregated"] == 0


def test_paired_comparison_sign_is_direct_minus_aggregated():
    target = np.array([0, 0, 1, 1] * 4)
    result = paired_binary_comparison(
        target, target, 1 - target, strata=target, iterations=100, seed=42
    )
    assert result["paired_95ci"] == [1.0, 1.0]
    assert result["mcnemar_exact_p"] < 0.001
