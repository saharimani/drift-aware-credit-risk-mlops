"""Delayed-evidence metrics and the logistic recalibration layer."""

import numpy as np
import pytest
from sklearn.metrics import brier_score_loss

import evidence as ev


def test_log_odds_round_trip():
    p = np.array([0.01, 0.2, 0.5, 0.9])
    back = 1 / (1 + np.exp(-ev.to_log_odds(p)))
    np.testing.assert_allclose(back, p)


def test_log_odds_clips_certain_predictions():
    assert np.all(np.isfinite(ev.to_log_odds([0.0, 1.0])))


def test_calibration_gap_is_actual_minus_predicted():
    y = np.array([1, 0, 0, 0])         # 25% bad
    p = np.full(4, 0.10)               # model says 10%
    result = ev.evaluate_predictions(y, p)
    assert result["calibration_gap"] == pytest.approx(0.15)
    assert result["absolute_calibration_gap"] == pytest.approx(0.15)
    assert result["brier_score"] == pytest.approx(brier_score_loss(y, p))


def test_single_class_cohort_does_not_crash():
    result = ev.evaluate_predictions(np.zeros(10), np.full(10, 0.1))
    assert np.isnan(result["roc_auc"]) and np.isnan(result["average_precision"])


def test_recalibration_removes_a_level_shift_and_keeps_the_ranking():
    """The situation Policy C recalibrates for: right ranking, wrong level."""
    rng = np.random.default_rng(0)
    true_p = rng.uniform(0.05, 0.40, size=20_000)
    y = rng.binomial(1, true_p)
    understated = 1 / (1 + np.exp(-(ev.to_log_odds(true_p) - 0.6)))  # same order, lower level

    recal = ev.fit_recalibration_map(understated, y)
    fixed = ev.apply_recalibration_map(recal, understated)

    before = ev.evaluate_predictions(y, understated)
    after = ev.evaluate_predictions(y, fixed)
    assert after["absolute_calibration_gap"] < 0.01 < before["absolute_calibration_gap"]
    assert after["brier_score"] < before["brier_score"]
    assert after["roc_auc"] == pytest.approx(before["roc_auc"])  # monotone map
