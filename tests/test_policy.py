"""Policy C decision logic: label availability, trigger rules, action fork and promotion gate."""

from pathlib import Path

import pandas as pd
import pytest

import policy as pol

RESULTS = Path(__file__).resolve().parents[1] / "results" / "policy_comparison"
MONTHS = [str(p) for p in pd.period_range("2014-01", "2016-12", freq="M")]


# ------------------------------------------------- label delay and windows

def test_only_months_with_matured_labels_are_available():
    available = pol.get_available_labelled_months("2014-11", 5, MONTHS)
    assert available == ["2014-01", "2014-02", "2014-03", "2014-04", "2014-05", "2014-06"]


def test_validation_window_is_the_most_recent_labelled_months():
    train, valid = pol.split_training_and_validation("2014-11", 5, 3, MONTHS)
    assert valid == ["2014-04", "2014-05", "2014-06"]
    assert train == ["2014-01", "2014-02", "2014-03"]
    assert max(train) < min(valid)  # no candidate is validated on rows it was fitted on


def test_no_split_until_there_is_more_than_one_window_of_labels():
    assert pol.split_training_and_validation("2014-08", 5, 3, MONTHS) == ([], [])


# ------------------------------------------------------- trigger rules

def _evidence(**overrides):
    base = {"performance_level": "low", "concept_level": "low",
            "residual_mean_abs_shift": 0.0, "target_drift_trigger": False,
            "warning": False}
    return {**base, **overrides}


def test_low_performance_never_fires(art):
    assert pol.choose_candidate_action(_evidence(), 5, True, art) == "none"


def test_high_performance_fires_r1(art):
    assert pol.choose_candidate_action(_evidence(performance_level="high"), 0, False, art) == "R1"


def test_moderate_performance_with_concept_shift_fires_r2(art):
    ev = _evidence(performance_level="moderate", concept_level="high")
    assert pol.choose_candidate_action(ev, 0, False, art) == "R2"


def test_r3_needs_persistence_and_corroboration(art):
    ev = _evidence(performance_level="moderate")
    needed = art["persistence_cohorts"]
    # persistent but uncorroborated: immediate drift alone is not enough
    assert pol.choose_candidate_action(ev, needed, False, art) == "none"
    # corroborated by the previous cohort's immediate warning
    assert pol.choose_candidate_action(ev, needed, True, art) == "R3"
    # corroborated but not yet persistent
    assert pol.choose_candidate_action(ev, needed - 1, True, art) == "none"


# ---------------------------------------------------------- action fork

def _fork_evidence(**overrides):
    base = {"auc_drop": 0.0, "calibration_slope_deviation": 0.0,
            "residual_mean_abs_shift": 0.0}
    return {**base, **overrides}


def test_pure_level_shift_is_recalibrated(art):
    action, _ = pol.choose_initial_action(_fork_evidence(), None, False, art)
    assert action == "recalibrate"


@pytest.mark.parametrize("field", ["auc_drop", "calibration_slope_deviation",
                                   "residual_mean_abs_shift"])
def test_ranking_or_shape_change_trains_a_challenger(art, field):
    threshold = {"auc_drop": art["fork"]["auc_drop_fork_threshold"],
                 "calibration_slope_deviation": art["fork"]["calibration_slope_deviation"],
                 "residual_mean_abs_shift": art["fork"]["residual_bin_shift_threshold"]}[field]
    action, _ = pol.choose_initial_action(_fork_evidence(**{field: threshold}), None, False, art)
    assert action == "train_challenger"


def test_second_recalibration_is_not_attempted(art):
    action, reason = pol.choose_initial_action(_fork_evidence(), None, True, art)
    assert action == "train_challenger" and "recalibration" in reason


# ------------------------------------------------------- promotion gate

def _metrics(brier, gap, log_loss, auc):
    return {"brier_score": brier, "absolute_calibration_gap": gap,
            "log_loss": log_loss, "roc_auc": auc}


def test_clearly_better_challenger_passes(primary_gate):
    champion = _metrics(0.140, 0.030, 0.450, 0.70)
    candidate = _metrics(0.139, 0.020, 0.449, 0.70)
    result = pol.evaluate_candidate_gate(champion, candidate, 50_000, "challenger", primary_gate)
    assert result["deployment_gate_passed"] and result["failed_criteria"] == "none"


def test_too_few_validation_rows_blocks_promotion(primary_gate):
    champion = _metrics(0.140, 0.030, 0.450, 0.70)
    candidate = _metrics(0.130, 0.000, 0.440, 0.72)
    rows = primary_gate["min_validation_rows"] - 1
    result = pol.evaluate_candidate_gate(champion, candidate, rows, "challenger", primary_gate)
    assert not result["deployment_gate_passed"]
    assert result["failed_criteria"] == "validation_size"


def test_recalibration_may_trade_a_little_brier_for_calibration(primary_gate):
    champion = _metrics(0.1400, 0.030, 0.450, 0.70)
    worse_brier = 0.1400 + primary_gate["recalibration_brier_tolerance"] / 2
    candidate = _metrics(worse_brier, 0.010, 0.450, 0.70)
    recal = pol.evaluate_candidate_gate(champion, candidate, 50_000, "recalibration", primary_gate)
    chall = pol.evaluate_candidate_gate(champion, candidate, 50_000, "challenger", primary_gate)
    assert recal["deployment_gate_passed"]
    assert not chall["deployment_gate_passed"]  # a challenger must actually improve Brier


def test_auc_guardrail(primary_gate):
    champion = _metrics(0.140, 0.030, 0.450, 0.70)
    candidate = _metrics(0.130, 0.010, 0.440, 0.70 - primary_gate["auc_drop_guardrail"] - 1e-6)
    result = pol.evaluate_candidate_gate(champion, candidate, 50_000, "challenger", primary_gate)
    assert result["failed_criteria"] == "auc_guardrail"


# ------------------------------------- regression against the recorded replay

ATTEMPTS = pd.read_csv(RESULTS / "policy_C_candidate_attempts.csv")


@pytest.mark.parametrize("row", ATTEMPTS.to_dict("records"),
                         ids=lambda r: f"{r['decision_month']}-{r['candidate_type']}")
def test_gate_reproduces_every_recorded_decision(primary_gate, row):
    """Re-run the gate on the metrics recorded in the offline replay.

    This includes the November 2014 challenger, which the gate rejected.
    """
    champion = _metrics(row["champion_brier"], row["champion_absolute_calibration_gap"],
                        row["champion_log_loss"], row["champion_roc_auc"])
    candidate = _metrics(row["candidate_brier"], row["candidate_absolute_calibration_gap"],
                         row["candidate_log_loss"], row["candidate_roc_auc"])

    result = pol.evaluate_candidate_gate(champion, candidate, row["validation_rows"],
                                         row["candidate_type"], primary_gate)

    assert result["deployment_gate_passed"] == bool(row["deployment_gate_passed"])
    assert result["failed_criteria"] == row["failed_criteria"]


def test_replay_promoted_exactly_four_candidates():
    assert ATTEMPTS["deployment_gate_passed"].sum() == 4
