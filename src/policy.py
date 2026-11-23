# Policy C decision logic.
# Copied from Notebook 05 cells 32-33, 61-62 and 65. Thresholds arrive from the
# saved policy metadata, so no gate or fork value is restated here.

import pandas as pd


def get_available_labelled_months(decision_month, label_delay, all_months):
    latest_available = pd.Period(decision_month, freq="M") - label_delay
    return [m for m in all_months if pd.Period(m, freq="M") <= latest_available]


def split_training_and_validation(decision_month, label_delay, validation_window, all_months):
    available = get_available_labelled_months(decision_month, label_delay, all_months)

    if len(available) <= validation_window:
        return [], []

    return available[:-validation_window], available[-validation_window:]


def choose_candidate_action(evidence, persistence_count, previous_cohort_warning, art):
    at_least_moderate = evidence["performance_level"] in {"moderate", "high"}
    residual_deterioration = (pd.notna(evidence["residual_mean_abs_shift"]) and
                              evidence["residual_mean_abs_shift"] >=
                              art["fork"]["residual_bin_shift_threshold"])

    rule_R1 = evidence["performance_level"] == "high"
    rule_R2 = at_least_moderate and (evidence["concept_level"] == "high" or residual_deterioration)

    corroborated = (evidence["target_drift_trigger"] or residual_deterioration or
                    evidence["warning"] or previous_cohort_warning)

    rule_R3 = (at_least_moderate and persistence_count >= art["persistence_cohorts"]
               and corroborated)

    fired = [name for name, rule in [("R1", rule_R1), ("R2", rule_R2), ("R3", rule_R3)] if rule]

    return "+".join(fired) if len(fired) > 0 else "none"


def choose_initial_action(evidence, champion_state, deterioration_after_recalibration, art):
    auc_fork = art["fork"]["auc_drop_fork_threshold"]
    slope_fork = art["fork"]["calibration_slope_deviation"]
    residual_fork = art["fork"]["residual_bin_shift_threshold"]

    slope_deviation = abs(evidence["calibration_slope_deviation"])
    residual_localised = (pd.notna(evidence["residual_mean_abs_shift"]) and
                          evidence["residual_mean_abs_shift"] >= residual_fork)

    if evidence["auc_drop"] >= auc_fork:
        return ("train_challenger",
                f"ROC-AUC drop {evidence['auc_drop']:.4f} >= {auc_fork}: ranking deteriorated")

    if slope_deviation >= slope_fork:
        return ("train_challenger",
                f"calibration slope moved {slope_deviation:.3f} from its reference: "
                "not a pure level shift")

    if residual_localised:
        return ("train_challenger",
                f"residual score-bin movement {evidence['residual_mean_abs_shift']:.4f} >= "
                f"{residual_fork} after removing the global shift: localised conditional change")

    if deterioration_after_recalibration:
        return ("train_challenger",
                "champion already carries a recalibration layer yet deterioration persisted: "
                "the level is not the problem")

    return ("recalibrate",
            "ranking and calibration slope broadly stable, movement is a global level shift "
            "with small residual bin movement: correct the level, keep the ranking")


def evaluate_candidate_gate(champion_result, candidate_result, validation_rows,
                            candidate_type, gate):
    brier_improvement = champion_result["brier_score"] - candidate_result["brier_score"]
    calibration_improvement = (champion_result["absolute_calibration_gap"] -
                               candidate_result["absolute_calibration_gap"])

    if candidate_type == "recalibration":
        gate_design = "recalibration"
        brier_required_min = -gate["recalibration_brier_tolerance"]
        calibration_required_min = gate["recalibration_calibration_gap_improvement_min"]
        brier_label = "brier_noninferiority"
    else:
        gate_design = "challenger"
        brier_required_min = gate["challenger_brier_improvement_min"]
        calibration_required_min = gate["challenger_calibration_gap_improvement_min"]
        brier_label = "brier_improvement"

    brier_criterion_passed = bool(brier_improvement >= brier_required_min)
    calibration_criterion_passed = bool(calibration_improvement >= calibration_required_min)

    log_loss_criterion_passed = bool(candidate_result["log_loss"] <=
                                     champion_result["log_loss"] + gate["log_loss_tolerance"])

    auc_guardrail_passed = bool(candidate_result["roc_auc"] >=
                                champion_result["roc_auc"] - gate["auc_drop_guardrail"])

    validation_size_passed = bool(validation_rows >= gate["min_validation_rows"])

    gate_passed = bool(brier_criterion_passed and calibration_criterion_passed and
                       log_loss_criterion_passed and auc_guardrail_passed and
                       validation_size_passed)

    failed_criteria = [name for name, passed in
                       [(brier_label, brier_criterion_passed),
                        ("calibration_gap", calibration_criterion_passed),
                        ("log_loss", log_loss_criterion_passed),
                        ("auc_guardrail", auc_guardrail_passed),
                        ("validation_size", validation_size_passed)]
                       if not passed]

    return {"gate_design": gate_design,
            "gate_name": gate["name"],
            "brier_improvement": brier_improvement,
            "calibration_improvement": calibration_improvement,
            "brier_required_min": brier_required_min,
            "calibration_required_min": calibration_required_min,
            "brier_criterion_passed": brier_criterion_passed,
            "calibration_criterion_passed": calibration_criterion_passed,
            "log_loss_criterion_passed": log_loss_criterion_passed,
            "auc_guardrail_passed": auc_guardrail_passed,
            "validation_size_passed": validation_size_passed,
            "deployment_gate_passed": gate_passed,
            "failed_criteria": ", ".join(failed_criteria) if failed_criteria else "none"}


def base_record(policy_name, policy_version, run_id, monitor_month, decision_month,
                champion_id, last_stream_month):
    """The Notebook 05 decision row, kept column-for-column so parity is a direct comparison."""
    return {"policy_name": policy_name,
            "policy_version": policy_version,
            "run_id": run_id,
            "monitor_month": monitor_month,
            "decision_month": decision_month,
            "in_stream_decision": bool(pd.Period(decision_month, freq="M") <=
                                       pd.Period(last_stream_month, freq="M")),
            "champion_id": champion_id,
            "scored_by": "",
            "evidence_relevant": None,
            "performance_level": "",
            "concept_level": "",
            "target_drift_level": "",
            "residual_mean_abs_shift": None,
            "persistence_count": None,
            "persistence_satisfied": None,
            "fired_rule": "none",
            "candidate_trigger": False,
            "suppressed_by_deployment_cooldown": False,
            "suppressed_by_candidate_evaluation_holdoff": False,
            "insufficient_labelled_data": False,
            "candidate_evaluated": False,
            "intended_action": "none",
            "action_reason": "",
            "final_action": "none",
            "candidate_promoted": False,
            "promotion_type": "none",
            "policy_reason": ""}
