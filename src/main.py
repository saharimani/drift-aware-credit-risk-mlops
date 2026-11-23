# One decision month of Policy C, as a stateless Cloud Run Job, plus the
# Policy A shadow request baseline.
#
# This replaces the run_governed_policy calendar loop from Notebook 05 with a
# single-month operational runner. The policy logic inside it is unchanged: the
# loop became a job, the in-memory state became a BigQuery table.
#
# The full replay is 41 executions: 2014-01 to 2016-12 score real cohorts;
# 2017-01 to 2017-05 are tail months that score nothing and only mature the
# remaining 2016 cohorts under the five-month label delay.

import argparse
import datetime as dt
import json
import os
import sys
import time
import uuid

import joblib
import numpy as np
import pandas as pd

import storage
import state as st
from storage import log
import signals as sig
import evidence as ev
import policy as pol
import candidate as cand


# ------------------------------------------------------------------ bootstrap

def bootstrap_state(cfg, art):
    """First run only: publish the original champion and open the state row."""
    champion_model = joblib.load(storage.fetch(cfg, cfg["gcs"]["champion_model"]))
    concept_baseline = pd.read_csv(storage.fetch(cfg, cfg["gcs"]["concept_drift_baseline"]))
    calibration_scored = pd.read_parquet(storage.fetch(cfg, cfg["gcs"]["calibration_scored"]))

    champion = ev.build_starting_champion(champion_model, concept_baseline,
                                          calibration_scored, art["score_edges"], cfg)

    model_uri, reference_uri = st.save_champion(cfg, champion)
    row = st.initialise_state(cfg, model_uri, reference_uri)

    record_promotion(cfg, {"decision_month": "start",
                           "trigger_monitor_month": "none",
                           "champion_id": champion["champion_id"],
                           "action_type": "original",
                           "validation_start_month": "2013-11",
                           "validation_end_month": "2013-12",
                           "model_uri": model_uri,
                           "reference_uri": reference_uri,
                           "run_id": "init"})
    return row


def record_promotion(cfg, promotion):
    row = {"policy_name": cfg["policy_name"],
           "policy_version": cfg["policy_version"],
           **promotion,
           "recorded_at": dt.datetime.now(dt.timezone.utc)}

    storage.merge_rows(cfg, "policy_promotions", [row],
                       ["policy_name", "policy_version", "champion_id"])


def reference_uri_for(cfg, champion_id):
    """Every champion that ever scored a cohort is recoverable from policy_promotions."""
    sql = f"""
    SELECT model_uri, reference_uri
    FROM `{storage.table_id(cfg, 'policy_promotions')}`
    WHERE policy_name = @policy_name
      AND policy_version = @policy_version
      AND champion_id = @champion_id
    """
    result = storage.query(cfg, sql,
                           policy_name=cfg["policy_name"],
                           policy_version=cfg["policy_version"],
                           champion_id=champion_id)

    if result.empty:
        raise RuntimeError(f"No stored artefacts for champion {champion_id}.")

    return result["model_uri"].iloc[0], result["reference_uri"].iloc[0]


# ------------------------------------------------------------- scoring a month

def score_and_monitor(cfg, art, champion, target_month, run_id):
    """Steps 4-7: load the cohort, score it, freeze the probabilities, monitor."""
    started = time.time()

    cohort = storage.load_cohort(cfg, target_month).reset_index(drop=True)
    scores = ev.score_champion(champion, cohort, art["champion_feature_cols"])

    label_available_month = str(pd.Period(target_month, freq="M") + art["label_delay_months"])

    frozen = pd.DataFrame({
        "policy_name": cfg["policy_name"],
        "policy_version": cfg["policy_version"],
        "monitor_month": target_month,
        "row_index": np.arange(len(cohort), dtype="int64"),
        "scoring_champion_id": champion["champion_id"],
        "predicted_bad_loan_probability": scores,
        "label_available_month": label_available_month,
        "target_bad_loan": pd.Series([None] * len(cohort), dtype="Int64"),
        "scored_at": dt.datetime.now(dt.timezone.utc),
        "run_id": run_id})

    storage.merge_rows(cfg, "prototype_scores", frozen,
                       ["policy_name", "policy_version", "monitor_month", "row_index"])

    historical = storage.load_frames(cfg)["historical"]
    immediate = sig.immediate_signals(historical, cohort, scores,
                                      champion["reference"], art)

    metrics_row = {"policy_name": cfg["policy_name"],
                   "policy_version": cfg["policy_version"],
                   "monitor_month": target_month,
                   "scoring_champion_id": champion["champion_id"],
                   "rows": int(len(cohort)),
                   "mean_predicted_probability": float(np.mean(scores)),
                   **{k: immediate[k] for k in
                      ["max_feature_psi", "mean_feature_psi",
                       "supported_feature_drift_count", "supported_high_drift_count",
                       "feature_drift_trigger", "prediction_jsd",
                       "prediction_drift_trigger", "unseen_category_count",
                       "unseen_category_detail", "mean_feature_missing_rate",
                       "data_quality_flag", "warning"]},
                   "computed_at": dt.datetime.now(dt.timezone.utc),
                   "run_id": run_id}

    storage.merge_rows(cfg, "monitoring_metrics", [metrics_row],
                       ["policy_name", "policy_version", "monitor_month"])

    # Policy A shadow baseline: the naive feature-drift request, written from the
    # trigger already computed above. No model is fitted, no champion is kept.
    policy_a_row = {"policy_name": "policy_a_naive_feature_drift",
                    "policy_version": cfg["policy_version"],
                    "monitor_month": target_month,
                    "feature_drift_trigger": bool(immediate["feature_drift_trigger"]),
                    "retraining_request": bool(immediate["feature_drift_trigger"]),
                    "request_reason": ("supported feature drift: >=1 supported high-PSI feature "
                                       "or >=5 supported drifting features"
                                       if immediate["feature_drift_trigger"] else "none"),
                    "recorded_at": dt.datetime.now(dt.timezone.utc),
                    "run_id": run_id}

    storage.merge_rows(cfg, "policy_a_requests", [policy_a_row],
                       ["policy_name", "policy_version", "monitor_month"])

    log("policy_a_shadow_request", monitor_month=target_month,
        retraining_request=policy_a_row["retraining_request"])

    log("cohort_scored_and_monitored", monitor_month=target_month,
        scoring_champion_id=champion["champion_id"], rows=len(cohort),
        feature_drift_trigger=immediate["feature_drift_trigger"],
        prediction_jsd=round(immediate["prediction_jsd"], 6),
        warning=immediate["warning"])

    return time.time() - started, len(cohort)


# ------------------------------------------------------- maturing the evidence

def mature_evidence(cfg, art, mature_month, decision_month, run_id):
    """Steps 8-9: recover the frozen probabilities, attach labels, compute evidence."""
    frozen = st.read_frozen_cohort(cfg, mature_month, decision_month)

    if frozen.empty:
        return None

    scoring_champion_id = frozen["scoring_champion_id"].unique()
    if len(scoring_champion_id) != 1:
        raise RuntimeError(f"{mature_month} was scored by more than one champion.")
    scoring_champion_id = scoring_champion_id[0]

    cohort = storage.load_cohort(cfg, mature_month).reset_index(drop=True)
    if len(cohort) != len(frozen):
        raise RuntimeError(f"{mature_month}: frozen rows do not match the cohort.")

    y_month = cohort[cfg["target_col"]].values
    deployed_scores = frozen["predicted_bad_loan_probability"].values

    # Labels are written only now, at the month they become available
    storage.merge_rows(cfg, "prototype_scores",
                       pd.DataFrame({"policy_name": cfg["policy_name"],
                                     "policy_version": cfg["policy_version"],
                                     "monitor_month": mature_month,
                                     "row_index": frozen["row_index"].values,
                                     "target_bad_loan": pd.Series(y_month, dtype="Int64")}),
                       ["policy_name", "policy_version", "monitor_month", "row_index"])

    _, reference_uri = reference_uri_for(cfg, scoring_champion_id)
    reference = storage.read_uri(cfg, reference_uri)

    evidence = ev.compute_champion_evidence(mature_month, y_month, deployed_scores,
                                            reference, art, cfg)
    evidence.update(ev.target_drift_for_cohort(y_month, art))
    evidence.update(st.read_immediate_signals(cfg, mature_month))
    evidence["scored_by"] = scoring_champion_id
    evidence["decision_month"] = decision_month
    evidence["label_available_month"] = frozen["label_available_month"].iloc[0]

    return evidence


def write_evidence(cfg, evidence, relevant, run_id):
    keep = ["monitor_month", "decision_month", "label_available_month", "scored_by",
            "rows", "actual_bad_loan_rate", "mean_predicted_probability",
            "calibration_gap", "absolute_calibration_gap", "brier_score", "log_loss",
            "roc_auc", "average_precision", "brier_change", "log_loss_change",
            "auc_drop", "performance_level", "score_bin_mean_absolute_shift",
            "score_bin_signed_shift", "residual_mean_abs_shift", "concept_level",
            "concept_shift_type", "calibration_slope", "calibration_slope_deviation",
            "target_rate_change", "target_drift_level", "target_drift_trigger",
            "feature_drift_trigger", "prediction_drift_trigger", "data_quality_flag",
            "warning"]

    row = {"policy_name": cfg["policy_name"],
           "policy_version": cfg["policy_version"],
           **{k: evidence[k] for k in keep},
           "relevant_to_current_champion": bool(relevant),
           "computed_at": dt.datetime.now(dt.timezone.utc),
           "run_id": run_id}

    storage.merge_rows(cfg, "delayed_evidence", [row],
                       ["policy_name", "policy_version", "decision_month", "monitor_month"])


# ------------------------------------------------------------- candidate stage

def try_candidate(cfg, art, champion, state, record, evidence, intended_action, run_id):
    """Steps 15-17: build the chronological windows, fit, gate, promote or reject."""
    months = storage.load_frames(cfg)["months"]
    decision_month = record["decision_month"]

    outcome = {"candidate_evaluated": False,
               "insufficient_labelled_data": False,
               "final_action": "none",
               "candidate_promoted": False,
               "promotion_type": "none",
               "policy_reason": ""}

    training_months, validation_months = pol.split_training_and_validation(
        decision_month, art["label_delay_months"], art["validation_window_months"], months)

    if len(training_months) == 0 or len(validation_months) < art["validation_window_months"]:
        outcome["insufficient_labelled_data"] = True
        outcome["policy_reason"] = "insufficient_labelled_data"
        return outcome, [], None, 0.0

    simulation = storage.load_frames(cfg)["simulation"]
    historical = storage.load_frames(cfg)["historical"]

    training_df = simulation[simulation["monitor_month"].isin(training_months)]
    validation_df = simulation[simulation["monitor_month"].isin(validation_months)]

    outcome["candidate_evaluated"] = True
    state["last_evaluation_month"] = decision_month

    y_validation = validation_df[cfg["target_col"]].values
    champion_scores = ev.score_champion(champion, validation_df, art["champion_feature_cols"])
    champion_result = ev.evaluate_predictions(y_validation, champion_scores)

    fit_started = time.time()
    attempts = []

    if intended_action == "recalibrate":
        attempts.append(("recalibration",
                         cand.build_recalibration_candidate(champion, training_df,
                                                            decision_month,
                                                            art["champion_feature_cols"])))
    else:
        attempts.append(("challenger",
                         cand.build_challenger_candidate(historical, training_df,
                                                         decision_month,
                                                         art["champion_feature_cols"],
                                                         cfg["xgb_n_jobs"])))

    attempt_rows = []
    promoted = None
    attempt_index = 0

    while attempt_index < len(attempts):
        attempt_type, candidate_state = attempts[attempt_index]
        candidate_scores = ev.score_champion(candidate_state, validation_df,
                                             art["champion_feature_cols"])
        candidate_result = ev.evaluate_predictions(y_validation, candidate_scores)

        gate_result = pol.evaluate_candidate_gate(champion_result, candidate_result,
                                                  validation_df.shape[0], attempt_type,
                                                  art["gate"])

        attempt_rows.append({
            "policy_name": cfg["policy_name"],
            "policy_version": cfg["policy_version"],
            "run_id": run_id,
            "decision_month": decision_month,
            "trigger_monitor_month": record["monitor_month"],
            "attempt_index": attempt_index + 1,
            "candidate_type": attempt_type,
            "current_champion": champion["champion_id"],
            "candidate_id": candidate_state["champion_id"],
            "simulation_training_start_month": training_months[0],
            "simulation_training_end_month": training_months[-1],
            "historical_training_rows": int(historical.shape[0]) if attempt_type == "challenger" else 0,
            "simulation_training_rows": int(training_df.shape[0]),
            "total_training_rows": int(training_df.shape[0] +
                                       (historical.shape[0] if attempt_type == "challenger" else 0)),
            "validation_start_month": validation_months[0],
            "validation_end_month": validation_months[-1],
            "validation_rows": int(validation_df.shape[0]),
            "champion_brier": champion_result["brier_score"],
            "candidate_brier": candidate_result["brier_score"],
            "champion_absolute_calibration_gap": champion_result["absolute_calibration_gap"],
            "candidate_absolute_calibration_gap": candidate_result["absolute_calibration_gap"],
            "champion_log_loss": champion_result["log_loss"],
            "candidate_log_loss": candidate_result["log_loss"],
            "log_loss_change": candidate_result["log_loss"] - champion_result["log_loss"],
            "champion_roc_auc": champion_result["roc_auc"],
            "candidate_roc_auc": candidate_result["roc_auc"],
            "auc_change": candidate_result["roc_auc"] - champion_result["roc_auc"],
            **gate_result,
            "evaluated_at": dt.datetime.now(dt.timezone.utc)})

        log("candidate_evaluated", decision_month=decision_month,
            candidate_type=attempt_type, candidate_id=candidate_state["champion_id"],
            brier_improvement=round(gate_result["brier_improvement"], 6),
            calibration_improvement=round(gate_result["calibration_improvement"], 6),
            deployment_gate_passed=gate_result["deployment_gate_passed"],
            failed_criteria=gate_result["failed_criteria"])

        if gate_result["deployment_gate_passed"]:
            promoted = (candidate_state, candidate_scores)
            break

        # Challenger fallback after a failed recalibration, as in Notebook 05
        if attempt_type == "recalibration" and attempt_index == len(attempts) - 1:
            attempts.append(("challenger",
                             cand.build_challenger_candidate(historical, training_df,
                                                             decision_month,
                                                             art["champion_feature_cols"],
                                                             cfg["xgb_n_jobs"])))

        attempt_index += 1

    fit_duration = time.time() - fit_started

    if promoted is None:
        outcome["final_action"] = "retain_champion"
        outcome["policy_reason"] = "all candidate attempts failed the deployment gate"
        return outcome, attempt_rows, None, fit_duration

    candidate_state, candidate_scores = promoted

    candidate_state["reference"] = ev.build_champion_reference(
        y_validation, candidate_scores,
        f"pre_deployment_validation_{validation_months[0]}_{validation_months[-1]}")

    model_uri, reference_uri = st.save_champion(cfg, candidate_state)

    record_promotion(cfg, {"decision_month": decision_month,
                           "trigger_monitor_month": record["monitor_month"],
                           "champion_id": candidate_state["champion_id"],
                           "action_type": candidate_state["action_type"],
                           "validation_start_month": validation_months[0],
                           "validation_end_month": validation_months[-1],
                           "model_uri": model_uri,
                           "reference_uri": reference_uri,
                           "run_id": run_id})

    # Policy state changes only here, and only after the gate has passed
    state["active_champion_id"] = candidate_state["champion_id"]
    state["active_action_type"] = candidate_state["action_type"]
    state["active_model_uri"] = model_uri
    state["active_reference_uri"] = reference_uri
    state["last_promotion_month"] = decision_month
    state["persistence_count"] = 0
    state["previous_cohort_warning"] = False

    outcome["final_action"] = candidate_state["action_type"]
    outcome["candidate_promoted"] = True
    outcome["promotion_type"] = candidate_state["action_type"]
    outcome["policy_reason"] = f"promoted: {candidate_state['action_type']} passed the deployment gate"

    return outcome, attempt_rows, candidate_state, fit_duration


# ---------------------------------------------------------------- the job body

def run_month(target_month, policy_name=None, config_path=None):
    started = time.time()
    run_id = uuid.uuid4().hex[:12]

    cfg = storage.load_config(config_path)

    # Only an explicitly supplied policy name overrides the resolved config
    if policy_name:
        cfg["policy_name"] = policy_name

    log("run_started", run_id=run_id, decision_month=target_month,
        policy_name=cfg["policy_name"], policy_version=cfg["policy_version"],
        project_id=cfg["project_id"], dataset=cfg["dataset"], bucket=cfg["bucket"])

    # 1. Idempotency
    if st.terminal_decision_exists(cfg, target_month):
        log("duplicate_run_skipped", severity="NOTICE", run_id=run_id,
            decision_month=target_month)
        write_run_row(cfg, run_id, target_month, started, "duplicate_run_skipped", {})
        return {"run_id": run_id, "decision_month": target_month,
                "outcome": "duplicate_run_skipped"}

    art = storage.load_artefacts(cfg)
    concept_baseline = pd.read_csv(storage.fetch(cfg, cfg["gcs"]["concept_drift_baseline"]))
    art["baseline_bad_loan_rate"] = float(concept_baseline["actual_bad_loan_rate"].iloc[0])

    # 2. Policy state, initialised from the original champion on the first run only
    row = st.read_state(cfg) or bootstrap_state(cfg, art)

    state = {"active_champion_id": row["active_champion_id"],
             "active_action_type": row["active_action_type"],
             "active_model_uri": row["active_model_uri"],
             "active_reference_uri": row["active_reference_uri"],
             "persistence_count": int(row["persistence_count"]),
             "last_promotion_month": row["last_promotion_month"],
             "last_evaluation_month": row["last_evaluation_month"],
             "previous_cohort_warning": bool(row["previous_cohort_warning"])}

    # 3. The active champion comes from state, never from the newest file in the bucket
    champion = st.load_champion(cfg, state["active_model_uri"], state["active_reference_uri"])
    log("champion_loaded", champion_id=champion["champion_id"],
        action_type=champion["action_type"], model_uri=state["active_model_uri"])

    months = storage.load_frames(cfg)["months"]
    last_stream_month = months[-1]

    # 4-10. Score this month's cohort with the champion active at the start of the
    # month, monitor it, and write the Policy A shadow request. Tail months
    # (2017-01 to 2017-05) have no cohort: they skip scoring entirely and only
    # mature delayed evidence below.
    monitoring_seconds, rows_scored = 0.0, 0
    is_tail_month = target_month not in months

    if is_tail_month:
        log("tail_month_no_scoring", decision_month=target_month,
            reason="beyond the 2014-2016 cohort stream; maturing delayed evidence only")
    elif st.cohort_already_scored(cfg, target_month):
        log("cohort_scoring_skipped", monitor_month=target_month,
            reason="cohort already scored and frozen")
    else:
        monitoring_seconds, rows_scored = score_and_monitor(cfg, art, champion,
                                                            target_month, run_id)

    # 8-9. Evidence for the cohort whose labels mature now
    mature_month = str(pd.Period(target_month, freq="M") - art["label_delay_months"])
    evidence = None

    if mature_month in months:
        evidence = mature_evidence(cfg, art, mature_month, target_month, run_id)

    if evidence is None:
        log("no_matured_cohort", decision_month=target_month, mature_month=mature_month)
        st.write_state(cfg, state, run_id)
        write_run_row(cfg, run_id, target_month, started, "no_decision",
                      {"monitoring_duration_seconds": monitoring_seconds,
                       "rows_scored": rows_scored})
        return {"run_id": run_id, "decision_month": target_month, "outcome": "no_decision"}

    # 10. Does this evidence belong to the active champion?
    relevant = evidence["scored_by"] == state["active_champion_id"]
    write_evidence(cfg, evidence, relevant, run_id)

    # 11. Champion-specific persistence
    if relevant:
        if evidence["performance_level"] in {"moderate", "high"}:
            state["persistence_count"] += 1
        else:
            state["persistence_count"] = 0

    record = pol.base_record(cfg["policy_name"], cfg["policy_version"], run_id,
                             mature_month, target_month, champion["champion_id"],
                             last_stream_month)

    record.update({"scored_by": evidence["scored_by"],
                   "evidence_relevant": bool(relevant),
                   "performance_level": evidence["performance_level"],
                   "concept_level": evidence["concept_level"],
                   "target_drift_level": evidence["target_drift_level"],
                   "residual_mean_abs_shift": evidence["residual_mean_abs_shift"],
                   "persistence_count": int(state["persistence_count"]),
                   "persistence_satisfied": bool(state["persistence_count"] >=
                                                 art["persistence_cohorts"])})

    attempt_rows = []
    fit_duration = 0.0
    candidate_type = "none"

    if not relevant:
        # 12. Evidence from a superseded champion cannot act against the current one
        record["policy_reason"] = ("cohort scored by a previous champion; evidence "
                                   "retained for retrospective evaluation only")
    else:
        fired_rule = pol.choose_candidate_action(evidence, state["persistence_count"],
                                                 state["previous_cohort_warning"], art)
        record["fired_rule"] = fired_rule
        record["candidate_trigger"] = fired_rule != "none"

        if not record["candidate_trigger"]:
            record["policy_reason"] = "no candidate trigger"
        else:
            # 13. Cooldown and hold-off, before anything is fitted
            since_promotion = st.months_between(target_month, state["last_promotion_month"])
            since_evaluation = st.months_between(target_month, state["last_evaluation_month"])

            cooldown_active = (since_promotion is not None and
                               since_promotion < art["deployment_cooldown_months"])
            holdoff_active = (since_evaluation is not None and
                              since_evaluation < art["candidate_evaluation_holdoff_months"])

            if cooldown_active:
                record["suppressed_by_deployment_cooldown"] = True
                record["policy_reason"] = (f"suppressed: within the "
                                           f"{art['deployment_cooldown_months']}-month "
                                           "deployment cooldown")
            elif holdoff_active:
                record["suppressed_by_candidate_evaluation_holdoff"] = True
                record["policy_reason"] = (f"suppressed: within the "
                                           f"{art['candidate_evaluation_holdoff_months']}-month "
                                           "candidate evaluation hold-off")
            else:
                # 14. Recalibrate or train a challenger
                deterioration_after_recalibration = (
                    champion["active_recalibration_map"] is not None and
                    state["persistence_count"] >= art["persistence_cohorts"])

                intended_action, action_reason = pol.choose_initial_action(
                    evidence, champion, deterioration_after_recalibration, art)

                record["intended_action"] = intended_action
                record["action_reason"] = action_reason

                outcome, attempt_rows, promoted_state, fit_duration = try_candidate(
                    cfg, art, champion, state, record, evidence, intended_action, run_id)

                record.update(outcome)
                candidate_type = attempt_rows[-1]["candidate_type"] if attempt_rows else "none"

    # The warning carried forward belongs to the champion active after this decision
    if evidence["scored_by"] == state["active_champion_id"]:
        state["previous_cohort_warning"] = bool(evidence["warning"])

    # 18-19. Persist state and the decision
    st.write_state(cfg, state, run_id)

    record["decided_at"] = dt.datetime.now(dt.timezone.utc)
    storage.merge_rows(cfg, "policy_decisions", [record],
                       ["policy_name", "policy_version", "decision_month"])

    if attempt_rows:
        storage.merge_rows(cfg, "candidate_attempts", attempt_rows,
                           ["policy_name", "policy_version", "decision_month", "attempt_index"])

    log("policy_decision", decision_month=target_month, monitor_month=mature_month,
        in_stream_decision=record["in_stream_decision"],
        scored_by=record["scored_by"], evidence_relevant=record["evidence_relevant"],
        fired_rule=record["fired_rule"], persistence_count=record["persistence_count"],
        intended_action=record["intended_action"], final_action=record["final_action"],
        candidate_promoted=record["candidate_promoted"],
        active_champion_after=state["active_champion_id"])

    write_run_row(cfg, run_id, target_month, started, "completed",
                  {"monitoring_duration_seconds": monitoring_seconds,
                   "rows_scored": rows_scored,
                   "candidate_fit_started": bool(attempt_rows),
                   "candidate_fit_duration_seconds": fit_duration,
                   "candidate_type": candidate_type,
                   "candidate_promoted": bool(record["candidate_promoted"])})

    # 20. Compact execution summary
    summary = {"run_id": run_id,
               "decision_month": target_month,
               "in_stream_decision": record["in_stream_decision"],
               "tail_month": is_tail_month,
               "monitor_month": mature_month,
               "scored_by": record["scored_by"],
               "fired_rule": record["fired_rule"],
               "intended_action": record["intended_action"],
               "final_action": record["final_action"],
               "candidate_promoted": record["candidate_promoted"],
               "active_champion_after": state["active_champion_id"],
               "job_duration_seconds": round(time.time() - started, 2)}

    log("run_completed", **summary)
    return summary


def write_run_row(cfg, run_id, decision_month, started, outcome, extra):
    row = {"run_id": run_id,
           "policy_name": cfg["policy_name"],
           "policy_version": cfg["policy_version"],
           "decision_month": decision_month,
           "started_at": dt.datetime.fromtimestamp(started, dt.timezone.utc),
           "ended_at": dt.datetime.now(dt.timezone.utc),
           "job_duration_seconds": round(time.time() - started, 2),
           "monitoring_duration_seconds": extra.get("monitoring_duration_seconds", 0.0),
           "rows_scored": int(extra.get("rows_scored", 0)),
           "candidate_fit_started": extra.get("candidate_fit_started", False),
           "candidate_fit_duration_seconds": extra.get("candidate_fit_duration_seconds", 0.0),
           "candidate_type": extra.get("candidate_type", "none"),
           "candidate_promoted": extra.get("candidate_promoted", False),
           "outcome": outcome,
           "image_digest": os.environ.get("IMAGE_DIGEST", "local"),
           "cpu": os.environ.get("JOB_CPU", "unknown"),
           "memory": os.environ.get("JOB_MEMORY", "unknown")}

    storage.merge_rows(cfg, "prototype_runs", [row], ["run_id"])


def main():
    parser = argparse.ArgumentParser(description="Run one Policy C decision month.")
    parser.add_argument("--month", default=os.environ.get("TARGET_MONTH"))
    parser.add_argument("--policy", default=None,
                        help="override the policy name from the resolved config")
    parser.add_argument("--config", default=os.environ.get("PROTOTYPE_CONFIG"))
    args = parser.parse_args()

    if not args.month:
        raise SystemExit("Provide --month YYYY-MM or set TARGET_MONTH.")

    try:
        summary = run_month(args.month, args.policy, args.config)
    except Exception as error:
        log("run_failed", severity="ERROR", decision_month=args.month, error=str(error))
        raise

    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
