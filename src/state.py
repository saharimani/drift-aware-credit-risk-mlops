# Policy state that the notebook engine held in memory and a stateless job cannot.
# The active champion is whatever policy_state says it is, never the newest file
# in the bucket.

import datetime as dt
import os
import tempfile

import joblib
import pandas as pd

import storage
from storage import log


def terminal_decision_exists(cfg, decision_month):
    sql = f"""
    SELECT COUNT(*) AS n
    FROM `{storage.table_id(cfg, 'policy_decisions')}`
    WHERE policy_name = @policy_name
      AND decision_month = @decision_month
      AND policy_version = @policy_version
    """
    result = storage.query(cfg, sql,
                           policy_name=cfg["policy_name"],
                           decision_month=decision_month,
                           policy_version=cfg["policy_version"])
    return int(result["n"].iloc[0]) > 0


def cohort_already_scored(cfg, monitor_month):
    sql = f"""
    SELECT COUNT(*) AS n
    FROM `{storage.table_id(cfg, 'prototype_scores')}`
    WHERE policy_name = @policy_name
      AND monitor_month = @monitor_month
      AND policy_version = @policy_version
    """
    result = storage.query(cfg, sql,
                           policy_name=cfg["policy_name"],
                           monitor_month=monitor_month,
                           policy_version=cfg["policy_version"])
    return int(result["n"].iloc[0]) > 0


def read_frozen_cohort(cfg, monitor_month, decision_month):
    """Frozen scores for a matured cohort.

    The label-availability condition is enforced in the query itself, so a decision
    can never reach a label that had not arrived by decision_month.
    """
    sql = f"""
    SELECT row_index, scoring_champion_id, predicted_bad_loan_probability,
           label_available_month
    FROM `{storage.table_id(cfg, 'prototype_scores')}`
    WHERE policy_name = @policy_name
      AND policy_version = @policy_version
      AND monitor_month = @monitor_month
      AND label_available_month <= @decision_month
    ORDER BY row_index
    """
    return storage.query(cfg, sql,
                         policy_name=cfg["policy_name"],
                         policy_version=cfg["policy_version"],
                         monitor_month=monitor_month,
                         decision_month=decision_month)


def read_immediate_signals(cfg, monitor_month):
    sql = f"""
    SELECT feature_drift_trigger, prediction_drift_trigger, data_quality_flag, warning
    FROM `{storage.table_id(cfg, 'monitoring_metrics')}`
    WHERE policy_name = @policy_name
      AND policy_version = @policy_version
      AND monitor_month = @monitor_month
    """
    result = storage.query(cfg, sql,
                           policy_name=cfg["policy_name"],
                           policy_version=cfg["policy_version"],
                           monitor_month=monitor_month)
    if result.empty:
        raise RuntimeError(f"No immediate signals recorded for {monitor_month}; "
                           "run that month before deciding on it.")
    return result.iloc[0].to_dict()


def read_state(cfg):
    sql = f"""
    SELECT *
    FROM `{storage.table_id(cfg, 'policy_state')}`
    WHERE policy_name = @policy_name AND policy_version = @policy_version
    """
    result = storage.query(cfg, sql,
                           policy_name=cfg["policy_name"],
                           policy_version=cfg["policy_version"])
    return None if result.empty else result.iloc[0].to_dict()


def initialise_state(cfg, model_uri, reference_uri):
    """One row, written once, describing the original Notebook 03 champion."""
    row = {"policy_name": cfg["policy_name"],
           "policy_version": cfg["policy_version"],
           "active_champion_id": cfg["starting_champion_id"],
           "active_action_type": "original",
           "active_model_uri": model_uri,
           "active_reference_uri": reference_uri,
           "persistence_count": 0,
           "last_promotion_month": None,
           "last_evaluation_month": None,
           "previous_cohort_warning": False,
           "updated_at": dt.datetime.now(dt.timezone.utc),
           "updated_by_run_id": "init"}

    storage.merge_rows(cfg, "policy_state", [row], ["policy_name", "policy_version"])
    log("policy_state_initialised", champion_id=cfg["starting_champion_id"])
    return row


def write_state(cfg, state, run_id):
    row = {"policy_name": cfg["policy_name"],
           "policy_version": cfg["policy_version"],
           "active_champion_id": state["active_champion_id"],
           "active_action_type": state["active_action_type"],
           "active_model_uri": state["active_model_uri"],
           "active_reference_uri": state["active_reference_uri"],
           "persistence_count": int(state["persistence_count"]),
           "last_promotion_month": state["last_promotion_month"],
           "last_evaluation_month": state["last_evaluation_month"],
           "previous_cohort_warning": bool(state["previous_cohort_warning"]),
           "updated_at": dt.datetime.now(dt.timezone.utc),
           "updated_by_run_id": run_id}

    # MERGE on the natural key, so a repeated run overwrites rather than appends
    storage.merge_rows(cfg, "policy_state", [row], ["policy_name", "policy_version"])
    log("policy_state_updated", **{k: row[k] for k in
                                   ["active_champion_id", "persistence_count",
                                    "last_promotion_month", "last_evaluation_month",
                                    "previous_cohort_warning"]})


def save_champion(cfg, champion_state):
    """Persist a promoted champion and its frozen monitoring reference.

    policy_version is part of the path, so a test replay and a cloud replay can
    never overwrite each other's promoted champion artefacts.
    """
    champion_id = champion_state["champion_id"]
    prefix = f"{cfg['gcs']['champions_prefix']}{cfg['policy_version']}/{champion_id}"

    with tempfile.TemporaryDirectory() as tmp:
        model_path = os.path.join(tmp, "model.joblib")
        reference_path = os.path.join(tmp, "reference.joblib")

        joblib.dump({"champion_id": champion_id,
                     "action_type": champion_state["action_type"],
                     "deployment_month": champion_state["deployment_month"],
                     "underlying_model": champion_state["underlying_model"],
                     "active_recalibration_map": champion_state["active_recalibration_map"]},
                    model_path)

        joblib.dump(champion_state["reference"], reference_path)

        model_uri = storage.upload(cfg, model_path, f"{prefix}/model.joblib")
        reference_uri = storage.upload(cfg, reference_path, f"{prefix}/reference.joblib")

    return model_uri, reference_uri


def load_champion(cfg, model_uri, reference_uri):
    bundle = storage.read_uri(cfg, model_uri)
    reference = storage.read_uri(cfg, reference_uri)

    return {"champion_id": bundle["champion_id"],
            "underlying_model": bundle["underlying_model"],
            "active_recalibration_map": bundle["active_recalibration_map"],
            "deployment_month": bundle["deployment_month"],
            "action_type": bundle["action_type"],
            "reference": reference}


def months_between(later, earlier):
    if earlier is None or (isinstance(earlier, float) and pd.isna(earlier)):
        return None
    return (pd.Period(str(later), freq="M") - pd.Period(str(earlier), freq="M")).n
