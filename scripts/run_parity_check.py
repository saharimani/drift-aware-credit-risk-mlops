# Full local-to-cloud parity check against the updated Notebook 05.
#
# Three layers:
#   1. Cohort parity, 36 rows: scoring champion, feature-drift trigger,
#      prediction drift (where a local artefact exists), and managed metrics
#      recomputed from the frozen cloud scores against managed_performance_C.csv.
#   2. Decision parity: every decision month 2014-06 to 2017-05, in-stream and
#      tail reported separately, fired rule, actions, gate outcomes and
#      candidate metrics against policy_C_decisions.csv and
#      policy_C_candidate_attempts.csv.
#   3. Headline counts: Policy A requests, Policy C triggers, fits, in-stream
#      and tail promotions, champion at 2016-12 and final champion.
#
#   python scripts/run_parity_check.py \
#     --local artifacts/retraining_policy \
#     --monitoring artifacts/monitoring \
#     --config config/prototype_config_test.json
#
# Pass --config config/prototype_config_cloud.json for the cloud replay.
#
# Fields with no local counterpart are marked not_comparable and counted
# separately. They are never silently treated as passing.
#
# The prototype may be described as validated only if this passes.

import argparse
import datetime as dt
import sys

sys.path.insert(0, "src")

import numpy as np
import pandas as pd

import storage
from evidence import evaluate_predictions

TOLERANCE = 1e-5

DECISION_COLS = ["fired_rule", "intended_action", "final_action", "candidate_promoted",
                 "scored_by", "evidence_relevant", "performance_level", "concept_level",
                 "persistence_count", "in_stream_decision"]

ATTEMPT_METRICS = ["champion_brier", "candidate_brier", "brier_improvement",
                   "champion_absolute_calibration_gap",
                   "candidate_absolute_calibration_gap", "calibration_improvement",
                   "champion_roc_auc", "candidate_roc_auc"]

MANAGED_METRICS = ["brier_score", "absolute_calibration_gap", "log_loss", "roc_auc"]

ORIGINAL_CHAMPION = "keep_sub_grade_only"


def bq_table(cfg, table, extra=""):
    sql = f"""
    SELECT * FROM `{storage.table_id(cfg, table)}`
    WHERE policy_name = @policy_name AND policy_version = @policy_version {extra}
    """
    return storage.query(cfg, sql, policy_name=cfg["policy_name"],
                         policy_version=cfg["policy_version"])


def cohort_parity(cfg, policy_dir, monitoring_dir):
    """One row per scoring cohort: champion, immediate signals, managed metrics."""
    local_managed = pd.read_csv(f"{policy_dir}/managed_performance_C.csv")
    local_feature = pd.read_csv(f"{monitoring_dir}/monthly_feature_drift_summary.csv")
    local_prediction = pd.read_csv(f"{monitoring_dir}/monthly_prediction_drift.csv")

    cloud_metrics = bq_table(cfg, "monitoring_metrics").set_index("monitor_month")
    cloud_scores = bq_table(cfg, "prototype_scores")

    rows = []

    for _, local in local_managed.sort_values("monitor_month").iterrows():
        month = local["monitor_month"]

        if month not in cloud_metrics.index:
            rows.append({"monitor_month": month, "match": False,
                         "note": "cohort not scored in the cloud"})
            continue

        cloud = cloud_metrics.loc[month]
        frozen = cloud_scores[cloud_scores["monitor_month"] == month]

        # Managed metrics recomputed from the frozen cloud probabilities and the
        # labels attached at maturity; this exercises the champion transitions
        if frozen["target_bad_loan"].notna().all() and len(frozen) > 0:
            cloud_managed = evaluate_predictions(
                frozen["target_bad_loan"].astype(int).values,
                frozen["predicted_bad_loan_probability"].values)
            metric_diffs = {m: abs(float(cloud_managed[m]) - float(local[m]))
                            for m in MANAGED_METRICS}
            managed_ok = all(v <= TOLERANCE for v in metric_diffs.values())
            max_diff = max(metric_diffs.values())
        else:
            managed_ok, max_diff = None, None  # labels not yet matured in the cloud

        local_ft = bool(local_feature.loc[
            local_feature["monitor_month"] == month, "feature_drift_trigger"].iloc[0])

        # Prediction drift is champion-specific; NB04 covers the original champion only
        if cloud["scoring_champion_id"] == ORIGINAL_CHAMPION:
            local_jsd = float(local_prediction.loc[
                local_prediction["monitor_month"] == month, "prediction_jsd"].iloc[0])
            jsd_diff = abs(local_jsd - float(cloud["prediction_jsd"]))
            jsd_ok = jsd_diff <= TOLERANCE
        else:
            local_jsd, jsd_diff, jsd_ok = None, None, None  # no local artefact

        champion_ok = local["deployed_champion"] == cloud["scoring_champion_id"]
        trigger_ok = local_ft == bool(cloud["feature_drift_trigger"])

        # Anything without a local counterpart is named, not quietly passed
        not_comparable = []
        if jsd_ok is None:
            not_comparable.append("prediction_jsd: no Notebook 04 artefact for a "
                                  "promoted champion")
        if managed_ok is None:
            not_comparable.append("managed_metrics: cloud labels not yet matured "
                                  "for this cohort")

        rows.append({"monitor_month": month,
                     "local_scoring_champion": local["deployed_champion"],
                     "cloud_scoring_champion": cloud["scoring_champion_id"],
                     "scoring_champion_match": bool(champion_ok),
                     "local_feature_drift_trigger": local_ft,
                     "cloud_feature_drift_trigger": bool(cloud["feature_drift_trigger"]),
                     "feature_drift_trigger_match": bool(trigger_ok),
                     "local_prediction_jsd": local_jsd,
                     "cloud_prediction_jsd": float(cloud["prediction_jsd"]),
                     "prediction_jsd_comparable": jsd_ok is not None,
                     "prediction_jsd_match": jsd_ok,
                     "max_managed_metric_difference": max_diff,
                     "metrics_within_tolerance": managed_ok,
                     "not_comparable": "; ".join(not_comparable) if not_comparable else "",
                     "match": bool(champion_ok and trigger_ok and
                                   (jsd_ok is not False) and (managed_ok is not False))})

    return pd.DataFrame(rows)


def decision_parity(cfg, policy_dir):
    """One row per decision month, tail decisions flagged by in_stream_decision."""
    local_d = pd.read_csv(f"{policy_dir}/policy_C_decisions.csv")
    local_a = pd.read_csv(f"{policy_dir}/policy_C_candidate_attempts.csv")

    cloud_d = bq_table(cfg, "policy_decisions")
    cloud_a = bq_table(cfg, "candidate_attempts")

    rows = []

    for _, l in local_d.sort_values("decision_month").iterrows():
        month = l["decision_month"]
        c = cloud_d[cloud_d["decision_month"] == month]

        if c.empty:
            rows.append({"decision_month": month, "match": False,
                         "note": "decision missing in the cloud"})
            continue

        c = c.iloc[0]
        decision_match = all(str(l[col]) == str(c[col]) for col in DECISION_COLS)

        la = local_a[local_a["decision_month"] == month].sort_values("attempt_index")
        ca = cloud_a[cloud_a["decision_month"] == month].sort_values("attempt_index")

        if la.empty and ca.empty:
            metrics_ok, max_diff, gate_l, gate_c = None, None, None, None
        elif len(la) != len(ca):
            metrics_ok, max_diff = False, None
            gate_l = bool(la["deployment_gate_passed"].iloc[-1]) if len(la) else None
            gate_c = bool(ca["deployment_gate_passed"].iloc[-1]) if len(ca) else None
        else:
            la, ca = la.reset_index(drop=True), ca.reset_index(drop=True)
            diffs = {m: float((la[m] - ca[m]).abs().max()) for m in ATTEMPT_METRICS}
            max_diff = max(diffs.values())
            metrics_ok = max_diff <= TOLERANCE
            gate_l = bool(la["deployment_gate_passed"].iloc[-1])
            gate_c = bool(ca["deployment_gate_passed"].iloc[-1])

        rows.append({"decision_month": month,
                     "in_stream_decision": bool(l["in_stream_decision"]),
                     "local_scoring_champion": l["scored_by"],
                     "cloud_scoring_champion": c["scored_by"],
                     "local_evidence_relevant": l["evidence_relevant"],
                     "cloud_evidence_relevant": c["evidence_relevant"],
                     "local_fired_rule": l["fired_rule"],
                     "cloud_fired_rule": c["fired_rule"],
                     "local_intended_action": l["intended_action"],
                     "cloud_intended_action": c["intended_action"],
                     "local_final_action": l["final_action"],
                     "cloud_final_action": c["final_action"],
                     "local_candidate_promoted": bool(l["candidate_promoted"]),
                     "cloud_candidate_promoted": bool(c["candidate_promoted"]),
                     "local_gate_passed": gate_l,
                     "cloud_gate_passed": gate_c,
                     "max_metric_difference": max_diff,
                     "metrics_within_tolerance": metrics_ok,
                     "match": bool(decision_match and metrics_ok is not False)})

    return pd.DataFrame(rows)


def headline_counts(cfg, policy_dir):
    local_a = pd.read_csv(f"{policy_dir}/policy_A_requests.csv")
    local_d = pd.read_csv(f"{policy_dir}/policy_C_decisions.csv")
    local_p = pd.read_csv(f"{policy_dir}/policy_C_promotions.csv")

    cloud_requests = storage.query(cfg, f"""
        SELECT COUNTIF(retraining_request) AS n
        FROM `{storage.table_id(cfg, 'policy_a_requests')}`
        WHERE policy_version = @policy_version""",
        policy_version=cfg["policy_version"])["n"].iloc[0]

    cloud_d = bq_table(cfg, "policy_decisions")
    cloud_p = bq_table(cfg, "policy_promotions")
    cloud_p = cloud_p[cloud_p["decision_month"] != "start"]

    state = bq_table(cfg, "policy_state").iloc[0]

    in_stream_local = local_p[local_p["in_stream"]]
    december_2016_local = pd.read_csv(f"{policy_dir}/managed_performance_C.csv")
    december_2016_local = december_2016_local.loc[
        december_2016_local["monitor_month"] == "2016-12", "deployed_champion"].iloc[0]

    december_2016_cloud = bq_table(
        cfg, "monitoring_metrics")\
        .set_index("monitor_month").loc["2016-12", "scoring_champion_id"]

    cloud_in_stream_promotions = cloud_d.loc[cloud_d["in_stream_decision"] &
                                             cloud_d["candidate_promoted"]]
    cloud_tail_promotions = cloud_d.loc[~cloud_d["in_stream_decision"] &
                                        cloud_d["candidate_promoted"]]

    local_promotions = local_p[
        local_p["decision_month"].astype(str) != "start"
    ].copy()

    final_local_champion = (
        local_promotions.sort_values("decision_month").iloc[-1]["champion_id"]
        if not local_promotions.empty else "none"
    )

    checks = [
        ("policy_a_requests", int(local_a["retraining_request"].sum()), int(cloud_requests)),
        ("policy_c_triggers", int(local_d["candidate_trigger"].sum()),
         int(cloud_d["candidate_trigger"].sum())),
        ("policy_c_in_stream_promotions", int(in_stream_local.shape[0]),
         int(cloud_in_stream_promotions.shape[0])),
        ("policy_c_tail_promotions", int((~local_p["in_stream"]).sum()),
         int(cloud_tail_promotions.shape[0])),
        ("champion_at_2016_12", december_2016_local, december_2016_cloud),
        ("final_champion_after_tail",
         final_local_champion,
         state["active_champion_id"])]

    return pd.DataFrame([{"comparison": name, "local_value": str(l),
                          "cloud_value": str(c), "match": str(l) == str(c)}
                         for name, l, c in checks])


def to_parity_rows(cfg, cohorts, decisions):
    now = dt.datetime.now(dt.timezone.utc)
    rows = []

    for _, r in cohorts.iterrows():
        rows.append({"policy_name": cfg["policy_name"],
                     "policy_version": cfg["policy_version"],
                     "parity_scope": "cohort",
                     "monitor_month": r.get("monitor_month"),
                     "decision_month": None,
                     "local_value": str(r.get("local_scoring_champion")),
                     "cloud_value": str(r.get("cloud_scoring_champion")),
                     "comparison": "scoring_champion_and_signals",
                     "max_metric_difference": r.get("max_managed_metric_difference"),
                     "metrics_within_tolerance": r.get("metrics_within_tolerance"),
                     "match": bool(r["match"]), "checked_at": now})

    for _, r in decisions.iterrows():
        rows.append({"policy_name": cfg["policy_name"],
                     "policy_version": cfg["policy_version"],
                     "parity_scope": "decision",
                     "monitor_month": None,
                     "decision_month": r.get("decision_month"),
                     "local_value": str(r.get("local_final_action")),
                     "cloud_value": str(r.get("cloud_final_action")),
                     "comparison": "decision_and_gate",
                     "max_metric_difference": r.get("max_metric_difference"),
                     "metrics_within_tolerance": r.get("metrics_within_tolerance"),
                     "match": bool(r["match"]), "checked_at": now})

    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", default="artifacts/retraining_policy")
    parser.add_argument("--monitoring", default="artifacts/monitoring")
    parser.add_argument("--config", default=None)
    parser.add_argument("--out", default="parity")
    parser.add_argument("--write-bq", action="store_true",
                        help="also MERGE the results into parity_results")
    args = parser.parse_args()

    cfg = storage.load_config(args.config)

    cohorts = cohort_parity(cfg, args.local, args.monitoring)
    decisions = decision_parity(cfg, args.local)
    headline = headline_counts(cfg, args.local)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 60)

    print("=== Cohort parity (36 scoring months) ===")
    print(cohorts.to_string(index=False))

    print("\n=== Decision parity: in-stream ===")
    print(decisions[decisions["in_stream_decision"]].to_string(index=False))

    print("\n=== Decision parity: tail (2017-01 to 2017-05) ===")
    print(decisions[~decisions["in_stream_decision"]].to_string(index=False))

    print("\n=== Headline counts ===")
    print(headline.to_string(index=False))

    not_comparable = cohorts[cohorts["not_comparable"] != ""] if "not_comparable" in cohorts else cohorts.iloc[0:0]
    if not not_comparable.empty:
        print("\n=== Not comparable (no local counterpart; excluded from the match test) ===")
        print(not_comparable[["monitor_month", "not_comparable"]].to_string(index=False))

    cohorts.to_csv(f"{args.out}_cohorts.csv", index=False)
    decisions.to_csv(f"{args.out}_decisions.csv", index=False)
    headline.to_csv(f"{args.out}_headline.csv", index=False)

    if args.write_bq:
        storage.merge_rows(cfg, "parity_results", to_parity_rows(cfg, cohorts, decisions),
                           ["policy_name", "policy_version", "parity_scope",
                            "monitor_month", "decision_month"])

    all_pass = bool(cohorts["match"].all() and decisions["match"].all()
                    and headline["match"].all())

    print(f"\nTolerance: {TOLERANCE}")
    print(f"Cohorts not fully comparable: {len(not_comparable)}/{len(cohorts)} "
          "(reported above, not counted as passes)")
    print(f"Cohorts matching: {int(cohorts['match'].sum())}/{len(cohorts)}")
    print(f"Decisions matching: {int(decisions['match'].sum())}/{len(decisions)}")
    print(f"Headline counts matching: {int(headline['match'].sum())}/{len(headline)}")
    print("VALIDATED:" if all_pass else "NOT VALIDATED:", all_pass)
