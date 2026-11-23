-- Minimum BigQuery schema for the drift-aware MLOps prototype.
--
-- Placeholders substituted by scripts/create_bigquery_tables.sh:
--   PROJECT_ID  Google Cloud project
--   DATASET_ID  drift_prototype_test or drift_prototype_cloud
--   REGION_ID   dataset location, e.g. europe-west2
--
-- Ten tables. In-stream versus tail is carried by in_stream_decision on
-- policy_decisions and derivable everywhere else from decision_month.

CREATE SCHEMA IF NOT EXISTS `PROJECT_ID.DATASET_ID`
  OPTIONS (location = "REGION_ID");

-- 1. Frozen deployed probabilities. One row per scored loan; in-stream only,
--    since tail months score nothing. target_bad_loan stays NULL until
--    label_available_month arrives, making the label delay visible in the data.
--    NB05 equivalent: deployed[month]["scores"].
CREATE TABLE IF NOT EXISTS `PROJECT_ID.DATASET_ID.prototype_scores` (
  policy_name STRING NOT NULL,
  policy_version STRING NOT NULL,
  monitor_month STRING NOT NULL,
  row_index INT64 NOT NULL,
  scoring_champion_id STRING NOT NULL,
  predicted_bad_loan_probability FLOAT64 NOT NULL,
  label_available_month STRING NOT NULL,
  target_bad_loan INT64,
  scored_at TIMESTAMP,
  run_id STRING
)
CLUSTER BY policy_name, monitor_month;

-- 2. Immediate monitoring signals. One row per scoring cohort (36 rows).
--    NB05 equivalent: immediate[month].
CREATE TABLE IF NOT EXISTS `PROJECT_ID.DATASET_ID.monitoring_metrics` (
  policy_name STRING NOT NULL,
  policy_version STRING NOT NULL,
  monitor_month STRING NOT NULL,
  scoring_champion_id STRING NOT NULL,
  `rows` INT64,
  mean_predicted_probability FLOAT64,
  max_feature_psi FLOAT64,
  mean_feature_psi FLOAT64,
  supported_feature_drift_count INT64,
  supported_high_drift_count INT64,
  feature_drift_trigger BOOL,
  prediction_jsd FLOAT64,
  prediction_drift_trigger BOOL,
  unseen_category_count INT64,
  unseen_category_detail STRING,
  mean_feature_missing_rate FLOAT64,
  data_quality_flag BOOL,
  warning BOOL,
  computed_at TIMESTAMP,
  run_id STRING
)
CLUSTER BY policy_name, monitor_month;

-- 3. Policy A shadow requests. One row per scoring cohort (36 rows, 33 with
--    retraining_request = TRUE). No model is fitted and no champion is kept.
--    NB05 equivalent: policy_A_requests.csv.
CREATE TABLE IF NOT EXISTS `PROJECT_ID.DATASET_ID.policy_a_requests` (
  policy_name STRING NOT NULL,
  policy_version STRING NOT NULL,
  monitor_month STRING NOT NULL,
  feature_drift_trigger BOOL,
  retraining_request BOOL,
  request_reason STRING,
  recorded_at TIMESTAMP,
  run_id STRING
);

-- 4. Delayed outcome-aware evidence. One row per matured cohort (36 rows,
--    the last five written by tail decision months).
--    NB05 equivalent: policy_c["evidence"].
CREATE TABLE IF NOT EXISTS `PROJECT_ID.DATASET_ID.delayed_evidence` (
  policy_name STRING NOT NULL,
  policy_version STRING NOT NULL,
  decision_month STRING NOT NULL,
  monitor_month STRING NOT NULL,
  label_available_month STRING,
  scored_by STRING,
  relevant_to_current_champion BOOL,
  `rows` INT64,
  actual_bad_loan_rate FLOAT64,
  mean_predicted_probability FLOAT64,
  calibration_gap FLOAT64,
  absolute_calibration_gap FLOAT64,
  brier_score FLOAT64,
  log_loss FLOAT64,
  roc_auc FLOAT64,
  average_precision FLOAT64,
  brier_change FLOAT64,
  log_loss_change FLOAT64,
  auc_drop FLOAT64,
  performance_level STRING,
  score_bin_mean_absolute_shift FLOAT64,
  score_bin_signed_shift FLOAT64,
  residual_mean_abs_shift FLOAT64,
  concept_level STRING,
  concept_shift_type STRING,
  calibration_slope FLOAT64,
  calibration_slope_deviation FLOAT64,
  target_rate_change FLOAT64,
  target_drift_level STRING,
  target_drift_trigger BOOL,
  feature_drift_trigger BOOL,
  prediction_drift_trigger BOOL,
  data_quality_flag BOOL,
  warning BOOL,
  computed_at TIMESTAMP,
  run_id STRING
)
CLUSTER BY policy_name, decision_month;

-- 5. Policy decisions. One row per decision month 2014-06 to 2017-05 (36 rows).
--    in_stream_decision separates the 31 in-stream from the 5 tail decisions.
--    NB05 equivalent: policy_C_decisions.csv, column for column.
CREATE TABLE IF NOT EXISTS `PROJECT_ID.DATASET_ID.policy_decisions` (
  policy_name STRING NOT NULL,
  policy_version STRING NOT NULL,
  decision_month STRING NOT NULL,
  monitor_month STRING,
  in_stream_decision BOOL,
  champion_id STRING,
  scored_by STRING,
  evidence_relevant BOOL,
  performance_level STRING,
  concept_level STRING,
  target_drift_level STRING,
  residual_mean_abs_shift FLOAT64,
  persistence_count INT64,
  persistence_satisfied BOOL,
  fired_rule STRING,
  candidate_trigger BOOL,
  suppressed_by_deployment_cooldown BOOL,
  suppressed_by_candidate_evaluation_holdoff BOOL,
  insufficient_labelled_data BOOL,
  candidate_evaluated BOOL,
  intended_action STRING,
  action_reason STRING,
  final_action STRING,
  candidate_promoted BOOL,
  promotion_type STRING,
  policy_reason STRING,
  decided_at TIMESTAMP,
  run_id STRING
)
CLUSTER BY policy_name, decision_month;

-- 6. Candidate attempts and gate outcomes (5 rows expected for Policy C).
--    NB05 equivalent: policy_C_candidate_attempts.csv.
CREATE TABLE IF NOT EXISTS `PROJECT_ID.DATASET_ID.candidate_attempts` (
  policy_name STRING NOT NULL,
  policy_version STRING NOT NULL,
  decision_month STRING NOT NULL,
  attempt_index INT64 NOT NULL,
  trigger_monitor_month STRING,
  candidate_type STRING,
  gate_design STRING,
  gate_name STRING,
  current_champion STRING,
  candidate_id STRING,
  simulation_training_start_month STRING,
  simulation_training_end_month STRING,
  historical_training_rows INT64,
  simulation_training_rows INT64,
  total_training_rows INT64,
  validation_start_month STRING,
  validation_end_month STRING,
  validation_rows INT64,
  champion_brier FLOAT64,
  candidate_brier FLOAT64,
  brier_improvement FLOAT64,
  brier_required_min FLOAT64,
  brier_criterion_passed BOOL,
  champion_absolute_calibration_gap FLOAT64,
  candidate_absolute_calibration_gap FLOAT64,
  calibration_improvement FLOAT64,
  calibration_required_min FLOAT64,
  calibration_criterion_passed BOOL,
  champion_log_loss FLOAT64,
  candidate_log_loss FLOAT64,
  log_loss_change FLOAT64,
  log_loss_criterion_passed BOOL,
  champion_roc_auc FLOAT64,
  candidate_roc_auc FLOAT64,
  auc_change FLOAT64,
  auc_guardrail_passed BOOL,
  validation_size_passed BOOL,
  deployment_gate_passed BOOL,
  failed_criteria STRING,
  evaluated_at TIMESTAMP,
  run_id STRING
)
CLUSTER BY policy_name, decision_month;

-- 7. Promotions. The only mapping from champion_id to its deployed artefacts,
--    which is what lets superseded-champion evidence be evaluated against the
--    correct reference. NB05 equivalent: policy_C_promotions.csv + champion_registry.
CREATE TABLE IF NOT EXISTS `PROJECT_ID.DATASET_ID.policy_promotions` (
  policy_name STRING NOT NULL,
  policy_version STRING NOT NULL,
  champion_id STRING NOT NULL,
  decision_month STRING,
  trigger_monitor_month STRING,
  action_type STRING,
  validation_start_month STRING,
  validation_end_month STRING,
  model_uri STRING,
  reference_uri STRING,
  recorded_at TIMESTAMP,
  run_id STRING
);

-- 8. Policy state. Exactly one row per policy_name + policy_version.
--    NB05 equivalent: the in-memory state dictionary and champion variable.
CREATE TABLE IF NOT EXISTS `PROJECT_ID.DATASET_ID.policy_state` (
  policy_name STRING NOT NULL,
  policy_version STRING NOT NULL,
  active_champion_id STRING NOT NULL,
  active_action_type STRING,
  active_model_uri STRING NOT NULL,
  active_reference_uri STRING NOT NULL,
  persistence_count INT64,
  last_promotion_month STRING,
  last_evaluation_month STRING,
  previous_cohort_warning BOOL,
  updated_at TIMESTAMP,
  updated_by_run_id STRING
);

-- 9. Execution metrics. One row per job execution (41 for the replay, plus any
--    duplicate skips). Carries the runtime evidence for the cost section.
CREATE TABLE IF NOT EXISTS `PROJECT_ID.DATASET_ID.prototype_runs` (
  run_id STRING NOT NULL,
  policy_name STRING,
  policy_version STRING,
  decision_month STRING,
  started_at TIMESTAMP,
  ended_at TIMESTAMP,
  job_duration_seconds FLOAT64,
  monitoring_duration_seconds FLOAT64,
  rows_scored INT64,
  candidate_fit_started BOOL,
  candidate_fit_duration_seconds FLOAT64,
  candidate_type STRING,
  candidate_promoted BOOL,
  outcome STRING,
  image_digest STRING,
  cpu STRING,
  memory STRING
);

-- 10. Parity results, written by scripts/run_parity_check.py so the acceptance
--     evidence is queryable next to everything it validates.
CREATE TABLE IF NOT EXISTS `PROJECT_ID.DATASET_ID.parity_results` (
  policy_name STRING,
  policy_version STRING,
  parity_scope STRING,          -- 'cohort' or 'decision'
  monitor_month STRING,
  decision_month STRING,
  local_value STRING,
  cloud_value STRING,
  comparison STRING,
  max_metric_difference FLOAT64,
  metrics_within_tolerance BOOL,
  match BOOL,
  checked_at TIMESTAMP
);
