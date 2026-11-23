#!/usr/bin/env bash
# Upload the existing Notebook 02/03/04/05 artefacts to Cloud Storage.
# Run from the repository root with the notebook directory layout intact.
# Nothing here reprocesses or regenerates anything: it is a copy.

set -euo pipefail

BUCKET="${BUCKET:?set BUCKET, e.g. export BUCKET=my-project-mlops-prototype}"
REGION="${REGION:-europe-west2}"

gcloud storage buckets create "gs://${BUCKET}" --location="${REGION}" 2>/dev/null || true

# Processed cohort data (Notebook 02). Uploaded whole, not split by month, so the
# categorical dtypes and level sets stay identical to the local run.
gcloud storage cp data/processed/historical_2013_clean.parquet "gs://${BUCKET}/data/processed/"
gcloud storage cp data/processed/simulation_2014_2016_clean.parquet "gs://${BUCKET}/data/processed/"

# Champion model and metadata (Notebook 03)
gcloud storage cp artifacts/models/champion_xgboost_model.joblib "gs://${BUCKET}/models/"
gcloud storage cp artifacts/models/champion_model_metadata.json  "gs://${BUCKET}/models/"

# Monitoring baselines and reference bins (Notebook 03)
for f in champion_feature_cols.csv \
         monitoring_feature_cols.csv \
         monitoring_numeric_cols.csv \
         monitoring_categorical_cols.csv \
         numeric_psi_cols.csv \
         discrete_monitoring_cols.csv \
         concept_drift_baseline.csv \
         feature_drift_numeric_bin_edges.joblib \
         prediction_drift_score_bin_edges.joblib
do
  gcloud storage cp "artifacts/baselines/${f}" "gs://${BUCKET}/baselines/"
done

# Scored 2013 calibration cohort: the original champion's monitoring reference
gcloud storage cp data/inference/calibration_2013_scored.parquet "gs://${BUCKET}/inference/"

# Threshold and policy metadata (Notebooks 04 and 05).
# The job reads every threshold from these files, so nothing is retyped in the cloud.
gcloud storage cp artifacts/monitoring/monitoring_threshold_metadata.json "gs://${BUCKET}/baselines/"
gcloud storage cp artifacts/retraining_policy/final_policy_metadata.json  "gs://${BUCKET}/baselines/"

# Local results used only as parity targets, never as job inputs
for f in policy_A_requests.csv \
         policy_C_decisions.csv \
         policy_C_candidate_attempts.csv \
         policy_C_promotions.csv \
         managed_performance_C.csv
do
  gcloud storage cp "artifacts/retraining_policy/${f}" "gs://${BUCKET}/local_results/"
done

# Notebook 04 monitoring summaries, parity targets for the immediate signals
gcloud storage cp artifacts/monitoring/monthly_feature_drift_summary.csv "gs://${BUCKET}/local_results/"
gcloud storage cp artifacts/monitoring/monthly_prediction_drift.csv      "gs://${BUCKET}/local_results/"

echo "Prototype input package uploaded to gs://${BUCKET}"
gcloud storage ls -r "gs://${BUCKET}" | head -60
