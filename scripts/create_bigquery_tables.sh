#!/usr/bin/env bash
# Create one prototype dataset and its ten tables.
# The same script creates both environments:
#
#   PROJECT=my-project DATASET=drift_prototype_test  REGION=europe-west2 \
#     bash scripts/create_bigquery_tables.sh
#
#   PROJECT=my-project DATASET=drift_prototype_cloud REGION=europe-west2 \
#     bash scripts/create_bigquery_tables.sh
#
# Run from the package root, so sql/schemas.sql resolves.

set -euo pipefail

PROJECT="${PROJECT:?export PROJECT=your-project-id}"
DATASET="${DATASET:?export DATASET=drift_prototype_test or drift_prototype_cloud}"
REGION="${REGION:-europe-west2}"

echo "Creating dataset ${PROJECT}.${DATASET} in ${REGION}"

sed -e "s/PROJECT_ID/${PROJECT}/g" \
    -e "s/DATASET_ID/${DATASET}/g" \
    -e "s/REGION_ID/${REGION}/g" \
    sql/schemas.sql \
  | bq --location="${REGION}" --project_id="${PROJECT}" query --use_legacy_sql=false

bq ls --project_id="${PROJECT}" "${DATASET}"
