#!/usr/bin/env bash
# Build the linux/amd64 image and deploy one Cloud Run Job for the chosen
# environment. Run from the package root, block by block, capturing console
# output as dissertation evidence.
#
#   PROJECT=my-project \
#   BUCKET=my-bucket \
#   DATASET=drift_prototype_cloud \
#   POLICY_VERSION=v1_primary_gate_delay5_cloud \
#   REGION=europe-west2 \
#   PYTHON_VERSION=3.12 \
#     bash scripts/deploy_cloud_run_job.sh
#
# Deploy the exact image that passed the local containerised test replay: build
# once, tag once, reuse the tag. Do not rebuild between test and cloud.

set -euo pipefail

PROJECT="${PROJECT:?export PROJECT=your-project-id}"
BUCKET="${BUCKET:?export BUCKET=your-bucket}"
DATASET="${DATASET:?export DATASET=drift_prototype_cloud}"
POLICY_VERSION="${POLICY_VERSION:?export POLICY_VERSION=v1_primary_gate_delay5_cloud}"
REGION="${REGION:-europe-west2}"
POLICY_NAME="${POLICY_NAME:-policy_c}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

REPO="${REPO:-drift-prototype}"
JOB="${JOB:-monitor-and-decide}"
SA="${SA:-drift-prototype-sa}"
IMAGE_TAG="${IMAGE_TAG:-v1}"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/${JOB}"
SA_EMAIL="${SA}@${PROJECT}.iam.gserviceaccount.com"
CONFIG_IN_IMAGE="${CONFIG_IN_IMAGE:-/app/config/prototype_config_cloud.json}"
CREATE_SCHEDULER="${CREATE_SCHEDULER:-false}"

# ---------------------------------------------------------------- 1. services
gcloud services enable run.googleapis.com bigquery.googleapis.com \
  storage.googleapis.com artifactregistry.googleapis.com \
  cloudscheduler.googleapis.com cloudbuild.googleapis.com --project="${PROJECT}"

# ---------------------------------------------------- 2. registry and identity
gcloud artifacts repositories describe "${REPO}" \
  --location="${REGION}" --project="${PROJECT}" >/dev/null 2>&1 \
  || gcloud artifacts repositories create "${REPO}" \
       --repository-format=docker --location="${REGION}" --project="${PROJECT}"

gcloud iam service-accounts describe "${SA_EMAIL}" --project="${PROJECT}" >/dev/null 2>&1 \
  || gcloud iam service-accounts create "${SA}" --project="${PROJECT}"

for ROLE in roles/storage.objectAdmin roles/bigquery.dataEditor roles/bigquery.jobUser
do
  gcloud projects add-iam-policy-binding "${PROJECT}" \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="${ROLE}" --condition=None >/dev/null
done

# ---------------------------------------------------------------- 3. the image
# The Dockerfile is pinned to the Python version established from Notebook 05.
gcloud builds submit . \
  --project="${PROJECT}" \
  --tag "${IMAGE}:${IMAGE_TAG}"

# If the local build used a non-default Python version, prefer building locally
# and pushing the tested image instead:
#   docker build --platform linux/amd64 --build-arg PYTHON_VERSION="${PYTHON_VERSION}" \
#     -t "${IMAGE}:${IMAGE_TAG}" .
#   gcloud auth configure-docker "${REGION}-docker.pkg.dev"
#   docker push "${IMAGE}:${IMAGE_TAG}"

# ------------------------------------------------------------------ 4. the job
# max-retries=0: the replay script owns retry decisions, and a silent platform
# retry would blur the runtime measurements and re-enter a partially written month.
JOB_ENV="PROJECT_ID=${PROJECT},BUCKET=${BUCKET},DATASET=${DATASET}"
JOB_ENV="${JOB_ENV},POLICY_NAME=${POLICY_NAME},POLICY_VERSION=${POLICY_VERSION}"
JOB_ENV="${JOB_ENV},PROTOTYPE_CONFIG=${CONFIG_IN_IMAGE},JOB_CPU=4,JOB_MEMORY=16Gi"

if gcloud run jobs describe "${JOB}" --region="${REGION}" --project="${PROJECT}" >/dev/null 2>&1
then
  gcloud run jobs update "${JOB}" \
    --image="${IMAGE}:${IMAGE_TAG}" --region="${REGION}" --project="${PROJECT}" \
    --service-account="${SA_EMAIL}" \
    --cpu=4 --memory=16Gi --task-timeout=7200s --max-retries=0 \
    --set-env-vars="${JOB_ENV}"
else
  gcloud run jobs create "${JOB}" \
    --image="${IMAGE}:${IMAGE_TAG}" --region="${REGION}" --project="${PROJECT}" \
    --service-account="${SA_EMAIL}" \
    --cpu=4 --memory=16Gi --task-timeout=7200s --max-retries=0 \
    --set-env-vars="${JOB_ENV}"
fi

# ------------------------------------- 5. let the scheduler identity invoke it
gcloud run jobs add-iam-policy-binding "${JOB}" \
  --region="${REGION}" \
  --project="${PROJECT}" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/run.invoker"

# ------------------------------------------------------------ 6. smoke test
gcloud run jobs execute "${JOB}" --region="${REGION}" --project="${PROJECT}" \
  --args="--month=2014-01,--policy=${POLICY_NAME}" --wait

# ------------------------------------------------------- 7. Cloud Scheduler
if [[ "${CREATE_SCHEDULER}" == "true" ]]
then
  # Demonstration evidence of managed scheduled execution. It does not replace the
  # historical replay script, which drives all 41 months.
  SCHEDULER_JOB="${SCHEDULER_JOB:-monthly-drift-decision}"
  RUN_URI="https://run.googleapis.com/v2/projects/${PROJECT}/locations/${REGION}/jobs/${JOB}:run"

  gcloud scheduler jobs describe "${SCHEDULER_JOB}" \
    --location="${REGION}" --project="${PROJECT}" >/dev/null 2>&1 \
    && SCHEDULER_ACTION=update || SCHEDULER_ACTION=create

  gcloud scheduler jobs "${SCHEDULER_ACTION}" http "${SCHEDULER_JOB}" \
    --project="${PROJECT}" \
    --location="${REGION}" \
    --schedule="0 6 5 * *" \
    --time-zone="Europe/London" \
    --uri="${RUN_URI}" \
    --http-method=POST \
    --oauth-service-account-email="${SA_EMAIL}" \
    --message-body='{"overrides":{"containerOverrides":[{"args":["--month=2015-04"]}]}}'

  # Trigger once, after the replay has already completed 2015-04, so the scheduled
  # invocation also demonstrates the duplicate-run skip:
  #   gcloud scheduler jobs run "${SCHEDULER_JOB}" --location="${REGION}" --project="${PROJECT}"
else
  echo "Skipping Cloud Scheduler creation (CREATE_SCHEDULER=false)."
fi

# ---------------------------------------------------------- 8. optional registry
# gcloud ai models upload --region="${REGION}" --project="${PROJECT}" \
#   --display-name="champion_keep_sub_grade_only" \
#   --container-image-uri="${REGION}-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-7:latest" \
#   --artifact-uri="gs://${BUCKET}/champions/${POLICY_VERSION}/keep_sub_grade_only/"

echo
echo "Deployed ${JOB} in ${REGION} from ${IMAGE}:${IMAGE_TAG}"
echo "  dataset        ${DATASET}"
echo "  policy_version ${POLICY_VERSION}"
echo "  config         ${CONFIG_IN_IMAGE}"
