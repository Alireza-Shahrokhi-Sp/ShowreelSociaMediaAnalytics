#!/usr/bin/env bash
# run_thumbnail_gcp.sh
#
# Launches the thumbnail feature extraction pipeline on GCP.
# Only steps that require your laptop:
#   - uploading thumbnails from local disk to GCS (once)
#   - submitting the Cloud Build job (returns in seconds)
#
# After this script exits you can close your laptop.
# Cloud Build builds the container on GCP (~45–90 min first time, faster on re-runs).
# Then 50 Cloud Run Job tasks execute in parallel (~15 min for 11,801 images).
#
# Prerequisites (one-time)
# ─────────────────────────
#   gcloud auth login
#   gcloud auth configure-docker us-central1-docker.pkg.dev
#   gcloud config set project project-10b142ae-d53f-4f87-81e
#
# Usage (from the repo root — the folder containing reza/)
# ─────────────────────────────────────────────────────────
#   bash reza/run_thumbnail_gcp.sh
#
# Skip uploading thumbnails on subsequent runs:
#   bash reza/run_thumbnail_gcp.sh --skip-upload
#
# Override number of parallel tasks (default 50):
#   bash reza/run_thumbnail_gcp.sh --tasks 100
#
# Monitor progress:
#   https://console.cloud.google.com/cloud-build/builds?project=project-10b142ae-d53f-4f87-81e
#   https://console.cloud.google.com/run/jobs?project=project-10b142ae-d53f-4f87-81e
#
# Download results after all tasks finish:
#   python reza/Data_Cleaned/merge_thumbnail_features.py

set -euo pipefail

PROJECT="project-10b142ae-d53f-4f87-81e"
BUCKET="socialmediaanalyticsproject"
REGION="us-central1"
TASKS="50"

# Default thumbnail source — adjust if your local thumbnails are elsewhere.
# Expects files named {video_id}.jpg
THUMB_LOCAL_DIR="reza/Data_Cleaned/thumbnails"

SKIP_UPLOAD=false
for arg in "$@"; do
  [[ $arg == "--skip-upload" ]] && SKIP_UPLOAD=true
  [[ $arg == --tasks=* ]]       && TASKS="${arg#*=}"
  [[ $arg == "--tasks" ]]       && shift && TASKS="$1"
done

# ── 1. Upload thumbnails ──────────────────────────────────────────────────────

if [ "$SKIP_UPLOAD" = false ]; then
  if [ ! -d "${THUMB_LOCAL_DIR}" ]; then
    echo "ERROR: Local thumbnail directory '${THUMB_LOCAL_DIR}' not found."
    echo "       Either set THUMB_LOCAL_DIR at the top of this script or use --skip-upload"
    echo "       if the thumbnails are already in gs://${BUCKET}/thumbnails/"
    exit 1
  fi
  THUMB_COUNT=$(find "${THUMB_LOCAL_DIR}" -name "*.jpg" | wc -l)
  echo "=== Uploading ${THUMB_COUNT} thumbnails to gs://${BUCKET}/thumbnails/ ==="
  gsutil -m cp "${THUMB_LOCAL_DIR}"/*.jpg "gs://${BUCKET}/thumbnails/"
  echo "Upload complete."
else
  echo "=== Skipping thumbnail upload (--skip-upload) ==="
fi

# ── 2. Enable required APIs (idempotent) ──────────────────────────────────────

echo "=== Enabling GCP APIs ==="
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  --project "${PROJECT}" --quiet

# ── 3. Ensure Artifact Registry repo exists ───────────────────────────────────

gcloud artifacts repositories describe showreel \
  --location="${REGION}" --project="${PROJECT}" --quiet 2>/dev/null \
  || gcloud artifacts repositories create showreel \
       --repository-format=docker \
       --location="${REGION}" \
       --project="${PROJECT}" \
       --quiet

# ── 4. Grant Cloud Build SA permission to deploy Cloud Run Jobs ───────────────

CLOUDBUILD_SA="$(gcloud projects describe "${PROJECT}" \
  --format='value(projectNumber)')@cloudbuild.gserviceaccount.com"

gcloud projects add-iam-policy-binding "${PROJECT}" \
  --member="serviceAccount:${CLOUDBUILD_SA}" \
  --role="roles/run.admin" --quiet

gcloud projects add-iam-policy-binding "${PROJECT}" \
  --member="serviceAccount:${CLOUDBUILD_SA}" \
  --role="roles/iam.serviceAccountUser" --quiet

# ── 5. Submit Cloud Build — async, returns immediately ────────────────────────

echo "=== Submitting Cloud Build (async) with --tasks=${TASKS} ==="
gcloud builds submit . \
  --config  "reza/cloudbuild-thumbnail.yaml" \
  --project "${PROJECT}" \
  --substitutions "_TASKS=${TASKS}" \
  --async

echo ""
echo "====================================================================="
echo " All done. You can now close your laptop."
echo "====================================================================="
echo ""
echo " Track the build (~45–90 min first time):"
echo "   https://console.cloud.google.com/cloud-build/builds?project=${PROJECT}"
echo ""
echo " Track the Cloud Run Job tasks (~15–25 min once tasks start):"
echo "   https://console.cloud.google.com/run/jobs?project=${PROJECT}"
echo ""
echo " Once all tasks show green, merge the shards locally:"
echo "   python reza/Data_Cleaned/merge_thumbnail_features.py"
echo ""
echo " Or inspect progress from the command line:"
echo "   gcloud run jobs executions list --job=extract-thumbnails \\"
echo "     --region=${REGION} --project=${PROJECT}"
echo ""
