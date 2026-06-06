#!/usr/bin/env bash
# run_ner_gcp.sh
#
# The ONLY part of this pipeline that needs your laptop:
#   - uploading the input parquet files to GCS
#   - submitting the Cloud Build job (returns in seconds)
#
# After this script exits you can close your laptop.
# Cloud Build builds the container on GCP, then Cloud Run executes the NER job.
# Both bill only for the seconds they are actually running and stop automatically.
#
# Prerequisites (one-time)
# ------------------------
#   gcloud auth login
#   gcloud auth configure-docker us-central1-docker.pkg.dev
#   gcloud config set project project-10b142ae-d53f-4f87-81e
#
# Usage (from the repo root — the folder containing reza/)
# ---------------------------------------------------------
#   bash reza/run_ner_gcp.sh
#
# Skip re-uploading files on subsequent runs:
#   bash reza/run_ner_gcp.sh --skip-upload
#
# Monitor progress after closing your laptop:
#   https://console.cloud.google.com/cloud-build/builds?project=project-10b142ae-d53f-4f87-81e
#   https://console.cloud.google.com/run/jobs?project=project-10b142ae-d53f-4f87-81e

set -euo pipefail

PROJECT="project-10b142ae-d53f-4f87-81e"
BUCKET="socialmediaanalyticsproject"
REGION="us-central1"

SKIP_UPLOAD=false
for arg in "$@"; do
  [[ $arg == "--skip-upload" ]] && SKIP_UPLOAD=true
done

# ── 1. Upload input data ──────────────────────────────────────────────────────

if [ "$SKIP_UPLOAD" = false ]; then
  echo "=== Uploading input files to gs://${BUCKET}/input/ ==="
  gsutil -m cp \
    reza/Data_Cleaned/yt_videos_with_local_transcripts.parquet \
    reza/Data_Cleaned/yt_comments_1_cleaned.parquet \
    reza/Data_Cleaned/yt_comments_2_cleaned.parquet \
    reza/Data_Cleaned/yt_comments_3_cleaned.parquet \
    reza/Data_Cleaned/yt_comments_4_cleaned.parquet \
    "gs://${BUCKET}/input/"
  echo "Upload complete."
else
  echo "=== Skipping upload (--skip-upload) ==="
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
# Cloud Build needs run.admin to create/execute Cloud Run Jobs, and
# iam.serviceAccountUser to attach the job's service account.

CLOUDBUILD_SA="$(gcloud projects describe "${PROJECT}" \
  --format='value(projectNumber)')@cloudbuild.gserviceaccount.com"

gcloud projects add-iam-policy-binding "${PROJECT}" \
  --member="serviceAccount:${CLOUDBUILD_SA}" \
  --role="roles/run.admin" --quiet

gcloud projects add-iam-policy-binding "${PROJECT}" \
  --member="serviceAccount:${CLOUDBUILD_SA}" \
  --role="roles/iam.serviceAccountUser" --quiet

# ── 5. Submit Cloud Build — async, returns immediately ────────────────────────

echo "=== Submitting Cloud Build (async) ==="
gcloud builds submit . \
  --config  "reza/cloudbuild-ner.yaml" \
  --project "${PROJECT}" \
  --async

echo ""
echo "=================================================="
echo " All done. You can now close your laptop."
echo "=================================================="
echo ""
echo " Track the build:"
echo "   https://console.cloud.google.com/cloud-build/builds?project=${PROJECT}"
echo ""
echo " Track the NER job:"
echo "   https://console.cloud.google.com/run/jobs?project=${PROJECT}"
echo ""
echo " Download results when the job finishes:"
echo "   gsutil -m cp -r gs://${BUCKET}/output/ner/ reza/Data_Cleaned/"
echo ""
