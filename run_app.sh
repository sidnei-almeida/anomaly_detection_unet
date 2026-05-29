#!/bin/bash
# Local run — same port as Hugging Face Spaces Docker image.

set -euo pipefail

export PORT="${PORT:-7860}"
export CORS_ORIGINS="${CORS_ORIGINS:-*}"

ARTIFACT_DIR="models/mvtec_structured_objects_dae_v1"

for f in best_model.pt category_error_profiles.npz thresholds.json \
         bbox_visualization_config.json config.json; do
  if [ ! -f "${ARTIFACT_DIR}/${f}" ]; then
    echo "⚠️  Missing: ${ARTIFACT_DIR}/${f} — run: git lfs pull"
  fi
done

exec uvicorn app:app --host 0.0.0.0 --port "${PORT}"
