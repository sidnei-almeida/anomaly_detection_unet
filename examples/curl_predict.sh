#!/usr/bin/env bash
# Example: call POST /predict on local Docker or Hugging Face Space (port 7860).

set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:7860}"
IMAGE_PATH="${1:-examples/bottle_anomaly.png}"
CATEGORY="${2:-bottle}"

echo "GET ${BASE_URL}/health"
curl -s "${BASE_URL}/health" | python -m json.tool

echo ""
echo "POST ${BASE_URL}/predict (category=${CATEGORY})"
curl -s -X POST "${BASE_URL}/predict" \
  -F "category=${CATEGORY}" \
  -F "include_images=true" \
  -F "include_debug=false" \
  -F "file=@${IMAGE_PATH}" | python -m json.tool
