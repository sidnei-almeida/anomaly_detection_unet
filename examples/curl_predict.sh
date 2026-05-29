#!/usr/bin/env bash
# Example: call POST /predict on the Hugging Face Space API.

set -euo pipefail

BASE_URL="${BASE_URL:-https://salmeida-bottle-anomaly-detection.hf.space}"
IMAGE_PATH="${1:-imagem/anomaly_1.png}"
CATEGORY="${2:-bottle}"

echo "API base: ${BASE_URL}"
echo ""
echo "GET ${BASE_URL}/health"
curl -sS "${BASE_URL}/health" | python -m json.tool

echo ""
echo "POST ${BASE_URL}/predict (category=${CATEGORY}, compact)"
curl -sS -X POST "${BASE_URL}/predict" \
  -F "category=${CATEGORY}" \
  -F "include_images=false" \
  -F "include_debug=false" \
  -F "file=@${IMAGE_PATH}" | python -m json.tool

echo ""
echo "Full response: BASE_URL=${BASE_URL} include_images=true file=@${IMAGE_PATH} > examples/response_full_sample.json"
