# syntax=docker/dockerfile:1.6
# Hugging Face Spaces — Docker SDK
# https://huggingface.co/docs/hub/spaces-sdks-docker

FROM python:3.10-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    CORS_ORIGINS="*"

WORKDIR /app

# OpenCV headless runtime dependency
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY app.py model_utils.py ./
COPY models/mvtec_structured_objects_dae_v1 ./models/mvtec_structured_objects_dae_v1/

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
