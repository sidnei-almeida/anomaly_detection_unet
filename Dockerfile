# syntax=docker/dockerfile:1.6

FROM python:3.10-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies required by OpenCV and PyTorch wheels.
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential libgl1 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY model_utils.py ./model_utils.py
COPY app.py ./app.py
COPY models ./models

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]

