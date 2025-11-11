---
title: Bottle Anomaly Detection
emoji: 📈
colorFrom: purple
colorTo: yellow
sdk: docker
pinned: false
license: mit
short_description: REST API for bottle anomaly detection with a U-Net model.
---

## Bottle Anomaly Detection API

REST API that serves a U-Net reconstruction model trained on the MVTec AD **bottle** subset.  
The service accepts an RGB image, classifies it as *Normal* or *Anomaly Detected*, and optionally returns visualization artifacts encoded as base64.

---

## Project Structure

```
anomaly_detection_unet/
├── app.py                  # FastAPI application entry point
├── model_utils.py          # Model loading, preprocessing and visualization helpers
├── models/
│   ├── bottle_unet_best.pth        # U-Net checkpoint (Git LFS)
│   └── bottle_unet_config.json     # Inference thresholds and post-processing configs
├── requirements.txt        # Runtime dependencies (CPU only)
├── Dockerfile              # Production container image
├── .dockerignore
└── README.md
```

Legacy assets such as notebooks, example images and training history are kept for reference but are excluded from the Docker build context.

---

## Quickstart (Local)

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

uvicorn app:app --host 0.0.0.0 --port 8000
```

Open http://localhost:8000/docs to explore the interactive Swagger UI.

---

## API Overview

| Endpoint      | Method | Description                                   |
|---------------|--------|-----------------------------------------------|
| `/health`     | GET    | Lightweight readiness probe                   |
| `/`           | GET    | Metadata and documentation links              |
| `/infer`      | POST   | Image classification and optional artifacts   |

### Request (multipart/form-data)

- `file` *(required)*: RGB image (`.png`, `.jpg`, `.jpeg`)
- `include_visualizations` *(boolean, default `true`)*: return anomaly map, mask, heatmap overlay and bounding box as base64-encoded PNGs

### Response

```json
{
  "prediction": "Anomaly Detected",
  "reconstruction_error": 0.00123,
  "thresholds": {
    "classification": 0.000205,
    "pixel_visualization": 20.0,
    "bounding_box": 1.5
  },
  "latency_ms": 87.421,
  "image_size": { "width": 1024, "height": 1024 },
  "artifacts": {
    "anomaly_map": { "format": "PNG", "encoding": "base64", "data": "..." },
    "binary_mask": { "format": "PNG", "encoding": "base64", "data": "..." },
    "heatmap_overlay": { "format": "PNG", "encoding": "base64", "data": "..." },
    "bounding_box": { "format": "PNG", "encoding": "base64", "data": "..." }
  }
}
```

Set `include_visualizations=false` to skip the `artifacts` payload.

---

## Docker

Build and run the container locally:

```bash
docker build -t bottle-anomaly-api .
docker run --rm -p 7860:7860 bottle-anomaly-api
```

The image exposes port **7860** by default, matching the requirement for Hugging Face Space deployments.

---

## Deploying to Hugging Face Spaces

1. **Enable Git LFS locally** (required for the ~100 MB `.pth` checkpoint):
   ```bash
   git lfs install
   git lfs track "*.pth"
   ```
2. Push the repository (including LFS files) to the Space.
3. Configure the Space as a **Docker** Space; the platform will detect the `Dockerfile`.
4. Hugging Face automatically sets `PORT=7860`. The container entrypoint already respects this value.

> ⚠️ Do **not** commit large model weights outside Git LFS. Hugging Face enforces a strict 5 GB limit for Git blobs.

---

## Configuration

`models/bottle_unet_config.json` controls inference behaviour:

- `classification_threshold`: reconstruction error threshold to flag anomalies
- `pixel_visualization_threshold`: pixel-level cut-off for binary masks
- `bounding_box_threshold`: sensitivity for contour detection
- `dilation_iterations`: morphological dilation applied before extracting bounding boxes

Adjust these values to tune false-positive/false-negative trade-offs.

---

## Tech Stack

- **FastAPI** + **Uvicorn** for high-performance REST serving
- **PyTorch** for U-Net reconstruction
- **torchvision.transforms** for preprocessing
- **OpenCV** + **Pillow** for post-processing and visualization

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

Maintained by [@sidnei-almeida](https://github.com/sidnei-almeida)

