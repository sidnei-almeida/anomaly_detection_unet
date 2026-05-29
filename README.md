# visual-anomaly-inspection-api

**FastAPI** service for multi-category visual anomaly inspection on **MVTec AD structured objects**, powered by experiment **`mvtec_structured_objects_dae_v1`** (`multi_product_denoising_conv_autoencoder`).

Designed for **Hugging Face Spaces (Docker)** and local Docker runs on port **7860**.

---

## API overview

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Service banner + doc links |
| `/health` | GET | Artifact readiness flags |
| `/metadata` | GET | Model/categories/outputs metadata |
| `/predict` | POST | **Primary** — multipart inference |
| `/infer` | POST | Legacy alias (deprecated) |

**Project name:** `visual-anomaly-inspection-api`  
**Swagger UI:** `/docs`

---

## Model overview

| Item | Value |
|------|-------|
| Architecture | `DenoisingConvAutoencoder` |
| Training | Denoising conv autoencoder, L1 loss, 256×256 RGB |
| Score | `top_1_z_score` (mean of top 1% z-map pixels) |
| Thresholds | Per-category, from validation (`thresholds.json`) |
| Localization | Category-normalized reconstruction error → heatmap, mask, approximate boxes |

### Supported categories

`bottle` · `capsule` · `hazelnut` · `metal_nut` · `pill` · `screw` · `zipper`

---

## Required artifacts

All inference files live under `models/mvtec_structured_objects_dae_v1/`:

| File | Required at startup |
|------|---------------------|
| `best_model.pt` | Yes |
| `category_error_profiles.npz` | Yes |
| `thresholds.json` | Yes |
| `bbox_visualization_config.json` | Yes |
| `config.json` | Yes |
| `manifest.json` | Optional metadata |
| `final_metrics.json` | Optional metadata |
| `training_history.json` | Optional metadata |
| `error_profile_metadata.json` | Optional metadata |

Startup validates:

- `.pt` exists and is not a tiny Git LFS pointer  
- Every category has `{category}_mean` / `{category}_std` in the `.npz`  
- Every category has a threshold in `thresholds.json`

> **Legacy (unused):** `models/legacy/bottle_unet_*` — old bottle-only U-Net; safe to ignore.

---

## Hugging Face Spaces deployment

1. Create a Space → **Docker** SDK.
2. Push this repository with **Git LFS** hydrated:

   ```bash
   git lfs install
   git lfs pull
   ```

3. Space settings:
   - SDK: Docker
   - Port: **7860** (default in `Dockerfile`)
4. Optional env vars:
   - `CORS_ORIGINS=https://sidnei-almeida.github.io` (comma-separated)
   - `CORS_ORIGINS=*` (default)

The container only **loads** versioned weights — no training or dataset download at runtime.

---

## Docker (local)

```bash
git lfs pull

docker build -t visual-anomaly-inspection-api .
docker run --rm -p 7860:7860 \
  -e CORS_ORIGINS="*" \
  visual-anomaly-inspection-api
```

---

## Local run (without Docker)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
git lfs pull

export PORT=7860
./run_app.sh
# or: uvicorn app:app --host 0.0.0.0 --port 7860
```

---

## Example request

```bash
curl -s http://localhost:7860/health | jq

curl -X POST "http://localhost:7860/predict" \
  -F "category=bottle" \
  -F "include_images=true" \
  -F "include_debug=false" \
  -F "file=@examples/bottle_anomaly.png" | jq
```

Or use the helper script:

```bash
chmod +x examples/curl_predict.sh
./examples/curl_predict.sh
```

---

## Example response (`POST /predict`)

```json
{
  "status": "anomaly",
  "is_anomaly": true,
  "category": "bottle",
  "model": {
    "experiment_name": "mvtec_structured_objects_dae_v1",
    "model_name": "multi_product_denoising_conv_autoencoder",
    "score_name": "top_1_z_score"
  },
  "scores": {
    "anomaly_score": 5.04,
    "threshold": 3.91,
    "error_mean": 0.018,
    "z_map_max": 6.72
  },
  "image_size": { "width": 256, "height": 256 },
  "boxes": [
    {
      "x": 100, "y": 184, "w": 7, "h": 10,
      "area": 42.0, "mean_z": 2.64, "max_z": 6.72, "score": 2.64
    }
  ],
  "images": {
    "original": "data:image/png;base64,...",
    "reconstruction": "data:image/png;base64,...",
    "heatmap": "data:image/png;base64,...",
    "mask": "data:image/png;base64,..."
  },
  "debug": {
    "bbox_method": "conservative_connected_components_on_z_map",
    "localization_note": "Bounding boxes are approximate suspicious regions derived from reconstruction error maps.",
    "latency_ms": 120.5
  }
}
```

### Response images (256×256)

| Field | Description |
|-------|-------------|
| `images.original` | Resized RGB input fed to the model |
| `images.reconstruction` | Autoencoder output |
| `images.heatmap` | Colored category-normalized z-map |
| `images.mask` | Binary suspicious-region mask |
| `images.overlay` | Optional — server-drawn boxes (`include_overlay=true`) |

**Front-end:** draw rectangles from `boxes` on `images.original` (coordinates are 256×256). Overlay is optional.

---

## Approximate bounding boxes

Boxes are **heuristic regions** from reconstruction error maps — **not** supervised detection (YOLO, etc.). Use them as visual hints, not ground-truth segmentation.

---

## Limitations

- Requires the correct `category` at inference time.
- Fixed **256×256** processing; boxes match that coordinate system.
- CPU by default on Hugging Face CPU Spaces (`device: cpu` in `/health`).
- No batch endpoint; one image per request.
- Large raw `z_map` / `error_map` arrays are not returned (use `include_debug=true` for grayscale debug images only).

---

## Git LFS

```bash
git lfs install
git lfs pull
git lfs ls-files
```

Tracked patterns (see `.gitattributes`): `*.pt`, `*.pth`, `*.npz`, `models/**`

---

## License

[MIT License](LICENSE)
