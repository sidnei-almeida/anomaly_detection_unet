---
title: Bottle Anomaly Detection
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: MVTec AD bottle visual anomaly detection API
tags:
  - anomaly-detection
  - computer-vision
  - fastapi
  - mvtec
---

# visual-anomaly-inspection-api

**FastAPI** service for **bottle** visual anomaly inspection on **MVTec AD**, powered by experiment **`mvtec_structured_objects_dae_v1`** (`multi_product_denoising_conv_autoencoder`).

This deployment is **bottle-only**: the underlying checkpoint was trained on multiple MVTec categories, but the public API accepts only `category=bottle` for reliable results.

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

### Base URL (production)

```text
https://salmeida-bottle-anomaly-detection.hf.space
```

| Resource | URL |
|----------|-----|
| API root | https://salmeida-bottle-anomaly-detection.hf.space/ |
| Swagger UI | https://salmeida-bottle-anomaly-detection.hf.space/docs |
| Health | https://salmeida-bottle-anomaly-detection.hf.space/health |
| Metadata | https://salmeida-bottle-anomaly-detection.hf.space/metadata |
| Predict | `POST` https://salmeida-bottle-anomaly-detection.hf.space/predict |

Space page: https://huggingface.co/spaces/salmeida/bottle-anomaly-detection

---

## Model overview

| Item | Value |
|------|-------|
| Architecture | `DenoisingConvAutoencoder` |
| Training | Denoising conv autoencoder, L1 loss, 256×256 RGB |
| Score | `top_1_z_score` on the **full** category-normalized z-map (compatible with `thresholds.json`) |
| Thresholds | Bottle-specific, from validation (`thresholds.json`) |
| Localization | Product foreground mask → `z_map_for_boxes` → heatmap (full z-map), mask, boxes |

### Supported category

**`bottle`** only. Other MVTec categories are rejected with HTTP 400.

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
- **Bottle** has `bottle_mean` / `bottle_std` in the `.npz`  
- **Bottle** has a threshold in `thresholds.json`

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

Set the API base URL (production Hugging Face Space):

```bash
export API_BASE_URL="https://salmeida-bottle-anomaly-detection.hf.space"
```

Compact response (no base64 — recommended for integrations and README):

```bash
curl -s "${API_BASE_URL}/health" | jq

curl -s -X POST "${API_BASE_URL}/predict" \
  -F "category=bottle" \
  -F "include_images=false" \
  -F "file=@imagem/anomaly_1.png" \
  -o examples/response_compact.json
```

Full response with images (`include_images=true`):

```bash
curl -s -X POST "${API_BASE_URL}/predict" \
  -F "category=bottle" \
  -F "include_images=true" \
  -F "file=@imagem/anomaly_1.png" \
  -o examples/response_full_sample.json
```

Or use the helper script (defaults to the HF Space URL):

```bash
chmod +x examples/curl_predict.sh
./examples/curl_predict.sh imagem/anomaly_1.png bottle
```

### Local development only

When running Docker or `uvicorn` on your machine, use `http://localhost:7860` instead of `API_BASE_URL`.

---

## Example response (`POST /predict`)

**Compact sample** (no `images` block): see [`examples/response_compact.json`](examples/response_compact.json) — generated with `include_images=false`.

**Full sample** (with base64 data URLs): see [`examples/response_full_sample.json`](examples/response_full_sample.json) — generated with `include_images=true`. Do not paste that file into the README (it is ~260 KB).

### Compact JSON shape

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
    "anomaly_score": 5.044606685638428,
    "threshold": 3.911202907562254,
    "error_mean": 0.01017017476260662,
    "z_map_max": 12.303437232971191
  },
  "image_size": {
    "width": 256,
    "height": 256
  },
  "boxes": [
    {
      "x": 100,
      "y": 184,
      "w": 7,
      "h": 10,
      "area": 42.0,
      "mean_z": 2.635521650314331,
      "max_z": 6.720283508300781,
      "score": 2.635521650314331,
      "foreground_ratio": 0.9461966753005981
    }
  ],
  "debug": {
    "bbox_method": "foreground_masked_conservative_connected_components_on_z_map",
    "score_region": "full_z_map",
    "localization_region": "product_foreground",
    "localization_note": "Classification uses top_1_z_score on the full category-normalized z-map. Bounding boxes and mask use the estimated product foreground only.",
    "latency_ms": 78.73
  }
}
```

When `include_images=true`, the same payload adds:

```json
"images": {
  "original": "data:image/png;base64,...",
  "reconstruction": "data:image/png;base64,...",
  "heatmap": "data:image/png;base64,...",
  "mask": "data:image/png;base64,..."
}
```

Image fields are returned as **base64 PNG data URLs** and can be used directly as `<img src="...">` values in the frontend.

### Response images (256×256)

| Field | Description |
|-------|-------------|
| `images.original` | Resized RGB input fed to the model |
| `images.reconstruction` | Autoencoder output |
| `images.heatmap` | Colored category-normalized z-map |
| `images.mask` | Binary suspicious-region mask |
| `images.overlay` | Optional — server-drawn boxes (`include_overlay=true`) |

### Front-end bounding boxes (256×256)

`image_size` is always `{ "width": 256, "height": 256 }`. Every box uses the same coordinate system as `images.original`:

- `x`, `y` — top-left corner in pixels  
- `w`, `h` — width and height in pixels  

Example (canvas or CSS on top of the 256×256 image):

```javascript
boxes.forEach(({ x, y, w, h }) => {
  // Draw on images.original (256×256) or scale if you upscale the image:
  // scaleX = displayWidth / response.image_size.width
  ctx.strokeRect(x * scaleX, y * scaleY, w * scaleX, h * scaleY);
});
```

Prefer drawing from the `boxes` array; `images.overlay` is optional server-side preview only.

---

## Approximate bounding boxes

Boxes are generated from **category-normalized reconstruction error maps constrained to the estimated product foreground** (Otsu-based product mask). They are **approximate visual hints** — not supervised detection (YOLO, etc.). Use them to highlight suspicious regions on the object, not as ground-truth segmentation.

Each box may include `foreground_ratio` (0–1): overlap with the estimated product region. Boxes with low foreground coverage are discarded server-side.

With `include_debug=true`, the API also returns `debug_images.product_mask` (estimated object region) and `debug_images.z_map_for_boxes` (z-map masked to the foreground for localization).

Classification (`scores.anomaly_score` vs `scores.threshold`) always uses the full z-map. The product mask is **not** applied to the score until thresholds are recalibrated.

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
