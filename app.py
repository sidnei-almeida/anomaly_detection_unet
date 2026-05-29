from __future__ import annotations

import base64
import io
import logging
import os
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

from model_utils import (
    API_NAME,
    BBOX_NOTE,
    EXPERIMENT_NAME,
    IMAGE_SIZE,
    METADATA_NOTE,
    MODEL_NAME,
    SCORE_NAME,
    SUPPORTED_CATEGORIES,
    ModelArtifacts,
    get_health_payload,
    get_metadata_payload,
    predict,
    setup_model_and_config,
)

LOGGER = logging.getLogger("anomaly_detection_api")
logging.basicConfig(level=logging.INFO)

DEFAULT_PORT = int(os.environ.get("PORT", "7860"))


def _parse_cors_origins() -> List[str]:
    """Parse CORS_ORIGINS env var (comma-separated). Default: allow all."""
    raw = os.environ.get("CORS_ORIGINS", "*").strip()
    if not raw or raw == "*":
        return ["*"]
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


app = FastAPI(
    title="Visual Anomaly Inspection API",
    version="3.1.0",
    description=(
        "Hugging Face / Docker ready API for MVTec AD structured-object anomaly "
        "detection using a multi-product DenoisingConvAutoencoder."
    ),
    contact={
        "name": "sidnei-almeida",
        "email": "sidnei.almeida1806@gmail.com",
        "url": "https://github.com/sidnei-almeida",
    },
    license_info={"name": "MIT License"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_parse_cors_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

ARTIFACTS: Optional[ModelArtifacts] = None
STARTUP_ERROR: Optional[str] = None


def _encode_image(image: Image.Image, format_: str = "PNG") -> str:
    buffer = io.BytesIO()
    image.save(buffer, format=format_)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _encode_data_url(image: Image.Image, format_: str = "PNG") -> str:
    mime = "image/png" if format_.upper() == "PNG" else f"image/{format_.lower()}"
    return f"data:{mime};base64,{_encode_image(image, format_)}"


def _model_block() -> Dict[str, str]:
    """Stable model metadata block (always real string fields, never placeholders)."""
    return {
        "experiment_name": EXPERIMENT_NAME,
        "model_name": MODEL_NAME,
        "score_name": SCORE_NAME,
    }


def _scores_block(results: Dict[str, Any]) -> Dict[str, float]:
    """Numeric scores as JSON-safe Python floats."""
    return {
        "anomaly_score": float(results["anomaly_score"]),
        "threshold": float(results["threshold"]),
        "error_mean": float(results["error_mean"]),
        "z_map_max": float(results["z_map_max"]),
    }


def _boxes_block(results: Dict[str, Any]) -> list[Dict[str, float | int]]:
    """
  Bounding boxes in 256x256 space for client-side drawing on images.original.

  Coordinates: x, y = top-left; w, h = width/height in pixels.
  """
    serialized: list[Dict[str, float | int]] = []
    for box in results["boxes"]:
        entry: Dict[str, float | int] = {
            "x": int(box["x"]),
            "y": int(box["y"]),
            "w": int(box["w"]),
            "h": int(box["h"]),
            "area": float(box["area"]),
            "mean_z": float(box["mean_z"]),
            "max_z": float(box["max_z"]),
            "score": float(box["score"]),
        }
        if "foreground_ratio" in box:
            entry["foreground_ratio"] = float(box["foreground_ratio"])
        serialized.append(entry)
    return serialized


def _image_size_block() -> Dict[str, int]:
    """Inference coordinate system — matches returned images and box coordinates."""
    return {"width": int(IMAGE_SIZE), "height": int(IMAGE_SIZE)}


def _debug_block(results: Dict[str, Any], latency_ms: float) -> Dict[str, Any]:
    return {
        "bbox_method": str(results["bbox_method"]),
        "localization_note": METADATA_NOTE,
        "latency_ms": round(float(latency_ms), 3),
    }


def _build_predict_payload(
    results: Dict[str, Any],
    latency_ms: float,
    *,
    include_images: bool = True,
    include_debug: bool = False,
    include_overlay: bool = False,
) -> Dict[str, Any]:
    """
    Build the /predict response for front-end consumption.

    Boxes are 256x256 coordinates; draw them client-side on images.original.
    """
    payload: Dict[str, Any] = {
        "status": str(results["status"]),
        "is_anomaly": bool(results["is_anomaly"]),
        "category": str(results["category"]),
        "model": _model_block(),
        "scores": _scores_block(results),
        "image_size": _image_size_block(),
        "boxes": _boxes_block(results),
        "debug": _debug_block(results, float(latency_ms)),
    }

    if include_images:
        payload["images"] = {
            "original": _encode_data_url(results["original_image"]),
            "reconstruction": _encode_data_url(results["reconstructed_image"]),
            "heatmap": _encode_data_url(results["heatmap_colored"]),
            "mask": _encode_data_url(results["binary_mask"]),
        }
        if include_overlay:
            payload["images"]["overlay"] = _encode_data_url(results["overlay"])

    if include_debug:
        payload["debug_images"] = {
            "error_map": _encode_data_url(results["error_map_gray"]),
            "z_map_gray": _encode_data_url(results["z_map_gray"]),
            "product_mask": _encode_data_url(results["product_mask_gray"]),
        }

    return payload


async def _read_upload_image(file: UploadFile) -> Image.Image:
    try:
        contents = await file.read()
        return Image.open(io.BytesIO(contents)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(
            status_code=400, detail="The uploaded file is not a valid image."
        ) from exc


def _ensure_ready() -> ModelArtifacts:
    if STARTUP_ERROR:
        raise HTTPException(status_code=503, detail=f"Startup failed: {STARTUP_ERROR}")
    if ARTIFACTS is None:
        raise HTTPException(
            status_code=503,
            detail="The model is still loading. Retry in a few seconds.",
        )
    return ARTIFACTS


@app.on_event("startup")
def load_artifacts() -> None:
    """Validate and load experiment artifacts (no training or dataset download)."""
    global ARTIFACTS, STARTUP_ERROR
    try:
        ARTIFACTS = setup_model_and_config()
        STARTUP_ERROR = None
        LOGGER.info(
            "%s ready — experiment=%s device=%s",
            API_NAME,
            EXPERIMENT_NAME,
            ARTIFACTS.device,
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        STARTUP_ERROR = str(exc)
        ARTIFACTS = None
        LOGGER.error("Startup validation failed: %s", exc)


@app.get("/", summary="API status")
def root() -> Dict[str, Any]:
    """Simple service banner with docs link."""
    status = "ready" if ARTIFACTS is not None and not STARTUP_ERROR else "error"
    if ARTIFACTS is None and not STARTUP_ERROR:
        status = "loading"
    return {
        "api_name": API_NAME,
        "status": status,
        "message": f"{API_NAME} — MVTec structured-object visual anomaly inspection",
        "docs_url": "/docs",
        "health_url": "/health",
        "metadata_url": "/metadata",
        "predict_url": "POST /predict",
    }


@app.get("/health", summary="Health and artifact readiness")
def health() -> Dict[str, Any]:
    """Return readiness flags; status is 'error' when startup validation failed."""
    return get_health_payload(ARTIFACTS, error=STARTUP_ERROR)


@app.get("/metadata", summary="Model and API metadata")
def metadata() -> Dict[str, Any]:
    """Return model metadata, supported categories, outputs, and limitations."""
    return get_metadata_payload(ARTIFACTS)


@app.post("/predict", summary="Run visual anomaly inspection")
async def predict_endpoint(
    file: UploadFile = File(..., description="RGB image (PNG or JPEG)."),
    category: str = Form(
        ...,
        description="MVTec category: bottle, capsule, hazelnut, metal_nut, pill, screw, zipper.",
    ),
    include_images: bool = Form(True, description="Return visualization data URLs."),
    include_debug: bool = Form(False, description="Include debug_images grayscale maps."),
    include_overlay: bool = Form(
        False,
        description="Include server-drawn overlay (front-end should draw boxes from JSON).",
    ),
) -> JSONResponse:
    """Run inference; returns 256x256 images and structured boxes for the front-end."""
    artifacts = _ensure_ready()
    image = await _read_upload_image(file)

    start_ts = time.perf_counter()
    try:
        results = predict(artifacts, image, category)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    latency_ms = round((time.perf_counter() - start_ts) * 1000, 3)
    payload = _build_predict_payload(
        results,
        latency_ms,
        include_images=include_images,
        include_debug=include_debug,
        include_overlay=include_overlay,
    )
    return JSONResponse(payload)


@app.post("/infer", summary="[Legacy] Alias for POST /predict", deprecated=True)
async def infer(
    file: UploadFile = File(...),
    category: str = Query(..., description="MVTec category (legacy query param)."),
    include_visualizations: bool = Query(True, description="Deprecated: use include_images."),
    include_images: Optional[bool] = Query(None),
    include_debug: bool = Query(False),
    include_overlay: bool = Query(False),
) -> JSONResponse:
    """Legacy alias — prefer POST /predict with multipart form fields."""
    artifacts = _ensure_ready()
    image = await _read_upload_image(file)
    images_flag = include_images if include_images is not None else include_visualizations

    start_ts = time.perf_counter()
    try:
        results = predict(artifacts, image, category)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    latency_ms = round((time.perf_counter() - start_ts) * 1000, 3)
    payload = _build_predict_payload(
        results,
        latency_ms,
        include_images=images_flag,
        include_debug=include_debug,
        include_overlay=include_overlay,
    )
    return JSONResponse(payload)


def get_application() -> FastAPI:
    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=DEFAULT_PORT, reload=False)
