from __future__ import annotations

import base64
import io
import logging
import time
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

from model_utils import (
    display_bounding_box,
    get_anomaly_map_image,
    get_binary_mask_image,
    get_heatmap_image,
    predict,
    setup_model_and_config,
)

LOGGER = logging.getLogger("anomaly_detection_api")
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Bottle Anomaly Detection API",
    version="1.0.0",
    description=(
        "REST API that serves the bottle anomaly detection U-Net model. "
        "Upload an RGB image to receive classification results, reconstruction error, "
        "and optional visualization artifacts encoded in base64."
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
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL: Optional[Any] = None
CONFIG: Optional[Dict[str, Any]] = None


def _encode_image(image: Image.Image, format_: str = "PNG") -> str:
    """Encode an image as a base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format=format_)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


@app.on_event("startup")
def load_artifacts() -> None:
    """Load the trained model and configuration once the application starts."""
    global MODEL, CONFIG
    MODEL, CONFIG = setup_model_and_config()
    LOGGER.info("Artifacts loaded successfully. API ready to serve requests.")


@app.get("/health", summary="Health check")
def health() -> Dict[str, str]:
    """Return a simple health status for readiness probes."""
    status = "ready" if MODEL is not None else "loading"
    return {"status": status}


@app.get("/", summary="API metadata")
def root() -> Dict[str, Any]:
    """Return high-level metadata and helpful links."""
    return {
        "name": app.title,
        "version": app.version,
        "description": app.description,
        "docs": {
            "openapi": "/openapi.json",
            "swagger_ui": "/docs",
            "redoc": "/redoc",
        },
    }


@app.post(
    "/infer",
    summary="Run anomaly detection on an uploaded image",
    responses={
        200: {"description": "Inference completed successfully"},
        400: {"description": "Invalid image payload"},
        503: {"description": "Model artifacts are not ready yet"},
    },
)
async def infer(
    file: UploadFile = File(..., description="RGB image (PNG or JPEG formats)."),
    include_visualizations: bool = Query(
        True,
        description="If true, return anomaly maps and masks encoded as base64 PNGs.",
    ),
) -> JSONResponse:
    """Run inference against the uploaded image and return model outputs."""
    if MODEL is None or CONFIG is None:
        raise HTTPException(
            status_code=503,
            detail="The model is still loading. Retry in a few seconds.",
        )

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(
            status_code=400, detail="The uploaded file is not a valid image."
        ) from exc

    start_ts = time.perf_counter()
    results = predict(MODEL, CONFIG, image)
    latency_ms = (time.perf_counter() - start_ts) * 1000

    payload: Dict[str, Any] = {
        "prediction": results["prediction"],
        "reconstruction_error": results["error"],
        "thresholds": {
            "classification": CONFIG["classification_threshold"],
            "pixel_visualization": CONFIG["pixel_visualization_threshold"],
            "bounding_box": CONFIG.get("bounding_box_threshold", 1.5),
        },
        "latency_ms": round(latency_ms, 3),
        "image_size": {"width": image.width, "height": image.height},
        "bounding_boxes": results.get("bounding_boxes", []),
        "detections": results.get("detections", []),
    }

    if include_visualizations:
        LOGGER.debug("Generating visualization artifacts.")
        anomaly_map_image = get_anomaly_map_image(results)
        binary_mask_image = get_binary_mask_image(results)
        heatmap_image = get_heatmap_image(results)
        bbox_image = display_bounding_box(results, CONFIG)

        payload["artifacts"] = {
            "anomaly_map": {
                "format": "PNG",
                "encoding": "base64",
                "data": _encode_image(anomaly_map_image),
            },
            "binary_mask": {
                "format": "PNG",
                "encoding": "base64",
                "data": _encode_image(binary_mask_image),
            },
            "heatmap_overlay": {
                "format": "PNG",
                "encoding": "base64",
                "data": _encode_image(heatmap_image),
            },
            "bounding_box": {
                "format": "PNG",
                "encoding": "base64",
                "data": _encode_image(bbox_image),
            },
        }

    # Remove internal-only metadata to avoid leaking implementation details
    results.pop("mask_boxes", None)

    return JSONResponse(payload)


def get_application() -> FastAPI:
    """Helper that returns the FastAPI instance (used by some ASGI servers)."""
    return app


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)

