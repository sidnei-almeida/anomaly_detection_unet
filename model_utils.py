from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

LOGGER = logging.getLogger(__name__)

API_NAME = "visual-anomaly-inspection-api"
EXPERIMENT_NAME = "mvtec_structured_objects_dae_v1"
MODEL_NAME = "multi_product_denoising_conv_autoencoder"
MODEL_CLASS = "DenoisingConvAutoencoder"

BASE_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = BASE_DIR / "models" / EXPERIMENT_NAME
MODEL_PATH = EXPERIMENT_DIR / "best_model.pt"
PROFILES_PATH = EXPERIMENT_DIR / "category_error_profiles.npz"
THRESHOLDS_PATH = EXPERIMENT_DIR / "thresholds.json"
BBOX_CONFIG_PATH = EXPERIMENT_DIR / "bbox_visualization_config.json"
CONFIG_PATH = EXPERIMENT_DIR / "config.json"
MANIFEST_PATH = EXPERIMENT_DIR / "manifest.json"
SUPPORTED_CATEGORIES = (
    "bottle",
    "capsule",
    "hazelnut",
    "metal_nut",
    "pill",
    "screw",
    "zipper",
)
SCORE_NAME = "top_1_z_score"
IMAGE_SIZE = 256
MIN_MODEL_BYTES = 1_000
LOCALIZATION_METHOD = "category-normalized reconstruction error"
BBOX_METHOD = "foreground_masked_conservative_connected_components_on_z_map"
METADATA_NOTE = (
    "Bounding boxes are generated from category-normalized reconstruction error maps "
    "constrained to the estimated product foreground. Boxes are approximate visual hints."
)
LIMITATION_NOTE = METADATA_NOTE
BBOX_NOTE = METADATA_NOTE

PRODUCT_MASK_MIN_AREA_RATIO = 0.03
PRODUCT_MASK_MAX_AREA_RATIO = 0.85
BOX_MIN_FOREGROUND_RATIO = 0.25
BOX_BORDER_FOREGROUND_RATIO = 0.4
BOX_BORDER_MARGIN = 3

METADATA_OUTPUTS = ["original", "reconstruction", "heatmap", "mask"]


class DenoisingConvAutoencoder(nn.Module):
    """Multi-product denoising convolutional autoencoder for MVTec AD structured objects."""

    def __init__(self, latent_channels: int = 256) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, latent_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(latent_channels),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                latent_channels, 128, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        return self.decoder(encoded)


@dataclass
class ArtifactPaths:
    """Resolved paths to experiment artifacts."""

    root: Path
    model: Path
    profiles: Path
    thresholds: Path
    bbox_config: Path
    config: Path
    manifest: Path


@dataclass
class ModelArtifacts:
    """Runtime artifacts required to reproduce category-aware inference."""

    model: DenoisingConvAutoencoder
    error_profiles: Dict[str, np.ndarray]
    thresholds: Dict[str, Any]
    bbox_config: Dict[str, Any]
    config: Dict[str, Any]
    manifest: Dict[str, Any]
    paths: ArtifactPaths
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    experiment_name: str = EXPERIMENT_NAME
    model_name: str = MODEL_NAME


def get_artifact_dir(base_dir: Path | None = None) -> Path:
    """Return the directory that holds all experiment artifacts."""
    if base_dir is not None:
        return Path(base_dir)
    return EXPERIMENT_DIR


def get_runtime_device() -> torch.device:
    """Select CUDA when available; Hugging Face CPU Spaces use CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_artifact_paths(artifact_dir: Path | None = None) -> ArtifactPaths:
    """Build absolute paths for every required artifact file."""
    root = get_artifact_dir(artifact_dir)
    return ArtifactPaths(
        root=root,
        model=root / MODEL_PATH.name,
        profiles=root / PROFILES_PATH.name,
        thresholds=root / THRESHOLDS_PATH.name,
        bbox_config=root / BBOX_CONFIG_PATH.name,
        config=root / CONFIG_PATH.name,
        manifest=root / MANIFEST_PATH.name,
    )


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_artifacts(paths: ArtifactPaths | None = None) -> Dict[str, Any]:
    """
    Validate artifact presence, size, and per-category coverage before loading weights.

    Raises ``FileNotFoundError`` or ``ValueError`` when validation fails.
    """
    paths = paths or resolve_artifact_paths()
    report: Dict[str, Any] = {
        "artifact_dir": str(paths.root),
        "categories_validated": list(SUPPORTED_CATEGORIES),
        "files_ok": True,
    }

    required = {
        "model": paths.model,
        "profiles": paths.profiles,
        "thresholds": paths.thresholds,
        "bbox_config": paths.bbox_config,
        "config": paths.config,
    }

    for label, path in required.items():
        if not path.exists():
            raise FileNotFoundError(f"Required artifact missing ({label}): {path}")

    model_size = paths.model.stat().st_size
    if model_size < MIN_MODEL_BYTES:
        raise ValueError(
            f"Model file at {paths.model} is unexpectedly small ({model_size} bytes). "
            "Run `git lfs pull` to hydrate Git LFS pointers."
        )
    report["model_bytes"] = model_size

    thresholds = _load_json(paths.thresholds)
    z_thresholds = thresholds.get("z_score_thresholds", {})
    profiles_npz = np.load(paths.profiles)

    missing_profiles: List[str] = []
    missing_thresholds: List[str] = []

    for category in SUPPORTED_CATEGORIES:
        mean_key = f"{category}_mean"
        std_key = f"{category}_std"
        if mean_key not in profiles_npz.files or std_key not in profiles_npz.files:
            missing_profiles.append(category)
            continue

        mean_map = profiles_npz[mean_key]
        std_map = profiles_npz[std_key]
        if mean_map.shape != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(
                f"Profile {mean_key} has shape {mean_map.shape}, expected ({IMAGE_SIZE}, {IMAGE_SIZE})."
            )
        if std_map.shape != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(
                f"Profile {std_key} has shape {std_map.shape}, expected ({IMAGE_SIZE}, {IMAGE_SIZE})."
            )

        if category not in z_thresholds:
            missing_thresholds.append(category)
        elif "threshold" not in z_thresholds[category]:
            missing_thresholds.append(category)

    if missing_profiles:
        raise ValueError(
            f"Missing mean/std profiles for categories: {', '.join(missing_profiles)}"
        )
    if missing_thresholds:
        raise ValueError(
            f"Missing z_score thresholds for categories: {', '.join(missing_thresholds)}"
        )

    report["profile_keys"] = list(profiles_npz.files)
    report["threshold_categories"] = list(z_thresholds.keys())
    return report


def _load_checkpoint(
    model: DenoisingConvAutoencoder, model_path: Path, device: torch.device
) -> None:
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)


@lru_cache(maxsize=1)
def setup_model_and_config(artifact_dir: str | os.PathLike[str] | None = None) -> ModelArtifacts:
    """
    Validate artifacts, load the DenoisingConvAutoencoder, and return runtime state.

    Expected layout::

        models/mvtec_structured_objects_dae_v1/
            best_model.pt
            category_error_profiles.npz
            thresholds.json
            bbox_visualization_config.json
            config.json
            manifest.json
    """
    paths = resolve_artifact_paths(
        Path(artifact_dir) if artifact_dir is not None else None
    )
    validation = validate_artifacts(paths)
    device = get_runtime_device()

    thresholds = _load_json(paths.thresholds)
    bbox_config = _load_json(paths.bbox_config)
    config = _load_json(paths.config) if paths.config.exists() else {}
    manifest = _load_json(paths.manifest) if paths.manifest.exists() else {}

    profiles_npz = np.load(paths.profiles)
    error_profiles = {key: profiles_npz[key] for key in profiles_npz.files}

    model = DenoisingConvAutoencoder(latent_channels=256).to(device)
    _load_checkpoint(model, paths.model, device)
    model.eval()

    LOGGER.info(
        "Artifacts loaded from %s (model %.2f MB, %d categories)",
        paths.root,
        validation["model_bytes"] / 1_048_576,
        len(SUPPORTED_CATEGORIES),
    )

    return ModelArtifacts(
        model=model,
        error_profiles=error_profiles,
        thresholds=thresholds,
        bbox_config=bbox_config,
        config=config,
        manifest=manifest,
        paths=paths,
        device=device,
    )


def get_health_payload(
    artifacts: ModelArtifacts | None,
    *,
    error: str | None = None,
) -> Dict[str, Any]:
    """Build the /health response with per-component load flags."""
    if error or artifacts is None:
        return {
            "status": "error" if error else "loading",
            "model_loaded": False,
            "profiles_loaded": False,
            "thresholds_loaded": False,
            "bbox_config_loaded": False,
            "supported_categories_count": len(SUPPORTED_CATEGORIES),
            "device": str(get_runtime_device()),
            **({"error": error} if error else {}),
        }

    return {
        "status": "ready",
        "model_loaded": artifacts.model is not None,
        "profiles_loaded": bool(artifacts.error_profiles),
        "thresholds_loaded": bool(artifacts.thresholds),
        "bbox_config_loaded": bool(artifacts.bbox_config),
        "supported_categories_count": len(SUPPORTED_CATEGORIES),
        "device": str(artifacts.device),
    }


def get_metadata_payload(artifacts: ModelArtifacts | None = None) -> Dict[str, Any]:
    """Build the /metadata response for API discovery."""
    localization = LOCALIZATION_METHOD
    if artifacts and artifacts.bbox_config:
        localization = str(
            artifacts.bbox_config.get("localization_method", localization)
        )

    return {
        "api_name": API_NAME,
        "experiment_name": EXPERIMENT_NAME,
        "model_name": MODEL_NAME,
        "supported_categories": list(SUPPORTED_CATEGORIES),
        "image_size": IMAGE_SIZE,
        "score_name": SCORE_NAME,
        "outputs": list(METADATA_OUTPUTS),
        "localization_method": localization,
        "note": METADATA_NOTE,
    }


def validate_category(category: str) -> str:
    """Normalize and validate the requested MVTec category."""
    normalized = category.strip().lower()
    if normalized not in SUPPORTED_CATEGORIES:
        supported = ", ".join(SUPPORTED_CATEGORIES)
        raise ValueError(
            f"Unsupported category '{category}'. Supported categories: {supported}."
        )
    return normalized


def _preprocess_image(image: Image.Image) -> Tuple[torch.Tensor, Image.Image]:
    """Convert an RGB PIL image to a [0, 1] float tensor at 256x256."""
    resized = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    tensor = transforms.ToTensor()(resized).unsqueeze(0)
    return tensor, resized


def _compute_error_map(
    image_tensor: torch.Tensor, reconstruction: torch.Tensor
) -> np.ndarray:
    """Mean absolute error per pixel across RGB channels."""
    error_map = torch.mean(torch.abs(image_tensor - reconstruction), dim=1)
    return error_map.squeeze(0).cpu().numpy()


def _compute_z_map(
    error_map: np.ndarray,
    category: str,
    error_profiles: Dict[str, np.ndarray],
) -> np.ndarray:
    """Category-normalized, clipped, and smoothed z-error map."""
    category_mean = error_profiles[f"{category}_mean"]
    category_std = error_profiles[f"{category}_std"]
    safe_std = np.maximum(category_std, 1e-8)

    z_map = (error_map - category_mean) / safe_std
    z_map = np.clip(z_map, 0, None)
    z_map = cv2.GaussianBlur(z_map.astype(np.float32), (5, 5), 0)
    return z_map


def _compute_top_1_z_score(z_map: np.ndarray) -> float:
    """Average of the top 1% highest z-scores in the map."""
    flat = z_map.reshape(-1)
    top_k = max(1, int(np.ceil(flat.size * 0.01)))
    top_values = np.partition(flat, -top_k)[-top_k:]
    return float(np.mean(top_values))


def _get_category_threshold(thresholds: Dict[str, Any], category: str) -> float:
    return float(thresholds["z_score_thresholds"][category]["threshold"])


def compute_product_mask(
    original_rgb: np.ndarray,
    category: str | None = None,
) -> np.ndarray:
    """
    Estimate a binary product/foreground mask [H, W] with 1 on the object and 0 on background.

    Uses Otsu thresholding on grayscale, picks the polarity whose largest connected
    component is central and has plausible area, then morphologically closes and dilates.

    ``original_rgb`` is a float array [H, W, 3] in [0, 1]. ``category`` is reserved for
    future category-specific tuning.
    """
    del category  # reserved for future per-category foreground heuristics

    height, width = original_rgb.shape[:2]
    gray = cv2.cvtColor(
        (np.clip(original_rgb, 0.0, 1.0) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY
    )
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    image_center_x = width / 2.0
    image_center_y = height / 2.0
    max_center_distance = float(
        np.hypot(image_center_x, image_center_y)
    ) or 1.0
    total_pixels = float(height * width)

    best_component: np.ndarray | None = None
    best_score = -1.0

    for candidate in (otsu, 255 - otsu):
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            candidate, connectivity=8
        )
        if num_labels <= 1:
            continue

        component_areas = stats[1:, cv2.CC_STAT_AREA]
        largest_idx = 1 + int(np.argmax(component_areas))
        area = float(stats[largest_idx, cv2.CC_STAT_AREA])
        area_ratio = area / total_pixels

        centroid_x, centroid_y = centroids[largest_idx]
        center_distance = float(
            np.hypot(centroid_x - image_center_x, centroid_y - image_center_y)
        )
        centrality = 1.0 - min(center_distance / max_center_distance, 1.0)

        area_plausible = (
            PRODUCT_MASK_MIN_AREA_RATIO <= area_ratio <= PRODUCT_MASK_MAX_AREA_RATIO
        )
        area_score = 1.0 if area_plausible else max(0.0, 1.0 - abs(area_ratio - 0.35))
        score = area_score * 0.55 + centrality * 0.45

        if score > best_score:
            best_score = score
            best_component = (labels == largest_idx).astype(np.uint8)

    if best_component is None:
        best_component = np.ones((height, width), dtype=np.uint8)

    closed = cv2.morphologyEx(
        best_component * 255, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)
    )
    dilated = cv2.dilate(closed, np.ones((5, 5), np.uint8), iterations=1)
    return (dilated > 0).astype(np.float32)


def _box_foreground_ratio(
    product_mask: np.ndarray, x: int, y: int, w: int, h: int
) -> float:
    """Mean foreground coverage inside a bounding box."""
    if w <= 0 or h <= 0:
        return 0.0
    region = product_mask[y : y + h, x : x + w]
    if region.size == 0:
        return 0.0
    return float(np.mean(region))


def _box_touches_border(x: int, y: int, w: int, h: int, size: int = IMAGE_SIZE) -> bool:
    return (
        x < BOX_BORDER_MARGIN
        or y < BOX_BORDER_MARGIN
        or x + w > size - BOX_BORDER_MARGIN
        or y + h > size - BOX_BORDER_MARGIN
    )


def _passes_foreground_filter(
    product_mask: np.ndarray, x: int, y: int, w: int, h: int
) -> bool:
    """Reject boxes that sit mostly on background or hug image borders without foreground."""
    foreground_ratio = _box_foreground_ratio(product_mask, x, y, w, h)
    if foreground_ratio < BOX_MIN_FOREGROUND_RATIO:
        return False
    if _box_touches_border(x, y, w, h) and foreground_ratio < BOX_BORDER_FOREGROUND_RATIO:
        return False
    return True


def _normalize_heatmap(z_map: np.ndarray) -> np.ndarray:
    """Normalize the z-map to an unsigned 8-bit heatmap for visualization."""
    if z_map.size == 0:
        return np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)

    positive = z_map[z_map > 0]
    if positive.size == 0:
        return np.zeros(z_map.shape, dtype=np.uint8)

    high = float(np.percentile(positive, 99.5))
    if high <= 0:
        return np.zeros(z_map.shape, dtype=np.uint8)

    normalized = np.clip(z_map / high, 0.0, 1.0)
    return (normalized * 255).astype(np.uint8)


def _build_localization_mask(
    z_map: np.ndarray, bbox_config: Dict[str, Any]
) -> np.ndarray:
    """
    Build a binary mask from the z-map using bbox visualization percentiles.

    Approximate suspicious regions — not ground-truth segmentation.
    """
    bbox_settings = bbox_config["bounding_boxes"]
    low_percentile = float(bbox_settings["low_percentile"])
    threshold = float(np.percentile(z_map, low_percentile))
    mask = (z_map >= threshold).astype(np.uint8) * 255

    kernel_size = int(bbox_settings.get("kernel_size", 3))
    dilate_iterations = int(bbox_settings.get("dilate_iterations", 0))
    if dilate_iterations > 0:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=dilate_iterations)

    return mask


def _extract_approximate_boxes(
    z_map: np.ndarray,
    mask: np.ndarray,
    bbox_config: Dict[str, Any],
    product_mask: np.ndarray | None = None,
) -> List[Dict[str, Any]]:
    """
    Derive approximate bounding boxes from connected components on a (masked) z-map.

    Coordinates are in the 256x256 model space for client-side drawing.
    Boxes overlapping background are filtered using ``product_mask``.
    """
    bbox_settings = bbox_config["bounding_boxes"]
    min_area = int(bbox_settings["min_area"])
    max_area = int(IMAGE_SIZE * IMAGE_SIZE * float(bbox_settings["max_area_ratio"]))
    min_mean_z = float(bbox_settings["min_mean_z"])
    max_boxes = int(bbox_settings["max_boxes"])

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    candidates: List[Dict[str, Any]] = []

    for label_idx in range(1, num_labels):
        x, y, w, h, area = stats[label_idx]
        if area < min_area or area > max_area or w == 0 or h == 0:
            continue

        if product_mask is not None and not _passes_foreground_filter(
            product_mask, int(x), int(y), int(w), int(h)
        ):
            continue

        region = z_map[y : y + h, x : x + w]
        mean_z = float(np.mean(region))
        max_z = float(np.max(region))
        if mean_z < min_mean_z:
            continue

        box: Dict[str, Any] = {
            "x": int(x),
            "y": int(y),
            "w": int(w),
            "h": int(h),
            "area": float(area),
            "mean_z": mean_z,
            "max_z": max_z,
            "score": mean_z,
        }
        if product_mask is not None:
            box["foreground_ratio"] = _box_foreground_ratio(
                product_mask, int(x), int(y), int(w), int(h)
            )

        candidates.append(box)

    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates[:max_boxes]


def _tensor_to_pil_image(tensor: torch.Tensor) -> Image.Image:
    array = tensor.squeeze(0).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((array * 255).astype(np.uint8), mode="RGB")


def _build_overlay_image(
    original: Image.Image,
    boxes: List[Dict[str, Any]],
    bbox_config: Dict[str, Any],
) -> Image.Image:
    """Optional server-side preview with boxes drawn (front-end should prefer JSON boxes)."""
    overlay_np = np.array(original.convert("RGB"))
    bbox_settings = bbox_config["bounding_boxes"]
    color = tuple(int(channel) for channel in bbox_settings["box_color_rgb"])
    thickness = int(bbox_settings.get("box_thickness", 2))

    for box in boxes:
        x, y, w, h = box["x"], box["y"], box["w"], box["h"]
        cv2.rectangle(overlay_np, (x, y), (x + w, y + h), color, thickness)

    return Image.fromarray(overlay_np, mode="RGB")


def _build_colored_heatmap_image(z_map: np.ndarray) -> Image.Image:
    heatmap_uint8 = _normalize_heatmap(z_map)
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    return Image.fromarray(cv2.cvtColor(colored, cv2.COLOR_BGR2RGB), mode="RGB")


def _normalize_array_to_gray_image(array: np.ndarray) -> Image.Image:
    if array.size == 0:
        return Image.fromarray(
            np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8), mode="L"
        )

    positive = array[array > 0]
    high = (
        float(np.percentile(positive, 99.5))
        if positive.size > 0
        else float(array.max())
    )
    if high <= 0:
        high = 1.0

    normalized = (np.clip(array / high, 0.0, 1.0) * 255).astype(np.uint8)
    return Image.fromarray(normalized, mode="L")


def predict(
    artifacts: ModelArtifacts,
    image: Image.Image,
    category: str,
) -> Dict[str, Any]:
    """Run category-aware reconstruction anomaly detection on a single RGB image."""
    category = validate_category(category)
    device = artifacts.device

    image_tensor, resized_original = _preprocess_image(image)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        reconstruction = artifacts.model(image_tensor)

    error_map = _compute_error_map(image_tensor, reconstruction)
    z_map = _compute_z_map(error_map, category, artifacts.error_profiles)
    top_1_z_score = _compute_top_1_z_score(z_map)
    threshold = _get_category_threshold(artifacts.thresholds, category)
    is_anomaly = top_1_z_score > threshold
    status = "anomaly" if is_anomaly else "normal"

    original_rgb = np.array(resized_original.convert("RGB"), dtype=np.float32) / 255.0
    product_mask = compute_product_mask(original_rgb, category)
    z_map_masked = z_map * product_mask

    mask = _build_localization_mask(z_map_masked, artifacts.bbox_config)
    boxes = _extract_approximate_boxes(
        z_map_masked, mask, artifacts.bbox_config, product_mask
    )

    return {
        "status": status,
        "is_anomaly": is_anomaly,
        "category": category,
        "score_name": SCORE_NAME,
        "anomaly_score": top_1_z_score,
        "threshold": threshold,
        "boxes": boxes,
        "error_mean": float(np.mean(error_map)),
        "z_map_max": float(np.max(z_map)),
        "bbox_method": BBOX_METHOD,
        "original_image": resized_original,
        "reconstructed_image": _tensor_to_pil_image(reconstruction),
        "heatmap_colored": _build_colored_heatmap_image(z_map),
        "binary_mask": Image.fromarray(mask, mode="L"),
        "overlay": _build_overlay_image(resized_original, boxes, artifacts.bbox_config),
        "error_map_gray": _normalize_array_to_gray_image(error_map),
        "z_map_gray": _normalize_array_to_gray_image(z_map),
        "product_mask_gray": Image.fromarray(
            (product_mask * 255).astype(np.uint8), mode="L"
        ),
    }
