from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# --------------------------------------------------------------------------
# 1. ARQUITETURA DO MODELO (Deve ser idêntica à do treino)
# --------------------------------------------------------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            self.encoder.append(DoubleConv(in_channels, feature))
            in_channels = feature

        for feature in reversed(features):
            self.decoder.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.decoder.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.encoder:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)
            skip_connection = skip_connections[idx // 2]
            if x.shape != skip_connection.shape:
                x = transforms.functional.resize(x, size=skip_connection.shape[2:])
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[idx + 1](concat_skip)

        x = self.final_conv(x)
        return torch.sigmoid(x)

# --------------------------------------------------------------------------
# 2. FUNÇÃO DE CONFIGURAÇÃO (Carrega modelo e thresholds)
# --------------------------------------------------------------------------
LOGGER = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def setup_model_and_config(
    model_path: str | os.PathLike[str] | None = None,
    config_path: str | os.PathLike[str] | None = None,
) -> Tuple[UNet, Dict[str, Any]]:
    """
    Load the trained U-Net model and configuration thresholds.

    The function is cached to avoid reloading the model multiple times.
    """
    device = torch.device("cpu")

    base_dir = Path(__file__).resolve().parent
    model_path = Path(model_path or base_dir / "models" / "bottle_unet_best.pth")
    config_path = Path(config_path or base_dir / "models" / "bottle_unet_config.json")

    with config_path.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")

    file_size = model_path.stat().st_size
    if file_size < 1_000:
        raise ValueError(
            f"Model file found at {model_path} is unexpectedly small ({file_size} bytes). "
            "Check whether Git LFS pulled the weights correctly."
        )

    model = UNet().to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    LOGGER.info("Model loaded from %s (%.2f MB)", model_path, file_size / 1_048_576)
    return model, config

# --------------------------------------------------------------------------
# 3. FUNÇÃO DE PREDIÇÃO
# --------------------------------------------------------------------------
def predict(model: UNet, config: Dict[str, Any], image: Image.Image) -> Dict[str, Any]:
    """
    Run a forward pass on a single image and return the prediction payload.
    """
    device = torch.device("cpu")

    class_threshold = float(config["classification_threshold"])
    pixel_threshold = float(config["pixel_visualization_threshold"])

    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ]
    )
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        reconstruction = model(image_tensor)
        error = torch.nn.functional.mse_loss(reconstruction, image_tensor)

    prediction_text = (
        "Anomaly Detected" if error.item() > class_threshold else "Normal"
    )

    anomaly_map = torch.mean(torch.abs(image_tensor - reconstruction), dim=1, keepdim=True)
    anomaly_map_scaled = (anomaly_map.squeeze().cpu().numpy() * 255).astype(np.uint8)
    _, binary_mask = cv2.threshold(
        anomaly_map_scaled, pixel_threshold, 255, cv2.THRESH_BINARY
    )

    bounding_boxes, mask_boxes = extract_bounding_boxes(
        results={
            "original_image": image,
            "anomaly_map_scaled": anomaly_map_scaled,
            "binary_mask": binary_mask,
        },
        config=config,
    )

    results_dict = {
        "prediction": prediction_text,
        "error": error.item(),
        "original_image": image,
        "reconstructed_image": reconstruction,
        "anomaly_map_scaled": anomaly_map_scaled,
        "binary_mask": binary_mask,
        "pixel_threshold": pixel_threshold,
        "bounding_boxes": bounding_boxes,
        "detections": [
            {
                "label": box["label"],
                "confidence": box["score"],
                "box": box["normalized_box"],
                "normalized": True,
            }
            for box in bounding_boxes
        ],
        "mask_boxes": mask_boxes,
    }

    return results_dict

# --------------------------------------------------------------------------
# 4. FUNÇÕES DE VISUALIZAÇÃO
# --------------------------------------------------------------------------
def get_anomaly_map_image(results: Dict[str, Any]) -> Image.Image:
    """
    Return the anomaly map as a grayscale PIL image.
    """
    anomaly_map = results["anomaly_map_scaled"]
    return Image.fromarray(anomaly_map, mode="L")

def get_binary_mask_image(results: Dict[str, Any]) -> Image.Image:
    """
    Return the binary mask as a grayscale PIL image.
    """
    binary_mask = results["binary_mask"]
    return Image.fromarray(binary_mask, mode="L")

def get_heatmap_image(results: Dict[str, Any]) -> Image.Image:
    """
    Return a heatmap overlay as a RGB PIL image.
    """
    original_np = np.array(results["original_image"].resize((256, 256)))
    heatmap = cv2.applyColorMap(results["anomaly_map_scaled"], cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original_np, 0.6, heatmap, 0.4, 0)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    return Image.fromarray(overlay_rgb)

def display_bounding_box(results: Dict[str, Any], config: Dict[str, Any]) -> Image.Image:
    """
    Return the original image with an anomaly bounding box drawn on top.
    """
    original_resized = results["original_image"].resize((256, 256))
    original_np = np.array(original_resized)

    mask_boxes: Sequence[Tuple[int, int, int, int]] = results.get("mask_boxes", [])

    if not mask_boxes:
        _, mask_boxes = extract_bounding_boxes(
            {
                "original_image": results["original_image"],
                "anomaly_map_scaled": results["anomaly_map_scaled"],
                "binary_mask": results["binary_mask"],
            },
            config,
        )

    result_image = original_np.copy()

    for x_start, y_start, x_end, y_end in mask_boxes:
        cv2.rectangle(result_image, (x_start, y_start), (x_end, y_end), (0, 194, 255), 2)
        cv2.putText(
            result_image,
            "Anomaly",
            (x_start, max(0, y_start - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 194, 255),
            1,
            cv2.LINE_AA,
        )

    result_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(result_rgb)


def extract_bounding_boxes(
    results: Dict[str, Any], config: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], List[Tuple[int, int, int, int]]]:
    """
    Extract bounding boxes from the anomaly map and return both pixel boxes and
    mask-space coordinates (for internal use).
    """
    original_image: Image.Image = results["original_image"]
    mask = results["anomaly_map_scaled"]

    width_orig, height_orig = original_image.size
    scale_x = width_orig / 256.0
    scale_y = height_orig / 256.0

    bounding_box_threshold = float(config.get("bounding_box_threshold", 1.5))
    _, sensitive_mask = cv2.threshold(
        mask,
        bounding_box_threshold,
        255,
        cv2.THRESH_BINARY,
    )

    dilation_iterations = int(config.get("dilation_iterations", 2))
    kernel = np.ones((3, 3), np.uint8)
    sensitive_mask = cv2.dilate(sensitive_mask, kernel, iterations=dilation_iterations)

    contours, _ = cv2.findContours(
        sensitive_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes: List[Dict[str, Any]] = []
    mask_boxes: List[Tuple[int, int, int, int]] = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        margin_x = int(w * 0.15)
        margin_y = int(h * 0.15)

        x_start = max(0, x - margin_x)
        y_start = max(0, y - margin_y)
        x_end = min(256, x + w + margin_x)
        y_end = min(256, y + h + margin_y)

        mask_boxes.append((x_start, y_start, x_end, y_end))

        xmin = int(x_start * scale_x)
        ymin = int(y_start * scale_y)
        xmax = int(x_end * scale_x)
        ymax = int(y_end * scale_y)

        width = xmax - xmin
        height = ymax - ymin

        crop = mask[y_start:y_end, x_start:x_end]
        score = float(np.clip(crop.mean() / 255.0, 0.0, 1.0)) if crop.size else 0.0

        boxes.append(
            {
                "label": "Anomaly",
                "score": score,
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
                "width": width,
                "height": height,
                "box": [xmin, ymin, xmax, ymax],
                "normalized_box": [
                    xmin / width_orig if width_orig else 0.0,
                    ymin / height_orig if height_orig else 0.0,
                    width / width_orig if width_orig else 0.0,
                    height / height_orig if height_orig else 0.0,
                ],
                "normalized": False,
            }
        )

    return boxes, mask_boxes

