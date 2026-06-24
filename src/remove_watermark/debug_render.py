from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .models import LAMA_MODE, Detection


def normalize_debug_map(map_array: np.ndarray | None, shape: tuple[int, int], *, smooth: bool = False) -> np.ndarray:
    if map_array is None:
        return np.zeros(shape, dtype=np.float32)
    data = np.asarray(map_array, dtype=np.float32)
    if data.ndim == 3:
        data = data[..., 0]
    if data.shape != shape:
        interpolation = cv2.INTER_LINEAR if smooth else cv2.INTER_NEAREST
        data = cv2.resize(data, (shape[1], shape[0]), interpolation=interpolation)
    maximum = float(data.max()) if data.size else 0.0
    if maximum > 1.0:
        data = data / maximum
    return np.clip(data, 0.0, 1.0).astype(np.float32)


def _normalize_debug_mask(mask: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    return normalize_debug_map(mask, shape)


def _normalize_residual_debug_map(residual_map: np.ndarray) -> np.ndarray:
    maximum = max(float(residual_map.max()) if residual_map.size else 0.0, 1.0)
    return np.clip(residual_map / maximum, 0.0, 1.0).astype(np.float32)


def _normalize_change_debug_map(before_patch: np.ndarray, after_patch: np.ndarray) -> np.ndarray:
    change = np.mean(np.abs(after_patch.astype(np.float32) - before_patch.astype(np.float32)), axis=2)
    return np.clip(change / 64.0, 0.0, 1.0).astype(np.float32)


def _json_safe_debug_value(value: object) -> object:
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, int) or value is None or isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe_debug_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe_debug_value(item) for key, item in value.items()}
    return str(value)


def render_debug_overlay(image_rgb: np.ndarray, detections: Iterable[Detection]) -> np.ndarray:
    canvas = Image.fromarray(np.clip(image_rgb, 0, 255).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(canvas)
    color_map = {LAMA_MODE: (0, 150, 136), "pending": (244, 180, 0)}
    for detection in detections:
        x, y, width, height = detection.bbox
        color = color_map.get(detection.method, (244, 180, 0))
        draw.rectangle((x, y, x + width, y + height), outline=color, width=2)
        label = _debug_overlay_label(detection)
        draw.rectangle((x, max(0, y - 16), x + max(60, len(label) * 7), y), fill=color)
        draw.text((x + 3, max(0, y - 14)), label, fill=(255, 255, 255))
    return np.asarray(canvas, dtype=np.uint8)


def _debug_overlay_label(detection: Detection) -> str:
    label = f"{detection.method} {detection.template_name} {detection.score:.2f}"
    return label.encode("latin-1", errors="replace").decode("latin-1")


def render_debug_map(map_array: np.ndarray) -> np.ndarray:
    data = np.asarray(map_array, dtype=np.float32)
    if data.ndim == 3 and data.shape[2] >= 3:
        maximum = float(data.max()) if data.size else 0.0
        scaled = data * 255.0 if maximum <= 1.0 else data
        return np.clip(scaled[..., :3], 0.0, 255.0).astype(np.uint8)
    if data.ndim == 3:
        data = data[..., 0]
    if data.size == 0:
        data = np.zeros((1, 1), dtype=np.float32)
    maximum = float(data.max())
    scaled = data * 255.0 if maximum <= 1.0 else data / max(maximum, 1.0) * 255.0
    gray = np.clip(scaled, 0.0, 255.0).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=2)

