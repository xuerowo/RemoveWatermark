from __future__ import annotations

import base64
import re
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from .masking import build_detection_mask
from .models import Detection, WatermarkRemoverConfig
from .templates import load_template


THUMBNAIL_SIZE = (224, 224)
THUMBNAIL_BG = (16, 19, 17)
THUMBNAIL_DIVIDER = (65, 74, 68)
THUMBNAIL_EXTREME_RATIO = 3.0


def _unique_archive_name(filename: str, used_names: set[str]) -> str:
    path = Path(filename)
    stem = path.stem or "image"
    suffix = path.suffix
    candidate = path.name
    counter = 2
    while candidate in used_names:
        candidate = f"{stem}_{counter}{suffix}"
        counter += 1
    used_names.add(candidate)
    return candidate


def _ascii_download_filename(filename: str) -> str:
    fallback = re.sub(r"[^A-Za-z0-9._-]+", "_", filename).strip("._")
    return fallback or "download"


def _file_version(path: Path) -> str:
    try:
        stat = path.stat()
    except OSError:
        return ""
    return f"{stat.st_mtime_ns}-{stat.st_size}"


def _decode_data_url(data_url: str) -> bytes:
    if "," not in data_url:
        raise ValueError("Invalid image data URL.")
    _, encoded = data_url.split(",", 1)
    return base64.b64decode(encoded)


def _decode_rgb_data_url(data_url: str, shape: tuple[int, int]) -> np.ndarray:
    with Image.open(BytesIO(_decode_data_url(data_url))) as source:
        image = source.convert("RGB")
    if image.size != (shape[1], shape[0]):
        image = image.resize((shape[1], shape[0]), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def _decode_mask_data_url(data_url: str, shape: tuple[int, int]) -> np.ndarray:
    with Image.open(BytesIO(_decode_data_url(data_url))) as image:
        return _image_to_mask(image, shape)


def _image_to_mask(image: Image.Image, shape: tuple[int, int]) -> np.ndarray:
    if image.mode == "RGBA":
        rgba = np.asarray(image, dtype=np.uint8)
        mask = np.maximum(rgba[..., 3], np.max(rgba[..., :3], axis=2))
        mask_image = Image.fromarray(mask, mode="L")
    else:
        mask_image = image.convert("L")
    if mask_image.size != (shape[1], shape[0]):
        mask_image = mask_image.resize((shape[1], shape[0]), Image.Resampling.NEAREST)
    return np.asarray(mask_image, dtype=np.uint8)


def _normalize_mask_u8(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    mask_array = np.asarray(mask)
    if mask_array.ndim == 3:
        mask_array = np.max(mask_array[..., :3], axis=2)
    if mask_array.size and np.issubdtype(mask_array.dtype, np.floating) and float(np.nanmax(mask_array)) <= 1.0:
        mask_array = mask_array * 255.0
    mask_image = Image.fromarray(np.clip(mask_array, 0, 255).astype(np.uint8), mode="L")
    if mask_image.size != (shape[1], shape[0]):
        mask_image = mask_image.resize((shape[1], shape[0]), Image.Resampling.NEAREST)
    return np.asarray(mask_image, dtype=np.uint8)


def _save_rgb(path: Path, image_rgb: np.ndarray) -> None:
    Image.fromarray(np.clip(image_rgb, 0, 255).astype(np.uint8), mode="RGB").save(path)


def _render_side_by_side(left_rgb: np.ndarray, right_rgb: np.ndarray) -> np.ndarray:
    left = np.clip(left_rgb, 0, 255).astype(np.uint8)
    right = np.clip(right_rgb, 0, 255).astype(np.uint8)
    separator = np.full((left.shape[0], 8, 3), 32, dtype=np.uint8)
    return np.concatenate([left, separator, right], axis=1)


def _render_mask_overlay(image_rgb: np.ndarray, mask_u8: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    base = np.clip(image_rgb, 0, 255).astype(np.float32)
    mask = (np.clip(mask_u8, 0, 255).astype(np.float32) / 255.0)[..., None]
    tint = np.zeros_like(base, dtype=np.float32)
    tint[..., 0] = color[0]
    tint[..., 1] = color[1]
    tint[..., 2] = color[2]
    alpha = mask * 0.55
    return np.clip(base * (1.0 - alpha) + tint * alpha, 0.0, 255.0).astype(np.uint8)


def _render_cleanup_change(original_rgb: np.ndarray, restored_rgb: np.ndarray) -> np.ndarray:
    change = _change_map(original_rgb, restored_rgb)
    heat = np.clip(change * 8.0, 0.0, 255.0).astype(np.uint8)
    change_rgb = np.zeros((*heat.shape, 3), dtype=np.uint8)
    change_rgb[..., 0] = heat
    change_rgb[..., 1] = np.clip(heat.astype(np.float32) * 0.45, 0, 255).astype(np.uint8)
    return change_rgb


def _render_detection_overlay(image_rgb: np.ndarray, detections: list[dict[str, Any]]) -> np.ndarray:
    image = Image.fromarray(np.clip(image_rgb, 0, 255).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    stroke = max(2, int(round(min(image.size) / 300)))
    for detection in detections:
        bbox = _clamped_box(detection.get("bbox"), image.size)
        content_bbox = _clamped_box(detection.get("content_bbox"), image.size)
        if bbox is not None:
            draw.rectangle(bbox, outline=(255, 96, 96, 255), width=stroke)
        if content_bbox is not None:
            draw.rectangle(content_bbox, outline=(94, 196, 167, 255), width=stroke)
        if bbox is not None:
            label = _detection_label(detection)
            if label:
                text_origin = (bbox[0], max(0, bbox[1] - 16))
                draw.rectangle(
                    (text_origin[0], text_origin[1], text_origin[0] + min(220, 8 * len(label) + 8), text_origin[1] + 15),
                    fill=(0, 0, 0, 150),
                )
                draw.text((text_origin[0] + 4, text_origin[1] + 1), label, fill=(255, 255, 255, 255))
    return np.asarray(image, dtype=np.uint8)


def _clamped_box(value: object, image_size: tuple[int, int]) -> tuple[int, int, int, int] | None:
    if not isinstance(value, list | tuple) or len(value) != 4:
        return None
    width, height = image_size
    try:
        x1, y1, box_width, box_height = [int(round(float(item))) for item in value]
    except (TypeError, ValueError):
        return None
    x2 = x1 + box_width
    y2 = y1 + box_height
    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width - 1, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _detection_label(detection: dict[str, Any]) -> str:
    template = str(detection.get("template") or "").encode("ascii", errors="ignore").decode("ascii").strip()
    score = detection.get("score")
    if isinstance(score, int | float):
        return f"{template} {score:.2f}".strip()
    return template


def _change_map(original_rgb: np.ndarray, restored_rgb: np.ndarray) -> np.ndarray:
    original = np.clip(original_rgb, 0, 255).astype(np.float32)
    restored = np.clip(restored_rgb, 0, 255).astype(np.float32)
    return np.mean(np.abs(restored - original), axis=2)


def thumbnail_bytes(image_path: Path, *, segment_extreme: bool = True) -> tuple[bytes, str]:
    with Image.open(image_path) as image:
        image = _prepare_thumbnail_source(image)
        image = _build_thumbnail(image, segment_extreme=segment_extreme)
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=82, optimize=True)
        return buffer.getvalue(), "image/jpeg"


def template_mask_preview_bytes(
    template_path: Path,
    config: WatermarkRemoverConfig,
    *,
    thumbnail: bool = True,
) -> tuple[bytes, str]:
    template = load_template(template_path)
    width, height = template.size
    background = np.broadcast_to(
        np.clip(template.background_rgb, 0, 255).astype(np.uint8).reshape(1, 1, 3),
        (height, width, 3),
    )
    detection = Detection(
        template_name=template.name,
        bbox=(0, 0, width, height),
        content_bbox=(0, 0, width, height),
        scale=1.0,
        score=1.0,
        color_score=1.0,
        strength=1.0,
        method="template_preview",
        objective=0.0,
        clip_ratio=0.0,
        residual=0.0,
        watermark_correlation=1.0,
    )
    mask = (build_detection_mask(background, [detection], [template], config) * 255.0).round().astype(np.uint8)

    with Image.open(template_path) as image:
        base = _prepare_thumbnail_source(image)
    if base.size != (width, height):
        base = base.resize((width, height), Image.Resampling.BILINEAR)
    preview = _render_mask_overlay(np.asarray(base, dtype=np.uint8), mask, (94, 196, 167))
    image = Image.fromarray(preview, mode="RGB")
    if thumbnail:
        image = _build_thumbnail(image, segment_extreme=False)
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue(), "image/png"


def _prepare_thumbnail_source(image: Image.Image) -> Image.Image:
    image = ImageOps.exif_transpose(image)
    if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, color=THUMBNAIL_BG)
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background
    if image.mode != "RGB":
        return image.convert("RGB")
    return image.copy()


def _build_thumbnail(image: Image.Image, *, segment_extreme: bool = True) -> Image.Image:
    width, height = image.size
    target_width, target_height = THUMBNAIL_SIZE
    canvas = Image.new("RGB", THUMBNAIL_SIZE, color=THUMBNAIL_BG)
    if width <= 0 or height <= 0:
        return canvas

    source_ratio = width / height
    target_ratio = target_width / target_height
    if segment_extreme and source_ratio < target_ratio / THUMBNAIL_EXTREME_RATIO:
        return _segmented_thumbnail(image, axis="vertical")
    if segment_extreme and source_ratio > target_ratio * THUMBNAIL_EXTREME_RATIO:
        return _segmented_thumbnail(image, axis="horizontal")

    preview = image.copy()
    preview.thumbnail(THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
    x = (target_width - preview.width) // 2
    y = (target_height - preview.height) // 2
    canvas.paste(preview, (x, y))
    return canvas


def _segmented_thumbnail(image: Image.Image, *, axis: str) -> Image.Image:
    target_width, target_height = THUMBNAIL_SIZE
    canvas = Image.new("RGB", THUMBNAIL_SIZE, color=THUMBNAIL_BG)
    gap = 2
    tile_width = (target_width - gap * 2) // 3
    anchors = (0.0, 0.5, 1.0)
    for index, anchor in enumerate(anchors):
        tile_x = index * (tile_width + gap)
        width = target_width - tile_x if index == len(anchors) - 1 else tile_width
        tile_ratio = width / target_height
        if axis == "vertical":
            crop_width = image.width
            crop_height = min(image.height, max(1, round(crop_width / tile_ratio)))
            source_y = round((image.height - crop_height) * anchor)
            crop_box = (0, source_y, image.width, source_y + crop_height)
        else:
            crop_height = image.height
            crop_width = min(image.width, max(1, round(crop_height * tile_ratio)))
            source_x = round((image.width - crop_width) * anchor)
            crop_box = (source_x, 0, source_x + crop_width, image.height)
        tile = image.crop(crop_box).resize((width, target_height), Image.Resampling.LANCZOS)
        canvas.paste(tile, (tile_x, 0))
        if index < len(anchors) - 1:
            divider_x = tile_x + width
            ImageDraw.Draw(canvas).rectangle((divider_x, 0, divider_x + gap - 1, target_height), fill=THUMBNAIL_DIVIDER)
    return canvas
