from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from .debug_render import _normalize_debug_mask
from .evidence_metrics import _filled_support_holes, _text_detail_mask
from .models import BodySourceResult, Detection, WatermarkRemoverConfig
from .templates import (
    ResizedTemplate,
    TemplateBundle,
    _binary_mask,
    _clean_mask,
    _close_mask,
    _compute_edge_map,
    _min_component_area,
    _remove_small_mask_components,
)


def build_detection_mask(
    image_rgb: np.ndarray,
    detections: Sequence[Detection],
    templates: TemplateBundle | Sequence[TemplateBundle],
    config: WatermarkRemoverConfig | None = None,
) -> np.ndarray:
    """Build a full-image LaMa mask preview for already accepted detections."""
    if not detections:
        return np.zeros(np.asarray(image_rgb).shape[:2], dtype=np.float32)

    runtime_config = config or WatermarkRemoverConfig(collect_debug_maps=False)
    template_list = [templates] if isinstance(templates, TemplateBundle) else list(templates)
    if not template_list:
        raise ValueError("At least one watermark template is required.")
    templates_by_name = {template.name: template for template in template_list}
    fallback_template = template_list[0]

    image = np.asarray(image_rgb, dtype=np.float32)
    height, width = image.shape[:2]
    full_mask = np.zeros((height, width), dtype=np.float32)
    for detection in detections:
        x, y, box_width, box_height = detection.bbox
        if box_width <= 0 or box_height <= 0:
            continue
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(width, x + box_width)
        y2 = min(height, y + box_height)
        if x2 <= x1 or y2 <= y1:
            continue
        patch = image[y1:y2, x1:x2].copy()
        template = templates_by_name.get(detection.template_name, fallback_template)
        patch_bbox = (x1, y1, x2 - x1, y2 - y1)
        resized = _aligned_template_for_patch(template, patch_bbox, detection.content_bbox or detection.bbox)
        lama_mask, _, _, _ = _build_lama_mask(patch, resized, runtime_config)
        full_mask[y1:y2, x1:x2] = np.maximum(
            full_mask[y1:y2, x1:x2],
            _normalize_debug_mask(lama_mask, (y2 - y1, x2 - x1)),
        )
    return np.clip(full_mask, 0.0, 1.0).astype(np.float32)
def _build_lama_mask(
    patch: np.ndarray,
    resized: ResizedTemplate,
    config: WatermarkRemoverConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, object], dict[str, np.ndarray]]:
    support_mask = resized.support_mask.astype(np.uint8)
    body_source_result = _template_body_source_mask(resized, config)
    body_source = body_source_result.mask
    if not np.any(body_source):
        body_source = support_mask
        body_source_result = BodySourceResult(body_source, "support_fallback")
    mask_body, mask_body_mode = _build_lama_mask_body(body_source, config)
    target = _soft_expand_mask(mask_body, config)

    text_detail_mask = _text_detail_mask(patch, resized, config).astype(bool)
    support = support_mask.astype(bool)
    text_detail = float(text_detail_mask[support].mean()) if np.any(support) else float(text_detail_mask.mean())
    support_pixels = int(support_mask.sum())
    body_source_pixels = int(body_source.sum())
    mask_body_pixels = int(mask_body.sum())
    effective_dilate_pixels = _effective_dilate_pixels(mask_body, config)
    lama_mask = target.astype(np.float32)
    stage_metrics = {
        "backend": "lama_direct",
        "mask_body_enabled": bool(config.mask_unify_body),
        "mask_body_mode": mask_body_mode,
        "mask_body_source_mode": body_source_result.mode,
        "mask_body_source_hole_ratio": body_source_result.hole_ratio,
        "mask_body_source_support_fill": body_source_result.support_fill,
        "mask_body_source_growth": body_source_result.growth,
        "mask_target_dilate_iterations": int(config.mask_dilate_iterations),
        "mask_target_effective_dilate_pixels": effective_dilate_pixels,
        "mask_body_source_pixels": body_source_pixels,
        "mask_body_source_components": _mask_component_count(body_source),
        "mask_body_pixels": mask_body_pixels,
        "mask_body_components": _mask_component_count(mask_body),
        "mask_body_area_growth": float(mask_body_pixels / max(body_source_pixels, support_pixels, 1)),
        "target_pixels": int((target >= 0.50).sum()),
        "lama_mask_pixels": int((lama_mask >= 0.50).sum()),
        "lama_mask_soft_pixels": int(((lama_mask > 0.0) & (lama_mask < 1.0)).sum()),
        "text_detail_mask_pixels": int(text_detail_mask.sum()),
        "text_detail": text_detail,
    }
    mask_debug_maps = {}
    if config.collect_debug_maps:
        mask_debug_maps = {
            "body_source": body_source.astype(np.float32),
            "mask_body": mask_body.astype(np.float32),
        }
    return lama_mask.astype(np.float32), text_detail_mask.astype(np.float32), stage_metrics, mask_debug_maps
def _aligned_template_for_patch(
    template: TemplateBundle,
    patch_bbox: tuple[int, int, int, int],
    content_bbox: tuple[int, int, int, int],
) -> ResizedTemplate:
    patch_x, patch_y, patch_width, patch_height = patch_bbox
    content_x, content_y, content_width, content_height = content_bbox
    content = template.resized_to(content_width, content_height)
    offset_x = int(content_x - patch_x)
    offset_y = int(content_y - patch_y)

    template_rgb = np.broadcast_to(content.background_rgb.reshape(1, 1, 3), (patch_height, patch_width, 3)).copy().astype(np.float32)
    rgb = np.zeros((patch_height, patch_width, 3), dtype=np.float32)
    gray = np.zeros((patch_height, patch_width), dtype=np.float32)
    alpha = np.zeros((patch_height, patch_width), dtype=np.float32)
    alpha_rgb = np.zeros((patch_height, patch_width, 3), dtype=np.float32)
    support = np.zeros((patch_height, patch_width), dtype=np.uint8)

    src_left = max(0, -offset_x)
    src_top = max(0, -offset_y)
    dst_left = max(0, offset_x)
    dst_top = max(0, offset_y)
    copy_width = min(content_width - src_left, patch_width - dst_left)
    copy_height = min(content_height - src_top, patch_height - dst_top)
    if copy_width > 0 and copy_height > 0:
        src = (slice(src_top, src_top + copy_height), slice(src_left, src_left + copy_width))
        dst = (slice(dst_top, dst_top + copy_height), slice(dst_left, dst_left + copy_width))
        template_rgb[dst] = content.template_rgb[src]
        rgb[dst] = content.rgb[src]
        gray[dst] = content.gray[src]
        alpha[dst] = content.alpha[src]
        alpha_rgb[dst] = content.alpha_rgb[src]
        support[dst] = content.support_mask[src]

    return ResizedTemplate(
        width=patch_width,
        height=patch_height,
        template_rgb=template_rgb,
        background_rgb=template.background_rgb.astype(np.float32),
        rgb=rgb,
        gray=gray,
        gray_u8=np.clip(gray, 0.0, 255.0).astype(np.uint8),
        alpha=alpha,
        alpha_rgb=alpha_rgb,
        support_mask=support,
        edge=_compute_edge_map(gray),
        polarity=template.polarity,
        chroma_strength=template.chroma_strength,
    )

def _template_visible_body_mask(resized: ResizedTemplate, config: WatermarkRemoverConfig) -> np.ndarray:
    return _template_body_source_mask(resized, config).mask


def _template_body_source_mask(resized: ResizedTemplate, config: WatermarkRemoverConfig) -> BodySourceResult:
    appearance = resized.rgb.astype(np.float32)
    visible_strength = np.linalg.norm(appearance, axis=2)
    support = _binary_mask(resized.support_mask)

    threshold = max(1.0, float(config.mask_body_visible_threshold))
    faint_threshold = max(1.0, threshold * max(0.0, float(config.mask_body_faint_threshold_ratio)))
    weak_visible = (visible_strength > faint_threshold).astype(np.uint8)
    if not np.any(support):
        return BodySourceResult(_clean_mask(weak_visible), "support_fallback")

    max_growth = max(1.0, float(config.mask_body_source_max_area_growth))
    filled_silhouette = _clean_mask(_filled_support_holes(support), min_area=_min_component_area(support))
    support_pixels = float(support.sum())
    silhouette_pixels = float(filled_silhouette.sum())
    silhouette_growth = silhouette_pixels / max(support_pixels, 1.0)
    hole_ratio = max(0.0, silhouette_pixels - support_pixels) / max(silhouette_pixels, 1.0)
    support_fill = support_pixels / max(silhouette_pixels, 1.0)
    if (
        silhouette_pixels > 0.0
        and silhouette_growth <= max_growth
        and hole_ratio >= config.mask_body_silhouette_min_hole_ratio
        and support_fill >= config.mask_body_silhouette_min_support_fill
    ):
        return BodySourceResult(
            filled_silhouette.astype(np.uint8),
            "filled_silhouette",
            hole_ratio,
            support_fill,
            silhouette_growth,
        )

    weak_connected = _seed_connected_mask(np.maximum(weak_visible, support), support)
    weak_growth = float(weak_connected.sum()) / max(support_pixels, 1.0)
    if weak_growth <= max_growth:
        return BodySourceResult(
            _clean_mask(weak_connected),
            "weak_connected",
            hole_ratio,
            support_fill,
            weak_growth,
        )

    strong_visible = (visible_strength > threshold).astype(np.uint8)
    strong_connected = _seed_connected_mask(np.maximum(strong_visible, support), support)
    strong_growth = float(strong_connected.sum()) / max(support_pixels, 1.0)
    return BodySourceResult(
        _clean_mask(strong_connected),
        "strong_connected_fallback",
        hole_ratio,
        support_fill,
        strong_growth,
    )


def _seed_connected_mask(mask: np.ndarray, seed: np.ndarray) -> np.ndarray:
    mask_u8 = _binary_mask(mask)
    seed_mask = _binary_mask(seed).astype(bool)
    if not np.any(mask_u8) or not np.any(seed_mask):
        return mask_u8

    count, labels, _, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if count <= 1:
        return mask_u8
    seed_labels = np.unique(labels[seed_mask & (labels > 0)])
    if seed_labels.size == 0:
        return np.zeros_like(mask_u8)
    return np.isin(labels, seed_labels).astype(np.uint8)


def _build_lama_mask_body(mask: np.ndarray, config: WatermarkRemoverConfig) -> tuple[np.ndarray, str]:
    base = _clean_mask(mask)
    if not config.mask_unify_body or not np.any(base):
        return base, "source"

    contour = _build_contour_mask_body(base, config)
    if contour is not None:
        max_growth = max(1.0, float(config.mask_body_max_area_growth))
        if float(contour.sum()) / max(float(base.sum()), 1.0) <= max_growth:
            return contour.astype(np.uint8), "contour"

    return _build_mask_body(base, config), "closed"


def _build_contour_mask_body(mask: np.ndarray, config: WatermarkRemoverConfig) -> np.ndarray | None:
    base = _binary_mask(mask)
    if not np.any(base):
        return None

    base = _close_mask(base, max(0.0, float(config.mask_contour_close_pixels)))
    return _clean_mask(base, min_area=1, fill_holes=True)


def _build_mask_body(mask: np.ndarray, config: WatermarkRemoverConfig) -> np.ndarray:
    base = _clean_mask(mask)
    if not config.mask_unify_body or not np.any(base):
        return base

    gap_ratio = max(0.0, float(config.mask_body_gap_ratio))
    if gap_ratio <= 0.0:
        return base

    ys, xs = np.where(base > 0)
    bbox_width = int(xs.max() - xs.min() + 1)
    bbox_height = int(ys.max() - ys.min() + 1)
    kernel_width = _odd_kernel_size(max(3, int(round(bbox_width * gap_ratio))))
    kernel_height = _odd_kernel_size(max(3, int(round(bbox_height * gap_ratio))))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, kernel_height))
    body = cv2.morphologyEx(base, cv2.MORPH_CLOSE, kernel, iterations=1)
    body = _filled_support_holes(body)
    if not np.any(body):
        return base

    max_growth = max(1.0, float(config.mask_body_max_area_growth))
    if float(body.sum()) / max(float(base.sum()), 1.0) > max_growth:
        return base

    return body.astype(np.uint8)


def _odd_kernel_size(value: int) -> int:
    return int(value + 1 if value % 2 == 0 else value)


def _mask_component_count(mask: np.ndarray) -> int:
    mask_u8 = _binary_mask(mask)
    if not np.any(mask_u8):
        return 0
    return int(cv2.connectedComponents(mask_u8, connectivity=8)[0] - 1)

def _soft_expand_mask(base_low: np.ndarray, config: WatermarkRemoverConfig) -> np.ndarray:
    scale = max(1, int(config.mask_antialias_scale))
    base_low = (base_low > 0).astype(np.uint8)
    base_low = _remove_small_mask_components(base_low, min_area=max(4, int(round(base_low.size * 0.0005))))
    base_low = _filled_support_holes(base_low)
    base = base_low
    if scale > 1:
        high_size = (base_low.shape[1] * scale, base_low.shape[0] * scale)
        base = cv2.resize(base, high_size, interpolation=cv2.INTER_NEAREST).astype(np.uint8)
    if not np.any(base):
        return np.zeros(base_low.shape, dtype=np.float32)

    effective_expand_pixels = _effective_dilate_pixels(base_low, config)
    expand_pixels = effective_expand_pixels * scale
    feather_pixels = max(0.1, float(config.mask_edge_feather_pixels)) * scale
    outside_distance = cv2.distanceTransform((base == 0).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    soft_high = np.clip((expand_pixels - outside_distance) / feather_pixels, 0.0, 1.0)
    soft_high[base > 0] = 1.0
    if scale > 1:
        soft = cv2.resize(soft_high, (base_low.shape[1], base_low.shape[0]), interpolation=cv2.INTER_AREA)
    else:
        soft = soft_high
    max_expand_pixels = effective_expand_pixels
    filled_low = base_low.astype(bool)
    low_distance = cv2.distanceTransform((~filled_low).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    soft[low_distance > max_expand_pixels] = 0.0
    return np.clip(soft, 0.0, 1.0).astype(np.float32)


def _effective_dilate_pixels(mask: np.ndarray, config: WatermarkRemoverConfig) -> float:
    requested = max(0.0, float(config.mask_dilate_iterations))
    ratio = max(0.0, float(config.mask_dilate_max_body_ratio))
    if requested <= 0.0 or ratio <= 0.0:
        return requested

    base = _binary_mask(mask)
    if not np.any(base):
        return requested
    ys, xs = np.where(base > 0)
    body_width = int(xs.max() - xs.min() + 1)
    body_height = int(ys.max() - ys.min() + 1)
    body_min_size = min(body_width, body_height)
    body_limit = max(1.0, float(body_min_size) * ratio)
    return min(requested, body_limit)
