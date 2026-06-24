from __future__ import annotations

import cv2
import numpy as np

from .debug_render import (
    _json_safe_debug_value,
    _normalize_change_debug_map,
    _normalize_debug_mask,
    _normalize_residual_debug_map,
)
from .evidence_metrics import _debug_mask_ratio, _residual_likelihood_map
from .lama import _run_lama
from .mask_builder import _build_lama_mask
from .models import (
    LAMA_MODE,
    FitResult,
    PatchContext,
    RestorationDiagnostics,
    RestorationResult,
    WatermarkRemoverConfig,
)
from .templates import ResizedTemplate


def restore_with_mask(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    config: WatermarkRemoverConfig | None = None,
) -> np.ndarray:
    """Run LaMa directly with a user-provided full-image mask."""
    image = np.asarray(image_rgb)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("image_rgb must be an RGB image array.")
    normalized_mask = _normalize_debug_mask(mask, image.shape[:2])
    if not np.any(normalized_mask > 0.0):
        raise ValueError("Manual mask is empty.")
    restored = _run_lama(image[..., :3], normalized_mask, config or WatermarkRemoverConfig())
    blended = _blend_restored_with_mask(image[..., :3], restored, normalized_mask)
    return np.clip(blended, 0, 255).astype(np.uint8)


def restore_original_regions(
    original_rgb: np.ndarray,
    edited_rgb: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Copy original pixels back into edited_rgb wherever mask is non-empty."""
    original = np.asarray(original_rgb)
    edited = np.asarray(edited_rgb)
    if original.shape != edited.shape:
        raise ValueError("original_rgb and edited_rgb must have the same shape.")
    if original.ndim != 3 or original.shape[2] < 3:
        raise ValueError("original_rgb and edited_rgb must be RGB image arrays.")
    normalized_mask = _normalize_debug_mask(mask, original.shape[:2]) > 0.0
    restored = edited[..., :3].copy()
    restored[normalized_mask] = original[..., :3][normalized_mask]
    return np.clip(restored, 0, 255).astype(np.uint8)

def _restore_lama_patch(
    patch: np.ndarray,
    resized: ResizedTemplate,
    fit: FitResult,
    score: float,
    config: WatermarkRemoverConfig,
    context: PatchContext | None = None,
) -> RestorationResult:
    lama_mask, text_detail_mask, stage_metrics, mask_debug_maps = _build_lama_mask(patch, resized, config)
    if int(lama_mask.sum()) < 1:
        raise RuntimeError("Expanded LaMa mask is empty.")
    lama_patch = _run_lama_with_context(patch, lama_mask, config, context)
    return _build_restoration_result(
        patch,
        lama_patch,
        resized,
        lama_mask,
        text_detail_mask,
        stage_metrics,
        fit,
        score,
        mask_debug_maps,
        collect_debug_maps=config.collect_debug_maps,
    )
def _run_lama_with_context(
    patch: np.ndarray,
    mask: np.ndarray,
    config: WatermarkRemoverConfig,
    context: PatchContext | None,
) -> np.ndarray:
    if context is None:
        return _run_lama(patch, mask, config).astype(np.float32)

    inner_x, inner_y, inner_width, inner_height = context.inner_bbox
    context_image = np.clip(context.image, 0.0, 255.0).astype(np.uint8)
    context_mask = np.zeros(context_image.shape[:2], dtype=np.uint8)
    inner_mask = (_normalize_debug_mask(mask, (inner_height, inner_width)) * 255.0).round().astype(np.uint8)
    context_mask[inner_y : inner_y + inner_height, inner_x : inner_x + inner_width] = inner_mask
    result = _run_lama(context_image, context_mask, config).astype(np.float32)
    return result[inner_y : inner_y + inner_height, inner_x : inner_x + inner_width]

def _build_restoration_result(
    original_patch: np.ndarray,
    restored_patch: np.ndarray,
    resized: ResizedTemplate,
    lama_mask: np.ndarray | None = None,
    text_detail_mask: np.ndarray | None = None,
    stage_metrics: dict[str, object] | None = None,
    fit: FitResult | None = None,
    score: float | None = None,
    extra_debug_maps: dict[str, np.ndarray] | None = None,
    collect_debug_maps: bool = True,
) -> RestorationResult:
    final_patch = _blend_restored_with_mask(original_patch, restored_patch, lama_mask) if lama_mask is not None else restored_patch
    diagnostics = _compute_restoration_diagnostics(
        original_patch,
        final_patch,
        resized,
        lama_mask,
        text_detail_mask,
        stage_metrics,
        fit,
        score,
        extra_debug_maps,
        collect_debug_maps=collect_debug_maps,
    )
    return RestorationResult(np.clip(final_patch, 0.0, 255.0).astype(np.float32), LAMA_MODE, diagnostics)


def _blend_restored_with_mask(original_rgb: np.ndarray, restored_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    original = np.asarray(original_rgb, dtype=np.float32)[..., :3]
    restored = _match_image_shape(np.asarray(restored_rgb, dtype=np.float32)[..., :3], original.shape[:2])
    alpha = _normalize_debug_mask(mask, original.shape[:2])[..., None]
    return original * (1.0 - alpha) + restored * alpha


def _match_image_shape(image_rgb: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    target_height, target_width = shape
    if image_rgb.shape[:2] == shape:
        return image_rgb
    if image_rgb.shape[0] >= target_height and image_rgb.shape[1] >= target_width:
        return image_rgb[:target_height, :target_width]
    return cv2.resize(image_rgb, (target_width, target_height), interpolation=cv2.INTER_LINEAR).astype(np.float32)


def _compute_restoration_diagnostics(
    original_patch: np.ndarray,
    restored_patch: np.ndarray,
    resized: ResizedTemplate,
    lama_mask: np.ndarray | None,
    text_detail_mask: np.ndarray | None,
    stage_metrics: dict[str, object] | None,
    fit: FitResult | None = None,
    score: float | None = None,
    extra_debug_maps: dict[str, np.ndarray] | None = None,
    collect_debug_maps: bool = True,
) -> RestorationDiagnostics:
    support = resized.support_mask.astype(bool)
    lama = _normalize_debug_mask(lama_mask, resized.support_mask.shape)
    raw_metrics = dict(stage_metrics or {})
    residual = _residual_likelihood_map(restored_patch, resized)
    residual_mean = float(residual[support].mean() / 255.0) if np.any(support) else float(residual.mean() / 255.0)
    if "text_detail" in raw_metrics:
        text_detail = float(raw_metrics["text_detail"])
        normalized_text_detail_mask = None
    else:
        normalized_text_detail_mask = _normalize_debug_mask(text_detail_mask, resized.support_mask.shape)
        text_detail = float(normalized_text_detail_mask[support].mean()) if np.any(support) else float(normalized_text_detail_mask.mean())
    lama_mask_ratio = _debug_mask_ratio(lama, resized.support_mask)
    metrics = raw_metrics
    metrics.update(
        {
            "fit_objective": float(fit.objective) if fit is not None else 0.0,
            "fit_watermark_correlation": float(fit.watermark_correlation) if fit is not None else 0.0,
            "detection_score": float(score) if score is not None else 0.0,
            "lama_mask_ratio": lama_mask_ratio,
        }
    )
    debug_maps = {}
    if collect_debug_maps:
        if normalized_text_detail_mask is None:
            normalized_text_detail_mask = _normalize_debug_mask(text_detail_mask, resized.support_mask.shape)
        debug_maps = {
            "residual_evidence": _normalize_residual_debug_map(residual),
            "text_detail_mask": normalized_text_detail_mask.astype(np.float32),
            "lama_mask": lama.astype(np.float32),
            "cleanup_change": _normalize_change_debug_map(original_patch, restored_patch),
        }
        for name, map_array in (extra_debug_maps or {}).items():
            debug_maps[name] = _normalize_debug_mask(map_array, resized.support_mask.shape)
    return RestorationDiagnostics(
        residual_score=residual_mean,
        text_detail=text_detail,
        lama_mask_ratio=lama_mask_ratio,
        stage_metrics=_json_safe_debug_value(metrics),
        debug_maps=debug_maps,
    )


def _build_patch_context(
    image_rgb: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    config: WatermarkRemoverConfig | None = None,
) -> PatchContext:
    config = config or WatermarkRemoverConfig()
    margin = max(config.min_context_margin, int(round(min(width, height) * config.context_margin_scale)))
    left = max(0, x - margin)
    top = max(0, y - margin)
    right = min(image_rgb.shape[1], x + width + margin)
    bottom = min(image_rgb.shape[0], y + height + margin)
    return PatchContext(image_rgb[top:bottom, left:right].copy(), (x - left, y - top, width, height))
