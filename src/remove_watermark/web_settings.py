from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any

from .ai_detection import AIDetectionConfig
from .models import WatermarkRemoverConfig
from .web_models import OperationCancelled


DEFAULT_TEMPLATE_PATHS = (Path("templates"),)
DEFAULT_OUTPUT_DIR = Path("output")
DEFAULT_WEB_TEMPLATE_SCORE_THRESHOLD = 0.50
DEFAULT_WEB_TEMPLATE_MIN_SCALE = 1.0
DEFAULT_WEB_TEMPLATE_MAX_SCALE = 1.0
TEMP_INPUT_ROOT_NAME = ".editor_imports"
WORKSPACE_INPUT_ROOT_NAME = ".editor_workspace"
DETECTOR_MODES = {"template", "sam3"}


def _path_key(path: Path) -> str:
    try:
        return str(path.resolve())
    except OSError:
        return str(path.absolute())


def _raise_if_cancelled(cancel_callback: Any | None) -> None:
    if cancel_callback is not None and cancel_callback():
        raise OperationCancelled("作業已中斷")


def _normalize_detector_mode(detector: str | None) -> str:
    value = (detector or "template").strip().lower() or "template"
    if value not in DETECTOR_MODES:
        raise ValueError(f"Unsupported detector mode: {detector}")
    return value


def _normalize_operation_mode(mode: str | None) -> str:
    value = (mode or "").strip()
    if value not in {"detect", "process", "detectProcess"}:
        raise ValueError(f"Unsupported operation mode: {mode}")
    return value


def _bool_setting(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _config_bool_setting(settings: dict[str, Any] | None, key: str, default: bool, kind: str) -> bool:
    if not isinstance(settings, dict) or key not in settings:
        return default
    value = settings[key]
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    elif isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    raise ValueError(f"Invalid {kind} setting: {key}")


def _ai_bool_setting(settings: dict[str, Any] | None, key: str, default: bool) -> bool:
    return _config_bool_setting(settings, key, default, "AI")


def _template_bool_setting(settings: dict[str, Any] | None, key: str, default: bool) -> bool:
    return _config_bool_setting(settings, key, default, "template")


def _normalize_batch_jobs(requested_jobs: int, image_count: int) -> int:
    if image_count <= 1:
        return 1
    if requested_jobs == 0:
        return max(1, min(image_count, os.cpu_count() or 1))
    return max(1, min(image_count, requested_jobs))


def _ai_settings_payload(config: AIDetectionConfig) -> dict[str, Any]:
    return {
        "sam3ConfidenceThreshold": float(config.sam3_confidence_threshold),
        "boxThreshold": float(config.box_threshold),
        "maxBoxAreaRatio": float(config.max_box_area_ratio),
        "nmsIouThreshold": float(config.nms_iou_threshold),
        "maxDetections": int(config.max_detections),
        "maskThreshold": float(config.mask_threshold),
        "maskDilatePixels": int(config.mask_dilate_pixels),
        "fallbackToBoxes": bool(config.fallback_to_boxes),
        "sam3MaxSide": int(config.sam3_max_side),
        "sam3TileOverlapRatio": float(config.sam3_tile_overlap_ratio),
    }


def _template_settings_payload(config: WatermarkRemoverConfig) -> dict[str, Any]:
    return {
        "scoreThreshold": float(config.score_threshold),
        "minScale": float(config.min_scale),
        "maxScale": float(config.max_scale),
        "scaleStep": float(config.scale_step),
        "maxDetections": int(config.max_detections),
        "nmsIouThreshold": float(config.nms_iou_threshold),
        "edgeScoreThreshold": float(config.edge_score_threshold),
        "colorScoreThreshold": float(config.color_score_threshold),
        "supportCorrelationThreshold": float(config.support_correlation_threshold),
        "maskDilateIterations": int(config.mask_dilate_iterations),
        "maskDilateMaxBodyRatio": float(config.mask_dilate_max_body_ratio),
        "maskEdgeFeatherPixels": float(config.mask_edge_feather_pixels),
        "maskUnifyBody": bool(config.mask_unify_body),
        "maskContourClosePixels": float(config.mask_contour_close_pixels),
        "maskBodyGapRatio": float(config.mask_body_gap_ratio),
        "sam3RefineMask": bool(config.sam3_refine_mask),
    }


def _ai_float_setting(
    settings: dict[str, Any] | None,
    key: str,
    default: float,
    lower: float,
    upper: float,
) -> float:
    if not isinstance(settings, dict) or key not in settings:
        return default
    try:
        value = float(settings[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid AI setting: {key}") from exc
    if not lower < value <= upper:
        raise ValueError(f"Invalid AI setting: {key}")
    return value


def _ai_float_setting_closed(
    settings: dict[str, Any] | None,
    key: str,
    default: float,
    lower: float,
    upper: float,
) -> float:
    if not isinstance(settings, dict) or key not in settings:
        return default
    try:
        value = float(settings[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid AI setting: {key}") from exc
    if value < lower or value > upper:
        raise ValueError(f"Invalid AI setting: {key}")
    return value


def _template_float_setting(
    settings: dict[str, Any] | None,
    key: str,
    default: float,
    lower: float,
    upper: float,
) -> float:
    if not isinstance(settings, dict) or key not in settings:
        return default
    try:
        value = float(settings[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid template setting: {key}") from exc
    if not lower < value <= upper:
        raise ValueError(f"Invalid template setting: {key}")
    return value


def _template_float_setting_closed(
    settings: dict[str, Any] | None,
    key: str,
    default: float,
    lower: float,
    upper: float,
) -> float:
    if not isinstance(settings, dict) or key not in settings:
        return default
    try:
        value = float(settings[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid template setting: {key}") from exc
    if value < lower or value > upper:
        raise ValueError(f"Invalid template setting: {key}")
    return value


def _ai_int_setting(
    settings: dict[str, Any] | None,
    key: str,
    default: int,
    lower: int,
    upper: int,
) -> int:
    if not isinstance(settings, dict) or key not in settings:
        return default
    try:
        value = int(round(float(settings[key])))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid AI setting: {key}") from exc
    if value < lower or value > upper:
        raise ValueError(f"Invalid AI setting: {key}")
    return value


def _template_int_setting(
    settings: dict[str, Any] | None,
    key: str,
    default: int,
    lower: int,
    upper: int,
) -> int:
    if not isinstance(settings, dict) or key not in settings:
        return default
    try:
        value = int(round(float(settings[key])))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid template setting: {key}") from exc
    if value < lower or value > upper:
        raise ValueError(f"Invalid template setting: {key}")
    return value


def _new_temporary_input_path(output_dir: Path) -> Path:
    return output_dir / TEMP_INPUT_ROOT_NAME / f"input-{uuid.uuid4().hex[:10]}"


def _new_workspace_input_path(output_dir: Path) -> Path:
    return output_dir / WORKSPACE_INPUT_ROOT_NAME / "images" / f"input-{uuid.uuid4().hex[:10]}"
