from __future__ import annotations

import gc
import re
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from PIL import Image

from .models import Detection, bbox_iou


DEFAULT_AI_PROMPT = "watermark. text watermark. transparent watermark. logo. stamp."
DEFAULT_SAM3_MODELSCOPE_MODEL = "facebook/sam3.1"
DEFAULT_SAM3_MODELSCOPE_FILE = "sam3.1_multiplex.pt"
DEFAULT_SAM3_CONFIDENCE_THRESHOLD = 0.05


@dataclass(slots=True)
class AIDetectionConfig:
    prompt: str = DEFAULT_AI_PROMPT
    sam3_model: str = DEFAULT_SAM3_MODELSCOPE_MODEL
    sam3_model_file: str = DEFAULT_SAM3_MODELSCOPE_FILE
    sam3_confidence_threshold: float = DEFAULT_SAM3_CONFIDENCE_THRESHOLD
    device: str = "auto"
    box_threshold: float = 0.20
    max_box_area_ratio: float = 0.35
    nms_iou_threshold: float = 0.30
    max_detections: int = 60
    mask_threshold: float = 0.0
    mask_dilate_pixels: int = 3
    fallback_to_boxes: bool = True
    sam3_max_side: int = 2048
    sam3_tile_overlap_ratio: float = 0.20
    sam3_crop_padding_ratio: float = 0.20
    sam3_crop_min_padding: int = 32


@dataclass(slots=True)
class AIDetectionResult:
    detections: list[Detection]
    mask: np.ndarray


@dataclass(slots=True)
class _PlacedMask:
    mask: np.ndarray
    bbox: tuple[int, int, int, int]


class AIDetectorUnavailableError(RuntimeError):
    pass


class Sam3TextWatermarkDetector:
    def __init__(self, config: AIDetectionConfig | None = None) -> None:
        self.config = config or AIDetectionConfig()
        self._torch = None
        self._sam_processor = None
        self._sam_model = None
        self._device = ""

    def detect(self, image_rgb: np.ndarray) -> AIDetectionResult:
        image = np.asarray(image_rgb, dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError("image_rgb must be an RGB image array.")

        labels = _prompt_to_labels(self.config.prompt)
        if not labels:
            raise ValueError("SAM 3.1 prompt is empty.")

        try:
            self._ensure_model_loaded()
            torch = self._torch
            assert torch is not None
            boxes, scores, detected_labels, masks = _detect_text_with_sam3_for_image(
                image[..., :3],
                labels,
                self._sam_processor,
                torch,
                self._device,
                self.config,
            )
            detections, kept_masks = _detections_and_masks_from_boxes(
                boxes,
                scores,
                detected_labels,
                masks,
                image.shape[:2],
                self.config,
            )
            if not detections:
                return AIDetectionResult(detections=[], mask=np.zeros(image.shape[:2], dtype=np.float32))

            mask = _combine_masks(kept_masks, detections, image.shape[:2], self.config)
            return AIDetectionResult(detections=detections, mask=mask)
        finally:
            self._release_transient_memory()

    def refine_mask(self, image_rgb: np.ndarray, detections: Sequence[Detection]) -> AIDetectionResult:
        image = np.asarray(image_rgb, dtype=np.uint8)
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError("image_rgb must be an RGB image array.")
        detection_list = list(detections)
        if not detection_list:
            return AIDetectionResult(detections=[], mask=np.zeros(image.shape[:2], dtype=np.float32))

        try:
            self._ensure_model_loaded()
            torch = self._torch
            assert torch is not None
            masks = _segment_boxes_with_sam3(
                image[..., :3],
                detection_list,
                self._sam_processor,
                torch,
                self._device,
                self.config,
            )
            mask = _combine_masks(masks, detection_list, image.shape[:2], self.config)
            return AIDetectionResult(detections=detection_list, mask=mask)
        finally:
            self._release_transient_memory()

    def _ensure_model_loaded(self) -> None:
        if self._sam_model is not None:
            return

        try:
            import torch
        except Exception as exc:  # pragma: no cover - exercised through integration environments
            raise AIDetectorUnavailableError("SAM 3.1 detection needs torch. Install the AI extras first.") from exc

        self._torch = torch
        self._device = _resolve_device(self.config.device, torch)
        try:
            self._sam_processor, self._sam_model = _load_sam3_model(self.config, self._device)
        except Exception as exc:  # pragma: no cover - downloads and device errors vary by machine
            raise AIDetectorUnavailableError(f"Failed to load SAM 3.1 detector model: {exc}") from exc

    def close(self) -> None:
        torch = self._torch
        self._sam_processor = None
        self._sam_model = None
        self._torch = None
        self._device = ""
        _release_torch_memory(torch)

    def _release_transient_memory(self) -> None:
        _release_torch_memory(self._torch)


def download_ai_models(config: AIDetectionConfig | None = None) -> None:
    download_sam3_model(config)


def download_sam3_model(config: AIDetectionConfig | None = None) -> str:
    runtime_config = config or AIDetectionConfig()
    return _resolve_modelscope_model_file(runtime_config.sam3_model, runtime_config.sam3_model_file)


def _empty_torch_cuda_cache(torch_module: Any | None) -> None:
    if torch_module is None:
        return
    try:
        cuda = getattr(torch_module, "cuda", None)
        if cuda is not None and cuda.is_available():
            empty_cache = getattr(cuda, "empty_cache", None)
            if callable(empty_cache):
                empty_cache()
            ipc_collect = getattr(cuda, "ipc_collect", None)
            if callable(ipc_collect):
                ipc_collect()
    except Exception:
        return


def _release_torch_memory(torch_module: Any | None) -> None:
    gc.collect()
    _empty_torch_cuda_cache(torch_module)


def _clear_sam3_state(sam3_processor: Any, state: Any | None) -> None:
    if state is None:
        return
    try:
        reset_all_prompts = getattr(sam3_processor, "reset_all_prompts", None)
        if callable(reset_all_prompts):
            reset_all_prompts(state)
    except Exception:
        pass
    try:
        clear = getattr(state, "clear", None)
        if callable(clear):
            clear()
    except Exception:
        pass


def _prompt_to_labels(prompt: str) -> list[str]:
    labels = []
    for part in re.split(r"[.;\n]+", prompt):
        label = part.strip().strip(",")
        if label:
            labels.append(label)
    return labels


def _resolve_device(requested: str, torch_module: Any) -> str:
    normalized = (requested or "auto").strip().lower()
    if normalized == "auto":
        return "cuda" if torch_module.cuda.is_available() else "cpu"
    if normalized == "cuda" and not torch_module.cuda.is_available():
        raise AIDetectorUnavailableError("SAM 3.1 detector was set to cuda, but CUDA is not available.")
    return normalized


def _load_sam3_model(config: AIDetectionConfig, device: str) -> tuple[Any, Any]:
    if not device.startswith("cuda"):
        raise AIDetectorUnavailableError("SAM 3.1 detection currently needs a CUDA GPU.")

    try:
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model
    except Exception as exc:  # pragma: no cover - depends on optional sam3 package
        raise AIDetectorUnavailableError(
            "SAM 3.1 detection needs Meta's sam3 package. Install the AI extras first."
        ) from exc

    checkpoint_path = _resolve_modelscope_model_file(config.sam3_model, config.sam3_model_file)
    model = build_sam3_image_model(
        checkpoint_path=checkpoint_path,
        load_from_HF=False,
        device=device,
    )
    if hasattr(model, "to"):
        model.to(device)
    if hasattr(model, "eval"):
        model.eval()
    processor = Sam3Processor(
        model,
        device=device,
        confidence_threshold=float(config.sam3_confidence_threshold),
    )
    return processor, model


def _detect_text_with_sam3(
    image_rgb: np.ndarray,
    labels: list[str],
    sam3_processor: Any,
    torch_module: Any,
    device: str,
    score_threshold: float = 0.0,
    limit: int | None = None,
) -> tuple[list[list[float]], list[float], list[str], list[np.ndarray]]:
    image = Image.fromarray(image_rgb, mode="RGB")
    image_shape = image_rgb.shape[:2]
    autocast = (
        torch_module.autocast(device_type="cuda", dtype=torch_module.bfloat16)
        if device.startswith("cuda")
        else nullcontext()
    )
    boxes: list[list[float]] = []
    scores: list[float] = []
    detected_labels: list[str] = []
    masks: list[np.ndarray] = []
    state: Any | None = None
    output: Any | None = None
    output_masks: Any | None = None
    try:
        with torch_module.no_grad(), autocast:
            state = sam3_processor.set_image(image)
            for label in labels:
                output = None
                output_masks = None
                if hasattr(sam3_processor, "reset_all_prompts"):
                    sam3_processor.reset_all_prompts(state)
                output = sam3_processor.set_text_prompt(prompt=label, state=state)
                output_boxes = _to_numpy_array(output.get("boxes", np.zeros((0, 4), dtype=np.float32)))
                output_boxes = (
                    np.asarray(output_boxes, dtype=np.float32).reshape(-1, 4)
                    if output_boxes.size
                    else np.zeros((0, 4), dtype=np.float32)
                )
                output_scores = _to_numpy_array(output.get("scores", np.zeros((len(output_boxes),), dtype=np.float32))).reshape(
                    -1
                )
                candidate_indices = [
                    index
                    for index in range(len(output_boxes))
                    if index < len(output_scores) and float(output_scores[index]) >= float(score_threshold)
                ]
                if limit is not None and len(candidate_indices) > limit:
                    candidate_indices = sorted(candidate_indices, key=lambda index: float(output_scores[index]), reverse=True)[:limit]
                output_masks = output.get("masks", np.zeros((0, *image_shape), dtype=np.float32))
                for index in candidate_indices:
                    box = output_boxes[index]
                    output_mask = _normalize_sam_mask_at(output_masks, index)
                    boxes.append([float(value) for value in box.tolist()])
                    scores.append(float(output_scores[index]))
                    detected_labels.append(label)
                    masks.append(_fit_mask_to_shape(output_mask, image_shape))
                output = None
                output_masks = None
    finally:
        output = None
        output_masks = None
        _clear_sam3_state(sam3_processor, state)
        _release_torch_memory(torch_module)
    return boxes, scores, detected_labels, masks


def _detect_text_with_sam3_for_image(
    image_rgb: np.ndarray,
    labels: list[str],
    sam3_processor: Any,
    torch_module: Any,
    device: str,
    config: AIDetectionConfig,
) -> tuple[list[list[float]], list[float], list[str], list[np.ndarray | _PlacedMask]]:
    image_shape = image_rgb.shape[:2]
    max_side = _sam3_max_side(config)
    candidate_limit = max(config.max_detections * 2, config.max_detections + 4)
    if max(image_shape) <= max_side:
        return _detect_text_with_sam3(
            image_rgb,
            labels,
            sam3_processor,
            torch_module,
            device,
            score_threshold=config.box_threshold,
            limit=candidate_limit,
        )

    boxes: list[list[float]] = []
    scores: list[float] = []
    detected_labels: list[str] = []
    masks: list[np.ndarray | _PlacedMask] = []
    for left, top, tile_width, tile_height in _sam3_tiles(
        image_shape,
        max_side,
        config.sam3_tile_overlap_ratio,
    ):
        tile = image_rgb[top : top + tile_height, left : left + tile_width]
        tile_boxes, tile_scores, tile_labels, tile_masks = _detect_text_with_sam3(
            tile,
            labels,
            sam3_processor,
            torch_module,
            device,
            score_threshold=config.box_threshold,
            limit=candidate_limit,
        )
        for index, raw_box in enumerate(tile_boxes):
            score = float(tile_scores[index]) if index < len(tile_scores) else 0.0
            if score < float(config.box_threshold):
                continue
            translated_box = _translate_xyxy_box(raw_box, left, top)
            global_bbox = _clamp_xyxy_box(translated_box, image_shape)
            if global_bbox is None or _bbox_area_ratio(global_bbox, image_shape) > config.max_box_area_ratio:
                continue
            if not _bbox_center_in_tile_core(
                global_bbox,
                (left, top, tile_width, tile_height),
                image_shape,
                config.sam3_tile_overlap_ratio,
            ):
                continue
            tile_mask = tile_masks[index] if index < len(tile_masks) else np.zeros(tile.shape[:2], dtype=np.float32)
            boxes.append(translated_box)
            scores.append(score)
            detected_labels.append(tile_labels[index] if index < len(tile_labels) else "watermark")
            masks.append(
                _PlacedMask(
                    _binary_sam_mask(_fit_mask_to_shape(tile_mask, tile.shape[:2]), config),
                    (left, top, tile_width, tile_height),
                )
            )
            boxes, scores, detected_labels, masks = _trim_sam3_candidates(
                boxes,
                scores,
                detected_labels,
                masks,
                candidate_limit,
            )
    return boxes, scores, detected_labels, masks


def _segment_boxes_with_sam3(
    image_rgb: np.ndarray,
    detections: Sequence[Detection],
    sam3_processor: Any,
    torch_module: Any,
    device: str,
    config: AIDetectionConfig | None = None,
) -> list[np.ndarray | _PlacedMask]:
    config = config or AIDetectionConfig()
    image_shape = image_rgb.shape[:2]
    autocast = (
        torch_module.autocast(device_type="cuda", dtype=torch_module.bfloat16)
        if device.startswith("cuda")
        else nullcontext()
    )
    masks: list[np.ndarray | _PlacedMask] = []
    with torch_module.no_grad(), autocast:
        for detection in detections:
            state: Any | None = None
            output: Any | None = None
            output_masks: Any | None = None
            target_bbox = detection.content_bbox or detection.bbox
            crop_bbox = _expanded_crop_bbox(target_bbox, image_shape, config)
            if crop_bbox is None:
                masks.append(_empty_placed_mask(target_bbox))
                continue
            left, top, crop_width, crop_height = crop_bbox
            crop = image_rgb[top : top + crop_height, left : left + crop_width]
            sam_crop, scale_x, scale_y = _resize_image_for_sam3(crop, config.sam3_max_side)
            local_bbox = _local_bbox_in_crop(target_bbox, crop_bbox)
            prompt_bbox = _scale_xywh_box(local_bbox, scale_x, scale_y, sam_crop.shape[:2])
            prompt_box = _bbox_to_normalized_cxcywh(prompt_bbox, sam_crop.shape[:2]) if prompt_bbox is not None else None
            if prompt_box is None:
                masks.append(_empty_placed_mask(target_bbox))
                continue
            try:
                state = sam3_processor.set_image(Image.fromarray(sam_crop, mode="RGB"))
                if hasattr(sam3_processor, "reset_all_prompts"):
                    sam3_processor.reset_all_prompts(state)
                output = sam3_processor.add_geometric_prompt(box=prompt_box, label=True, state=state)
                output_boxes = _to_numpy_array(output.get("boxes", np.zeros((0, 4), dtype=np.float32)))
                output_boxes = (
                    np.asarray(output_boxes, dtype=np.float32).reshape(-1, 4)
                    if output_boxes.size
                    else np.zeros((0, 4), dtype=np.float32)
                )
                output_scores = _to_numpy_array(output.get("scores", np.zeros((len(output_boxes),), dtype=np.float32))).reshape(
                    -1
                )
                output_masks = _normalize_sam_masks(output.get("masks", np.zeros((0, *sam_crop.shape[:2]), dtype=np.float32)))
                crop_mask = _best_mask_for_detection(output_boxes, output_scores, output_masks, prompt_bbox, sam_crop.shape[:2])
                crop_mask = _fit_mask_to_shape(crop_mask, crop.shape[:2])
                masks.append(_PlacedMask(_binary_sam_mask(crop_mask, config), crop_bbox))
                output = None
                output_masks = None
            finally:
                output = None
                output_masks = None
                _clear_sam3_state(sam3_processor, state)
                _release_torch_memory(torch_module)
    return masks


def _sam3_max_side(config: AIDetectionConfig) -> int:
    return max(8, int(config.sam3_max_side))


def _sam3_tiles(
    image_shape: tuple[int, int],
    max_side: int,
    overlap_ratio: float,
) -> list[tuple[int, int, int, int]]:
    height, width = image_shape
    tile_width = min(width, max_side)
    tile_height = min(height, max_side)
    overlap = min(max(float(overlap_ratio), 0.0), 0.80)
    step_x = max(1, tile_width - int(round(tile_width * overlap)))
    step_y = max(1, tile_height - int(round(tile_height * overlap)))
    return [
        (left, top, tile_width, tile_height)
        for top in _tile_starts(height, tile_height, step_y)
        for left in _tile_starts(width, tile_width, step_x)
    ]


def _bbox_center_in_tile_core(
    bbox: tuple[int, int, int, int],
    tile_bbox: tuple[int, int, int, int],
    image_shape: tuple[int, int],
    overlap_ratio: float,
) -> bool:
    x, y, width, height = bbox
    tile_x, tile_y, tile_width, tile_height = tile_bbox
    image_height, image_width = image_shape
    margin_x = int(round(tile_width * min(max(float(overlap_ratio), 0.0), 0.80) * 0.5))
    margin_y = int(round(tile_height * min(max(float(overlap_ratio), 0.0), 0.80) * 0.5))
    center_x = x + width / 2.0
    center_y = y + height / 2.0
    if margin_x > 0 and tile_x > 0 and center_x < tile_x + margin_x:
        return False
    if margin_x > 0 and tile_x + tile_width < image_width and center_x >= tile_x + tile_width - margin_x:
        return False
    if margin_y > 0 and tile_y > 0 and center_y < tile_y + margin_y:
        return False
    if margin_y > 0 and tile_y + tile_height < image_height and center_y >= tile_y + tile_height - margin_y:
        return False
    return True


def _trim_sam3_candidates(
    boxes: list[list[float]],
    scores: list[float],
    labels: list[str],
    masks: list[np.ndarray | _PlacedMask],
    limit: int,
) -> tuple[list[list[float]], list[float], list[str], list[np.ndarray | _PlacedMask]]:
    if len(scores) <= limit:
        return boxes, scores, labels, masks
    kept_indices = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)[:limit]
    return (
        [boxes[index] for index in kept_indices],
        [scores[index] for index in kept_indices],
        [labels[index] for index in kept_indices],
        [masks[index] for index in kept_indices],
    )


def _binary_sam_mask(mask: np.ndarray, config: AIDetectionConfig) -> np.ndarray:
    return (_fit_mask_to_shape(mask, mask.shape[:2]) > float(config.mask_threshold)).astype(np.uint8)


def _tile_starts(length: int, tile_size: int, step: int) -> list[int]:
    if length <= tile_size:
        return [0]
    final_start = length - tile_size
    starts = list(range(0, final_start + 1, step))
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def _translate_xyxy_box(raw_box: list[float], left: int, top: int) -> list[float]:
    return [
        float(raw_box[0] + left),
        float(raw_box[1] + top),
        float(raw_box[2] + left),
        float(raw_box[3] + top),
    ]


def _resize_image_for_sam3(image_rgb: np.ndarray, max_side: int) -> tuple[np.ndarray, float, float]:
    image = np.asarray(image_rgb, dtype=np.uint8)
    height, width = image.shape[:2]
    target_side = max(8, int(max_side))
    longest = max(height, width)
    if longest <= target_side:
        return image, 1.0, 1.0
    scale = float(target_side / longest)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    return resized.astype(np.uint8), float(resized_width / width), float(resized_height / height)


def _expanded_crop_bbox(
    bbox: tuple[int, int, int, int],
    image_shape: tuple[int, int],
    config: AIDetectionConfig,
) -> tuple[int, int, int, int] | None:
    x, y, width, height = bbox
    if width <= 0 or height <= 0:
        return None
    image_height, image_width = image_shape
    min_padding = max(0, int(config.sam3_crop_min_padding))
    pad_x = max(min_padding, int(round(width * max(0.0, float(config.sam3_crop_padding_ratio)))))
    pad_y = max(min_padding, int(round(height * max(0.0, float(config.sam3_crop_padding_ratio)))))
    left = max(0, x - pad_x)
    top = max(0, y - pad_y)
    right = min(image_width, x + width + pad_x)
    bottom = min(image_height, y + height + pad_y)
    if right <= left or bottom <= top:
        return None
    return left, top, right - left, bottom - top


def _local_bbox_in_crop(
    bbox: tuple[int, int, int, int],
    crop_bbox: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    x, y, width, height = bbox
    left, top, crop_width, crop_height = crop_bbox
    local_left = max(0, x - left)
    local_top = max(0, y - top)
    local_right = min(crop_width, x + width - left)
    local_bottom = min(crop_height, y + height - top)
    return local_left, local_top, max(0, local_right - local_left), max(0, local_bottom - local_top)


def _scale_xywh_box(
    bbox: tuple[int, int, int, int],
    scale_x: float,
    scale_y: float,
    image_shape: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    x, y, width, height = bbox
    image_height, image_width = image_shape
    left = max(0, min(image_width, int(round(x * scale_x))))
    top = max(0, min(image_height, int(round(y * scale_y))))
    right = max(0, min(image_width, int(round((x + width) * scale_x))))
    bottom = max(0, min(image_height, int(round((y + height) * scale_y))))
    if right - left < 1 or bottom - top < 1:
        return None
    return left, top, right - left, bottom - top


def _resolve_modelscope_model_file(model: str, model_file: str) -> str:
    model_value = (model or "").strip()
    if not model_value:
        raise ValueError("ModelScope model is empty.")
    filename = (model_file or DEFAULT_SAM3_MODELSCOPE_FILE).strip() or DEFAULT_SAM3_MODELSCOPE_FILE

    local_path = Path(model_value)
    if local_path.is_file():
        return str(local_path)
    if local_path.is_dir():
        candidate = local_path / filename
        if candidate.is_file():
            return str(candidate)

    try:
        from modelscope.hub.file_download import model_file_download
    except Exception as exc:  # pragma: no cover - depends on optional modelscope package
        raise AIDetectorUnavailableError(
            "SAM 3.1 ModelScope download needs the modelscope package. Install the AI extras first."
        ) from exc
    return str(model_file_download(model_id=model_value, file_path=filename))


def _to_numpy_array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if str(getattr(value, "dtype", "")) == "torch.bfloat16" and hasattr(value, "float"):
        value = value.float()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _detections_and_masks_from_boxes(
    boxes: list[list[float]],
    scores: list[float],
    labels: list[str],
    masks: list[np.ndarray | _PlacedMask],
    image_shape: tuple[int, int],
    config: AIDetectionConfig,
) -> tuple[list[Detection], list[np.ndarray | _PlacedMask]]:
    candidates: list[tuple[float, str, tuple[int, int, int, int], np.ndarray | _PlacedMask]] = []
    for index, raw_box in enumerate(boxes):
        bbox = _clamp_xyxy_box(raw_box, image_shape)
        if bbox is None:
            continue
        if _bbox_area_ratio(bbox, image_shape) > config.max_box_area_ratio:
            continue
        score = float(scores[index]) if index < len(scores) else 0.0
        if score < float(config.box_threshold):
            continue
        label = labels[index] if index < len(labels) and labels[index] else "watermark"
        mask = masks[index] if index < len(masks) else np.zeros(image_shape, dtype=np.float32)
        candidates.append((score, label, bbox, mask))

    candidates.sort(key=lambda item: item[0], reverse=True)
    kept: list[tuple[float, str, tuple[int, int, int, int], np.ndarray | _PlacedMask]] = []
    for candidate in candidates:
        if any(bbox_iou(candidate[2], existing[2]) >= config.nms_iou_threshold for existing in kept):
            continue
        kept.append(candidate)
        if len(kept) >= config.max_detections:
            break

    detections = [
        Detection(
            template_name=f"sam3:{label}",
            bbox=bbox,
            scale=1.0,
            score=score,
            color_score=0.0,
            strength=1.0,
            method="sam3",
            objective=0.0,
            clip_ratio=0.0,
            residual=0.0,
            watermark_correlation=0.0,
            content_bbox=bbox,
        )
        for score, label, bbox, _mask in kept
    ]
    return detections, [_normalize_mask_candidate(mask) for _score, _label, _bbox, mask in kept]


def _normalize_mask_candidate(mask: np.ndarray | _PlacedMask) -> np.ndarray | _PlacedMask:
    if isinstance(mask, _PlacedMask):
        return _PlacedMask(np.asarray(mask.mask, dtype=np.uint8), mask.bbox)
    return np.asarray(mask, dtype=np.float32)


def _empty_placed_mask(bbox: tuple[int, int, int, int]) -> _PlacedMask:
    return _PlacedMask(np.zeros((1, 1), dtype=np.uint8), bbox)


def _combine_masks(
    masks: list[np.ndarray | _PlacedMask],
    detections: list[Detection],
    image_shape: tuple[int, int],
    config: AIDetectionConfig,
) -> np.ndarray:
    height, width = image_shape
    combined = np.zeros((height, width), dtype=np.float32)
    for index, detection in enumerate(detections):
        if index < len(masks):
            mask_candidate = masks[index]
        else:
            if config.fallback_to_boxes:
                combined = np.maximum(combined, _box_mask(detection.bbox, image_shape))
            continue

        if isinstance(mask_candidate, _PlacedMask):
            merged = _merge_placed_mask(combined, mask_candidate, detection.bbox, image_shape, config)
            if not merged and config.fallback_to_boxes:
                combined = np.maximum(combined, _box_mask(detection.bbox, image_shape))
            continue

        mask = _fit_mask_to_shape(mask_candidate, image_shape)
        mask = (mask > float(config.mask_threshold)).astype(np.float32)
        if config.fallback_to_boxes and not np.any(mask):
            mask = _box_mask(detection.bbox, image_shape)
        combined = np.maximum(combined, _clip_mask_to_bbox(mask, detection.bbox, image_shape))

    if config.mask_dilate_pixels > 0 and np.any(combined):
        kernel_size = int(config.mask_dilate_pixels) * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        combined = cv2.dilate(combined.astype(np.uint8), kernel, iterations=1).astype(np.float32)
    return np.clip(combined, 0.0, 1.0).astype(np.float32)


def _normalize_sam_masks(masks: Any) -> list[np.ndarray]:
    mask_array = _to_numpy_array(masks)
    mask_array = np.asarray(mask_array)
    if mask_array.ndim == 2:
        return [mask_array.astype(np.float32)]
    if mask_array.ndim == 3:
        return [mask_array[index].astype(np.float32) for index in range(mask_array.shape[0])]
    if mask_array.ndim == 4:
        reduced = mask_array.max(axis=1)
        return [reduced[index].astype(np.float32) for index in range(reduced.shape[0])]
    raise ValueError(f"Unexpected SAM mask shape: {mask_array.shape}")


def _normalize_sam_mask_at(masks: Any, index: int) -> np.ndarray:
    ndim_value = getattr(masks, "ndim", None)
    if ndim_value is None and not isinstance(masks, np.ndarray) and hasattr(masks, "__getitem__"):
        mask = _to_numpy_array(masks[index])
        mask_array = np.asarray(mask)
        if mask_array.ndim == 3:
            mask_array = mask_array.max(axis=0)
        if mask_array.ndim != 2:
            raise ValueError(f"Unexpected SAM mask shape: {mask_array.shape}")
        return mask_array.astype(np.float32)

    ndim = int(ndim_value) if ndim_value is not None else int(np.asarray(masks).ndim)
    if ndim == 2:
        if index != 0:
            raise IndexError("SAM mask index out of range.")
        mask = _to_numpy_array(masks)
    elif ndim == 3:
        mask = _to_numpy_array(masks[index])
    elif ndim == 4:
        mask = _to_numpy_array(masks[index])
        if mask.ndim == 3:
            mask = mask.max(axis=0)
    else:
        mask = _to_numpy_array(masks)
    mask_array = np.asarray(mask)
    if mask_array.ndim != 2:
        raise ValueError(f"Unexpected SAM mask shape: {mask_array.shape}")
    return mask_array.astype(np.float32)


def _fit_mask_to_shape(mask: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    height, width = image_shape
    if mask.shape[:2] == (height, width):
        return mask.astype(np.float32)
    return cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST)


def _merge_placed_mask(
    combined: np.ndarray,
    placed_mask: _PlacedMask,
    bbox: tuple[int, int, int, int],
    image_shape: tuple[int, int],
    config: AIDetectionConfig,
) -> bool:
    placed_bbox = _clamp_xywh_box(placed_mask.bbox, image_shape)
    if placed_bbox is None:
        return False
    mask = _fit_mask_to_shape(placed_mask.mask, (placed_bbox[3], placed_bbox[2]))
    mask = (mask > float(config.mask_threshold)).astype(np.float32)
    if not np.any(mask):
        return False

    place_x, place_y, place_width, place_height = placed_bbox
    clip_x, clip_y, clip_width, clip_height = _expand_bbox(bbox, image_shape, ratio=0.04)
    left = max(place_x, clip_x)
    top = max(place_y, clip_y)
    right = min(place_x + place_width, clip_x + clip_width)
    bottom = min(place_y + place_height, clip_y + clip_height)
    if right <= left or bottom <= top:
        return False

    src_x = left - place_x
    src_y = top - place_y
    source = mask[src_y : src_y + bottom - top, src_x : src_x + right - left]
    if not np.any(source):
        return False
    current = combined[top:bottom, left:right]
    combined[top:bottom, left:right] = np.maximum(current, source)
    return True


def _clip_mask_to_bbox(mask: np.ndarray, bbox: tuple[int, int, int, int], image_shape: tuple[int, int]) -> np.ndarray:
    clipped = np.zeros(image_shape, dtype=np.float32)
    x, y, width, height = _expand_bbox(bbox, image_shape, ratio=0.04)
    clipped[y : y + height, x : x + width] = mask[y : y + height, x : x + width]
    return clipped


def _clamp_xywh_box(
    bbox: tuple[int, int, int, int],
    image_shape: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    x, y, width, height = bbox
    image_height, image_width = image_shape
    left = max(0, min(image_width, int(x)))
    top = max(0, min(image_height, int(y)))
    right = max(0, min(image_width, int(x + width)))
    bottom = max(0, min(image_height, int(y + height)))
    if right <= left or bottom <= top:
        return None
    return left, top, right - left, bottom - top


def _box_mask(bbox: tuple[int, int, int, int], image_shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(image_shape, dtype=np.float32)
    x, y, width, height = bbox
    mask[y : y + height, x : x + width] = 1.0
    return mask


def _expand_bbox(
    bbox: tuple[int, int, int, int],
    image_shape: tuple[int, int],
    *,
    ratio: float,
) -> tuple[int, int, int, int]:
    x, y, width, height = bbox
    image_height, image_width = image_shape
    pad_x = int(round(width * ratio))
    pad_y = int(round(height * ratio))
    left = max(0, x - pad_x)
    top = max(0, y - pad_y)
    right = min(image_width, x + width + pad_x)
    bottom = min(image_height, y + height + pad_y)
    return left, top, max(1, right - left), max(1, bottom - top)


def _bbox_to_normalized_cxcywh(bbox: tuple[int, int, int, int], image_shape: tuple[int, int]) -> list[float] | None:
    x, y, width, height = bbox
    image_height, image_width = image_shape
    if width <= 0 or height <= 0 or image_width <= 0 or image_height <= 0:
        return None
    center_x = (x + width / 2.0) / image_width
    center_y = (y + height / 2.0) / image_height
    return [
        float(max(0.0, min(1.0, center_x))),
        float(max(0.0, min(1.0, center_y))),
        float(max(0.0, min(1.0, width / image_width))),
        float(max(0.0, min(1.0, height / image_height))),
    ]


def _best_mask_for_detection(
    boxes: np.ndarray,
    scores: np.ndarray,
    masks: list[np.ndarray],
    target_bbox: tuple[int, int, int, int],
    image_shape: tuple[int, int],
) -> np.ndarray:
    if not masks:
        return np.zeros(image_shape, dtype=np.float32)
    best_index = 0
    best_key = (-1.0, -1.0)
    for index, mask in enumerate(masks):
        bbox = _clamp_xyxy_box(boxes[index].tolist(), image_shape) if index < len(boxes) else None
        overlap = bbox_iou(target_bbox, bbox) if bbox is not None else 0.0
        score = float(scores[index]) if index < len(scores) else 0.0
        key = (overlap, score)
        if key > best_key:
            best_key = key
            best_index = index
    return _fit_mask_to_shape(masks[best_index], image_shape)


def _clamp_xyxy_box(raw_box: list[float], image_shape: tuple[int, int]) -> tuple[int, int, int, int] | None:
    if len(raw_box) != 4:
        return None
    height, width = image_shape
    left = max(0, min(width, int(round(raw_box[0]))))
    top = max(0, min(height, int(round(raw_box[1]))))
    right = max(0, min(width, int(round(raw_box[2]))))
    bottom = max(0, min(height, int(round(raw_box[3]))))
    if right - left < 2 or bottom - top < 2:
        return None
    return left, top, right - left, bottom - top


def _bbox_area_ratio(bbox: tuple[int, int, int, int], image_shape: tuple[int, int]) -> float:
    _, _, width, height = bbox
    image_height, image_width = image_shape
    return float((width * height) / max(image_width * image_height, 1))


def _stringify_label(label: Any) -> str:
    if isinstance(label, str):
        return label
    if isinstance(label, (list, tuple)):
        return ", ".join(_stringify_label(item) for item in label)
    return str(label)
