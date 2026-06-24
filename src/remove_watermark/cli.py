from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .ai_detection import (
    DEFAULT_AI_PROMPT,
    DEFAULT_SAM3_CONFIDENCE_THRESHOLD,
    DEFAULT_SAM3_MODELSCOPE_FILE,
    DEFAULT_SAM3_MODELSCOPE_MODEL,
    AIDetectionConfig,
    Sam3TextWatermarkDetector,
    download_ai_models,
    download_sam3_model,
)
from .debug_render import normalize_debug_map, render_debug_map, render_debug_overlay
from .detection import WatermarkRemover
from .masking import restore_with_mask
from .models import Detection, WatermarkRemoverConfig
from .templates import (
    list_input_images,
    load_image_rgb,
    load_templates,
    save_image_like,
    save_image_rgb,
)

DEBUG_OVERLAY_SPECS = (
    ("lama_mask", (33, 150, 243)),
    ("cleanup_change", (255, 193, 7)),
    ("residual_evidence", (0, 188, 212)),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect and remove a fixed-style semi-transparent watermark from images.")
    parser.add_argument("--input", default="input", help="Input image file or directory. Defaults to input/.")
    parser.add_argument("--template", action="append", help="Template image file or directory. Repeat to use multiple templates. Defaults to templates/.")
    parser.add_argument("--output", default="output", help="Output directory for processed images. Defaults to output/.")
    parser.add_argument("--min-scale", type=float, default=0.35, help="Minimum template scale.")
    parser.add_argument("--max-scale", type=float, default=1.20, help="Maximum template scale.")
    parser.add_argument("--scale-step", type=float, default=0.05, help="Template scale step.")
    parser.add_argument("--score-threshold", type=float, default=0.50, help="Primary detection score threshold.")
    parser.add_argument("--jobs", type=int, default=0, help="Number of CPU detection jobs. Defaults to CPU count; CUDA LaMa runs one image at a time.")
    parser.add_argument("--lama-device", default="auto", help="Torch device for LaMa, for example auto, cuda, or cpu. Defaults to auto.")
    parser.add_argument("--detector", choices=("template", "sam3"), default="template", help="Detection backend. Defaults to template.")
    parser.add_argument("--ai-prompt", default=DEFAULT_AI_PROMPT, help="SAM 3.1 text prompts, separated by periods.")
    parser.add_argument("--sam3-model", default=DEFAULT_SAM3_MODELSCOPE_MODEL, help="ModelScope SAM 3.1 model id or local checkpoint directory/path.")
    parser.add_argument("--sam3-model-file", default=DEFAULT_SAM3_MODELSCOPE_FILE, help="SAM 3.1 checkpoint filename inside the ModelScope repo.")
    parser.add_argument(
        "--sam3-confidence-threshold",
        type=float,
        default=DEFAULT_SAM3_CONFIDENCE_THRESHOLD,
        help="SAM 3.1 mask confidence threshold.",
    )
    parser.add_argument("--ai-device", default="auto", help="Torch device for SAM 3.1 detection, for example auto or cuda. Defaults to auto.")
    parser.add_argument("--ai-box-threshold", type=float, default=0.20, help="SAM 3.1 detection confidence threshold.")
    parser.add_argument("--ai-max-box-area-ratio", type=float, default=0.35, help="Largest SAM 3.1 detection box area as a fraction of the image.")
    parser.add_argument("--ai-max-detections", type=int, default=60, help="Maximum SAM 3.1 detections per image.")
    parser.add_argument("--download-ai-models", action="store_true", help="Download and cache the SAM 3.1 detector model, then exit.")
    parser.add_argument("--download-sam3-model", action="store_true", help="Download and cache only the ModelScope SAM 3.1 model, then exit.")
    parser.add_argument("--save-debug", action="store_true", help="Save detection JSON and overlay previews.")
    parser.add_argument("--no-mask-body", action="store_true", help="Disable mask body unification before LaMa inpainting.")
    parser.add_argument("--mask-body-gap-ratio", type=float, default=0.05, help="Relative gap used to join nearby mask components into one body.")
    parser.add_argument(
        "--mask-dilate-max-body-ratio",
        type=float,
        default=0.10,
        help="Maximum mask dilation as a fraction of the mask body's shortest side. Use 0 to disable the relative cap.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_path = Path(args.input)
    template_paths = [Path(value) for value in (args.template or ["templates"])]
    output_dir = Path(args.output)

    if args.min_scale <= 0 or args.max_scale <= 0 or args.scale_step <= 0 or args.min_scale > args.max_scale:
        print("Invalid scale configuration.", file=sys.stderr)
        return 1
    if args.jobs < 0:
        print("Invalid jobs configuration.", file=sys.stderr)
        return 1
    if args.mask_body_gap_ratio < 0:
        print("Invalid mask body gap ratio.", file=sys.stderr)
        return 1
    if args.mask_dilate_max_body_ratio < 0:
        print("Invalid mask dilation body ratio.", file=sys.stderr)
        return 1
    if (
        not 0.0 < args.ai_box_threshold <= 1.0
        or not 0.0 < args.ai_max_box_area_ratio <= 1.0
        or args.ai_max_detections < 1
        or not 0.0 <= args.sam3_confidence_threshold <= 1.0
    ):
        print("Invalid AI detector configuration.", file=sys.stderr)
        return 1

    ai_config = build_ai_config(args)
    if args.download_sam3_model:
        try:
            path = download_sam3_model(ai_config)
            print(f"Downloaded SAM 3.1 model: {path}", flush=True)
            return 0
        except Exception as exc:
            print(f"Failed to download SAM 3.1 model: {exc}", file=sys.stderr)
            return 1

    if args.download_ai_models:
        try:
            download_ai_models(ai_config)
            print(
                f"Downloaded AI models: {model_download_label(ai_config)}",
                flush=True,
            )
            return 0
        except Exception as exc:
            print(f"Failed to download AI models: {exc}", file=sys.stderr)
            return 1

    if args.detector == "template":
        missing_templates = [path for path in template_paths if not path.exists()]
        if missing_templates:
            print(f"Template path not found: {missing_templates[0]}", file=sys.stderr)
            return 1
    if not input_path.exists():
        print(f"Input path not found: {input_path}", file=sys.stderr)
        return 1

    image_paths = list_input_images(input_path)
    if not image_paths:
        print(f"No supported images found under: {input_path}", file=sys.stderr)
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir = output_dir / "debug"
    if args.save_debug:
        debug_dir.mkdir(parents=True, exist_ok=True)

    config = WatermarkRemoverConfig(
        min_scale=args.min_scale,
        max_scale=args.max_scale,
        scale_step=args.scale_step,
        score_threshold=args.score_threshold,
        mask_unify_body=not args.no_mask_body,
        mask_body_gap_ratio=args.mask_body_gap_ratio,
        mask_dilate_max_body_ratio=args.mask_dilate_max_body_ratio,
        lama_device=args.lama_device,
        collect_debug_maps=args.save_debug,
    )
    if args.detector == "sam3":
        detector = build_ai_detector(ai_config)
        failures = process_ai_images_sequential(
            image_paths,
            output_dir=output_dir,
            detector=detector,
            lama_config=config,
            save_debug=args.save_debug,
            debug_dir=debug_dir,
        )
        return 0 if failures == 0 else 1

    templates = load_templates(template_paths)
    if not templates:
        print("No supported template images found.", file=sys.stderr)
        return 1
    remover = WatermarkRemover(templates, config=config)
    jobs = normalize_jobs(args.jobs, len(image_paths))
    if jobs > 1:
        warm_template_cache(templates, config)

    if jobs > 1:
        failures = process_images_staged(
            image_paths,
            output_dir=output_dir,
            remover=remover,
            save_debug=args.save_debug,
            debug_dir=debug_dir,
            detect_jobs=jobs,
        )
    else:
        failures = process_images_sequential(
            image_paths,
            output_dir=output_dir,
            remover=remover,
            save_debug=args.save_debug,
            debug_dir=debug_dir,
        )

    return 0 if failures == 0 else 1


@dataclass(slots=True)
class ProcessingResult:
    image_path: Path
    detection_count: int = 0
    error: str | None = None


@dataclass(slots=True)
class DetectionResult:
    image_path: Path
    image_rgb: np.ndarray | None = None
    detections: list[Detection] | None = None
    error: str | None = None


def build_ai_config(args: argparse.Namespace) -> AIDetectionConfig:
    return AIDetectionConfig(
        prompt=args.ai_prompt,
        sam3_model=args.sam3_model,
        sam3_model_file=args.sam3_model_file,
        sam3_confidence_threshold=args.sam3_confidence_threshold,
        device=args.ai_device,
        box_threshold=args.ai_box_threshold,
        max_box_area_ratio=args.ai_max_box_area_ratio,
        max_detections=args.ai_max_detections,
    )


def build_ai_detector(config: AIDetectionConfig):
    return Sam3TextWatermarkDetector(config)


def model_download_label(config: AIDetectionConfig) -> str:
    return f"{config.sam3_model}/{config.sam3_model_file}"


def report_processing_result(result: ProcessingResult) -> bool:
    if result.error is None:
        print(
            f"Processed {result.image_path.name}: {result.detection_count} detections",
            flush=True,
        )
        return True
    print(result.error, file=sys.stderr, flush=True)
    return False


def normalize_jobs(requested_jobs: int, image_count: int) -> int:
    if image_count <= 1:
        return 1
    if requested_jobs == 0:
        return max(1, min(image_count, os.cpu_count() or 1))
    return max(1, min(image_count, requested_jobs))


def warm_template_cache(templates, config: WatermarkRemoverConfig) -> None:
    for template in templates:
        for scale in config.scales():
            template.resized_for_scale(scale)


def process_images_sequential(
    image_paths: list[Path],
    *,
    output_dir: Path,
    remover: WatermarkRemover,
    save_debug: bool,
    debug_dir: Path,
) -> int:
    failures = 0
    for image_path in image_paths:
        result = process_image(
            image_path,
            output_dir=output_dir,
            remover=remover,
            save_debug=save_debug,
            debug_dir=debug_dir,
        )
        if not report_processing_result(result):
            failures += 1
    return failures


def process_images_staged(
    image_paths: list[Path],
    *,
    output_dir: Path,
    remover: WatermarkRemover,
    save_debug: bool,
    debug_dir: Path,
    detect_jobs: int,
) -> int:
    failures = 0
    iterator = iter(image_paths)
    pending = {}
    with ThreadPoolExecutor(max_workers=detect_jobs) as executor:
        for _ in range(detect_jobs):
            if not submit_next_detection(pending, executor, iterator, remover):
                break
        while pending:
            for future in as_completed(pending):
                image_path = pending.pop(future)
                try:
                    detected = future.result()
                except Exception as exc:  # pragma: no cover - detect_image normally wraps errors
                    detected = DetectionResult(
                        image_path=image_path,
                        error=f"Failed to process {image_path}: {exc}",
                    )
                result = restore_detected_image(
                    detected,
                    output_dir=output_dir,
                    remover=remover,
                    save_debug=save_debug,
                    debug_dir=debug_dir,
                )
                if not report_processing_result(result):
                    failures += 1
                submit_next_detection(pending, executor, iterator, remover)
                break
    return failures


def process_ai_images_sequential(
    image_paths: list[Path],
    *,
    output_dir: Path,
    detector: Sam3TextWatermarkDetector,
    lama_config: WatermarkRemoverConfig,
    save_debug: bool,
    debug_dir: Path,
) -> int:
    failures = 0
    for image_path in image_paths:
        result = process_ai_image(
            image_path,
            output_dir=output_dir,
            detector=detector,
            lama_config=lama_config,
            save_debug=save_debug,
            debug_dir=debug_dir,
        )
        if not report_processing_result(result):
            failures += 1
    return failures


def process_ai_image(
    image_path: Path,
    *,
    output_dir: Path,
    detector: Sam3TextWatermarkDetector,
    lama_config: WatermarkRemoverConfig,
    save_debug: bool,
    debug_dir: Path,
) -> ProcessingResult:
    output_path = output_dir / image_path.name
    if output_path.resolve() == image_path.resolve():
        return ProcessingResult(image_path=image_path, error=f"Refusing to overwrite input image: {image_path}")
    try:
        image_rgb = load_image_rgb(image_path)
        detected = detector.detect(image_rgb)
        restored = restore_with_mask(image_rgb, detected.mask, lama_config) if np.any(detected.mask > 0.0) else image_rgb.copy()
        save_image_like(output_path, restored, image_path)
        if save_debug:
            save_ai_debug_artifacts(debug_dir, image_path.stem, image_rgb, restored, detected.detections, detected.mask)
        return ProcessingResult(image_path=image_path, detection_count=len(detected.detections))
    except Exception as exc:  # pragma: no cover - surfaced through CLI tests
        return ProcessingResult(image_path=image_path, error=f"Failed to process {image_path}: {exc}")


def submit_next_detection(
    pending,
    executor: ThreadPoolExecutor,
    iterator,
    remover: WatermarkRemover,
) -> bool:
    try:
        image_path = next(iterator)
    except StopIteration:
        return False
    pending[executor.submit(detect_image, image_path, remover=remover)] = image_path
    return True


def detect_image(image_path: Path, *, remover: WatermarkRemover) -> DetectionResult:
    try:
        image_rgb = load_image_rgb(image_path)
        detections = remover.detect(image_rgb)
        return DetectionResult(image_path=image_path, image_rgb=image_rgb, detections=detections)
    except Exception as exc:  # pragma: no cover - surfaced through CLI tests
        return DetectionResult(image_path=image_path, error=f"Failed to process {image_path}: {exc}")


def restore_detected_image(
    detected: DetectionResult,
    *,
    output_dir: Path,
    remover: WatermarkRemover,
    save_debug: bool,
    debug_dir: Path,
) -> ProcessingResult:
    image_path = detected.image_path
    output_path = output_dir / image_path.name
    if output_path.resolve() == image_path.resolve():
        return ProcessingResult(image_path=image_path, error=f"Refusing to overwrite input image: {image_path}")
    if detected.error is not None:
        return ProcessingResult(image_path=image_path, error=detected.error)
    if detected.image_rgb is None or detected.detections is None:
        return ProcessingResult(image_path=image_path, error=f"Failed to process {image_path}: missing detection result")
    try:
        restored, detections = remover.restore_detections(detected.image_rgb, detected.detections)
        save_image_like(output_path, restored, image_path)
        if save_debug:
            save_debug_artifacts(debug_dir, image_path.stem, detected.image_rgb, restored, detections)
        return ProcessingResult(image_path=image_path, detection_count=len(detections))
    except Exception as exc:  # pragma: no cover - surfaced through CLI tests
        return ProcessingResult(image_path=image_path, error=f"Failed to process {image_path}: {exc}")


def process_image(
    image_path: Path,
    *,
    output_dir: Path,
    remover: WatermarkRemover,
    save_debug: bool,
    debug_dir: Path,
) -> ProcessingResult:
    detected = detect_image(image_path, remover=remover)
    return restore_detected_image(
        detected,
        output_dir=output_dir,
        remover=remover,
        save_debug=save_debug,
        debug_dir=debug_dir,
    )


def save_debug_artifacts(debug_dir: Path, stem: str, image_rgb, restored_rgb, detections: list[Detection]) -> None:
    image_debug_dir = debug_dir / stem
    previews_dir = image_debug_dir / "previews"
    overlays_dir = image_debug_dir / "overlays"
    maps_dir = image_debug_dir / "maps"
    patches_dir = image_debug_dir / "patches"
    for directory in (previews_dir, overlays_dir, maps_dir, patches_dir):
        directory.mkdir(parents=True, exist_ok=True)

    detection_boxes_filename = f"{stem}/previews/{stem}_detection_boxes.png"
    save_image_rgb(debug_dir / detection_boxes_filename, render_debug_overlay(image_rgb, detections))
    input_output_filename = f"{stem}/previews/{stem}_input_output_compare.png"
    save_image_rgb(debug_dir / input_output_filename, render_side_by_side(image_rgb, restored_rgb))

    debug_overlays = {}
    for map_name, color in DEBUG_OVERLAY_SPECS:
        if not any(map_name in detection.debug_maps for detection in detections):
            continue
        full_map = build_full_debug_map(image_rgb.shape[:2], detections, map_name)
        if map_name == "lama_mask" and float(full_map.max()) <= 0.0:
            continue
        filename = f"{stem}/overlays/{stem}_{map_name}_overlay.png"
        save_image_rgb(debug_dir / filename, render_mask_overlay(image_rgb, full_map, color))
        debug_overlays[map_name] = filename

    detection_payloads = []
    for index, detection in enumerate(detections, start=1):
        debug_map_files = {}
        for name, map_array in detection.debug_maps.items():
            filename = f"{stem}/maps/{stem}_{index:03d}_{name}.png"
            save_image_rgb(debug_dir / filename, render_debug_map(map_array))
            debug_map_files[name] = filename
        patch_compare_filename = f"{stem}/patches/{stem}_{index:03d}_patch_compare.png"
        save_image_rgb(debug_dir / patch_compare_filename, render_patch_compare(image_rgb, restored_rgb, detection.bbox))
        detection_payloads.append(_detection_payload(detection, patch_compare_filename, debug_map_files))
    payload = {
        "detections": detection_payloads,
        "detection_boxes": detection_boxes_filename,
        "debug_overlays": debug_overlays,
        "input_output_compare": input_output_filename,
    }
    (debug_dir / f"{stem}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_ai_debug_artifacts(
    debug_dir: Path,
    stem: str,
    image_rgb: np.ndarray,
    restored_rgb: np.ndarray,
    detections: list[Detection],
    mask: np.ndarray,
) -> None:
    image_debug_dir = debug_dir / stem
    previews_dir = image_debug_dir / "previews"
    overlays_dir = image_debug_dir / "overlays"
    maps_dir = image_debug_dir / "maps"
    patches_dir = image_debug_dir / "patches"
    for directory in (previews_dir, overlays_dir, maps_dir, patches_dir):
        directory.mkdir(parents=True, exist_ok=True)

    detection_boxes_filename = f"{stem}/previews/{stem}_ai_detection_boxes.png"
    save_image_rgb(debug_dir / detection_boxes_filename, render_debug_overlay(image_rgb, detections))
    input_output_filename = f"{stem}/previews/{stem}_input_output_compare.png"
    save_image_rgb(debug_dir / input_output_filename, render_side_by_side(image_rgb, restored_rgb))
    mask_filename = f"{stem}/maps/{stem}_ai_mask.png"
    save_image_rgb(debug_dir / mask_filename, render_debug_map(mask))
    mask_overlay_filename = f"{stem}/overlays/{stem}_ai_mask_overlay.png"
    save_image_rgb(debug_dir / mask_overlay_filename, render_mask_overlay(image_rgb, mask, (94, 196, 167)))

    detection_payloads = []
    for index, detection in enumerate(detections, start=1):
        patch_compare_filename = f"{stem}/patches/{stem}_{index:03d}_patch_compare.png"
        save_image_rgb(debug_dir / patch_compare_filename, render_patch_compare(image_rgb, restored_rgb, detection.bbox))
        detection_payloads.append(_detection_payload(detection, patch_compare_filename, {}))
    payload = {
        "detections": detection_payloads,
        "detection_boxes": detection_boxes_filename,
        "debug_overlays": {"ai_mask": mask_overlay_filename},
        "debug_maps": {"ai_mask": mask_filename},
        "input_output_compare": input_output_filename,
    }
    (debug_dir / f"{stem}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _detection_payload(
    detection: Detection,
    patch_compare_filename: str,
    debug_map_files: dict[str, str],
) -> dict[str, object]:
    return {
        "bbox": list(detection.bbox),
        "content_bbox": list(detection.content_bbox or detection.bbox),
        "template": detection.template_name,
        "scale": detection.scale,
        "score": detection.score,
        "color_score": detection.color_score,
        "strength": detection.strength,
        "method": detection.method,
        "objective": detection.objective,
        "clip_ratio": detection.clip_ratio,
        "residual": detection.residual,
        "watermark_correlation": detection.watermark_correlation,
        "residual_score": detection.residual_score,
        "text_detail": detection.text_detail,
        "lama_mask_ratio": detection.lama_mask_ratio,
        "stage_metrics": detection.stage_metrics,
        "patch_compare": patch_compare_filename,
        "debug_maps": debug_map_files,
    }


def render_side_by_side(left_rgb: np.ndarray, right_rgb: np.ndarray) -> np.ndarray:
    left = np.clip(left_rgb, 0, 255).astype(np.uint8)
    right = np.clip(right_rgb, 0, 255).astype(np.uint8)
    separator = np.full((left.shape[0], 8, 3), 245, dtype=np.uint8)
    return np.concatenate([left, separator, right], axis=1)


def render_patch_compare(image_rgb: np.ndarray, restored_rgb: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x, y, width, height = bbox
    original_patch = image_rgb[y : y + height, x : x + width]
    restored_patch = restored_rgb[y : y + height, x : x + width]
    change = np.mean(np.abs(restored_patch.astype(np.float32) - original_patch.astype(np.float32)), axis=2)
    change = np.clip(change * 8.0, 0.0, 255.0).astype(np.uint8)
    change_rgb = np.zeros((*change.shape, 3), dtype=np.uint8)
    change_rgb[..., 0] = change
    change_rgb[..., 1] = np.clip(change * 0.45, 0, 255).astype(np.uint8)
    separator = np.full((height, 4, 3), 245, dtype=np.uint8)
    return np.concatenate(
        [
            original_patch.astype(np.uint8),
            separator,
            restored_patch.astype(np.uint8),
            separator,
            change_rgb,
        ],
        axis=1,
    )


def build_full_debug_map(shape: tuple[int, int], detections: list[Detection], map_name: str) -> np.ndarray:
    height, width = shape
    full_map = np.zeros((height, width), dtype=np.float32)
    for detection in detections:
        map_array = detection.debug_maps.get(map_name)
        if map_array is None:
            continue
        x, y, box_width, box_height = detection.bbox
        normalized = normalize_debug_map(map_array, (box_height, box_width), smooth=True)
        y2 = min(height, y + box_height)
        x2 = min(width, x + box_width)
        if y2 <= y or x2 <= x:
            continue
        patch_map = normalized[: y2 - y, : x2 - x]
        full_map[y:y2, x:x2] = np.maximum(full_map[y:y2, x:x2], patch_map)
    return full_map


def render_mask_overlay(image_rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    base = np.clip(image_rgb, 0, 255).astype(np.float32)
    mask_float = np.clip(mask.astype(np.float32), 0.0, 1.0)[..., None]
    tint = np.zeros_like(base, dtype=np.float32)
    tint[..., 0] = color[0]
    tint[..., 1] = color[1]
    tint[..., 2] = color[2]
    alpha = mask_float * 0.55
    return np.clip(base * (1.0 - alpha) + tint * alpha, 0.0, 255.0).astype(np.uint8)


def run() -> None:
    raise SystemExit(main())
