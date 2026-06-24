from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, wait
from pathlib import Path
from typing import Any

import numpy as np

from .ai_detection import DEFAULT_AI_PROMPT, AIDetectionConfig
from .models import WatermarkRemoverConfig
from .templates import load_image_rgb, load_templates
from .web_images import _decode_mask_data_url
from .web_models import BatchDetectionResult, BatchProcessResult, OperationCancelled
from .web_settings import (
    DEFAULT_WEB_TEMPLATE_MAX_SCALE,
    DEFAULT_WEB_TEMPLATE_MIN_SCALE,
    DEFAULT_WEB_TEMPLATE_SCORE_THRESHOLD,
    _ai_bool_setting,
    _ai_float_setting,
    _ai_float_setting_closed,
    _ai_int_setting,
    _ai_settings_payload,
    _bool_setting,
    _normalize_batch_jobs,
    _normalize_detector_mode,
    _path_key,
    _raise_if_cancelled,
    _template_bool_setting,
    _template_float_setting,
    _template_float_setting_closed,
    _template_int_setting,
    _template_settings_payload,
)


class EditorProcessingMixin:
    def detect(
        self,
        index: int,
        selected_templates: list[str] | None = None,
        *,
        detector: str = "template",
        ai_prompt: str | None = None,
        ai_settings: dict[str, Any] | None = None,
        template_settings: dict[str, Any] | None = None,
        operation_id: str | None = None,
    ) -> dict[str, Any]:
        detector = _normalize_detector_mode(detector)
        self._raise_if_operation_cancelled(operation_id)
        if detector == "sam3":
            return self._detect_with_ai(index, ai_prompt, ai_settings, detector=detector, operation_id=operation_id)

        template_paths = self._selected_template_paths(selected_templates)
        if not template_paths:
            raise ValueError("No templates selected.")
        templates = load_templates(template_paths)
        if not templates:
            raise ValueError("No supported template images found.")

        config = self._template_detection_config(template_settings)
        remover = self._new_template_remover(templates, config)
        return self._detect_with_loaded_templates(
            index,
            templates,
            template_paths,
            config,
            remover,
            operation_id=operation_id,
        )

    def detect_all(
        self,
        selected_templates: list[str] | None = None,
        *,
        detector: str = "template",
        ai_prompt: str | None = None,
        ai_settings: dict[str, Any] | None = None,
        template_settings: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._detect_all(
            selected_templates,
            detector=detector,
            ai_prompt=ai_prompt,
            ai_settings=ai_settings,
            template_settings=template_settings,
        )

    def _execute_operation_batch_detect(self, job_id: str, item: dict[str, Any]) -> int:
        image_path = Path(str(item["imagePath"]))

        def cancelled() -> bool:
            return self._batch_cancel_requested(job_id)

        try:
            detector = _normalize_detector_mode(str(item.get("detector", "template")))
            _raise_if_cancelled(cancelled)
            if detector == "sam3":
                ai_config = self._ai_detection_config(item.get("aiPrompt"), item.get("aiSettings"))
                _raise_if_cancelled(cancelled)
                result = self._detect_for_batch_ai_path(image_path, ai_config)
                _raise_if_cancelled(cancelled)
                if result.error is not None:
                    raise ValueError(result.error)
                self._commit_batch_detection_for_result(
                    result,
                    [],
                    detector=detector,
                    ai_prompt=ai_config.prompt,
                    ai_settings=_ai_settings_payload(ai_config),
                )
                return len(result.detections or [])

            template_paths, templates, config, remover, template_settings_payload = self._operation_batch_template_detector(
                job_id,
                item,
            )
            _raise_if_cancelled(cancelled)
            result = self._detect_for_batch_path(image_path, templates, config, remover)
            _raise_if_cancelled(cancelled)
            if result.error is not None:
                raise ValueError(result.error)
            self._commit_batch_detection_for_result(result, template_paths, template_settings=template_settings_payload)
            return len(result.detections or [])
        except OperationCancelled:
            raise
        except Exception as exc:
            self._mark_detection_error_for_path(image_path, str(exc))
            raise

    def _operation_batch_template_detector(
        self,
        job_id: str,
        item: dict[str, Any],
    ) -> tuple[list[Path], list[Any], WatermarkRemoverConfig, Any, dict[str, Any]]:
        template_paths = self._selected_template_paths(item.get("templates"))
        if not template_paths:
            raise ValueError("No templates selected.")
        config = self._template_detection_config(item.get("templateSettings"))
        cache_key = self._operation_batch_template_cache_key(template_paths, config)

        with self._batch_lock:
            job = self._batch_jobs.get(job_id)
        if job is None:
            raise ValueError("Batch job not found.")

        with job.operation_cache_lock:
            cached = job.operation_cache.get(cache_key)
            if cached is None:
                templates = load_templates(template_paths)
                if not templates:
                    raise ValueError("No supported template images found.")
                cached = (
                    list(template_paths),
                    templates,
                    config,
                    self._new_template_remover(templates, config),
                    _template_settings_payload(config),
                )
                job.operation_cache[cache_key] = cached
            return cached

    def _operation_batch_template_cache_key(
        self,
        template_paths: list[Path],
        config: WatermarkRemoverConfig,
    ) -> tuple[Any, ...]:
        return (
            tuple(self._operation_batch_template_path_key(path) for path in template_paths),
            tuple(sorted(_template_settings_payload(config).items())),
        )

    def _operation_batch_template_path_key(self, path: Path) -> tuple[str, int | None, int | None]:
        key = _path_key(path)
        try:
            stat = path.stat()
        except OSError:
            return (key, None, None)
        return (key, stat.st_size, stat.st_mtime_ns)

    def _execute_operation_batch_process(self, job_id: str, item: dict[str, Any]) -> None:
        image_path = Path(str(item["imagePath"]))

        def cancelled() -> bool:
            return self._batch_cancel_requested(job_id)

        try:
            image_rgb = load_image_rgb(image_path)
            mask_data = item.get("maskData")
            if mask_data:
                mask = _decode_mask_data_url(str(mask_data), image_rgb.shape[:2])
                _raise_if_cancelled(cancelled)
                with self._batch_lock:
                    self._save_mask(self._image_index_for_path(image_path), mask)
            else:
                with self._batch_lock:
                    index = self._image_index_for_path(image_path)
                    if not self._mask_path(index).is_file():
                        raise ValueError("No saved mask for this image.")
                    mask = self._load_mask(index, image_rgb.shape[:2])
            if not np.any(mask > 0):
                raise ValueError("Saved mask is empty.")
            _raise_if_cancelled(cancelled)
            config = WatermarkRemoverConfig(lama_device=self.config.lama_device)
            restored = self._restore_with_mask(image_rgb, mask, config)
            _raise_if_cancelled(cancelled)
            self._commit_batch_process_for_result(
                BatchProcessResult(index=-1, image_path=image_path, image_rgb=image_rgb, restored=restored, mask=mask),
                keep_detections=_bool_setting(item.get("keepDetectionsAfterProcess")),
            )
        except OperationCancelled:
            raise
        except Exception as exc:
            self._mark_process_error_for_path(image_path, str(exc))
            raise

    def _detect_all(
        self,
        selected_templates: list[str] | None = None,
        *,
        detector: str = "template",
        ai_prompt: str | None = None,
        ai_settings: dict[str, Any] | None = None,
        template_settings: dict[str, Any] | None = None,
        image_paths: list[Path] | None = None,
        progress_callback: Any | None = None,
        cancel_callback: Any | None = None,
    ) -> dict[str, Any]:
        if image_paths is not None:
            batch_image_paths = list(image_paths)
        else:
            with self._batch_lock:
                batch_image_paths = list(self.image_paths)
        detector = _normalize_detector_mode(detector)
        if detector == "sam3":
            return self._detect_all_ai(
                ai_prompt,
                ai_settings,
                detector=detector,
                image_paths=batch_image_paths,
                progress_callback=progress_callback,
                cancel_callback=cancel_callback,
            )

        template_paths = self._selected_template_paths(selected_templates)
        if not template_paths:
            raise ValueError("No templates selected.")
        templates = load_templates(template_paths)
        if not templates:
            raise ValueError("No supported template images found.")

        config = self._template_detection_config(template_settings)
        remover = self._new_template_remover(templates, config)
        processed = 0
        failed = 0
        detection_count = 0
        jobs = _normalize_batch_jobs(self.config.batch_detect_jobs, len(batch_image_paths))
        if jobs <= 1:
            results = (
                self._detect_for_batch_path(image_path, templates, config, remover, index=index)
                for index, image_path in enumerate(batch_image_paths)
            )
            for result in results:
                _raise_if_cancelled(cancel_callback)
                if result.error is None:
                    self._commit_batch_detection_for_result(
                        result,
                        template_paths,
                        template_settings=_template_settings_payload(config),
                    )
                    processed += 1
                    detection_count += len(result.detections or [])
                else:
                    failed += 1
                    if result.image_path is not None:
                        self._mark_detection_error_for_path(result.image_path, result.error)
                    else:
                        self._mark_detection_error(result.index, result.error)
                if progress_callback is not None:
                    progress_callback(processed, failed, detection_count)
                _raise_if_cancelled(cancel_callback)
        else:
            executor = self._executor_cls()(max_workers=jobs)
            cancelled = False
            try:
                pending = {
                    executor.submit(self._detect_for_batch_path, image_path, templates, config, remover, index=index): index
                    for index, image_path in enumerate(batch_image_paths)
                }
                while pending:
                    _raise_if_cancelled(cancel_callback)
                    done, _not_done = wait(pending, timeout=0.1, return_when=FIRST_COMPLETED)
                    if not done:
                        continue
                    for future in done:
                        index = pending.pop(future)
                        try:
                            result = future.result()
                        except Exception as exc:  # pragma: no cover - _detect_for_batch normally wraps errors
                            result = BatchDetectionResult(index=index, error=str(exc))
                        _raise_if_cancelled(cancel_callback)
                        if result.error is None:
                            self._commit_batch_detection_for_result(result, template_paths, template_settings=_template_settings_payload(config))
                            processed += 1
                            detection_count += len(result.detections or [])
                        else:
                            failed += 1
                            if result.image_path is not None:
                                self._mark_detection_error_for_path(result.image_path, result.error)
                            else:
                                self._mark_detection_error(result.index, result.error)
                        if progress_callback is not None:
                            progress_callback(processed, failed, detection_count)
                        _raise_if_cancelled(cancel_callback)
            except OperationCancelled:
                cancelled = True
                raise
            finally:
                shutdown = getattr(executor, "shutdown", None)
                if callable(shutdown):
                    shutdown(wait=not cancelled, cancel_futures=True)

        summary = self.summary()
        summary["batch"] = {
            "processedImages": processed,
            "failedImages": failed,
            "detectionCount": detection_count,
            "jobs": jobs,
        }
        return summary

    def _detect_all_ai(
        self,
        ai_prompt: str | None,
        ai_settings: dict[str, Any] | None,
        *,
        detector: str = "sam3",
        image_paths: list[Path] | None = None,
        progress_callback: Any | None = None,
        cancel_callback: Any | None = None,
    ) -> dict[str, Any]:
        if image_paths is not None:
            batch_image_paths = list(image_paths)
        else:
            with self._batch_lock:
                batch_image_paths = list(self.image_paths)
        ai_config = self._ai_detection_config(ai_prompt, ai_settings)
        processed = 0
        failed = 0
        detection_count = 0
        for index, image_path in enumerate(batch_image_paths):
            _raise_if_cancelled(cancel_callback)
            result = self._detect_for_batch_ai_path(image_path, ai_config, index=index)
            _raise_if_cancelled(cancel_callback)
            if result.error is None:
                self._commit_batch_detection_for_result(
                    result,
                    [],
                    detector=detector,
                    ai_prompt=ai_config.prompt,
                    ai_settings=_ai_settings_payload(ai_config),
                )
                processed += 1
                detection_count += len(result.detections or [])
            else:
                failed += 1
                if result.image_path is not None:
                    self._mark_detection_error_for_path(result.image_path, result.error)
                else:
                    self._mark_detection_error(result.index, result.error)
            if progress_callback is not None:
                progress_callback(processed, failed, detection_count)
            _raise_if_cancelled(cancel_callback)

        summary = self.summary()
        summary["batch"] = {
            "processedImages": processed,
            "failedImages": failed,
            "detectionCount": detection_count,
            "jobs": 1,
        }
        return summary

    def _detect_for_batch(
        self,
        index: int,
        templates: list[Any],
        config: WatermarkRemoverConfig,
        remover: Any,
    ) -> BatchDetectionResult:
        return self._detect_for_batch_path(self._image_path(index), templates, config, remover, index=index)

    def _detect_for_batch_path(
        self,
        image_path: Path,
        templates: list[Any],
        config: WatermarkRemoverConfig,
        remover: Any,
        *,
        index: int = -1,
    ) -> BatchDetectionResult:
        try:
            image_rgb = load_image_rgb(image_path)
            detections = remover.detect(image_rgb)
            mask = self._build_template_detection_mask(image_rgb, detections, templates, config)
            return BatchDetectionResult(index=index, image_path=image_path, image_rgb=image_rgb, detections=detections, mask=mask)
        except Exception as exc:
            return BatchDetectionResult(index=index, image_path=image_path, error=str(exc))

    def _detect_for_batch_ai(
        self,
        index: int,
        config: AIDetectionConfig,
    ) -> BatchDetectionResult:
        return self._detect_for_batch_ai_path(self._image_path(index), config, index=index)

    def _detect_for_batch_ai_path(
        self,
        image_path: Path,
        config: AIDetectionConfig,
        *,
        index: int = -1,
    ) -> BatchDetectionResult:
        try:
            image_rgb = load_image_rgb(image_path)
            with self._sam3_detector_lock:
                detector = self._new_sam3_detector(config)
                result = detector.detect(image_rgb)
            return BatchDetectionResult(index=index, image_path=image_path, image_rgb=image_rgb, detections=result.detections, mask=result.mask)
        except Exception as exc:
            return BatchDetectionResult(index=index, image_path=image_path, error=str(exc))

    def _commit_batch_detection_for_result(
        self,
        result: BatchDetectionResult,
        template_paths: list[Path],
        *,
        detector: str = "template",
        ai_prompt: str | None = None,
        ai_settings: dict[str, Any] | None = None,
        template_settings: dict[str, Any] | None = None,
    ) -> None:
        with self._batch_lock:
            if result.image_path is not None:
                result.index = self._image_index_for_path(result.image_path)
            self._commit_batch_detection(
                result,
                template_paths,
                detector=detector,
                ai_prompt=ai_prompt,
                ai_settings=ai_settings,
                template_settings=template_settings,
            )

    def _commit_batch_detection(
        self,
        result: BatchDetectionResult,
        template_paths: list[Path],
        *,
        detector: str = "template",
        ai_prompt: str | None = None,
        ai_settings: dict[str, Any] | None = None,
        template_settings: dict[str, Any] | None = None,
    ) -> None:
        if result.image_rgb is None or result.detections is None or result.mask is None:
            raise ValueError("Batch detection result is incomplete.")
        self._save_mask(result.index, (result.mask * 255.0).round().astype(np.uint8))
        state = self._read_state(result.index)
        state.update(
            {
                "status": "detected",
                "selectedTemplates": [str(path) for path in template_paths],
                "detections": [self._detection_payload(detection) for detection in result.detections],
                "detector": detector,
                "aiPrompt": ai_prompt,
                "aiSettings": ai_settings,
                "templateSettings": template_settings,
                "error": None,
            }
        )
        self._write_state(result.index, state)
        self._save_ui_debug(result.index, result.image_rgb, result.image_rgb, result.mask, "batch_detect")

    def _mark_detection_error(self, index: int, error: str) -> None:
        state = self._read_state(index)
        state["status"] = "error"
        state["detections"] = []
        state["error"] = error
        self._write_state(index, state)
        self._mask_path(index).unlink(missing_ok=True)

    def _mark_detection_error_for_path(self, image_path: Path, error: str) -> None:
        with self._batch_lock:
            self._mark_detection_error(self._image_index_for_path(image_path), error)

    def _detect_with_loaded_templates(
        self,
        index: int,
        templates: list[Any],
        template_paths: list[Path],
        config: WatermarkRemoverConfig,
        remover: Any,
        operation_id: str | None = None,
    ) -> dict[str, Any]:
        image_path = self._image_path(index)
        image_rgb = load_image_rgb(image_path)
        self._raise_if_operation_cancelled(operation_id)
        detections = remover.detect(image_rgb)
        mask = self._build_template_detection_mask(image_rgb, detections, templates, config)
        self._raise_if_operation_cancelled(operation_id)
        self._save_mask(index, (mask * 255.0).round().astype(np.uint8))

        state = self._read_state(index)
        state.update(
            {
                "status": "detected",
                "selectedTemplates": [str(path) for path in template_paths],
                "detections": [self._detection_payload(detection) for detection in detections],
                "detector": "template",
                "aiPrompt": None,
                "aiSettings": None,
                "templateSettings": _template_settings_payload(config),
                "error": None,
            }
        )
        self._write_state(index, state)
        self._save_ui_debug(index, image_rgb, image_rgb, mask, "detect")
        return self._image_summary(index, image_path)

    def _build_template_detection_mask(
        self,
        image_rgb: np.ndarray,
        detections: list[Any],
        templates: list[Any],
        config: WatermarkRemoverConfig,
    ) -> np.ndarray:
        template_mask = self._build_detection_mask(image_rgb, detections, templates, config)
        if not config.sam3_refine_mask or not detections:
            return template_mask

        try:
            ai_config = AIDetectionConfig(
                mask_dilate_pixels=0,
                fallback_to_boxes=False,
                sam3_crop_padding_ratio=0.20,
            )
            with self._sam3_detector_lock:
                refined = self._new_sam3_detector(ai_config).refine_mask(image_rgb, detections)
        except Exception:
            return template_mask

        refined_mask = np.asarray(refined.mask, dtype=np.float32)
        if refined_mask.shape != template_mask.shape:
            return template_mask
        if not np.any(refined_mask > 0.0):
            return template_mask
        return np.clip(refined_mask, 0.0, 1.0).astype(np.float32)

    def _detect_with_ai(
        self,
        index: int,
        ai_prompt: str | None,
        ai_settings: dict[str, Any] | None,
        *,
        detector: str = "sam3",
        operation_id: str | None = None,
    ) -> dict[str, Any]:
        image_path = self._image_path(index)
        image_rgb = load_image_rgb(image_path)
        ai_config = self._ai_detection_config(ai_prompt, ai_settings)
        self._raise_if_operation_cancelled(operation_id)
        with self._sam3_detector_lock:
            ai_detector = self._new_sam3_detector(ai_config)
            result = ai_detector.detect(image_rgb)
        self._raise_if_operation_cancelled(operation_id)
        self._save_mask(index, (result.mask * 255.0).round().astype(np.uint8))

        state = self._read_state(index)
        state.update(
            {
                "status": "detected",
                "selectedTemplates": [],
                "detections": [self._detection_payload(detection) for detection in result.detections],
                "detector": detector,
                "aiPrompt": ai_config.prompt,
                "aiSettings": _ai_settings_payload(ai_config),
                "templateSettings": None,
                "error": None,
            }
        )
        self._write_state(index, state)
        self._save_ui_debug(index, image_rgb, image_rgb, result.mask, "ai_detect")
        return self._image_summary(index, image_path)

    def process_mask(
        self,
        index: int,
        mask_data: str | None = None,
        *,
        keep_detections: bool = False,
        operation_id: str | None = None,
    ) -> dict[str, Any]:
        image_path = self._image_path(index)
        image_rgb = load_image_rgb(image_path)
        mask = _decode_mask_data_url(mask_data, image_rgb.shape[:2]) if mask_data else self._load_mask(index, image_rgb.shape[:2])
        self._raise_if_operation_cancelled(operation_id)
        self._save_mask(index, mask)
        config = WatermarkRemoverConfig(lama_device=self.config.lama_device)
        restored = self._restore_with_mask(image_rgb, mask, config)
        self._raise_if_operation_cancelled(operation_id)
        self._save_result(index, restored)
        state = self._read_state(index)
        state["status"] = "processed"
        state["error"] = None
        if not keep_detections:
            state["detections"] = []
        self._write_state(index, state)
        self._save_ui_debug(index, image_rgb, restored, mask, "process")
        return self._image_summary(index, image_path)

    def process_all(self, *, keep_detections: bool = False) -> dict[str, Any]:
        return self._process_all(keep_detections=keep_detections)

    def _batch_process_image_paths(self, image_paths: list[Path] | None = None) -> list[Path]:
        with self._batch_lock:
            paths = list(self.image_paths) if image_paths is None else list(image_paths)
            return [
                image_path
                for image_path in paths
                if self._mask_has_pixels(self._image_index_for_path(image_path))
            ]

    def _process_all(
        self,
        *,
        keep_detections: bool = False,
        image_paths: list[Path] | None = None,
        progress_callback: Any | None = None,
        cancel_callback: Any | None = None,
    ) -> dict[str, Any]:
        batch_image_paths = self._batch_process_image_paths(image_paths)
        config = WatermarkRemoverConfig(lama_device=self.config.lama_device)
        processed = 0
        failed = 0
        jobs = _normalize_batch_jobs(self.config.batch_process_jobs, len(batch_image_paths))
        if jobs <= 1:
            results = (
                self._process_for_batch_path(image_path, config, index=index)
                for index, image_path in enumerate(batch_image_paths)
            )
            for result in results:
                _raise_if_cancelled(cancel_callback)
                if result.error is None:
                    self._commit_batch_process_for_result(result, keep_detections=keep_detections)
                    processed += 1
                else:
                    if result.image_path is not None:
                        self._mark_process_error_for_path(result.image_path, result.error)
                    else:
                        self._mark_process_error(result.index, result.error)
                    failed += 1
                if progress_callback is not None:
                    progress_callback(processed, failed)
                _raise_if_cancelled(cancel_callback)
        else:
            executor = self._executor_cls()(max_workers=jobs)
            cancelled = False
            try:
                pending = {
                    executor.submit(self._process_for_batch_path, image_path, config, index=index): index
                    for index, image_path in enumerate(batch_image_paths)
                }
                while pending:
                    _raise_if_cancelled(cancel_callback)
                    done, _not_done = wait(pending, timeout=0.1, return_when=FIRST_COMPLETED)
                    if not done:
                        continue
                    for future in done:
                        index = pending.pop(future)
                        try:
                            result = future.result()
                        except Exception as exc:  # pragma: no cover - _process_for_batch normally wraps errors
                            result = BatchProcessResult(index=index, error=str(exc))
                        _raise_if_cancelled(cancel_callback)
                        if result.error is None:
                            self._commit_batch_process_for_result(result, keep_detections=keep_detections)
                            processed += 1
                        else:
                            if result.image_path is not None:
                                self._mark_process_error_for_path(result.image_path, result.error)
                            else:
                                self._mark_process_error(result.index, result.error)
                            failed += 1
                        if progress_callback is not None:
                            progress_callback(processed, failed)
                        _raise_if_cancelled(cancel_callback)
            except OperationCancelled:
                cancelled = True
                raise
            finally:
                shutdown = getattr(executor, "shutdown", None)
                if callable(shutdown):
                    shutdown(wait=not cancelled, cancel_futures=True)

        summary = self.summary()
        summary["batch"] = {
            "processedImages": processed,
            "failedImages": failed,
            "jobs": jobs,
        }
        return summary

    def _process_for_batch(self, index: int, config: WatermarkRemoverConfig) -> BatchProcessResult:
        return self._process_for_batch_path(self._image_path(index), config, index=index)

    def _process_for_batch_path(self, image_path: Path, config: WatermarkRemoverConfig, *, index: int = -1) -> BatchProcessResult:
        try:
            image_rgb = load_image_rgb(image_path)
            with self._batch_lock:
                current_index = self._image_index_for_path(image_path)
                if not self._mask_path(current_index).is_file():
                    raise ValueError("No saved mask for this image.")
                mask = self._load_mask(current_index, image_rgb.shape[:2])
            if not np.any(mask > 0):
                raise ValueError("Saved mask is empty.")
            restored = self._restore_with_mask(image_rgb, mask, config)
            return BatchProcessResult(index=index, image_path=image_path, image_rgb=image_rgb, restored=restored, mask=mask)
        except Exception as exc:
            return BatchProcessResult(index=index, image_path=image_path, error=str(exc))

    def _commit_batch_process_for_result(self, result: BatchProcessResult, *, keep_detections: bool = False) -> None:
        with self._batch_lock:
            if result.image_path is not None:
                result.index = self._image_index_for_path(result.image_path)
            self._commit_batch_process(result, keep_detections=keep_detections)

    def _commit_batch_process(self, result: BatchProcessResult, *, keep_detections: bool = False) -> None:
        if result.image_rgb is None or result.restored is None or result.mask is None:
            raise ValueError("Batch process result is incomplete.")
        self._save_result(result.index, result.restored)
        state = self._read_state(result.index)
        state["status"] = "processed"
        state["error"] = None
        if not keep_detections:
            state["detections"] = []
        self._write_state(result.index, state)
        self._save_ui_debug(result.index, result.image_rgb, result.restored, result.mask, "batch_process")

    def _mark_process_error(self, index: int, error: str) -> None:
        state = self._read_state(index)
        state["status"] = "error"
        state["error"] = error
        self._write_state(index, state)

    def _mark_process_error_for_path(self, image_path: Path, error: str) -> None:
        with self._batch_lock:
            self._mark_process_error(self._image_index_for_path(image_path), error)

    def _ai_detection_config(self, prompt: str | None, settings: dict[str, Any] | None) -> AIDetectionConfig:
        defaults = AIDetectionConfig()
        return AIDetectionConfig(
            prompt=(prompt or DEFAULT_AI_PROMPT).strip() or DEFAULT_AI_PROMPT,
            sam3_confidence_threshold=_ai_float_setting_closed(
                settings,
                "sam3ConfidenceThreshold",
                defaults.sam3_confidence_threshold,
                0.0,
                1.0,
            ),
            device="auto",
            box_threshold=_ai_float_setting(settings, "boxThreshold", defaults.box_threshold, 0.0, 1.0),
            max_box_area_ratio=_ai_float_setting(settings, "maxBoxAreaRatio", defaults.max_box_area_ratio, 0.0, 1.0),
            nms_iou_threshold=_ai_float_setting_closed(settings, "nmsIouThreshold", defaults.nms_iou_threshold, 0.01, 1.0),
            max_detections=_ai_int_setting(settings, "maxDetections", defaults.max_detections, 1, 96),
            mask_threshold=_ai_float_setting_closed(settings, "maskThreshold", defaults.mask_threshold, 0.0, 1.0),
            mask_dilate_pixels=_ai_int_setting(settings, "maskDilatePixels", defaults.mask_dilate_pixels, 0, 64),
            fallback_to_boxes=_ai_bool_setting(settings, "fallbackToBoxes", defaults.fallback_to_boxes),
            sam3_max_side=_ai_int_setting(settings, "sam3MaxSide", defaults.sam3_max_side, 256, 8192),
            sam3_crop_padding_ratio=_ai_float_setting_closed(
                settings,
                "sam3CropPaddingRatio",
                defaults.sam3_crop_padding_ratio,
                0.0,
                2.0,
            ),
            sam3_tile_overlap_ratio=_ai_float_setting_closed(
                settings,
                "sam3TileOverlapRatio",
                defaults.sam3_tile_overlap_ratio,
                0.0,
                0.80,
            ),
        )

    def _template_detection_config(self, settings: dict[str, Any] | None) -> WatermarkRemoverConfig:
        defaults = WatermarkRemoverConfig(
            score_threshold=DEFAULT_WEB_TEMPLATE_SCORE_THRESHOLD,
            min_scale=DEFAULT_WEB_TEMPLATE_MIN_SCALE,
            max_scale=DEFAULT_WEB_TEMPLATE_MAX_SCALE,
        )
        min_scale = _template_float_setting(settings, "minScale", defaults.min_scale, 0.0, 5.0)
        max_scale = _template_float_setting(settings, "maxScale", defaults.max_scale, 0.0, 5.0)
        if min_scale > max_scale:
            raise ValueError("Invalid template setting: minScale")
        return WatermarkRemoverConfig(
            min_scale=min_scale,
            max_scale=max_scale,
            scale_step=_template_float_setting(settings, "scaleStep", defaults.scale_step, 0.0, 1.0),
            score_threshold=_template_float_setting(settings, "scoreThreshold", defaults.score_threshold, 0.0, 1.0),
            max_detections=_template_int_setting(settings, "maxDetections", defaults.max_detections, 1, 240),
            nms_iou_threshold=_template_float_setting(settings, "nmsIouThreshold", defaults.nms_iou_threshold, 0.0, 1.0),
            edge_score_threshold=_template_float_setting(settings, "edgeScoreThreshold", defaults.edge_score_threshold, 0.0, 1.0),
            color_score_threshold=_template_float_setting(settings, "colorScoreThreshold", defaults.color_score_threshold, 0.0, 1.0),
            support_correlation_threshold=_template_float_setting(
                settings,
                "supportCorrelationThreshold",
                defaults.support_correlation_threshold,
                0.0,
                1.0,
            ),
            mask_dilate_iterations=_template_int_setting(
                settings,
                "maskDilateIterations",
                defaults.mask_dilate_iterations,
                0,
                64,
            ),
            mask_dilate_max_body_ratio=_template_float_setting_closed(
                settings,
                "maskDilateMaxBodyRatio",
                defaults.mask_dilate_max_body_ratio,
                0.0,
                1.0,
            ),
            mask_edge_feather_pixels=_template_float_setting_closed(
                settings,
                "maskEdgeFeatherPixels",
                defaults.mask_edge_feather_pixels,
                0.0,
                16.0,
            ),
            mask_unify_body=_template_bool_setting(settings, "maskUnifyBody", defaults.mask_unify_body),
            mask_contour_close_pixels=_template_float_setting_closed(
                settings,
                "maskContourClosePixels",
                defaults.mask_contour_close_pixels,
                0.0,
                32.0,
            ),
            mask_body_gap_ratio=_template_float_setting_closed(
                settings,
                "maskBodyGapRatio",
                defaults.mask_body_gap_ratio,
                0.0,
                0.5,
            ),
            sam3_refine_mask=_template_bool_setting(settings, "sam3RefineMask", defaults.sam3_refine_mask),
            lama_device=self.config.lama_device,
            collect_debug_maps=False,
        )
