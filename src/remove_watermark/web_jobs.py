from __future__ import annotations

import uuid
from pathlib import Path
from threading import Thread
from typing import Any

from .web_models import BatchJob, OperationCancelled
from .web_settings import (
    _bool_setting,
    _normalize_batch_jobs,
    _normalize_detector_mode,
    _normalize_operation_mode,
    _path_key,
    _raise_if_cancelled,
)


_TERMINAL_BATCH_STATUSES = {"completed", "cancelled", "failed"}
_RETAINED_FINISHED_BATCH_JOBS = 8


class EditorJobsMixin:
    def start_detect_all(
        self,
        selected_templates: list[str] | None = None,
        *,
        detector: str = "template",
        ai_prompt: str | None = None,
        ai_settings: dict[str, Any] | None = None,
        template_settings: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        detector = _normalize_detector_mode(detector)
        with self._batch_lock:
            self._prune_finished_batch_jobs_locked()
            running = next((job for job in self._batch_jobs.values() if job.status == "running"), None)
            if running is not None:
                return {"job": running.payload()}

            image_paths = list(self.image_paths)
            job = BatchJob(
                id=uuid.uuid4().hex,
                kind="detect",
                status="running",
                total=len(image_paths),
                jobs=1
                if detector == "sam3" else _normalize_batch_jobs(self.config.batch_detect_jobs, len(image_paths)),
            )
            self._batch_jobs[job.id] = job

        thread = Thread(
            target=self._run_detect_job,
            args=(job.id, image_paths, selected_templates, detector, ai_prompt, ai_settings, template_settings),
            daemon=True,
        )
        thread.start()
        return {"job": job.payload()}

    def batch_job(self, job_id: str) -> dict[str, Any]:
        with self._batch_lock:
            job = self._batch_jobs.get(job_id)
            if job is None:
                raise ValueError("Batch job not found.")
            return {"job": job.payload()}

    def _finalize_batch_job_locked(self, job: BatchJob) -> None:
        job.release_payloads()
        job.active_indices.clear()
        job.active_paths.clear()
        self._refresh_operation_batch_active_locked(job)
        self._prune_finished_batch_jobs_locked()

    def _prune_finished_batch_jobs_locked(self) -> None:
        finished_ids = [
            job_id
            for job_id, job in self._batch_jobs.items()
            if job.status in _TERMINAL_BATCH_STATUSES
        ]
        overflow = len(finished_ids) - _RETAINED_FINISHED_BATCH_JOBS
        if overflow <= 0:
            return
        for job_id in finished_ids[:overflow]:
            self._batch_jobs.pop(job_id, None)

    def start_operation_batch(self, mode: str, payload: dict[str, Any]) -> dict[str, Any]:
        mode = _normalize_operation_mode(mode)
        with self._batch_condition:
            self._prune_finished_batch_jobs_locked()
            item = self._operation_batch_item(mode, payload)
            detector = self._operation_batch_detector(item) if mode in {"detect", "detectProcess"} else ""
            running = next((job for job in self._batch_jobs.values() if job.status == "running"), None)
            if running is not None:
                if running.kind == "operation" and running.mode == mode and not running.cancel_requested:
                    self._validate_operation_batch_item_locked(running, item)
                    self._append_operation_batch_item_locked(running, item)
                    return {"job": running.payload()}
                raise ValueError("Another batch job is already running.")

            job = BatchJob(
                id=uuid.uuid4().hex,
                kind="operation",
                status="running",
                total=0,
                jobs=self._operation_batch_jobs_for_item(mode, item),
                mode=mode,
                detector=detector,
            )
            self._append_operation_batch_item_locked(job, item)
            self._batch_jobs[job.id] = job
            job_id = job.id
            jobs = job.jobs
            payload = job.payload()

        for _ in range(jobs):
            thread = Thread(target=self._run_operation_batch_job, args=(job_id,), daemon=True)
            thread.start()
        return {"job": payload}

    def add_operation_batch(self, job_id: str, mode: str, payload: dict[str, Any]) -> dict[str, Any]:
        mode = _normalize_operation_mode(mode)
        with self._batch_condition:
            item = self._operation_batch_item(mode, payload)
            job = self._batch_jobs.get(job_id)
            if job is None:
                raise ValueError("Batch job not found.")
            if job.status != "running":
                raise ValueError("Batch job is no longer running.")
            if job.cancel_requested:
                raise ValueError("Batch job is cancelling.")
            if job.kind != "operation" or job.mode != mode:
                raise ValueError("Only the same operation type can be added.")
            self._validate_operation_batch_item_locked(job, item)
            self._append_operation_batch_item_locked(job, item)
            return {"job": job.payload()}

    def _operation_batch_item(self, mode: str, payload: dict[str, Any]) -> dict[str, Any]:
        index = int(payload["index"])
        image_path = self._image_path_from_payload(payload, index)
        return {
            "mode": mode,
            "index": index,
            "imagePath": str(image_path),
            "templates": payload.get("templates"),
            "detector": str(payload.get("detector", "template")),
            "aiPrompt": payload.get("aiPrompt"),
            "aiSettings": payload.get("aiSettings"),
            "templateSettings": payload.get("templateSettings"),
            "maskData": payload.get("maskData"),
            "keepDetectionsAfterProcess": _bool_setting(payload.get("keepDetectionsAfterProcess")),
        }

    def _operation_batch_detector(self, item: dict[str, Any]) -> str:
        return _normalize_detector_mode(str(item.get("detector", "template")))

    def _operation_batch_jobs_for_item(self, mode: str, item: dict[str, Any]) -> int:
        if mode in {"detect", "detectProcess"} and self._operation_batch_detector(item) == "template":
            return _normalize_batch_jobs(self.config.batch_detect_jobs, len(self.image_paths))
        return 1

    def _validate_operation_batch_item_locked(self, job: BatchJob, item: dict[str, Any]) -> None:
        if job.mode not in {"detect", "detectProcess"}:
            return
        detector = self._operation_batch_detector(item)
        if job.detector and detector != job.detector:
            raise ValueError("Only the same detector type can be added.")

    def _append_operation_batch_item_locked(self, job: BatchJob, item: dict[str, Any]) -> None:
        if job.status != "running":
            raise ValueError("Batch job is no longer running.")
        if job.cancel_requested:
            raise ValueError("Batch job is cancelling.")
        index = int(item["index"])
        image_key = _path_key(Path(str(item["imagePath"])))
        if image_key in job.item_paths:
            raise ValueError("This image is already in the current batch job.")
        job.items.append(item)
        job.item_indices.add(index)
        job.item_paths.add(image_key)
        job.total = len(job.items)
        job.summary = self._operation_batch_summary(job)
        self._batch_condition.notify_all()

    def _operation_batch_summary(self, job: BatchJob) -> dict[str, Any]:
        summary = self.summary()
        batch: dict[str, Any] = {
            "processedImages": job.processed,
            "failedImages": job.failed,
            "jobs": job.jobs,
            "totalImages": job.total,
        }
        if job.mode in {"detect", "detectProcess"}:
            batch["detectionCount"] = job.detection_count
        summary["batch"] = batch
        return summary

    def _refresh_operation_batch_active_locked(self, job: BatchJob) -> None:
        active_indices = sorted(job.active_indices)
        active_paths = sorted(job.active_paths)
        job.active_index = active_indices[0] if active_indices else None
        job.active_path = active_paths[0] if active_paths else None

    def _run_operation_batch_job(self, job_id: str) -> None:
        try:
            while True:
                with self._batch_condition:
                    job = self._batch_jobs[job_id]
                    while True:
                        if job.status != "running":
                            return
                        if job.next_item >= len(job.items) and not job.active_indices:
                            job.status = "completed"
                            self._refresh_operation_batch_active_locked(job)
                            job.summary = self._operation_batch_summary(job)
                            self._finalize_batch_job_locked(job)
                            self._batch_condition.notify_all()
                            return
                        if job.cancel_requested:
                            raise OperationCancelled("作業已中斷")
                        if job.next_item < len(job.items):
                            item = dict(job.items[job.next_item])
                            job.next_item += 1
                            active_path = Path(str(item["imagePath"]))
                            active_index = self._image_index_for_path(active_path)
                            job.active_indices.add(active_index)
                            job.active_paths.add(str(active_path))
                            self._refresh_operation_batch_active_locked(job)
                            job.summary = self._operation_batch_summary(job)
                            self._batch_condition.notify_all()
                            break
                        self._batch_condition.wait(timeout=0.2)

                try:
                    detection_delta = self._execute_operation_batch_item(job_id, item)
                    with self._batch_condition:
                        job = self._batch_jobs[job_id]
                        if job.status != "running":
                            return
                        job.processed += 1
                        job.detection_count += detection_delta
                        active_path = Path(str(item["imagePath"]))
                        job.active_indices.discard(self._image_index_for_path(active_path))
                        job.active_paths.discard(str(active_path))
                        self._refresh_operation_batch_active_locked(job)
                        job.summary = self._operation_batch_summary(job)
                        self._batch_condition.notify_all()
                except OperationCancelled:
                    raise
                except Exception:
                    with self._batch_condition:
                        job = self._batch_jobs[job_id]
                        if job.status != "running":
                            return
                        job.failed += 1
                        active_path = Path(str(item["imagePath"]))
                        job.active_indices.discard(self._image_index_for_path(active_path))
                        job.active_paths.discard(str(active_path))
                        self._refresh_operation_batch_active_locked(job)
                        job.summary = self._operation_batch_summary(job)
                        self._batch_condition.notify_all()
        except OperationCancelled as exc:
            with self._batch_condition:
                job = self._batch_jobs[job_id]
                job.status = "cancelled"
                job.error = str(exc)
                job.active_indices.clear()
                job.active_paths.clear()
                self._refresh_operation_batch_active_locked(job)
                job.summary = self._operation_batch_summary(job)
                self._finalize_batch_job_locked(job)
                self._batch_condition.notify_all()
        except Exception as exc:
            with self._batch_condition:
                job = self._batch_jobs[job_id]
                job.status = "failed"
                job.error = str(exc)
                job.active_indices.clear()
                job.active_paths.clear()
                self._refresh_operation_batch_active_locked(job)
                if job.summary is None:
                    job.summary = self.summary()
                self._finalize_batch_job_locked(job)
                self._batch_condition.notify_all()

    def _execute_operation_batch_item(self, job_id: str, item: dict[str, Any]) -> int:
        mode = _normalize_operation_mode(str(item.get("mode", "")))
        if mode == "detect":
            return self._execute_operation_batch_detect(job_id, item)
        if mode == "process":
            self._execute_operation_batch_process(job_id, item)
            return 0

        detection_count = self._execute_operation_batch_detect(job_id, item)
        if detection_count > 0:
            process_item = dict(item)
            process_item["maskData"] = None
            with self._operation_process_lock:
                _raise_if_cancelled(lambda: self._batch_cancel_requested(job_id))
                self._execute_operation_batch_process(job_id, process_item)
        return detection_count

    def cancel_processing(self, operation_id: str | None = None, job_id: str | None = None) -> dict[str, Any]:
        cancelled_operation = False
        cancelled_job: dict[str, Any] | None = None
        with self._batch_condition:
            if operation_id:
                self._cancelled_operations.add(operation_id)
                cancelled_operation = True
            if job_id:
                job = self._batch_jobs.get(job_id)
                if job is None:
                    raise ValueError("Batch job not found.")
                if job.status == "running":
                    job.cancel_requested = True
                cancelled_job = job.payload()
            self._batch_condition.notify_all()
        return {"cancelledOperation": cancelled_operation, "job": cancelled_job}

    def _operation_cancelled(self, operation_id: str | None) -> bool:
        if not operation_id:
            return False
        with self._batch_lock:
            if operation_id not in self._cancelled_operations:
                return False
            self._cancelled_operations.discard(operation_id)
            return True

    def _raise_if_operation_cancelled(self, operation_id: str | None) -> None:
        if self._operation_cancelled(operation_id):
            raise OperationCancelled("作業已中斷")

    def _batch_cancel_requested(self, job_id: str) -> bool:
        with self._batch_lock:
            job = self._batch_jobs.get(job_id)
            return bool(job and job.cancel_requested)

    def start_process_all(self, *, keep_detections: bool = False) -> dict[str, Any]:
        with self._batch_lock:
            self._prune_finished_batch_jobs_locked()
            running = next((job for job in self._batch_jobs.values() if job.status == "running"), None)
            if running is not None:
                return {"job": running.payload()}

            image_paths = self._batch_process_image_paths()
            job = BatchJob(
                id=uuid.uuid4().hex,
                kind="process",
                status="running",
                total=len(image_paths),
                jobs=_normalize_batch_jobs(self.config.batch_process_jobs, len(image_paths)),
            )
            self._batch_jobs[job.id] = job

        thread = Thread(target=self._run_process_job, args=(job.id, image_paths, keep_detections), daemon=True)
        thread.start()
        return {"job": job.payload()}

    def _run_detect_job(
        self,
        job_id: str,
        image_paths: list[Path],
        selected_templates: list[str] | None,
        detector: str,
        ai_prompt: str | None,
        ai_settings: dict[str, Any] | None,
        template_settings: dict[str, Any] | None,
    ) -> None:
        def cancelled() -> bool:
            return self._batch_cancel_requested(job_id)

        def update(processed: int, failed: int, detection_count: int) -> None:
            summary = self.summary()
            summary["batch"] = {
                "processedImages": processed,
                "failedImages": failed,
                "detectionCount": detection_count,
                "jobs": 1 if detector == "sam3" else _normalize_batch_jobs(self.config.batch_detect_jobs, len(image_paths)),
            }
            with self._batch_lock:
                job = self._batch_jobs.get(job_id)
                if job is None:
                    return
                job.processed = processed
                job.failed = failed
                job.detection_count = detection_count
                job.summary = summary

        try:
            summary = self._detect_all(
                selected_templates,
                detector=detector,
                ai_prompt=ai_prompt,
                ai_settings=ai_settings,
                template_settings=template_settings,
                image_paths=image_paths,
                progress_callback=update,
                cancel_callback=cancelled,
            )
            with self._batch_lock:
                job = self._batch_jobs[job_id]
                batch = summary.get("batch", {})
                job.status = "completed"
                job.processed = int(batch.get("processedImages", job.processed))
                job.failed = int(batch.get("failedImages", job.failed))
                job.detection_count = int(batch.get("detectionCount", job.detection_count))
                job.jobs = int(batch.get("jobs", job.jobs))
                job.summary = summary
                self._finalize_batch_job_locked(job)
        except OperationCancelled as exc:
            with self._batch_lock:
                job = self._batch_jobs[job_id]
                job.status = "cancelled"
                job.error = str(exc)
                if job.summary is None:
                    job.summary = self.summary()
                self._finalize_batch_job_locked(job)
        except Exception as exc:
            with self._batch_lock:
                job = self._batch_jobs[job_id]
                job.status = "failed"
                job.error = str(exc)
                self._finalize_batch_job_locked(job)

    def _run_process_job(self, job_id: str, image_paths: list[Path], keep_detections: bool) -> None:
        def cancelled() -> bool:
            return self._batch_cancel_requested(job_id)

        def update(processed: int, failed: int) -> None:
            summary = self.summary()
            summary["batch"] = {
                "processedImages": processed,
                "failedImages": failed,
                "jobs": _normalize_batch_jobs(self.config.batch_process_jobs, len(image_paths)),
            }
            with self._batch_lock:
                job = self._batch_jobs.get(job_id)
                if job is None:
                    return
                job.processed = processed
                job.failed = failed
                job.summary = summary

        try:
            summary = self._process_all(
                keep_detections=keep_detections,
                image_paths=image_paths,
                progress_callback=update,
                cancel_callback=cancelled,
            )
            with self._batch_lock:
                job = self._batch_jobs[job_id]
                batch = summary.get("batch", {})
                job.status = "completed"
                job.processed = int(batch.get("processedImages", job.processed))
                job.failed = int(batch.get("failedImages", job.failed))
                job.jobs = int(batch.get("jobs", job.jobs))
                job.summary = summary
                self._finalize_batch_job_locked(job)
        except OperationCancelled as exc:
            with self._batch_lock:
                job = self._batch_jobs[job_id]
                job.status = "cancelled"
                job.error = str(exc)
                if job.summary is None:
                    job.summary = self.summary()
                self._finalize_batch_job_locked(job)
        except Exception as exc:
            with self._batch_lock:
                job = self._batch_jobs[job_id]
                job.status = "failed"
                job.error = str(exc)
                self._finalize_batch_job_locked(job)
