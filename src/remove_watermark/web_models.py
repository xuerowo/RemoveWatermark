from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np

from .models import Detection


@dataclass(slots=True)
class EditorConfig:
    input_path: Path
    template_paths: list[Path]
    output_dir: Path
    lama_device: str = "auto"
    input_is_temporary: bool = False
    source_input_path: Path | None = None
    batch_detect_jobs: int = 0
    batch_process_jobs: int = 1
    save_debug: bool = False
    settings_file: Path | None = None


@dataclass(slots=True)
class BatchDetectionResult:
    index: int
    image_path: Path | None = None
    image_rgb: np.ndarray | None = None
    detections: list[Detection] | None = None
    mask: np.ndarray | None = None
    error: str | None = None


@dataclass(slots=True)
class BatchProcessResult:
    index: int
    image_path: Path | None = None
    image_rgb: np.ndarray | None = None
    restored: np.ndarray | None = None
    mask: np.ndarray | None = None
    error: str | None = None


@dataclass(slots=True)
class BatchJob:
    id: str
    kind: str
    status: str
    total: int
    processed: int = 0
    failed: int = 0
    detection_count: int = 0
    jobs: int = 1
    error: str | None = None
    summary: dict[str, Any] | None = None
    cancel_requested: bool = False
    mode: str = ""
    active_index: int | None = None
    active_path: str | None = None
    detector: str = ""
    next_item: int = 0
    active_indices: set[int] = field(default_factory=set)
    active_paths: set[str] = field(default_factory=set)
    items: list[dict[str, Any]] = field(default_factory=list)
    item_indices: set[int] = field(default_factory=set)
    item_paths: set[str] = field(default_factory=set)
    operation_cache: dict[Any, Any] = field(default_factory=dict, repr=False)
    operation_cache_lock: Any = field(default_factory=Lock, repr=False)

    def release_payloads(self) -> None:
        self.items.clear()
        self.operation_cache.clear()

    def payload(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "totalImages": self.total,
            "processedImages": self.processed,
            "failedImages": self.failed,
            "remainingImages": max(0, self.total - self.processed - self.failed),
            "detectionCount": self.detection_count,
            "jobs": self.jobs,
            "error": self.error,
            "summary": self.summary,
            "cancelRequested": self.cancel_requested,
            "mode": self.mode,
            "detector": self.detector,
            "activeIndex": self.active_index,
            "activeImagePath": self.active_path,
            "activeIndices": sorted(self.active_indices),
            "activeImagePaths": sorted(self.active_paths),
            "queuedImages": max(0, self.total - self.processed - self.failed - len(self.active_indices)),
            "itemIndices": sorted(self.item_indices),
            "itemPaths": sorted(self.item_paths),
        }


class OperationCancelled(RuntimeError):
    pass
