from __future__ import annotations

import hashlib
import json
import mimetypes
import re
import shutil
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .masking import restore_original_regions
from .models import IMAGE_EXTENSIONS, Detection
from .templates import list_input_images, load_image_rgb, save_image_like
from .web_images import (
    _change_map,
    _decode_data_url,
    _decode_mask_data_url,
    _decode_rgb_data_url,
    _file_version,
    _image_to_mask,
    _normalize_mask_u8,
    _render_cleanup_change,
    _render_detection_overlay,
    _render_mask_overlay,
    _render_side_by_side,
    _save_rgb,
    _unique_archive_name,
    template_mask_preview_bytes,
    thumbnail_bytes,
)
from .web_settings import DEFAULT_TEMPLATE_PATHS, WORKSPACE_INPUT_ROOT_NAME, _new_temporary_input_path, _new_workspace_input_path, _path_key


class EditorWorkspaceMixin:
    def _input_path_or_default(self, value: str | None, output_dir: Path) -> tuple[Path, bool]:
        if value is None:
            return self.config.source_input_path or self.config.input_path, self.config.input_is_temporary
        value = value.strip()
        if value:
            return Path(value), False
        if self.config.input_is_temporary:
            return self.config.input_path, True
        return _new_temporary_input_path(output_dir), True

    def _summary_input_path(self) -> Path:
        return self.config.source_input_path or self.config.input_path

    def _activate_input_workspace(self) -> None:
        if self.config.input_is_temporary:
            self.config.source_input_path = None
            self.config.input_path.mkdir(parents=True, exist_ok=True)
            return

        source_path = self.config.source_input_path or self.config.input_path
        self._ensure_collection_path(source_path, "Input")
        workspace_path = _new_workspace_input_path(self.config.output_dir)
        workspace_path.mkdir(parents=True, exist_ok=True)
        self.config.source_input_path = source_path
        self.config.input_path = workspace_path
        self._sync_source_input_to_workspace()

    def _sync_source_input_to_workspace(self) -> None:
        if self.config.input_is_temporary or self.config.source_input_path is None:
            return
        self._ensure_collection_path(self.config.source_input_path, "Input")
        self.config.input_path.mkdir(parents=True, exist_ok=True)
        for source_image in list_input_images(self.config.source_input_path):
            if source_image.name in self._deleted_source_image_names:
                continue
            target = self.config.input_path / source_image.name
            try:
                same_file = source_image.resolve() == target.resolve()
            except OSError:
                same_file = False
            if same_file:
                continue
            if target.exists():
                source_stat = source_image.stat()
                target_stat = target.stat()
                if source_stat.st_size == target_stat.st_size and source_stat.st_mtime_ns == target_stat.st_mtime_ns:
                    continue
            shutil.copy2(source_image, target)

    def _archive_active_workspace_input(self) -> None:
        workspace_path = self.config.input_path
        if self.config.input_is_temporary or not workspace_path.is_dir():
            return
        workspace_root = self.config.output_dir / WORKSPACE_INPUT_ROOT_NAME / "images"
        try:
            is_workspace = workspace_path.parent.resolve() == workspace_root.resolve()
        except OSError:
            is_workspace = False
        if not is_workspace or not any(workspace_path.iterdir()):
            return
        trash_dir = self.config.output_dir / ".editor_trash" / "workspace"
        trash_dir.mkdir(parents=True, exist_ok=True)
        target = trash_dir / self._unique_file_name(trash_dir, workspace_path.name)
        shutil.move(str(workspace_path), str(target))

    def _path_or_default(self, value: str | None, current: Path, default: Path) -> Path:
        if value is None:
            return current
        value = value.strip()
        return Path(value) if value else default

    def _template_paths_or_default(self, values: list[str] | None) -> list[Path]:
        if values is None:
            return self.config.template_paths
        paths = [Path(value.strip()) for value in values if value.strip()]
        return paths or list(DEFAULT_TEMPLATE_PATHS)

    def save_mask(self, index: int, mask_data: str, detections: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        image_rgb = load_image_rgb(self._image_path(index))
        mask = _decode_mask_data_url(mask_data, image_rgb.shape[:2])
        self._save_mask(index, mask)
        state = self._read_state(index)
        state["status"] = "edited"
        state["error"] = None
        if detections is not None:
            state["detections"] = detections
        self._write_state(index, state)
        return self._image_summary(index, self._image_path(index))

    def save_result(self, index: int, result_data: str | None = None, mask_data: str | None = None) -> dict[str, Any]:
        image_path = self._image_path(index)
        image_rgb = load_image_rgb(image_path)
        if mask_data:
            self._save_mask(index, _decode_mask_data_url(mask_data, image_rgb.shape[:2]))
        if result_data:
            restored = _decode_rgb_data_url(result_data, image_rgb.shape[:2])
            self._save_result(index, restored)
        elif not self._has_result(index):
            raise ValueError("No result image to save.")
        else:
            restored = load_image_rgb(self._result_path(index))
        state = self._read_state(index)
        state["status"] = "saved"
        state["error"] = None
        self._write_state(index, state)
        self._save_ui_debug(index, image_rgb, restored, self._load_mask(index, image_rgb.shape[:2]), "save")
        return self._image_summary(index, image_path)

    def restore_original(self, index: int, brush_mask_data: str, result_data: str | None = None) -> dict[str, Any]:
        image_path = self._image_path(index)
        original = load_image_rgb(image_path)
        if result_data:
            edited = _decode_rgb_data_url(result_data, original.shape[:2])
        elif self._result_path(index).is_file():
            edited = load_image_rgb(self._result_path(index))
        else:
            edited = original.copy()
        brush_mask = _decode_mask_data_url(brush_mask_data, original.shape[:2])
        restored = restore_original_regions(original, edited, brush_mask)
        self._save_result(index, restored)
        state = self._read_state(index)
        state["status"] = "edited"
        state["error"] = None
        self._write_state(index, state)
        self._save_ui_debug(index, original, restored, self._load_mask(index, original.shape[:2]), "restore_original")
        return self._image_summary(index, image_path)

    def reset_image(self, index: int) -> dict[str, Any]:
        reset = self._reset_image_state(index)
        summary = self._image_summary(index, self._image_path(index))
        summary["resetImage"] = reset
        return summary

    def reset_all_images(self) -> dict[str, Any]:
        reset = 0
        for index, _ in enumerate(self.image_paths):
            if self._reset_image_state(index):
                reset += 1
        summary = self.summary()
        summary["resetImages"] = reset
        return summary

    def create_template(self, index: int, mask_data: str, name: str | None = None) -> dict[str, Any]:
        image_path = self._image_path(index)
        image_rgb = load_image_rgb(image_path)
        mask = _decode_mask_data_url(mask_data, image_rgb.shape[:2])
        ys, xs = np.where(mask > 0)
        if xs.size == 0 or ys.size == 0:
            raise ValueError("Cannot create a template from an empty mask.")

        pad = max(6, int(round(max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1) * 0.08)))
        left = max(0, int(xs.min()) - pad)
        top = max(0, int(ys.min()) - pad)
        right = min(image_rgb.shape[1], int(xs.max()) + pad + 1)
        bottom = min(image_rgb.shape[0], int(ys.max()) + pad + 1)

        crop = image_rgb[top:bottom, left:right].copy()
        crop_mask = mask[top:bottom, left:right] > 0
        outside = crop[~crop_mask]
        if outside.size:
            background = np.median(outside.reshape(-1, 3), axis=0)
        else:
            background = np.median(image_rgb.reshape(-1, 3), axis=0)
        template_rgb = np.empty_like(crop)
        template_rgb[:] = np.clip(background, 0, 255).astype(np.uint8)
        template_rgb[crop_mask] = crop[crop_mask]

        template_dir = self._template_output_dir()
        template_dir.mkdir(parents=True, exist_ok=True)
        filename = self._unique_template_name(template_dir, name or f"{image_path.stem}_template")
        template_path = template_dir / filename
        Image.fromarray(template_rgb, mode="RGB").save(template_path)
        self.reload()

        state = self._read_state(index)
        created = state.setdefault("createdTemplates", [])
        created.append(str(template_path))
        state["status"] = "edited"
        state["error"] = None
        self._write_state(index, state)
        summary = self.summary()
        template_index = self._template_index(template_path)
        summary["createdTemplate"] = self._template_payload(template_index, template_path)
        return summary

    def add_images(self, files: list[dict[str, str]]) -> dict[str, Any]:
        if not self.config.input_path.is_dir():
            raise ValueError("新增圖片需要圖片路徑是資料夾。")
        validated = self._validated_upload_images(files, item_label="圖片")

        added: list[Path] = []
        for raw_name, body in validated:
            filename = self._unique_file_name(self.config.input_path, raw_name)
            target = self.config.input_path / filename
            target.write_bytes(body)
            added.append(target)

        with self._batch_lock:
            self.reload()
            self._refresh_running_job_summaries_locked()
            summary = self.summary()
        summary["addedImages"] = [str(path) for path in added]
        return summary

    def _refresh_running_job_summaries_locked(self) -> None:
        for job in self._batch_jobs.values():
            if job.status != "running":
                continue
            if job.kind == "operation":
                job.summary = self._operation_batch_summary(job)
                continue
            summary = self.summary()
            batch: dict[str, Any] = {
                "processedImages": job.processed,
                "failedImages": job.failed,
                "jobs": job.jobs,
            }
            if job.kind == "detect":
                batch["detectionCount"] = job.detection_count
            summary["batch"] = batch
            job.summary = summary

    def add_templates(self, files: list[dict[str, str]]) -> dict[str, Any]:
        template_dir = self._template_output_dir()
        template_dir.mkdir(parents=True, exist_ok=True)
        if not template_dir.is_dir():
            raise ValueError("新增模板需要模板路徑是資料夾。")
        validated = self._validated_upload_images(files, item_label="模板")

        added: list[Path] = []
        for raw_name, body in validated:
            filename = self._unique_file_name(template_dir, raw_name)
            target = template_dir / filename
            target.write_bytes(body)
            added.append(target)

        self.reload()
        summary = self.summary()
        summary["addedTemplates"] = [str(path) for path in added]
        return summary

    def delete_template(self, template: str) -> dict[str, Any]:
        template_path = self._template_path(template)
        trashed = self._move_to_trash(template_path, "templates")
        self._replace_deleted_template_root(template_path)
        self.reload()
        summary = self.summary()
        summary["deletedTemplate"] = str(template_path)
        summary["trashedFiles"] = [str(trashed)]
        return summary

    def delete_image(self, index: int) -> dict[str, Any]:
        input_was_single_file = self.config.input_path.is_file()
        image_path = self._image_path(index)
        related_paths = [self._state_path(index), self._mask_path(index), self._result_path(index)]
        trashed = [self._move_to_trash(image_path, "images")]
        self._mark_source_image_deleted_if_present(image_path)
        for path in related_paths:
            if path.is_file():
                if path == self._result_path(index):
                    trashed.append(self._move_to_trash(path, "results", target_name=image_path.name))
                else:
                    trashed.append(self._move_to_trash(path, "state"))
        if input_was_single_file:
            self._use_empty_temporary_input()
        self.reload()
        summary = self.summary()
        summary["deletedImage"] = str(image_path)
        summary["trashedFiles"] = [str(path) for path in trashed]
        return summary

    def delete_all_images(self) -> dict[str, Any]:
        input_was_single_file = self.config.input_path.is_file()
        trashed: list[Path] = []
        deleted_count = 0
        for index, image_path in enumerate(list(self.image_paths)):
            related_paths = [self._state_path(index), self._mask_path(index), self._result_path(index)]
            if image_path.is_file():
                trashed.append(self._move_to_trash(image_path, "images"))
                self._mark_source_image_deleted_if_present(image_path)
                deleted_count += 1
            for path in related_paths:
                if path.is_file():
                    if path == self._result_path(index):
                        trashed.append(self._move_to_trash(path, "results", target_name=image_path.name))
                    else:
                        trashed.append(self._move_to_trash(path, "state"))
        if input_was_single_file:
            self._use_empty_temporary_input()
        self.reload()
        summary = self.summary()
        summary["deletedImages"] = deleted_count
        summary["trashedFiles"] = [str(path) for path in trashed]
        return summary

    def clear_all_masks(self) -> dict[str, Any]:
        cleared = 0
        for index, _ in enumerate(self.image_paths):
            mask_path = self._mask_path(index)
            if mask_path.is_file():
                mask_path.unlink()
                cleared += 1
            state = self._read_state(index)
            if state.get("status") == "edited" and not state.get("detections") and not self._has_result(index):
                state["status"] = "pending"
                self._write_state(index, state)
        summary = self.summary()
        summary["clearedMasks"] = cleared
        return summary

    def clear_all_detections(self) -> dict[str, Any]:
        cleared = 0
        for index, _ in enumerate(self.image_paths):
            state = self._read_state(index)
            detections = state.get("detections", [])
            if detections:
                cleared += len(detections)
                state["detections"] = []
                if state.get("status") == "detected":
                    state["status"] = "edited" if self._mask_path(index).is_file() else "pending"
                state["error"] = None
                self._write_state(index, state)
        summary = self.summary()
        summary["clearedDetections"] = cleared
        return summary

    def image_bytes(self, index: int, kind: str) -> tuple[bytes, str]:
        image_path = self._image_path(index)
        if kind == "original":
            return image_path.read_bytes(), mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
        if kind == "thumbnail":
            return self._thumbnail_bytes(image_path)
        if kind == "result":
            path = self._result_path(index)
            source = path if self._has_result(index) else image_path
            return source.read_bytes(), mimetypes.guess_type(source.name)[0] or "application/octet-stream"
        if kind == "mask":
            original = load_image_rgb(image_path)
            mask_path = self._mask_path(index)
            if mask_path.is_file():
                return mask_path.read_bytes(), "image/png"
            blank = Image.new("L", (original.shape[1], original.shape[0]), color=0)
            buffer = BytesIO()
            blank.save(buffer, format="PNG")
            return buffer.getvalue(), "image/png"
        raise ValueError(f"Unsupported image kind: {kind}")

    def result_download(self, index: int) -> tuple[bytes, str, str]:
        result_path = self._result_path(index)
        if not self._has_result(index):
            raise ValueError("No result image to download.")
        content_type = mimetypes.guess_type(result_path.name)[0] or "application/octet-stream"
        return result_path.read_bytes(), content_type, self._image_path(index).name

    def results_zip_download(self) -> tuple[bytes, str, str]:
        buffer = BytesIO()
        names: set[str] = set()
        count = 0
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for index, _ in enumerate(self.image_paths):
                result_path = self._result_path(index)
                if not self._has_result(index):
                    continue
                archive.write(result_path, _unique_archive_name(self._image_path(index).name, names))
                count += 1
        if count == 0:
            raise ValueError("No result images to download.")
        return buffer.getvalue(), "application/zip", "remove-watermark-results.zip"

    def template_bytes(
        self,
        index: int,
        kind: str,
        template_settings: dict[str, Any] | None = None,
    ) -> tuple[bytes, str]:
        template_path = self._template_path_by_index(index)
        if kind == "thumbnail":
            return self._thumbnail_bytes(template_path, segment_extreme=False)
        if kind == "mask-preview":
            return template_mask_preview_bytes(
                template_path,
                self._template_detection_config(template_settings),
            )
        if kind == "mask-preview-full":
            return template_mask_preview_bytes(
                template_path,
                self._template_detection_config(template_settings),
                thumbnail=False,
            )
        if kind == "original":
            return template_path.read_bytes(), mimetypes.guess_type(template_path.name)[0] or "application/octet-stream"
        raise ValueError(f"Unsupported template kind: {kind}")

    def _thumbnail_bytes(self, image_path: Path, *, segment_extreme: bool = True) -> tuple[bytes, str]:
        return thumbnail_bytes(image_path, segment_extreme=segment_extreme)

    def _image_summary(self, index: int, image_path: Path) -> dict[str, Any]:
        state = self._read_state(index)
        return {
            "index": index,
            "name": image_path.name,
            "path": str(image_path),
            "imageKey": _path_key(image_path),
            "originalVersion": _file_version(image_path),
            "maskVersion": _file_version(self._mask_path(index)),
            "resultVersion": _file_version(self._result_path(index)),
            "status": state.get("status", "pending"),
            "detections": state.get("detections", []),
            "selectedTemplates": state.get("selectedTemplates", []),
            "detector": state.get("detector", "template"),
            "aiPrompt": state.get("aiPrompt"),
            "aiSettings": state.get("aiSettings"),
            "templateSettings": state.get("templateSettings"),
            "hasMask": self._mask_has_pixels(index),
            "hasResult": self._has_result(index),
            "resultPath": str(self._result_path(index)),
            "error": state.get("error"),
        }

    def _template_payload(self, index: int, path: Path) -> dict[str, Any]:
        return {"index": index, "name": path.stem, "path": str(path)}

    def _template_index(self, template_path: Path) -> int:
        for index, path in enumerate(self.template_files):
            try:
                if path.resolve() == template_path.resolve():
                    return index
            except OSError:
                if path == template_path:
                    return index
        raise ValueError(f"Template not found: {template_path}")

    def _selected_template_paths(self, selected_templates: list[str] | None) -> list[Path]:
        if not selected_templates:
            return self.template_files
        selected = set(selected_templates)
        return [
            path
            for path in self.template_files
            if str(path) in selected or path.name in selected or path.stem in selected
        ]

    def _template_image_candidates(self, paths: list[Path]) -> list[Path]:
        candidates: list[Path] = []
        for path in paths:
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                candidates.append(path)
            elif path.is_dir():
                images = sorted(file for file in path.iterdir() if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS)
                if images:
                    candidates.extend(images)
                elif (path / "templates").is_dir():
                    candidates.extend(self._template_image_candidates([path / "templates"]))
        return list(dict.fromkeys(candidates))

    def _image_path(self, index: int) -> Path:
        if index < 0 or index >= len(self.image_paths):
            raise ValueError(f"Image index out of range: {index}")
        return self.image_paths[index]

    def _image_index_for_path(self, image_path: Path) -> int:
        image_key = _path_key(image_path)
        for index, path in enumerate(self.image_paths):
            if _path_key(path) == image_key:
                return index
        raise ValueError(f"Image is no longer in the list: {image_path}")

    def _image_path_from_payload(self, payload: dict[str, Any], fallback_index: int) -> Path:
        requested = str(payload.get("imageKey") or payload.get("imagePath") or payload.get("path") or "").strip()
        if requested:
            requested_path = Path(requested)
            image_key = _path_key(requested_path)
            for path in self.image_paths:
                if _path_key(path) == image_key:
                    return path
            source_match = self._workspace_image_for_source_path(requested_path)
            if source_match is not None:
                return source_match
            raise ValueError(f"Image is no longer in the list: {requested}")
        return self._image_path(fallback_index)

    def _workspace_image_for_source_path(self, requested_path: Path) -> Path | None:
        source_root = self.config.source_input_path
        if source_root is None:
            return None
        try:
            if source_root.is_file():
                matches_source = requested_path.resolve() == source_root.resolve()
            else:
                matches_source = requested_path.parent.resolve() == source_root.resolve()
        except OSError:
            matches_source = False
        if not matches_source:
            return None

        workspace_path = self.config.input_path / requested_path.name
        for path in self.image_paths:
            try:
                if path.resolve() == workspace_path.resolve():
                    return path
            except OSError:
                if path == workspace_path:
                    return path
        return None

    def _mark_source_image_deleted_if_present(self, workspace_image_path: Path) -> None:
        source_root = self.config.source_input_path
        if source_root is None:
            return
        source_path = source_root if source_root.is_file() else source_root / workspace_image_path.name
        if source_path.is_file():
            self._deleted_source_image_names.add(workspace_image_path.name)

    def _template_path_by_index(self, index: int) -> Path:
        if index < 0 or index >= len(self.template_files):
            raise ValueError(f"Template index out of range: {index}")
        return self.template_files[index]

    def _state_key(self, index: int) -> str:
        image_path = self._image_path(index)
        digest = hashlib.sha1(str(image_path.resolve()).encode("utf-8")).hexdigest()[:10]
        return f"{image_path.stem}-{digest}"

    def _state_path(self, index: int) -> Path:
        return self.state_dir / f"{self._state_key(index)}.json"

    def _mask_path(self, index: int) -> Path:
        return self.state_dir / f"{self._state_key(index)}_mask.png"

    def _mask_has_pixels(self, index: int) -> bool:
        path = self._mask_path(index)
        if not path.is_file():
            return False
        with Image.open(path) as image:
            return image.convert("L").getbbox() is not None

    def _result_path(self, index: int) -> Path:
        image_path = self._image_path(index)
        return self.state_dir / "results" / f"{self._state_key(index)}{image_path.suffix.lower()}"

    def _has_result(self, index: int) -> bool:
        result_path = self._result_path(index)
        return result_path.is_file() and _path_key(result_path) != _path_key(self._image_path(index))

    def _read_state(self, index: int) -> dict[str, Any]:
        path = self._state_path(index)
        if not path.is_file():
            return {"status": "pending", "detections": [], "selectedTemplates": []}
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_state(self, index: int, state: dict[str, Any]) -> None:
        self._state_path(index).write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

    def _reset_image_state(self, index: int) -> bool:
        changed = False
        for path in (self._state_path(index), self._mask_path(index)):
            if path.is_file():
                path.unlink()
                changed = True
        result_path = self._result_path(index)
        if self._has_result(index):
            result_path.unlink()
            changed = True
        debug_dir = self.config.output_dir / "debug-ui" / self._image_path(index).stem
        if debug_dir.is_dir():
            shutil.rmtree(debug_dir)
            changed = True
        return changed

    def _save_mask(self, index: int, mask: np.ndarray) -> None:
        mask_u8 = np.clip(mask, 0, 255).astype(np.uint8)
        path = self._mask_path(index)
        if not np.any(mask_u8 > 0):
            path.unlink(missing_ok=True)
            return
        Image.fromarray(mask_u8, mode="L").save(path)

    def _load_mask(self, index: int, shape: tuple[int, int]) -> np.ndarray:
        path = self._mask_path(index)
        if not path.is_file():
            return np.zeros(shape, dtype=np.uint8)
        with Image.open(path) as image:
            return _image_to_mask(image, shape)

    def _save_result(self, index: int, image_rgb: np.ndarray) -> None:
        image_path = self._image_path(index)
        result_path = self._result_path(index)
        if result_path.resolve() == image_path.resolve():
            raise ValueError(f"Refusing to overwrite input image: {image_path}")
        result_path.parent.mkdir(parents=True, exist_ok=True)
        save_image_like(result_path, image_rgb, image_path)

    def _save_ui_debug(self, index: int, original_rgb: np.ndarray, restored_rgb: np.ndarray, mask: np.ndarray, action: str) -> None:
        if not self.config.save_debug:
            return
        image_path = self._image_path(index)
        debug_dir = self.config.output_dir / "debug-ui" / image_path.stem
        debug_dir.mkdir(parents=True, exist_ok=True)
        normalized_mask = _normalize_mask_u8(mask, original_rgb.shape[:2])
        state = self._read_state(index)
        files = {
            "input_output_compare": "input_output_compare.png",
            "manual_mask": "manual_mask.png",
            "manual_mask_overlay": "manual_mask_overlay.png",
            "cleanup_change": "cleanup_change.png",
            "detection_overlay": "detection_overlay.png",
        }
        _save_rgb(debug_dir / files["input_output_compare"], _render_side_by_side(original_rgb, restored_rgb))
        Image.fromarray(normalized_mask, mode="L").save(debug_dir / files["manual_mask"])
        _save_rgb(debug_dir / files["manual_mask_overlay"], _render_mask_overlay(original_rgb, normalized_mask, (94, 196, 167)))
        _save_rgb(debug_dir / files["cleanup_change"], _render_cleanup_change(original_rgb, restored_rgb))
        _save_rgb(debug_dir / files["detection_overlay"], _render_detection_overlay(original_rgb, state.get("detections", [])))
        payload = {
            "action": action,
            "image": str(image_path),
            "result": str(self._result_path(index)),
            "mask": str(self._mask_path(index)),
            "mask_pixels": int((normalized_mask > 0).sum()),
            "changed_pixels": int((_change_map(original_rgb, restored_rgb) > 0).sum()),
            "detections": state.get("detections", []),
            "selectedTemplates": state.get("selectedTemplates", []),
            "detector": state.get("detector", "template"),
            "aiPrompt": state.get("aiPrompt"),
            "aiSettings": state.get("aiSettings"),
            "templateSettings": state.get("templateSettings"),
            "files": files,
        }
        (debug_dir / "state.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _template_output_dir(self) -> Path:
        for path in self.config.template_paths:
            if path.is_dir():
                return path
            if path.parent.exists():
                return path.parent
        return Path("templates")

    def _template_path(self, value: str) -> Path:
        if not value:
            raise ValueError("沒有指定要刪除的模板。")
        value_path = Path(value)
        for path in self.template_files:
            if value in {str(path), path.name, path.stem}:
                return path
            try:
                if value_path.resolve() == path.resolve():
                    return path
            except OSError:
                continue
        raise ValueError(f"找不到模板：{value}")

    def _replace_deleted_template_root(self, deleted_path: Path) -> None:
        next_roots: list[Path] = []
        for root in self.config.template_paths:
            try:
                root_matches_deleted_file = root.resolve() == deleted_path.resolve()
            except OSError:
                root_matches_deleted_file = False
            if root_matches_deleted_file:
                replacement = deleted_path.parent
                if replacement not in next_roots:
                    next_roots.append(replacement)
            elif root not in next_roots:
                next_roots.append(root)
        self.config.template_paths = next_roots or list(DEFAULT_TEMPLATE_PATHS)

    def _validated_upload_images(self, files: list[dict[str, str]], *, item_label: str) -> list[tuple[str, bytes]]:
        if not files:
            raise ValueError(f"沒有選擇{item_label}。")

        validated: list[tuple[str, bytes]] = []
        for item in files:
            raw_name = item.get("name", "")
            data = item.get("data", "")
            suffix = Path(raw_name).suffix.lower()
            if suffix not in IMAGE_EXTENSIONS:
                raise ValueError(f"不支援的{item_label}格式：{raw_name}")
            body = _decode_data_url(data)
            try:
                with Image.open(BytesIO(body)) as image:
                    image.verify()
            except Exception as exc:
                raise ValueError(f"不是有效{item_label}：{raw_name}") from exc
            validated.append((raw_name, body))
        return validated

    def _ensure_collection_path(self, path: Path, label: str) -> None:
        if path.exists():
            return
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            raise ValueError(f"{label} path not found: {path}")
        path.mkdir(parents=True, exist_ok=True)

    def _use_empty_temporary_input(self) -> None:
        self.config.input_path = _new_temporary_input_path(self.config.output_dir)
        self.config.input_is_temporary = True

    def _unique_template_name(self, template_dir: Path, raw_name: str) -> str:
        base = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", raw_name).strip(" ._")
        if not base:
            base = "template"
        candidate = f"{base}.png"
        counter = 2
        while (template_dir / candidate).exists():
            candidate = f"{base}_{counter}.png"
            counter += 1
        return candidate

    def _unique_file_name(self, directory: Path, raw_name: str) -> str:
        source_name = Path(raw_name).name
        suffix = Path(source_name).suffix.lower()
        base = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", Path(source_name).stem).strip(" ._")
        if not base:
            base = "image"
        candidate = f"{base}{suffix}"
        counter = 2
        while (directory / candidate).exists():
            candidate = f"{base}_{counter}{suffix}"
            counter += 1
        return candidate

    def _move_to_trash(self, path: Path, category: str, *, target_name: str | None = None) -> Path:
        trash_dir = self.config.output_dir / ".editor_trash" / category
        trash_dir.mkdir(parents=True, exist_ok=True)
        target = trash_dir / self._unique_file_name(trash_dir, target_name or path.name)
        shutil.move(str(path), str(target))
        return target

    def _detection_payload(self, detection: Detection) -> dict[str, Any]:
        payload = {
            "bbox": list(detection.bbox),
            "content_bbox": list(detection.content_bbox or detection.bbox),
            "template": detection.template_name,
            "scale": detection.scale,
            "score": detection.score,
            "color_score": detection.color_score,
            "method": detection.method,
        }
        diagnostics = _detection_diagnostics_payload(detection)
        if diagnostics:
            payload["diagnostics"] = diagnostics
        return payload


def _detection_diagnostics_payload(detection: Detection) -> dict[str, Any]:
    final_decision = detection.stage_metrics.get("final_decision")
    if not isinstance(final_decision, dict):
        return {}
    score = final_decision.get("score")
    if not isinstance(score, dict):
        score = {}
    return {
        "profile": final_decision.get("profile"),
        "acceptanceThreshold": final_decision.get("acceptanceThreshold"),
        "fitThreshold": final_decision.get("fitThreshold"),
        "activeGroups": final_decision.get("activeGroups") or [],
        "scoreGroups": final_decision.get("scoreGroups") or {},
        "risk": score.get("risk"),
        "finalScore": score.get("final_total"),
        "evidenceCount": score.get("evidence_count"),
    }
