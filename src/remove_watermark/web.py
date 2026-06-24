from __future__ import annotations

import argparse
import json
import mimetypes
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Condition, RLock, Thread
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlparse

import numpy as np

from .ai_detection import DEFAULT_AI_PROMPT, AIDetectionConfig, Sam3TextWatermarkDetector
from .detection import WatermarkRemover
from .masking import build_detection_mask, restore_with_mask
from .models import Detection, WatermarkRemoverConfig
from .templates import list_input_images
from .web_images import (
    THUMBNAIL_BG as THUMBNAIL_BG,
    THUMBNAIL_DIVIDER as THUMBNAIL_DIVIDER,
    THUMBNAIL_EXTREME_RATIO as THUMBNAIL_EXTREME_RATIO,
    THUMBNAIL_SIZE as THUMBNAIL_SIZE,
    _ascii_download_filename,
)
from .web_models import BatchJob, EditorConfig, OperationCancelled
from .web_jobs import EditorJobsMixin
from .web_processing import EditorProcessingMixin
from .web_workspace import EditorWorkspaceMixin
from .web_settings import (
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TEMPLATE_PATHS,
    DEFAULT_WEB_TEMPLATE_MAX_SCALE,
    DEFAULT_WEB_TEMPLATE_MIN_SCALE,
    DEFAULT_WEB_TEMPLATE_SCORE_THRESHOLD,
    DETECTOR_MODES as DETECTOR_MODES,
    TEMP_INPUT_ROOT_NAME as TEMP_INPUT_ROOT_NAME,
    _ai_settings_payload,
    _bool_setting,
    _new_temporary_input_path,
    _normalize_batch_jobs,
    _template_settings_payload,
)


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
STATIC_DIR = Path(__file__).resolve().parent / "web_static"
ADVANCED_SETTINGS_FILE_NAME = "advanced_settings.json"
TEMPLATE_MASK_PREVIEW_SETTING_KEYS = {
    "maskDilateIterations",
    "maskDilateMaxBodyRatio",
    "maskEdgeFeatherPixels",
    "maskUnifyBody",
    "maskContourClosePixels",
    "maskBodyGapRatio",
}


class EditorSession(EditorProcessingMixin, EditorJobsMixin, EditorWorkspaceMixin):
    def __init__(self, config: EditorConfig) -> None:
        self.config = config
        if self.config.settings_file is None:
            self.config.settings_file = self.config.output_dir / ".editor_state" / ADVANCED_SETTINGS_FILE_NAME
        self.image_paths: list[Path] = []
        self.template_files: list[Path] = []
        self._batch_jobs: dict[str, BatchJob] = {}
        self._cancelled_operations: set[str] = set()
        self._batch_lock = RLock()
        self._batch_condition = Condition(self._batch_lock)
        self._operation_process_lock = RLock()
        self._sam3_detector_lock = RLock()
        self._sam3_detector: Any | None = None
        self._sam3_detector_key: tuple[Any, ...] | None = None
        self._deleted_source_image_names: set[str] = set()
        self._closed = False
        self._activate_input_workspace()
        self.reload()

    def _new_template_remover(self, templates: list[Any], config: WatermarkRemoverConfig) -> Any:
        return WatermarkRemover(templates, config=config)

    def _new_sam3_detector(self, config: AIDetectionConfig) -> Any:
        if self._closed:
            raise OperationCancelled("作業已中斷")
        key = _sam3_detector_cache_key(config)
        if self._sam3_detector is not None and self._sam3_detector_key == key:
            self._sam3_detector.config = config
            return self._sam3_detector

        self._close_sam3_detector_locked()
        self._sam3_detector = _build_ai_detector(config)
        self._sam3_detector_key = key
        return self._sam3_detector

    def _close_sam3_detector_locked(self) -> None:
        detector = self._sam3_detector
        self._sam3_detector = None
        self._sam3_detector_key = None
        close = getattr(detector, "close", None)
        if callable(close):
            close()

    def close(self) -> None:
        with self._sam3_detector_lock:
            self._closed = True
            self._close_sam3_detector_locked()
        with self._batch_condition:
            for job in self._batch_jobs.values():
                if job.status == "running":
                    job.cancel_requested = True
            self._batch_condition.notify_all()

    def _build_detection_mask(
        self,
        image_rgb: np.ndarray,
        detections: list[Detection],
        templates: list[Any],
        config: WatermarkRemoverConfig,
    ) -> np.ndarray:
        return build_detection_mask(image_rgb, detections, templates, config)

    def _restore_with_mask(
        self,
        image_rgb: np.ndarray,
        mask_u8: np.ndarray,
        config: WatermarkRemoverConfig,
    ) -> np.ndarray:
        return restore_with_mask(image_rgb, mask_u8, config)

    def _executor_cls(self) -> type[Any]:
        return ThreadPoolExecutor

    @property
    def state_dir(self) -> Path:
        return self.config.output_dir / ".editor_state"

    @property
    def advanced_settings_path(self) -> Path:
        return self.config.settings_file or (self.state_dir / ADVANCED_SETTINGS_FILE_NAME)

    def reload(self) -> None:
        self._ensure_collection_path(self.config.input_path, "Input")
        for template_path in self.config.template_paths:
            self._ensure_collection_path(template_path, "Template")
        if not self.config.input_path.exists():
            raise ValueError(f"Input path not found: {self.config.input_path}")
        missing_templates = [path for path in self.config.template_paths if not path.exists()]
        if missing_templates:
            raise ValueError(f"Template path not found: {missing_templates[0]}")
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._sync_source_input_to_workspace()
        self.image_paths = list_input_images(self.config.input_path)
        self.template_files = self._template_image_candidates(self.config.template_paths)

    def configure(
        self,
        *,
        input_path: str | None = None,
        template_paths: list[str] | None = None,
        output_dir: str | None = None,
        lama_device: str | None = None,
    ) -> dict[str, Any]:
        next_output_dir = self._path_or_default(output_dir, self.config.output_dir, DEFAULT_OUTPUT_DIR)
        next_input_path, next_input_is_temporary = self._input_path_or_default(input_path, next_output_dir)
        self._archive_active_workspace_input()
        self._deleted_source_image_names.clear()
        self.config = EditorConfig(
            input_path=next_input_path,
            template_paths=self._template_paths_or_default(template_paths),
            output_dir=next_output_dir,
            lama_device=lama_device if lama_device is not None else self.config.lama_device,
            input_is_temporary=next_input_is_temporary,
            source_input_path=None if next_input_is_temporary else next_input_path,
            settings_file=self.config.settings_file,
            batch_detect_jobs=self.config.batch_detect_jobs,
            batch_process_jobs=self.config.batch_process_jobs,
            save_debug=self.config.save_debug,
        )
        self._activate_input_workspace()
        self.reload()
        return self.summary()

    def _normalize_advanced_settings(self, settings: dict[str, Any], *, strict: bool) -> dict[str, Any]:
        if not isinstance(settings, dict):
            if strict:
                raise ValueError("Invalid advanced settings.")
            return {}
        normalized: dict[str, Any] = {}
        for key in ("aiSettings", "templateSettings"):
            if key not in settings or settings[key] is None:
                continue
            if not isinstance(settings[key], dict):
                if strict:
                    raise ValueError(f"Invalid advanced settings: {key}")
                continue
            try:
                if key == "aiSettings":
                    normalized[key] = _ai_settings_payload(self._ai_detection_config(DEFAULT_AI_PROMPT, settings[key]))
                else:
                    normalized[key] = _template_settings_payload(self._template_detection_config(settings[key]))
            except ValueError:
                if strict:
                    raise
        return normalized

    def saved_advanced_settings(self) -> dict[str, Any]:
        path = self.advanced_settings_path
        if not path.is_file():
            return {}
        try:
            parsed = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return self._normalize_advanced_settings(parsed, strict=False)

    def save_advanced_settings(self, settings: dict[str, Any]) -> dict[str, Any]:
        normalized = self._normalize_advanced_settings(settings, strict=True)
        path = self.advanced_settings_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if normalized:
            path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        elif path.exists():
            path.unlink()
        return self.summary()

    def summary(self) -> dict[str, Any]:
        return {
            "input": str(self._summary_input_path()),
            "inputIsTemporary": self.config.input_is_temporary,
            "templates": [self._template_payload(index, path) for index, path in enumerate(self.template_files)],
            "templateRoots": [str(path) for path in self.config.template_paths],
            "output": str(self.config.output_dir),
            "lamaDevice": self.config.lama_device,
            "aiPrompt": DEFAULT_AI_PROMPT,
            "aiSettings": _ai_settings_payload(AIDetectionConfig()),
            "templateSettings": _template_settings_payload(
                WatermarkRemoverConfig(
                    score_threshold=DEFAULT_WEB_TEMPLATE_SCORE_THRESHOLD,
                    min_scale=DEFAULT_WEB_TEMPLATE_MIN_SCALE,
                    max_scale=DEFAULT_WEB_TEMPLATE_MAX_SCALE,
                )
            ),
            "savedAdvancedSettings": self.saved_advanced_settings(),
            "batchDetectJobs": _normalize_batch_jobs(self.config.batch_detect_jobs, len(self.image_paths)),
            "batchProcessJobs": _normalize_batch_jobs(self.config.batch_process_jobs, len(self.image_paths)),
            "images": [self._image_summary(index, path) for index, path in enumerate(self.image_paths)],
        }


def _build_ai_detector(config: AIDetectionConfig):
    return Sam3TextWatermarkDetector(config)


def _sam3_detector_cache_key(config: AIDetectionConfig) -> tuple[Any, ...]:
    return (
        config.sam3_model,
        config.sam3_model_file,
        round(float(config.sam3_confidence_threshold), 6),
        config.device,
    )


def _path_key(path: Path) -> str:
    try:
        return str(path.resolve())
    except OSError:
        return str(path.absolute())


def _raise_if_cancelled(cancel_callback: Any | None) -> None:
    if cancel_callback is not None and cancel_callback():
        raise OperationCancelled("作業已中斷")


def make_handler(session: EditorSession) -> type[BaseHTTPRequestHandler]:
    class EditorRequestHandler(BaseHTTPRequestHandler):
        server_version = "RemoveWatermarkEditor/0.1"

        def do_GET(self) -> None:
            try:
                parsed = urlparse(self.path)
                if parsed.path == "/":
                    self._serve_static("index.html")
                elif parsed.path.startswith("/static/"):
                    self._serve_static(unquote(parsed.path.removeprefix("/static/")))
                elif parsed.path == "/api/state":
                    self._send_json(session.summary())
                elif parsed.path == "/api/batch-progress":
                    query = parse_qs(parsed.query)
                    self._send_json(session.batch_job(query.get("id", [""])[0]))
                elif parsed.path == "/api/image":
                    query = parse_qs(parsed.query)
                    index = int(query.get("index", ["0"])[0])
                    kind = query.get("kind", ["original"])[0]
                    body, content_type = session.image_bytes(index, kind)
                    cache_control = "private, max-age=31536000, immutable" if query.get("v", [""])[0] else "no-store"
                    self._send_bytes(body, content_type, cache_control=cache_control)
                elif parsed.path == "/api/download-image":
                    query = parse_qs(parsed.query)
                    index = int(query.get("index", ["0"])[0])
                    body, content_type, filename = session.result_download(index)
                    self._send_download(body, content_type, filename)
                elif parsed.path == "/api/download-all-images":
                    body, content_type, filename = session.results_zip_download()
                    self._send_download(body, content_type, filename)
                elif parsed.path == "/api/template":
                    query = parse_qs(parsed.query)
                    index = int(query.get("index", ["0"])[0])
                    kind = query.get("kind", ["thumbnail"])[0]
                    template_settings = {
                        key: query[key][0]
                        for key in TEMPLATE_MASK_PREVIEW_SETTING_KEYS
                        if key in query and query[key]
                    } if kind in {"mask-preview", "mask-preview-full"} else None
                    body, content_type = session.template_bytes(index, kind, template_settings)
                    self._send_bytes(body, content_type)
                else:
                    self._send_json({"error": "Not found."}, HTTPStatus.NOT_FOUND)
            except Exception as exc:
                self._send_error(exc)

        def do_POST(self) -> None:
            try:
                payload = self._read_json()
                parsed = urlparse(self.path)
                if parsed.path == "/api/workspace":
                    self._send_json(
                        session.configure(
                            input_path=payload.get("input"),
                            template_paths=payload.get("templates"),
                            output_dir=payload.get("output"),
                            lama_device=payload.get("lamaDevice"),
                        )
                    )
                elif parsed.path == "/api/settings":
                    self._send_json(session.save_advanced_settings(payload))
                elif parsed.path == "/api/detect":
                    self._send_json(
                        session.detect(
                            int(payload["index"]),
                            payload.get("templates"),
                            detector=str(payload.get("detector", "template")),
                            ai_prompt=payload.get("aiPrompt"),
                            ai_settings=payload.get("aiSettings"),
                            template_settings=payload.get("templateSettings"),
                            operation_id=str(payload.get("operationId") or ""),
                        )
                    )
                elif parsed.path == "/api/batch-detect":
                    self._send_json(
                        session.detect_all(
                            payload.get("templates"),
                            detector=str(payload.get("detector", "template")),
                            ai_prompt=payload.get("aiPrompt"),
                            ai_settings=payload.get("aiSettings"),
                            template_settings=payload.get("templateSettings"),
                        )
                    )
                elif parsed.path == "/api/start-batch-detect":
                    self._send_json(
                        session.start_detect_all(
                            payload.get("templates"),
                            detector=str(payload.get("detector", "template")),
                            ai_prompt=payload.get("aiPrompt"),
                            ai_settings=payload.get("aiSettings"),
                            template_settings=payload.get("templateSettings"),
                        )
                    )
                elif parsed.path == "/api/start-operation-batch":
                    self._send_json(session.start_operation_batch(str(payload.get("mode", "")), payload))
                elif parsed.path == "/api/add-operation-batch":
                    self._send_json(
                        session.add_operation_batch(
                            str(payload.get("jobId", "")),
                            str(payload.get("mode", "")),
                            payload,
                        )
                    )
                elif parsed.path == "/api/save-mask":
                    self._send_json(session.save_mask(int(payload["index"]), payload["maskData"], payload.get("detections")))
                elif parsed.path == "/api/process":
                    self._send_json(
                        session.process_mask(
                            int(payload["index"]),
                            payload.get("maskData"),
                            keep_detections=_bool_setting(payload.get("keepDetectionsAfterProcess")),
                            operation_id=str(payload.get("operationId") or ""),
                        )
                    )
                elif parsed.path == "/api/batch-process":
                    self._send_json(session.process_all(keep_detections=_bool_setting(payload.get("keepDetectionsAfterProcess"))))
                elif parsed.path == "/api/start-batch-process":
                    self._send_json(session.start_process_all(keep_detections=_bool_setting(payload.get("keepDetectionsAfterProcess"))))
                elif parsed.path == "/api/cancel-processing":
                    self._send_json(
                        session.cancel_processing(
                            operation_id=str(payload.get("operationId") or ""),
                            job_id=str(payload.get("jobId") or ""),
                        )
                    )
                elif parsed.path == "/api/save":
                    self._send_json(session.save_result(int(payload["index"]), payload.get("resultData"), payload.get("maskData")))
                elif parsed.path == "/api/restore-original":
                    self._send_json(session.restore_original(int(payload["index"]), payload["brushMaskData"], payload.get("resultData")))
                elif parsed.path == "/api/reset-image":
                    self._send_json(session.reset_image(int(payload["index"])))
                elif parsed.path == "/api/reset-all-images":
                    self._send_json(session.reset_all_images())
                elif parsed.path == "/api/create-template":
                    self._send_json(session.create_template(int(payload["index"]), payload["maskData"], payload.get("name")))
                elif parsed.path == "/api/add-images":
                    self._send_json(session.add_images(payload.get("files", [])))
                elif parsed.path == "/api/add-templates":
                    self._send_json(session.add_templates(payload.get("files", [])))
                elif parsed.path == "/api/delete-template":
                    self._send_json(session.delete_template(str(payload.get("template", ""))))
                elif parsed.path == "/api/delete-image":
                    self._send_json(session.delete_image(int(payload["index"])))
                elif parsed.path == "/api/delete-all-images":
                    self._send_json(session.delete_all_images())
                elif parsed.path == "/api/clear-all-masks":
                    self._send_json(session.clear_all_masks())
                elif parsed.path == "/api/clear-all-detections":
                    self._send_json(session.clear_all_detections())
                else:
                    self._send_json({"error": "Not found."}, HTTPStatus.NOT_FOUND)
            except Exception as exc:
                self._send_error(exc)

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _serve_static(self, relative_path: str) -> None:
            path = (STATIC_DIR / relative_path).resolve()
            if STATIC_DIR.resolve() not in path.parents and path != STATIC_DIR.resolve():
                self._send_json({"error": "Invalid static path."}, HTTPStatus.BAD_REQUEST)
                return
            if not path.is_file():
                self._send_json({"error": "Static file not found."}, HTTPStatus.NOT_FOUND)
                return
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            self._send_bytes(path.read_bytes(), content_type)

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                return {}
            return json.loads(self.rfile.read(length).decode("utf-8"))

        def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(int(status))
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_bytes(self, body: bytes, content_type: str, *, cache_control: str = "no-store") -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", cache_control)
            self.end_headers()
            self.wfile.write(body)

        def _send_download(self, body: bytes, content_type: str, filename: str) -> None:
            fallback = _ascii_download_filename(filename)
            encoded = quote(filename, safe="")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Disposition", f"attachment; filename=\"{fallback}\"; filename*=UTF-8''{encoded}")
            self.end_headers()
            self.wfile.write(body)

        def _send_error(self, exc: Exception) -> None:
            status = HTTPStatus.BAD_REQUEST if isinstance(exc, (ValueError, KeyError, json.JSONDecodeError)) else HTTPStatus.INTERNAL_SERVER_ERROR
            self._send_json({"error": str(exc)}, status)

    return EditorRequestHandler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Start the local Remove Watermark editor UI.")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"Host to bind. Defaults to {DEFAULT_HOST}.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Port to bind. Defaults to {DEFAULT_PORT}.")
    parser.add_argument("--input", help="Input image file or directory. Omit to start with an empty image list.")
    parser.add_argument("--template", action="append", help="Template image file or directory. Repeat to use multiple templates. Defaults to templates/.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR), help="Output directory for edited images. Defaults to output/.")
    parser.add_argument(
        "--settings-file",
        help="JSON file for saved Web UI advanced settings. Defaults to output/.editor_state/advanced_settings.json.",
    )
    parser.add_argument("--lama-device", default="auto", help="Torch device for LaMa, for example auto, cuda, or cpu. Defaults to auto.")
    parser.add_argument("--batch-detect-jobs", type=int, default=0, help="Number of UI batch detection jobs. Defaults to CPU count.")
    parser.add_argument("--batch-process-jobs", type=int, default=1, help="Number of UI batch LaMa removal jobs. Defaults to 1.")
    parser.add_argument("--save-debug", action="store_true", help="Save UI debug artifacts under the output debug-ui directory.")
    parser.add_argument("--no-open-browser", action="store_true", help="Do not automatically open the browser after the UI starts.")
    return parser


def _browser_url(host: str, port: int) -> str:
    browser_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    return f"http://{browser_host}:{port}"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.batch_detect_jobs < 0 or args.batch_process_jobs < 1:
        raise SystemExit("Invalid UI batch jobs configuration.")
    output_dir = Path(args.output)
    input_value = args.input.strip() if args.input else ""
    input_is_temporary = not input_value
    session = EditorSession(
        EditorConfig(
            input_path=Path(input_value) if input_value else _new_temporary_input_path(output_dir),
            template_paths=[Path(value) for value in (args.template or [str(DEFAULT_TEMPLATE_PATHS[0])])],
            output_dir=output_dir,
            settings_file=Path(args.settings_file) if args.settings_file else None,
            lama_device=args.lama_device,
            input_is_temporary=input_is_temporary,
            batch_detect_jobs=args.batch_detect_jobs,
            batch_process_jobs=args.batch_process_jobs,
            save_debug=args.save_debug,
        )
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(session))
    url = f"http://{args.host}:{args.port}"
    print(f"Remove Watermark editor running at {url}", flush=True)
    if not args.no_open_browser:
        Thread(target=webbrowser.open, args=(_browser_url(args.host, args.port),), daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        session.close()
        server.server_close()
    return 0


def run() -> None:
    raise SystemExit(main())


if __name__ == "__main__":
    run()
