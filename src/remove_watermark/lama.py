from __future__ import annotations

import sys
from threading import RLock

import numpy as np
from PIL import Image

from .debug_render import _normalize_debug_mask
from .models import BackendUnavailableError, LamaCallable, WatermarkRemoverConfig




_LAMA_CACHE: dict[str, LamaCallable] = {}
_LAMA_CACHE_LOCK = RLock()


def _run_lama(image_rgb: np.ndarray, mask: np.ndarray, config: WatermarkRemoverConfig) -> np.ndarray:
    image_u8 = np.clip(image_rgb, 0.0, 255.0).astype(np.uint8)
    mask_u8 = (_normalize_debug_mask(mask, image_u8.shape[:2]) * 255.0).round().astype(np.uint8)
    if config.lama_backend is not None:
        return np.asarray(config.lama_backend(image_u8, mask_u8), dtype=np.uint8)

    key = config.lama_device or "auto"
    with _LAMA_CACHE_LOCK:
        runner = _LAMA_CACHE.get(key)
        if runner is None:
            core_module = sys.modules.get("remove_watermark.core")
            build_runner = getattr(core_module, "build_simple_lama_runner", build_simple_lama_runner)
            runner = build_runner(config.lama_device)
            _LAMA_CACHE[key] = runner
        return np.asarray(runner(image_u8, mask_u8), dtype=np.uint8)


def build_simple_lama_runner(device_name: str | None) -> LamaCallable:
    try:
        import torch
        from simple_lama_inpainting import SimpleLama
    except Exception as exc:  # pragma: no cover - depends on optional runtime
        raise BackendUnavailableError("LaMa backend unavailable: install simple-lama-inpainting.") from exc

    requested_device = (device_name or "auto").strip()
    auto_device = requested_device.lower() == "auto"

    def build_model(device_label: str):
        device = torch.device(device_label)
        return SimpleLama(device=device)

    def build_or_raise(device_label: str, *, fallback_error: Exception | None = None):
        try:
            return build_model(device_label)
        except Exception as exc:  # pragma: no cover - depends on local model/runtime
            if fallback_error is not None:
                raise BackendUnavailableError(
                    f"LaMa backend unavailable: CUDA failed ({fallback_error}); CPU fallback failed ({exc})"
                ) from exc
            raise BackendUnavailableError(f"LaMa backend unavailable: {exc}") from exc

    try:
        if auto_device:
            if torch.cuda.is_available():
                try:
                    active_device = "cuda"
                    model = build_model(active_device)
                except Exception as exc:  # pragma: no cover - depends on local model/runtime
                    active_device = "cpu"
                    model = build_or_raise(active_device, fallback_error=exc)
            else:
                active_device = "cpu"
                model = build_or_raise(active_device)
        elif requested_device.lower().startswith("cuda") and not torch.cuda.is_available():
            raise BackendUnavailableError("LaMa CUDA requested but CUDA is not available. Use --lama-device cpu or --lama-device auto.")
        else:
            active_device = requested_device
            model = build_or_raise(active_device)
    except BackendUnavailableError:
        raise
    except Exception as exc:  # pragma: no cover - depends on local model/runtime
        raise BackendUnavailableError(f"LaMa backend unavailable: {exc}") from exc

    def runner(image_rgb: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
        nonlocal active_device, model
        image = Image.fromarray(image_rgb.astype(np.uint8), mode="RGB")
        mask = Image.fromarray(mask_u8.astype(np.uint8), mode="L")
        try:
            result = model(image, mask)
        except Exception as exc:
            if auto_device and active_device.lower().startswith("cuda"):
                active_device = "cpu"
                model = build_or_raise(active_device, fallback_error=exc)
                try:
                    result = model(image, mask)
                except Exception as cpu_exc:
                    raise BackendUnavailableError(
                        f"LaMa backend unavailable: CUDA failed ({exc}); CPU fallback failed ({cpu_exc})"
                    ) from cpu_exc
            else:
                raise BackendUnavailableError(f"LaMa backend unavailable: {exc}") from exc
        return np.asarray(result.convert("RGB"), dtype=np.uint8)

    return runner

