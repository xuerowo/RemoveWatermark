from pathlib import Path


_SRC_PACKAGE = Path(__file__).resolve().parent.parent / "src" / "remove_watermark"
__path__ = [str(_SRC_PACKAGE)]

from .core import (  # noqa: E402
    Detection,
    TemplateBundle,
    WatermarkRemover,
    WatermarkRemoverConfig,
    build_detection_mask,
    load_template,
    restore_original_regions,
    restore_with_mask,
)

__all__ = [
    "Detection",
    "TemplateBundle",
    "WatermarkRemover",
    "WatermarkRemoverConfig",
    "build_detection_mask",
    "load_template",
    "restore_original_regions",
    "restore_with_mask",
]
