from .core import (
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
