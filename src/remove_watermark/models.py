from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
LAMA_MODE = "lama"
EPSILON = 1e-6
LamaCallable = Callable[[np.ndarray, np.ndarray], np.ndarray]
CandidateSource = Literal["match", "faint_presence", "chromatic_presence"]
CANDIDATE_SOURCE_MATCH: CandidateSource = "match"
CANDIDATE_SOURCE_FAINT_PRESENCE: CandidateSource = "faint_presence"
CANDIDATE_SOURCE_CHROMATIC_PRESENCE: CandidateSource = "chromatic_presence"
LOCAL_MATCH_EDGE_WEIGHT = 0.30
LOCAL_MATCH_SCORE_THRESHOLD = 0.18
LOCAL_MATCH_EDGE_THRESHOLD = 0.26


@dataclass(slots=True)
class FitResult:
    strength: float = 1.0
    objective: float = 0.0
    clip_ratio: float = 0.0
    residual: float = 0.0
    watermark_correlation: float = 0.0


@dataclass(slots=True)
class Candidate:
    template_name: str
    score: float
    gray_score: float
    edge_score: float
    x: int
    y: int
    width: int
    height: int
    scale: float
    source: CandidateSource = CANDIDATE_SOURCE_MATCH

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.width, self.height


@dataclass(slots=True)
class CandidatePrefit:
    passed: bool
    score: float = 0.0
    reason: str = ""


@dataclass(slots=True)
class CandidateEvidence:
    patch_gray: np.ndarray
    support_corr: float
    support_mean: float
    text_detail: float = 0.0


@dataclass(slots=True)
class Detection:
    template_name: str
    bbox: tuple[int, int, int, int]
    scale: float
    score: float
    color_score: float
    strength: float
    method: str
    objective: float
    clip_ratio: float
    residual: float
    watermark_correlation: float
    content_bbox: tuple[int, int, int, int] | None = None
    residual_score: float = 0.0
    text_detail: float = 0.0
    lama_mask_ratio: float = 0.0
    stage_metrics: dict[str, object] = field(default_factory=dict)
    debug_maps: dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    @property
    def x(self) -> int:
        return int(self.bbox[0])

    @property
    def y(self) -> int:
        return int(self.bbox[1])

    @property
    def width(self) -> int:
        return int(self.bbox[2])

    @property
    def height(self) -> int:
        return int(self.bbox[3])


@dataclass(slots=True)
class RestorationDiagnostics:
    residual_score: float = 0.0
    text_detail: float = 0.0
    lama_mask_ratio: float = 0.0
    stage_metrics: dict[str, object] = field(default_factory=dict)
    debug_maps: dict[str, np.ndarray] = field(default_factory=dict, repr=False)


@dataclass(slots=True)
class RestorationResult:
    patch: np.ndarray
    method: str
    diagnostics: RestorationDiagnostics


@dataclass(slots=True)
class PatchContext:
    image: np.ndarray
    inner_bbox: tuple[int, int, int, int]


@dataclass(slots=True)
class BodySourceResult:
    mask: np.ndarray
    mode: str
    hole_ratio: float = 0.0
    support_fill: float = 1.0
    growth: float = 1.0


class BackendUnavailableError(RuntimeError):
    pass


@dataclass(slots=True)
class WatermarkRemoverConfig:
    min_scale: float = 0.35
    max_scale: float = 1.20
    scale_step: float = 0.05
    score_threshold: float = 0.50
    candidate_margin: float = 0.16
    candidate_limit: int = 240
    max_detections: int = 96
    nms_iou_threshold: float = 0.30
    bbox_padding_ratio: float = 0.02
    bbox_edge_snap_ratio: float = 0.03
    polarity_flexible_matching: bool = True
    polarity_flexible_min_template_area: int = 60000
    large_template_area_threshold: int = 60000
    large_template_text_detail_threshold: float = 0.70
    small_template_area_threshold: int = 60000
    small_template_evidence_delta_threshold: float = 15.0
    color_match_evidence_delta_threshold: float = 15.0
    sparse_chromatic_support_ratio: float = 0.20
    sparse_chromatic_chroma_threshold: float = 10.0
    sparse_chromatic_min_shape_correlation: float = 0.45
    sparse_chromatic_strong_shape_correlation: float = 0.75
    sparse_chromatic_max_overstrength_delta: float = 55.0
    sparse_chromatic_color_rescue_threshold: float = 0.92
    sparse_chromatic_evidence_rescue_delta: float = 45.0
    sparse_chromatic_chroma_delta_threshold: float = 0.08
    sparse_chromatic_edge_candidate_threshold_ratio: float = 0.95
    sparse_chromatic_min_best_score: float = 0.70
    sparse_chromatic_relative_score: float = 0.65
    chromatic_text_presence_threshold: float = 0.58
    chromatic_text_min_support_ratio: float = 0.25
    chromatic_text_min_chroma: float = 8.0
    faint_presence_max_chroma: float = 2.0
    faint_presence_min_support_ratio: float = 0.06
    faint_presence_score_threshold: float = 0.30
    solid_dark_min_component_area_ratio: float = 0.20
    solid_dark_support_luma_threshold: float = 96.0
    solid_dark_fill_threshold: float = 0.72
    solid_dark_hole_luma_threshold: float = 145.0
    solid_dark_hole_bright_threshold: float = 0.55
    edge_score_threshold: float = 0.30
    support_correlation_threshold: float = 0.12
    color_score_threshold: float = 0.55
    tiny_scale_threshold: float = 0.45
    small_scale_threshold: float = 0.55
    foreground_shape_min_coverage: float = 0.18
    foreground_shape_max_leakage: float = 2.00
    foreground_edge_orientation_min_score: float = 0.45
    foreground_edge_min_coverage: float = 0.08
    foreground_strong_shape_correlation: float = 0.34
    foreground_residual_delta_threshold: float = 8.0
    profile_acceptance_thresholds: dict[str, float] = field(default_factory=dict)
    profile_fit_thresholds: dict[str, float] = field(default_factory=dict)
    strength_min: float = 0.35
    strength_max: float = 1.45
    strength_step: float = 0.05
    min_context_margin: int = 32
    context_margin_scale: float = 1.25
    mask_dilate_iterations: int = 10
    mask_dilate_max_body_ratio: float = 0.10
    mask_antialias_scale: int = 4
    mask_edge_feather_pixels: float = 2.0
    mask_unify_body: bool = True
    mask_body_gap_ratio: float = 0.05
    mask_body_max_area_growth: float = 1.85
    mask_body_visible_threshold: float = 16.0
    mask_body_faint_threshold_ratio: float = 0.35
    mask_body_source_max_area_growth: float = 2.50
    mask_body_silhouette_min_hole_ratio: float = 0.03
    mask_body_silhouette_min_support_fill: float = 0.55
    mask_contour_close_pixels: float = 3.0
    sam3_refine_mask: bool = False
    protect_edge_dilate: int = 1
    protect_contrast_threshold: float = 15.0
    protect_luma_dark: float = 96.0
    protect_luma_light: float = 232.0
    lama_device: str | None = None
    lama_backend: LamaCallable | None = field(default=None, repr=False, compare=False)
    collect_debug_maps: bool = True

    def scales(self) -> list[float]:
        values = np.arange(self.min_scale, self.max_scale + self.scale_step * 0.5, self.scale_step)
        return [float(scale) for scale in values]

    @property
    def candidate_threshold(self) -> float:
        return max(0.18, self.score_threshold - self.candidate_margin)

def _pad_bbox(
    bbox: tuple[int, int, int, int],
    image_shape: tuple[int, int],
    config: WatermarkRemoverConfig,
) -> tuple[int, int, int, int]:
    x, y, width, height = bbox
    image_height, image_width = int(image_shape[0]), int(image_shape[1])
    safe_ratio = max(0.0, config.bbox_padding_ratio)
    pad_x = int(round(width * safe_ratio))
    pad_y = int(round(height * safe_ratio))
    left = max(0, x - pad_x)
    top = max(0, y - pad_y)
    right = min(image_width, x + width + pad_x)
    bottom = min(image_height, y + height + pad_y)
    edge_snap_ratio = max(0.0, config.bbox_edge_snap_ratio)
    snap_x = int(round(width * edge_snap_ratio))
    snap_y = int(round(height * edge_snap_ratio))
    if left <= snap_x:
        left = 0
    if top <= snap_y:
        top = 0
    if image_width - right <= snap_x:
        right = image_width
    if image_height - bottom <= snap_y:
        bottom = image_height
    return left, top, max(1, right - left), max(1, bottom - top)


def bbox_iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    inter_left, inter_top = max(ax, bx), max(ay, by)
    inter_right, inter_bottom = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, inter_right - inter_left) * max(0, inter_bottom - inter_top)
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union else 0.0


def _is_center_contained(inner: tuple[int, int, int, int], outer: tuple[int, int, int, int]) -> bool:
    ix, iy, iw, ih = inner
    ox, oy, ow, oh = outer
    cx, cy = ix + iw / 2.0, iy + ih / 2.0
    return bool(ox <= cx <= ox + ow and oy <= cy <= oy + oh and iw * ih < ow * oh)

