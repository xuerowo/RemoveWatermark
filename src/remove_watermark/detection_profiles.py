from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .evidence_metrics import _template_chroma_strength, _template_has_solid_dark_region
from .models import EPSILON
from .templates import ResizedTemplate, TemplateBundle, _rgb_to_gray


@dataclass(slots=True)
class TemplateProfile:
    name: str
    width: int
    height: int
    support_ratio: float
    polarity: str
    chroma_strength: float
    alpha_coverage: float
    component_density: float
    foreground_bbox: tuple[int, int, int, int]
    is_faint: bool
    is_sparse: bool
    is_chromatic: bool
    is_sparse_chromatic: bool
    is_chromatic_text: bool
    is_solid_dark: bool


class DetectionProfilesMixin:
    def _scales_for_template(self, template: TemplateBundle) -> list[float]:
        if not self._template_profile(template).is_solid_dark:
            return self.config.scales()

        step = max(self.config.scale_step * 0.5, EPSILON)
        values = np.arange(self.config.min_scale, self.config.max_scale + step * 0.5, step)
        return [float(scale) for scale in values]

    def _template_profile(self, template: TemplateBundle) -> TemplateProfile:
        key = (
            id(template),
            float(self.config.faint_presence_max_chroma),
            float(self.config.faint_presence_min_support_ratio),
            float(self.config.sparse_chromatic_support_ratio),
            float(self.config.sparse_chromatic_chroma_threshold),
            float(self.config.chromatic_text_min_chroma),
            float(self.config.chromatic_text_min_support_ratio),
            float(self.config.solid_dark_min_component_area_ratio),
        )
        with self._cache_lock:
            cached = self._template_profile_cache.get(key)
            if cached is None:
                cached = self._build_template_profile(template)
                self._template_profile_cache[key] = cached
            return cached

    def _build_template_profile(self, template: TemplateBundle) -> TemplateProfile:
        width, height = template.size
        template_area = max(width * height, 1)
        support = template.support_mask.astype(bool)
        support_pixels = int(support.sum())
        support_ratio = float(support_pixels / template_area)
        alpha = getattr(template, "soft_alpha", np.zeros_like(template.gray, dtype=np.float32))
        alpha_coverage = float((alpha > 0.20).mean()) if alpha.size else 0.0
        component_density = 0.0
        foreground_bbox = (0, 0, width, height)
        if support_pixels > 0:
            count, _, stats, _ = cv2.connectedComponentsWithStats(support.astype(np.uint8), connectivity=8)
            components = max(count - 1, 0)
            component_density = float(components / max(support_pixels, 1))
            ys, xs = np.where(support)
            left = int(xs.min())
            top = int(ys.min())
            right = int(xs.max()) + 1
            bottom = int(ys.max()) + 1
            foreground_bbox = (left, top, max(1, right - left), max(1, bottom - top))

        chroma_strength = _template_chroma_strength(template)
        solid_dark = self._template_has_solid_dark_region(template)
        chromatic_directions = self._template_chromatic_foreground_directions(template)
        is_chromatic = bool(chroma_strength >= self.config.faint_presence_max_chroma or chromatic_directions.size > 0)
        is_sparse = support_ratio <= self.config.sparse_chromatic_support_ratio
        is_faint = bool(chroma_strength <= self.config.faint_presence_max_chroma and support_ratio >= self.config.faint_presence_min_support_ratio)
        is_sparse_chromatic = bool(is_sparse and chromatic_directions.size > 0)
        is_chromatic_text = bool(
            chroma_strength >= self.config.chromatic_text_min_chroma
            and support_ratio >= self.config.chromatic_text_min_support_ratio
            and not solid_dark
            and not (is_sparse and chroma_strength >= self.config.sparse_chromatic_chroma_threshold)
        )
        return TemplateProfile(
            name=template.name,
            width=width,
            height=height,
            support_ratio=support_ratio,
            polarity=template.polarity,
            chroma_strength=float(chroma_strength),
            alpha_coverage=alpha_coverage,
            component_density=component_density,
            foreground_bbox=foreground_bbox,
            is_faint=is_faint,
            is_sparse=is_sparse,
            is_chromatic=is_chromatic,
            is_sparse_chromatic=is_sparse_chromatic,
            is_chromatic_text=is_chromatic_text,
            is_solid_dark=solid_dark,
        )

    def _template_has_solid_dark_region(self, template: TemplateBundle) -> bool:
        key = (id(template), float(self.config.solid_dark_min_component_area_ratio))
        with self._cache_lock:
            cached = self._solid_dark_template_cache.get(key)
            if cached is None:
                cached = _template_has_solid_dark_region(template, self.config.solid_dark_min_component_area_ratio)
                self._solid_dark_template_cache[key] = cached
            return cached

    def _template_uses_faint_presence_matching(self, template: TemplateBundle) -> bool:
        return self._template_profile(template).is_faint

    def _template_uses_sparse_achromatic_presence(self, template: TemplateBundle) -> bool:
        profile = self._template_profile(template)
        return bool(profile.is_faint and profile.is_sparse and not profile.is_chromatic)

    def _template_uses_sparse_chromatic_presence(self, template: TemplateBundle) -> bool:
        return self._template_profile(template).is_sparse_chromatic

    def _template_uses_chromatic_text_presence(self, template: TemplateBundle) -> bool:
        return self._template_profile(template).is_chromatic_text

    @staticmethod
    def _chromatic_text_presence_image_mask(image_rgb: np.ndarray) -> np.ndarray:
        image = image_rgb.astype(np.float32)
        chroma = image.max(axis=2) - image.min(axis=2)
        gray = _rgb_to_gray(np.clip(image, 0, 255).astype(np.uint8)).astype(np.float32)
        local = cv2.medianBlur(np.clip(gray, 0, 255).astype(np.uint8), 31).astype(np.float32)
        contrast = np.abs(gray - local)
        return ((chroma > 35.0) & (contrast > 5.0)).astype(np.float32)

    @staticmethod
    def _chromatic_text_presence_template_mask(resized: ResizedTemplate) -> np.ndarray:
        template = resized.rgb.astype(np.float32)
        chroma = template.max(axis=2) - template.min(axis=2)
        return ((resized.support_mask > 0) & (chroma > 25.0)).astype(np.float32)

    def _sparse_chromatic_presence_image_mask(self, image_rgb: np.ndarray, template: TemplateBundle) -> np.ndarray:
        directions = self._template_chromatic_foreground_directions(template)
        if directions.size == 0:
            return np.zeros(image_rgb.shape[:2], dtype=np.float32)

        image = image_rgb.astype(np.float32)
        chroma = image.max(axis=2) - image.min(axis=2)
        gray = _rgb_to_gray(np.clip(image, 0, 255).astype(np.uint8)).astype(np.float32)
        local = cv2.medianBlur(np.clip(gray, 0, 255).astype(np.uint8), 31).astype(np.float32)
        search_mask = (chroma > 24.0) & (np.abs(gray - local) > 2.0)
        if int(search_mask.sum()) < 64:
            return np.zeros(search_mask.shape, dtype=np.float32)

        ys, xs = np.where(search_mask)
        image_directions = self._chromatic_color_directions(image[ys, xs])
        similarity = image_directions @ directions.T
        mask = np.zeros(search_mask.shape, dtype=np.float32)
        mask[ys, xs] = (similarity.max(axis=1) >= 0.80).astype(np.float32)
        return mask

    @staticmethod
    def _chromatic_color_directions(rgb: np.ndarray) -> np.ndarray:
        centered = rgb.astype(np.float32) - rgb.astype(np.float32).mean(axis=-1, keepdims=True)
        norm = np.linalg.norm(centered, axis=-1, keepdims=True)
        return centered / np.maximum(norm, EPSILON)

    def _template_chromatic_foreground_directions(self, template: TemplateBundle) -> np.ndarray:
        support = template.support_mask.astype(bool)
        template_rgb = template.template_rgb.astype(np.float32)
        chroma = template_rgb.max(axis=2) - template_rgb.min(axis=2)
        foreground = support & (chroma > 25.0)
        if int(foreground.sum()) < 16:
            return np.empty((0, 3), dtype=np.float32)

        colors = template_rgb[foreground]
        directions = self._chromatic_color_directions(colors)
        chroma_values = chroma[foreground]
        order = np.argsort(-chroma_values)[:4096]
        representatives: list[np.ndarray] = []
        for index in order.tolist():
            direction = directions[index]
            if not np.all(np.isfinite(direction)):
                continue
            if all(float(np.dot(direction, existing)) < 0.92 for existing in representatives):
                representatives.append(direction.astype(np.float32))
            if len(representatives) >= 8:
                break
        if not representatives:
            return np.empty((0, 3), dtype=np.float32)
        return np.stack(representatives).astype(np.float32)
