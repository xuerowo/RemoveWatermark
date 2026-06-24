from __future__ import annotations

from typing import NamedTuple, Sequence

import cv2
import numpy as np

from .evidence_metrics import (
    _local_maxima_mask,
    _median_kernel_for,
    _needs_sparse_chromatic_confidence_gate,
    _ranking_score,
)
from .models import (
    CANDIDATE_SOURCE_CHROMATIC_PRESENCE,
    CANDIDATE_SOURCE_FAINT_PRESENCE,
    LOCAL_MATCH_EDGE_WEIGHT,
    LOCAL_MATCH_SCORE_THRESHOLD,
    Candidate,
    Detection,
    FitResult,
    _is_center_contained,
    _pad_bbox,
    bbox_iou,
)
from .templates import ResizedTemplate, TemplateBundle


class PresenceContext(NamedTuple):
    image: np.ndarray
    chroma: np.ndarray
    gray: np.ndarray
    local: np.ndarray


class DetectionCandidatesMixin:
    def _collect_evidence_map_candidates(
        self,
        image_rgb: np.ndarray,
        image_gray: np.ndarray,
        edge_map: np.ndarray,
    ) -> list[Candidate]:
        candidates = self._collect_candidates(image_gray, edge_map)
        sparse_chromatic_templates = [template for template in self.templates if self._template_uses_sparse_chromatic_presence(template)]
        chromatic_text_templates = [template for template in self.templates if self._template_uses_chromatic_text_presence(template)]
        needs_presence_context = bool(sparse_chromatic_templates or chromatic_text_templates)
        presence_context = self._build_presence_context(image_rgb, image_gray) if needs_presence_context else None
        candidates.extend(self._collect_sparse_chromatic_presence_candidates(image_rgb, sparse_chromatic_templates, presence_context))
        candidates.extend(self._collect_chromatic_text_presence_candidates(image_rgb, chromatic_text_templates, presence_context))
        return sorted(candidates, key=lambda item: _ranking_score(item.score, item.scale), reverse=True)

    @staticmethod
    def _build_presence_context(image_rgb: np.ndarray, image_gray: np.ndarray) -> PresenceContext:
        image = image_rgb.astype(np.float32)
        chroma = image.max(axis=2) - image.min(axis=2)
        gray = image_gray.astype(np.float32, copy=False)
        local = cv2.medianBlur(np.clip(gray, 0, 255).astype(np.uint8), 31).astype(np.float32)
        return PresenceContext(image=image, chroma=chroma, gray=gray, local=local)

    def _collect_candidates(self, image_gray: np.ndarray, edge_map: np.ndarray) -> list[Candidate]:
        collected: list[Candidate] = []
        image_gray_f32 = image_gray.astype(np.float32, copy=False)
        image_gray_u8 = np.clip(image_gray_f32, 0.0, 255.0).astype(np.uint8)
        inverted_gray = (255.0 - image_gray_f32).astype(np.float32, copy=False)
        dark_fill_mask = (image_gray_f32 < self.config.solid_dark_support_luma_threshold).astype(np.float32)
        local_gray_cache: dict[int, np.ndarray] = {}
        local_response_cache: dict[tuple[int, str], np.ndarray] = {}
        for template in self.templates:
            template_candidates: list[Candidate] = []
            sparse_chromatic_gate = _needs_sparse_chromatic_confidence_gate(template, self.config)
            solid_dark_template = self._template_has_solid_dark_region(template)
            local_match_template = self._template_uses_local_match_map(template)
            gray_source = inverted_gray if template.polarity == "dark" else image_gray_f32
            alternate_source = image_gray_f32 if template.polarity == "dark" else inverted_gray
            flexible_matching = (
                self.config.polarity_flexible_matching
                and template.size[0] * template.size[1] >= self.config.polarity_flexible_min_template_area
            )
            generic_scales = self.config.scales() if solid_dark_template else self._scales_for_template(template)
            for scale in generic_scales:
                scaled_width, scaled_height = template.size_for_scale(scale)
                if scaled_width >= image_gray.shape[1] or scaled_height >= image_gray.shape[0]:
                    continue
                resized = template.resized_to(scaled_width, scaled_height)
                score_gray = cv2.matchTemplate(gray_source, resized.gray, cv2.TM_CCOEFF_NORMED)
                if flexible_matching:
                    alternate_gray = cv2.matchTemplate(alternate_source, resized.gray, cv2.TM_CCOEFF_NORMED)
                    score_gray = np.maximum(score_gray, alternate_gray)
                score_edge = cv2.matchTemplate(edge_map, resized.edge, cv2.TM_CCOEFF_NORMED)
                if local_match_template:
                    median_kernel = _median_kernel_for(resized, 31)
                    local_gray = local_gray_cache.get(median_kernel)
                    if local_gray is None:
                        local_gray = cv2.medianBlur(image_gray_u8, median_kernel).astype(np.float32)
                        local_gray_cache[median_kernel] = local_gray
                    local_response_key = (median_kernel, template.polarity)
                    local_response = local_response_cache.get(local_response_key)
                    if local_response is None:
                        local_response = (
                            np.maximum(local_gray - image_gray_f32, 0.0)
                            if template.polarity == "dark"
                            else np.maximum(image_gray_f32 - local_gray, 0.0)
                        ).astype(np.float32, copy=False)
                        local_response_cache[local_response_key] = local_response
                    score_local = cv2.matchTemplate(local_response, resized.gray, cv2.TM_CCOEFF_NORMED)
                else:
                    score_local = np.zeros_like(score_gray, dtype=np.float32)
                shape_combined = np.maximum(score_gray + 0.16 * np.maximum(score_edge, 0.0), score_edge)
                local_combined = score_edge + LOCAL_MATCH_EDGE_WEIGHT * np.maximum(score_local, 0.0)
                combined = np.maximum(shape_combined, local_combined)
                maxima = _local_maxima_mask(combined, resized.width, resized.height)
                local_evidence_mask = (
                    (score_local >= LOCAL_MATCH_SCORE_THRESHOLD)
                    & (score_edge >= self.config.edge_score_threshold * 0.70)
                    & (local_combined >= self.config.candidate_threshold)
                )
                candidate_mask = (
                    (shape_combined >= self.config.candidate_threshold)
                    | (score_gray >= self.config.candidate_threshold)
                    | (score_edge >= self.config.edge_score_threshold)
                    | local_evidence_mask
                )
                if sparse_chromatic_gate and scale >= 0.85:
                    candidate_mask |= score_edge >= (
                        self.config.edge_score_threshold * self.config.sparse_chromatic_edge_candidate_threshold_ratio
                    )
                ys, xs = np.where(candidate_mask & maxima)
                for x, y in zip(xs.tolist(), ys.tolist()):
                    template_candidates.append(
                        Candidate(
                            template_name=template.name,
                            score=float(combined[y, x]),
                            gray_score=float(max(score_gray[y, x], score_local[y, x])),
                            edge_score=float(score_edge[y, x]),
                            x=int(x),
                            y=int(y),
                            width=resized.width,
                            height=resized.height,
                            scale=scale,
                        )
                    )

                if self._template_uses_faint_presence_matching(template):
                    sparse_achromatic_presence = self._template_uses_sparse_achromatic_presence(template)
                    if sparse_achromatic_presence and scale < 0.85:
                        continue
                    local_presence = cv2.matchTemplate(
                        local_response,
                        resized.gray,
                        cv2.TM_CCORR_NORMED,
                    )
                    edge_presence = cv2.matchTemplate(edge_map, resized.edge, cv2.TM_CCORR_NORMED)
                    presence_combined = 0.70 * np.maximum(local_presence, 0.0) + 0.30 * np.maximum(edge_presence, 0.0)
                    if sparse_achromatic_presence:
                        presence_combined = np.maximum(local_presence, presence_combined)
                    presence_maxima = _local_maxima_mask(presence_combined, resized.width, resized.height)
                    presence_threshold = self.config.faint_presence_score_threshold
                    if self._template_uses_sparse_achromatic_presence(template):
                        presence_threshold = min(presence_threshold, 0.23)
                    presence_mask = presence_combined >= presence_threshold
                    ys, xs = np.where(presence_mask & presence_maxima)
                    offsets = [(0.0, 0.0)]
                    if sparse_achromatic_presence and template.polarity == "light":
                        offsets.append((-0.38, 0.0))
                    seen_presence_candidates: set[tuple[int, int]] = set()
                    for x, y in zip(xs.tolist(), ys.tolist()):
                        for offset_x, offset_y in offsets:
                            candidate_x = int(round(x + offset_x * resized.width))
                            candidate_y = int(round(y + offset_y * resized.height))
                            if (
                                candidate_x < 0
                                or candidate_y < 0
                                or candidate_x >= presence_combined.shape[1]
                                or candidate_y >= presence_combined.shape[0]
                                or (candidate_x, candidate_y) in seen_presence_candidates
                                or presence_combined[candidate_y, candidate_x] < presence_threshold
                            ):
                                continue
                            seen_presence_candidates.add((candidate_x, candidate_y))
                            template_candidates.append(
                                Candidate(
                                    template_name=template.name,
                                    score=float(presence_combined[candidate_y, candidate_x]),
                                    gray_score=float(max(score_local[candidate_y, candidate_x], local_presence[candidate_y, candidate_x])),
                                    edge_score=float(max(score_edge[candidate_y, candidate_x], edge_presence[candidate_y, candidate_x])),
                                    x=int(candidate_x),
                                    y=int(candidate_y),
                                    width=resized.width,
                                    height=resized.height,
                                    scale=scale,
                                    source=CANDIDATE_SOURCE_FAINT_PRESENCE,
                                )
                            )

                if solid_dark_template:
                    self._append_solid_dark_candidates(template_candidates, template, resized, dark_fill_mask, scale)
            if solid_dark_template:
                generic_sizes = {template.size_for_scale(scale) for scale in generic_scales}
                for scale in self._scales_for_template(template):
                    scaled_width, scaled_height = template.size_for_scale(scale)
                    if (scaled_width, scaled_height) in generic_sizes:
                        continue
                    if scaled_width >= image_gray.shape[1] or scaled_height >= image_gray.shape[0]:
                        continue
                    resized = template.resized_to(scaled_width, scaled_height)
                    self._append_solid_dark_candidates(template_candidates, template, resized, dark_fill_mask, scale)
            grouped_candidates: dict[str, list[Candidate]] = {}
            for candidate in template_candidates:
                grouped_candidates.setdefault(candidate.source, []).append(candidate)
            for source_candidates in grouped_candidates.values():
                collected.extend(self._rank_and_dedupe_candidates(source_candidates))
        return sorted(collected, key=lambda item: _ranking_score(item.score, item.scale), reverse=True)

    def _append_solid_dark_candidates(
        self,
        template_candidates: list[Candidate],
        template: TemplateBundle,
        resized: ResizedTemplate,
        dark_fill_mask: np.ndarray,
        scale: float,
    ) -> None:
        solid_support = (resized.support_mask.astype(bool) & (resized.alpha > 0.20)).astype(np.float32)
        if not np.any(solid_support):
            solid_support = resized.support_mask.astype(np.float32)
        solid_response = cv2.matchTemplate(dark_fill_mask, solid_support, cv2.TM_CCORR_NORMED)
        solid_maxima = _local_maxima_mask(solid_response, resized.width, resized.height)
        ys, xs = np.where((solid_response >= self.config.solid_dark_fill_threshold) & solid_maxima)
        for x, y in zip(xs.tolist(), ys.tolist()):
            template_candidates.append(
                Candidate(
                    template_name=template.name,
                    score=float(solid_response[y, x]),
                    gray_score=float(solid_response[y, x]),
                    edge_score=0.0,
                    x=int(x),
                    y=int(y),
                    width=resized.width,
                    height=resized.height,
                    scale=scale,
                )
            )
        for edge_x in {0, solid_response.shape[1] - 1}:
            if edge_x < 0:
                continue
            column = solid_response[:, edge_x]
            if column.size == 0:
                continue
            window = max(3, resized.height // 2)
            kernel = np.ones((window,), dtype=np.float32)
            column_maxima = column == cv2.dilate(column.reshape(-1, 1), kernel.reshape(-1, 1)).reshape(-1)
            strong_column = column >= self.config.solid_dark_fill_threshold
            run_starts = np.where(strong_column & ~np.r_[False, strong_column[:-1]])[0]
            ys = np.union1d(np.where(strong_column & column_maxima)[0], run_starts)
            for y in ys.tolist():
                edge_bonus = 0.16 if y in set(run_starts.tolist()) else 0.08
                template_candidates.append(
                    Candidate(
                        template_name=template.name,
                        score=float(min(0.95, solid_response[y, edge_x] + edge_bonus)),
                        gray_score=float(solid_response[y, edge_x]),
                        edge_score=0.0,
                        x=int(edge_x),
                        y=int(y),
                        width=resized.width,
                        height=resized.height,
                        scale=scale,
                    )
                )

    def _template_uses_local_match_map(self, template: TemplateBundle) -> bool:
        return bool(self._template_uses_faint_presence_matching(template) or self._template_uses_sparse_chromatic_presence(template))

    def _collect_sparse_chromatic_presence_candidates(
        self,
        image_rgb: np.ndarray,
        target_templates: Sequence[TemplateBundle],
        presence_context: PresenceContext | None,
    ) -> list[Candidate]:
        if not target_templates:
            return []

        image_height, image_width = image_rgb.shape[:2]
        if presence_context is None:
            return []
        search_mask = (presence_context.chroma > 24.0) & (np.abs(presence_context.gray - presence_context.local) > 2.0)
        if int(search_mask.sum()) < 64:
            return []

        search_y, search_x = np.where(search_mask)
        image_directions = self._chromatic_color_directions(presence_context.image[search_y, search_x])
        collected: list[Candidate] = []
        for template in target_templates:
            directions = self._template_chromatic_foreground_directions(template)
            if directions.size == 0:
                continue

            similarity = image_directions @ directions.T
            image_mask = np.zeros(search_mask.shape, dtype=np.float32)
            image_mask[search_y, search_x] = (similarity.max(axis=1) >= 0.80).astype(np.float32)
            if int(image_mask.sum()) < 64:
                continue

            template_candidates: list[Candidate] = []
            for scale in self._scales_for_template(template):
                if scale < 0.85:
                    continue
                scaled_width, scaled_height = template.size_for_scale(scale)
                if scaled_width >= image_width or scaled_height >= image_height:
                    continue
                resized = template.resized_to(scaled_width, scaled_height)
                template_mask = self._chromatic_text_presence_template_mask(resized)
                if float(template_mask.mean()) < 0.03:
                    continue
                response = cv2.matchTemplate(image_mask, template_mask, cv2.TM_CCORR_NORMED)
                maxima = _local_maxima_mask(response, resized.width, resized.height)
                ys, xs = np.where((response >= 0.35) & maxima)
                for x, y in zip(xs.tolist(), ys.tolist()):
                    template_candidates.append(
                        Candidate(
                            template_name=template.name,
                            score=float(response[y, x]),
                            gray_score=0.0,
                            edge_score=0.0,
                            x=int(x),
                            y=int(y),
                            width=resized.width,
                            height=resized.height,
                            scale=scale,
                            source=CANDIDATE_SOURCE_CHROMATIC_PRESENCE,
                        )
                    )
            collected.extend(self._rank_and_dedupe_candidates(template_candidates))
        return sorted(collected, key=lambda item: _ranking_score(item.score, item.scale), reverse=True)

    def _collect_chromatic_text_presence_candidates(
        self,
        image_rgb: np.ndarray,
        target_templates: Sequence[TemplateBundle],
        presence_context: PresenceContext | None,
    ) -> list[Candidate]:
        if not target_templates:
            return []

        if presence_context is None:
            return []
        contrast = np.abs(presence_context.gray - presence_context.local)
        image_mask = ((presence_context.chroma > 35.0) & (contrast > 5.0)).astype(np.float32)
        if int(image_mask.sum()) < 64:
            return []

        image_height, image_width = image_rgb.shape[:2]
        collected: list[Candidate] = []
        for template in target_templates:
            foreground_components = self._template_chromatic_foreground_components(presence_context, template, image_mask)
            template_candidates: list[Candidate] = []
            for scale in self.config.scales():
                scaled_width, scaled_height = template.size_for_scale(scale)
                if scaled_width >= image_width or scaled_height >= image_height:
                    continue
                resized = template.resized_to(scaled_width, scaled_height)
                template_mask = self._chromatic_text_presence_template_mask(resized)
                if float(template_mask.mean()) < 0.05:
                    continue
                response = cv2.matchTemplate(image_mask, template_mask, cv2.TM_CCORR_NORMED)
                maxima = _local_maxima_mask(response, resized.width, resized.height)
                ys, xs = np.where((response >= self.config.chromatic_text_presence_threshold) & maxima)
                for x, y in zip(xs.tolist(), ys.tolist()):
                    score = float(response[y, x])
                    adjusted = self._adjust_chromatic_text_candidate_bbox(
                        template,
                        int(x),
                        int(y),
                        resized.width,
                        resized.height,
                        scale,
                        score,
                        foreground_components,
                        image_rgb.shape[:2],
                    )
                    if adjusted is None:
                        continue
                    x, y, width, height, scale, score = adjusted
                    template_candidates.append(
                        Candidate(
                            template_name=template.name,
                            score=score,
                            gray_score=0.0,
                            edge_score=0.0,
                            x=x,
                            y=y,
                            width=width,
                            height=height,
                            scale=scale,
                            source=CANDIDATE_SOURCE_CHROMATIC_PRESENCE,
                        )
                    )
            collected.extend(self._rank_and_dedupe_candidates(template_candidates))
        return sorted(collected, key=lambda item: _ranking_score(item.score, item.scale), reverse=True)

    def _template_chromatic_foreground_components(
        self,
        presence_context: PresenceContext | np.ndarray,
        template: TemplateBundle,
        presence_mask: np.ndarray,
    ) -> list[tuple[int, int, int, int, int]]:
        directions = self._template_chromatic_foreground_directions(template)
        if directions.size == 0:
            return []

        if isinstance(presence_context, PresenceContext):
            image = presence_context.image
            chroma = presence_context.chroma
        else:
            image = presence_context.astype(np.float32)
            chroma = image.max(axis=2) - image.min(axis=2)

        search_mask = (presence_mask > 0) | (chroma > 55.0)
        if int(search_mask.sum()) < 64:
            return []

        ys, xs = np.where(search_mask)
        image_directions = self._chromatic_color_directions(image[ys, xs])
        similarity = image_directions @ directions.T
        matched = np.zeros(search_mask.shape, dtype=np.uint8)
        matched[ys, xs] = (similarity.max(axis=1) >= 0.72).astype(np.uint8)
        mask = matched
        if int(mask.sum()) < 64:
            return []
        joined = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (45, 21)), iterations=2)
        joined = cv2.erode(joined, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 5)), iterations=1)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(joined.astype(np.uint8), connectivity=8)
        components: list[tuple[int, int, int, int, int]] = []
        for label in range(1, count):
            x, y, width, height, area = (int(value) for value in stats[label])
            if area < 1000 or width < 64 or height < 24:
                continue
            raw_pixels = int(mask[labels == label].sum())
            if raw_pixels < 512:
                continue
            components.append((x, y, width, height, raw_pixels))
        return components

    def _adjust_chromatic_text_candidate_bbox(
        self,
        template: TemplateBundle,
        x: int,
        y: int,
        width: int,
        height: int,
        scale: float,
        score: float,
        components: Sequence[tuple[int, int, int, int, int]],
        image_shape: tuple[int, int],
    ) -> tuple[int, int, int, int, float, float] | None:
        if not components:
            return None

        expected_aspect = template.size[0] / max(template.size[1], 1)
        candidate_box = (x, y, width, height)
        best: tuple[float, tuple[int, int, int, int], float] | None = None
        best_near_score = 0.0
        has_near_component = False
        for component_x, component_y, component_width, component_height, raw_pixels in components:
            component_box = (component_x, component_y, component_width, component_height)
            overlap = bbox_iou(candidate_box, component_box)
            center_inside = _is_center_contained(candidate_box, component_box) or _is_center_contained(component_box, candidate_box)
            if overlap >= 0.20 or center_inside:
                has_near_component = True
                best_near_score = max(best_near_score, self._chromatic_component_candidate_score(overlap, raw_pixels, component_width, component_height))
            aspect = component_width / max(component_height, 1)
            if aspect < expected_aspect * 0.85 or aspect > expected_aspect * 1.25:
                continue
            component_scale = 0.5 * (component_width / template.size[0] + component_height / template.size[1])
            if component_scale < self.config.min_scale or component_scale > self.config.max_scale:
                continue
            if abs(component_scale - scale) > max(self.config.scale_step * 2.5, 0.14):
                continue
            if overlap < 0.35 and not center_inside:
                continue
            component_score = self._chromatic_component_candidate_score(overlap, raw_pixels, component_width, component_height)
            if best is None or component_score > best[0]:
                best = (component_score, component_box, component_scale)

        if best is None:
            if not has_near_component:
                return None
            return x, y, width, height, scale, max(best_near_score, float(score) * 0.90)

        component_score, component_box, component_scale = best
        image_height, image_width = image_shape
        component_x, component_y, component_width, component_height = component_box
        left = max(0, component_x)
        top = max(0, component_y)
        right = min(image_width, component_x + component_width)
        bottom = min(image_height, component_y + component_height)
        return left, top, max(1, right - left), max(1, bottom - top), float(component_scale), component_score

    @staticmethod
    def _chromatic_component_candidate_score(overlap: float, raw_pixels: int, width: int, height: int) -> float:
        density = min(raw_pixels / max(width * height, 1), 0.60)
        return float(min(0.95, 0.50 + 0.18 * max(overlap, 0.0) + 0.52 * density))

    def _rank_and_dedupe_candidates(self, candidates: list[Candidate]) -> list[Candidate]:
        candidates.sort(key=lambda item: _ranking_score(item.score, item.scale), reverse=True)
        return self._dedupe_candidates(candidates)[: self.config.candidate_limit]

    def _candidate_patch(
        self,
        image_rgb: np.ndarray,
        candidate: Candidate,
    ) -> tuple[TemplateBundle, np.ndarray, ResizedTemplate]:
        template = self.templates_by_name[candidate.template_name]
        x, y, width, height = candidate.bbox
        patch = image_rgb[y : y + height, x : x + width].astype(np.float32)
        return template, patch, template.resized_to(width, height)

    def _detection_from_candidate(
        self,
        candidate: Candidate,
        template: TemplateBundle,
        fit: FitResult,
        color_score: float,
        image_shape: tuple[int, int],
        *,
        score: float,
        stage_metrics: dict[str, object] | None = None,
    ) -> Detection:
        return Detection(
            template_name=template.name,
            bbox=_pad_bbox(candidate.bbox, image_shape, self.config),
            scale=candidate.scale,
            score=score,
            color_score=color_score,
            strength=fit.strength,
            method="pending",
            objective=fit.objective,
            clip_ratio=fit.clip_ratio,
            residual=fit.residual,
            watermark_correlation=fit.watermark_correlation,
            content_bbox=candidate.bbox,
            stage_metrics=stage_metrics or {},
        )

    def _dedupe_candidates(self, candidates: list[Candidate]) -> list[Candidate]:
        if self.config.nms_iou_threshold <= 0.0:
            return self._dedupe_candidates_linear(candidates)

        kept: list[Candidate] = []
        grid: dict[tuple[int, int], list[int]] = {}
        cell_size = self._candidate_dedupe_cell_size(candidates)
        for candidate in candidates:
            nearby = self._candidate_dedupe_nearby_candidates(candidate, kept, grid, cell_size)
            if any(
                bbox_iou(candidate.bbox, existing.bbox) >= self._candidate_dedupe_iou_threshold(candidate, existing)
                and not self._keep_overlapping_candidate_scale(candidate, existing)
                for existing in nearby
            ):
                continue
            kept.append(candidate)
            self._add_candidate_to_dedupe_grid(candidate, len(kept) - 1, grid, cell_size)
            if len(kept) >= self.config.candidate_limit:
                break
        return kept

    def _dedupe_candidates_linear(self, candidates: list[Candidate]) -> list[Candidate]:
        kept: list[Candidate] = []
        for candidate in candidates:
            if any(
                bbox_iou(candidate.bbox, existing.bbox) >= self._candidate_dedupe_iou_threshold(candidate, existing)
                and not self._keep_overlapping_candidate_scale(candidate, existing)
                for existing in kept
            ):
                continue
            kept.append(candidate)
            if len(kept) >= self.config.candidate_limit:
                break
        return kept

    @staticmethod
    def _candidate_dedupe_cell_size(candidates: list[Candidate]) -> int:
        sample = candidates[:512]
        if not sample:
            return 64
        dimensions = [max(1, min(candidate.width, candidate.height)) for candidate in sample]
        return int(max(32, min(128, np.median(dimensions))))

    @staticmethod
    def _candidate_dedupe_cells(candidate: Candidate, cell_size: int) -> range:
        x0 = candidate.x // cell_size
        x1 = (candidate.x + max(1, candidate.width) - 1) // cell_size
        return range(x0, x1 + 1)

    @staticmethod
    def _candidate_dedupe_rows(candidate: Candidate, cell_size: int) -> range:
        y0 = candidate.y // cell_size
        y1 = (candidate.y + max(1, candidate.height) - 1) // cell_size
        return range(y0, y1 + 1)

    def _candidate_dedupe_nearby_candidates(
        self,
        candidate: Candidate,
        kept: list[Candidate],
        grid: dict[tuple[int, int], list[int]],
        cell_size: int,
    ) -> list[Candidate]:
        nearby: list[Candidate] = []
        seen: set[int] = set()
        for row in self._candidate_dedupe_rows(candidate, cell_size):
            for column in self._candidate_dedupe_cells(candidate, cell_size):
                for index in grid.get((column, row), ()):
                    if index not in seen:
                        seen.add(index)
                        nearby.append(kept[index])
        return nearby

    def _add_candidate_to_dedupe_grid(
        self,
        candidate: Candidate,
        index: int,
        grid: dict[tuple[int, int], list[int]],
        cell_size: int,
    ) -> None:
        for row in self._candidate_dedupe_rows(candidate, cell_size):
            for column in self._candidate_dedupe_cells(candidate, cell_size):
                grid.setdefault((column, row), []).append(index)

    def _candidate_dedupe_iou_threshold(self, candidate: Candidate, existing: Candidate) -> float:
        template = self.templates_by_name.get(candidate.template_name)
        if (
            template is not None
            and template.polarity == "light"
            and candidate.source == existing.source == CANDIDATE_SOURCE_FAINT_PRESENCE
        ):
            return max(self.config.nms_iou_threshold, 0.45)
        return self.config.nms_iou_threshold

    @staticmethod
    def _keep_overlapping_candidate_scale(candidate: Candidate, existing: Candidate) -> bool:
        if candidate.template_name != existing.template_name or candidate.source != existing.source:
            return False
        if candidate.edge_score != 0.0 or existing.edge_score != 0.0:
            return False
        if bbox_iou(candidate.bbox, existing.bbox) >= 0.65:
            return False
        return bool(abs(candidate.scale - existing.scale) >= 0.08)

    def _dedupe_candidates_by_template(self, candidates: list[Candidate]) -> list[Candidate]:
        grouped: dict[tuple[str, str], list[Candidate]] = {}
        for candidate in candidates:
            grouped.setdefault((candidate.template_name, candidate.source), []).append(candidate)

        kept: list[Candidate] = []
        for template_candidates in grouped.values():
            kept.extend(self._rank_and_dedupe_candidates(template_candidates))
        return sorted(kept, key=lambda item: _ranking_score(item.score, item.scale), reverse=True)

    def _dedupe_detections(self, detections: list[Detection]) -> list[Detection]:
        kept: list[Detection] = []
        for detection in detections:
            duplicate_index = next(
                (
                    index
                    for index, existing in enumerate(kept)
                    if (
                        bbox_iou(detection.bbox, existing.bbox) >= self.config.nms_iou_threshold
                        or _is_center_contained(detection.bbox, existing.bbox)
                        or self._is_near_duplicate_detection(detection, existing)
                    )
                ),
                None,
            )
            if duplicate_index is not None:
                existing = kept[duplicate_index]
                if (
                    detection.objective + 0.02 < existing.objective
                    and detection.watermark_correlation + 0.10 < existing.watermark_correlation
                    and detection.score >= existing.score * 0.90
                ):
                    kept[duplicate_index] = detection
                    continue
                if (
                    self._is_near_duplicate_detection(detection, existing)
                    and bbox_iou(detection.bbox, existing.bbox) < 0.60
                    and detection.objective + 0.01 < existing.objective
                    and detection.score >= existing.score * 0.96
                    and detection.scale >= existing.scale * 0.90
                ):
                    kept[duplicate_index] = detection
                continue
            kept.append(detection)
            if len(kept) >= self.config.max_detections:
                break
        return kept

    def _filter_detections_by_template_confidence(self, detections: list[Detection]) -> list[Detection]:
        grouped: dict[str, list[Detection]] = {}
        for detection in detections:
            grouped.setdefault(detection.template_name, []).append(detection)

        kept: list[Detection] = []
        for template_name, group in grouped.items():
            template = self.templates_by_name.get(template_name)
            if template is not None and self._template_profile(template).is_solid_dark:
                best_score = max(float(detection.score) for detection in group)
                min_score = best_score * 0.75
                kept.extend(detection for detection in group if detection.score >= min_score)
                continue
            if template is not None:
                profile = self._template_profile(template)
                if profile.is_faint and not profile.is_sparse:
                    best_score = max(float(detection.score) for detection in group)
                    if best_score >= 1.40:
                        min_score = best_score * 0.70
                        kept.extend(detection for detection in group if detection.score >= min_score)
                        continue
            if template is None or not _needs_sparse_chromatic_confidence_gate(template, self.config):
                kept.extend(group)
                continue

            best_score = max(float(detection.score) for detection in group)
            if best_score < self.config.sparse_chromatic_min_best_score:
                kept.extend(group)
                continue

            min_score = best_score * float(self.config.sparse_chromatic_relative_score)
            kept.extend(
                detection
                for detection in group
                if (
                    detection.score >= min_score
                    and (
                        detection.scale >= 0.85
                        or detection.color_score >= self.config.sparse_chromatic_color_rescue_threshold
                    )
                )
                or (
                    detection.scale >= 0.85
                    and detection.color_score >= self.config.sparse_chromatic_color_rescue_threshold
                )
            )
        return kept

    @staticmethod
    def _is_near_duplicate_detection(a: Detection, b: Detection) -> bool:
        if a.template_name != b.template_name:
            return False
        ax, ay, aw, ah = a.bbox
        bx, by, bw, bh = b.bbox
        overlap_x = max(0, min(ax + aw, bx + bw) - max(ax, bx))
        overlap_y = max(0, min(ay + ah, by + bh) - max(ay, by))
        if overlap_x / max(min(aw, bw), 1) < 0.72:
            return False
        center_y_gap = abs((ay + ah * 0.5) - (by + bh * 0.5))
        return bool(overlap_y > 0 and center_y_gap <= max(ah, bh) * 0.75)
