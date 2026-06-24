from __future__ import annotations

from dataclasses import asdict, replace
from threading import RLock
from typing import Any, Sequence

import numpy as np

from .detection_candidates import DetectionCandidatesMixin
from .detection_profiles import DetectionProfilesMixin, TemplateProfile
from .detection_scoring import (
    CHROMATIC_TEXT_MATCH_LOCAL_THRESHOLD,
    DEFAULT_RISK_WEIGHTS,
    EVIDENCE_GROUP_THRESHOLDS,
    EXCELLENT_FIT_GROUP_THRESHOLD,
    GOOD_FIT_GROUP_THRESHOLD,
    GOOD_GROUP_THRESHOLD,
    LOW_EDGE_GROUP_THRESHOLD,
    SMALL_SCALE_RISK_THRESHOLD,
    STRONG_COLOR_GROUP_THRESHOLD,
    STRONG_GROUP_THRESHOLD,
    STRONG_TEXT_GROUP_THRESHOLD,
    TINY_SCALE_RISK_THRESHOLD,
    VERY_STRONG_GROUP_THRESHOLD,
    WEAK_SHAPE_GROUP_THRESHOLD,
    WEAK_TEXT_GROUP_THRESHOLD,
    CandidateDecision,
    CandidateDecisionMetrics,
    CandidateEvidenceScore,
    DetectionScoringMixin,
    EvidenceProfile,
)
from .evidence_metrics import (
    _estimate_best_fit,
    _ranking_score,
    _template_color_score,
)
from .mask_builder import _aligned_template_for_patch
from .restoration import (
    _build_patch_context,
    _restore_lama_patch,
)
from .models import Candidate, Detection, FitResult, WatermarkRemoverConfig
from .templates import TemplateBundle, _compute_edge_map, _rgb_to_gray

__all__ = [
    "CHROMATIC_TEXT_MATCH_LOCAL_THRESHOLD",
    "DEFAULT_RISK_WEIGHTS",
    "EVIDENCE_GROUP_THRESHOLDS",
    "EXCELLENT_FIT_GROUP_THRESHOLD",
    "GOOD_FIT_GROUP_THRESHOLD",
    "GOOD_GROUP_THRESHOLD",
    "LOW_EDGE_GROUP_THRESHOLD",
    "SMALL_SCALE_RISK_THRESHOLD",
    "STRONG_COLOR_GROUP_THRESHOLD",
    "STRONG_GROUP_THRESHOLD",
    "STRONG_TEXT_GROUP_THRESHOLD",
    "TINY_SCALE_RISK_THRESHOLD",
    "VERY_STRONG_GROUP_THRESHOLD",
    "WEAK_SHAPE_GROUP_THRESHOLD",
    "WEAK_TEXT_GROUP_THRESHOLD",
    "CandidateDecision",
    "CandidateDecisionMetrics",
    "CandidateEvidenceScore",
    "Detection",
    "EvidenceProfile",
    "TemplateProfile",
    "TemplateBundle",
    "WatermarkRemover",
    "WatermarkRemoverConfig",
]


class WatermarkRemover(DetectionCandidatesMixin, DetectionScoringMixin, DetectionProfilesMixin):
    def __init__(self, template: TemplateBundle | Sequence[TemplateBundle], config: WatermarkRemoverConfig | None = None) -> None:
        self.templates = [template] if isinstance(template, TemplateBundle) else list(template)
        if not self.templates:
            raise ValueError("At least one watermark template is required.")
        self.template = self.templates[0]
        self.templates_by_name = {item.name: item for item in self.templates}
        self.config = config or WatermarkRemoverConfig()
        self._solid_dark_template_cache: dict[tuple[int, float], bool] = {}
        self._template_profile_cache: dict[tuple[int, float, float, float, float, float, float, float], TemplateProfile] = {}
        self._cache_lock = RLock()

    def detect(self, image_rgb: np.ndarray) -> list[Detection]:
        image_gray = _rgb_to_gray(image_rgb).astype(np.float32)
        edge_map = _compute_edge_map(image_gray)
        candidates = self._collect_evidence_map_candidates(image_rgb, image_gray, edge_map)
        candidates = self._dedupe_candidates_by_template(candidates)
        accepted = self._accept_candidates(image_rgb, candidates)

        accepted = self._filter_detections_by_template_confidence(accepted)
        accepted.sort(key=lambda item: _ranking_score(item.score, item.scale), reverse=True)
        return self._dedupe_detections(accepted)

    def _accept_candidates(self, image_rgb: np.ndarray, candidates: Sequence[Candidate]) -> list[Detection]:
        accepted: list[Detection] = []
        for candidate in candidates:
            template, patch, resized = self._candidate_patch(image_rgb, candidate)
            color_score = _template_color_score(patch, resized)
            prefit = self._prefilter_candidate(candidate, patch, resized, color_score, template)
            if not prefit.passed:
                continue
            original_candidate = candidate
            original_color_score = color_score
            candidate, color_score, resized, patch = self._refine_colored_solid_dark_candidate(
                image_rgb,
                candidate,
                template,
                color_score,
            )
            # Unchanged refinement keeps the same patch/resized inputs, so the first prefilter still applies.
            if candidate == original_candidate and float(color_score) == float(original_color_score):
                refined_prefit = prefit
            else:
                refined_prefit = self._prefilter_candidate(candidate, patch, resized, color_score, template)
            if not refined_prefit.passed:
                continue
            fit = _estimate_best_fit(patch, resized, self.config)
            final_decision = self._candidate_decision(candidate, patch, resized, color_score, template, fit)
            if final_decision.passed and self._accept_candidate_fit(fit, refined_prefit):
                evidence_profile = self._evidence_profile(self._template_profile(template))
                accepted.append(
                    self._detection_from_candidate(
                        candidate,
                        template,
                        fit,
                        color_score,
                        image_rgb.shape[:2],
                        score=final_decision.score.final_total,
                        stage_metrics={
                            "final_decision": _decision_payload(
                                final_decision,
                                evidence_profile,
                                acceptance_threshold=self._candidate_acceptance_threshold(evidence_profile, fit),
                            )
                        },
                    )
                )
        return accepted

    def diagnose_candidates(
        self,
        image_rgb: np.ndarray,
        candidates: Sequence[Candidate] | None = None,
    ) -> list[dict[str, Any]]:
        if candidates is None:
            image_gray = _rgb_to_gray(image_rgb).astype(np.float32)
            edge_map = _compute_edge_map(image_gray)
            candidates = self._dedupe_candidates_by_template(
                self._collect_evidence_map_candidates(image_rgb, image_gray, edge_map)
            )
        return [self._diagnose_candidate(image_rgb, candidate) for candidate in candidates]

    def _diagnose_candidate(self, image_rgb: np.ndarray, candidate: Candidate) -> dict[str, Any]:
        template, patch, resized = self._candidate_patch(image_rgb, candidate)
        color_score = _template_color_score(patch, resized)
        payload = _candidate_diagnostic_payload(candidate, color_score)
        profile = self._template_profile(template)
        evidence_profile = self._evidence_profile(profile)
        payload["profile"] = _profile_payload(
            evidence_profile,
            prefit_acceptance_threshold=self._candidate_acceptance_threshold(evidence_profile, fit=None),
        )

        prefit = self._prefilter_candidate(candidate, patch, resized, color_score, template)
        payload["prefit"] = _prefit_payload(prefit)
        if not prefit.passed:
            payload.update({"status": "rejected", "guard": prefit.reason or "prefilter_rejected"})
            return payload

        refined_candidate, color_score, resized, patch = self._refine_colored_solid_dark_candidate(
            image_rgb,
            candidate,
            template,
            color_score,
        )
        if refined_candidate.bbox != candidate.bbox or abs(refined_candidate.scale - candidate.scale) > 1e-9:
            payload["refined"] = _candidate_diagnostic_payload(refined_candidate, color_score)

        refined_prefit = self._prefilter_candidate(refined_candidate, patch, resized, color_score, template)
        payload["refinedPrefit"] = _prefit_payload(refined_prefit)
        if not refined_prefit.passed:
            payload.update({"status": "rejected", "guard": refined_prefit.reason or "refined_prefilter_rejected"})
            return payload

        fit = _estimate_best_fit(patch, resized, self.config)
        payload["fit"] = _fit_payload(fit)
        final_decision = self._candidate_decision(refined_candidate, patch, resized, color_score, template, fit)
        payload["finalDecision"] = _decision_payload(
            final_decision,
            evidence_profile,
            acceptance_threshold=self._candidate_acceptance_threshold(evidence_profile, fit),
        )
        if not final_decision.passed:
            payload.update({"status": "rejected", "guard": final_decision.reason or "final_decision_rejected"})
            return payload
        if not self._accept_candidate_fit(fit, refined_prefit):
            payload.update({"status": "rejected", "guard": "fit_objective_rejected"})
            return payload
        payload.update({"status": "accepted", "guard": "accepted"})
        return payload

    def remove(self, image_rgb: np.ndarray) -> tuple[np.ndarray, list[Detection]]:
        detections = self.detect(image_rgb)
        return self.restore_detections(image_rgb, detections)

    def restore_detections(
        self,
        image_rgb: np.ndarray,
        detections: Sequence[Detection],
    ) -> tuple[np.ndarray, list[Detection]]:
        if not detections:
            return image_rgb.copy(), []

        working = image_rgb.astype(np.float32).copy()
        finalized: list[Detection] = []
        for detection in sorted(detections, key=lambda item: item.score, reverse=True):
            x, y, width, height = detection.bbox
            patch = working[y : y + height, x : x + width].copy()
            template = self.templates_by_name.get(detection.template_name, self.template)
            resized = _aligned_template_for_patch(template, detection.bbox, detection.content_bbox or detection.bbox)
            fit = FitResult(
                strength=detection.strength,
                objective=detection.objective,
                clip_ratio=detection.clip_ratio,
                residual=detection.residual,
                watermark_correlation=detection.watermark_correlation,
            )
            context = _build_patch_context(working, x, y, width, height, self.config)
            result = _restore_lama_patch(patch, resized, fit, detection.score, self.config, context)
            working[y : y + height, x : x + width] = result.patch
            finalized.append(
                replace(
                    detection,
                    method=result.method,
                    residual_score=result.diagnostics.residual_score,
                    text_detail=result.diagnostics.text_detail,
                    lama_mask_ratio=result.diagnostics.lama_mask_ratio,
                    stage_metrics={**detection.stage_metrics, **result.diagnostics.stage_metrics},
                    debug_maps=result.diagnostics.debug_maps,
                )
            )
        return np.clip(working, 0.0, 255.0).astype(np.uint8), finalized


def _candidate_diagnostic_payload(candidate: Candidate, color_score: float) -> dict[str, Any]:
    return {
        "template": candidate.template_name,
        "source": candidate.source,
        "bbox": list(candidate.bbox),
        "scale": _round_float(candidate.scale),
        "score": _round_float(candidate.score),
        "gray_score": _round_float(candidate.gray_score),
        "edge_score": _round_float(candidate.edge_score),
        "color_score": _round_float(color_score),
    }


def _prefit_payload(prefit: Any) -> dict[str, Any]:
    return {
        "passed": bool(prefit.passed),
        "score": _round_float(prefit.score),
        "reason": prefit.reason,
    }


def _profile_payload(evidence_profile: EvidenceProfile, *, prefit_acceptance_threshold: float) -> dict[str, Any]:
    return {
        "kind": evidence_profile.kind,
        "acceptanceThreshold": _round_float(evidence_profile.acceptance_threshold),
        "prefitAcceptanceThreshold": _round_float(prefit_acceptance_threshold),
        "fitThreshold": _round_float(evidence_profile.fit_threshold),
        "minGroups": int(evidence_profile.min_groups),
        "requiredAnyGroups": sorted(evidence_profile.required_any_groups),
        "strongSingleGroups": sorted(evidence_profile.strong_single_groups),
    }


def _decision_payload(
    decision: CandidateDecision,
    evidence_profile: EvidenceProfile,
    *,
    acceptance_threshold: float,
) -> dict[str, Any]:
    score = asdict(decision.score)
    group_values = _decision_group_values(decision.score)
    active_groups = sorted(
        name
        for name, value in group_values.items()
        if value >= EVIDENCE_GROUP_THRESHOLDS[name]
    )
    return {
        "passed": bool(decision.passed),
        "reason": decision.reason,
        "profile": evidence_profile.kind,
        "acceptanceThreshold": _round_float(acceptance_threshold),
        "fitThreshold": _round_float(evidence_profile.fit_threshold),
        "activeGroups": active_groups,
        "scoreGroups": {key: _round_float(value) for key, value in group_values.items()},
        "score": {key: _round_float(value) if isinstance(value, float) else value for key, value in score.items()},
    }


def _decision_group_values(score: CandidateEvidenceScore) -> dict[str, float]:
    return {
        "shape": max(score.shape, score.support_coverage),
        "edge": score.edge,
        "local": score.local,
        "color": max(score.color_direction, score.chroma_delta),
        "component": score.component_fit,
        "solid": score.solid_fill,
        "fit": score.fit_quality,
        "text": score.text_detail,
    }


def _fit_payload(fit: FitResult) -> dict[str, float]:
    return {key: _round_float(value) for key, value in asdict(fit).items()}


def _round_float(value: float) -> float:
    return round(float(value), 6)
