from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .evidence_metrics import (
    _candidate_evidence,
    _debug_mask_ratio,
    _has_foreground_geometry_evidence,
    _has_low_score_match_evidence,
    _has_polarity_background_conflict,
    _has_solid_dark_template_evidence,
    _has_sparse_chromatic_color_edge_evidence,
    _has_sparse_chromatic_spatial_evidence,
    _solid_dark_template_match_metrics,
    _text_detail_mask,
    _template_chroma_delta,
    _template_chroma_strength,
    _template_color_score,
    _watermark_evidence_delta,
)
from .models import (
    CANDIDATE_SOURCE_CHROMATIC_PRESENCE,
    CANDIDATE_SOURCE_FAINT_PRESENCE,
    CANDIDATE_SOURCE_MATCH,
    EPSILON,
    Candidate,
    CandidateEvidence,
    CandidatePrefit,
    FitResult,
)
from .templates import ResizedTemplate, TemplateBundle
from .detection_profiles import TemplateProfile


@dataclass(slots=True)
class CandidateDecisionMetrics:
    evidence_delta: float
    chroma_delta: float
    color_score: float
    support_ratio: float
    template_chroma: float
    solid_dark_fill_ratio: float
    solid_dark_hole_bright_ratio: float
    foreground_geometry: bool
    low_score_evidence: bool
    sparse_chromatic_template: bool
    sparse_chromatic_color_edge: bool
    sparse_chromatic_spatial: bool
    solid_dark_template: bool
    solid_dark_evidence: bool

@dataclass(frozen=True, slots=True)
class EvidenceProfile:
    kind: str
    min_groups: int
    required_any_groups: frozenset[str]
    strong_single_groups: frozenset[str]
    acceptance_threshold: float
    fit_threshold: float
    risk_weights: dict[str, float]

@dataclass(slots=True)
class CandidateEvidenceScore:
    shape: float
    edge: float
    local: float
    color_direction: float
    chroma_delta: float
    support_coverage: float
    component_fit: float
    solid_fill: float
    text_detail: float
    scale_quality: float
    fit_quality: float
    evidence_total: float
    risk: float
    final_total: float
    evidence_count: int

@dataclass(slots=True)
class CandidateDecision:
    passed: bool
    score: CandidateEvidenceScore
    reason: str = ""


EVIDENCE_GROUP_THRESHOLDS: dict[str, float] = {
    "shape": 0.45,
    "edge": 0.45,
    "local": 0.45,
    "color": 0.45,
    "component": 0.55,
    "solid": 0.50,
    "fit": 0.55,
    "text": 0.55,
}

DEFAULT_RISK_WEIGHTS: dict[str, float] = {
    "tiny_scale": 0.30,
    "small_scale": 0.16,
    "weak_shape": 0.18,
    "weak_edge": 0.16,
    "weak_local": 0.18,
    "weak_color": 0.20,
    "weak_component": 0.22,
    "weak_solid": 0.26,
    "weak_text": 0.18,
    "color_without_shape": 0.30,
    "shape_without_support": 0.18,
    "presence_without_structure": 0.24,
    "polarity_conflict": 0.30,
    "weak_fit": 0.22,
    "poor_fit": 0.32,
}

LOW_EDGE_GROUP_THRESHOLD = 0.20
WEAK_SHAPE_GROUP_THRESHOLD = 0.35
WEAK_TEXT_GROUP_THRESHOLD = 0.20
TINY_SCALE_RISK_THRESHOLD = 0.45
SMALL_SCALE_RISK_THRESHOLD = 0.55
GOOD_GROUP_THRESHOLD = 0.65
STRONG_GROUP_THRESHOLD = 0.70
CHROMATIC_TEXT_MATCH_LOCAL_THRESHOLD = 0.85
STRONG_COLOR_GROUP_THRESHOLD = 0.90
STRONG_TEXT_GROUP_THRESHOLD = 0.80
VERY_STRONG_GROUP_THRESHOLD = 0.90
GOOD_FIT_GROUP_THRESHOLD = 0.85
EXCELLENT_FIT_GROUP_THRESHOLD = 0.98


class DetectionScoringMixin:
    def _prefilter_candidate(
        self,
        candidate: Candidate,
        patch: np.ndarray,
        resized: ResizedTemplate,
        color_score: float,
        template: TemplateBundle,
    ) -> CandidatePrefit:
        decision = self._candidate_decision(candidate, patch, resized, color_score, template, fit=None)
        return CandidatePrefit(decision.passed, score=decision.score.final_total, reason=decision.reason)

    def _candidate_decision(
        self,
        candidate: Candidate,
        patch: np.ndarray,
        resized: ResizedTemplate,
        color_score: float,
        template: TemplateBundle,
        fit: FitResult | None,
    ) -> CandidateDecision:
        evidence = _candidate_evidence(patch, resized)
        evidence.text_detail = _debug_mask_ratio(_text_detail_mask(patch, resized, self.config), resized.support_mask)
        metrics = self._candidate_decision_metrics(candidate, patch, resized, color_score, template, evidence)
        profile = self._template_profile(template)
        evidence_profile = self._evidence_profile(profile)
        score = self._candidate_evidence_score(candidate, resized, evidence, metrics, profile, evidence_profile, fit)
        guard_failure = self._candidate_safety_guard_failure(
            candidate,
            resized,
            evidence,
            metrics,
            profile,
            evidence_profile,
            score,
            fit,
        )
        if guard_failure:
            return CandidateDecision(False, score, guard_failure)
        threshold = self._candidate_acceptance_threshold(evidence_profile, fit)
        passed = score.final_total >= threshold
        return CandidateDecision(passed, score, "" if passed else "score_below_threshold")

    def _candidate_evidence_score(
        self,
        candidate: Candidate,
        resized: ResizedTemplate,
        evidence: CandidateEvidence,
        metrics: CandidateDecisionMetrics,
        profile: TemplateProfile,
        evidence_profile: EvidenceProfile,
        fit: FitResult | None,
    ) -> CandidateEvidenceScore:
        candidate_signal = self._unit_interval(
            candidate.score,
            self.config.candidate_threshold,
            self.config.score_threshold + 0.18,
        )
        gray_signal = self._unit_interval(
            candidate.gray_score,
            self.config.candidate_threshold,
            self.config.score_threshold + 0.12,
        )
        support_signal = self._unit_interval(
            evidence.support_corr,
            self.config.support_correlation_threshold * 0.50,
            self.config.foreground_strong_shape_correlation,
        )
        shape = max(candidate_signal, gray_signal, support_signal)
        edge = self._unit_interval(
            candidate.edge_score,
            self.config.edge_score_threshold * 0.70,
            self.config.edge_score_threshold + 0.25,
        )
        local = max(
            self._unit_interval(metrics.evidence_delta, 0.0, self.config.color_match_evidence_delta_threshold),
            self._unit_interval(candidate.score, self.config.faint_presence_score_threshold, 0.60)
            if candidate.source == CANDIDATE_SOURCE_FAINT_PRESENCE
            else 0.0,
        )
        if metrics.low_score_evidence:
            local = max(local, 0.82)
        color_direction = self._unit_interval(
            metrics.color_score,
            self.config.color_score_threshold,
            self.config.sparse_chromatic_color_rescue_threshold,
        )
        chroma_delta = self._unit_interval(
            metrics.chroma_delta,
            self.config.sparse_chromatic_chroma_delta_threshold * 0.50,
            self.config.sparse_chromatic_chroma_delta_threshold * 1.50,
        )
        component_fit = (
            self._unit_interval(candidate.score, self.config.candidate_threshold, self.config.score_threshold + 0.08)
            if profile.is_chromatic_text and candidate.source == CANDIDATE_SOURCE_CHROMATIC_PRESENCE
            else 0.0
        )
        fill = self._unit_interval(
            metrics.solid_dark_fill_ratio,
            self.config.solid_dark_fill_threshold * 0.70,
            self.config.solid_dark_fill_threshold,
        )
        hole = self._unit_interval(
            metrics.solid_dark_hole_bright_ratio,
            self.config.solid_dark_hole_bright_threshold * 0.40,
            self.config.solid_dark_hole_bright_threshold,
        )
        solid_fill = max(min(fill, hole), fill * 0.72 + hole * 0.28 if profile.is_solid_dark else 0.0)
        text_detail = self._unit_interval(evidence.text_detail, 0.50, self.config.large_template_text_detail_threshold)
        scale_quality = self._unit_interval(candidate.scale, 0.55, 0.85)
        fit_quality = self._fit_quality(fit)

        evidence_total = 0.0
        evidence_total += 0.30 * shape
        evidence_total += 0.22 * edge
        evidence_total += 0.24 * local
        evidence_total += 0.22 * color_direction
        evidence_total += 0.18 * chroma_delta
        evidence_total += 0.25 * support_signal
        evidence_total += 0.18 * component_fit
        evidence_total += 0.22 * solid_fill
        evidence_total += 0.16 * text_detail
        evidence_total += 0.12 * scale_quality
        evidence_total += 0.22 * fit_quality
        if metrics.foreground_geometry:
            evidence_total += 0.16
        if metrics.low_score_evidence:
            evidence_total += 0.18
        if metrics.sparse_chromatic_color_edge:
            evidence_total += 0.24
        if metrics.sparse_chromatic_spatial:
            evidence_total += 0.14
        if metrics.solid_dark_evidence:
            evidence_total += 0.18
        if candidate.source == CANDIDATE_SOURCE_CHROMATIC_PRESENCE:
            evidence_total += 0.08 + 0.18 * min(max(candidate_signal, component_fit), max(color_direction, chroma_delta))
            if profile.is_sparse_chromatic:
                evidence_total += 0.35 * min(chroma_delta, text_detail)
        if candidate.source == CANDIDATE_SOURCE_FAINT_PRESENCE:
            evidence_total += 0.10 * max(local, edge)

        group_values = self._evidence_group_values(
            shape=shape,
            edge=edge,
            local=local,
            color_direction=color_direction,
            chroma_delta=chroma_delta,
            support_coverage=support_signal,
            component_fit=component_fit,
            solid_fill=solid_fill,
            text_detail=text_detail,
            fit_quality=fit_quality,
        )
        evidence_count = sum(
            value >= EVIDENCE_GROUP_THRESHOLDS[name] for name, value in group_values.items()
        )
        risk = self._candidate_risk(
            candidate,
            resized,
            evidence,
            metrics,
            profile,
            evidence_profile,
            group_values,
            fit,
        )
        final_total = evidence_total - risk

        return CandidateEvidenceScore(
            shape=shape,
            edge=edge,
            local=local,
            color_direction=color_direction,
            chroma_delta=chroma_delta,
            support_coverage=support_signal,
            component_fit=component_fit,
            solid_fill=solid_fill,
            text_detail=text_detail,
            scale_quality=scale_quality,
            fit_quality=fit_quality,
            evidence_total=float(evidence_total),
            risk=float(risk),
            final_total=float(final_total),
            evidence_count=int(evidence_count),
        )

    def _candidate_safety_guard_failure(
        self,
        candidate: Candidate,
        resized: ResizedTemplate,
        evidence: CandidateEvidence,
        metrics: CandidateDecisionMetrics,
        profile: TemplateProfile,
        evidence_profile: EvidenceProfile,
        score: CandidateEvidenceScore,
        fit: FitResult | None,
    ) -> str:
        if candidate.width <= 0 or candidate.height <= 0:
            return "invalid_size"
        if resized.support_mask.sum() <= 0:
            return "empty_template_support"
        if candidate.scale < self.config.min_scale - EPSILON or candidate.scale > self.config.max_scale + EPSILON:
            return "invalid_scale"

        group_values = self._score_group_values(score)
        active_groups = {
            name for name, value in group_values.items() if value >= EVIDENCE_GROUP_THRESHOLDS[name]
        }
        strong_single = any(group_values[name] >= 0.92 for name in evidence_profile.strong_single_groups)
        if fit is not None:
            strong_single = strong_single and score.fit_quality >= evidence_profile.fit_threshold
        if len(active_groups) < evidence_profile.min_groups and not strong_single:
            return "insufficient_evidence"

        if not any(group_values[name] >= EVIDENCE_GROUP_THRESHOLDS[name] for name in evidence_profile.required_any_groups):
            return "missing_required_evidence"
        if profile.is_solid_dark and score.solid_fill < 0.10:
            return "solid_dark_missing_fill"
        if profile.is_solid_dark and candidate.scale < self.config.tiny_scale_threshold:
            return "solid_dark_too_small"
        if evidence_profile.kind == "faint_sparse_achromatic" and candidate.scale < self.config.tiny_scale_threshold:
            return "sparse_achromatic_too_small"
        if (
            fit is not None
            and fit.objective > 0.30
            and score.fit_quality < evidence_profile.fit_threshold
            and score.final_total < evidence_profile.acceptance_threshold + 0.10
        ):
            return "poor_fit_quality"
        if (
            _has_polarity_background_conflict(candidate, resized, evidence.support_mean, self.config)
            and score.fit_quality < 0.35
            and max(group_values["local"], group_values["color"], group_values["shape"]) < 0.45
        ):
            return "polarity_background_conflict"
        return ""

    def _candidate_acceptance_threshold(
        self,
        evidence_profile: EvidenceProfile,
        fit: FitResult | None,
    ) -> float:
        if fit is None:
            if evidence_profile.kind == "solid_dark":
                return -0.30
            if evidence_profile.kind == "chromatic_text":
                return 0.55
            if evidence_profile.kind in {"faint_dense", "faint_sparse_achromatic", "sparse_chromatic"}:
                return 0.60
            return float(max(0.72, min(1.00, evidence_profile.acceptance_threshold - 0.16)))
        return float(evidence_profile.acceptance_threshold)

    def _with_profile_threshold_overrides(self, evidence_profile: EvidenceProfile) -> EvidenceProfile:
        acceptance = self.config.profile_acceptance_thresholds.get(evidence_profile.kind)
        fit = self.config.profile_fit_thresholds.get(evidence_profile.kind)
        if acceptance is None and fit is None:
            return evidence_profile
        return replace(
            evidence_profile,
            acceptance_threshold=float(acceptance) if acceptance is not None else evidence_profile.acceptance_threshold,
            fit_threshold=float(fit) if fit is not None else evidence_profile.fit_threshold,
        )

    def _evidence_profile(self, profile: TemplateProfile) -> EvidenceProfile:
        if profile.is_solid_dark:
            return self._with_profile_threshold_overrides(EvidenceProfile(
                kind="solid_dark",
                min_groups=2,
                required_any_groups=frozenset({"solid", "text", "shape"}),
                strong_single_groups=frozenset({"solid"}),
                acceptance_threshold=0.79,
                fit_threshold=0.35,
                risk_weights={
                    **DEFAULT_RISK_WEIGHTS,
                    "tiny_scale": 0.40,
                    "weak_shape": 0.24,
                    "weak_solid": 0.48,
                    "weak_text": 0.24,
                    "weak_fit": 0.34,
                },
            ))
        if profile.is_chromatic_text:
            return self._with_profile_threshold_overrides(EvidenceProfile(
                kind="chromatic_text",
                min_groups=2,
                required_any_groups=frozenset({"component", "color", "shape"}),
                strong_single_groups=frozenset({"component", "color"}),
                acceptance_threshold=0.76,
                fit_threshold=0.25,
                risk_weights={
                    **DEFAULT_RISK_WEIGHTS,
                    "tiny_scale": 0.35,
                    "weak_component": 0.36,
                    "weak_color": 0.22,
                    "color_without_shape": 0.34,
                },
            ))
        if profile.is_sparse_chromatic:
            return self._with_profile_threshold_overrides(EvidenceProfile(
                kind="sparse_chromatic",
                min_groups=2,
                required_any_groups=frozenset({"color", "shape", "edge"}),
                strong_single_groups=frozenset({"color"}),
                acceptance_threshold=1.06,
                fit_threshold=0.28,
                risk_weights={
                    **DEFAULT_RISK_WEIGHTS,
                    "tiny_scale": 0.26,
                    "weak_edge": 0.05,
                    "weak_color": 0.34,
                    "color_without_shape": 0.36,
                    "presence_without_structure": 0.28,
                },
            ))
        if profile.is_faint and profile.is_sparse and not profile.is_chromatic:
            return self._with_profile_threshold_overrides(EvidenceProfile(
                kind="faint_sparse_achromatic",
                min_groups=2,
                required_any_groups=frozenset({"local", "edge", "text", "shape"}),
                strong_single_groups=frozenset({"local", "text"}),
                acceptance_threshold=0.88,
                fit_threshold=0.30,
                risk_weights={
                    **DEFAULT_RISK_WEIGHTS,
                    "tiny_scale": 0.40,
                    "weak_edge": 0.12,
                    "weak_local": 0.24,
                    "weak_text": 0.36,
                    "presence_without_structure": 0.36,
                    "weak_fit": 0.28,
                },
            ))
        if profile.is_faint:
            return self._with_profile_threshold_overrides(EvidenceProfile(
                kind="faint_dense",
                min_groups=2,
                required_any_groups=frozenset({"local", "shape", "edge"}),
                strong_single_groups=frozenset({"local"}),
                acceptance_threshold=1.00,
                fit_threshold=0.25,
                risk_weights={
                    **DEFAULT_RISK_WEIGHTS,
                    "weak_edge": 0.22,
                    "weak_local": 0.32,
                    "presence_without_structure": 0.30,
                },
            ))
        return self._with_profile_threshold_overrides(EvidenceProfile(
            kind="general",
            min_groups=2,
            required_any_groups=frozenset({"shape", "edge", "fit"}),
            strong_single_groups=frozenset({"shape", "edge"}),
            acceptance_threshold=1.12,
            fit_threshold=0.30,
            risk_weights=DEFAULT_RISK_WEIGHTS.copy(),
        ))

    def _evidence_group_values(
        self,
        *,
        shape: float,
        edge: float,
        local: float,
        color_direction: float,
        chroma_delta: float,
        support_coverage: float,
        component_fit: float,
        solid_fill: float,
        text_detail: float,
        fit_quality: float,
    ) -> dict[str, float]:
        return {
            "shape": max(shape, support_coverage),
            "edge": edge,
            "local": local,
            "color": max(color_direction, chroma_delta),
            "component": component_fit,
            "solid": solid_fill,
            "fit": fit_quality,
            "text": text_detail,
        }

    def _score_group_values(self, score: CandidateEvidenceScore) -> dict[str, float]:
        return self._evidence_group_values(
            shape=score.shape,
            edge=score.edge,
            local=score.local,
            color_direction=score.color_direction,
            chroma_delta=score.chroma_delta,
            support_coverage=score.support_coverage,
            component_fit=score.component_fit,
            solid_fill=score.solid_fill,
            text_detail=score.text_detail,
            fit_quality=score.fit_quality,
        )

    def _solid_dark_background_block_risk(
        self,
        evidence_profile: EvidenceProfile,
        evidence: CandidateEvidence,
        metrics: CandidateDecisionMetrics,
        group_values: dict[str, float],
    ) -> float:
        if evidence_profile.kind != "solid_dark" or group_values["edge"] >= LOW_EDGE_GROUP_THRESHOLD:
            return 0.0

        weak_hole = metrics.solid_dark_hole_bright_ratio < self.config.solid_dark_hole_bright_threshold * 0.37
        weak_structure = (
            group_values["local"] < 0.80
            or group_values["text"] < 0.55
            or evidence.support_corr < self.config.support_correlation_threshold
        )
        if weak_hole and weak_structure:
            return 0.55
        return 0.0

    def _low_edge_texture_risk(
        self,
        candidate: Candidate,
        evidence: CandidateEvidence,
        metrics: CandidateDecisionMetrics,
        profile: TemplateProfile,
        evidence_profile: EvidenceProfile,
        group_values: dict[str, float],
        fit: FitResult | None,
    ) -> float:
        if evidence_profile.kind not in {"faint_sparse_achromatic", "faint_dense"}:
            return 0.0
        if candidate.source not in {CANDIDATE_SOURCE_FAINT_PRESENCE, CANDIDATE_SOURCE_MATCH}:
            return 0.0
        if group_values["edge"] >= LOW_EDGE_GROUP_THRESHOLD:
            return 0.0

        risk = 0.0
        if group_values["local"] >= 0.80:
            texture_like = evidence.support_corr >= self.config.support_correlation_threshold * 0.95
            weak_delta = metrics.evidence_delta < self.config.foreground_residual_delta_threshold * 2.30
            fit_texture = fit is not None and fit.watermark_correlation >= 0.05
            shape_texture = group_values["shape"] >= GOOD_GROUP_THRESHOLD
            if texture_like and (weak_delta or fit_texture or shape_texture):
                risk += 0.28 if candidate.source == CANDIDATE_SOURCE_FAINT_PRESENCE else 0.18

            low_shape_dark_texture = (
                group_values["shape"] < 0.20
                and group_values["text"] >= STRONG_TEXT_GROUP_THRESHOLD
                and evidence.support_corr >= self.config.support_correlation_threshold * 0.70
                and 120.0 <= evidence.support_mean <= 190.0
            )
            if low_shape_dark_texture:
                risk += 0.30
            weak_alignment = (
                fit is not None
                and profile.polarity == "light"
                and group_values["shape"] < WEAK_SHAPE_GROUP_THRESHOLD
                and evidence.support_corr < self.config.support_correlation_threshold
                and group_values["fit"] < EXCELLENT_FIT_GROUP_THRESHOLD
            )
            if weak_alignment:
                risk += 0.30
        weak_dark_low_contrast_alignment = (
            fit is not None
            and profile.polarity == "dark"
            and group_values["local"] < 0.80
            and group_values["shape"] < WEAK_SHAPE_GROUP_THRESHOLD
            and self.config.support_correlation_threshold <= evidence.support_corr < self.config.foreground_strong_shape_correlation
            and metrics.evidence_delta < self.config.foreground_residual_delta_threshold * 1.50
            and group_values["fit"] < EXCELLENT_FIT_GROUP_THRESHOLD
        )
        if weak_dark_low_contrast_alignment:
            risk += 0.30
        return float(risk)

    def _missing_evidence_amount(self, group_values: dict[str, float], group: str) -> float:
        threshold = EVIDENCE_GROUP_THRESHOLDS[group]
        return 1.0 - self._unit_interval(group_values[group], 0.0, threshold)

    def _scale_risk(self, candidate: Candidate, evidence_profile: EvidenceProfile) -> float:
        weights = evidence_profile.risk_weights
        if candidate.scale < self.config.tiny_scale_threshold:
            return float(weights["tiny_scale"])
        if candidate.scale < self.config.small_scale_threshold:
            return float(weights["small_scale"])
        return 0.0

    def _required_group_risk(
        self,
        evidence_profile: EvidenceProfile,
        group_values: dict[str, float],
        *,
        strong_sparse_color: bool,
        strong_achromatic_structure: bool,
        strong_chromatic_text_match: bool,
    ) -> float:
        weights = evidence_profile.risk_weights
        risk = 0.0

        for group, weight_name in (
            ("shape", "weak_shape"),
            ("edge", "weak_edge"),
            ("local", "weak_local"),
            ("color", "weak_color"),
            ("component", "weak_component"),
            ("solid", "weak_solid"),
            ("text", "weak_text"),
        ):
            if group not in evidence_profile.required_any_groups:
                continue
            if group_values[group] >= EVIDENCE_GROUP_THRESHOLDS[group]:
                continue
            if strong_sparse_color and group in {"shape", "edge"}:
                continue
            if strong_achromatic_structure and group == "text":
                continue
            if strong_chromatic_text_match and group == "component":
                continue
            risk += weights[weight_name] * self._missing_evidence_amount(group_values, group)
        return float(risk)

    def _generic_balance_risk(
        self,
        candidate: Candidate,
        metrics: CandidateDecisionMetrics,
        profile: TemplateProfile,
        evidence_profile: EvidenceProfile,
        group_values: dict[str, float],
        *,
        strong_sparse_color: bool,
    ) -> float:
        weights = evidence_profile.risk_weights
        risk = 0.0
        if (
            group_values["color"] >= 0.55
            and group_values["shape"] < WEAK_SHAPE_GROUP_THRESHOLD
            and not strong_sparse_color
        ):
            risk += weights["color_without_shape"] * (
                1.0 - self._unit_interval(group_values["shape"], 0.0, WEAK_SHAPE_GROUP_THRESHOLD)
            )
        if (
            group_values["shape"] >= GOOD_GROUP_THRESHOLD
            and max(group_values["local"], group_values["color"], group_values["edge"]) < 0.30
        ):
            risk += weights["shape_without_support"]
        if candidate.source in {CANDIDATE_SOURCE_CHROMATIC_PRESENCE, CANDIDATE_SOURCE_FAINT_PRESENCE}:
            if max(group_values["shape"], group_values["edge"], group_values["text"]) < WEAK_SHAPE_GROUP_THRESHOLD:
                risk += weights["presence_without_structure"]
        if candidate.source == CANDIDATE_SOURCE_MATCH and metrics.color_score < self.config.color_score_threshold:
            risk += 0.10 * self._unit_interval(
                self.config.foreground_residual_delta_threshold - metrics.evidence_delta,
                0.0,
                self.config.foreground_residual_delta_threshold,
            )
        if profile.is_sparse and max(group_values["color"], group_values["local"]) < WEAK_SHAPE_GROUP_THRESHOLD:
            risk += 0.06
        return float(risk)

    def _sparse_chromatic_risk(
        self,
        candidate: Candidate,
        metrics: CandidateDecisionMetrics,
        profile: TemplateProfile,
        evidence_profile: EvidenceProfile,
        group_values: dict[str, float],
    ) -> float:
        if evidence_profile.kind != "sparse_chromatic":
            return 0.0

        risk = 0.0
        if group_values["color"] < 0.80 and group_values["edge"] < LOW_EDGE_GROUP_THRESHOLD:
            risk += 0.24
        if (
            candidate.source == CANDIDATE_SOURCE_CHROMATIC_PRESENCE
            and group_values["color"] < STRONG_COLOR_GROUP_THRESHOLD
            and group_values["edge"] < LOW_EDGE_GROUP_THRESHOLD
        ):
            risk += 0.60
        if (
            candidate.source == CANDIDATE_SOURCE_MATCH
            and group_values["text"] < WEAK_TEXT_GROUP_THRESHOLD
            and group_values["color"] < STRONG_COLOR_GROUP_THRESHOLD
            and not metrics.sparse_chromatic_spatial
        ):
            risk += 0.55
        if (
            group_values["shape"] < WEAK_SHAPE_GROUP_THRESHOLD
            and group_values["edge"] < LOW_EDGE_GROUP_THRESHOLD
            and metrics.chroma_delta < self.config.sparse_chromatic_chroma_delta_threshold
            and not metrics.sparse_chromatic_spatial
            and not metrics.sparse_chromatic_color_edge
        ):
            risk += 0.35
        if (
            profile.chroma_strength >= self.config.sparse_chromatic_chroma_threshold * 2.40
            and not metrics.sparse_chromatic_spatial
        ):
            risk += 0.75
        return float(risk)

    def _faint_sparse_achromatic_risk(
        self,
        candidate: Candidate,
        evidence: CandidateEvidence,
        metrics: CandidateDecisionMetrics,
        profile: TemplateProfile,
        evidence_profile: EvidenceProfile,
        group_values: dict[str, float],
        fit: FitResult | None,
    ) -> float:
        if evidence_profile.kind != "faint_sparse_achromatic":
            return 0.0

        risk = self._low_edge_texture_risk(candidate, evidence, metrics, profile, evidence_profile, group_values, fit)
        ultra_sparse_template = profile.support_ratio < 0.09 and profile.alpha_coverage < 0.08
        if ultra_sparse_template and candidate.source in {CANDIDATE_SOURCE_FAINT_PRESENCE, CANDIDATE_SOURCE_MATCH}:
            if (
                group_values["edge"] < LOW_EDGE_GROUP_THRESHOLD
                and group_values["shape"] < WEAK_SHAPE_GROUP_THRESHOLD
                and evidence.support_corr < self.config.support_correlation_threshold
            ):
                risk += 0.45
            if (
                group_values["edge"] < LOW_EDGE_GROUP_THRESHOLD
                and group_values["local"] < STRONG_GROUP_THRESHOLD
                and metrics.evidence_delta < self.config.foreground_residual_delta_threshold * 1.50
            ):
                risk += 0.35
        if (
            profile.polarity == "light"
            and candidate.source == CANDIDATE_SOURCE_FAINT_PRESENCE
            and group_values["edge"] < LOW_EDGE_GROUP_THRESHOLD
            and group_values["shape"] < 0.45
            and evidence.support_corr < 0.20
            and evidence.support_mean < 165.0
        ):
            risk += 0.45
        if candidate.source == CANDIDATE_SOURCE_MATCH and group_values["edge"] < LOW_EDGE_GROUP_THRESHOLD:
            risk += 0.55
        if candidate.source == CANDIDATE_SOURCE_MATCH and group_values["text"] < WEAK_TEXT_GROUP_THRESHOLD:
            risk += 0.45
        if (
            candidate.source == CANDIDATE_SOURCE_MATCH
            and group_values["edge"] < LOW_EDGE_GROUP_THRESHOLD
            and group_values["fit"] < 0.75
        ):
            risk += 0.25
        if candidate.source == CANDIDATE_SOURCE_FAINT_PRESENCE:
            smooth_background_alignment = (
                fit is not None
                and group_values["edge"] < LOW_EDGE_GROUP_THRESHOLD
                and group_values["color"] < 0.10
                and group_values["shape"] >= GOOD_GROUP_THRESHOLD
                and group_values["local"] < STRONG_GROUP_THRESHOLD
                and fit.watermark_correlation < 0.05
                and metrics.evidence_delta < self.config.foreground_residual_delta_threshold * 1.80
            )
            if smooth_background_alignment:
                risk += 0.45
            if (
                fit is not None
                and group_values["edge"] < LOW_EDGE_GROUP_THRESHOLD
                and group_values["shape"] >= WEAK_SHAPE_GROUP_THRESHOLD
            ):
                risk += 0.60 * self._unit_interval(fit.watermark_correlation, 0.08, 0.18)
            if fit is not None and group_values["edge"] < LOW_EDGE_GROUP_THRESHOLD and group_values["fit"] < GOOD_FIT_GROUP_THRESHOLD:
                risk += 0.20
            if group_values["edge"] < LOW_EDGE_GROUP_THRESHOLD and group_values["shape"] >= WEAK_SHAPE_GROUP_THRESHOLD:
                risk += 0.20
            if (
                group_values["edge"] < LOW_EDGE_GROUP_THRESHOLD
                and group_values["shape"] > GOOD_GROUP_THRESHOLD
                and group_values["local"] < GOOD_GROUP_THRESHOLD
            ):
                risk += 0.35
            if candidate.scale > 1.10 and group_values["edge"] < LOW_EDGE_GROUP_THRESHOLD:
                risk += 0.70
            if (
                candidate.scale < 0.98
                and group_values["edge"] < LOW_EDGE_GROUP_THRESHOLD
                and group_values["shape"] < WEAK_SHAPE_GROUP_THRESHOLD
            ):
                risk += 0.30
            if fit is not None and group_values["text"] < WEAK_TEXT_GROUP_THRESHOLD and group_values["fit"] < GOOD_FIT_GROUP_THRESHOLD:
                risk += 0.40
        if (
            candidate.scale < 0.70
            and not (
                group_values["shape"] >= 0.75
                and group_values["local"] >= STRONG_GROUP_THRESHOLD
                and (group_values["edge"] >= EVIDENCE_GROUP_THRESHOLDS["edge"] or group_values["text"] >= EVIDENCE_GROUP_THRESHOLDS["text"])
            )
        ):
            risk += 0.65
        if (
            fit is not None
            and candidate.source == CANDIDATE_SOURCE_MATCH
            and candidate.scale < 0.70
            and evidence.support_corr >= self.config.support_correlation_threshold
            and fit.watermark_correlation >= 0.16
        ):
            risk += 0.65
        if (
            profile.polarity == "dark"
            and group_values["text"] < WEAK_TEXT_GROUP_THRESHOLD
            and group_values["local"] >= STRONG_GROUP_THRESHOLD
            and evidence.support_corr >= self.config.support_correlation_threshold
            and evidence.support_mean < 145.0
        ):
            risk += 0.42
        return float(risk)

    def _faint_dense_risk(
        self,
        candidate: Candidate,
        evidence: CandidateEvidence,
        metrics: CandidateDecisionMetrics,
        profile: TemplateProfile,
        evidence_profile: EvidenceProfile,
        group_values: dict[str, float],
        fit: FitResult | None,
    ) -> float:
        if evidence_profile.kind != "faint_dense":
            return 0.0

        risk = self._low_edge_texture_risk(candidate, evidence, metrics, profile, evidence_profile, group_values, fit)
        if (
            candidate.source == CANDIDATE_SOURCE_FAINT_PRESENCE
            and group_values["edge"] < LOW_EDGE_GROUP_THRESHOLD
            and group_values["shape"] >= WEAK_SHAPE_GROUP_THRESHOLD
            and group_values["fit"] < 0.90
        ):
            risk += 0.25
        return float(risk)

    def _solid_dark_risk(
        self,
        candidate: Candidate,
        evidence: CandidateEvidence,
        metrics: CandidateDecisionMetrics,
        evidence_profile: EvidenceProfile,
        group_values: dict[str, float],
        fit: FitResult | None,
    ) -> float:
        if evidence_profile.kind != "solid_dark":
            return 0.0

        risk = self._solid_dark_background_block_risk(evidence_profile, evidence, metrics, group_values)
        if group_values["solid"] < 0.85 and group_values["text"] > STRONG_TEXT_GROUP_THRESHOLD and group_values["color"] > 0.40:
            risk += 0.75
        if (
            fit is not None
            and group_values["solid"] < 0.85
            and group_values["text"] < 0.50
            and group_values["local"] < STRONG_GROUP_THRESHOLD
            and fit.watermark_correlation > 0.08
        ):
            risk += 0.35
        if (
            candidate.scale > 0.75
            and group_values["text"] < 0.10
            and group_values["color"] < 0.10
            and group_values["edge"] < LOW_EDGE_GROUP_THRESHOLD
        ):
            risk += 0.45
        if candidate.scale > 0.62 and group_values["text"] < 0.10 and group_values["edge"] < LOW_EDGE_GROUP_THRESHOLD:
            risk += 0.50
        if (
            group_values["solid"] < 0.78
            and group_values["text"] < 0.10
            and group_values["color"] < 0.10
            and group_values["edge"] < LOW_EDGE_GROUP_THRESHOLD
        ):
            risk += 0.35
        if (
            fit is not None
            and group_values["text"] < 0.10
            and group_values["edge"] < LOW_EDGE_GROUP_THRESHOLD
            and fit.watermark_correlation > 0.08
        ):
            risk += 0.35
        if (
            group_values["text"] < 0.10
            and group_values["color"] < 0.10
            and group_values["edge"] < LOW_EDGE_GROUP_THRESHOLD
            and group_values["fit"] < GOOD_FIT_GROUP_THRESHOLD
        ):
            risk += 0.35
        if (
            group_values["solid"] <= self.config.solid_dark_fill_threshold
            and candidate.scale <= 0.50
            and group_values["text"] > STRONG_GROUP_THRESHOLD
            and group_values["edge"] < LOW_EDGE_GROUP_THRESHOLD
        ):
            risk += 0.35
        if (
            group_values["solid"] <= self.config.solid_dark_fill_threshold
            and candidate.scale <= 0.50
            and group_values["color"] < 0.30
            and group_values["text"] > 0.60
            and group_values["edge"] < LOW_EDGE_GROUP_THRESHOLD
        ):
            risk += 0.35
        if (
            group_values["solid"] <= self.config.solid_dark_fill_threshold
            and group_values["local"] < 0.10
            and group_values["text"] > STRONG_TEXT_GROUP_THRESHOLD
            and group_values["edge"] < LOW_EDGE_GROUP_THRESHOLD
        ):
            risk += 0.45
        if (
            group_values["solid"] < 0.80
            and group_values["color"] < 0.30
            and group_values["local"] < STRONG_GROUP_THRESHOLD
            and group_values["text"] > 0.40
        ):
            risk += 0.55
        if group_values["fit"] < STRONG_GROUP_THRESHOLD and group_values["edge"] < LOW_EDGE_GROUP_THRESHOLD:
            risk += 0.65
        return float(risk)

    def _chromatic_text_candidate_risk(self, candidate: Candidate, evidence_profile: EvidenceProfile) -> float:
        if (
            evidence_profile.kind == "chromatic_text"
            and candidate.source == CANDIDATE_SOURCE_CHROMATIC_PRESENCE
            and candidate.scale < 0.65
        ):
            return 0.55
        return 0.0

    def _polarity_conflict_risk(
        self,
        candidate: Candidate,
        resized: ResizedTemplate,
        evidence: CandidateEvidence,
        evidence_profile: EvidenceProfile,
        fit: FitResult | None,
    ) -> float:
        if not _has_polarity_background_conflict(candidate, resized, evidence.support_mean, self.config):
            return 0.0
        return float(evidence_profile.risk_weights["polarity_conflict"] * (1.0 if fit is not None else 0.50))

    def _fit_risk(
        self,
        evidence_profile: EvidenceProfile,
        group_values: dict[str, float],
        fit: FitResult | None,
    ) -> float:
        if fit is None:
            return 0.0

        weights = evidence_profile.risk_weights
        risk = 0.0
        if group_values["fit"] < evidence_profile.fit_threshold:
            risk += weights["weak_fit"] * (
                1.0 - self._unit_interval(group_values["fit"], 0.0, evidence_profile.fit_threshold)
            )
        if fit.objective > 0.22:
            risk += weights["poor_fit"] * self._unit_interval(fit.objective, 0.22, 0.40)
        return float(risk)

    def _candidate_risk(
        self,
        candidate: Candidate,
        resized: ResizedTemplate,
        evidence: CandidateEvidence,
        metrics: CandidateDecisionMetrics,
        profile: TemplateProfile,
        evidence_profile: EvidenceProfile,
        group_values: dict[str, float],
        fit: FitResult | None,
    ) -> float:
        strong_sparse_color = (
            evidence_profile.kind == "sparse_chromatic"
            and group_values["color"] >= STRONG_COLOR_GROUP_THRESHOLD
        )
        strong_achromatic_structure = (
            evidence_profile.kind == "faint_sparse_achromatic"
            and candidate.source == CANDIDATE_SOURCE_FAINT_PRESENCE
            and group_values["shape"] >= STRONG_GROUP_THRESHOLD
            and group_values["edge"] >= EVIDENCE_GROUP_THRESHOLDS["edge"]
            and group_values["local"] >= STRONG_GROUP_THRESHOLD
        )
        strong_chromatic_text_match = (
            evidence_profile.kind == "chromatic_text"
            and candidate.source == CANDIDATE_SOURCE_MATCH
            and group_values["shape"] >= VERY_STRONG_GROUP_THRESHOLD
            and group_values["local"] >= CHROMATIC_TEXT_MATCH_LOCAL_THRESHOLD
            and group_values["color"] >= STRONG_GROUP_THRESHOLD
        )

        risk = self._scale_risk(candidate, evidence_profile)
        risk += self._required_group_risk(
            evidence_profile,
            group_values,
            strong_sparse_color=strong_sparse_color,
            strong_achromatic_structure=strong_achromatic_structure,
            strong_chromatic_text_match=strong_chromatic_text_match,
        )
        risk += self._generic_balance_risk(
            candidate,
            metrics,
            profile,
            evidence_profile,
            group_values,
            strong_sparse_color=strong_sparse_color,
        )
        risk += self._sparse_chromatic_risk(candidate, metrics, profile, evidence_profile, group_values)
        risk += self._faint_sparse_achromatic_risk(candidate, evidence, metrics, profile, evidence_profile, group_values, fit)
        risk += self._faint_dense_risk(candidate, evidence, metrics, profile, evidence_profile, group_values, fit)
        risk += self._solid_dark_risk(candidate, evidence, metrics, evidence_profile, group_values, fit)
        risk += self._chromatic_text_candidate_risk(candidate, evidence_profile)
        risk += self._polarity_conflict_risk(candidate, resized, evidence, evidence_profile, fit)
        risk += self._fit_risk(evidence_profile, group_values, fit)
        return float(risk)

    def _fit_quality(self, fit: FitResult | None) -> float:
        if fit is None:
            return 0.0
        objective = 1.0 - self._unit_interval(fit.objective, 0.02, 0.18)
        residual = 1.0 - self._unit_interval(fit.residual, 0.00, 0.10)
        clip = 1.0 - self._unit_interval(fit.clip_ratio, 0.00, 0.45)
        correlation = 1.0 - self._unit_interval(fit.watermark_correlation, 0.00, 0.45)
        return float(np.clip(0.40 * objective + 0.25 * residual + 0.20 * clip + 0.15 * correlation, 0.0, 1.0))

    def _candidate_decision_metrics(
        self,
        candidate: Candidate,
        patch: np.ndarray,
        resized: ResizedTemplate,
        color_score: float,
        template: TemplateBundle,
        evidence: CandidateEvidence,
    ) -> CandidateDecisionMetrics:
        evidence_delta = _watermark_evidence_delta(patch, resized)
        sparse_chromatic_template = self._template_uses_sparse_chromatic_presence(template)
        chroma_delta = _template_chroma_delta(patch, resized) if sparse_chromatic_template else 0.0
        support_ratio = float(template.support_mask.sum() / max(int(template.support_mask.size), 1))
        template_chroma = _template_chroma_strength(resized)
        solid_dark_template = self._template_has_solid_dark_region(template)
        solid_dark_evidence = False
        fill_ratio = 0.0
        hole_bright_ratio = 0.0
        if solid_dark_template:
            fill_ratio, hole_bright_ratio = _solid_dark_template_match_metrics(evidence.patch_gray, resized, self.config)
            solid_dark_evidence = bool(
                _has_solid_dark_template_evidence(evidence.patch_gray, resized, self.config)
                or (
                    fill_ratio >= self.config.solid_dark_fill_threshold
                    and evidence_delta >= self.config.foreground_residual_delta_threshold
                    and hole_bright_ratio >= self.config.solid_dark_hole_bright_threshold * 0.70
                )
                or (
                    fill_ratio >= max(self.config.solid_dark_fill_threshold, 0.80)
                    and hole_bright_ratio >= self.config.solid_dark_hole_bright_threshold * 0.43
                    and evidence.text_detail >= 0.45
                )
                or (
                    fill_ratio >= self.config.solid_dark_fill_threshold * 0.85
                    and hole_bright_ratio >= self.config.solid_dark_hole_bright_threshold * 0.43
                    and evidence.text_detail >= 0.55
                    and evidence_delta >= self.config.foreground_residual_delta_threshold
                )
            )
        foreground_geometry = solid_dark_evidence or _has_foreground_geometry_evidence(
            evidence.patch_gray,
            resized,
            candidate,
            evidence.support_corr,
            self.config,
        )
        low_score_evidence = _has_low_score_match_evidence(
            candidate,
            evidence.text_detail,
            evidence.support_corr,
            evidence_delta,
            self.config,
        )
        if not low_score_evidence and self._template_uses_sparse_achromatic_presence(template):
            low_score_evidence = self._has_sparse_achromatic_local_text_evidence(
                candidate,
                template,
                evidence,
                evidence_delta,
                foreground_geometry,
            )
        sparse_chromatic_color_edge = sparse_chromatic_template and _has_sparse_chromatic_color_edge_evidence(
            evidence,
            resized,
            color_score,
            self.config,
            candidate_scale=candidate.scale,
            candidate_edge_score=candidate.edge_score,
            chroma_delta=chroma_delta,
        )
        sparse_chromatic_spatial = sparse_chromatic_template and _has_sparse_chromatic_spatial_evidence(
            evidence,
            resized,
            color_score,
            evidence_delta,
            self.config,
            candidate_scale=candidate.scale,
            candidate_score=candidate.score,
            candidate_edge_score=candidate.edge_score,
            chroma_delta=chroma_delta,
        )
        return CandidateDecisionMetrics(
            evidence_delta=evidence_delta,
            chroma_delta=chroma_delta,
            color_score=color_score,
            support_ratio=support_ratio,
            template_chroma=template_chroma,
            solid_dark_fill_ratio=fill_ratio,
            solid_dark_hole_bright_ratio=hole_bright_ratio,
            foreground_geometry=foreground_geometry,
            low_score_evidence=low_score_evidence,
            sparse_chromatic_template=sparse_chromatic_template,
            sparse_chromatic_color_edge=sparse_chromatic_color_edge,
            sparse_chromatic_spatial=sparse_chromatic_spatial,
            solid_dark_template=solid_dark_template,
            solid_dark_evidence=solid_dark_evidence,
        )

    def _has_sparse_achromatic_local_text_evidence(
        self,
        candidate: Candidate,
        template: TemplateBundle,
        evidence: CandidateEvidence,
        evidence_delta: float,
        foreground_geometry: bool,
    ) -> bool:
        if candidate.scale < 0.85 or not foreground_geometry or evidence_delta < self.config.foreground_residual_delta_threshold:
            return False

        faint_presence_evidence = (
            candidate.source == CANDIDATE_SOURCE_FAINT_PRESENCE
            and candidate.score >= self.config.faint_presence_score_threshold
            and evidence.text_detail >= 0.85
            and (
                evidence.support_corr >= self.config.support_correlation_threshold * 0.75
                or candidate.edge_score >= self.config.edge_score_threshold * 0.55
            )
        )
        strong_local_presence_evidence = (
            template.polarity == "light"
            and candidate.source == CANDIDATE_SOURCE_FAINT_PRESENCE
            and candidate.score >= 0.50
            and candidate.edge_score >= self.config.edge_score_threshold * 0.80
        )
        weak_local_presence_evidence = (
            template.polarity == "light"
            and candidate.source == CANDIDATE_SOURCE_FAINT_PRESENCE
            and candidate.score >= 0.23
            and candidate.edge_score >= self.config.edge_score_threshold * 0.55
            and evidence.text_detail >= 0.62
            and evidence.support_corr >= self.config.support_correlation_threshold * 0.75
        )
        local_match_evidence = (
            candidate.source == CANDIDATE_SOURCE_MATCH
            and candidate.score >= self.config.candidate_threshold
            and candidate.edge_score >= self.config.edge_score_threshold * 0.70
            and evidence.text_detail >= 0.82
        )
        return bool(
            faint_presence_evidence
            or strong_local_presence_evidence
            or weak_local_presence_evidence
            or local_match_evidence
        )

    def _has_colored_solid_dark_alignment(
        self,
        candidate: Candidate,
        evidence: CandidateEvidence,
        metrics: CandidateDecisionMetrics,
    ) -> bool:
        aligned_shape = (
            evidence.support_corr >= self.config.support_correlation_threshold
            and metrics.color_score >= self.config.color_score_threshold - 0.05
        )
        strong_shape_color = (
            candidate.score >= self.config.score_threshold + 0.36
            and metrics.color_score >= self.config.color_score_threshold - 0.10
        )
        achromatic_banner = (
            candidate.score >= self.config.score_threshold + 0.36
            and evidence.text_detail >= 0.55
            and metrics.evidence_delta >= self.config.foreground_residual_delta_threshold
            and metrics.solid_dark_fill_ratio >= self.config.solid_dark_fill_threshold
            and metrics.solid_dark_hole_bright_ratio >= self.config.solid_dark_hole_bright_threshold * 0.90
        )
        strong_shape_only = evidence.support_corr >= self.config.foreground_strong_shape_correlation
        return bool(aligned_shape or strong_shape_color or achromatic_banner or strong_shape_only)

    def _refine_colored_solid_dark_candidate(
        self,
        image_rgb: np.ndarray,
        candidate: Candidate,
        template: TemplateBundle,
        color_score: float,
    ) -> tuple[Candidate, float, ResizedTemplate, np.ndarray]:
        resized = template.resized_to(candidate.width, candidate.height)
        patch = image_rgb[candidate.y : candidate.y + candidate.height, candidate.x : candidate.x + candidate.width].astype(np.float32)
        if not (
            self._template_has_solid_dark_region(template)
            and _template_chroma_strength(resized) > self.config.faint_presence_max_chroma
        ):
            return candidate, color_score, resized, patch

        best_candidate = candidate
        best_resized = resized
        best_patch = patch
        best_color_score = color_score
        base_quality = self._colored_solid_dark_refinement_quality(candidate, patch, resized, color_score)
        best_quality = base_quality
        image_height, image_width = image_rgb.shape[:2]
        scale_step = max(self.config.scale_step * 0.5, EPSILON)
        scale_values = [
            float(np.clip(candidate.scale + delta, self.config.min_scale, self.config.max_scale))
            for delta in (-self.config.scale_step, -scale_step, 0.0, scale_step, self.config.scale_step)
        ]
        for scale in sorted(set(scale_values)):
            scaled_width, scaled_height = template.size_for_scale(float(scale))
            if scaled_width >= image_width or scaled_height >= image_height:
                continue
            trial_resized = template.resized_to(scaled_width, scaled_height)
            for offset_y in (-18, -12, -6, 0, 6, 12, 18):
                y = int(np.clip(candidate.y + offset_y, 0, image_height - trial_resized.height))
                for offset_x in (-18, -12, -6, 0, 6, 12, 18, 24):
                    x = int(np.clip(candidate.x + offset_x, 0, image_width - trial_resized.width))
                    trial_candidate = replace(
                        candidate,
                        x=x,
                        y=y,
                        width=trial_resized.width,
                        height=trial_resized.height,
                        scale=float(scale),
                    )
                    trial_patch = image_rgb[y : y + trial_resized.height, x : x + trial_resized.width].astype(np.float32)
                    trial_color_score = _template_color_score(trial_patch, trial_resized)
                    quality = self._colored_solid_dark_refinement_quality(
                        trial_candidate,
                        trial_patch,
                        trial_resized,
                        trial_color_score,
                    )
                    if quality > best_quality:
                        best_candidate = trial_candidate
                        best_resized = trial_resized
                        best_patch = trial_patch
                        best_color_score = trial_color_score
                        best_quality = quality

        if best_quality < base_quality + 0.12:
            return candidate, color_score, resized, patch
        return best_candidate, best_color_score, best_resized, best_patch

    def _colored_solid_dark_refinement_quality(
        self,
        candidate: Candidate,
        patch: np.ndarray,
        resized: ResizedTemplate,
        color_score: float,
    ) -> float:
        del candidate
        evidence = _candidate_evidence(patch, resized)
        evidence_delta = _watermark_evidence_delta(patch, resized)
        return float(
            0.45 * color_score
            + 0.30 * self._unit_interval(evidence.support_corr, 0.0, self.config.foreground_strong_shape_correlation)
            + 0.25 * self._unit_interval(evidence_delta, 0.0, self.config.sparse_chromatic_evidence_rescue_delta)
        )

    @staticmethod
    def _unit_interval(value: float, low: float, high: float) -> float:
        if high <= low:
            return 0.0
        return float(np.clip((value - low) / (high - low), 0.0, 1.0))

    def _accept_candidate_fit(self, fit: FitResult, prefit: CandidatePrefit) -> bool:
        if fit.objective <= 0.18:
            return True
        return bool(fit.objective <= 0.28 and prefit.score >= 1.35)
