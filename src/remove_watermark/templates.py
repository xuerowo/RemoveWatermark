from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Sequence

import cv2
import numpy as np
from PIL import Image, ImageOps

from .models import IMAGE_EXTENSIONS


@dataclass(slots=True)
class ResizedTemplate:
    width: int
    height: int
    template_rgb: np.ndarray
    background_rgb: np.ndarray
    rgb: np.ndarray
    gray: np.ndarray
    gray_u8: np.ndarray
    alpha: np.ndarray
    alpha_rgb: np.ndarray
    support_mask: np.ndarray
    edge: np.ndarray
    polarity: str
    chroma_strength: float


@dataclass(slots=True)
class TemplateBundle:
    path: Path
    name: str
    template_rgb: np.ndarray
    background_rgb: np.ndarray
    rgb: np.ndarray
    gray: np.ndarray
    gray_u8: np.ndarray
    soft_alpha: np.ndarray
    soft_alpha_rgb: np.ndarray
    support_mask: np.ndarray
    edge: np.ndarray
    polarity: str
    chroma_strength: float
    _cache: dict[tuple[int, int], ResizedTemplate] = field(default_factory=dict, repr=False)
    _cache_lock: RLock = field(default_factory=RLock, repr=False, compare=False)

    @property
    def size(self) -> tuple[int, int]:
        return int(self.gray.shape[1]), int(self.gray.shape[0])

    def size_for_scale(self, scale: float) -> tuple[int, int]:
        return max(8, int(round(self.size[0] * scale))), max(8, int(round(self.size[1] * scale)))

    def resized_for_scale(self, scale: float) -> ResizedTemplate:
        width, height = self.size_for_scale(scale)
        return self.resized_to(width, height)

    def resized_to(self, width: int, height: int) -> ResizedTemplate:
        key = (int(width), int(height))
        with self._cache_lock:
            cached = self._cache.get(key)
            if cached is not None:
                return cached

            interpolation = cv2.INTER_AREA if key[0] <= self.size[0] else cv2.INTER_CUBIC
            template_rgb = cv2.resize(self.template_rgb, key, interpolation=interpolation).astype(np.float32)
            rgb = cv2.resize(self.rgb, key, interpolation=interpolation).astype(np.float32)
            gray = cv2.resize(self.gray, key, interpolation=interpolation).astype(np.float32)
            alpha_rgb = cv2.resize(self.soft_alpha_rgb, key, interpolation=interpolation).astype(np.float32)
            alpha = cv2.resize(self.soft_alpha, key, interpolation=interpolation).astype(np.float32)
            support = cv2.resize(self.support_mask, key, interpolation=cv2.INTER_NEAREST).astype(np.uint8)
            resized = ResizedTemplate(
                width=key[0],
                height=key[1],
                template_rgb=np.clip(template_rgb, 0.0, 255.0).astype(np.float32),
                background_rgb=self.background_rgb.astype(np.float32),
                rgb=np.clip(rgb, 0.0, 255.0).astype(np.float32),
                gray=np.clip(gray, 0.0, 255.0).astype(np.float32),
                gray_u8=np.clip(gray, 0.0, 255.0).astype(np.uint8),
                alpha=np.clip(alpha, 0.0, 1.0).astype(np.float32),
                alpha_rgb=np.clip(alpha_rgb, 0.0, 1.0).astype(np.float32),
                support_mask=support,
                edge=_compute_edge_map(gray),
                polarity=self.polarity,
                chroma_strength=self.chroma_strength,
            )
            self._cache[key] = resized
            return resized

def load_template(path: str | Path) -> TemplateBundle:
    template_path = Path(path)
    with Image.open(template_path) as image:
        has_transparency = "A" in image.getbands() or "transparency" in image.info
        template_rgba = np.asarray(image.convert("RGBA"), dtype=np.float32)
    template_rgb = template_rgba[..., :3]
    template_alpha = template_rgba[..., 3] / 255.0
    alpha_defines_shape = has_transparency and float(template_alpha.min()) < 1.0
    if alpha_defines_shape:
        return _load_alpha_template(template_path, template_rgb, template_alpha)

    template_background = _estimate_template_background(template_rgb)
    background_gray = float(_rgb_to_gray(template_background.reshape(1, 1, 3).astype(np.uint8))[0, 0])
    polarity = "dark" if background_gray >= 160.0 else "light"
    reference = template_background.reshape(1, 1, 3)
    appearance_rgb = np.clip(reference - template_rgb, 0.0, 255.0) if polarity == "dark" else np.clip(template_rgb - reference, 0.0, 255.0)
    gray = _rgb_to_gray(np.clip(appearance_rgb, 0.0, 255.0).astype(np.uint8)).astype(np.float32)
    support_mask = build_template_support_mask(gray, polarity)
    return TemplateBundle(
        path=template_path,
        name=template_path.stem,
        template_rgb=np.clip(template_rgb, 0.0, 255.0).astype(np.float32),
        background_rgb=np.clip(template_background, 0.0, 255.0).astype(np.float32),
        rgb=np.clip(appearance_rgb, 0.0, 255.0).astype(np.float32),
        gray=gray,
        gray_u8=np.clip(gray, 0.0, 255.0).astype(np.uint8),
        soft_alpha=np.clip(gray / 255.0, 0.0, 1.0).astype(np.float32),
        soft_alpha_rgb=np.clip(appearance_rgb / 255.0, 0.0, 1.0).astype(np.float32),
        support_mask=support_mask,
        edge=_compute_edge_map(gray),
        polarity=polarity,
        chroma_strength=float(np.std(_color_deviation(appearance_rgb))),
    )


def _load_alpha_template(template_path: Path, template_rgb: np.ndarray, template_alpha: np.ndarray) -> TemplateBundle:
    alpha_gray = np.clip(template_alpha * 255.0, 0.0, 255.0).astype(np.float32)
    support_mask = build_template_support_mask(alpha_gray, "light")
    visible = support_mask.astype(bool)
    if not np.any(visible):
        visible = template_alpha > 0.01
    visible_gray = _rgb_to_gray(np.clip(template_rgb, 0.0, 255.0).astype(np.uint8)).astype(np.float32)
    weights = template_alpha[visible]
    visible_mean = float(np.average(visible_gray[visible], weights=weights)) if float(weights.sum()) > 0.0 else 255.0
    polarity = "dark" if visible_mean < 160.0 else "light"
    template_background = np.array([255.0, 255.0, 255.0] if polarity == "dark" else [0.0, 0.0, 0.0], dtype=np.float32)
    reference = template_background.reshape(1, 1, 3)
    if polarity == "dark":
        appearance_rgb = np.clip(reference - template_rgb, 0.0, 255.0) * template_alpha[..., None]
    else:
        appearance_rgb = np.clip(template_rgb - reference, 0.0, 255.0) * template_alpha[..., None]
    appearance_rgb = np.where(template_alpha[..., None] > 0.0, appearance_rgb, 0.0).astype(np.float32)
    return TemplateBundle(
        path=template_path,
        name=template_path.stem,
        template_rgb=np.clip(template_rgb, 0.0, 255.0).astype(np.float32),
        background_rgb=template_background,
        rgb=np.clip(appearance_rgb, 0.0, 255.0).astype(np.float32),
        gray=alpha_gray,
        gray_u8=np.clip(alpha_gray, 0.0, 255.0).astype(np.uint8),
        soft_alpha=np.clip(template_alpha, 0.0, 1.0).astype(np.float32),
        soft_alpha_rgb=np.repeat(np.clip(template_alpha, 0.0, 1.0)[..., None], 3, axis=2).astype(np.float32),
        support_mask=support_mask,
        edge=_compute_edge_map(alpha_gray),
        polarity=polarity,
        chroma_strength=float(np.std(_color_deviation(appearance_rgb))),
    )


def load_templates(paths: str | Path | Sequence[str | Path]) -> list[TemplateBundle]:
    raw_paths = [paths] if isinstance(paths, (str, Path)) else paths
    template_paths: list[Path] = []
    for raw_path in raw_paths:
        path = Path(raw_path)
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            template_paths.append(path)
        elif path.is_dir():
            template_paths.extend(_template_image_candidates(path))
    return [load_template(path) for path in dict.fromkeys(template_paths)]


def _template_image_candidates(path: Path) -> list[Path]:
    images = sorted(file for file in path.iterdir() if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS)
    if images:
        return images
    nested = path / "templates"
    return _template_image_candidates(nested) if nested.is_dir() else []


def list_input_images(input_path: str | Path) -> list[Path]:
    path = Path(input_path)
    if path.is_file():
        return [path] if path.suffix.lower() in IMAGE_EXTENSIONS else []
    if not path.is_dir():
        return []
    return sorted(image for image in path.iterdir() if image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS)


def load_image_rgb(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(ImageOps.exif_transpose(image).convert("RGB"), dtype=np.uint8)


def save_image_rgb(path: str | Path, image_rgb: np.ndarray) -> None:
    Image.fromarray(np.clip(image_rgb, 0, 255).astype(np.uint8), mode="RGB").save(path)


def save_image_like(path: str | Path, image_rgb: np.ndarray, reference_path: str | Path) -> None:
    rgb_image = Image.fromarray(np.clip(image_rgb, 0, 255).astype(np.uint8), mode="RGB")
    with Image.open(reference_path) as reference:
        reference = ImageOps.exif_transpose(reference)
        mode = reference.mode
        alpha = reference.getchannel("A").copy() if "A" in reference.getbands() else None
    if mode == "RGBA" and alpha is not None:
        output = rgb_image.convert("RGBA")
        output.putalpha(alpha)
    elif mode == "LA" and alpha is not None:
        output = Image.merge("LA", (rgb_image.convert("L"), alpha))
    else:
        try:
            output = rgb_image if mode == "RGB" else rgb_image.convert(mode)
        except ValueError:
            output = rgb_image
    output.save(path)


def _rgb_to_gray(image_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.clip(image_rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)


def _estimate_template_background(template_rgb: np.ndarray) -> np.ndarray:
    border = np.concatenate([template_rgb[0], template_rgb[-1], template_rgb[:, 0], template_rgb[:, -1]], axis=0)
    return np.median(border, axis=0).astype(np.float32)


def build_template_support_mask(gray_image: np.ndarray, polarity: str) -> np.ndarray:
    gray = np.clip(gray_image.astype(np.float32), 0.0, 255.0)
    if gray.size == 0 or float(gray.max()) <= 0.0:
        return np.zeros(gray.shape, dtype=np.uint8)
    nonzero = gray[gray > 0]
    adaptive = float(np.percentile(nonzero, 70)) * 0.35 if nonzero.size else 0.0
    threshold = max(3.0 if polarity == "light" else 8.0, float(gray.max()) * 0.15, adaptive)
    mask = (gray > threshold).astype(np.uint8)
    return _clean_mask(mask, min_area=_min_component_area(mask, ratio=0.00035))


def _binary_mask(mask: np.ndarray) -> np.ndarray:
    return (mask > 0).astype(np.uint8)


def _min_component_area(mask: np.ndarray, ratio: float = 0.0005) -> int:
    return max(4, int(round(mask.size * ratio)))


def _close_mask(mask: np.ndarray, pixels: float, shape: int = cv2.MORPH_ELLIPSE) -> np.ndarray:
    mask_u8 = _binary_mask(mask)
    if not np.any(mask_u8) or pixels <= 0.0:
        return mask_u8
    kernel_size = _odd_kernel_size(max(3, int(round(pixels * 2.0 + 1.0))))
    return cv2.morphologyEx(
        mask_u8,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(shape, (kernel_size, kernel_size)),
        iterations=1,
    )


def _odd_kernel_size(value: int) -> int:
    return int(value + 1 if value % 2 == 0 else value)


def _clean_mask(
    mask: np.ndarray,
    *,
    min_area: int | None = None,
    fill_holes: bool = False,
    restrict_to: np.ndarray | None = None,
) -> np.ndarray:
    cleaned = _binary_mask(mask)
    if fill_holes:
        cleaned = _filled_support_holes(cleaned)
    if restrict_to is not None:
        cleaned = (cleaned.astype(bool) & _binary_mask(restrict_to).astype(bool)).astype(np.uint8)
    return _remove_small_mask_components(cleaned, min_area=_min_component_area(cleaned) if min_area is None else min_area)


def _remove_small_mask_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    mask_u8 = _binary_mask(mask)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if count <= 1:
        return mask_u8
    cleaned = np.zeros_like(mask_u8)
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= min_area:
            cleaned[labels == label] = 1
    return cleaned.astype(np.uint8)


def _filled_support_holes(mask: np.ndarray) -> np.ndarray:
    base = _binary_mask(mask)
    if not np.any(base):
        return base
    inverted = np.pad(1 - base, 1, constant_values=1).astype(np.uint8)
    flood = inverted.copy()
    cv2.floodFill(flood, None, (0, 0), 0)
    holes = flood[1:-1, 1:-1].astype(bool)
    return (base.astype(bool) | holes).astype(np.uint8)


def _compute_edge_map(gray: np.ndarray) -> np.ndarray:
    return (cv2.Canny(np.clip(gray, 0, 255).astype(np.uint8), 20, 80).astype(np.float32) / 255.0).astype(np.float32)


def _color_deviation(image_rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(image_rgb, dtype=np.float32)
    return rgb - rgb.mean(axis=2, keepdims=True)

