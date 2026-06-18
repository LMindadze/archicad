from __future__ import annotations

import random
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import yaml

from sprinkler_hd_gan.semantics import (
    HazardClass,
    SemanticRGB,
    paper_outdoor_tint_bgr,
    sprinkled_area_mask,
)


def _draw_floorplate(
    canvas: np.ndarray,
    poly_xy: List[Tuple[int, int]],
    outdoor_bgr: Tuple[int, int, int],
) -> None:
    pts = np.array(poly_xy, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(canvas, [pts], outdoor_bgr)
    cv2.fillPoly(canvas, [pts], SemanticRGB.SPRINKLERED_SPACE, lineType=cv2.LINE_AA)


def _draw_rect_room(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, color: tuple) -> None:
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=-1)


def _spacing_px_for_hazard(hazard: HazardClass, mm_per_pixel: float) -> float:
    # Illustrative max spacing (mm) -> pixel spacing for synthetic grid.
    max_spacing_mm = {HazardClass.LOW: 4400, HazardClass.MEDIUM: 3600, HazardClass.HIGH: 3000}[hazard]
    return max_spacing_mm / mm_per_pixel


def _grid_centers(mask: np.ndarray, spacing_px: float, jitter: float = 0.12) -> list[tuple[int, int]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return []
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    pts: list[tuple[int, int]] = []
    y = float(y0) + spacing_px / 2
    while y <= y1:
        x = float(x0) + spacing_px / 2
        while x <= x1:
            xi, yi = int(round(x)), int(round(y))
            if 0 <= yi < mask.shape[0] and 0 <= xi < mask.shape[1] and mask[yi, xi]:
                jx = (random.random() * 2 - 1) * jitter * spacing_px
                jy = (random.random() * 2 - 1) * jitter * spacing_px
                pts.append((int(round(xi + jx)), int(round(yi + jy))))
            x += spacing_px
        y += spacing_px
    return pts


def _draw_heads(target: np.ndarray, centers: list[tuple[int, int]], color: tuple[int, int, int]) -> None:
    r = 3
    for x, y in centers:
        cv2.circle(target, (x, y), r, color, thickness=-1, lineType=cv2.LINE_AA)


def _draw_subpipes(
    target: np.ndarray,
    centers: list[tuple[int, int]],
    color: tuple[int, int, int],
    *,
    max_neighbors: int = 4,
    max_dist_px: float,
) -> None:
    if not centers:
        return
    pts = np.array(centers, dtype=np.float32)
    for i, p in enumerate(centers):
        d = np.linalg.norm(pts - p, axis=1)
        d[i] = 1e9
        order = np.argsort(d)
        for j in order[:max_neighbors]:
            if d[j] <= max_dist_px:
                q = centers[int(j)]
                cv2.line(target, (p[0], p[1]), (q[0], q[1]), color, thickness=2, lineType=cv2.LINE_AA)


def _nearest_to_riser(centers: list[tuple[int, int]], riser_xy: tuple[int, int]) -> tuple[int, int]:
    bx, by = riser_xy
    best = centers[0]
    bd = 1e18
    for x, y in centers:
        d = (x - bx) ** 2 + (y - by) ** 2
        if d < bd:
            bd = d
            best = (x, y)
    return best


def _draw_main_stub(target: np.ndarray, centers: list[tuple[int, int]], riser_xy: tuple[int, int]) -> None:
    if not centers:
        return
    hub = _nearest_to_riser(centers, riser_xy)
    main_color = (0, 140, 255)
    cv2.line(target, riser_xy, hub, main_color, thickness=3, lineType=cv2.LINE_AA)


def generate_one_synthetic_pair(
    width: int,
    height: int,
    hazard: HazardClass,
    *,
    mm_per_pixel: float = 100.0,
    use_paper_outdoor_tint: bool = True,
    seed: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, HazardClass]:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    outdoor_bgr = paper_outdoor_tint_bgr(hazard) if use_paper_outdoor_tint else (40, 40, 40)

    semantic = np.zeros((height, width, 3), dtype=np.uint8)
    target = np.zeros((height, width, 3), dtype=np.uint8)

    plate = [(40, 40), (width - 40, 60), (width - 60, height - 80), (80, height - 50)]
    _draw_floorplate(semantic, plate, outdoor_bgr)

    # Shafts / rooms
    _draw_rect_room(semantic, 120, 120, 200, 280, SemanticRGB.NON_SPRINKLERED)
    _draw_rect_room(semantic, 220, 340, 520, 430, SemanticRGB.CARPARK)
    _draw_rect_room(semantic, width // 2, 90, width - 120, 240, SemanticRGB.OFFICE_ZONE)

    riser_xy = (95, height // 2)
    cv2.rectangle(semantic, (riser_xy[0] - 18, riser_xy[1] - 18), (riser_xy[0] + 18, riser_xy[1] + 18), SemanticRGB.RISER_SHAFT, -1)

    mask = sprinkled_area_mask(semantic).astype(np.uint8)
    spacing_px = _spacing_px_for_hazard(hazard, mm_per_pixel)
    centers = _grid_centers(mask, spacing_px=spacing_px)

    head_color = (200, 80, 80)
    sub_color = (180, 180, 60)
    _draw_heads(target, centers, head_color)
    _draw_subpipes(target, centers, sub_color, max_dist_px=spacing_px * 1.15)
    _draw_main_stub(target, centers, riser_xy)

    return semantic, target, hazard


def write_synthetic_dataset(
    out_root: Path,
    n_train: int,
    n_val: int,
    *,
    width: int,
    height: int,
    mm_per_pixel: float,
    hazard_mode_weights: Optional[dict[str, float]] = None,
) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    splits = [("train", n_train), ("val", n_val)]

    if hazard_mode_weights is None:
        hazard_mode_weights = {"low": 0.25, "medium": 0.5, "high": 0.25}

    def _hazard_from_key(k: str) -> HazardClass:
        m = {"low": HazardClass.LOW, "medium": HazardClass.MEDIUM, "high": HazardClass.HIGH}
        return m[k]

    def sample_hazard() -> HazardClass:
        r = random.random()
        acc = 0.0
        for k, w in hazard_mode_weights.items():
            acc += w
            if r <= acc:
                return _hazard_from_key(k)
        return HazardClass.MEDIUM

    idx = 0
    for split, count in splits:
        d_in = out_root / split / "input"
        d_tgt = out_root / split / "target"
        d_meta = out_root / split / "meta"
        d_in.mkdir(parents=True, exist_ok=True)
        d_tgt.mkdir(parents=True, exist_ok=True)
        d_meta.mkdir(parents=True, exist_ok=True)

        for _ in range(count):
            hz = sample_hazard()
            sem, tgt, hz2 = generate_one_synthetic_pair(
                width,
                height,
                hz,
                mm_per_pixel=mm_per_pixel,
                use_paper_outdoor_tint=True,
                seed=idx,
            )
            stem = f"{idx:05d}"
            cv2.imwrite(str(d_in / f"{stem}.png"), sem)
            cv2.imwrite(str(d_tgt / f"{stem}.png"), tgt)
            (d_meta / f"{stem}.yaml").write_text(
                yaml.safe_dump({"hazard": hz2.name.lower()}, sort_keys=False),
                encoding="utf-8",
            )
            idx += 1
