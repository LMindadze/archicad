from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np

from sprinkler_hd_gan.semantics import HazardClass, sprinkled_area_mask


@dataclass
class CoverageReport:
    coverage_fraction: float
    sprinkled_area_px: int
    covered_px: int
    head_count: int

    @property
    def coverage_percent(self) -> float:
        return 100.0 * self.coverage_fraction


def estimate_heads_from_target_bgr(
    target_bgr: np.ndarray,
    *,
    h_min: int = 80,
    s_max: int = 200,
    v_min: int = 40,
    v_max: int = 220,
) -> list[tuple[int, int]]:
    """Very rough blob centroids for synthetic-style red heads in HSV."""
    hsv = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    mask = (h >= h_min) & (h <= 140) & (s >= 40) & (s <= s_max) & (v >= v_min) & (v <= v_max)
    mask_u8 = (mask.astype(np.uint8) * 255)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pts: list[tuple[int, int]] = []
    for c in cnts:
        m = cv2.moments(c)
        if m["m00"] <= 1e-6:
            continue
        cx = int(m["m10"] / m["m00"])
        cy = int(m["m01"] / m["m00"])
        pts.append((cx, cy))
    return pts


def protection_radius_px(hazard: HazardClass, mm_per_pixel: float, radius_m: dict[str, float]) -> float:
    key = {HazardClass.LOW: "low", HazardClass.MEDIUM: "medium", HazardClass.HIGH: "high"}[hazard]
    meters = float(radius_m[key])
    return (meters * 1000.0) / mm_per_pixel


def coverage_from_heads(
    semantic_bgr: np.ndarray,
    heads_xy: Iterable[tuple[int, int]],
    hazard: HazardClass,
    *,
    mm_per_pixel: float,
    hazard_radius_m: dict[str, float],
) -> CoverageReport:
    mask = sprinkled_area_mask(semantic_bgr)
    sprinkled_px = int(mask.sum())
    if sprinkled_px == 0:
        return CoverageReport(1.0, 0, 0, len(list(heads_xy)))

    cov = np.zeros_like(mask, dtype=np.uint8)
    r = protection_radius_px(hazard, mm_per_pixel, hazard_radius_m)
    for x, y in heads_xy:
        cv2.circle(cov, (int(x), int(y)), int(round(r)), 1, thickness=-1, lineType=cv2.LINE_AA)

    covered = int(np.logical_and(mask, cov > 0).sum())
    frac = covered / sprinkled_px if sprinkled_px else 1.0
    heads_list = list(heads_xy)
    return CoverageReport(frac, sprinkled_px, covered, len(heads_list))


def highlight_gaps_bgr(
    semantic_bgr: np.ndarray,
    heads_xy: Iterable[tuple[int, int]],
    hazard: HazardClass,
    *,
    mm_per_pixel: float,
    hazard_radius_m: dict[str, float],
) -> np.ndarray:
    """Orange overlay on sprinkled pixels lacking coverage (paper-style quick check)."""
    mask = sprinkled_area_mask(semantic_bgr)
    cov = np.zeros_like(mask, dtype=np.uint8)
    r = protection_radius_px(hazard, mm_per_pixel, hazard_radius_m)
    for x, y in heads_xy:
        cv2.circle(cov, (int(x), int(y)), int(round(r)), 1, thickness=-1, lineType=cv2.LINE_AA)

    gap = np.logical_and(mask, cov == 0)
    vis = semantic_bgr.copy()
    orange = (0, 165, 255)  # BGR
    vis[gap] = orange
    return vis
