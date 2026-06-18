from __future__ import annotations

from enum import IntEnum
from typing import Tuple

import numpy as np


class HazardClass(IntEnum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2


# BGR for OpenCV-style arrays (H, W, 3), uint8 — paper Fig.3 style.
class SemanticRGB:
    SPRINKLERED_SPACE = (255, 255, 255)
    NON_SPRINKLERED = (180, 180, 180)
    RISER_SHAFT = (255, 0, 0)  # "blue" in paper BGR = (255,0,0)? Paper says blue: in RGB blue=(0,0,255) -> BGR (255,0,0)
    OFFICE_ZONE = (0, 255, 200)
    CORRIDOR = (0, 200, 100)
    CARPARK = (100, 100, 220)


def paper_outdoor_tint_bgr(hazard: HazardClass) -> Tuple[int, int, int]:
    """Paper encodes hazard in outdoor background: green low, blue medium, orange high (RGB)."""
    if hazard == HazardClass.LOW:
        return (0, 255, 0)  # green BGR
    if hazard == HazardClass.MEDIUM:
        return (255, 0, 0)  # blue in RGB -> BGR
    return (0, 165, 255)  # orange-ish in BGR


def rgb_mask_equal(image_bgr: np.ndarray, color_bgr: Tuple[int, int, int]) -> np.ndarray:
    c = np.array(color_bgr, dtype=np.uint8).reshape(1, 1, 3)
    return np.all(image_bgr == c, axis=-1)


def sprinkled_area_mask(image_bgr: np.ndarray) -> np.ndarray:
    return rgb_mask_equal(image_bgr, SemanticRGB.SPRINKLERED_SPACE)


def hazard_onehot_maps(hazard: HazardClass, height: int, width: int) -> np.ndarray:
    """Shape (3, H, W) float32 {0,1}."""
    m = np.zeros((3, height, width), dtype=np.float32)
    m[int(hazard), :, :] = 1.0
    return m


def hazard_scalar_map(hazard: HazardClass, height: int, width: int) -> np.ndarray:
    """Single channel full-frame constant: 0.0 / 0.5 / 1.0."""
    v = {HazardClass.LOW: 0.0, HazardClass.MEDIUM: 0.5, HazardClass.HIGH: 1.0}[hazard]
    return np.full((1, height, width), v, dtype=np.float32)


def stack_model_input(
    semantic_bgr: np.ndarray,
    hazard: HazardClass,
    *,
    hazard_mode: str,
    paper_background_tint: bool,
) -> np.ndarray:
    """
    Returns float32 tensor CxHxW in [0,1], BGR first then hazard channel(s).
    hazard_mode: 'scalar' | 'onehot'
    """
    h, w = semantic_bgr.shape[:2]
    rgb = semantic_bgr.astype(np.float32) / 255.0
    chw = np.transpose(rgb, (2, 0, 1))
    if paper_background_tint:
        # Callers should paint outdoor regions using `paper_outdoor_tint_bgr` before stacking;
        # the dedicated hazard channel(s) remain the primary signal.
        pass
    if hazard_mode == "scalar":
        haz = hazard_scalar_map(hazard, h, w)
        return np.concatenate([chw, haz], axis=0)
    if hazard_mode == "onehot":
        haz = hazard_onehot_maps(hazard, h, w)
        return np.concatenate([chw, haz], axis=0)
    raise ValueError(f"Unknown hazard_mode: {hazard_mode}")
