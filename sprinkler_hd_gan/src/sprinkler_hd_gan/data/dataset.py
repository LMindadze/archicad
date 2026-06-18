from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset

from sprinkler_hd_gan.semantics import HazardClass, stack_model_input


def _load_hazard(meta_path: Path) -> HazardClass:
    data = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
    key = str(data.get("hazard", "medium")).lower()
    return {"low": HazardClass.LOW, "medium": HazardClass.MEDIUM, "high": HazardClass.HIGH}[key]


def _apply_geom_aug(
    x: np.ndarray,
    y: np.ndarray,
    *,
    hflip: bool,
    vflip: bool,
    rot180: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if hflip and np.random.rand() < 0.5:
        x = np.flip(x, axis=2).copy()
        y = np.flip(y, axis=2).copy()
    if vflip and np.random.rand() < 0.5:
        x = np.flip(x, axis=1).copy()
        y = np.flip(y, axis=1).copy()
    if rot180 and np.random.rand() < 0.5:
        x = np.rot90(x, k=2, axes=(1, 2)).copy()
        y = np.rot90(y, k=2, axes=(1, 2)).copy()
    return x, y


class SprinklerLayoutDataset(Dataset):
    """
    Expects:
      root/train|val/input/*.png
      root/train|val/target/*.png
      root/train|val/meta/*.yaml  (hazard: low|medium|high)
    """

    def __init__(
        self,
        root: Path,
        split: str,
        *,
        hazard_mode: str,
        paper_background_tint: bool,
        augment: dict[str, bool],
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.hazard_mode = hazard_mode
        self.paper_background_tint = paper_background_tint
        self.augment = augment

        in_dir = self.root / split / "input"
        self.ids = sorted(p.stem for p in in_dir.glob("*.png"))
        if not self.ids:
            raise FileNotFoundError(f"No input PNGs in {in_dir}")

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int) -> dict[str, Any]:
        stem = self.ids[i]
        sem = cv2.imread(str(self.root / self.split / "input" / f"{stem}.png"), cv2.IMREAD_COLOR)
        tgt = cv2.imread(str(self.root / self.split / "target" / f"{stem}.png"), cv2.IMREAD_COLOR)
        if sem is None or tgt is None:
            raise FileNotFoundError(f"Missing pair for {stem}")

        hazard = _load_hazard(self.root / self.split / "meta" / f"{stem}.yaml")

        x = stack_model_input(
            sem,
            hazard,
            hazard_mode=self.hazard_mode,
            paper_background_tint=self.paper_background_tint,
        )
        y = np.transpose(tgt.astype(np.float32) / 255.0, (2, 0, 1))

        if self.split == "train" and self.augment:
            x, y = _apply_geom_aug(
                x,
                y,
                hflip=bool(self.augment.get("hflip", False)),
                vflip=bool(self.augment.get("vflip", False)),
                rot180=bool(self.augment.get("rot180", False)),
            )

        return {
            "input": torch.from_numpy(x),
            "target": torch.from_numpy(y),
            "id": stem,
            "hazard": int(hazard),
        }
