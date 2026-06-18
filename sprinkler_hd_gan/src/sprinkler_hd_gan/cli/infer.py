from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import torch
import yaml

from sprinkler_hd_gan.models.networks import GlobalGenerator
from sprinkler_hd_gan.semantics import HazardClass, stack_model_input


def _load_hazard_dict(meta_yaml: Path) -> HazardClass:
    data = yaml.safe_load(meta_yaml.read_text(encoding="utf-8"))
    key = str(data.get("hazard", "medium")).lower()
    return {"low": HazardClass.LOW, "medium": HazardClass.MEDIUM, "high": HazardClass.HIGH}[key]


def main() -> None:
    p = argparse.ArgumentParser(description="Run trained generator on one floorplan image.")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True, help="Semantic input PNG (BGR).")
    p.add_argument("--meta", type=Path, required=True, help="YAML with hazard: low|medium|high")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Full dict checkpoint (weights + cfg); not weights-only tensors.
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    hazard_mode = str(ckpt.get("hazard_mode", "onehot"))
    in_ch = int(ckpt["in_ch"])

    cfg = ckpt.get("cfg") or {}
    haz_cfg = cfg.get("hazard_encoding") or {}
    paper_tint = bool(haz_cfg.get("paper_background_tint_for_film", False))

    sem = cv2.imread(str(args.input), cv2.IMREAD_COLOR)
    if sem is None:
        raise FileNotFoundError(args.input)

    hazard = _load_hazard_dict(args.meta)
    x = stack_model_input(sem, hazard, hazard_mode=hazard_mode, paper_background_tint=paper_tint)
    xt = torch.from_numpy(x).unsqueeze(0).to(device)

    tcfg = cfg.get("training") or {}
    g = GlobalGenerator(
        in_ch,
        3,
        ngf=int(tcfg.get("ngf", 64)),
        n_downsampling=int(tcfg.get("n_downsample_global", 4)),
        n_blocks=int(tcfg.get("n_blocks_global", 9)),
    ).to(device)
    g.load_state_dict(ckpt["generator"])
    g.eval()

    with torch.no_grad():
        y = g(xt)
    y = (y.clamp(-1, 1) + 1.0) * 0.5
    y = y[0].detach().cpu().numpy().transpose(1, 2, 0)
    y = (y * 255.0).clip(0, 255).astype("uint8")
    y_bgr = y

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), y_bgr)


if __name__ == "__main__":
    main()
