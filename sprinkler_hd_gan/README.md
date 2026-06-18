# Sprinkler layout GAN (pix2pixHD-style + hazard channel)

This repo implements the **Zeng et al. (2025)** paradigm—paired floorplan semantics to sprinkler/pipe rasters at **640×512** and **100 mm/pixel**—with deliberate upgrades:

1. **Dedicated hazard signal** — full-frame **one-hot** (default) or **scalar** channels so hazard class does not rely on outdoor background tint size (addresses §3.3 in the paper).
2. **Multi-scale discriminators + feature matching + L1**, following the pix2pixHD recipe (global generator; [NvLabs architecture reference](https://github.com/NVIDIA/pix2pixHD)).
3. **Synthetic paired data** for end-to-end smoke tests without proprietary drawings.
4. **Coverage / gap overlay** helpers using configurable equivalent protection radii (`configs/default.yaml` → `standards.hazard_radius_m`).

Paper: *AI-powered automatic design of fire sprinkler layout for random building floorplans* — [DOI 10.1016/j.iintel.2025.100167](https://doi.org/10.1016/j.iintel.2025.100167).

## GPU setup (Windows, NVIDIA)

The repo depends on PyTorch. If `python -c "import torch; print(torch.cuda.is_available())"` prints `False`, you likely have the **CPU-only** wheel. Reinstall the **CUDA** build (matches driver CUDA 12.x; RTX 30xx works with cu124):

```powershell
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Training prints `Device: CUDA ...` on startup when the GPU is used.

## Quickstart

```bash
cd sprinkler_hd_gan
python -m venv .venv
.venv\Scripts\activate
pip install -e .

# 1) Generate synthetic pairs
sprinkler-hd-synth --out data/synthetic --train 200 --val 20

# 2) Train (GPU recommended)
sprinkler-hd-train --data data/synthetic --out runs/demo --epochs 5

# 3) Infer
sprinkler-hd-infer --checkpoint runs/demo/epoch_0005.pt --input data/synthetic/val/input/00000.png --meta data/synthetic/val/meta/00000.yaml --out out/pred.png
```

For step 3, use any matching `val/input/*.png` and `val/meta/*.yaml` pair.

### IFC → semantic PNG (e.g. `გარემო.ifc`, floor -2)

Install IFC extras, then export a storey to `semantic.png` + `meta.yaml`:

```powershell
pip install -e ".[ifc]"
sprinkler-hd-ifc-export --ifc path\to\გარემო.ifc --storey "-2" --out out\my_floor
# or exact name:
sprinkler-hd-ifc-export --ifc path\to\გარემო.ifc --storey "-2. Story" --out out\my_floor
sprinkler-hd-infer --checkpoint runs\demo\epoch_0005.pt --input out\my_floor\semantic.png --meta out\my_floor\meta.yaml --out out\my_floor\pred.png
```

`--list-storeys` prints all `IfcBuildingStorey` names in the file. Short hints like `-2` resolve to names containing `-2` (e.g. `-2. Story`).

## Data layout (real projects)

```
dataset_root/
  train/
    input/*.png     # semantic BGR (paper legend + optional outdoor tint)
    target/*.png    # engineer-style raster
    meta/*.yaml     # hazard: low | medium | high
  val/ ... same
```

`hazard_encoding.mode` in `configs/default.yaml` must stay **`onehot`** or **`scalar`** consistently across preprocess → train → infer.

## Hazard encoding modes

- **`onehot`**: 3 extra channels `[1,0,0] / [0,1,0] / [0,0,1]` tiled over the canvas (clearest signal).
- **`scalar`**: 1 channel with `0.0 / 0.5 / 1.0` (compact).

Optional legacy **`paper_background_tint_for_film`**: paint outdoor regions as in the paper *in addition* to the dedicated channels.

## NVIDIA pix2pixHD upstream

To train with the official implementation instead, align folder layout to `datasets/` expectations from [NVIDIA/pix2pixHD](https://github.com/NVIDIA/pix2pixHD) and set generator input channels to **3 + hazard channels**.
