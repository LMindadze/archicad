from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import yaml

from sprinkler_hd_gan.semantics import HazardClass
from sprinkler_hd_gan.ifc_semantic import export_storey_semantic, list_storey_names


def main() -> None:
    p = argparse.ArgumentParser(
        description="Rasterize an IFC building storey into GAN semantic PNG (+ meta.yaml). "
        "Requires: pip install -e \".[ifc]\""
    )
    p.add_argument("--ifc", type=Path, required=True, help="Path to .ifc (e.g. გარემო.ifc).")
    p.add_argument(
        "--storey",
        type=str,
        default="-2",
        help='IfcBuildingStorey match: exact name, or short hints like "-2" / "-2 floor" (default: -2).',
    )
    p.add_argument("--list-storeys", action="store_true", help="Print storey names and exit.")
    p.add_argument("--out", type=Path, default=Path("out/ifc_semantic"), help="Output directory.")
    p.add_argument("--hazard", choices=("low", "medium", "high"), default="medium")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--mm-per-pixel", type=float, default=100.0)
    p.add_argument("--margin-m", type=float, default=0.5, help="World margin around slab bounds (m).")
    p.add_argument(
        "--riser-contains",
        type=str,
        default=None,
        help="If set, IfcSpace whose Name contains this substring (case-insensitive) is drawn as riser (blue).",
    )
    args = p.parse_args()

    if not args.ifc.exists():
        raise SystemExit(f"IFC not found: {args.ifc}")

    import ifcopenshell  # noqa: F401 — checked at runtime for helpful error

    if args.list_storeys:
        model = ifcopenshell.open(str(args.ifc))
        for n in list_storey_names(model):
            print(n)
        return

    hz = {"low": HazardClass.LOW, "medium": HazardClass.MEDIUM, "high": HazardClass.HIGH}[args.hazard]

    img, resolved, debug = export_storey_semantic(
        args.ifc,
        args.storey,
        width=args.width,
        height=args.height,
        mm_per_pixel=args.mm_per_pixel,
        margin_m=args.margin_m,
        hazard=hz,
        riser_name_contains=args.riser_contains,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    png_path = args.out / "semantic.png"
    meta_path = args.out / "meta.yaml"
    dbg_path = args.out / "ifc_export_debug.json"

    cv2.imwrite(str(png_path), img)
    meta_path.write_text(
        yaml.safe_dump(
            {
                "hazard": args.hazard,
                "storey": resolved,
                "ifc": str(args.ifc.resolve()),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    dbg_path.write_text(json.dumps(debug, indent=2), encoding="utf-8")

    print(f"Wrote {png_path}")
    print(f"Wrote {meta_path} (storey={resolved})")
    print(f"Wrote {dbg_path}")
    print("Next: sprinkler-hd-infer --checkpoint <runs/.../epoch_XXXX.pt> --input {0} --meta {1} --out pred.png".format(png_path, meta_path))


if __name__ == "__main__":
    main()
