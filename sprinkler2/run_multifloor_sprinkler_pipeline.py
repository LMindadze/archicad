"""
Run full sprinkler draft pipeline for every IfcBuildingStorey in an IFC:

  1) detect_parking_geometry.py  → detected_geometry.json (+ preview)
  2) sam2_corridor_trunk.py        → trunk (manual 2 clicks per floor, never --auto-points)
  3) generate_draft_layout.py    → layout_result.json, layout_preview.png, DXF

Outputs under: <out_root>/<project_name>/floor_XX_<storey_slug>/
Final PDF:     <out_root>/<project_name>/all_floors_layout_previews.pdf

Usage (from repo root):
  python sprinkler2/run_multifloor_sprinkler_pipeline.py --ifc path/to/model.ifc \\
    --sam2-config path/to/config.yaml --sam2-checkpoint path/to/sam2.pt
"""
from __future__ import annotations

import argparse
import os
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import ifcopenshell
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def safe_folder_name(name: str | None, max_len: int = 72) -> str:
    raw = (name or "unnamed").strip() or "unnamed"
    raw = re.sub(r'[<>:"/\\|?*\n\r\t]', "_", raw)
    raw = re.sub(r"\s+", "_", raw)
    return raw[:max_len] if len(raw) > max_len else raw


def storey_sort_key(entry: dict[str, Any]) -> tuple[float, str]:
    el = entry.get("elevation")
    try:
        z = float(el) if el is not None else 0.0
    except (TypeError, ValueError):
        z = 0.0
    name = str(entry.get("name") or "")
    return (z, name.casefold())


def collect_storeys_from_ifc(ifc_path: Path) -> list[dict[str, Any]]:
    model = ifcopenshell.open(str(ifc_path))
    storeys: list[dict[str, Any]] = []
    for s in model.by_type("IfcBuildingStorey"):
        storeys.append(
            {
                "ifc_id": s.id(),
                "global_id": getattr(s, "GlobalId", None),
                "name": getattr(s, "Name", None),
                "elevation": getattr(s, "Elevation", None),
            }
        )
    storeys.sort(key=storey_sort_key)
    return storeys


def run_python(
    script: Path,
    args: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    cmd = [sys.executable, str(script)] + args
    print("\n→ " + " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=str(cwd) if cwd else None, env=env)
    if r.returncode != 0:
        raise RuntimeError(f"Command failed (exit {r.returncode}): {' '.join(cmd)}")


def build_pdf_from_pngs(png_paths: list[Path], out_pdf: Path) -> None:
    if not png_paths:
        print("No layout_preview.png files found; skipping PDF.", flush=True)
        return
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out_pdf) as pdf:
        for p in png_paths:
            fig, ax = plt.subplots(figsize=(13, 8))
            ax.imshow(mpimg.imread(p))
            ax.set_title(p.parent.name, fontsize=10)
            ax.axis("off")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    print(f"Combined PDF: {out_pdf}", flush=True)


def build_subprocess_env(sam2_repo: Path | None) -> dict[str, str]:
    env = os.environ.copy()
    if sam2_repo is not None:
        py_path = env.get("PYTHONPATH", "")
        entries = [str(sam2_repo)]
        if py_path.strip():
            entries.append(py_path)
        env["PYTHONPATH"] = os.pathsep.join(entries)
    return env


def verify_sam2_import(env: dict[str, str], cwd: Path) -> tuple[bool, str]:
    probe = "import sam2,sys;print(getattr(sam2,'__file__','<unknown>'))"
    r = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
    )
    if r.returncode == 0:
        return True, (r.stdout.strip() or "<sam2 import ok>")
    msg = (r.stderr or r.stdout or "").strip()
    return False, msg


def normalize_sam2_config_name(raw: str, sam2_repo: Path | None) -> str:
    """
    Hydra in sam2.build_sam expects config_name like:
      configs/sam2.1/sam2.1_hiera_t.yaml
    not an absolute filesystem path.
    """
    txt = (raw or "").strip().replace("\\", "/")
    if not txt:
        return txt
    # Already a hydra-style config path.
    if txt.startswith("configs/"):
        return txt
    p = Path(raw)
    if p.is_absolute() and p.suffix.lower() in {".yaml", ".yml"}:
        if sam2_repo is not None:
            # Try make path relative to <sam2_repo>/sam2
            base = (sam2_repo / "sam2").resolve()
            try:
                rel = p.resolve().relative_to(base)
                return str(rel).replace("\\", "/")
            except Exception:
                pass
            # Try relative to repo root and keep trailing part from /sam2/
            try:
                rel2 = p.resolve().relative_to(sam2_repo.resolve())
                rel2s = str(rel2).replace("\\", "/")
                if rel2s.startswith("sam2/"):
                    return rel2s[len("sam2/") :]
            except Exception:
                pass
        # Fallback for absolute path that contains '/configs/...'
        low = txt.lower()
        idx = low.find("/configs/")
        if idx >= 0:
            return txt[idx + 1 :]
    # Relative filesystem path; if it contains sam2/ prefix strip it.
    txt = txt.lstrip("./")
    if txt.startswith("sam2/"):
        return txt[len("sam2/") :]
    return txt


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-floor IFC → detection → SAM2 trunk (clicks) → draft layout + PDF.",
    )
    parser.add_argument("--ifc", type=str, required=True, help="Path to IFC file.")
    parser.add_argument(
        "--project-name",
        type=str,
        default=None,
        help="Project folder name (default: IFC stem).",
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default=None,
        help="Parent directory for project output (default: ./outputs).",
    )
    parser.add_argument(
        "--sam2-config",
        type=str,
        required=True,
        help="SAM2 YAML config (passed to sam2_corridor_trunk.py).",
    )
    parser.add_argument(
        "--sam2-checkpoint",
        type=str,
        required=True,
        help="SAM2 checkpoint .pt (passed to sam2_corridor_trunk.py).",
    )
    parser.add_argument(
        "--sam2-repo",
        type=str,
        default=None,
        help="Path to segment-anything-2 repo root (contains sam2 package). If omitted, auto-detects next to workspace.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="SAM2 device (cuda|cpu).")
    parser.add_argument("--pixels-per-meter", type=float, default=14.0)
    parser.add_argument("--wall-buffer-m", type=float, default=0.25)
    parser.add_argument("--column-buffer-m", type=float, default=0.3)
    parser.add_argument(
        "--strict-storey",
        action="store_true",
        help="Forward to detect_parking_geometry.py",
    )
    parser.add_argument("--simplify-m", type=float, default=0.0)
    parser.add_argument(
        "--only-storeys",
        type=str,
        default=None,
        help="Comma-separated storey names to process (exact strings as in IFC).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List storeys and folders only; do not run pipeline steps.",
    )
    args = parser.parse_args()

    ifc_path = Path(args.ifc).resolve()
    if not ifc_path.is_file():
        print(f"IFC not found: {ifc_path}", file=sys.stderr)
        sys.exit(1)

    project_name = args.project_name or ifc_path.stem
    safe_project = safe_folder_name(project_name, max_len=96)
    out_root = Path(args.out_root).resolve() if args.out_root else (Path.cwd() / "outputs").resolve()
    project_dir = out_root / safe_project
    project_dir.mkdir(parents=True, exist_ok=True)

    sprinkler2 = Path(__file__).resolve().parent
    repo_root = sprinkler2.parent
    auto_sam2_repo = (repo_root / "segment-anything-2").resolve()
    sam2_repo = Path(args.sam2_repo).resolve() if args.sam2_repo else (auto_sam2_repo if auto_sam2_repo.is_dir() else None)
    proc_env = build_subprocess_env(sam2_repo)
    sam2_config_name = normalize_sam2_config_name(args.sam2_config, sam2_repo)

    detect_script = sprinkler2 / "detect_parking_geometry.py"
    sam2_script = sprinkler2 / "sam2_corridor_trunk.py"
    layout_script = sprinkler2 / "generate_draft_layout.py"

    for s in (detect_script, sam2_script, layout_script):
        if not s.is_file():
            print(f"Missing script: {s}", file=sys.stderr)
            sys.exit(1)

    storeys = collect_storeys_from_ifc(ifc_path)
    if not storeys:
        print("No IfcBuildingStorey entities found.", file=sys.stderr)
        sys.exit(1)

    filter_names: set[str] | None = None
    if args.only_storeys:
        filter_names = {x.strip() for x in args.only_storeys.split(",") if x.strip()}

    run_plan: list[dict[str, Any]] = []
    for st in storeys:
        name = st.get("name")
        if filter_names is not None and name not in filter_names:
            continue
        run_plan.append(st)

    if not run_plan:
        print("No storeys match --only-storeys filter.", file=sys.stderr)
        sys.exit(1)

    print("\n" + "=" * 72, flush=True)
    print("[Pipeline] Multifloor sprinkler draft — geometry → SAM2 trunk (clicks) → layout", flush=True)
    print("=" * 72, flush=True)
    print(f"[Pipeline] IFC file: {ifc_path}", flush=True)
    print(f"[Pipeline] IfcBuildingStorey entities in file: {len(storeys)}", flush=True)
    print(f"[Pipeline] Floors in this run: {len(run_plan)}", flush=True)
    print(f"[Pipeline] Project output root: {project_dir}", flush=True)
    print(f"[Pipeline] SAM2 repo: {sam2_repo if sam2_repo else '<not set>'}", flush=True)
    print(f"[Pipeline] SAM2 config name passed to Hydra: {sam2_config_name}", flush=True)
    print("=" * 72 + "\n", flush=True)
    if args.dry_run:
        for idx, st in enumerate(run_plan, start=1):
            nm = st.get("name") or f"storey_{idx}"
            slug = safe_folder_name(str(nm))
            print(f"  [{idx:02d}] {nm} → floor_{idx:02d}_{slug}")
        return

    manifest_all: dict[str, Any] = {
        "ifc": str(ifc_path),
        "project_name": safe_project,
        "storeys": [],
    }

    layout_pngs: list[Path] = []

    ok_sam2, sam2_msg = verify_sam2_import(proc_env, repo_root)
    if not ok_sam2:
        print("[Pipeline] SAM2 import check failed before floor processing.", file=sys.stderr)
        print("[Pipeline] Set --sam2-repo to your segment-anything-2 root or install sam2 in current env.", file=sys.stderr)
        print(f"[Pipeline] Import error:\n{sam2_msg}", file=sys.stderr)
        sys.exit(2)
    print(f"[Pipeline] SAM2 import OK: {sam2_msg}", flush=True)

    for seq_idx, st in enumerate(run_plan, start=1):
        storey_name = st.get("name")
        if not storey_name:
            print(f"Skipping storey without Name at index {seq_idx}", file=sys.stderr)
            continue
        slug = safe_folder_name(str(storey_name))
        floor_dir = project_dir / f"floor_{seq_idx:02d}_{slug}"
        floor_dir.mkdir(parents=True, exist_ok=True)

        detected_json = floor_dir / "detected_geometry.json"
        sam2_preview = floor_dir / "sam2_trunk_preview.png"
        layout_png = floor_dir / "layout_preview.png"

        floor_badge = (
            f"Floor {seq_idx}/{len(run_plan)} | Storey: {storey_name} "
            f"| IfcBuildingStorey id={st.get('ifc_id')}"
        )

        print("\n" + "=" * 72, flush=True)
        print(f"[Floor {seq_idx} / {len(run_plan)}] Storey name: {storey_name}", flush=True)
        print(f"[Floor {seq_idx} / {len(run_plan)}] Ifc id: {st.get('ifc_id')}", flush=True)
        print(f"[Floor {seq_idx} / {len(run_plan)}] Output folder: {floor_dir}", flush=True)
        print("=" * 72, flush=True)

        detect_args = [
            "--ifc",
            str(ifc_path),
            "--storey",
            str(storey_name),
            "--output-dir",
            str(floor_dir),
            "--no-auto-trunk",
            "--simplify-m",
            str(args.simplify_m),
            "--preview-floor-label",
            floor_badge,
        ]
        if args.strict_storey:
            detect_args.append("--strict-storey")

        print(
            f"\n[Floor {seq_idx}/{len(run_plan)}] Stage 1/3 — IFC geometry detection "
            f"(detected_geometry.json + preview PNG)…",
            flush=True,
        )
        run_python(detect_script, detect_args, cwd=repo_root, env=proc_env)
        print(f"[Floor {seq_idx}/{len(run_plan)}] Stage 1/3 — done.\n", flush=True)

        print(
            f"[Floor {seq_idx}/{len(run_plan)}] Stage 2/3 — SAM2 corridor trunk "
            f"(manual: click START wall, then END wall). Auto-points: OFF.\n",
            flush=True,
        )
        sam2_args = [
            "--json",
            str(detected_json),
            "--output-json",
            str(detected_json),
            "--sam2-config",
            sam2_config_name,
            "--sam2-checkpoint",
            str(Path(args.sam2_checkpoint).resolve()),
            "--device",
            args.device,
            "--pixels-per-meter",
            str(args.pixels_per_meter),
            "--wall-buffer-m",
            str(args.wall_buffer_m),
            "--column-buffer-m",
            str(args.column_buffer_m),
            "--save-preview",
            str(sam2_preview),
            "--preview-floor-label",
            floor_badge,
        ]
        # Deliberately never pass --auto-points (manual clicks every floor).
        run_python(sam2_script, sam2_args, cwd=repo_root, env=proc_env)
        print(f"[Floor {seq_idx}/{len(run_plan)}] Stage 2/3 — done.\n", flush=True)

        print(
            f"[Floor {seq_idx}/{len(run_plan)}] Stage 3/3 — Draft layout "
            f"(layout_preview.png, DXF, JSON)…",
            flush=True,
        )
        run_python(
            layout_script,
            [
                "--input-json",
                str(detected_json),
                "--output-dir",
                str(floor_dir),
                "--preview-floor-label",
                floor_badge,
            ],
            cwd=repo_root,
            env=proc_env,
        )
        print(f"[Floor {seq_idx}/{len(run_plan)}] Stage 3/3 — done.\n", flush=True)

        if layout_png.is_file():
            layout_pngs.append(layout_png)

        manifest_all["storeys"].append(
            {
                "floor_index": seq_idx,
                "storey_name": storey_name,
                "ifc_building_storey_id": st.get("ifc_id"),
                "preview_floor_label": floor_badge,
                "folder": str(floor_dir.relative_to(project_dir)),
                "detected_geometry": str(detected_json.name),
                "layout_preview": str(layout_png.name) if layout_png.is_file() else None,
            }
        )

    manifest_path = project_dir / "pipeline_manifest.json"
    manifest_path.write_text(json.dumps(manifest_all, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[Pipeline] Manifest: {manifest_path}", flush=True)

    pdf_out = project_dir / "all_floors_layout_previews.pdf"
    print(f"[Pipeline] Building combined PDF ({len(layout_pngs)} pages)…", flush=True)
    build_pdf_from_pngs(layout_pngs, pdf_out)
    print("[Pipeline] All floors complete.", flush=True)


if __name__ == "__main__":
    main()
