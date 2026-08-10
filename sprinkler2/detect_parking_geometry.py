from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import ifcopenshell
import ifcopenshell.geom
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import GeometryCollection, LineString, MultiLineString, MultiPolygon, Polygon
from shapely.ops import unary_union

TARGET_STOREY_NAME = "-2. Story"
DEFAULT_IFC_CANDIDATES = [
    Path("archicad") / "გარემო დიღომი (მშენებლობა).ifc",
]

DEFAULT_SLAB_MIN_ABS_NORMAL_Z = 0.88
_TRIANGLE_UNION_CHUNK = 800


def _norm_storey_label(name: str | None) -> str:
    if not name:
        return ""
    return " ".join(name.strip().split()).casefold()


def _longest_linestring_from_intersection(geom: Any, min_length: float = 1e-6) -> LineString | None:
    if geom is None or geom.is_empty:
        return None
    if isinstance(geom, LineString):
        return geom if geom.length >= min_length else None
    if isinstance(geom, MultiLineString):
        segs = [g for g in geom.geoms if isinstance(g, LineString) and g.length >= min_length]
        if not segs:
            return None
        return max(segs, key=lambda g: g.length)
    if hasattr(geom, "geoms"):
        candidates: list[LineString] = []
        for g in geom.geoms:
            found = _longest_linestring_from_intersection(g, min_length=min_length)
            if found is not None:
                candidates.append(found)
        if not candidates:
            return None
        return max(candidates, key=lambda g: g.length)
    return None


def _union_triangles_chunked(triangles: list[Polygon], chunk_size: int = _TRIANGLE_UNION_CHUNK) -> Any:
    if not triangles:
        raise ValueError("triangles must be non-empty")
    if len(triangles) <= chunk_size:
        return unary_union(triangles)
    chunks: list[Any] = []
    for i in range(0, len(triangles), chunk_size):
        chunks.append(unary_union(triangles[i : i + chunk_size]))
    return unary_union(chunks)


@dataclass
class ElementFootprint:
    ifc_id: int
    global_id: str | None
    ifc_class: str
    name: str | None
    storey: str | None
    polygon: Polygon | MultiPolygon | None
    error: str | None = None


def build_geom_settings() -> ifcopenshell.geom.settings:
    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)
    return settings


def resolve_ifc_path(cli_path: str | None) -> Path:
    if cli_path:
        candidate = Path(cli_path)
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"IFC path not found: {candidate}")

    for candidate in DEFAULT_IFC_CANDIDATES:
        if candidate.exists():
            return candidate

    all_ifc_files = sorted(Path(".").rglob("*.ifc"))
    for candidate in all_ifc_files:
        if ".venv" not in candidate.parts:
            return candidate
    raise FileNotFoundError("No IFC file found. Pass --ifc to specify one.")


def normalize_polygon(geom: Any) -> Polygon | MultiPolygon | None:
    if geom is None or geom.is_empty:
        return None
    cleaned = geom.buffer(0)
    if cleaned.is_empty:
        return None
    if isinstance(cleaned, (Polygon, MultiPolygon)):
        return cleaned
    if isinstance(cleaned, GeometryCollection):
        polys = [g for g in cleaned.geoms if isinstance(g, (Polygon, MultiPolygon)) and not g.is_empty]
        if not polys:
            return None
        return unary_union(polys).buffer(0)
    return None


def shape_to_2d_footprint(
    elem: Any,
    settings: ifcopenshell.geom.settings,
    *,
    min_abs_normal_z: float | None = None,
) -> tuple[Polygon | MultiPolygon | None, str | None]:
    try:
        shape = ifcopenshell.geom.create_shape(settings, elem)
        verts = np.array(shape.geometry.verts, dtype=float).reshape(-1, 3)
        faces = np.array(shape.geometry.faces, dtype=int).reshape(-1, 3)
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)

    if len(verts) == 0 or len(faces) == 0:
        return None, "empty mesh"

    triangles: list[Polygon] = []
    skipped_normal = 0
    for i0, i1, i2 in faces:
        try:
            p0 = verts[i0]
            p1 = verts[i1]
            p2 = verts[i2]
            if min_abs_normal_z is not None:
                e1 = p1 - p0
                e2 = p2 - p0
                cn = np.cross(e1, e2)
                norm = float(np.linalg.norm(cn))
                if norm < 1e-18:
                    continue
                nz = abs(float(cn[2] / norm))
                if nz < min_abs_normal_z:
                    skipped_normal += 1
                    continue

            tri_xy = [
                (float(p0[0]), float(p0[1])),
                (float(p1[0]), float(p1[1])),
                (float(p2[0]), float(p2[1])),
            ]
            poly = Polygon(tri_xy)
            if poly.is_valid and not poly.is_empty and poly.area > 1e-8:
                triangles.append(poly)
        except Exception:  # noqa: BLE001
            continue

    if not triangles:
        hint = (
            f" (all {skipped_normal} faces dropped by |n_z| filter; try lowering min_abs_normal_z)"
            if skipped_normal and min_abs_normal_z is not None
            else ""
        )
        return None, f"no valid triangles in XY projection{hint}"

    footprint = normalize_polygon(_union_triangles_chunked(triangles))
    if footprint is None:
        return None, "failed to build valid polygon footprint"
    return footprint, None


def get_storey_name(element: Any) -> str | None:
    for rel in getattr(element, "ContainedInStructure", []) or []:
        structure = getattr(rel, "RelatingStructure", None)
        if structure and structure.is_a("IfcBuildingStorey"):
            return structure.Name

    for rel in getattr(element, "Decomposes", []) or []:
        parent = getattr(rel, "RelatingObject", None)
        if parent and parent.is_a("IfcBuildingStorey"):
            return parent.Name
    return None


def collect_storeys(model: Any) -> list[dict[str, Any]]:
    storeys = []
    for s in model.by_type("IfcBuildingStorey"):
        storeys.append(
            {
                "ifc_id": s.id(),
                "global_id": getattr(s, "GlobalId", None),
                "name": getattr(s, "Name", None),
                "elevation": getattr(s, "Elevation", None),
            }
        )
    return storeys


def extract_elements(
    model: Any,
    settings: ifcopenshell.geom.settings,
    ifc_type: str,
    target_storey: str,
    exact_class_only: bool = False,
    *,
    strict_storey_match: bool = False,
    slab_horizontal_faces_only: bool = True,
    slab_min_abs_normal_z: float = DEFAULT_SLAB_MIN_ABS_NORMAL_Z,
) -> tuple[list[ElementFootprint], list[ElementFootprint], list[ElementFootprint]]:
    selected: list[ElementFootprint] = []
    failed: list[ElementFootprint] = []
    out_of_storey: list[ElementFootprint] = []

    def storey_matches(elem_storey: str | None) -> bool:
        if strict_storey_match:
            return elem_storey == target_storey
        return _norm_storey_label(elem_storey) == _norm_storey_label(target_storey)

    min_abs_z: float | None = None
    if ifc_type == "IfcSlab" and slab_horizontal_faces_only:
        min_abs_z = slab_min_abs_normal_z

    for elem in model.by_type(ifc_type):
        if exact_class_only and elem.is_a() != ifc_type:
            continue

        storey_name = get_storey_name(elem)
        if not storey_matches(storey_name):
            out_of_storey.append(
                ElementFootprint(
                    ifc_id=elem.id(),
                    global_id=getattr(elem, "GlobalId", None),
                    ifc_class=ifc_type,
                    name=getattr(elem, "Name", None),
                    storey=storey_name,
                    polygon=None,
                )
            )
            continue

        poly, err = shape_to_2d_footprint(elem, settings, min_abs_normal_z=min_abs_z)
        record = ElementFootprint(
            ifc_id=elem.id(),
            global_id=getattr(elem, "GlobalId", None),
            ifc_class=ifc_type,
            name=getattr(elem, "Name", None),
            storey=storey_name,
            polygon=poly,
            error=err,
        )
        if poly is None:
            failed.append(record)
        else:
            selected.append(record)

    return selected, failed, out_of_storey


def polygons_union(items: Iterable[ElementFootprint]) -> Polygon | MultiPolygon | None:
    polys = [it.polygon for it in items if it.polygon is not None]
    if not polys:
        return None
    return normalize_polygon(unary_union(polys))


def geometry_to_json_dict(geom: Polygon | MultiPolygon | None) -> dict[str, Any] | None:
    if geom is None or geom.is_empty:
        return None
    if isinstance(geom, Polygon):
        return {
            "type": "Polygon",
            "exterior": [list(pt) for pt in geom.exterior.coords],
            "holes": [[list(pt) for pt in ring.coords] for ring in geom.interiors],
            "area": float(geom.area),
        }
    if isinstance(geom, MultiPolygon):
        return {
            "type": "MultiPolygon",
            "parts": [geometry_to_json_dict(g) for g in geom.geoms],
            "area": float(geom.area),
        }
    return None


def bounds_dict(geom: Polygon | MultiPolygon | None) -> dict[str, float] | None:
    if geom is None or geom.is_empty:
        return None
    minx, miny, maxx, maxy = geom.bounds
    return {"min_x": float(minx), "min_y": float(miny), "max_x": float(maxx), "max_y": float(maxy)}


def maybe_simplify(geom: Polygon | MultiPolygon | None, tolerance_m: float) -> Polygon | MultiPolygon | None:
    if geom is None or tolerance_m <= 0:
        return geom
    simplified = geom.simplify(tolerance_m, preserve_topology=True)
    return normalize_polygon(simplified)


def ifc_instance_counts(model: Any, type_names: Iterable[str]) -> dict[str, int]:
    return {name: len(model.by_type(name)) for name in type_names}


def principal_axis_from_floorplate(geom: Polygon | MultiPolygon | None) -> dict[str, Any] | None:
    if geom is None or geom.is_empty:
        return None

    mrr = geom.minimum_rotated_rectangle
    coords = list(mrr.exterior.coords)
    if len(coords) < 5:
        return None

    edges = []
    for i in range(4):
        p0 = np.array(coords[i], dtype=float)
        p1 = np.array(coords[i + 1], dtype=float)
        vec = p1 - p0
        length = float(np.linalg.norm(vec))
        if length > 1e-9:
            edges.append((length, vec))
    if not edges:
        return None

    edges.sort(key=lambda x: x[0], reverse=True)
    main_vec = edges[0][1] / np.linalg.norm(edges[0][1])
    main_angle_deg = math.degrees(math.atan2(main_vec[1], main_vec[0]))
    branch_vec = np.array([-main_vec[1], main_vec[0]], dtype=float)
    branch_angle_deg = math.degrees(math.atan2(branch_vec[1], branch_vec[0]))

    return {
        "main_axis": {
            "unit_vector_xy": [float(main_vec[0]), float(main_vec[1])],
            "angle_deg_from_x": float(main_angle_deg),
        },
        "branch_axis": {
            "unit_vector_xy": [float(branch_vec[0]), float(branch_vec[1])],
            "angle_deg_from_x": float(branch_angle_deg),
        },
    }


def infer_trunk_line(
    floorplate: Polygon | MultiPolygon | None, axis_info: dict[str, Any] | None
) -> LineString | None:
    """Straight trunk along the slab long axis (MRR), clipped to the floor polygon (original behaviour)."""
    if floorplate is None or floorplate.is_empty or not axis_info:
        return None

    vec = np.array(axis_info["main_axis"]["unit_vector_xy"], dtype=float)
    centroid = np.array([floorplate.centroid.x, floorplate.centroid.y], dtype=float)
    boundary_points = np.array(list(floorplate.convex_hull.exterior.coords), dtype=float)
    rel = boundary_points - centroid
    projections = rel @ vec
    if len(projections) == 0:
        return None

    p0 = centroid + vec * float(np.min(projections))
    p1 = centroid + vec * float(np.max(projections))
    candidate = LineString([tuple(p0), tuple(p1)])
    clipped = candidate.intersection(floorplate)
    return _longest_linestring_from_intersection(clipped)


def element_record(item: ElementFootprint) -> dict[str, Any]:
    return {
        "ifc_id": item.ifc_id,
        "global_id": item.global_id,
        "ifc_class": item.ifc_class,
        "name": item.name,
        "storey": item.storey,
        "error": item.error,
        "footprint": geometry_to_json_dict(item.polygon),
    }


def draw_geom(ax: Any, geom: Polygon | MultiPolygon | None, color: str, alpha: float, label: str) -> None:
    if geom is None or geom.is_empty:
        return
    geoms = [geom] if isinstance(geom, Polygon) else list(geom.geoms)
    for idx, g in enumerate(geoms):
        ext = np.array(g.exterior.coords)
        ax.add_patch(
            MplPolygon(
                ext,
                closed=True,
                facecolor=color,
                edgecolor=color,
                linewidth=1.0,
                alpha=alpha,
                label=label if idx == 0 else None,
            )
        )
        for ring in g.interiors:
            ring_xy = np.array(ring.coords)
            ax.add_patch(
                MplPolygon(
                    ring_xy,
                    closed=True,
                    facecolor="white",
                    edgecolor=color,
                    linewidth=0.8,
                    alpha=1.0,
                )
            )


def _draw_preview_floor_badge(ax: Any, label: str) -> None:
    if not label or not str(label).strip():
        return
    ax.text(
        0.01,
        0.98,
        str(label).strip(),
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        color="#111111",
        va="top",
        ha="left",
        bbox={
            "boxstyle": "round,pad=0.4",
            "fc": "white",
            "ec": "#ff2d2d",
            "linewidth": 1.0,
            "alpha": 0.95,
        },
        zorder=100,
    )


def save_preview(
    out_png: Path,
    slab_items: list[ElementFootprint],
    unified_area: Polygon | MultiPolygon | None,
    column_items: list[ElementFootprint],
    stair_items: list[ElementFootprint],
    wall_items: list[ElementFootprint],
    trunk_line: LineString | None,
    preview_floor_label: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 8))

    for i, slab in enumerate(slab_items):
        draw_geom(ax, slab.polygon, color="#4f9dda", alpha=0.35, label="Slab footprint" if i == 0 else "")
    draw_geom(ax, unified_area, color="#24577a", alpha=0.12, label="Unified protected area")
    for i, col in enumerate(column_items):
        draw_geom(ax, col.polygon, color="#d2842f", alpha=0.8, label="Columns" if i == 0 else "")
    for i, st in enumerate(stair_items):
        draw_geom(ax, st.polygon, color="#6f42c1", alpha=0.75, label="Stairs exclusion" if i == 0 else "")
    for i, wall in enumerate(wall_items):
        draw_geom(ax, wall.polygon, color="#888888", alpha=0.5, label="Wall footprints" if i == 0 else "")

    if trunk_line is not None and not trunk_line.is_empty:
        x, y = trunk_line.xy
        ax.plot(x, y, color="red", linewidth=2.5, label="Suggested trunk line (auto)")

    ax.set_title("Detected Geometry Preview")
    ax.set_xlabel("X (world)")
    ax.set_ylabel("Y (world)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2)
    handles, labels = ax.get_legend_handles_labels()
    dedup = {}
    for h, l in zip(handles, labels):
        if l and l not in dedup:
            dedup[l] = h
    if dedup:
        ax.legend(dedup.values(), dedup.keys(), loc="best")
    _draw_preview_floor_badge(ax, preview_floor_label or "")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect IFC floor geometry for sprinkler layout (simple auto trunk).")
    parser.add_argument("--ifc", type=str, default=None, help="Path to IFC file. If omitted, script auto-detects.")
    parser.add_argument("--storey", type=str, default=TARGET_STOREY_NAME, help="Target IfcBuildingStorey name.")
    parser.add_argument(
        "--strict-storey",
        action="store_true",
        help="Require exact IfcBuildingStorey.Name match (default: ignore case/extra spaces).",
    )
    parser.add_argument(
        "--slab-all-faces",
        action="store_true",
        help="For IfcSlab, union all mesh triangles (default: mostly horizontal faces only).",
    )
    parser.add_argument(
        "--slab-min-abs-normal-z",
        type=float,
        default=DEFAULT_SLAB_MIN_ABS_NORMAL_Z,
        help=f"IfcSlab horizontal face filter |n_z| threshold (default {DEFAULT_SLAB_MIN_ABS_NORMAL_Z}).",
    )
    parser.add_argument(
        "--simplify-m",
        type=float,
        default=0.0,
        help="Optional Shapely simplify tolerance in metres on merged regions (0 disables).",
    )
    parser.add_argument(
        "--no-auto-trunk",
        action="store_true",
        help="Do not write suggested_trunk_line (use pick_trunk_line.py to set it interactively).",
    )
    parser.add_argument("--output-dir", type=str, default="outputs/output", help="Output directory for JSON and PNG.")
    parser.add_argument(
        "--preview-floor-label",
        type=str,
        default="",
        help="If set, draws this text on detected_geometry_preview.png",
    )
    args = parser.parse_args()

    ifc_path = resolve_ifc_path(args.ifc)
    out_dir = Path(args.output_dir)
    out_json = out_dir / "detected_geometry.json"
    out_png = out_dir / "detected_geometry_preview.png"

    model = ifcopenshell.open(str(ifc_path))
    settings = build_geom_settings()

    storeys = collect_storeys(model)

    tracked_types = (
        "IfcWall",
        "IfcWallStandardCase",
        "IfcSlab",
        "IfcColumn",
        "IfcStair",
        "IfcSpace",
    )

    extract_kw: dict[str, Any] = {
        "strict_storey_match": args.strict_storey,
        "slab_horizontal_faces_only": not args.slab_all_faces,
        "slab_min_abs_normal_z": float(args.slab_min_abs_normal_z),
    }

    slabs, slabs_failed, _ = extract_elements(model, settings, "IfcSlab", args.storey, **extract_kw)
    columns, columns_failed, _ = extract_elements(model, settings, "IfcColumn", args.storey, **extract_kw)
    stairs, stairs_failed, _ = extract_elements(model, settings, "IfcStair", args.storey, **extract_kw)
    wall_std, wall_std_failed, _ = extract_elements(model, settings, "IfcWallStandardCase", args.storey, **extract_kw)
    wall_generic, wall_generic_failed, _ = extract_elements(
        model,
        settings,
        "IfcWall",
        args.storey,
        exact_class_only=True,
        **extract_kw,
    )
    spaces, spaces_failed, _ = extract_elements(model, settings, "IfcSpace", args.storey, **extract_kw)

    slab_union = maybe_simplify(polygons_union(slabs), args.simplify_m)
    column_union = maybe_simplify(polygons_union(columns), args.simplify_m)
    stair_union = maybe_simplify(polygons_union(stairs), args.simplify_m)
    wall_std_union = polygons_union(wall_std)
    wall_generic_union = polygons_union(wall_generic)
    all_walls_union = maybe_simplify(
        normalize_polygon(unary_union([g for g in [wall_std_union, wall_generic_union] if g is not None])),
        args.simplify_m,
    )

    axis_info = principal_axis_from_floorplate(slab_union)
    trunk_line: LineString | None = None if args.no_auto_trunk else infer_trunk_line(slab_union, axis_info)

    data: dict[str, Any] = {
        "detection_meta": {
            "detector": "detect_parking_geometry.py",
            "slab_face_filter": (not args.slab_all_faces),
            "slab_min_abs_normal_z": float(args.slab_min_abs_normal_z) if not args.slab_all_faces else None,
            "storey_match": "exact" if args.strict_storey else "normalized_name",
            "simplify_m": float(args.simplify_m),
            "trunk_source": "none" if args.no_auto_trunk else "auto_mrr_axis",
            "note": "Override trunk with: python sprinkler2/pick_trunk_line.py --json <detected_geometry.json>",
        },
        "input_ifc": str(ifc_path),
        "target_storey": args.storey,
        "storeys_available": storeys,
        "ifc_instance_counts_total_file": ifc_instance_counts(model, tracked_types),
        "detected_counts_on_target_storey": {
            "IfcSlab_success": len(slabs),
            "IfcSlab_failed": len(slabs_failed),
            "IfcColumn_success": len(columns),
            "IfcColumn_failed": len(columns_failed),
            "IfcStair_success": len(stairs),
            "IfcStair_failed": len(stairs_failed),
            "IfcWallStandardCase_success": len(wall_std),
            "IfcWallStandardCase_failed": len(wall_std_failed),
            "IfcWall_success": len(wall_generic),
            "IfcWall_failed": len(wall_generic_failed),
            "IfcSpace_success": len(spaces),
            "IfcSpace_failed": len(spaces_failed),
        },
        "slab_footprints": [element_record(s) for s in slabs],
        "unified_protected_floor_area": geometry_to_json_dict(slab_union),
        "columns": [element_record(c) for c in columns],
        "columns_union": geometry_to_json_dict(column_union),
        "stairs": [element_record(s) for s in stairs],
        "stairs_union": geometry_to_json_dict(stair_union),
        "walls_standard_case": [element_record(w) for w in wall_std],
        "walls_standard_case_union": geometry_to_json_dict(wall_std_union),
        "walls_generic": [element_record(w) for w in wall_generic],
        "walls_generic_union": geometry_to_json_dict(wall_generic_union),
        "walls_all_union": geometry_to_json_dict(all_walls_union),
        "spaces": [element_record(s) for s in spaces],
        "generic_walls_failed_geometry": [element_record(w) for w in wall_generic_failed],
        "other_failures": {
            "slabs_failed": [element_record(x) for x in slabs_failed],
            "columns_failed": [element_record(x) for x in columns_failed],
            "stairs_failed": [element_record(x) for x in stairs_failed],
            "wall_standard_case_failed": [element_record(x) for x in wall_std_failed],
            "wall_generic_failed": [element_record(x) for x in wall_generic_failed],
            "spaces_failed": [element_record(x) for x in spaces_failed],
        },
        "overall_floor_bounds": bounds_dict(slab_union),
        "routing_candidate_area": None,
        "routing_parameters": None,
        "candidate_axes": axis_info,
        "suggested_trunk_line": None if trunk_line is None else [list(c) for c in trunk_line.coords],
        "trunk_endpoints": None,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    save_preview(
        out_png,
        slabs,
        slab_union,
        columns,
        stairs,
        wall_std + wall_generic,
        trunk_line,
        preview_floor_label=(args.preview_floor_label.strip() or None),
    )

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass

    print(f"IFC: {str(ifc_path).encode('ascii', errors='backslashreplace').decode('ascii')}")
    print(f"Storeys detected: {len(storeys)}")
    print(f"Target storey: {args.storey}")
    print()
    print("Extraction summary:")
    print(f"- Slabs: {len(slabs)} success, {len(slabs_failed)} failed")
    print(f"- Columns: {len(columns)} success, {len(columns_failed)} failed")
    print(f"- Stairs: {len(stairs)} success, {len(stairs_failed)} failed")
    print(f"- IfcWallStandardCase: {len(wall_std)} success, {len(wall_std_failed)} failed")
    print(f"- IfcWall (generic): {len(wall_generic)} success, {len(wall_generic_failed)} failed")
    print(f"- IfcSpace: {len(spaces)} success, {len(spaces_failed)} failed")
    print()
    if args.no_auto_trunk:
        print("Auto trunk disabled. Set the main line with: python sprinkler2/pick_trunk_line.py")
    else:
        print("Trunk: straight segment along slab long axis (MRR), clipped to floor. Override with pick_trunk_line.py")
    print()
    print(f"Saved JSON: {out_json}")
    print(f"Saved preview: {out_png}")


if __name__ == "__main__":
    main()
