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
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import GeometryCollection, LineString, MultiPolygon, Polygon
from shapely.ops import unary_union


TARGET_STOREY_NAME = "-2. Story"
DEFAULT_IFC_CANDIDATES = [
    Path("archicad") / "გარემო.ifc",
]


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
    elem: Any, settings: ifcopenshell.geom.settings
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
    for i0, i1, i2 in faces:
        try:
            tri_xy = [
                (float(verts[i0][0]), float(verts[i0][1])),
                (float(verts[i1][0]), float(verts[i1][1])),
                (float(verts[i2][0]), float(verts[i2][1])),
            ]
            poly = Polygon(tri_xy)
            if poly.is_valid and not poly.is_empty and poly.area > 1e-8:
                triangles.append(poly)
        except Exception:  # noqa: BLE001
            continue

    if not triangles:
        return None, "no valid triangles in XY projection"

    footprint = normalize_polygon(unary_union(triangles))
    if footprint is None:
        return None, "failed to build valid polygon footprint"
    return footprint, None


def get_storey_name(element: Any) -> str | None:
    # Primary path: IfcRelContainedInSpatialStructure
    for rel in getattr(element, "ContainedInStructure", []) or []:
        structure = getattr(rel, "RelatingStructure", None)
        if structure and structure.is_a("IfcBuildingStorey"):
            return structure.Name

    # Fallback path: decomposition / nesting links
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
) -> tuple[list[ElementFootprint], list[ElementFootprint], list[ElementFootprint]]:
    selected: list[ElementFootprint] = []
    failed: list[ElementFootprint] = []
    out_of_storey: list[ElementFootprint] = []

    for elem in model.by_type(ifc_type):
        if exact_class_only and elem.is_a() != ifc_type:
            continue

        storey_name = get_storey_name(elem)
        if storey_name != target_storey:
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

        poly, err = shape_to_2d_footprint(elem, settings)
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
    if clipped.is_empty:
        return None

    if isinstance(clipped, LineString):
        return clipped
    if hasattr(clipped, "geoms"):
        line_candidates = [g for g in clipped.geoms if isinstance(g, LineString)]
        if line_candidates:
            line_candidates.sort(key=lambda g: g.length, reverse=True)
            return line_candidates[0]
    return None


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


def save_preview(
    out_png: Path,
    slab_items: list[ElementFootprint],
    unified_area: Polygon | MultiPolygon | None,
    column_items: list[ElementFootprint],
    stair_items: list[ElementFootprint],
    wall_items: list[ElementFootprint],
    trunk_line: LineString | None,
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
        ax.plot(x, y, color="red", linewidth=2.5, label="Suggested trunk line")

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
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect IFC floor geometry for future suppression layout generation.")
    parser.add_argument("--ifc", type=str, default=None, help="Path to IFC file. If omitted, script auto-detects.")
    parser.add_argument("--storey", type=str, default=TARGET_STOREY_NAME, help="Target IfcBuildingStorey name.")
    parser.add_argument("--output-dir", type=str, default="outputs/output", help="Output directory for JSON and PNG.")
    args = parser.parse_args()

    ifc_path = resolve_ifc_path(args.ifc)
    out_dir = Path(args.output_dir)
    out_json = out_dir / "detected_geometry.json"
    out_png = out_dir / "detected_geometry_preview.png"

    model = ifcopenshell.open(str(ifc_path))
    settings = build_geom_settings()

    storeys = collect_storeys(model)

    slabs, slabs_failed, _ = extract_elements(model, settings, "IfcSlab", args.storey)
    columns, columns_failed, _ = extract_elements(model, settings, "IfcColumn", args.storey)
    stairs, stairs_failed, _ = extract_elements(model, settings, "IfcStair", args.storey)
    wall_std, wall_std_failed, _ = extract_elements(model, settings, "IfcWallStandardCase", args.storey)
    wall_generic, wall_generic_failed, _ = extract_elements(
        model,
        settings,
        "IfcWall",
        args.storey,
        exact_class_only=True,
    )
    spaces, spaces_failed, _ = extract_elements(model, settings, "IfcSpace", args.storey)

    slab_union = polygons_union(slabs)
    column_union = polygons_union(columns)
    stair_union = polygons_union(stairs)
    wall_std_union = polygons_union(wall_std)
    wall_generic_union = polygons_union(wall_generic)
    all_walls_union = normalize_polygon(
        unary_union([g for g in [wall_std_union, wall_generic_union] if g is not None])
    )

    axis_info = principal_axis_from_floorplate(slab_union)
    trunk_line = infer_trunk_line(slab_union, axis_info)

    data = {
        "input_ifc": str(ifc_path),
        "target_storey": args.storey,
        "storeys_available": storeys,
        "known_counts_reference": {
            "IfcWall": 186,
            "IfcWallStandardCase": 166,
            "IfcSlab": 3,
            "IfcColumn": 81,
            "IfcStair": 4,
            "IfcSpace": 0,
        },
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
        "generic_walls_failed_geometry": [element_record(w) for w in wall_generic_failed],
        "other_failures": {
            "slabs_failed": [element_record(x) for x in slabs_failed],
            "columns_failed": [element_record(x) for x in columns_failed],
            "stairs_failed": [element_record(x) for x in stairs_failed],
            "wall_standard_case_failed": [element_record(x) for x in wall_std_failed],
            "spaces_failed": [element_record(x) for x in spaces_failed],
        },
        "overall_floor_bounds": bounds_dict(slab_union),
        "candidate_axes": axis_info,
        "suggested_trunk_line": list(trunk_line.coords) if trunk_line is not None else None,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    save_preview(out_png, slabs, slab_union, columns, stairs, wall_std + wall_generic, trunk_line)

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
    print("What is good enough for layout generation:")
    print("- Unified slab area provides the primary protected zone.")
    print("- Column and stair footprints provide practical obstacle/exclusion zones.")
    print("- Main axis, branch axis, and trunk line candidate are inferable from slab geometry.")
    print("- WallStandardCase and any successful generic walls can be used as supplemental routing constraints.")
    print()
    print(f"Saved JSON: {out_json}")
    print(f"Saved preview: {out_png}")


if __name__ == "__main__":
    main()
