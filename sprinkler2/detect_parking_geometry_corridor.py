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
import networkx as nx
import numpy as np
from matplotlib.patches import Polygon as MplPolygon
from scipy import ndimage as ndi
from shapely.geometry import GeometryCollection, LineString, MultiLineString, MultiPolygon, Point, Polygon
from shapely.ops import linemerge, unary_union
from skimage.morphology import skeletonize


TARGET_STOREY_NAME = "-2. Story"
DEFAULT_IFC_CANDIDATES = [
    Path("archicad") / "გარემო დიღომი (მშენებლობა).ifc",
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


# ---------------------------
# IFC extraction helpers
# ---------------------------
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


def geometry_to_json_dict(geom: Polygon | MultiPolygon | LineString | MultiLineString | None) -> dict[str, Any] | None:
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
    if isinstance(geom, LineString):
        return {
            "type": "LineString",
            "coordinates": [list(pt) for pt in geom.coords],
            "length": float(geom.length),
        }
    if isinstance(geom, MultiLineString):
        return {
            "type": "MultiLineString",
            "parts": [geometry_to_json_dict(g) for g in geom.geoms],
            "length": float(geom.length),
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


# ---------------------------
# Corridor / trunk inference
# ---------------------------
def extract_polygon_components(geom: Polygon | MultiPolygon | None) -> list[Polygon]:
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    return [g for g in geom.geoms if isinstance(g, Polygon) and not g.is_empty]


def build_routing_candidate_area(
    slab_union: Polygon | MultiPolygon | None,
    walls_union: Polygon | MultiPolygon | None,
    columns_union: Polygon | MultiPolygon | None,
    stairs_union: Polygon | MultiPolygon | None,
    wall_clearance: float,
    column_clearance: float,
    stair_clearance: float,
    boundary_inset: float,
    min_component_area: float,
) -> Polygon | MultiPolygon | None:
    if slab_union is None or slab_union.is_empty:
        return None

    routing_area = slab_union
    if boundary_inset > 0:
        inset = normalize_polygon(slab_union.buffer(-boundary_inset))
        if inset is not None and not inset.is_empty:
            routing_area = inset

    blockers: list[Polygon | MultiPolygon] = []
    if walls_union is not None and not walls_union.is_empty:
        blockers.append(walls_union.buffer(wall_clearance))
    if columns_union is not None and not columns_union.is_empty and column_clearance > 0:
        blockers.append(columns_union.buffer(column_clearance))
    if stairs_union is not None and not stairs_union.is_empty:
        blockers.append(stairs_union.buffer(stair_clearance))

    if blockers:
        routing_area = normalize_polygon(routing_area.difference(unary_union(blockers)))

    if routing_area is None or routing_area.is_empty:
        return None

    kept = [poly for poly in extract_polygon_components(routing_area) if poly.area >= min_component_area]
    if not kept:
        return None
    return normalize_polygon(unary_union(kept))


def polygon_to_mask(
    geom: Polygon | MultiPolygon,
    resolution: float,
    margin: float = 1.0,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    minx, miny, maxx, maxy = geom.bounds
    minx -= margin
    miny -= margin
    maxx += margin
    maxy += margin

    width = max(1, int(math.ceil((maxx - minx) / resolution)))
    height = max(1, int(math.ceil((maxy - miny) / resolution)))

    xs = minx + (np.arange(width) + 0.5) * resolution
    ys = miny + (np.arange(height) + 0.5) * resolution
    xv, yv = np.meshgrid(xs, ys)

    mask = np.zeros((height, width), dtype=bool)
    polys = extract_polygon_components(geom)
    for row in range(height):
        pts = [Point(float(x), float(y)) for x, y in zip(xv[row], yv[row])]
        row_mask = np.zeros(width, dtype=bool)
        for poly in polys:
            row_mask |= np.array([poly.covers(pt) for pt in pts], dtype=bool)
        mask[row] = row_mask

    return mask, (minx, miny, maxx, maxy)


def skeleton_graph_from_mask(skel: np.ndarray) -> nx.Graph:
    g = nx.Graph()
    rows, cols = np.where(skel)
    active = set(zip(rows.tolist(), cols.tolist()))
    neighbors = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]
    for r, c in active:
        g.add_node((r, c))
        for dr, dc in neighbors:
            nr, nc = r + dr, c + dc
            if (nr, nc) in active:
                g.add_edge((r, c), (nr, nc), weight=math.hypot(dr, dc))
    return g


def largest_graph_component(g: nx.Graph) -> nx.Graph:
    if g.number_of_nodes() == 0:
        return g
    comp_nodes = max(nx.connected_components(g), key=len)
    return g.subgraph(comp_nodes).copy()


def farthest_node(g: nx.Graph, source: tuple[int, int]) -> tuple[tuple[int, int], dict[tuple[int, int], float]]:
    lengths = nx.single_source_dijkstra_path_length(g, source, weight="weight")
    target = max(lengths.items(), key=lambda x: x[1])[0]
    return target, lengths


def longest_skeleton_path(g: nx.Graph) -> list[tuple[int, int]]:
    if g.number_of_nodes() == 0:
        return []

    degrees = dict(g.degree())
    endpoints = [n for n, d in degrees.items() if d == 1]
    if not endpoints:
        endpoints = list(g.nodes)

    start = endpoints[0]
    far_a, _ = farthest_node(g, start)
    far_b, _ = farthest_node(g, far_a)
    return nx.shortest_path(g, far_a, far_b, weight="weight")


def pixel_path_to_world(path: list[tuple[int, int]], bounds: tuple[float, float, float, float], resolution: float) -> list[tuple[float, float]]:
    minx, miny, _, _ = bounds
    coords: list[tuple[float, float]] = []
    for r, c in path:
        x = minx + (c + 0.5) * resolution
        y = miny + (r + 0.5) * resolution
        coords.append((float(x), float(y)))
    return coords


def orthogonalize_polyline(points: list[tuple[float, float]], min_run: float = 0.75) -> LineString | None:
    if len(points) < 2:
        return None

    simplified = LineString(points).simplify(min_run, preserve_topology=False)
    coords = list(simplified.coords)
    if len(coords) < 2:
        coords = points

    ortho: list[tuple[float, float]] = [tuple(coords[0])]
    for p in coords[1:]:
        x0, y0 = ortho[-1]
        x1, y1 = p
        dx = x1 - x0
        dy = y1 - y0
        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            continue
        if abs(dx) >= abs(dy):
            candidate = (x1, y0)
        else:
            candidate = (x0, y1)
        if math.hypot(candidate[0] - x0, candidate[1] - y0) >= 1e-6:
            ortho.append(candidate)
        if math.hypot(x1 - ortho[-1][0], y1 - ortho[-1][1]) >= 1e-6:
            ortho.append((x1, y1))

    # collapse collinear / tiny segments
    cleaned: list[tuple[float, float]] = [ortho[0]]
    for p in ortho[1:]:
        if math.hypot(p[0] - cleaned[-1][0], p[1] - cleaned[-1][1]) < 1e-6:
            continue
        cleaned.append(p)

    reduced: list[tuple[float, float]] = []
    for p in cleaned:
        if len(reduced) < 2:
            reduced.append(p)
            continue
        x0, y0 = reduced[-2]
        x1, y1 = reduced[-1]
        x2, y2 = p
        if (abs(x0 - x1) < 1e-6 and abs(x1 - x2) < 1e-6) or (abs(y0 - y1) < 1e-6 and abs(y1 - y2) < 1e-6):
            reduced[-1] = p
        else:
            reduced.append(p)

    if len(reduced) < 2:
        return None
    return LineString(reduced)


def clip_line_to_polygon_segments(line: LineString, poly: Polygon | MultiPolygon | None) -> LineString | None:
    if poly is None or poly.is_empty or line.is_empty:
        return None
    clipped = line.intersection(poly)
    if clipped.is_empty:
        return None
    if isinstance(clipped, LineString):
        return clipped if clipped.length > 0.5 else None
    if isinstance(clipped, MultiLineString):
        longest = max(clipped.geoms, key=lambda g: g.length, default=None)
        return longest if longest is not None and longest.length > 0.5 else None
    merged = linemerge(clipped)
    if isinstance(merged, LineString):
        return merged if merged.length > 0.5 else None
    if isinstance(merged, MultiLineString):
        longest = max(merged.geoms, key=lambda g: g.length, default=None)
        return longest if longest is not None and longest.length > 0.5 else None
    return None


def infer_trunk_line_from_corridor(
    routing_area: Polygon | MultiPolygon | None,
    raster_resolution: float,
    simplify_tolerance: float,
) -> LineString | None:
    if routing_area is None or routing_area.is_empty:
        return None

    mask, bounds = polygon_to_mask(routing_area, resolution=raster_resolution, margin=1.0)
    if mask.sum() == 0:
        return None

    # Remove tiny holes and get a more corridor-like band.
    mask = ndi.binary_opening(mask, structure=np.ones((3, 3), dtype=bool))
    skel = skeletonize(mask)
    if skel.sum() == 0:
        return None

    g = largest_graph_component(skeleton_graph_from_mask(skel))
    path = longest_skeleton_path(g)
    if len(path) < 2:
        return None

    world_points = pixel_path_to_world(path, bounds, raster_resolution)
    line = orthogonalize_polyline(world_points, min_run=max(simplify_tolerance, raster_resolution * 2.0))
    if line is None:
        line = LineString(world_points)
    line = LineString(line.simplify(simplify_tolerance, preserve_topology=False).coords)
    return clip_line_to_polygon_segments(line, routing_area)


# ---------------------------
# Reporting / preview
# ---------------------------
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
    routing_area: Polygon | MultiPolygon | None,
    column_items: list[ElementFootprint],
    stair_items: list[ElementFootprint],
    wall_items: list[ElementFootprint],
    trunk_line: LineString | None,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 8))

    for i, slab in enumerate(slab_items):
        draw_geom(ax, slab.polygon, color="#4f9dda", alpha=0.35, label="Slab footprint" if i == 0 else "")
    draw_geom(ax, unified_area, color="#24577a", alpha=0.10, label="Unified protected area")
    draw_geom(ax, routing_area, color="#cbd5e1", alpha=0.22, label="Routing candidate")
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


# ---------------------------
# Main
# ---------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Detect IFC floor geometry and infer a corridor-based trunk line.")
    parser.add_argument("--ifc", type=str, default=None, help="Path to IFC file. If omitted, script auto-detects.")
    parser.add_argument("--storey", type=str, default=TARGET_STOREY_NAME, help="Target IfcBuildingStorey name.")
    parser.add_argument("--output-dir", type=str, default="outputs/output", help="Output directory for JSON and PNG.")
    parser.add_argument("--wall-clearance", type=float, default=0.45, help="Wall buffer for routing candidate area (m).")
    parser.add_argument("--column-clearance", type=float, default=0.20, help="Column buffer for routing candidate area (m).")
    parser.add_argument("--stair-clearance", type=float, default=0.80, help="Stair buffer for routing candidate area (m).")
    parser.add_argument("--boundary-inset", type=float, default=0.40, help="Inset from slab boundary to keep trunk off perimeter (m).")
    parser.add_argument("--min-routing-area", type=float, default=8.0, help="Drop tiny routing pockets smaller than this area (m²).")
    parser.add_argument("--raster-resolution", type=float, default=0.20, help="Skeleton raster resolution in world units (m / pixel).")
    parser.add_argument("--simplify-tolerance", type=float, default=0.60, help="Polyline simplification tolerance (m).")
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
    routing_area = build_routing_candidate_area(
        slab_union=slab_union,
        walls_union=all_walls_union,
        columns_union=column_union,
        stairs_union=stair_union,
        wall_clearance=args.wall_clearance,
        column_clearance=args.column_clearance,
        stair_clearance=args.stair_clearance,
        boundary_inset=args.boundary_inset,
        min_component_area=args.min_routing_area,
    )
    trunk_line = infer_trunk_line_from_corridor(
        routing_area=routing_area,
        raster_resolution=args.raster_resolution,
        simplify_tolerance=args.simplify_tolerance,
    )

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
        "routing_candidate_area": geometry_to_json_dict(routing_area),
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
        "routing_parameters": {
            "wall_clearance": args.wall_clearance,
            "column_clearance": args.column_clearance,
            "stair_clearance": args.stair_clearance,
            "boundary_inset": args.boundary_inset,
            "min_routing_area": args.min_routing_area,
            "raster_resolution": args.raster_resolution,
            "simplify_tolerance": args.simplify_tolerance,
        },
        "suggested_trunk_line": list(trunk_line.coords) if trunk_line is not None else None,
        "suggested_trunk_line_geojson": geometry_to_json_dict(trunk_line),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    save_preview(out_png, slabs, slab_union, routing_area, columns, stairs, wall_std + wall_generic, trunk_line)

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
    print("Routing inference summary:")
    if routing_area is not None:
        print(f"- Routing candidate area: {routing_area.area:.2f} m²")
    else:
        print("- Routing candidate area: none")
    if trunk_line is not None:
        print(f"- Trunk length: {trunk_line.length:.2f} m")
        print(f"- Trunk coords: {list(map(lambda p: (round(p[0], 2), round(p[1], 2)), trunk_line.coords))}")
    else:
        print("- Trunk line: none")
    print()
    print(f"Saved JSON: {out_json}")
    print(f"Saved preview: {out_png}")


if __name__ == "__main__":
    main()
