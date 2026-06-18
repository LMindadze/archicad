from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple, Union

import cv2
import ifcopenshell
import ifcopenshell.geom
import numpy as np
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.ops import unary_union

from sprinkler_hd_gan.semantics import HazardClass, SemanticRGB, paper_outdoor_tint_bgr


@dataclass
class ElementFootprint:
    ifc_id: int
    global_id: Optional[str]
    ifc_class: str
    name: Optional[str]
    storey: Optional[str]
    polygon: Union[Polygon, MultiPolygon, None]
    error: Optional[str] = None


def build_geom_settings() -> ifcopenshell.geom.settings:
    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)
    return settings


def normalize_polygon(geom: Any) -> Union[Polygon, MultiPolygon, None]:
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
        return normalize_polygon(unary_union(polys))
    return None


def shape_to_2d_footprint(
    elem: Any,
    settings: ifcopenshell.geom.settings,
) -> Tuple[Union[Polygon, MultiPolygon, None], Optional[str]]:
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
            tri = Polygon(tri_xy)
            if tri.is_valid and not tri.is_empty and tri.area > 1e-8:
                triangles.append(tri)
        except Exception:  # noqa: BLE001
            continue

    if not triangles:
        return None, "no valid triangles in XY projection"

    footprint = normalize_polygon(unary_union(triangles))
    if footprint is None:
        return None, "failed to build valid polygon footprint"
    return footprint, None


def get_storey_name(element: Any) -> Optional[str]:
    for rel in getattr(element, "ContainedInStructure", []) or []:
        structure = getattr(rel, "RelatingStructure", None)
        if structure and structure.is_a("IfcBuildingStorey"):
            return structure.Name

    for rel in getattr(element, "Decomposes", []) or []:
        parent = getattr(rel, "RelatingObject", None)
        if parent and parent.is_a("IfcBuildingStorey"):
            return parent.Name
    return None


def list_storey_names(model: Any) -> list[str]:
    out: list[str] = []
    for s in model.by_type("IfcBuildingStorey"):
        n = getattr(s, "Name", None)
        if n:
            out.append(str(n))
    return sorted(set(out))


def resolve_target_storey(model: Any, storey: str) -> str:
    """Match exact name, or fuzzy '-2' / 'level -2' style hints."""
    names = list_storey_names(model)
    if storey in names:
        return storey

    s = storey.strip()
    if s in names:
        return s

    sl = s.lower()
    if sl in ("-2", "-2 floor", "basement 2", "b2"):
        for n in names:
            nl = n.lower()
            if "-2" in n or "−2" in n:
                return n
        for n in names:
            if re.search(r"\b-2\b", n) or re.search(r"\b2\s*\.\s*story", nl):
                return n

    for n in names:
        if sl and sl in n.lower():
            return n

    raise ValueError(f"Unknown storey {storey!r}. Available: {names}")


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


def polygons_union(items: Iterable[ElementFootprint]) -> Union[Polygon, MultiPolygon, None]:
    polys = [it.polygon for it in items if it.polygon is not None]
    if not polys:
        return None
    return normalize_polygon(unary_union(polys))


def _ring_to_cvpts(ring: Any, world_to_px: Any) -> np.ndarray:
    pts = []
    for x, y in ring.coords:
        px, py = world_to_px(float(x), float(y))
        pts.append([px, py])
    return np.array([pts], dtype=np.int32)


def fill_polygon_with_holes(
    img: np.ndarray,
    geom: Union[Polygon, MultiPolygon],
    fill: Tuple[int, int, int],
    hole_fill: Tuple[int, int, int],
    world_to_px: Any,
) -> None:
    if isinstance(geom, MultiPolygon):
        for g in geom.geoms:
            fill_polygon_with_holes(img, g, fill, hole_fill, world_to_px)
        return
    if not isinstance(geom, Polygon):
        return
    cv2.fillPoly(img, [_ring_to_cvpts(geom.exterior, world_to_px)], fill)
    for hole in geom.interiors:
        cv2.fillPoly(img, [_ring_to_cvpts(hole, world_to_px)], hole_fill)


def build_semantic_bgr_from_floor(
    slab: Union[Polygon, MultiPolygon, None],
    obstacles: Union[Polygon, MultiPolygon, None],
    riser: Union[Polygon, MultiPolygon, None],
    *,
    width: int,
    height: int,
    mm_per_pixel: float,
    margin_m: float,
    hazard: HazardClass,
) -> np.ndarray:
    """Raster IFC floor geometry into paper-style BGR semantic image."""
    if slab is None or slab.is_empty:
        raise ValueError("No slab footprint for this storey — check IFC / storey name.")

    minx, miny, maxx, maxy = slab.bounds
    minx -= margin_m
    miny -= margin_m
    maxx += margin_m
    maxy += margin_m
    w_m = maxx - minx
    h_m = maxy - miny
    if w_m <= 0 or h_m <= 0:
        raise ValueError("Invalid slab bounds")

    m_per_px = mm_per_pixel / 1000.0
    nat_w = w_m / m_per_px
    nat_h = h_m / m_per_px
    scale = min(width / nat_w, height / nat_h)
    scaled_w = nat_w * scale
    scaled_h = nat_h * scale
    offx = (width - scaled_w) / 2.0
    offy = (height - scaled_h) / 2.0

    outdoor = paper_outdoor_tint_bgr(hazard)

    def world_to_px(x: float, y: float) -> Tuple[int, int]:
        px = (x - minx) / m_per_px * scale + offx
        py = (maxy - y) / m_per_px * scale + offy
        return int(round(px)), int(round(py))

    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :] = outdoor

    fill_polygon_with_holes(img, slab, SemanticRGB.SPRINKLERED_SPACE, outdoor, world_to_px)
    if obstacles is not None and not obstacles.is_empty:
        fill_polygon_with_holes(
            img,
            obstacles,
            SemanticRGB.NON_SPRINKLERED,
            SemanticRGB.SPRINKLERED_SPACE,
            world_to_px,
        )
    if riser is not None and not riser.is_empty:
        fill_polygon_with_holes(
            img,
            riser,
            SemanticRGB.RISER_SHAFT,
            SemanticRGB.SPRINKLERED_SPACE,
            world_to_px,
        )

    return img


def export_storey_semantic(
    ifc_path: Path,
    storey: str,
    *,
    width: int = 640,
    height: int = 512,
    mm_per_pixel: float = 100.0,
    margin_m: float = 0.5,
    hazard: HazardClass = HazardClass.MEDIUM,
    riser_name_contains: Optional[str] = None,
) -> tuple[np.ndarray, str, dict[str, Any]]:
    """
    Returns (semantic_bgr, resolved_storey_name, debug_info).
    Requires `ifcopenshell` and `shapely` (install: pip install -e ".[ifc]").
    """
    model = ifcopenshell.open(str(ifc_path))
    resolved = resolve_target_storey(model, storey)
    settings = build_geom_settings()

    slabs, _, _ = extract_elements(model, settings, "IfcSlab", resolved)
    wall_std, _, _ = extract_elements(model, settings, "IfcWallStandardCase", resolved)
    wall_generic, _, _ = extract_elements(model, settings, "IfcWall", resolved, exact_class_only=True)
    columns, _, _ = extract_elements(model, settings, "IfcColumn", resolved)
    stairs, _, _ = extract_elements(model, settings, "IfcStair", resolved)

    slab_union = polygons_union(slabs)
    wall_u = polygons_union(wall_std + wall_generic)
    col_u = polygons_union(columns)
    stair_u = polygons_union(stairs)

    obstacle_parts = [g for g in [wall_u, col_u, stair_u] if g is not None and not g.is_empty]
    obstacles = normalize_polygon(unary_union(obstacle_parts)) if obstacle_parts else None

    riser_union: Optional[Union[Polygon, MultiPolygon]] = None
    if riser_name_contains:
        sub = riser_name_contains.lower()
        spaces, _, _ = extract_elements(model, settings, "IfcSpace", resolved)
        riser_polys = [e.polygon for e in spaces if e.polygon and e.name and sub in e.name.lower()]
        if riser_polys:
            riser_union = normalize_polygon(unary_union(riser_polys))

    img = build_semantic_bgr_from_floor(
        slab_union,
        obstacles,
        riser_union,
        width=width,
        height=height,
        mm_per_pixel=mm_per_pixel,
        margin_m=margin_m,
        hazard=hazard,
    )

    debug = {
        "resolved_storey": resolved,
        "slab_area_m2": float(slab_union.area) if slab_union is not None else None,
        "n_slabs": len(slabs),
        "n_walls": len(wall_std) + len(wall_generic),
        "n_columns": len(columns),
        "n_stairs": len(stairs),
    }
    return img, resolved, debug
