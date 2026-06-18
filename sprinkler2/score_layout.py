from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from shapely.geometry import GeometryCollection, LineString, MultiLineString, MultiPolygon, Point, Polygon
from shapely.ops import unary_union


def geometry_from_json(data: dict[str, Any] | None) -> Polygon | MultiPolygon | None:
    if data is None:
        return None
    geom_type = data.get("type")
    if geom_type == "Polygon":
        exterior = data.get("exterior", [])
        holes = data.get("holes", [])
        if len(exterior) < 3:
            return None
        return Polygon(exterior, holes=holes)
    if geom_type == "MultiPolygon":
        polys: list[Polygon] = []
        for part in data.get("parts", []):
            geom = geometry_from_json(part)
            if isinstance(geom, Polygon) and not geom.is_empty:
                polys.append(geom)
            elif isinstance(geom, MultiPolygon):
                polys.extend([poly for poly in geom.geoms if not poly.is_empty])
        return MultiPolygon(polys) if polys else None
    return None


def normalize_polygon(geom: Any) -> Polygon | MultiPolygon | None:
    if geom is None or geom.is_empty:
        return None
    cleaned = geom.buffer(0)
    if cleaned.is_empty:
        return None
    if isinstance(cleaned, (Polygon, MultiPolygon)):
        return cleaned
    if isinstance(cleaned, GeometryCollection):
        polys = [part for part in cleaned.geoms if isinstance(part, (Polygon, MultiPolygon)) and not part.is_empty]
        if polys:
            return normalize_polygon(unary_union(polys))
    return None


def build_exclusion_area(
    detected: dict[str, Any],
    column_buffer: float,
    stair_buffer: float,
    wall_clearance: float,
) -> Polygon | MultiPolygon | None:
    columns = geometry_from_json(detected.get("columns_union"))
    stairs = geometry_from_json(detected.get("stairs_union"))
    walls = geometry_from_json(detected.get("walls_all_union"))
    parts: list[Polygon | MultiPolygon] = []
    if columns is not None and not columns.is_empty and column_buffer > 0:
        parts.append(columns.buffer(column_buffer))
    if stairs is not None and not stairs.is_empty and stair_buffer > 0:
        parts.append(stairs.buffer(stair_buffer))
    if walls is not None and not walls.is_empty and wall_clearance > 0:
        parts.append(walls.buffer(wall_clearance))
    return normalize_polygon(unary_union(parts)) if parts else None


def sample_points_inside_polygon(geom: Polygon | MultiPolygon | None, step: float) -> list[Point]:
    if geom is None or geom.is_empty:
        return []
    step = max(float(step), 0.25)
    minx, miny, maxx, maxy = geom.bounds
    pts: list[Point] = []
    for x in np.arange(minx, maxx + step * 0.5, step):
        for y in np.arange(miny, maxy + step * 0.5, step):
            p = Point(float(x), float(y))
            if geom.contains(p):
                pts.append(p)
    return pts


def sample_boundary_points(geom: Polygon | MultiPolygon | None, step: float) -> list[Point]:
    if geom is None or geom.is_empty:
        return []
    step = max(float(step), 0.25)
    polys = [geom] if isinstance(geom, Polygon) else list(geom.geoms)
    lines: list[LineString] = []
    for poly in polys:
        lines.append(LineString(poly.exterior.coords))
        for ring in poly.interiors:
            lines.append(LineString(ring.coords))
    pts: list[Point] = []
    for line in lines:
        if line.length <= 0:
            continue
        n = max(1, int(math.ceil(line.length / step)))
        for idx in range(n + 1):
            pts.append(line.interpolate(min(line.length, idx * step)))
    return pts


def lines_from_coords(items: list[Any]) -> list[LineString]:
    out: list[LineString] = []
    for coords in items:
        if len(coords) >= 2:
            out.append(LineString([(float(x), float(y)) for x, y in coords]))
    return out


def trunk_lines_from_layout(layout: dict[str, Any]) -> list[LineString]:
    geoms = layout.get("geometries") or {}
    lines: list[LineString] = []
    for segment in geoms.get("trunk_segments") or []:
        start = segment.get("start") or []
        end = segment.get("end") or []
        if len(start) >= 2 and len(end) >= 2:
            lines.append(LineString([(float(start[0]), float(start[1])), (float(end[0]), float(end[1]))]))
    if lines:
        return lines
    trunk = geoms.get("trunk_line") or geoms.get("main_trunk_line") or []
    if len(trunk) >= 2:
        coords = [(float(x), float(y)) for x, y in trunk]
        for left, right in zip(coords, coords[1:]):
            lines.append(LineString([left, right]))
    return lines


def graph_connectivity(branches: list[LineString], trunks: list[LineString], tolerance: float) -> dict[str, Any]:
    lines = trunks + branches
    if not lines:
        return {"component_count": 0, "disconnected_branch_lines": 0}
    parent = list(range(len(lines)))

    def find(idx: int) -> int:
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = parent[idx]
        return idx

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left in range(len(lines)):
        for right in range(left + 1, len(lines)):
            if lines[left].intersects(lines[right]) or lines[left].distance(lines[right]) <= tolerance:
                union(left, right)
    trunk_roots = {find(idx) for idx in range(len(trunks))}
    disconnected = []
    for branch_idx in range(len(branches)):
        if find(len(trunks) + branch_idx) not in trunk_roots:
            disconnected.append(branch_idx)
    return {
        "component_count": len({find(idx) for idx in range(len(lines))}),
        "trunk_component_count": len(trunk_roots),
        "disconnected_branch_lines": len(disconnected),
        "disconnected_branch_indices": disconnected[:50],
    }


def score_layout(
    detected: dict[str, Any],
    layout: dict[str, Any],
    *,
    demand_step: float,
    generator_cover_radius: float,
    nfpa_max_spacing: float,
    nfpa_min_spacing: float,
    nfpa_max_wall_distance: float,
    nfpa_min_wall_distance: float,
    min_obstruction_clearance: float,
    max_avg_area_per_head: float,
    column_clearance: float,
    stair_clearance: float,
    wall_clearance: float,
) -> dict[str, Any]:
    geoms = layout.get("geometries") or {}
    protected = normalize_polygon(geometry_from_json(detected.get("unified_protected_floor_area")))
    if protected is None:
        raise RuntimeError("No protected floor area found.")
    exclusion = build_exclusion_area(detected, column_clearance, stair_clearance, wall_clearance)
    valid_area = protected
    if exclusion is not None and not exclusion.is_empty:
        diff = normalize_polygon(protected.difference(exclusion))
        if diff is not None and not diff.is_empty:
            valid_area = diff

    heads = [Point(float(item["x"]), float(item["y"])) for item in geoms.get("sprinkler_heads") or []]
    branches = lines_from_coords(geoms.get("branch_lines") or [])
    trunks = trunk_lines_from_layout(layout)

    samples = sample_points_inside_polygon(valid_area, demand_step)

    def coverage(radius: float) -> dict[str, Any]:
        covered = sum(1 for sample in samples if any(sample.distance(head) <= radius for head in heads))
        return {
            "radius_m": float(radius),
            "covered_points": int(covered),
            "sample_points": len(samples),
            "ratio": float(covered / max(1, len(samples))),
        }

    generator_coverage = coverage(generator_cover_radius)
    nfpa_coverage = coverage(0.5 * nfpa_max_spacing)

    min_pair = None
    too_close_pairs = 0
    max_nearest = 0.0
    too_far_heads = 0
    for left in range(len(heads)):
        nearest = None
        for right in range(len(heads)):
            if left == right:
                continue
            distance = heads[left].distance(heads[right])
            if min_pair is None or distance < min_pair:
                min_pair = distance
            if nearest is None or distance < nearest:
                nearest = distance
            if distance < nfpa_min_spacing - 1e-6:
                too_close_pairs += 1
        if nearest is not None:
            max_nearest = max(max_nearest, nearest)
            if nearest > nfpa_max_spacing + 1e-6:
                too_far_heads += 1
    too_close_pairs //= 2

    boundary_points = sample_boundary_points(valid_area, max(0.5, demand_step))
    uncovered_boundary = 0
    for sample in boundary_points:
        if not any(sample.distance(head) <= nfpa_max_wall_distance for head in heads):
            uncovered_boundary += 1

    wall_distances = [head.distance(protected.boundary) for head in heads] if heads else []
    near_wall_too_close = sum(1 for distance in wall_distances if distance < nfpa_min_wall_distance - 1e-6)
    far_from_wall = sum(1 for distance in wall_distances if distance > nfpa_max_wall_distance + 1e-6)

    obstruction_violations = 0
    min_obstruction_distance = None
    if exclusion is not None and not exclusion.is_empty:
        for head in heads:
            distance = head.distance(exclusion)
            if min_obstruction_distance is None or distance < min_obstruction_distance:
                min_obstruction_distance = distance
            if distance < min_obstruction_clearance - 1e-6:
                obstruction_violations += 1

    valid_area_m2 = float(valid_area.area)
    avg_area_per_head = valid_area_m2 / max(1, len(heads))
    branch_length = float(sum(line.length for line in branches))
    trunk_length = float(sum(line.length for line in trunks))
    connectivity = graph_connectivity(branches, trunks, tolerance=0.08)

    hard_failures = {
        "disconnected_branch_lines": connectivity["disconnected_branch_lines"],
        "too_close_pairs": too_close_pairs,
        "obstruction_violations": obstruction_violations,
    }
    score = 100.0
    score -= max(0.0, 0.96 - generator_coverage["ratio"]) * 65.0
    score -= max(0.0, avg_area_per_head - max_avg_area_per_head) * 1.5
    score -= min(20.0, too_close_pairs * 1.5)
    score -= min(12.0, too_far_heads * 0.7)
    score -= min(12.0, uncovered_boundary * 0.08)
    score -= min(20.0, connectivity["disconnected_branch_lines"] * 2.0)
    score = max(0.0, min(100.0, score))

    recommendations: list[str] = []
    if generator_coverage["ratio"] < 0.90:
        recommendations.append("Increase selected branch packages or reduce CP-SAT head/branch penalties; coverage is the main gap.")
    if avg_area_per_head > max_avg_area_per_head:
        recommendations.append("Add heads or split uncovered zones; average protected area per head is above target.")
    if too_close_pairs:
        recommendations.append("Keep NFPA min spacing constraint; do not add heads too close to existing heads.")
    if uncovered_boundary:
        recommendations.append("Boundary/wall-adjacent sample points remain uncovered; add perimeter-support packages where spacing allows.")
    if connectivity["disconnected_branch_lines"]:
        recommendations.append("Run main-trunk connector post-process before accepting a candidate.")
    if not recommendations:
        recommendations.append("No high-priority geometric gaps found under current scoring.")

    return {
        "score": round(score, 2),
        "counts": {
            "heads": len(heads),
            "branch_lines": len(branches),
            "trunk_lines": len(trunks),
            "total_branch_length_m": branch_length,
            "total_trunk_length_m": trunk_length,
            "valid_area_m2": valid_area_m2,
        },
        "coverage": {
            "generator_radius": generator_coverage,
            "nfpa_half_max_spacing_radius": nfpa_coverage,
        },
        "spacing": {
            "min_pair_distance_m": min_pair,
            "too_close_pairs": too_close_pairs,
            "max_nearest_neighbor_m": max_nearest,
            "too_far_heads": too_far_heads,
            "min_spacing_limit_m": nfpa_min_spacing,
            "max_spacing_limit_m": nfpa_max_spacing,
        },
        "wall_distance": {
            "near_wall_too_close_heads": near_wall_too_close,
            "far_from_wall_heads": far_from_wall,
            "min_wall_distance_observed_m": min(wall_distances) if wall_distances else None,
            "max_wall_distance_observed_m": max(wall_distances) if wall_distances else None,
            "boundary_sample_points": len(boundary_points),
            "uncovered_boundary_points": uncovered_boundary,
            "uncovered_boundary_ratio": uncovered_boundary / max(1, len(boundary_points)),
        },
        "obstruction_clearance": {
            "violating_heads": obstruction_violations,
            "min_distance_to_exclusion_m": min_obstruction_distance,
            "required_clearance_m": min_obstruction_clearance,
        },
        "area_per_head": {
            "avg_area_per_head_m2": avg_area_per_head,
            "limit_m2": max_avg_area_per_head,
        },
        "connectivity": connectivity,
        "hard_failures": hard_failures,
        "recommendations": recommendations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score sprinkler layout coverage, spacing, clearances, and trunk connectivity.")
    parser.add_argument("--detected-json", default="outputs/output/detected_geometry.json")
    parser.add_argument("--layout-json", required=True)
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--demand-step", type=float, default=1.0)
    parser.add_argument("--generator-cover-radius", type=float, default=1.92)
    parser.add_argument("--nfpa-max-spacing", type=float, default=4.572)
    parser.add_argument("--nfpa-min-spacing", type=float, default=1.8288)
    parser.add_argument("--nfpa-max-wall-distance", type=float, default=2.286)
    parser.add_argument("--nfpa-min-wall-distance", type=float, default=0.1016)
    parser.add_argument("--min-obstruction-clearance", type=float, default=0.2)
    parser.add_argument("--max-avg-area-per-head", type=float, default=12.1)
    parser.add_argument("--column-clearance", type=float, default=0.55)
    parser.add_argument("--stair-clearance", type=float, default=0.8)
    parser.add_argument("--wall-clearance", type=float, default=0.3)
    args = parser.parse_args()

    detected = json.loads(Path(args.detected_json).read_text(encoding="utf-8"))
    layout = json.loads(Path(args.layout_json).read_text(encoding="utf-8"))
    score = score_layout(
        detected,
        layout,
        demand_step=args.demand_step,
        generator_cover_radius=args.generator_cover_radius,
        nfpa_max_spacing=args.nfpa_max_spacing,
        nfpa_min_spacing=args.nfpa_min_spacing,
        nfpa_max_wall_distance=args.nfpa_max_wall_distance,
        nfpa_min_wall_distance=args.nfpa_min_wall_distance,
        min_obstruction_clearance=args.min_obstruction_clearance,
        max_avg_area_per_head=args.max_avg_area_per_head,
        column_clearance=args.column_clearance,
        stair_clearance=args.stair_clearance,
        wall_clearance=args.wall_clearance,
    )
    text = json.dumps(score, indent=2, ensure_ascii=False)
    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
