from __future__ import annotations

import argparse
import heapq
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import GeometryCollection, LineString, MultiLineString, MultiPolygon, Point, Polygon
from shapely.ops import linemerge, substring, unary_union


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
        parts = data.get("parts", [])
        polys = []
        for p in parts:
            g = geometry_from_json(p)
            if isinstance(g, Polygon) and not g.is_empty:
                polys.append(g)
            elif isinstance(g, MultiPolygon):
                polys.extend(list(g.geoms))
        if not polys:
            return None
        return MultiPolygon(polys)
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
        polys = [g for g in cleaned.geoms if isinstance(g, (Polygon, MultiPolygon)) and not g.is_empty]
        if polys:
            return normalize_polygon(unary_union(polys))
    return None


def geometry_to_json_dict(geom: Polygon | MultiPolygon | None) -> dict[str, Any] | None:
    if geom is None or geom.is_empty:
        return None
    if isinstance(geom, Polygon):
        return {
            "type": "Polygon",
            "exterior": [list(c) for c in geom.exterior.coords],
            "holes": [[list(c) for c in ring.coords] for ring in geom.interiors],
            "area": float(geom.area),
        }
    return {
        "type": "MultiPolygon",
        "parts": [geometry_to_json_dict(p) for p in geom.geoms],
        "area": float(geom.area),
    }


def choose_axes(detected: dict[str, Any], protected_area: Polygon | MultiPolygon) -> tuple[np.ndarray, np.ndarray]:
    axes = detected.get("candidate_axes")
    if axes and axes.get("main_axis") and axes.get("branch_axis"):
        main = np.array(axes["main_axis"]["unit_vector_xy"], dtype=float)
        branch = np.array(axes["branch_axis"]["unit_vector_xy"], dtype=float)
    else:
        mrr = protected_area.minimum_rotated_rectangle
        pts = list(mrr.exterior.coords)
        edges = []
        for i in range(4):
            p0 = np.array(pts[i], dtype=float)
            p1 = np.array(pts[i + 1], dtype=float)
            vec = p1 - p0
            length = float(np.linalg.norm(vec))
            if length > 1e-9:
                edges.append((length, vec / length))
        edges.sort(key=lambda x: x[0], reverse=True)
        main = edges[0][1]
        branch = np.array([-main[1], main[0]], dtype=float)

    main = main / np.linalg.norm(main)
    branch = branch / np.linalg.norm(branch)
    return main, branch


def trunk_line_from_detected(
    detected: dict[str, Any],
    protected_area: Polygon | MultiPolygon,
    main_vec: np.ndarray,
) -> LineString | None:
    suggested = detected.get("suggested_trunk_line")
    if suggested and len(suggested) >= 2:
        line = LineString([tuple(p) for p in suggested])
        clipped = line.intersection(protected_area)
        if isinstance(clipped, LineString) and not clipped.is_empty:
            return clipped
        if isinstance(clipped, MultiLineString):
            segments = sorted(clipped.geoms, key=lambda g: g.length, reverse=True)
            if segments:
                return segments[0]
        merged = linemerge(clipped)
        if isinstance(merged, LineString) and not merged.is_empty:
            return merged
        if isinstance(merged, MultiLineString) and merged.geoms:
            return max(merged.geoms, key=lambda g: g.length)

    c = np.array([protected_area.centroid.x, protected_area.centroid.y], dtype=float)
    hull = np.array(list(protected_area.convex_hull.exterior.coords), dtype=float)
    rel = hull - c
    proj = rel @ main_vec
    p0 = c + main_vec * float(np.min(proj))
    p1 = c + main_vec * float(np.max(proj))
    line = LineString([tuple(p0), tuple(p1)])
    clipped = line.intersection(protected_area)
    if isinstance(clipped, LineString) and not clipped.is_empty:
        return clipped
    if isinstance(clipped, MultiLineString):
        segments = sorted(clipped.geoms, key=lambda g: g.length, reverse=True)
        if segments:
            return segments[0]
    return None


def lines_from_intersection(geom: Any, min_length: float = 0.1) -> list[LineString]:
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, LineString):
        return [geom] if geom.length >= min_length else []
    if isinstance(geom, MultiLineString):
        return [g for g in geom.geoms if g.length >= min_length]
    if hasattr(geom, "geoms"):
        out: list[LineString] = []
        for g in geom.geoms:
            out.extend(lines_from_intersection(g, min_length=min_length))
        return out
    return []


def rays_from_point(
    point_xy: np.ndarray,
    direction_xy: np.ndarray,
    span: float,
) -> tuple[LineString, LineString]:
    unit = direction_xy / (np.linalg.norm(direction_xy) + 1e-12)
    p_pos = point_xy + unit * span
    p_neg = point_xy - unit * span
    return LineString([tuple(point_xy), tuple(p_pos)]), LineString([tuple(point_xy), tuple(p_neg)])


def pick_segment_touching_origin(segments: list[LineString], origin: Point, tol: float = 1e-3) -> LineString | None:
    if not segments:
        return None
    touching = [s for s in segments if s.distance(origin) <= tol]
    if touching:
        return max(touching, key=lambda s: s.length)
    return min(segments, key=lambda s: s.distance(origin))


def build_branch_lines(
    protected_area: Polygon | MultiPolygon,
    trunk: LineString | None,
    main_vec: np.ndarray,
    branch_vec: np.ndarray,
    exclusion_area: Polygon | MultiPolygon | None,
    branch_spacing: float,
    branch_offset: float = 0.0,
    min_branch_length: float = 1.0,
) -> list[LineString]:
    if trunk is None or trunk.is_empty:
        return []

    minx, miny, maxx, maxy = protected_area.bounds
    span_branch = max(maxx - minx, maxy - miny) + 20.0

    branch_lines: list[LineString] = []
    length = float(trunk.length)
    d = max(0.0, branch_offset)
    blocked = exclusion_area if exclusion_area is not None and not exclusion_area.is_empty else None
    while d <= length + 1e-9:
        s = min(d, length)
        pt = trunk.interpolate(s)
        d0 = max(0.0, s - 0.25)
        d1 = min(length, s + 0.25)
        p0 = trunk.interpolate(d0)
        p1 = trunk.interpolate(d1)
        tx = p1.x - p0.x
        ty = p1.y - p0.y
        n = math.hypot(tx, ty)
        if n < 1e-9:
            d += branch_spacing
            continue
        tx /= n
        ty /= n
        perp = np.array([-ty, tx], dtype=float)
        origin = np.array([pt.x, pt.y], dtype=float)
        ray_a, ray_b = rays_from_point(origin, perp, span_branch)
        origin_pt = Point(float(origin[0]), float(origin[1]))

        for ray in (ray_a, ray_b):
            clipped = ray.intersection(protected_area)
            segments = lines_from_intersection(clipped, min_length=min_branch_length)
            seg = pick_segment_touching_origin(segments, origin_pt)
            if seg is None or seg.length < min_branch_length:
                continue
            if blocked is not None:
                seg = seg.difference(blocked)
                cleaned = lines_from_intersection(seg, min_length=min_branch_length)
                seg = pick_segment_touching_origin(cleaned, origin_pt)
                if seg is None or seg.length < min_branch_length:
                    continue
            branch_lines.append(seg)
        d += branch_spacing

    return branch_lines


def points_along_line(line: LineString, spacing: float, endpoint_margin: float = 0.0) -> list[Point]:
    if line.length <= endpoint_margin * 2:
        return []
    usable = line.length - endpoint_margin * 2
    if usable <= 0:
        return []
    n = max(1, int(math.floor(usable / spacing)) + 1)
    pts: list[Point] = []
    for i in range(n):
        dist = endpoint_margin + i * spacing
        if dist > line.length - endpoint_margin + 1e-9:
            break
        pts.append(line.interpolate(dist))
    return pts


def points_along_line_with_phase(
    line: LineString,
    spacing: float,
    phase_offset: float,
    endpoint_margin: float = 0.0,
) -> list[Point]:
    if line.length <= endpoint_margin * 2:
        return []
    start = endpoint_margin + max(0.0, phase_offset)
    end = line.length - endpoint_margin
    if start > end:
        return []
    pts: list[Point] = []
    d = start
    while d <= end + 1e-9:
        pts.append(line.interpolate(d))
        d += spacing
    return pts


def build_exclusion_area(
    detected: dict[str, Any],
    column_buffer: float,
    stair_buffer: float,
    wall_clearance: float,
) -> Polygon | MultiPolygon | None:
    cols = geometry_from_json(detected.get("columns_union"))
    stairs = geometry_from_json(detected.get("stairs_union"))
    walls = geometry_from_json(detected.get("walls_all_union"))

    parts = []
    if cols is not None and not cols.is_empty and column_buffer > 0:
        parts.append(cols.buffer(column_buffer))
    if stairs is not None and not stairs.is_empty:
        parts.append(stairs.buffer(stair_buffer))
    if walls is not None and not walls.is_empty and wall_clearance > 0:
        parts.append(walls.buffer(wall_clearance))
    if not parts:
        return None
    return normalize_polygon(unary_union(parts))


def sample_points_inside_polygon(
    geom: Polygon | MultiPolygon | None,
    step: float,
) -> list[Point]:
    if geom is None or geom.is_empty:
        return []
    minx, miny, maxx, maxy = geom.bounds
    xs = np.arange(minx, maxx + step * 0.5, step)
    ys = np.arange(miny, maxy + step * 0.5, step)
    pts: list[Point] = []
    for x in xs:
        for y in ys:
            p = Point(float(x), float(y))
            if geom.contains(p):
                pts.append(p)
    return pts


def choose_layout_offsets(
    valid_area: Polygon | MultiPolygon | None,
    trunk: LineString | None,
    protected_area: Polygon | MultiPolygon,
    exclusion: Polygon | MultiPolygon | None,
    main_vec: np.ndarray,
    branch_vec: np.ndarray,
    branch_spacing: float,
    head_spacing: float,
    branch_end_margin: float,
    min_obstacle_clearance: float,
) -> tuple[float, float]:
    if trunk is None or trunk.is_empty:
        return 0.0, 0.0

    # Coarse search over phase offsets; maximize area coverage score.
    branch_candidates = [0.0, 0.25 * branch_spacing, 0.5 * branch_spacing, 0.75 * branch_spacing]
    head_candidates = [0.0, 0.25 * head_spacing, 0.5 * head_spacing, 0.75 * head_spacing]
    samples = sample_points_inside_polygon(valid_area, step=1.0)
    if not samples:
        return 0.0, 0.0
    cover_radius = 0.6 * head_spacing

    best_score = -1e9
    best = (0.0, 0.0)
    for boff in branch_candidates:
        branches = build_branch_lines(
            protected_area=protected_area,
            trunk=trunk,
            main_vec=main_vec,
            branch_vec=branch_vec,
            exclusion_area=exclusion,
            branch_spacing=branch_spacing,
            branch_offset=boff,
        )
        for hoff in head_candidates:
            heads: list[Point] = []
            for bl in branches:
                heads.extend(
                    points_along_line_with_phase(
                        bl,
                        spacing=head_spacing,
                        phase_offset=hoff,
                        endpoint_margin=branch_end_margin,
                    )
                )
            kept: list[Point] = []
            for p in heads:
                if valid_area is None or valid_area.is_empty:
                    continue
                if not (valid_area.contains(p) or valid_area.buffer(1e-6).contains(p)):
                    continue
                if exclusion is not None and not exclusion.is_empty and p.distance(exclusion) < min_obstacle_clearance:
                    continue
                kept.append(p)
            if not kept:
                continue

            covered = 0
            for s in samples:
                if any(s.distance(h) <= cover_radius for h in kept):
                    covered += 1
            coverage_ratio = covered / max(1, len(samples))
            # Prefer higher coverage with modest head count.
            score = 1000.0 * coverage_ratio - 0.20 * len(kept)
            if score > best_score:
                best_score = score
                best = (boff, hoff)
    return best


def build_candidate_heads(
    branches: list[LineString],
    head_spacing: float,
    branch_end_margin: float,
    head_offset: float,
    valid_area: Polygon | MultiPolygon | None,
    exclusion: Polygon | MultiPolygon | None,
    min_obstacle_clearance: float,
) -> list[Point]:
    heads: list[Point] = []
    for bl in branches:
        heads.extend(
            points_along_line_with_phase(
                bl,
                head_spacing,
                phase_offset=head_offset,
                endpoint_margin=branch_end_margin,
            )
        )

    kept: list[Point] = []
    for p in heads:
        if valid_area is None or valid_area.is_empty:
            continue
        if not (valid_area.contains(p) or valid_area.buffer(1e-6).contains(p)):
            continue
        if exclusion is not None and not exclusion.is_empty and p.distance(exclusion) < min_obstacle_clearance:
            continue
        kept.append(p)
    return kept


def sample_demand_points(
    valid_area: Polygon | MultiPolygon | None,
    step: float,
) -> list[Point]:
    return sample_points_inside_polygon(valid_area, step=step)


def dedupe_points(points: list[Point], tol: float = 0.12) -> list[Point]:
    if not points:
        return []
    kept: list[Point] = []
    for p in points:
        if all(p.distance(q) > tol for q in kept):
            kept.append(p)
    return kept


def branch_station_candidates(
    trunk: LineString | None,
    branch_spacing: float,
) -> list[float]:
    if trunk is None or trunk.is_empty:
        return []
    length = float(trunk.length)
    stations: list[float] = []
    d = 0.0
    while d <= length + 1e-9:
        stations.append(min(d, length))
        d += max(0.6, branch_spacing * 0.5)
    if length > 0 and (not stations or abs(stations[-1] - length) > 1e-6):
        stations.append(length)
    return stations


def build_branch_from_station(
    protected_area: Polygon | MultiPolygon,
    trunk: LineString,
    station_s: float,
    side: int,
    exclusion_area: Polygon | MultiPolygon | None,
    min_branch_length: float = 1.0,
) -> LineString | None:
    minx, miny, maxx, maxy = protected_area.bounds
    span_branch = max(maxx - minx, maxy - miny) + 20.0

    s = float(max(0.0, min(station_s, trunk.length)))
    pt = trunk.interpolate(s)
    d0 = max(0.0, s - 0.25)
    d1 = min(trunk.length, s + 0.25)
    p0 = trunk.interpolate(d0)
    p1 = trunk.interpolate(d1)
    tx = p1.x - p0.x
    ty = p1.y - p0.y
    n = math.hypot(tx, ty)
    if n < 1e-9:
        return None
    tx /= n
    ty /= n
    perp = np.array([-ty, tx], dtype=float) * (1.0 if side >= 0 else -1.0)
    origin = np.array([pt.x, pt.y], dtype=float)

    ray, _ = rays_from_point(origin, perp, span_branch)
    origin_pt = Point(float(origin[0]), float(origin[1]))
    clipped = ray.intersection(protected_area)
    segments = lines_from_intersection(clipped, min_length=min_branch_length)
    seg = pick_segment_touching_origin(segments, origin_pt)
    if seg is None or seg.length < min_branch_length:
        return None
    if exclusion_area is not None and not exclusion_area.is_empty:
        seg = seg.difference(exclusion_area)
        cleaned = lines_from_intersection(seg, min_length=min_branch_length)
        seg = pick_segment_touching_origin(cleaned, origin_pt)
        if seg is None or seg.length < min_branch_length:
            return None
    return seg


def heads_for_branch(
    branch: LineString,
    head_spacing: float,
    branch_end_margin: float,
    head_offset: float,
    valid_area: Polygon | MultiPolygon | None,
    exclusion: Polygon | MultiPolygon | None,
    min_obstacle_clearance: float,
) -> list[Point]:
    return build_candidate_heads(
        branches=[branch],
        head_spacing=head_spacing,
        branch_end_margin=branch_end_margin,
        head_offset=head_offset,
        valid_area=valid_area,
        exclusion=exclusion,
        min_obstacle_clearance=min_obstacle_clearance,
    )


def adaptive_branch_and_head_layout(
    protected_area: Polygon | MultiPolygon,
    valid_area: Polygon | MultiPolygon | None,
    trunk: LineString | None,
    exclusion: Polygon | MultiPolygon | None,
    branch_spacing: float,
    head_spacing: float,
    branch_end_margin: float,
    min_obstacle_clearance: float,
    demand_step: float = 1.0,
    target_coverage_ratio: float = 0.96,
) -> tuple[list[LineString], list[Point], dict[str, Any]]:
    if trunk is None or trunk.is_empty:
        return [], [], {"achieved_coverage_ratio": 0.0, "target_coverage_ratio": target_coverage_ratio}

    demand = sample_demand_points(valid_area, step=demand_step)
    if not demand:
        return [], [], {"achieved_coverage_ratio": 0.0, "target_coverage_ratio": target_coverage_ratio}
    covered = np.zeros(len(demand), dtype=bool)
    cover_radius = 0.6 * head_spacing

    # Head phasing still matters; evaluate both quickly.
    head_offsets = [0.0, 0.5 * head_spacing]
    stations = branch_station_candidates(trunk, branch_spacing=branch_spacing)
    candidates: list[dict[str, Any]] = []
    for s in stations:
        for side in (-1, 1):
            branch = build_branch_from_station(
                protected_area=protected_area,
                trunk=trunk,
                station_s=s,
                side=side,
                exclusion_area=exclusion,
                min_branch_length=1.0,
            )
            if branch is None:
                continue
            for h_off in head_offsets:
                phase_index = 0 if abs(h_off) < 1e-9 else 1
                branch_heads = heads_for_branch(
                    branch=branch,
                    head_spacing=head_spacing,
                    branch_end_margin=branch_end_margin,
                    head_offset=h_off,
                    valid_area=valid_area,
                    exclusion=exclusion,
                    min_obstacle_clearance=min_obstacle_clearance,
                )
                if not branch_heads:
                    continue
                cover_idx: set[int] = set()
                for i, dp in enumerate(demand):
                    if any(dp.distance(h) <= cover_radius for h in branch_heads):
                        cover_idx.add(i)
                if not cover_idx:
                    continue
                candidates.append(
                    {
                        "station": float(s),
                        "side": int(side),
                        "branch": branch,
                        "heads": branch_heads,
                        "covers": cover_idx,
                        "cost": max(1.0, float(branch.length)) + 0.5 * len(branch_heads),
                    }
                )

    selected_branches: list[LineString] = []
    selected_heads: list[Point] = []
    used_station_side: set[tuple[int, int]] = set()
    while True:
        uncovered_idx = {i for i in range(len(demand)) if not covered[i]}
        if not uncovered_idx:
            break
        if covered.mean() >= target_coverage_ratio:
            break
        best_idx = -1
        best_score = 0.0
        for i, c in enumerate(candidates):
            key = (int(round(c["station"] * 10.0)), int(c["side"]))
            if key in used_station_side:
                continue
            gain = len(c["covers"] & uncovered_idx)
            if gain <= 0:
                continue
            score = gain / c["cost"]
            if score > best_score:
                best_score = score
                best_idx = i
        if best_idx < 0:
            break

        best = candidates[best_idx]
        key = (int(round(best["station"] * 10.0)), int(best["side"]))
        used_station_side.add(key)
        selected_branches.append(best["branch"])
        selected_heads.extend(best["heads"])
        for j in best["covers"]:
            covered[j] = True

    selected_heads = dedupe_points(selected_heads, tol=0.15)
    achieved = float(covered.mean()) if len(demand) else 0.0
    diag = {
        "achieved_coverage_ratio": achieved,
        "target_coverage_ratio": float(target_coverage_ratio),
        "demand_points": int(len(demand)),
        "candidate_branch_variants": int(len(candidates)),
    }
    return selected_branches, selected_heads, diag


def _build_branch_packages_for_cpsat(
    protected_area: Polygon | MultiPolygon,
    valid_area: Polygon | MultiPolygon | None,
    trunk: LineString,
    exclusion: Polygon | MultiPolygon | None,
    branch_spacing: float,
    head_spacing: float,
    branch_end_margin: float,
    min_obstacle_clearance: float,
    demand: list[Point],
    cover_radius: float,
) -> list[dict[str, Any]]:
    """One package = trunk station + side + head phase; heads are fixed along that branch."""
    head_offsets = [0.0, 0.5 * head_spacing]
    stations = branch_station_candidates(trunk, branch_spacing=branch_spacing)
    packages: list[dict[str, Any]] = []
    seen: set[tuple[int, int, float]] = set()
    for s in stations:
        for side in (-1, 1):
            branch = build_branch_from_station(
                protected_area=protected_area,
                trunk=trunk,
                station_s=s,
                side=side,
                exclusion_area=exclusion,
                min_branch_length=1.0,
            )
            if branch is None:
                continue
            for h_off in head_offsets:
                phase_index = 0 if abs(float(h_off)) < 1e-9 else 1
                branch_heads = heads_for_branch(
                    branch=branch,
                    head_spacing=head_spacing,
                    branch_end_margin=branch_end_margin,
                    head_offset=h_off,
                    valid_area=valid_area,
                    exclusion=exclusion,
                    min_obstacle_clearance=min_obstacle_clearance,
                )
                if not branch_heads:
                    continue
                key = (int(round(float(s) * 1000)), int(side), float(h_off))
                if key in seen:
                    continue
                seen.add(key)
                cover_idx: set[int] = set()
                for i, dp in enumerate(demand):
                    if any(dp.distance(h) <= cover_radius for h in branch_heads):
                        cover_idx.add(i)
                if not cover_idx:
                    continue
                packages.append(
                    {
                        "station": float(s),
                        "side": int(side),
                        "head_offset": float(h_off),
                        "phase_index": int(phase_index),
                        "branch": branch,
                        "heads": branch_heads,
                        "covers": cover_idx,
                        "num_heads": len(branch_heads),
                        "branch_length": float(branch.length),
                    }
                )
    return packages


def _package_pair_conflicts(
    pkgs: list[dict[str, Any]],
    min_head_spacing: float,
    min_branch_station_gap: float,
) -> list[tuple[int, int]]:
    """Undirected conflict edges i<j (for x_i + x_j <= 1)."""
    conflicts: list[tuple[int, int]] = []
    n = len(pkgs)
    for i in range(n):
        for j in range(i + 1, n):
            pi, pj = pkgs[i], pkgs[j]
            if pi["side"] == pj["side"] and abs(pi["station"] - pj["station"]) < min_branch_station_gap:
                conflicts.append((i, j))
                continue
            too_close = False
            for ha in pi["heads"]:
                for hb in pj["heads"]:
                    if ha.distance(hb) < min_head_spacing - 1e-6:
                        too_close = True
                        break
                if too_close:
                    break
            if too_close:
                conflicts.append((i, j))
    return conflicts


def cpsat_branch_head_layout(
    protected_area: Polygon | MultiPolygon,
    valid_area: Polygon | MultiPolygon | None,
    trunk: LineString | None,
    exclusion: Polygon | MultiPolygon | None,
    branch_spacing: float,
    head_spacing: float,
    branch_end_margin: float,
    min_obstacle_clearance: float,
    demand_step: float,
    min_head_spacing_nfpa: float = 1.8288,
    min_branch_station_gap: float | None = None,
    cover_radius: float | None = None,
    weight_per_demand: float = 200.0,
    coeff_heads: float = 18.0,
    coeff_branch_len: float = 2.5,
    coeff_phase_switch: float = 80.0,
    enforce_single_phase_per_side: bool = False,
    time_limit_s: float = 60.0,
    max_demand_points: int = 4000,
) -> tuple[list[LineString], list[Point], dict[str, Any]]:
    """
    CP-SAT: choose branch packages (trunk-originated) to maximize weighted coverage
    minus head/branch penalties, subject to NFPA-inspired min head spacing and
    branch-station separation. Not a substitute for full hydraulic review.
    """
    try:
        from ortools.sat.python import cp_model
    except ImportError as e:
        raise RuntimeError(
            "CP-SAT layout requires OR-Tools. Install with: pip install ortools"
        ) from e

    if trunk is None or trunk.is_empty:
        return [], [], {"solver": "cpsat", "status": "no_trunk"}

    if min_branch_station_gap is None:
        min_branch_station_gap = max(1.2, 0.42 * branch_spacing)
    if cover_radius is None:
        cover_radius = 0.6 * head_spacing

    demand = sample_demand_points(valid_area, step=demand_step)
    if len(demand) > max_demand_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(demand), size=max_demand_points, replace=False)
        demand = [demand[i] for i in sorted(idx)]

    pkgs = _build_branch_packages_for_cpsat(
        protected_area=protected_area,
        valid_area=valid_area,
        trunk=trunk,
        exclusion=exclusion,
        branch_spacing=branch_spacing,
        head_spacing=head_spacing,
        branch_end_margin=branch_end_margin,
        min_obstacle_clearance=min_obstacle_clearance,
        demand=demand,
        cover_radius=cover_radius,
    )
    if not pkgs:
        return [], [], {"solver": "cpsat", "status": "no_packages", "demand_points": len(demand)}

    conflicts = _package_pair_conflicts(pkgs, min_head_spacing_nfpa, min_branch_station_gap)

    n_p = len(demand)
    n_b = len(pkgs)
    coverers: list[list[int]] = [[] for _ in range(n_p)]
    for j, p in enumerate(pkgs):
        for i in p["covers"]:
            if 0 <= i < n_p:
                coverers[i].append(j)

    model = cp_model.CpModel()
    x = [model.NewBoolVar(f"b_{j}") for j in range(n_b)]
    z = [model.NewBoolVar(f"d_{i}") for i in range(n_p)]
    u = {
        (-1, 0): model.NewBoolVar("phase_side_neg_0"),
        (-1, 1): model.NewBoolVar("phase_side_neg_1"),
        (1, 0): model.NewBoolVar("phase_side_pos_0"),
        (1, 1): model.NewBoolVar("phase_side_pos_1"),
    }

    for i in range(n_p):
        if coverers[i]:
            model.Add(sum(x[j] for j in coverers[i]) >= z[i])
        else:
            model.Add(z[i] == 0)

    for a, b in conflicts:
        model.Add(x[a] + x[b] <= 1)
    for j, pkg in enumerate(pkgs):
        model.Add(x[j] <= u[(int(pkg["side"]), int(pkg["phase_index"]))])
    if enforce_single_phase_per_side:
        model.Add(u[(-1, 0)] + u[(-1, 1)] <= 1)
        model.Add(u[(1, 0)] + u[(1, 1)] <= 1)

    obj_terms: list[cp_model.LinearExpr] = []
    for i in range(n_p):
        obj_terms.append(int(round(weight_per_demand)) * z[i])
    for j in range(n_b):
        obj_terms.append(-int(round(coeff_heads * pkgs[j]["num_heads"])) * x[j])
        obj_terms.append(-int(round(coeff_branch_len * pkgs[j]["branch_length"])) * x[j])
    for side_phase in u.values():
        obj_terms.append(-int(round(coeff_phase_switch)) * side_phase)
    model.Maximize(sum(obj_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_s)
    solver.parameters.num_search_workers = 8
    status = solver.Solve(model)

    status_name = solver.StatusName(status)
    selected_branches: list[LineString] = []
    selected_heads: list[Point] = []
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for j in range(n_b):
            if solver.Value(x[j]) == 1:
                selected_branches.append(pkgs[j]["branch"])
                selected_heads.extend(pkgs[j]["heads"])

    selected_heads = dedupe_points(selected_heads, tol=min(0.2, 0.1 * head_spacing))

    covered_count = 0
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for i in range(n_p):
            if solver.Value(z[i]) == 1:
                covered_count += 1
    achieved = covered_count / max(1, n_p)

    diag: dict[str, Any] = {
        "solver": "cpsat",
        "status": status_name,
        "demand_points": n_p,
        "branch_packages": n_b,
        "conflict_edges": len(conflicts),
        "selected_packages": len(selected_branches),
        "objective": float(solver.ObjectiveValue()) if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else None,
        "achieved_coverage_ratio": achieved,
        "wall_time_s": float(solver.WallTime()),
        "min_head_spacing_m": min_head_spacing_nfpa,
        "min_branch_station_gap_m": min_branch_station_gap,
        "cover_radius_m": cover_radius,
        "phase_usage": {
            "-1_0": int(solver.Value(u[(-1, 0)])) if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else 0,
            "-1_1": int(solver.Value(u[(-1, 1)])) if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else 0,
            "+1_0": int(solver.Value(u[(1, 0)])) if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else 0,
            "+1_1": int(solver.Value(u[(1, 1)])) if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else 0,
        },
        "enforce_single_phase_per_side": bool(enforce_single_phase_per_side),
    }
    return selected_branches, selected_heads, diag


def optimize_heads_set_cover(
    candidates: list[Point],
    valid_area: Polygon | MultiPolygon | None,
    cover_radius: float,
    sample_step: float = 1.0,
) -> list[Point]:
    """Greedy set-cover model over sampled valid-area points."""
    if not candidates:
        return []
    samples = sample_points_inside_polygon(valid_area, step=sample_step)
    if not samples:
        return candidates

    candidate_cover: list[set[int]] = []
    for c in candidates:
        covered = {i for i, s in enumerate(samples) if s.distance(c) <= cover_radius}
        candidate_cover.append(covered)

    uncovered = set(range(len(samples)))
    selected_idx: list[int] = []
    while uncovered:
        best_i = -1
        best_gain = 0
        for i, covers in enumerate(candidate_cover):
            gain = len(covers & uncovered)
            if gain > best_gain:
                best_gain = gain
                best_i = i
        if best_i < 0 or best_gain == 0:
            break
        selected_idx.append(best_i)
        uncovered -= candidate_cover[best_i]

    if not selected_idx:
        return candidates
    return [candidates[i] for i in selected_idx]


def select_branches_graph_like(branches: list[LineString], min_station_gap: float = 2.0) -> list[LineString]:
    """
    Graph-like branch selection:
    - node score = branch length
    - conflict edge when two branches are too close (by midpoint distance)
    - greedy maximal independent set by descending score
    """
    if not branches:
        return []

    mids = [bl.interpolate(0.5, normalized=True) for bl in branches]
    order = sorted(range(len(branches)), key=lambda i: branches[i].length, reverse=True)
    kept: list[int] = []
    for idx in order:
        m = mids[idx]
        conflict = any(m.distance(mids[j]) < min_station_gap for j in kept)
        if not conflict:
            kept.append(idx)
    kept.sort()
    return [branches[i] for i in kept]


def generate_secondary_branches(
    primary_branches: list[LineString],
    protected_area: Polygon | MultiPolygon,
    exclusion: Polygon | MultiPolygon | None,
    trunk: LineString | None,
    spacing: float,
    min_primary_length: float,
    min_secondary_length: float = 1.4,
    max_per_primary: int = 1,
    direct_from_trunk_tolerance: float = 1.6,
) -> list[LineString]:
    """
    Generate stricter branch-from-branch stubs (orthogonal twigs) on long primary branches.
    Prefer direct straight branches from trunk whenever feasible.
    """
    if not primary_branches:
        return []
    minx, miny, maxx, maxy = protected_area.bounds
    span = max(maxx - minx, maxy - miny) + 20.0
    out: list[LineString] = []
    blocked = exclusion if exclusion is not None and not exclusion.is_empty else None
    for br in primary_branches:
        if br.length < min_primary_length:
            continue
        added_this_primary = 0
        d = spacing
        while d <= br.length - spacing + 1e-9:
            if added_this_primary >= max_per_primary:
                break
            p = br.interpolate(d)
            d0 = max(0.0, d - 0.2)
            d1 = min(br.length, d + 0.2)
            p0 = br.interpolate(d0)
            p1 = br.interpolate(d1)
            vx = p1.x - p0.x
            vy = p1.y - p0.y
            n = math.hypot(vx, vy)
            if n < 1e-9:
                d += spacing
                continue
            tangent = np.array([vx / n, vy / n], dtype=float)
            normal = np.array([-tangent[1], tangent[0]], dtype=float)
            origin = np.array([p.x, p.y], dtype=float)
            ray_a, ray_b = rays_from_point(origin, normal, span)
            origin_pt = Point(float(origin[0]), float(origin[1]))

            # Prefer direct branch from trunk if a straight trunk branch can reach this point.
            if trunk is not None and not trunk.is_empty:
                s_trunk = trunk.project(origin_pt)
                trunk_direct_found = False
                for side in (-1, 1):
                    direct = build_branch_from_station(
                        protected_area=protected_area,
                        trunk=trunk,
                        station_s=s_trunk,
                        side=side,
                        exclusion_area=exclusion,
                        min_branch_length=min_secondary_length,
                    )
                    if direct is not None and direct.distance(origin_pt) <= direct_from_trunk_tolerance:
                        trunk_direct_found = True
                        break
                if trunk_direct_found:
                    d += spacing
                    continue

            for ray in (ray_a, ray_b):
                clipped = ray.intersection(protected_area)
                segs = lines_from_intersection(clipped, min_length=min_secondary_length)
                seg = pick_segment_touching_origin(segs, origin_pt)
                if seg is None or seg.length < min_secondary_length:
                    continue
                if blocked is not None:
                    seg = seg.difference(blocked)
                    cleaned = lines_from_intersection(seg, min_length=min_secondary_length)
                    seg = pick_segment_touching_origin(cleaned, origin_pt)
                    if seg is None or seg.length < min_secondary_length:
                        continue
                out.append(seg)
                added_this_primary += 1
                if added_this_primary >= max_per_primary:
                    break
            d += spacing
    return out


def _segment_is_routable(
    p0: tuple[float, float],
    p1: tuple[float, float],
    valid_area: Polygon | MultiPolygon | None,
    exclusion: Polygon | MultiPolygon | None,
) -> bool:
    if p0 == p1:
        return False
    seg = LineString([p0, p1])
    if valid_area is None or valid_area.is_empty:
        return False
    mid = seg.interpolate(0.5, normalized=True)
    if not (valid_area.contains(mid) or valid_area.buffer(1e-6).contains(mid)):
        return False
    if exclusion is not None and not exclusion.is_empty and seg.distance(exclusion) <= 1e-4:
        return False
    clipped = seg.intersection(valid_area)
    return isinstance(clipped, LineString) and clipped.length >= seg.length * 0.95


def _dijkstra_shortest_path(
    adj: list[list[tuple[int, float]]],
    src: int,
    dst: int,
) -> tuple[float, list[int]]:
    n = len(adj)
    dist = [float("inf")] * n
    prev = [-1] * n
    dist[src] = 0.0
    pq: list[tuple[float, int]] = [(0.0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u] + 1e-12:
            continue
        if u == dst:
            break
        for v, w in adj[u]:
            nd = d + w
            if nd + 1e-12 < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(pq, (nd, v))
    if not math.isfinite(dist[dst]):
        return float("inf"), []
    path: list[int] = []
    cur = dst
    while cur >= 0:
        path.append(cur)
        if cur == src:
            break
        cur = prev[cur]
    path.reverse()
    return dist[dst], path


def route_heads_with_rectilinear_steiner(
    heads: list[Point],
    trunk: LineString | None,
    valid_area: Polygon | MultiPolygon | None,
    exclusion: Polygon | MultiPolygon | None,
    root_step: float,
    nearest_roots: int = 3,
) -> tuple[list[LineString], dict[str, Any]]:
    """
    Rectilinear obstacle-aware Steiner approximation:
    - terminals = heads
    - roots = sampled points on trunk (connected to virtual source at zero cost)
    - graph nodes include terminals, roots, and one-bend Steiner candidates
    - solve Steiner approximation via metric closure MST + path expansion
    """
    if trunk is None or trunk.is_empty or not heads:
        return [], {"routing_model": "steiner_rectilinear", "status": "no_inputs"}

    roots = points_along_line(trunk, spacing=max(1.0, root_step), endpoint_margin=0.0)
    if not roots:
        roots = [trunk.interpolate(0.0), trunk.interpolate(trunk.length)]

    terminals_xy = [(float(h.x), float(h.y)) for h in heads]
    roots_xy = [(float(r.x), float(r.y)) for r in roots]

    node_set: set[tuple[float, float]] = set(terminals_xy) | set(roots_xy)
    for tx, ty in terminals_xy:
        dists = sorted(((rx - tx) ** 2 + (ry - ty) ** 2, (rx, ry)) for rx, ry in roots_xy)
        for _, (rx, ry) in dists[: max(1, nearest_roots)]:
            node_set.add((tx, ry))
            node_set.add((rx, ty))

    nodes = list(node_set)
    idx_of = {p: i for i, p in enumerate(nodes)}
    n = len(nodes)

    x_buckets: dict[float, list[tuple[float, int]]] = {}
    y_buckets: dict[float, list[tuple[float, int]]] = {}
    for i, (x, y) in enumerate(nodes):
        x_buckets.setdefault(round(x, 6), []).append((y, i))
        y_buckets.setdefault(round(y, 6), []).append((x, i))

    adj: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for _, arr in x_buckets.items():
        arr.sort(key=lambda t: t[0])
        for k in range(len(arr) - 1):
            y0, i0 = arr[k]
            y1, i1 = arr[k + 1]
            p0, p1 = nodes[i0], nodes[i1]
            if _segment_is_routable(p0, p1, valid_area, exclusion):
                w = abs(y1 - y0)
                adj[i0].append((i1, w))
                adj[i1].append((i0, w))
    for _, arr in y_buckets.items():
        arr.sort(key=lambda t: t[0])
        for k in range(len(arr) - 1):
            x0, i0 = arr[k]
            x1, i1 = arr[k + 1]
            p0, p1 = nodes[i0], nodes[i1]
            if _segment_is_routable(p0, p1, valid_area, exclusion):
                w = abs(x1 - x0)
                adj[i0].append((i1, w))
                adj[i1].append((i0, w))

    required = [idx_of[p] for p in terminals_xy]
    root_indices = [idx_of[p] for p in roots_xy]
    if not required or not root_indices:
        return [], {"routing_model": "steiner_rectilinear", "status": "no_required_or_roots"}

    virtual_root = n
    req_plus_root = [virtual_root] + required
    m = len(req_plus_root)

    metric_w = [[float("inf")] * m for _ in range(m)]
    metric_path: dict[tuple[int, int], list[int]] = {}
    for i in range(m):
        metric_w[i][i] = 0.0

    # virtual-root to terminal = best shortest path through any root index
    for j in range(1, m):
        t_idx = req_plus_root[j]
        best = (float("inf"), [])
        for r_idx in root_indices:
            d, path = _dijkstra_shortest_path(adj, r_idx, t_idx)
            if d < best[0]:
                best = (d, path)
        metric_w[0][j] = metric_w[j][0] = best[0]
        metric_path[(0, j)] = best[1]
        metric_path[(j, 0)] = list(reversed(best[1])) if best[1] else []

    for i in range(1, m):
        for j in range(i + 1, m):
            d, path = _dijkstra_shortest_path(adj, req_plus_root[i], req_plus_root[j])
            metric_w[i][j] = metric_w[j][i] = d
            metric_path[(i, j)] = path
            metric_path[(j, i)] = list(reversed(path)) if path else []

    # Prim MST on metric closure
    in_tree = [False] * m
    key = [float("inf")] * m
    parent = [-1] * m
    key[0] = 0.0
    for _ in range(m):
        u = -1
        bestk = float("inf")
        for i in range(m):
            if not in_tree[i] and key[i] < bestk:
                bestk = key[i]
                u = i
        if u < 0:
            break
        in_tree[u] = True
        for v in range(m):
            if not in_tree[v] and metric_w[u][v] < key[v]:
                key[v] = metric_w[u][v]
                parent[v] = u

    edge_set: set[tuple[int, int]] = set()
    for v in range(1, m):
        u = parent[v]
        if u < 0:
            continue
        path = metric_path.get((u, v), [])
        for a, b in zip(path, path[1:]):
            e = (min(a, b), max(a, b))
            edge_set.add(e)

    segments: list[LineString] = []
    for a, b in sorted(edge_set):
        p0, p1 = nodes[a], nodes[b]
        seg = LineString([p0, p1])
        if seg.length > 1e-6:
            segments.append(seg)

    diag = {
        "routing_model": "steiner_rectilinear",
        "status": "ok",
        "terminals": len(required),
        "roots": len(root_indices),
        "graph_nodes": len(nodes),
        "graph_edges": sum(len(v) for v in adj) // 2,
        "steiner_segments": len(segments),
    }
    return segments, diag


def route_heads_direct_from_trunk(
    heads: list[Point],
    trunk: LineString | None,
    valid_area: Polygon | MultiPolygon | None,
    exclusion: Polygon | MultiPolygon | None,
    root_step: float,
) -> tuple[list[LineString], dict[str, Any]]:
    """
    Strict routing: connect each head directly from trunk whenever possible.
    Preference order:
    1) straight segment from nearest trunk point
    2) one-bend orthogonal path via nearest trunk root
    This intentionally reduces secondary turns/splits in branch network.
    """
    if trunk is None or trunk.is_empty or not heads:
        return [], {"routing_model": "direct_from_trunk", "status": "no_inputs"}

    roots = points_along_line(trunk, spacing=max(1.0, root_step), endpoint_margin=0.0)
    if not roots:
        roots = [trunk.interpolate(0.0), trunk.interpolate(trunk.length)]
    roots_xy = [(float(r.x), float(r.y)) for r in roots]

    segments: list[LineString] = []
    straight_used = 0
    one_bend_used = 0
    dropped = 0

    for h in heads:
        hp = (float(h.x), float(h.y))
        s = trunk.project(h)
        tp = trunk.interpolate(s)
        txy = (float(tp.x), float(tp.y))

        # 1) straight from nearest trunk projection
        if _segment_is_routable(txy, hp, valid_area, exclusion):
            segments.append(LineString([txy, hp]))
            straight_used += 1
            continue

        # 2) one-bend via nearest trunk roots
        root_order = sorted(roots_xy, key=lambda r: (r[0] - hp[0]) ** 2 + (r[1] - hp[1]) ** 2)
        connected = False
        for rx, ry in root_order[:6]:
            elbow1 = (rx, hp[1])
            elbow2 = (hp[0], ry)
            path1_ok = _segment_is_routable((rx, ry), elbow1, valid_area, exclusion) and _segment_is_routable(
                elbow1, hp, valid_area, exclusion
            )
            path2_ok = _segment_is_routable((rx, ry), elbow2, valid_area, exclusion) and _segment_is_routable(
                elbow2, hp, valid_area, exclusion
            )
            if path1_ok or path2_ok:
                if path1_ok:
                    segments.append(LineString([(rx, ry), elbow1]))
                    segments.append(LineString([elbow1, hp]))
                else:
                    segments.append(LineString([(rx, ry), elbow2]))
                    segments.append(LineString([elbow2, hp]))
                one_bend_used += 1
                connected = True
                break
        if not connected:
            dropped += 1

    # Deduplicate exact duplicate segments
    uniq: dict[tuple[tuple[float, float], tuple[float, float]], LineString] = {}
    for seg in segments:
        c = list(seg.coords)
        a = (round(float(c[0][0]), 4), round(float(c[0][1]), 4))
        b = (round(float(c[-1][0]), 4), round(float(c[-1][1]), 4))
        key = (a, b) if a <= b else (b, a)
        uniq[key] = seg

    out = list(uniq.values())
    diag = {
        "routing_model": "direct_from_trunk",
        "status": "ok",
        "terminals": len(heads),
        "roots": len(roots_xy),
        "segments": len(out),
        "straight_used": straight_used,
        "one_bend_used": one_bend_used,
        "dropped_heads": dropped,
    }
    return out, diag


def evaluate_nfpa13_compliance(
    heads: list[Point],
    branches: list[LineString],
    protected_area: Polygon | MultiPolygon | None,
    valid_area: Polygon | MultiPolygon | None,
    exclusion: Polygon | MultiPolygon | None,
    *,
    max_spacing_m: float,
    min_spacing_m: float,
    max_wall_distance_m: float,
    min_wall_distance_m: float,
    min_obstruction_clearance_m: float,
    max_avg_area_per_head_m2: float,
    target_coverage_ratio: float,
    demand_step: float,
) -> dict[str, Any]:
    """Draft NFPA-13 oriented compliance checks (geometric subset)."""
    checks: list[dict[str, Any]] = []
    violations: dict[str, Any] = {}

    def add_check(name: str, passed: bool, detail: str, metrics: dict[str, Any] | None = None) -> None:
        checks.append(
            {
                "rule": name,
                "pass": bool(passed),
                "detail": detail,
                "metrics": metrics or {},
            }
        )

    # 1) Coverage ratio over sampled valid area
    samples = sample_points_inside_polygon(valid_area, step=max(0.5, demand_step))
    cover_radius = 0.5 * max_spacing_m
    covered = 0
    for s in samples:
        if any(s.distance(h) <= cover_radius for h in heads):
            covered += 1
    coverage_ratio = covered / max(1, len(samples))
    ok_coverage = coverage_ratio >= target_coverage_ratio
    add_check(
        "coverage_ratio",
        ok_coverage,
        f"Coverage ratio {coverage_ratio:.3f} vs target {target_coverage_ratio:.3f}.",
        {"coverage_ratio": coverage_ratio, "target": target_coverage_ratio, "sample_points": len(samples)},
    )
    if not ok_coverage:
        violations["coverage_ratio"] = {"actual": coverage_ratio, "target": target_coverage_ratio}

    # 2) Head-to-head spacing min/max
    min_pair = float("inf")
    max_nearest = 0.0
    too_close = 0
    too_far = 0
    for i in range(len(heads)):
        nearest = float("inf")
        for j in range(len(heads)):
            if i == j:
                continue
            d = heads[i].distance(heads[j])
            min_pair = min(min_pair, d)
            nearest = min(nearest, d)
            if d < min_spacing_m - 1e-6:
                too_close += 1
        if math.isfinite(nearest):
            max_nearest = max(max_nearest, nearest)
            if nearest > max_spacing_m + 1e-6:
                too_far += 1
    too_close = too_close // 2
    ok_spacing = too_close == 0 and too_far == 0
    add_check(
        "sprinkler_spacing",
        ok_spacing,
        f"Too-close pairs={too_close}, isolated heads beyond max spacing={too_far}.",
        {
            "min_pair_distance_m": None if not math.isfinite(min_pair) else min_pair,
            "max_nearest_neighbor_m": max_nearest,
            "min_spacing_limit_m": min_spacing_m,
            "max_spacing_limit_m": max_spacing_m,
        },
    )
    if not ok_spacing:
        violations["sprinkler_spacing"] = {"too_close_pairs": too_close, "too_far_heads": too_far}

    # 3) Distance from walls (approx by nearest boundary distance)
    near_wall_too_close = 0
    far_from_wall = 0
    wall_dists: list[float] = []
    boundary = protected_area.boundary if protected_area is not None and not protected_area.is_empty else None
    if boundary is not None:
        for h in heads:
            d = h.distance(boundary)
            wall_dists.append(d)
            if d < min_wall_distance_m - 1e-6:
                near_wall_too_close += 1
            if d > max_wall_distance_m + 1e-6:
                far_from_wall += 1
    ok_wall = near_wall_too_close == 0 and far_from_wall == 0
    add_check(
        "distance_from_walls",
        ok_wall,
        f"Too close to wall={near_wall_too_close}, farther than max wall distance={far_from_wall}.",
        {
            "min_wall_distance_observed_m": min(wall_dists) if wall_dists else None,
            "max_wall_distance_observed_m": max(wall_dists) if wall_dists else None,
            "min_wall_limit_m": min_wall_distance_m,
            "max_wall_limit_m": max_wall_distance_m,
        },
    )
    if not ok_wall:
        violations["distance_from_walls"] = {
            "too_close": near_wall_too_close,
            "too_far": far_from_wall,
        }

    # 4) Obstruction clearance (columns/stairs/wall buffer aggregate)
    obstruction_viol = 0
    if exclusion is not None and not exclusion.is_empty:
        for h in heads:
            if h.distance(exclusion) < min_obstruction_clearance_m - 1e-6:
                obstruction_viol += 1
    ok_obs = obstruction_viol == 0
    add_check(
        "obstruction_clearance",
        ok_obs,
        f"Heads violating obstruction clearance={obstruction_viol}.",
        {
            "min_required_clearance_m": min_obstruction_clearance_m,
            "violating_heads": obstruction_viol,
        },
    )
    if not ok_obs:
        violations["obstruction_clearance"] = {"violating_heads": obstruction_viol}

    # 5) Average area per head sanity check (not substitute for exact NFPA table application)
    valid_area_m2 = float(valid_area.area) if valid_area is not None and not valid_area.is_empty else 0.0
    avg_area = valid_area_m2 / max(1, len(heads))
    ok_area = avg_area <= max_avg_area_per_head_m2 + 1e-6
    add_check(
        "avg_area_per_head",
        ok_area,
        f"Average area/head {avg_area:.2f} m2 vs limit {max_avg_area_per_head_m2:.2f} m2.",
        {
            "avg_area_per_head_m2": avg_area,
            "limit_m2": max_avg_area_per_head_m2,
            "valid_area_m2": valid_area_m2,
            "sprinkler_count": len(heads),
        },
    )
    if not ok_area:
        violations["avg_area_per_head"] = {"actual": avg_area, "limit": max_avg_area_per_head_m2}

    # 6) Branch network sanity
    total_branch_length = float(sum(bl.length for bl in branches))
    add_check(
        "network_sanity",
        total_branch_length > 0 and len(branches) > 0 and len(heads) > 0,
        "Branch and head network non-empty.",
        {"branch_lines": len(branches), "total_branch_length_m": total_branch_length, "sprinkler_heads": len(heads)},
    )

    passed_all = all(c["pass"] for c in checks)
    return {
        "standard": "NFPA13_draft_geometric_checks",
        "pass": passed_all,
        "checks": checks,
        "violations": violations,
        "note": (
            "Geometric draft checks only. Full NFPA 13 compliance still requires sprinkler listing constraints, "
            "ceiling/deflector rules, hazard tables, and hydraulic calculations by a licensed engineer."
        ),
    }


def analyze_branch_topology(
    branches: list[LineString],
    trunk: LineString | None,
    snap_m: float = 0.01,
    trunk_touch_tol: float = 0.15,
    collinear_tol: float = 1e-3,
) -> dict[str, Any]:
    """Detect turns/splits in final routed branch network."""
    if not branches:
        return {
            "nodes": 0,
            "edges": 0,
            "turn_nodes": 0,
            "split_nodes": 0,
            "secondary_events": 0,
            "secondary_segments_estimate": 0,
        }

    def snap_pt(x: float, y: float) -> tuple[float, float]:
        return (round(x / snap_m) * snap_m, round(y / snap_m) * snap_m)

    nodes: list[tuple[float, float]] = []
    idx_of: dict[tuple[float, float], int] = {}
    adj: dict[int, set[int]] = {}

    def get_idx(p: tuple[float, float]) -> int:
        if p in idx_of:
            return idx_of[p]
        idx = len(nodes)
        nodes.append(p)
        idx_of[p] = idx
        adj[idx] = set()
        return idx

    for seg in branches:
        coords = list(seg.coords)
        if len(coords) < 2:
            continue
        p0 = snap_pt(float(coords[0][0]), float(coords[0][1]))
        p1 = snap_pt(float(coords[-1][0]), float(coords[-1][1]))
        if p0 == p1:
            continue
        i0 = get_idx(p0)
        i1 = get_idx(p1)
        adj[i0].add(i1)
        adj[i1].add(i0)

    trunk_touch = set()
    if trunk is not None and not trunk.is_empty:
        for i, (x, y) in enumerate(nodes):
            if Point(x, y).distance(trunk) <= trunk_touch_tol:
                trunk_touch.add(i)

    turn_nodes = 0
    split_nodes = 0
    for i, nbrs in adj.items():
        deg = len(nbrs)
        if deg < 2:
            continue
        if deg >= 3 and i not in trunk_touch:
            split_nodes += 1
            continue
        if deg == 2 and i not in trunk_touch:
            n1, n2 = list(nbrs)
            p = np.array(nodes[i], dtype=float)
            v1 = np.array(nodes[n1], dtype=float) - p
            v2 = np.array(nodes[n2], dtype=float) - p
            nrm1 = float(np.linalg.norm(v1))
            nrm2 = float(np.linalg.norm(v2))
            if nrm1 > 1e-9 and nrm2 > 1e-9:
                v1 /= nrm1
                v2 /= nrm2
                # Collinear if dot is near +/-1
                if abs(abs(float(np.dot(v1, v2))) - 1.0) > collinear_tol:
                    turn_nodes += 1

    edge_count = sum(len(v) for v in adj.values()) // 2
    secondary_events = turn_nodes + split_nodes
    return {
        "nodes": len(nodes),
        "edges": edge_count,
        "turn_nodes": turn_nodes,
        "split_nodes": split_nodes,
        "secondary_events": secondary_events,
        "secondary_segments_estimate": secondary_events,
    }


def draw_polygon(ax: Any, geom: Polygon | MultiPolygon | None, color: str, alpha: float, label: str) -> None:
    if geom is None or geom.is_empty:
        return
    polys = [geom] if isinstance(geom, Polygon) else list(geom.geoms)
    for idx, p in enumerate(polys):
        ext = np.array(p.exterior.coords)
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
        for ring in p.interiors:
            pts = np.array(ring.coords)
            ax.add_patch(
                MplPolygon(
                    pts,
                    closed=True,
                    facecolor="white",
                    edgecolor=color,
                    linewidth=0.7,
                    alpha=1.0,
                )
            )


def annotate_pipe_segment(
    ax: Any,
    p0: np.ndarray,
    p1: np.ndarray,
    diameter_label: str,
    color: str = "#111111",
    offset_scale: float = 0.45,
    font_size: float = 2.6,
    leader_color: str | None = None,
) -> None:
    vec = p1 - p0
    length = float(np.linalg.norm(vec))
    if length <= 1e-6:
        return

    midpoint = (p0 + p1) * 0.5
    dx = float(vec[0])
    dy = float(vec[1])
    is_vertical = abs(dy) >= abs(dx)

    if is_vertical:
        # CAD-style side note for vertical risers/branches.
        leader_start = midpoint
        leader_end = midpoint + np.array([offset_scale, 0.0], dtype=float)
        text_pos = leader_end + np.array([0.04, 0.04], dtype=float)
        ha, va = "left", "bottom"
    else:
        # CAD-style top note for horizontal trunk sections.
        leader_start = midpoint
        leader_end = midpoint + np.array([0.0, offset_scale], dtype=float)
        text_pos = leader_end + np.array([0.03, 0.03], dtype=float)
        ha, va = "left", "bottom"

    lc = leader_color if leader_color is not None else "#444444"
    ax.plot(
        [leader_start[0], leader_end[0]],
        [leader_start[1], leader_end[1]],
        color=lc,
        linewidth=0.8,
        alpha=0.85,
        zorder=7,
    )

    label = f"Ø{diameter_label}\nL={length:.2f} m"
    ax.text(
        text_pos[0],
        text_pos[1],
        label,
        fontsize=font_size,
        color=color,
        ha=ha,
        va=va,
        zorder=8,
    )


def add_equipment_notes(ax: Any) -> None:
    equipment_lines = [
        "ფოლადის მილი",
        "სახანძრო წყარადა 30 მ-იანი შლანგით, DN50 დიამეტრით",
        "სახანძრო სისტემის დასარეგულირებელი ონკანი, DN77",
        "სარქველი, ზევით მიმართველი DN15, K=5.6, T=68°C",
        "კევკლა სარქველი, DN65",
        "უკუსარქველი, D65",
        "სახანძრო ონკანი PRV DN65, D77იანი დიამეტრით",
        "წნევის მაკონტროლებელი სარქველი DN150–DN100",
        "სარქველის მაკონტროლებელი კვანძი წყლის რულერი, მანომეტრია და ტესტირების ურელეო",
    ]
    notes = "\n".join(f"• {line}" for line in equipment_lines)
    ax.text(
        0.99,
        0.99,
        notes,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.6,
        color="#111111",
        bbox={"boxstyle": "square,pad=0.30", "fc": "white", "ec": "#9ca3af", "alpha": 0.92},
        zorder=10,
    )


def draw_preview_floor_badge(ax: Any, label: str) -> None:
    """Visible floor / storey id on PNG (axes coordinates, top-left)."""
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
    protected_area: Polygon | MultiPolygon | None,
    exclusion: Polygon | MultiPolygon | None,
    trunk: LineString | None,
    branches: list[LineString],
    heads: list[Point],
    trunk_diameter_label: str,
    branch_diameter_label: str,
    preview_floor_label: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.set_facecolor("#f7f7f7")
    draw_polygon(ax, protected_area, color="#d1d5db", alpha=0.12, label="Protected area")
    draw_polygon(ax, exclusion, color="#fecaca", alpha=0.15, label="Exclusion zones")

    # Main trunk line (DN100 labels drawn between branch starts below, or fallback here).
    MAIN_PIPE_LABEL_COLOR = "#1565c0"
    BRANCH_LABEL_FONT = 1.9 / 1.3
    MAIN_PIPE_LABEL_FONT = 2.4 / 1.3

    if trunk is not None:
        tx, ty = trunk.xy
        ax.plot(tx, ty, color="#ff2d2d", linewidth=1.4, label="Main trunk", zorder=3)

    head_xy = np.array([[p.x, p.y] for p in heads], dtype=float) if heads else np.empty((0, 2), dtype=float)

    def near_any_head(pt: np.ndarray, tol: float = 0.22) -> bool:
        if head_xy.size == 0:
            return False
        d = np.linalg.norm(head_xy - pt.reshape(1, 2), axis=1)
        return bool(np.any(d <= tol))

    def split_line_by_heads(line: LineString, tol: float = 0.22) -> list[LineString]:
        if line.length <= 1e-6:
            return []
        cuts = [0.0, float(line.length)]
        if head_xy.size > 0:
            for h in heads:
                if line.distance(h) <= tol:
                    cuts.append(float(line.project(h)))
        cuts = sorted(cuts)
        uniq: list[float] = []
        for d in cuts:
            if not uniq or abs(d - uniq[-1]) > 1e-4:
                uniq.append(d)
        out: list[LineString] = []
        for d0, d1 in zip(uniq, uniq[1:]):
            if d1 - d0 < 0.08:
                continue
            seg = substring(line, d0, d1)
            if isinstance(seg, LineString) and seg.length > 0.05:
                out.append(seg)
        return out

    branch_starts_on_trunk: list[Point] = []
    for idx, bl in enumerate(branches):
        bx, by = bl.xy
        ax.plot(bx, by, color="#ff2d2d", linewidth=1.05, alpha=0.95, label="Branch lines" if idx == 0 else None, zorder=3)
        if trunk is not None and not trunk.is_empty:
            c0 = Point(float(bl.coords[0][0]), float(bl.coords[0][1]))
            c1 = Point(float(bl.coords[-1][0]), float(bl.coords[-1][1]))
            d0 = c0.distance(trunk)
            d1 = c1.distance(trunk)
            if d0 <= 0.15 and d0 <= d1:
                branch_starts_on_trunk.append(c0)
            elif d1 <= 0.15 and d1 < d0:
                branch_starts_on_trunk.append(c1)
        for seg in split_line_by_heads(bl):
            sc = [np.array(c, dtype=float) for c in seg.coords]
            if len(sc) < 2:
                continue
            p0 = sc[0]
            p1 = sc[-1]
            # Diameter rule:
            # - trunk handled above as DN100
            # - main-to-first sprinkler: DN32
            # - sprinkler-to-sprinkler: DN25
            dia = "DN25" if (near_any_head(p0) and near_any_head(p1)) else "DN32"
            annotate_pipe_segment(
                ax,
                p0,
                p1,
                dia,
                color="#111111",
                offset_scale=0.42,
                font_size=BRANCH_LABEL_FONT,
            )

    if trunk is not None and not trunk.is_empty and branch_starts_on_trunk:
        # Deduplicate/sort branch starts along trunk.
        svals = sorted(trunk.project(p) for p in branch_starts_on_trunk)
        uniq_s: list[float] = []
        for s in svals:
            if not uniq_s or abs(s - uniq_s[-1]) > 0.12:
                uniq_s.append(s)
        # Trunk readings from one branch start to the next.
        for s0, s1 in zip(uniq_s, uniq_s[1:]):
            if s1 - s0 < 0.12:
                continue
            seg = substring(trunk, s0, s1)
            if not isinstance(seg, LineString) or seg.length <= 0.1:
                continue
            sc = [np.array(c, dtype=float) for c in seg.coords]
            annotate_pipe_segment(
                ax,
                sc[0],
                sc[-1],
                "DN100",
                color=MAIN_PIPE_LABEL_COLOR,
                offset_scale=0.5,
                font_size=MAIN_PIPE_LABEL_FONT,
                leader_color=MAIN_PIPE_LABEL_COLOR,
            )
    elif trunk is not None and not trunk.is_empty:
        trunk_coords = [np.array(c, dtype=float) for c in trunk.coords]
        for i in range(len(trunk_coords) - 1):
            annotate_pipe_segment(
                ax,
                trunk_coords[i],
                trunk_coords[i + 1],
                "DN100",
                color=MAIN_PIPE_LABEL_COLOR,
                offset_scale=0.5,
                font_size=MAIN_PIPE_LABEL_FONT,
                leader_color=MAIN_PIPE_LABEL_COLOR,
            )

    if heads:
        hx = [p.x for p in heads]
        hy = [p.y for p in heads]
        # Hollow red circles to match conventional fire-system drafting style.
        ax.scatter(hx, hy, facecolors="none", edgecolors="#ff2d2d", linewidths=1.0, s=42, label="Sprinkler heads", zorder=6)
        # Center dot inside sprinkler ring; ~3x smaller than previous s=7 (marker area ÷ 3).
        ax.scatter(hx, hy, c="#ff2d2d", s=7.0 / 3.0, zorder=7)

    add_equipment_notes(ax)
    ax.set_title("Fire Suppression Layout")
    ax.set_xlabel("X (world)")
    ax.set_ylabel("Y (world)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    draw_preview_floor_badge(ax, preview_floor_label or "")

    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        if l and l not in seen:
            seen[l] = h
    if seen:
        ax.legend(seen.values(), seen.keys(), loc="best")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=360)
    plt.close(fig)


def write_dxf_fallback(out_dxf: Path, trunk: LineString | None, branches: list[LineString], heads: list[Point]) -> None:
    # Minimal R12 ASCII DXF exporter (LINES + POINTS), avoids extra dependencies.
    lines = ["0", "SECTION", "2", "ENTITIES"]

    def add_line(x1: float, y1: float, x2: float, y2: float, layer: str) -> None:
        lines.extend(
            [
                "0",
                "LINE",
                "8",
                layer,
                "10",
                f"{x1}",
                "20",
                f"{y1}",
                "30",
                "0.0",
                "11",
                f"{x2}",
                "21",
                f"{y2}",
                "31",
                "0.0",
            ]
        )

    if trunk is not None and not trunk.is_empty:
        coords = list(trunk.coords)
        for i in range(len(coords) - 1):
            add_line(coords[i][0], coords[i][1], coords[i + 1][0], coords[i + 1][1], "TRUNK")

    for bl in branches:
        coords = list(bl.coords)
        for i in range(len(coords) - 1):
            add_line(coords[i][0], coords[i][1], coords[i + 1][0], coords[i + 1][1], "BRANCH")

    for p in heads:
        lines.extend(
            [
                "0",
                "POINT",
                "8",
                "SPRINKLER",
                "10",
                f"{p.x}",
                "20",
                f"{p.y}",
                "30",
                "0.0",
            ]
        )

    lines.extend(["0", "ENDSEC", "0", "EOF"])
    out_dxf.parent.mkdir(parents=True, exist_ok=True)
    out_dxf.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic draft suppression layout from detected geometry.")
    parser.add_argument("--input-json", default="outputs/output/detected_geometry.json", help="Detected geometry JSON path")
    parser.add_argument("--output-dir", default="outputs/output", help="Output directory")
    parser.add_argument(
        "--preview-floor-label",
        type=str,
        default="",
        help="If set, draws this text (floor id) on layout_preview.png",
    )
    parser.add_argument("--branch-spacing", type=float, default=3.8, help="Spacing between branch lines (m)")
    parser.add_argument("--head-spacing", type=float, default=3.2, help="Spacing between sprinkler heads on branch (m)")
    parser.add_argument("--column-clearance", type=float, default=0.55, help="Column buffer clearance (m)")
    parser.add_argument("--stair-clearance", type=float, default=0.8, help="Stair exclusion clearance (m)")
    parser.add_argument("--wall-clearance", type=float, default=0.3, help="Optional wall clearance buffer (m, 0 to disable)")
    parser.add_argument("--branch-end-margin", type=float, default=0.8, help="Trim heads near branch ends (m)")
    parser.add_argument("--min-obstacle-clearance", type=float, default=0.2, help="Extra min clearance from exclusion edges")
    parser.add_argument("--trunk-diameter", default="DN100", help="Main trunk diameter label")
    parser.add_argument("--branch-diameter", default="DN65", help="Branch line diameter label")
    parser.add_argument(
        "--layout-model",
        choices=["baseline", "hybrid", "adaptive", "cpsat"],
        default="cpsat",
        help="baseline|hybrid|adaptive heuristics; cpsat=OR-Tools CP-SAT (coverage + NFPA-style spacing constraints)",
    )
    parser.add_argument("--demand-step", type=float, default=1.0, help="Demand grid step (m); used by adaptive and cpsat")
    parser.add_argument("--target-coverage", type=float, default=0.96, help="Target coverage ratio for adaptive model only (0..1)")
    parser.add_argument("--cpsat-time-limit", type=float, default=60.0, help="CP-SAT max wall time (seconds)")
    parser.add_argument(
        "--cpsat-max-demand",
        type=int,
        default=4000,
        help="Cap demand sample size for CP-SAT (random subsample if exceeded)",
    )
    parser.add_argument(
        "--cpsat-min-head-spacing",
        type=float,
        default=1.8288,
        help="Minimum sprinkler-to-sprinkler spacing (m), NFPA 6 ft default",
    )
    parser.add_argument(
        "--cpsat-weight-demand",
        type=float,
        default=200.0,
        help="CP-SAT objective weight per covered demand point",
    )
    parser.add_argument("--cpsat-penalty-head", type=float, default=18.0, help="CP-SAT penalty per sprinkler head in objective")
    parser.add_argument("--cpsat-penalty-branch-m", type=float, default=2.5, help="CP-SAT penalty per meter of selected branch length")
    parser.add_argument(
        "--cpsat-penalty-phase-switch",
        type=float,
        default=80.0,
        help="CP-SAT penalty for using extra side/phase combinations (encourages line-up)",
    )
    parser.add_argument("--allow-secondary-branches", action="store_true", help="Allow branch-from-branch secondary stubs")
    parser.add_argument(
        "--cpsat-enforce-single-phase-per-side",
        action="store_true",
        help="Hard alignment: for each side of trunk use one head phase only",
    )
    parser.add_argument("--secondary-branch-spacing", type=float, default=6.0, help="Spacing between secondary stubs on long branches (m)")
    parser.add_argument("--secondary-min-primary-length", type=float, default=10.5, help="Primary branch minimum length for secondary stubs (m)")
    parser.add_argument("--secondary-min-length", type=float, default=1.8, help="Minimum kept secondary branch length (m)")
    parser.add_argument("--secondary-max-per-primary", type=int, default=1, help="Maximum secondary branches generated per primary branch")
    parser.add_argument(
        "--secondary-direct-trunk-tolerance",
        type=float,
        default=1.6,
        help="If a direct trunk branch can reach secondary origin within this distance, skip secondary branch (m)",
    )
    parser.add_argument(
        "--routing-model",
        choices=["legacy", "steiner", "direct"],
        default="direct",
        help="legacy uses generated branch segments; steiner builds shared rectilinear tree; direct is strict trunk-first with minimal secondary branching",
    )
    parser.add_argument("--steiner-root-step", type=float, default=3.8, help="Trunk root sampling step for steiner routing (m)")
    parser.add_argument("--steiner-nearest-roots", type=int, default=3, help="Nearest trunk roots used to build one-bend Steiner candidates")
    parser.add_argument("--verbose-layout", action="store_true", help="Print detailed layout diagnostics")
    parser.add_argument("--nfpa-checks", action="store_true", help="Run NFPA-13 oriented geometric compliance checks")
    parser.add_argument("--nfpa-max-spacing", type=float, default=4.572, help="NFPA spacing max (m), default 15 ft")
    parser.add_argument("--nfpa-min-spacing", type=float, default=1.8288, help="NFPA spacing min (m), default 6 ft")
    parser.add_argument("--nfpa-max-wall-distance", type=float, default=2.286, help="NFPA wall distance max (m), default 7.5 ft")
    parser.add_argument("--nfpa-min-wall-distance", type=float, default=0.1016, help="NFPA wall distance min (m), default 4 in")
    parser.add_argument(
        "--nfpa-max-avg-area-per-head",
        type=float,
        default=12.1,
        help="Draft average area/head limit (m2), e.g. 130 ft2 for ordinary hazard standard spray",
    )
    args = parser.parse_args()

    input_json = Path(args.input_json)
    out_dir = Path(args.output_dir)
    out_json = out_dir / "layout_result.json"
    out_png = out_dir / "layout_preview.png"
    out_dxf = out_dir / "layout_overlay.dxf"

    detected = json.loads(input_json.read_text(encoding="utf-8"))
    protected_area = normalize_polygon(geometry_from_json(detected.get("unified_protected_floor_area")))
    if protected_area is None:
        raise RuntimeError("No unified_protected_floor_area found in detection JSON.")

    main_vec, branch_vec = choose_axes(detected, protected_area)
    trunk = trunk_line_from_detected(detected, protected_area, main_vec)

    exclusion = build_exclusion_area(
        detected,
        column_buffer=args.column_clearance,
        stair_buffer=args.stair_clearance,
        wall_clearance=args.wall_clearance,
    )

    valid_area = protected_area
    if exclusion is not None and not exclusion.is_empty:
        valid_area = normalize_polygon(protected_area.difference(exclusion))
        if valid_area is None:
            valid_area = protected_area

    branch_offset, head_offset = choose_layout_offsets(
        valid_area=valid_area,
        trunk=trunk,
        protected_area=protected_area,
        exclusion=exclusion,
        main_vec=main_vec,
        branch_vec=branch_vec,
        branch_spacing=args.branch_spacing,
        head_spacing=args.head_spacing,
        branch_end_margin=args.branch_end_margin,
        min_obstacle_clearance=args.min_obstacle_clearance,
    )
    layout_diagnostics: dict[str, Any] = {}
    if args.layout_model == "cpsat":
        branches, kept_heads, layout_diagnostics = cpsat_branch_head_layout(
            protected_area=protected_area,
            valid_area=valid_area,
            trunk=trunk,
            exclusion=exclusion,
            branch_spacing=args.branch_spacing,
            head_spacing=args.head_spacing,
            branch_end_margin=args.branch_end_margin,
            min_obstacle_clearance=args.min_obstacle_clearance,
            demand_step=args.demand_step,
            min_head_spacing_nfpa=args.cpsat_min_head_spacing,
            weight_per_demand=args.cpsat_weight_demand,
            coeff_heads=args.cpsat_penalty_head,
            coeff_branch_len=args.cpsat_penalty_branch_m,
            coeff_phase_switch=args.cpsat_penalty_phase_switch,
            enforce_single_phase_per_side=args.cpsat_enforce_single_phase_per_side,
            time_limit_s=args.cpsat_time_limit,
            max_demand_points=args.cpsat_max_demand,
        )
    elif args.layout_model == "adaptive":
        branches, kept_heads, layout_diagnostics = adaptive_branch_and_head_layout(
            protected_area=protected_area,
            valid_area=valid_area,
            trunk=trunk,
            exclusion=exclusion,
            branch_spacing=args.branch_spacing,
            head_spacing=args.head_spacing,
            branch_end_margin=args.branch_end_margin,
            min_obstacle_clearance=args.min_obstacle_clearance,
            demand_step=args.demand_step,
            target_coverage_ratio=args.target_coverage,
        )
    else:
        branches = build_branch_lines(
            protected_area=protected_area,
            trunk=trunk,
            main_vec=main_vec,
            branch_vec=branch_vec,
            exclusion_area=exclusion,
            branch_spacing=args.branch_spacing,
            branch_offset=branch_offset,
        )
        if args.layout_model == "hybrid":
            branches = select_branches_graph_like(branches, min_station_gap=max(1.6, 0.45 * args.branch_spacing))

        kept_heads = build_candidate_heads(
            branches=branches,
            head_spacing=args.head_spacing,
            branch_end_margin=args.branch_end_margin,
            head_offset=head_offset,
            valid_area=valid_area,
            exclusion=exclusion,
            min_obstacle_clearance=args.min_obstacle_clearance,
        )
        if args.layout_model == "hybrid":
            kept_heads = optimize_heads_set_cover(
                candidates=kept_heads,
                valid_area=valid_area,
                cover_radius=0.6 * args.head_spacing,
                sample_step=1.0,
            )

    secondary_branches: list[LineString] = []
    secondary_heads: list[Point] = []
    if args.allow_secondary_branches and branches:
        secondary_branches = generate_secondary_branches(
            primary_branches=branches,
            protected_area=protected_area,
            exclusion=exclusion,
            trunk=trunk,
            spacing=args.secondary_branch_spacing,
            min_primary_length=args.secondary_min_primary_length,
            min_secondary_length=args.secondary_min_length,
            max_per_primary=max(0, int(args.secondary_max_per_primary)),
            direct_from_trunk_tolerance=args.secondary_direct_trunk_tolerance,
        )
        if secondary_branches:
            secondary_heads = build_candidate_heads(
                branches=secondary_branches,
                head_spacing=args.head_spacing,
                branch_end_margin=args.branch_end_margin,
                head_offset=head_offset,
                valid_area=valid_area,
                exclusion=exclusion,
                min_obstacle_clearance=args.min_obstacle_clearance,
            )
            branches = branches + secondary_branches
            kept_heads = dedupe_points(kept_heads + secondary_heads, tol=min(0.2, 0.1 * args.head_spacing))

    routed_diag: dict[str, Any] = {}
    if kept_heads and trunk is not None and not trunk.is_empty:
        if args.routing_model == "steiner":
            routed_segments, routed_diag = route_heads_with_rectilinear_steiner(
                heads=kept_heads,
                trunk=trunk,
                valid_area=valid_area,
                exclusion=exclusion,
                root_step=args.steiner_root_step,
                nearest_roots=max(1, int(args.steiner_nearest_roots)),
            )
            if routed_segments:
                branches = routed_segments
        elif args.routing_model == "direct":
            routed_segments, routed_diag = route_heads_direct_from_trunk(
                heads=kept_heads,
                trunk=trunk,
                valid_area=valid_area,
                exclusion=exclusion,
                root_step=args.steiner_root_step,
            )
            if routed_segments:
                branches = routed_segments

    nfpa_report: dict[str, Any] = {}
    if args.nfpa_checks:
        nfpa_report = evaluate_nfpa13_compliance(
            heads=kept_heads,
            branches=branches,
            protected_area=protected_area,
            valid_area=valid_area,
            exclusion=exclusion,
            max_spacing_m=args.nfpa_max_spacing,
            min_spacing_m=args.nfpa_min_spacing,
            max_wall_distance_m=args.nfpa_max_wall_distance,
            min_wall_distance_m=args.nfpa_min_wall_distance,
            min_obstruction_clearance_m=args.min_obstacle_clearance,
            max_avg_area_per_head_m2=args.nfpa_max_avg_area_per_head,
            target_coverage_ratio=args.target_coverage,
            demand_step=args.demand_step,
        )

    topology_diag = analyze_branch_topology(branches=branches, trunk=trunk)

    result = {
        "meta": {
            "status": "draft_layout_non_hydraulic",
            "note": "NFPA 13-style spacing inspiration only; engineer review required.",
            "input_detected_json": str(input_json),
            "layout_model": args.layout_model,
            "layout_diagnostics": layout_diagnostics,
            "routing_model": args.routing_model,
            "routing_diagnostics": routed_diag,
            "topology_diagnostics": topology_diag,
            "nfpa_compliance": nfpa_report,
        },
        "parameters": {
            "branch_spacing": args.branch_spacing,
            "head_spacing": args.head_spacing,
            "branch_offset": branch_offset,
            "head_offset": head_offset,
            "demand_step": args.demand_step,
            "target_coverage": args.target_coverage,
            "cpsat_time_limit": args.cpsat_time_limit,
            "cpsat_max_demand": args.cpsat_max_demand,
            "cpsat_min_head_spacing": args.cpsat_min_head_spacing,
            "cpsat_weight_demand": args.cpsat_weight_demand,
            "cpsat_penalty_head": args.cpsat_penalty_head,
            "cpsat_penalty_branch_m": args.cpsat_penalty_branch_m,
            "cpsat_penalty_phase_switch": args.cpsat_penalty_phase_switch,
            "cpsat_enforce_single_phase_per_side": args.cpsat_enforce_single_phase_per_side,
            "allow_secondary_branches": args.allow_secondary_branches,
            "secondary_branch_spacing": args.secondary_branch_spacing,
            "secondary_min_primary_length": args.secondary_min_primary_length,
            "secondary_min_length": args.secondary_min_length,
            "secondary_max_per_primary": args.secondary_max_per_primary,
            "secondary_direct_trunk_tolerance": args.secondary_direct_trunk_tolerance,
            "routing_model": args.routing_model,
            "steiner_root_step": args.steiner_root_step,
            "steiner_nearest_roots": args.steiner_nearest_roots,
            "nfpa_checks": args.nfpa_checks,
            "nfpa_max_spacing": args.nfpa_max_spacing,
            "nfpa_min_spacing": args.nfpa_min_spacing,
            "nfpa_max_wall_distance": args.nfpa_max_wall_distance,
            "nfpa_min_wall_distance": args.nfpa_min_wall_distance,
            "nfpa_max_avg_area_per_head": args.nfpa_max_avg_area_per_head,
            "column_clearance": args.column_clearance,
            "stair_clearance": args.stair_clearance,
            "wall_clearance": args.wall_clearance,
            "branch_end_margin": args.branch_end_margin,
            "min_obstacle_clearance": args.min_obstacle_clearance,
        },
        "decision": {
            "main_pipe_direction_unit_xy": [float(main_vec[0]), float(main_vec[1])],
            "branch_direction_unit_xy": [float(branch_vec[0]), float(branch_vec[1])],
            "main_pipe_angle_deg": float(math.degrees(math.atan2(main_vec[1], main_vec[0]))),
            "branch_angle_deg": float(math.degrees(math.atan2(branch_vec[1], branch_vec[0]))),
        },
        "geometries": {
            "protected_floor_area": geometry_to_json_dict(protected_area),
            "exclusion_area": geometry_to_json_dict(exclusion),
            "valid_coverage_area": geometry_to_json_dict(valid_area),
            "trunk_line": list(trunk.coords) if trunk is not None else [],
            "branch_lines": [list(bl.coords) for bl in branches],
            "secondary_branch_lines": [list(bl.coords) for bl in secondary_branches],
            "sprinkler_heads": [{"x": float(p.x), "y": float(p.y)} for p in kept_heads],
            "equipment_placeholders": {
                "steel_pipe_runs": [],
                "fire_hose_cabinet": [],
                "fire_department_connection": [],
                "prv": [],
                "pressure_control_valve": [],
                "floor_control_assembly": [],
            },
        },
        "counts": {
            "branch_lines": len(branches),
            "secondary_branch_lines_generated": len(secondary_branches),
            "secondary_branch_events_detected": int(topology_diag.get("secondary_events", 0)),
            "sprinkler_heads": len(kept_heads),
            "secondary_sprinkler_heads": len(secondary_heads),
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    save_preview(
        out_png,
        protected_area,
        exclusion,
        trunk,
        branches,
        kept_heads,
        trunk_diameter_label=args.trunk_diameter,
        branch_diameter_label=args.branch_diameter,
        preview_floor_label=(args.preview_floor_label.strip() or None),
    )
    write_dxf_fallback(out_dxf, trunk, branches, kept_heads)

    print("Draft layout stage complete.")
    print(f"- Branch lines: {len(branches)}")
    print(f"- Secondary branches (generated stubs): {len(secondary_branches)}")
    print(f"- Secondary branches (detected turns/splits): {int(topology_diag.get('secondary_events', 0))}")
    print(f"- Sprinkler heads: {len(kept_heads)}")
    print(f"- Secondary sprinkler heads: {len(secondary_heads)}")
    if args.verbose_layout:
        print("- Layout diagnostics:")
        for k, v in layout_diagnostics.items():
            print(f"  - {k}: {v}")
        print("- Routing diagnostics:")
        for k, v in routed_diag.items():
            print(f"  - {k}: {v}")
        print("- Topology diagnostics:")
        for k, v in topology_diag.items():
            print(f"  - {k}: {v}")
        if args.nfpa_checks:
            print("- NFPA compliance:")
            print(f"  - pass: {nfpa_report.get('pass')}")
            for chk in nfpa_report.get("checks", []):
                print(f"  - {chk.get('rule')}: pass={chk.get('pass')} | {chk.get('detail')}")
    print(f"- JSON: {out_json}")
    print(f"- Preview: {out_png}")
    print(f"- DXF: {out_dxf}")


if __name__ == "__main__":
    main()
