from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import GeometryCollection, LineString, MultiLineString, MultiPolygon, Point, Polygon
from shapely.ops import nearest_points, substring, unary_union


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
        polys: list[Polygon | MultiPolygon] = [
            part for part in cleaned.geoms if isinstance(part, (Polygon, MultiPolygon)) and not part.is_empty
        ]
        if polys:
            return normalize_polygon(unary_union(polys))
    return None


def lines_from_intersection(geom: Any, min_length: float = 0.05) -> list[LineString]:
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, LineString):
        return [geom] if geom.length >= min_length else []
    if isinstance(geom, MultiLineString):
        return [line for line in geom.geoms if line.length >= min_length]
    if hasattr(geom, "geoms"):
        out: list[LineString] = []
        for part in geom.geoms:
            out.extend(lines_from_intersection(part, min_length=min_length))
        return out
    return []


def _layout_heads(layout: dict[str, Any]) -> list[dict[str, float]]:
    heads = (layout.get("geometries") or {}).get("sprinkler_heads") or []
    return [{"x": float(item["x"]), "y": float(item["y"])} for item in heads]


def _layout_branch_lines(layout: dict[str, Any]) -> list[LineString]:
    lines: list[LineString] = []
    for coords in (layout.get("geometries") or {}).get("branch_lines") or []:
        if len(coords) >= 2:
            lines.append(LineString([(float(x), float(y)) for x, y in coords]))
    return lines


def _layout_trunk_line(layout: dict[str, Any]) -> list[list[float]]:
    return [[float(x), float(y)] for x, y in (layout.get("geometries") or {}).get("trunk_line") or []]


def _line_to_coords(line: LineString) -> list[list[float]]:
    return [[float(x), float(y)] for x, y in line.coords]


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
    if not parts:
        return None
    return normalize_polygon(unary_union(parts))


def _trunk_lines_from_segments(trunk_segments: list[dict[str, Any]]) -> list[LineString]:
    lines: list[LineString] = []
    for segment in trunk_segments:
        start = segment.get("start") or []
        end = segment.get("end") or []
        if len(start) >= 2 and len(end) >= 2:
            lines.append(LineString([(float(start[0]), float(start[1])), (float(end[0]), float(end[1]))]))
    return lines


def _existing_trunk_segments(layout: dict[str, Any]) -> tuple[list[tuple[float, float]], list[dict[str, Any]], dict[str, Any]]:
    geoms = layout.get("geometries") or {}
    segments = geoms.get("trunk_segments") or []
    trunk_segments: list[dict[str, Any]] = []
    for segment in segments:
        start = segment.get("start") or []
        end = segment.get("end") or []
        if len(start) >= 2 and len(end) >= 2:
            trunk_segments.append(
                {
                    "kind": str(segment.get("kind") or f"preserved_trunk_{len(trunk_segments) + 1}"),
                    "start": [float(start[0]), float(start[1])],
                    "end": [float(end[0]), float(end[1])],
                    "diameter": str(segment.get("diameter") or "DN100"),
                }
            )
    if not trunk_segments:
        coords = geoms.get("main_trunk_line") or geoms.get("trunk_line") or []
        if len(coords) >= 2:
            for idx, (left, right) in enumerate(zip(coords, coords[1:]), start=1):
                if len(left) >= 2 and len(right) >= 2:
                    trunk_segments.append(
                        {
                            "kind": f"preserved_trunk_{idx}",
                            "start": [float(left[0]), float(left[1])],
                            "end": [float(right[0]), float(right[1])],
                            "diameter": "DN100",
                        }
                    )
    if not trunk_segments:
        return [], [], {"status": "no_existing_trunk"}
    points: list[tuple[float, float]] = []
    for segment in trunk_segments:
        start = tuple(segment["start"])
        end = tuple(segment["end"])
        if not points:
            points.append((float(start[0]), float(start[1])))
        if points[-1] != (float(end[0]), float(end[1])):
            points.append((float(end[0]), float(end[1])))
    return points, trunk_segments, {
        "status": "preserved_existing_trunk",
        "trunk_segments": len(trunk_segments),
    }


def _connector_path(
    branch_point: Point,
    trunk_point: Point,
    protected: Polygon | MultiPolygon,
) -> LineString:
    bp = (float(branch_point.x), float(branch_point.y))
    tp = (float(trunk_point.x), float(trunk_point.y))
    if abs(bp[0] - tp[0]) <= 1e-6 or abs(bp[1] - tp[1]) <= 1e-6:
        return LineString([bp, tp])

    candidates = [
        LineString([bp, (bp[0], tp[1]), tp]),
        LineString([bp, (tp[0], bp[1]), tp]),
    ]
    covered = [line for line in candidates if protected.buffer(1e-6).covers(line)]
    if covered:
        return min(covered, key=lambda line: line.length)
    return min(candidates, key=lambda line: line.length)


def build_trunk_connectors(
    branches: list[LineString],
    trunk_segments: list[dict[str, Any]],
    protected: Polygon | MultiPolygon,
    *,
    tolerance: float = 0.08,
) -> tuple[list[LineString], dict[str, Any]]:
    trunk_lines = _trunk_lines_from_segments(trunk_segments)
    if not branches or not trunk_lines:
        return [], {"status": "no_inputs", "connectors_added": 0}

    lines = trunk_lines + branches
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

    trunk_roots = {find(idx) for idx in range(len(trunk_lines))}
    branch_groups: dict[int, list[int]] = {}
    for branch_idx in range(len(branches)):
        line_idx = len(trunk_lines) + branch_idx
        root = find(line_idx)
        if root not in trunk_roots:
            branch_groups.setdefault(root, []).append(branch_idx)

    if not branch_groups:
        return [], {
            "status": "already_connected",
            "connectors_added": 0,
            "disconnected_branch_components_before": 0,
            "tolerance_m": float(tolerance),
        }

    trunk_union = unary_union(trunk_lines)
    connectors: list[LineString] = []
    diagnostics: list[dict[str, Any]] = []
    for branch_root, branch_indices in sorted(branch_groups.items(), key=lambda item: min(item[1])):
        component_geom = unary_union([branches[idx] for idx in branch_indices])
        branch_point, trunk_point = nearest_points(component_geom, trunk_union)
        connector = _connector_path(branch_point, trunk_point, protected)
        if connector.length <= tolerance:
            continue
        connectors.append(connector)
        diagnostics.append(
            {
                "branch_component": int(branch_root),
                "branch_indices": [int(idx) for idx in branch_indices],
                "branch_count": len(branch_indices),
                "nearest_distance_m": float(branch_point.distance(trunk_point)),
                "connector": _line_to_coords(connector),
            }
        )

    return connectors, {
        "status": "ok",
        "connectors_added": len(connectors),
        "disconnected_branch_components_before": len(branch_groups),
        "tolerance_m": float(tolerance),
        "connectors": diagnostics,
    }


def infer_main_trunk(
    protected_geom: Polygon | MultiPolygon,
    wall_union: Polygon | MultiPolygon | None,
    heads: list[dict[str, float]],
    *,
    routing_wall_clearance_m: float,
) -> tuple[list[tuple[float, float]], list[dict[str, Any]], dict[str, Any]]:
    """Port of the donor repo's trunk-first main-trunk placement only.

    This keeps the existing sprinkler/head package untouched. The inferred geometry is
    just the donor-style main trunk plus optional left riser/lower-trunk feed.
    """

    if len(heads) < 2:
        return [], [], {"status": "no_heads"}

    xs = [float(head["x"]) for head in heads]
    ys = [float(head["y"]) for head in heads]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    width = max(maxx - minx, 1.0)
    height = max(maxy - miny, 1.0)
    x_split_fraction = 0.27 if width / max(height, 1.0) >= 2.0 else 0.20
    x_split = minx + width * x_split_fraction

    left_heads = [head for head in heads if float(head["x"]) < x_split]
    wing_heads = [head for head in heads if head not in left_heads]
    if not wing_heads:
        wing_heads = heads[:]
        left_heads = []

    # Match donor behavior for trunk inference without mutating the saved sprinkler heads.
    infer_heads = [dict(head) for head in heads]
    infer_left_ids = {id(head) for head in left_heads}
    infer_left = [head for head in infer_heads if any(abs(head["x"] - src["x"]) < 1e-9 and abs(head["y"] - src["y"]) < 1e-9 for src in left_heads)]
    infer_wing = [head for head in infer_heads if head not in infer_left]
    lower_band = miny + height * 0.28
    upper_band = maxy - height * 0.18
    lower_span = max(lower_band - miny, 1e-6)
    upper_span = max(maxy - upper_band, 1e-6)
    for head in infer_heads:
        hy = float(head["y"])
        if hy < lower_band:
            head["y"] = hy + 2.0 * min(1.0, (lower_band - hy) / lower_span)
        elif hy > upper_band:
            head["y"] = hy - 1.35 * min(1.0, (hy - upper_band) / upper_span)

    if infer_left:
        infer_wing = [head for head in infer_heads if head not in infer_left]
    if not infer_wing:
        infer_wing = infer_heads[:]
        infer_left = []

    wing_xs = [float(head["x"]) for head in infer_wing]
    wing_ys = [float(head["y"]) for head in infer_wing]
    trunk_y = float(np.quantile(wing_ys, 0.70))

    wall_cut = wall_union.buffer(routing_wall_clearance_m) if wall_union is not None and not wall_union.is_empty else None
    trunk_free_area = protected_geom
    if wall_cut is not None and not wall_cut.is_empty:
        free_candidate = normalize_polygon(protected_geom.difference(wall_cut))
        if free_candidate is not None and not free_candidate.is_empty:
            trunk_free_area = free_candidate

    def horizontal_trunk_clear(x0: float, x1: float, y: float) -> bool:
        if abs(x1 - x0) < 1e-6:
            return True
        line = LineString([(float(x0), float(y)), (float(x1), float(y))])
        if not protected_geom.buffer(1e-6).covers(line):
            return False
        if wall_cut is not None and not wall_cut.is_empty and line.intersects(wall_cut):
            return False
        return True

    def vertical_trunk_clear(x: float, y0: float, y1: float) -> bool:
        if abs(y1 - y0) < 1e-6:
            return True
        line = LineString([(float(x), float(y0)), (float(x), float(y1))])
        if not protected_geom.buffer(1e-6).covers(line):
            return False
        if wall_cut is not None and not wall_cut.is_empty and line.intersects(wall_cut):
            return False
        return True

    def snap_clear_vertical_x(x: float, y0: float, y1: float) -> float:
        if abs(y1 - y0) < 1e-6:
            return float(x)
        offsets = [0.0]
        for idx in range(1, 21):
            offsets.extend([idx * 0.35, -idx * 0.35])
        best_x = float(x)
        best_distance = float("inf")
        for offset in offsets:
            candidate = float(x + offset)
            if not vertical_trunk_clear(candidate, y0, y1):
                continue
            distance = abs(candidate - x)
            if distance < best_distance:
                best_x = candidate
                best_distance = distance
        return best_x

    def free_hall_interval_at_x(x: float, y_hint: float) -> dict[str, float] | None:
        _, py0, _, py1 = protected_geom.bounds
        scan = LineString([(float(x), float(py0 - 2.0)), (float(x), float(py1 + 2.0))])
        pieces = lines_from_intersection(scan.intersection(trunk_free_area), min_length=0.65)
        intervals: list[tuple[float, float, float, float]] = []
        min_width = max(1.05, routing_wall_clearance_m * 2.0 + 0.35)
        for piece in pieces:
            ys_local = [float(coord[1]) for coord in piece.coords]
            lo = min(ys_local)
            hi = max(ys_local)
            interval_width = hi - lo
            if interval_width < min_width:
                continue
            intervals.append((lo, hi, (lo + hi) * 0.5, interval_width))
        if not intervals:
            return None
        containing = [item for item in intervals if item[0] - 0.45 <= y_hint <= item[1] + 0.45]
        if containing:
            lo, hi, center, interval_width = min(containing, key=lambda item: (abs(item[2] - y_hint), -item[3]))
        else:
            lo, hi, center, interval_width = min(
                intervals,
                key=lambda item: (
                    0.0 if item[0] <= y_hint <= item[1] else min(abs(y_hint - item[0]), abs(y_hint - item[1])),
                    abs(item[2] - y_hint),
                ),
            )
        if abs(center - y_hint) > max(4.75, height * 0.22):
            return None
        return {"min_y": float(lo), "max_y": float(hi), "center_y": float(center), "width_m": float(interval_width)}

    primary_riser_x: float | None = None
    riser_x: float | None = None
    riser_jog_y: float | None = None
    lower_trunk_y: float | None = None
    lower_trunk_x1: float | None = None
    if infer_left:
        riser_head_band = miny + height * 0.30
        riser_heads = [head for head in infer_left if float(head["y"]) > riser_head_band] or infer_left
        left_xs = [float(head["x"]) for head in riser_heads]
        left_ys = [float(head["y"]) for head in riser_heads]
        riser_x = float(np.quantile(left_xs, 0.52))
        primary_riser_x = float(riser_x - min(2.8, width * 0.04))
        riser_jog_y = float(np.quantile(left_ys, 0.32))
        riser_jog_y = max(min(riser_jog_y, max(left_ys)), trunk_y + min(2.2, height * 0.08))
        riser_jog_y = max(riser_jog_y, trunk_y + min(7.5, height * 0.22))
        lower_trunk_y = trunk_y - min(1.8, height * 0.05)
        lower_trunk_x1 = max(x_split, riser_x + width * 0.08)
        transfer_heads = [head for head in infer_wing if float(head["x"]) < lower_trunk_x1 - 0.25]
        if transfer_heads:
            transfer_ids = {id(head) for head in transfer_heads}
            infer_left.extend(transfer_heads)
            infer_wing = [head for head in infer_wing if id(head) not in transfer_ids]
            wing_xs = [float(head["x"]) for head in infer_wing]
            wing_ys = [float(head["y"]) for head in infer_wing]
        if wing_xs:
            lower_trunk_x1 = max(lower_trunk_x1, min(wing_xs))
        trunk_x0 = float(lower_trunk_x1)
    else:
        trunk_x0 = float(min(wing_xs))
    trunk_x1 = float(max(wing_xs))

    def build_local_centered_main_trunk(x0: float, x1: float, base_y: float) -> tuple[list[tuple[float, float]], dict[str, Any]]:
        if x1 < x0:
            x0, x1 = x1, x0
        span = max(0.0, x1 - x0)
        if span < 1e-6:
            return [(float(x0), float(base_y)), (float(x1), float(base_y))], {"local_centered": False, "reason": "zero_length"}
        sample_count = max(2, int(math.ceil(span / 1.9)) + 1)
        sample_xs = [x0 + span * idx / (sample_count - 1) for idx in range(sample_count)]
        samples: list[dict[str, Any]] = []
        for sample_x in sample_xs:
            interval = free_hall_interval_at_x(sample_x, base_y)
            sample_y = float(base_y if interval is None else interval["center_y"])
            samples.append(
                {
                    "x": float(sample_x),
                    "y": sample_y,
                    "raw_y": sample_y,
                    "source": "fallback" if interval is None else "hall_center",
                    "interval": interval,
                }
            )
        if len(samples) >= 3:
            smoothed: list[float] = []
            for idx, _sample in enumerate(samples):
                near = [float(samples[jdx]["raw_y"]) for jdx in range(max(0, idx - 1), min(len(samples), idx + 2))]
                smoothed.append(float(np.median(near)))
            for sample, sample_y in zip(samples, smoothed):
                raw_y = float(sample["raw_y"])
                sample["y"] = sample_y if abs(raw_y - sample_y) <= 0.65 else raw_y

        center_groups: list[dict[str, Any]] = []
        for sample in samples:
            sample_y = float(sample["y"])
            for group in center_groups:
                if abs(sample_y - float(group["median_y"])) <= 0.75:
                    group["values"].append(sample_y)
                    group["xs"].append(float(sample["x"]))
                    group["median_y"] = float(np.median(group["values"]))
                    break
            else:
                center_groups.append({"values": [sample_y], "xs": [float(sample["x"])], "median_y": sample_y})
        if center_groups:
            dominant = max(
                center_groups,
                key=lambda group: (
                    len(group["values"]),
                    max(group["xs"]) - min(group["xs"]) if len(group["xs"]) >= 2 else 0.0,
                    -abs(float(group["median_y"]) - base_y),
                ),
            )
            dominant_y = float(dominant["median_y"])
            candidate_ys = {dominant_y, float(base_y)}
            for group in center_groups:
                candidate_ys.add(float(group["median_y"]))
            for sample in samples:
                candidate_ys.add(float(sample["y"]))
                candidate_ys.add(float(sample["raw_y"]))
            for scan_idx in range(55):
                candidate_ys.add(float(dominant_y - 1.35 + 2.70 * scan_idx / 54))

            def score_y(candidate_y: float) -> tuple[float, float, float, int, float, float] | None:
                if not horizontal_trunk_clear(x0, x1, candidate_y):
                    return None
                support = sum(1 for sample in samples if abs(float(sample["y"]) - candidate_y) <= 0.85)
                center_offsets = [abs(float(sample["y"]) - candidate_y) for sample in samples]
                avg_center_offset = float(sum(center_offsets) / len(center_offsets)) if center_offsets else 0.0
                if wall_union is None or wall_union.is_empty:
                    return (float("inf"), float("inf"), float("inf"), support, -avg_center_offset, -abs(candidate_y - dominant_y))
                wall_distances = [Point(float(x0 + (x1 - x0) * idx / 96.0), candidate_y).distance(wall_union) for idx in range(97)]
                ordered = sorted(wall_distances)
                p10 = ordered[max(0, min(len(ordered) - 1, int(len(ordered) * 0.10)))]
                return (
                    float(min(wall_distances)),
                    float(p10),
                    float(sum(wall_distances) / len(wall_distances)),
                    support,
                    -avg_center_offset,
                    -abs(candidate_y - dominant_y),
                )

            scored: list[tuple[tuple[float, float, float, int, float, float], float]] = []
            for candidate_y in sorted(candidate_ys):
                score = score_y(float(candidate_y))
                if score is not None:
                    scored.append((score, float(candidate_y)))
            if scored:
                best_score, selected_y = max(scored, key=lambda item: item[0])
                return [(float(x0), selected_y), (float(x1), selected_y)], {
                    "local_centered": True,
                    "mode": "clearance_optimized_straight_spine",
                    "base_y": float(base_y),
                    "selected_y": float(selected_y),
                    "dominant_y": dominant_y,
                    "selected_wall_clearance": {
                        "min_m": float(best_score[0]),
                        "p10_m": float(best_score[1]),
                        "avg_m": float(best_score[2]),
                    },
                    "sample_count": len(samples),
                    "jogs": 0,
                }

        return [(float(x0), float(base_y)), (float(x1), float(base_y))], {
            "local_centered": False,
            "reason": "no_clear_centered_candidate",
            "base_y": float(base_y),
            "sample_count": len(samples),
        }

    main_points, main_diag = build_local_centered_main_trunk(trunk_x0, trunk_x1, trunk_y)
    segments: list[dict[str, Any]] = []

    def add_segment(start: tuple[float, float], end: tuple[float, float], route_model: str) -> None:
        if math.hypot(float(end[0] - start[0]), float(end[1] - start[1])) < 1e-6:
            return
        segments.append(
            {
                "start": [float(start[0]), float(start[1])],
                "end": [float(end[0]), float(end[1])],
                "kind": "trunk",
                "diameter": "DN100",
                "route_model": route_model,
            }
        )

    for start, end in zip(main_points, main_points[1:]):
        route_model = "main_trunk_jog" if abs(float(start[1]) - float(end[1])) > 0.05 else "main_trunk"
        add_segment(start, end, route_model)

    if infer_left and primary_riser_x is not None and riser_x is not None and riser_jog_y is not None and lower_trunk_y is not None and lower_trunk_x1 is not None:
        left_ys_for_riser = [float(head["y"]) for head in infer_left]
        primary_riser_x = snap_clear_vertical_x(primary_riser_x, max(left_ys_for_riser), riser_jog_y)
        riser_x = snap_clear_vertical_x(riser_x, riser_jog_y, lower_trunk_y)
        add_segment((primary_riser_x, max(left_ys_for_riser)), (primary_riser_x, riser_jog_y), "left_primary_riser")
        add_segment((primary_riser_x, riser_jog_y), (riser_x, riser_jog_y), "left_riser_jog")
        add_segment((riser_x, riser_jog_y), (riser_x, lower_trunk_y), "left_riser")
        add_segment((riser_x, lower_trunk_y), (lower_trunk_x1, lower_trunk_y), "left_lower_trunk")
        add_segment((lower_trunk_x1, lower_trunk_y), (lower_trunk_x1, main_points[0][1]), "left_lower_trunk_riser")

    return main_points, segments, {
        "status": "ok",
        "source": "donor_trunk_first_main_only",
        "trunk_y_fallback": float(trunk_y),
        "x_split": float(x_split),
        "left_head_count_for_inference": len(infer_left),
        "wing_head_count_for_inference": len(infer_wing),
        "main_trunk": main_diag,
        "preserved_existing_sprinklers": len(heads),
    }


def draw_polygon(ax: Any, geom: Polygon | MultiPolygon | None, color: str, alpha: float, label: str) -> None:
    if geom is None or geom.is_empty:
        return
    polys = [geom] if isinstance(geom, Polygon) else list(geom.geoms)
    for idx, poly in enumerate(polys):
        ext = np.array(poly.exterior.coords)
        ax.add_patch(
            MplPolygon(ext, closed=True, facecolor=color, edgecolor=color, linewidth=0.8, alpha=alpha, label=label if idx == 0 else None)
        )
        for ring in poly.interiors:
            pts = np.array(ring.coords)
            ax.add_patch(MplPolygon(pts, closed=True, facecolor="white", edgecolor=color, linewidth=0.5, alpha=1.0))


def annotate_pipe_segment(
    ax: Any,
    p0: np.ndarray,
    p1: np.ndarray,
    diameter_label: str,
    color: str = "#111111",
    offset_scale: float = 0.45,
    font_size: float = 1.5,
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
        leader_start = midpoint
        leader_end = midpoint + np.array([offset_scale, 0.0], dtype=float)
        text_pos = leader_end + np.array([0.04, 0.04], dtype=float)
        ha, va = "left", "bottom"
    else:
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
    label = f"DN {diameter_label.replace('DN', '')}\nL={length:.2f} m"
    ax.text(text_pos[0], text_pos[1], label, fontsize=font_size, color=color, ha=ha, va=va, zorder=8)


def add_equipment_notes(ax: Any) -> None:
    equipment_lines = [
        "Steel sprinkler pipe",
        "Fire hose cabinet with listed hose, DN50",
        "Fire sprinkler control valve assembly, DN77",
        "Sprinkler head, DN15, K=5.6, T=68 C",
        "Branch pipe, DN65",
        "Check valve, D65",
        "Pressure reducing valve, DN65 / D77",
        "Pressure control valve DN150-DN100",
        "Floor control assembly with flow switch, gauge, and test drain",
    ]
    notes = "\n".join(f"- {line}" for line in equipment_lines)
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
        bbox={"boxstyle": "round,pad=0.4", "fc": "white", "ec": "#ff2d2d", "linewidth": 1.0, "alpha": 0.95},
        zorder=100,
    )


def save_preview(
    out_png: Path,
    protected: Polygon | MultiPolygon,
    exclusion: Polygon | MultiPolygon | None,
    branches: list[LineString],
    heads: list[dict[str, float]],
    trunk_segments: list[dict[str, Any]],
    preview_floor_label: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.set_facecolor("#f7f7f7")
    draw_polygon(ax, protected, color="#d1d5db", alpha=0.12, label="Protected area")
    draw_polygon(ax, exclusion, color="#fecaca", alpha=0.15, label="Exclusion zones")

    head_points = [Point(float(head["x"]), float(head["y"])) for head in heads]
    head_xy = np.array([[p.x, p.y] for p in head_points], dtype=float) if head_points else np.empty((0, 2), dtype=float)

    def near_any_head(pt: np.ndarray, tol: float = 0.22) -> bool:
        if head_xy.size == 0:
            return False
        d = np.linalg.norm(head_xy - pt.reshape(1, 2), axis=1)
        return bool(np.any(d <= tol))

    def split_line_by_heads(line: LineString, tol: float = 0.22) -> list[LineString]:
        if line.length <= 1e-6:
            return []
        cuts = [0.0, float(line.length)]
        for head in head_points:
            if line.distance(head) <= tol:
                cuts.append(float(line.project(head)))
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

    for idx, branch in enumerate(branches):
        x, y = branch.xy
        ax.plot(x, y, color="#ff2d2d", linewidth=1.05, alpha=0.95, label="Branch lines" if idx == 0 else None, zorder=3)
        for seg in split_line_by_heads(branch):
            coords = [np.array(c, dtype=float) for c in seg.coords]
            if len(coords) < 2:
                continue
            p0 = coords[0]
            p1 = coords[-1]
            dia = "DN25" if (near_any_head(p0) and near_any_head(p1)) else "DN32"
            annotate_pipe_segment(ax, p0, p1, dia, color="#111111", offset_scale=0.42, font_size=1.46)

    main_color = "#1565c0"
    for idx, segment in enumerate(trunk_segments):
        sx, sy = segment["start"]
        ex, ey = segment["end"]
        ax.plot([sx, ex], [sy, ey], color="#ff2d2d", linewidth=1.55, label="Main trunk" if idx == 0 else None, zorder=4)
        annotate_pipe_segment(
            ax,
            np.array([sx, sy], dtype=float),
            np.array([ex, ey], dtype=float),
            "DN100",
            color=main_color,
            offset_scale=0.5,
            font_size=1.85,
            leader_color=main_color,
        )

    if heads:
        ax.scatter(
            [head["x"] for head in heads],
            [head["y"] for head in heads],
            facecolors="none",
            edgecolors="#ff2d2d",
            linewidths=1.0,
            s=42,
            label="Sprinkler heads",
            zorder=6,
        )
        ax.scatter([head["x"] for head in heads], [head["y"] for head in heads], c="#ff2d2d", s=7.0 / 3.0, zorder=7)

    add_equipment_notes(ax)
    ax.set_title("Fire Suppression Layout")
    ax.set_xlabel("X (world)")
    ax.set_ylabel("Y (world)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)
    draw_preview_floor_badge(ax, preview_floor_label or "")
    handles, labels = ax.get_legend_handles_labels()
    seen: dict[str, Any] = {}
    for handle, label in zip(handles, labels):
        if label and label not in seen:
            seen[label] = handle
    if seen:
        ax.legend(seen.values(), seen.keys(), loc="best")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=360)
    plt.close(fig)


def write_dxf(out_dxf: Path, branches: list[LineString], heads: list[dict[str, float]], trunk_segments: list[dict[str, Any]]) -> None:
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

    for segment in trunk_segments:
        sx, sy = segment["start"]
        ex, ey = segment["end"]
        add_line(sx, sy, ex, ey, "MAIN_TRUNK")
    for branch in branches:
        coords = list(branch.coords)
        for idx in range(len(coords) - 1):
            add_line(coords[idx][0], coords[idx][1], coords[idx + 1][0], coords[idx + 1][1], "BRANCH_EXISTING")
    for head in heads:
        lines.extend(["0", "POINT", "8", "SPRINKLER_EXISTING", "10", f"{head['x']}", "20", f"{head['y']}", "30", "0.0"])
    lines.extend(["0", "ENDSEC", "0", "EOF"])
    out_dxf.parent.mkdir(parents=True, exist_ok=True)
    out_dxf.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Add donor-style auto main trunk to an existing sprinkler layout without regenerating sprinklers.")
    parser.add_argument("--detected-json", default="outputs/output/detected_geometry.json")
    parser.add_argument("--layout-json", default="outputs/output/layout_result.json")
    parser.add_argument("--output-dir", default="outputs/output_main_trunk_only")
    parser.add_argument("--routing-wall-clearance", type=float, default=0.08)
    parser.add_argument("--column-clearance", type=float, default=0.55)
    parser.add_argument("--stair-clearance", type=float, default=0.8)
    parser.add_argument("--wall-clearance", type=float, default=0.3)
    parser.add_argument("--preview-floor-label", default=None)
    parser.add_argument(
        "--preserve-existing-trunk",
        action="store_true",
        help="Use trunk_segments already present in the source layout instead of inferring a new trunk.",
    )
    args = parser.parse_args()

    detected_path = Path(args.detected_json)
    layout_path = Path(args.layout_json)
    out_dir = Path(args.output_dir)
    out_json = out_dir / "layout_result.json"
    out_png = out_dir / "layout_preview.png"
    out_dxf = out_dir / "layout_overlay.dxf"

    detected = json.loads(detected_path.read_text(encoding="utf-8"))
    layout = json.loads(layout_path.read_text(encoding="utf-8"))

    protected = normalize_polygon(geometry_from_json(detected.get("unified_protected_floor_area")))
    if protected is None:
        raise RuntimeError("No unified_protected_floor_area found in detected geometry.")
    wall_union = normalize_polygon(geometry_from_json(detected.get("walls_all_union")))
    exclusion = build_exclusion_area(
        detected,
        column_buffer=float(args.column_clearance),
        stair_buffer=float(args.stair_clearance),
        wall_clearance=float(args.wall_clearance),
    )
    heads = _layout_heads(layout)
    branches = _layout_branch_lines(layout)

    if args.preserve_existing_trunk:
        main_points, trunk_segments, diagnostics = _existing_trunk_segments(layout)
    else:
        main_points, trunk_segments, diagnostics = infer_main_trunk(
            protected,
            wall_union,
            heads,
            routing_wall_clearance_m=float(args.routing_wall_clearance),
        )
    if not main_points:
        raise RuntimeError(f"Could not infer main trunk: {diagnostics}")

    trunk_connectors, connector_diagnostics = build_trunk_connectors(branches, trunk_segments, protected)
    output_branches = branches + trunk_connectors

    geometries = layout.setdefault("geometries", {})
    geometries["previous_trunk_line"] = _layout_trunk_line(layout)
    geometries["trunk_line"] = [[float(x), float(y)] for x, y in main_points]
    geometries["main_trunk_line"] = [[float(x), float(y)] for x, y in main_points]
    geometries["trunk_segments"] = trunk_segments
    geometries["main_trunk_connectors"] = [_line_to_coords(line) for line in trunk_connectors]
    geometries["branch_lines"] = [_line_to_coords(line) for line in output_branches]

    counts = layout.setdefault("counts", {})
    counts["branch_lines"] = len(output_branches)
    counts["original_branch_lines"] = len(branches)
    counts["main_trunk_connectors"] = len(trunk_connectors)

    meta = layout.setdefault("meta", {})
    meta["main_trunk_generation"] = diagnostics
    meta["main_trunk_generation"]["detected_json"] = str(detected_path)
    meta["main_trunk_generation"]["source_layout_json"] = str(layout_path)
    meta["main_trunk_generation"]["sprinkler_generation_preserved"] = True
    meta["main_trunk_generation"]["branch_connectivity"] = connector_diagnostics

    out_dir.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(layout, indent=2, ensure_ascii=False), encoding="utf-8")
    floor_label = args.preview_floor_label
    if floor_label is None:
        floor_label = str(detected.get("target_storey") or "").strip()
    save_preview(out_png, protected, exclusion, output_branches, heads, trunk_segments, preview_floor_label=floor_label)
    write_dxf(out_dxf, output_branches, heads, trunk_segments)

    print("Main trunk post-process complete.")
    print(f"- Existing heads preserved: {len(heads)}")
    print(f"- Existing branch lines preserved: {len(branches)}")
    print(f"- Main trunk connectors added: {len(trunk_connectors)}")
    print(f"- Main trunk points: {len(main_points)}")
    print(f"- Trunk segments: {len(trunk_segments)}")
    print(f"- JSON: {out_json}")
    print(f"- Preview: {out_png}")
    print(f"- DXF: {out_dxf}")


if __name__ == "__main__":
    main()
