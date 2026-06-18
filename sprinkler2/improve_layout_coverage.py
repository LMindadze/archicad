from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from shapely.geometry import LineString, MultiPolygon, Point, Polygon

from score_layout import (
    build_exclusion_area,
    geometry_from_json,
    lines_from_coords,
    normalize_polygon,
    sample_points_inside_polygon,
    trunk_lines_from_layout,
)


def _heads_from_layout(layout: dict[str, Any]) -> list[Point]:
    return [
        Point(float(item["x"]), float(item["y"]))
        for item in (layout.get("geometries") or {}).get("sprinkler_heads") or []
    ]


def _point_array(points: list[Point]) -> np.ndarray:
    if not points:
        return np.empty((0, 2), dtype=float)
    return np.array([[point.x, point.y] for point in points], dtype=float)


def _line_to_coords(line: LineString) -> list[list[float]]:
    return [[float(x), float(y)] for x, y in line.coords]


def _coverage_mask(sample_xy: np.ndarray, head_xy: np.ndarray, radius: float) -> np.ndarray:
    if len(sample_xy) == 0:
        return np.zeros(0, dtype=bool)
    if len(head_xy) == 0:
        return np.zeros(len(sample_xy), dtype=bool)
    covered = np.zeros(len(sample_xy), dtype=bool)
    radius_sq = radius * radius
    for head in head_xy:
        delta = sample_xy - head.reshape(1, 2)
        covered |= np.sum(delta * delta, axis=1) <= radius_sq + 1e-9
    return covered


def _far_enough(point_xy: np.ndarray, head_xy: np.ndarray, min_spacing: float) -> bool:
    if len(head_xy) == 0:
        return True
    delta = head_xy - point_xy.reshape(1, 2)
    return bool(np.all(np.sum(delta * delta, axis=1) >= (min_spacing * min_spacing) - 1e-9))


def _route_allowed(
    line: LineString,
    route_area: Polygon | MultiPolygon,
    exclusion: Polygon | MultiPolygon | None,
    max_connector_length: float,
) -> bool:
    if line.is_empty or line.length <= 0.05 or line.length > max_connector_length:
        return False
    if not route_area.covers(line):
        return False
    if exclusion is not None and not exclusion.is_empty and line.crosses(exclusion):
        return False
    return True


def _anchors_from_lines(network_lines: list[LineString], step: float = 1.0) -> list[Point]:
    anchors: list[Point] = []
    for line in network_lines:
        if line.is_empty:
            continue
        coords = list(line.coords)
        if coords:
            anchors.append(Point(coords[0]))
            anchors.append(Point(coords[-1]))
        if line.length > step:
            count = max(1, int(math.floor(line.length / max(step, 0.25))))
            for idx in range(1, count):
                anchors.append(line.interpolate(idx * line.length / count))
    out: list[Point] = []
    seen: set[tuple[int, int]] = set()
    for point in anchors:
        key = (round(point.x * 1000), round(point.y * 1000))
        if key in seen:
            continue
        seen.add(key)
        out.append(point)
    return out


def _route_to_anchors(
    head: Point,
    anchors: list[Point],
    anchor_xy: np.ndarray,
    route_area: Polygon | MultiPolygon,
    exclusion: Polygon | MultiPolygon | None,
    max_connector_length: float,
    anchor_limit: int,
    allow_diagonal: bool,
) -> LineString | None:
    if not anchors:
        return None
    hp_xy = (float(head.x), float(head.y))
    delta = anchor_xy - np.array(hp_xy, dtype=float).reshape(1, 2)
    order = np.argsort(np.sum(delta * delta, axis=1))[: max(1, int(anchor_limit))]
    allowed: list[LineString] = []
    for anchor_idx in order:
        network_point = anchors[int(anchor_idx)]
        np_xy = (float(network_point.x), float(network_point.y))
        candidates = [
            LineString([np_xy, (np_xy[0], hp_xy[1]), hp_xy]),
            LineString([np_xy, (hp_xy[0], np_xy[1]), hp_xy]),
        ]
        if allow_diagonal or abs(np_xy[0] - hp_xy[0]) <= 1e-6 or abs(np_xy[1] - hp_xy[1]) <= 1e-6:
            candidates.insert(0, LineString([np_xy, hp_xy]))
        allowed.extend(
            line
            for line in candidates
            if _route_allowed(line, route_area, exclusion, max_connector_length=max_connector_length)
        )
    if not allowed:
        return None
    return min(allowed, key=lambda line: line.length)


def improve_coverage(
    detected: dict[str, Any],
    layout: dict[str, Any],
    *,
    demand_step: float,
    cover_radius: float,
    min_head_spacing: float,
    min_obstruction_clearance: float,
    max_added_heads: int,
    min_gain: int,
    max_connector_length: float,
    anchor_limit: int,
    allow_diagonal_connectors: bool,
    route_through_walls: bool,
    column_clearance: float,
    stair_clearance: float,
    wall_clearance: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    protected = normalize_polygon(geometry_from_json(detected.get("unified_protected_floor_area")))
    if protected is None:
        raise RuntimeError("No unified_protected_floor_area found in detected geometry.")
    exclusion = build_exclusion_area(
        detected,
        column_buffer=column_clearance,
        stair_buffer=stair_clearance,
        wall_clearance=wall_clearance,
    )
    valid_area: Polygon | MultiPolygon = protected
    if exclusion is not None and not exclusion.is_empty:
        diff = normalize_polygon(protected.difference(exclusion))
        if diff is not None and not diff.is_empty:
            valid_area = diff
    route_exclusion = exclusion
    route_valid_area: Polygon | MultiPolygon = valid_area
    if route_through_walls:
        route_exclusion = build_exclusion_area(
            detected,
            column_buffer=column_clearance,
            stair_buffer=stair_clearance,
            wall_clearance=0.0,
        )
        route_valid_area = protected
        if route_exclusion is not None and not route_exclusion.is_empty:
            route_diff = normalize_polygon(protected.difference(route_exclusion))
            if route_diff is not None and not route_diff.is_empty:
                route_valid_area = route_diff

    geoms = layout.setdefault("geometries", {})
    heads = _heads_from_layout(layout)
    branches = lines_from_coords(geoms.get("branch_lines") or [])
    trunk_lines = trunk_lines_from_layout(layout)
    network_lines = trunk_lines + branches
    if not network_lines:
        raise RuntimeError("No connected pipe network found in layout.")
    base_anchors = _anchors_from_lines(network_lines, step=1.0)
    base_anchor_xy = _point_array(base_anchors)
    route_area = route_valid_area.buffer(1e-6)

    samples = sample_points_inside_polygon(valid_area, demand_step)
    sample_xy = _point_array(samples)
    head_xy = _point_array(heads)
    initially_covered = _coverage_mask(sample_xy, head_xy, cover_radius)
    uncovered = ~initially_covered

    candidate_indices: list[int] = []
    for idx, sample in enumerate(samples):
        if not uncovered[idx]:
            continue
        if not _far_enough(sample_xy[idx], head_xy, min_head_spacing):
            continue
        if sample.distance(protected.boundary) < 0.1016 - 1e-9:
            continue
        if exclusion is not None and not exclusion.is_empty and sample.distance(exclusion) < min_obstruction_clearance - 1e-9:
            continue
        candidate_indices.append(idx)

    if candidate_indices:
        candidate_xy = sample_xy[candidate_indices]
        delta = candidate_xy[:, None, :] - sample_xy[None, :, :]
        candidate_covers = np.sum(delta * delta, axis=2) <= (cover_radius * cover_radius) + 1e-9
    else:
        candidate_xy = np.empty((0, 2), dtype=float)
        candidate_covers = np.empty((0, len(sample_xy)), dtype=bool)

    candidate_routes: list[LineString | None] = []
    for point_xy in candidate_xy:
        candidate_routes.append(
            _route_to_anchors(
                Point(float(point_xy[0]), float(point_xy[1])),
                base_anchors,
                base_anchor_xy,
                route_area,
                route_exclusion,
                max_connector_length=max_connector_length,
                anchor_limit=anchor_limit,
                allow_diagonal=allow_diagonal_connectors,
            )
        )

    selected_heads: list[Point] = []
    selected_lines: list[LineString] = []
    selected_anchors: list[Point] = []
    selected_candidate_rows: set[int] = set()
    head_xy_current = head_xy.copy()

    for _ in range(max(0, int(max_added_heads))):
        best_row: int | None = None
        best_gain = 0
        best_route: LineString | None = None
        selected_anchor_xy = _point_array(selected_anchors)
        gains = candidate_covers[:, uncovered].sum(axis=1) if len(candidate_indices) else np.array([], dtype=int)
        order = np.argsort(-gains)
        for row in order:
            if int(row) in selected_candidate_rows:
                continue
            gain = int(gains[row])
            if gain < min_gain:
                break
            point_xy = candidate_xy[row]
            route = candidate_routes[int(row)]
            if route is None and selected_anchors:
                route = _route_to_anchors(
                    Point(float(point_xy[0]), float(point_xy[1])),
                    selected_anchors,
                    selected_anchor_xy,
                    route_area,
                    route_exclusion,
                    max_connector_length=max_connector_length,
                    anchor_limit=min(anchor_limit, 24),
                    allow_diagonal=allow_diagonal_connectors,
                )
            if route is None:
                continue
            if not _far_enough(point_xy, head_xy_current, min_head_spacing):
                continue
            best_row = int(row)
            best_gain = gain
            best_route = route
            break

        if best_row is None or best_route is None or best_gain < min_gain:
            break

        point_xy = candidate_xy[best_row]
        head = Point(float(point_xy[0]), float(point_xy[1]))
        selected_heads.append(head)
        selected_lines.append(best_route)
        selected_anchors.extend(_anchors_from_lines([best_route], step=1.0))
        selected_candidate_rows.add(best_row)
        uncovered &= ~candidate_covers[best_row]
        head_xy_current = np.vstack([head_xy_current, point_xy.reshape(1, 2)])

    final_heads = heads + selected_heads
    final_branches = branches + selected_lines
    final_covered = _coverage_mask(sample_xy, _point_array(final_heads), cover_radius)

    geoms["sprinkler_heads"] = [{"x": float(point.x), "y": float(point.y)} for point in final_heads]
    geoms["branch_lines"] = [_line_to_coords(line) for line in final_branches]
    geoms["coverage_added_branch_lines"] = [_line_to_coords(line) for line in selected_lines]

    counts = layout.setdefault("counts", {})
    counts["sprinkler_heads"] = len(final_heads)
    counts["branch_lines"] = len(final_branches)
    counts["coverage_added_heads"] = len(selected_heads)
    counts["coverage_added_branch_lines"] = len(selected_lines)

    diagnostics = {
        "status": "ok",
        "source": "v1_connected_layout_coverage_postprocess",
        "initial_heads": len(heads),
        "added_heads": len(selected_heads),
        "initial_branch_lines": len(branches),
        "added_branch_lines": len(selected_lines),
        "sample_points": len(samples),
        "initial_covered_points": int(initially_covered.sum()),
        "final_covered_points": int(final_covered.sum()),
        "initial_coverage_ratio": float(initially_covered.sum() / max(1, len(samples))),
        "final_coverage_ratio": float(final_covered.sum() / max(1, len(samples))),
        "remaining_uncovered_points": int((~final_covered).sum()),
        "candidate_points": len(candidate_indices),
        "route_feasible_candidates": sum(1 for route in candidate_routes if route is not None),
        "cover_radius_m": float(cover_radius),
        "min_head_spacing_m": float(min_head_spacing),
        "max_connector_length_m": float(max_connector_length),
        "anchor_limit": int(anchor_limit),
        "allow_diagonal_connectors": bool(allow_diagonal_connectors),
        "route_through_walls": bool(route_through_walls),
        "min_gain_points": int(min_gain),
    }
    meta = layout.setdefault("meta", {})
    meta["coverage_improvement"] = diagnostics
    return layout, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description="Add connected coverage heads to an existing trunk-connected layout.")
    parser.add_argument("--detected-json", default="outputs/output/detected_geometry.json")
    parser.add_argument("--layout-json", required=True)
    parser.add_argument("--output-dir", default="outputs/output_v2_coverage")
    parser.add_argument("--demand-step", type=float, default=1.0)
    parser.add_argument("--cover-radius", type=float, default=1.92)
    parser.add_argument("--min-head-spacing", type=float, default=1.8288)
    parser.add_argument("--min-obstruction-clearance", type=float, default=0.2)
    parser.add_argument("--max-added-heads", type=int, default=80)
    parser.add_argument("--min-gain", type=int, default=4)
    parser.add_argument("--max-connector-length", type=float, default=5.0)
    parser.add_argument("--anchor-limit", type=int, default=32)
    parser.add_argument("--allow-diagonal-connectors", action="store_true")
    parser.add_argument(
        "--route-through-walls",
        action="store_true",
        help="Keep heads in valid area, but let branch connector routing ignore wall-clearance exclusion.",
    )
    parser.add_argument("--column-clearance", type=float, default=0.55)
    parser.add_argument("--stair-clearance", type=float, default=0.8)
    parser.add_argument("--wall-clearance", type=float, default=0.3)
    args = parser.parse_args()

    detected = json.loads(Path(args.detected_json).read_text(encoding="utf-8"))
    layout = json.loads(Path(args.layout_json).read_text(encoding="utf-8"))
    improved, diagnostics = improve_coverage(
        detected,
        layout,
        demand_step=args.demand_step,
        cover_radius=args.cover_radius,
        min_head_spacing=args.min_head_spacing,
        min_obstruction_clearance=args.min_obstruction_clearance,
        max_added_heads=args.max_added_heads,
        min_gain=args.min_gain,
        max_connector_length=args.max_connector_length,
        anchor_limit=args.anchor_limit,
        allow_diagonal_connectors=args.allow_diagonal_connectors,
        route_through_walls=args.route_through_walls,
        column_clearance=args.column_clearance,
        stair_clearance=args.stair_clearance,
        wall_clearance=args.wall_clearance,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "layout_result.json"
    out_json.write_text(json.dumps(improved, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Coverage improvement complete.")
    print(f"- Initial coverage: {diagnostics['initial_coverage_ratio']:.3f}")
    print(f"- Final coverage: {diagnostics['final_coverage_ratio']:.3f}")
    print(f"- Added heads: {diagnostics['added_heads']}")
    print(f"- Added branch lines: {diagnostics['added_branch_lines']}")
    print(f"- Remaining uncovered samples: {diagnostics['remaining_uncovered_points']}")
    print(f"- JSON: {out_json}")


if __name__ == "__main__":
    main()
