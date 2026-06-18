from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPolygon,
    Point,
    Polygon,
)
from shapely.ops import unary_union


@dataclass
class ZoneLayout:
    zone_id: int
    polygon: Polygon | MultiPolygon
    center: Point
    main_axis: np.ndarray
    branch_axis: np.ndarray
    main_segment: LineString | None
    branch_lines: list[LineString]
    sprinkler_heads: list[Point]


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
        polys: list[Polygon] = []
        for p in parts:
            g = geometry_from_json(p)
            if isinstance(g, Polygon) and not g.is_empty:
                polys.append(g)
            elif isinstance(g, MultiPolygon):
                polys.extend(list(g.geoms))
        if polys:
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


def choose_global_axes(detected: dict[str, Any], protected_area: Polygon | MultiPolygon) -> tuple[np.ndarray, np.ndarray]:
    axes = detected.get("candidate_axes")
    if axes and axes.get("main_axis") and axes.get("branch_axis"):
        main = np.array(axes["main_axis"]["unit_vector_xy"], dtype=float)
        branch = np.array(axes["branch_axis"]["unit_vector_xy"], dtype=float)
    else:
        main, branch = principal_axes(protected_area)
    main = main / np.linalg.norm(main)
    branch = branch / np.linalg.norm(branch)
    return main, branch


def principal_axes(geom: Polygon | MultiPolygon) -> tuple[np.ndarray, np.ndarray]:
    mrr = geom.minimum_rotated_rectangle
    pts = list(mrr.exterior.coords)
    edges: list[tuple[float, np.ndarray]] = []
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
    return main, branch


def build_exclusion_area(
    detected: dict[str, Any],
    column_buffer: float,
    stair_buffer: float,
    wall_clearance: float,
) -> Polygon | MultiPolygon | None:
    cols = geometry_from_json(detected.get("columns_union"))
    stairs = geometry_from_json(detected.get("stairs_union"))
    walls = geometry_from_json(detected.get("walls_all_union"))

    parts: list[Polygon | MultiPolygon] = []
    if cols is not None and not cols.is_empty and column_buffer > 0:
        parts.append(cols.buffer(column_buffer))
    if stairs is not None and not stairs.is_empty and stair_buffer > 0:
        parts.append(stairs.buffer(stair_buffer))
    if walls is not None and not walls.is_empty and wall_clearance > 0:
        parts.append(walls.buffer(wall_clearance))
    if not parts:
        return None
    return normalize_polygon(unary_union(parts))


def split_into_zones(
    valid_area: Polygon | MultiPolygon,
    neck_width: float,
    min_area: float,
) -> list[Polygon | MultiPolygon]:
    pieces = [valid_area] if isinstance(valid_area, Polygon) else list(valid_area.geoms)
    all_zones: list[Polygon | MultiPolygon] = []

    for poly in pieces:
        if poly.area < min_area:
            all_zones.append(poly)
            continue

        eroded = poly.buffer(-neck_width)
        if eroded.is_empty:
            all_zones.append(poly)
            continue

        eroded_parts: list[Polygon]
        if isinstance(eroded, Polygon):
            eroded_parts = [eroded]
        elif isinstance(eroded, MultiPolygon):
            eroded_parts = list(eroded.geoms)
        else:
            all_zones.append(poly)
            continue

        grown = [normalize_polygon(part.buffer(neck_width).intersection(poly)) for part in eroded_parts]
        grown = [g for g in grown if g is not None and not g.is_empty]

        if len(grown) <= 1:
            all_zones.append(poly)
            continue

        for zone in grown:
            if zone.area >= min_area * 0.35:
                all_zones.append(zone)

        assigned = normalize_polygon(unary_union(all_zones))
        if assigned is not None:
            remainder = normalize_polygon(poly.difference(unary_union(grown)))
            if remainder is not None and not remainder.is_empty:
                if isinstance(remainder, Polygon):
                    if remainder.area >= min_area * 0.2:
                        all_zones.append(remainder)
                else:
                    for r in remainder.geoms:
                        if r.area >= min_area * 0.2:
                            all_zones.append(r)

    filtered = [z for z in all_zones if z.area >= min_area * 0.15]
    if not filtered:
        return [valid_area]
    filtered.sort(key=lambda g: g.area, reverse=True)
    return filtered


def line_through_point(point: np.ndarray, vec: np.ndarray, half_span: float) -> LineString:
    return LineString([tuple(point - vec * half_span), tuple(point + vec * half_span)])


def longest_clipped_line(polygon: Polygon | MultiPolygon, point: np.ndarray, vec: np.ndarray) -> LineString | None:
    minx, miny, maxx, maxy = polygon.bounds
    diag = math.hypot(maxx - minx, maxy - miny) + 10.0
    raw = line_through_point(point, vec, diag)
    clipped = raw.intersection(polygon)
    segments = lines_from_intersection(clipped, min_length=0.5)
    if not segments:
        return None
    segments.sort(key=lambda s: s.length, reverse=True)
    return segments[0]


def order_line_endpoints(line: LineString, vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.array(line.coords[0], dtype=float)
    b = np.array(line.coords[-1], dtype=float)
    if float(np.dot(b - a, vec)) >= 0.0:
        return a, b
    return b, a


def sample_positions_along_line(line: LineString, step: float, end_margin: float) -> list[tuple[Point, float]]:
    usable = max(0.0, line.length - 2.0 * end_margin)
    if usable <= 0.01:
        return []
    count = max(1, int(math.floor(usable / step)) + 1)
    out: list[tuple[Point, float]] = []
    for i in range(count):
        d = end_margin + i * step
        if d > line.length - end_margin + 1e-9:
            break
        out.append((line.interpolate(d), d))
    return out


def choose_side_segments(
    zone: Polygon | MultiPolygon,
    main_line: LineString,
    branch_vec: np.ndarray,
    branch_length_cap: float,
    min_branch_length: float,
    main_sample_step: float,
    main_end_margin: float,
) -> list[LineString]:
    branches: list[LineString] = []
    samples = sample_positions_along_line(main_line, step=main_sample_step, end_margin=main_end_margin)
    if not samples:
        return branches

    for point, _ in samples:
        base = np.array([point.x, point.y], dtype=float)
        full = LineString(
            [
                tuple(base - branch_vec * branch_length_cap),
                tuple(base + branch_vec * branch_length_cap),
            ]
        )
        clipped = lines_from_intersection(full.intersection(zone), min_length=min_branch_length)
        if not clipped:
            continue

        positive_best: LineString | None = None
        negative_best: LineString | None = None
        for seg in clipped:
            coords = [np.array(c, dtype=float) for c in seg.coords]
            mid = (coords[0] + coords[-1]) * 0.5
            sign = float(np.dot(mid - base, branch_vec))
            if sign >= 0:
                if positive_best is None or seg.length > positive_best.length:
                    positive_best = seg
            else:
                if negative_best is None or seg.length > negative_best.length:
                    negative_best = seg

        pos_len = positive_best.length if positive_best is not None else 0.0
        neg_len = negative_best.length if negative_best is not None else 0.0

        keep_positive = pos_len >= min_branch_length and pos_len >= 0.9 * neg_len
        keep_negative = neg_len >= min_branch_length and neg_len >= 0.9 * pos_len

        if abs(pos_len - neg_len) > main_sample_step * 0.6:
            keep_positive = pos_len > neg_len and pos_len >= min_branch_length
            keep_negative = neg_len > pos_len and neg_len >= min_branch_length

        if pos_len >= min_branch_length and neg_len < min_branch_length:
            keep_positive = True
            keep_negative = False
        if neg_len >= min_branch_length and pos_len < min_branch_length:
            keep_negative = True
            keep_positive = False

        if keep_positive and positive_best is not None:
            branches.append(trim_line_near_main(positive_best, base, branch_vec, main_connection_trim=0.18))
        if keep_negative and negative_best is not None:
            branches.append(trim_line_near_main(negative_best, base, -branch_vec, main_connection_trim=0.18))

    return dedupe_similar_lines(branches, tolerance=0.25)


def trim_line_near_main(line: LineString, base: np.ndarray, out_vec: np.ndarray, main_connection_trim: float) -> LineString:
    coords = [np.array(c, dtype=float) for c in line.coords]
    start = coords[0]
    end = coords[-1]
    if float(np.dot(end - start, out_vec)) < 0:
        start, end = end, start
    vec = end - start
    length = float(np.linalg.norm(vec))
    if length <= main_connection_trim + 0.05:
        return LineString([tuple(start), tuple(end)])
    new_start = start + (vec / length) * main_connection_trim
    return LineString([tuple(new_start), tuple(end)])


def dedupe_similar_lines(lines: Iterable[LineString], tolerance: float) -> list[LineString]:
    kept: list[LineString] = []
    signatures: set[tuple[int, int, int, int]] = set()
    for line in lines:
        a = np.array(line.coords[0], dtype=float)
        b = np.array(line.coords[-1], dtype=float)
        sig = (
            int(round(a[0] / tolerance)),
            int(round(a[1] / tolerance)),
            int(round(b[0] / tolerance)),
            int(round(b[1] / tolerance)),
        )
        rev = (sig[2], sig[3], sig[0], sig[1])
        if sig in signatures or rev in signatures:
            continue
        signatures.add(sig)
        kept.append(line)
    return kept


def points_along_branch(line: LineString, spacing: float, endpoint_margin: float = 0.0) -> list[Point]:
    usable = max(0.0, line.length - 2.0 * endpoint_margin)
    if usable <= 0.01:
        return []
    count = max(1, int(math.floor(usable / spacing)) + 1)
    pts: list[Point] = []
    for i in range(count):
        d = endpoint_margin + i * spacing
        if d > line.length - endpoint_margin + 1e-9:
            break
        pts.append(line.interpolate(d))
    return pts


def point_clear(p: Point, valid_area: Polygon | MultiPolygon, exclusion: Polygon | MultiPolygon | None, min_clearance: float) -> bool:
    if not (valid_area.contains(p) or valid_area.buffer(1e-6).contains(p)):
        return False
    if exclusion is not None and not exclusion.is_empty and p.distance(exclusion) < min_clearance:
        return False
    return True


def build_zone_layout(
    zone_id: int,
    zone: Polygon | MultiPolygon,
    global_main: np.ndarray,
    branch_spacing: float,
    head_spacing: float,
    branch_end_margin: float,
    min_obstacle_clearance: float,
    exclusion: Polygon | MultiPolygon | None,
) -> ZoneLayout:
    local_main, local_branch = principal_axes(zone)
    if abs(float(np.dot(local_main, global_main))) < abs(float(np.dot(local_branch, global_main))):
        local_main, local_branch = local_branch, local_main
    if float(np.dot(local_main, global_main)) < 0:
        local_main = -local_main
    local_branch = np.array([-local_main[1], local_main[0]], dtype=float)

    center = zone.representative_point()
    main_line = longest_clipped_line(zone, np.array([center.x, center.y], dtype=float), local_main)

    branch_lines: list[LineString] = []
    sprinkler_heads: list[Point] = []
    if main_line is not None:
        minx, miny, maxx, maxy = zone.bounds
        span = math.hypot(maxx - minx, maxy - miny) + 6.0
        branch_lines = choose_side_segments(
            zone=zone,
            main_line=main_line,
            branch_vec=local_branch,
            branch_length_cap=span,
            min_branch_length=max(1.0, head_spacing * 0.8),
            main_sample_step=branch_spacing,
            main_end_margin=min(branch_spacing * 0.5, 1.0),
        )

        for bl in branch_lines:
            for head in points_along_branch(bl, spacing=head_spacing, endpoint_margin=branch_end_margin):
                if point_clear(head, zone, exclusion, min_obstacle_clearance):
                    sprinkler_heads.append(head)

    return ZoneLayout(
        zone_id=zone_id,
        polygon=zone,
        center=center,
        main_axis=local_main,
        branch_axis=local_branch,
        main_segment=main_line,
        branch_lines=branch_lines,
        sprinkler_heads=sprinkler_heads,
    )


def connect_main_runs(layouts: list[ZoneLayout], global_main: np.ndarray, valid_area: Polygon | MultiPolygon) -> list[LineString]:
    usable = [l for l in layouts if l.main_segment is not None]
    if not usable:
        return []

    usable.sort(key=lambda z: float(np.dot(np.array([z.center.x, z.center.y]), global_main)))
    runs: list[LineString] = [z.main_segment for z in usable if z.main_segment is not None]

    connectors: list[LineString] = []
    for a, b in zip(usable, usable[1:]):
        if a.main_segment is None or b.main_segment is None:
            continue
        a0, a1 = order_line_endpoints(a.main_segment, global_main)
        b0, b1 = order_line_endpoints(b.main_segment, global_main)
        candidates = [
            LineString([tuple(a1), tuple(b0)]),
            LineString([tuple(a1), tuple(b1)]),
            LineString([tuple(a0), tuple(b0)]),
            LineString([tuple(a0), tuple(b1)]),
        ]
        best_line: LineString | None = None
        best_score = float("inf")
        for line in candidates:
            score = line.length
            clipped = line.intersection(valid_area.buffer(0.25))
            if clipped.is_empty:
                score += 1000.0
            elif not isinstance(clipped, (LineString, MultiLineString)):
                score += 500.0
            if score < best_score:
                best_score = score
                best_line = line
        if best_line is not None:
            connectors.append(best_line)

    return runs + connectors


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


def add_notes(ax: Any) -> None:
    notes = (
        "Draft sprinkler grid from IFC geometry\n"
        "Heuristic coordination layout only\n"
        "Not code-compliant or hydraulic design\n"
        "Peer review required before construction"
    )
    ax.text(
        0.77,
        0.94,
        notes,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color="#111827",
    )


def add_equipment_tags(ax: Any) -> None:
    text = (
        "Typical equipment tags\n\n"
        "P — Carbon steel sprinkler piping (schedule per spec)\n"
        "FH — Fire hose station with listed hose and cabinet\n"
        "FDC — Fire department connection — listed, sized per water supply\n"
        "PRV — Pressure reducing valve — listed, set per calculation\n"
        "DRV — Deluge / release trim — per system type\n"
        "FCA — Floor control assembly — listed\n"
        "GV — General duty valve — supervised where required"
    )
    ax.text(
        0.77,
        0.58,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.6,
        color="#111827",
    )


def save_preview(
    out_png: Path,
    protected_area: Polygon | MultiPolygon | None,
    valid_area: Polygon | MultiPolygon | None,
    exclusion: Polygon | MultiPolygon | None,
    zones: list[ZoneLayout],
    main_runs: list[LineString],
    sprinkler_heads: list[Point],
) -> None:
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_facecolor("#f4f4f5")
    draw_polygon(ax, protected_area, color="#94a3b8", alpha=0.10, label="Floor boundary (slab union)")
    draw_polygon(ax, valid_area, color="#cbd5e1", alpha=0.14, label="Head placement domain")
    draw_polygon(ax, exclusion, color="#fca5a5", alpha=0.18, label="Obstacle buffer")

    for idx, zone in enumerate(zones):
        draw_polygon(ax, zone.polygon, color="#dbeafe", alpha=0.08, label="Zones" if idx == 0 else "")

    for idx, run in enumerate(main_runs):
        x, y = run.xy
        ax.plot(x, y, color="#d62828", linewidth=1.7, label="Main" if idx == 0 else None, zorder=4)

    branch_label_used = False
    for zone in zones:
        for bl in zone.branch_lines:
            x, y = bl.xy
            ax.plot(
                x,
                y,
                color="#ef4444",
                linewidth=0.95,
                alpha=0.95,
                label="Branches" if not branch_label_used else None,
                zorder=4,
            )
            branch_label_used = True

    if sprinkler_heads:
        hx = [p.x for p in sprinkler_heads]
        hy = [p.y for p in sprinkler_heads]
        ax.scatter(
            hx,
            hy,
            s=26,
            facecolors="none",
            edgecolors="#ea580c",
            linewidths=0.8,
            label="Sprinklers (r≈250)",
            zorder=6,
        )

    add_notes(ax)
    add_equipment_tags(ax)

    ax.set_title("Sprinkler layout — draft coordination drawing", fontsize=14, weight="bold", color="#1f2937")
    ax.set_xlabel("Easting / IFC X (m)")
    ax.set_ylabel("Northing / IFC Y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.18)
    handles, labels = ax.get_legend_handles_labels()
    seen: dict[str, Any] = {}
    for h, l in zip(handles, labels):
        if l and l not in seen:
            seen[l] = h
    if seen:
        ax.legend(seen.values(), seen.keys(), loc="upper left", fontsize=7.5)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def write_dxf_fallback(out_dxf: Path, main_runs: list[LineString], branches: list[LineString], heads: list[Point]) -> None:
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

    for line in main_runs:
        coords = list(line.coords)
        for i in range(len(coords) - 1):
            add_line(coords[i][0], coords[i][1], coords[i + 1][0], coords[i + 1][1], "MAIN")

    for line in branches:
        coords = list(line.coords)
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
    parser = argparse.ArgumentParser(description="Generate an improved draft sprinkler coordination layout from detected IFC geometry.")
    parser.add_argument("--input-json", default="outputs/output/detected_geometry.json", help="Detected geometry JSON path")
    parser.add_argument("--output-dir", default="outputs/output", help="Output directory")
    parser.add_argument("--branch-spacing", type=float, default=3.8, help="Spacing between branch takeoffs along zone main (m)")
    parser.add_argument("--head-spacing", type=float, default=3.2, help="Spacing between sprinkler heads on branch (m)")
    parser.add_argument("--column-clearance", type=float, default=0.60, help="Column buffer clearance (m)")
    parser.add_argument("--stair-clearance", type=float, default=0.90, help="Stair exclusion clearance (m)")
    parser.add_argument("--wall-clearance", type=float, default=0.30, help="Wall buffer clearance (m)")
    parser.add_argument("--branch-end-margin", type=float, default=0.70, help="Trim heads near branch ends (m)")
    parser.add_argument("--min-obstacle-clearance", type=float, default=0.22, help="Extra min clearance from exclusion edges (m)")
    parser.add_argument("--zone-neck-width", type=float, default=1.40, help="Morphological neck width used to split zones (m)")
    parser.add_argument("--min-zone-area", type=float, default=12.0, help="Ignore tiny broken zones below this area (m²)")
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

    global_main, _ = choose_global_axes(detected, protected_area)
    exclusion = build_exclusion_area(
        detected,
        column_buffer=args.column_clearance,
        stair_buffer=args.stair_clearance,
        wall_clearance=args.wall_clearance,
    )

    valid_area = protected_area
    if exclusion is not None and not exclusion.is_empty:
        diff = normalize_polygon(protected_area.difference(exclusion))
        if diff is not None and not diff.is_empty:
            valid_area = diff

    zones = split_into_zones(valid_area, neck_width=args.zone_neck_width, min_area=args.min_zone_area)
    zone_layouts: list[ZoneLayout] = []
    for i, zone in enumerate(zones, start=1):
        zone_layouts.append(
            build_zone_layout(
                zone_id=i,
                zone=zone,
                global_main=global_main,
                branch_spacing=args.branch_spacing,
                head_spacing=args.head_spacing,
                branch_end_margin=args.branch_end_margin,
                min_obstacle_clearance=args.min_obstacle_clearance,
                exclusion=exclusion,
            )
        )

    main_runs = connect_main_runs(zone_layouts, global_main=global_main, valid_area=valid_area)
    all_branches = [bl for zone in zone_layouts for bl in zone.branch_lines]
    sprinkler_heads = [p for zone in zone_layouts for p in zone.sprinkler_heads]

    result = {
        "meta": {
            "status": "draft_heuristic_layout_non_hydraulic",
            "note": "Heuristic sprinkler coordination layout from IFC geometry; engineer review required.",
            "input_detected_json": str(input_json),
        },
        "parameters": {
            "branch_spacing": args.branch_spacing,
            "head_spacing": args.head_spacing,
            "column_clearance": args.column_clearance,
            "stair_clearance": args.stair_clearance,
            "wall_clearance": args.wall_clearance,
            "branch_end_margin": args.branch_end_margin,
            "min_obstacle_clearance": args.min_obstacle_clearance,
            "zone_neck_width": args.zone_neck_width,
            "min_zone_area": args.min_zone_area,
        },
        "decision": {
            "global_main_direction_unit_xy": [float(global_main[0]), float(global_main[1])],
            "global_main_angle_deg": float(math.degrees(math.atan2(global_main[1], global_main[0]))),
            "zone_count": len(zone_layouts),
        },
        "geometries": {
            "protected_floor_area": geometry_to_json_dict(protected_area),
            "exclusion_area": geometry_to_json_dict(exclusion),
            "valid_coverage_area": geometry_to_json_dict(valid_area),
            "main_runs": [list(line.coords) for line in main_runs],
            "zones": [
                {
                    "zone_id": z.zone_id,
                    "polygon": geometry_to_json_dict(z.polygon),
                    "center": [float(z.center.x), float(z.center.y)],
                    "main_axis": [float(z.main_axis[0]), float(z.main_axis[1])],
                    "branch_axis": [float(z.branch_axis[0]), float(z.branch_axis[1])],
                    "main_segment": list(z.main_segment.coords) if z.main_segment is not None else [],
                    "branch_lines": [list(bl.coords) for bl in z.branch_lines],
                    "sprinkler_heads": [{"x": float(p.x), "y": float(p.y)} for p in z.sprinkler_heads],
                }
                for z in zone_layouts
            ],
            "sprinkler_heads": [{"x": float(p.x), "y": float(p.y)} for p in sprinkler_heads],
        },
        "counts": {
            "zones": len(zone_layouts),
            "main_runs": len(main_runs),
            "branch_lines": len(all_branches),
            "sprinkler_heads": len(sprinkler_heads),
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    save_preview(out_png, protected_area, valid_area, exclusion, zone_layouts, main_runs, sprinkler_heads)
    write_dxf_fallback(out_dxf, main_runs, all_branches, sprinkler_heads)

    print("Improved draft layout stage complete.")
    print(f"- Zones: {len(zone_layouts)}")
    print(f"- Main runs: {len(main_runs)}")
    print(f"- Branch lines: {len(all_branches)}")
    print(f"- Sprinkler heads: {len(sprinkler_heads)}")
    print(f"- JSON: {out_json}")
    print(f"- Preview: {out_png}")
    print(f"- DXF: {out_dxf}")


if __name__ == "__main__":
    main()
