from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon
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
        for p in data.get("parts", []):
            g = geometry_from_json(p)
            if isinstance(g, Polygon) and not g.is_empty:
                polys.append(g)
            elif isinstance(g, MultiPolygon):
                polys.extend([x for x in g.geoms if not x.is_empty])
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


def choose_axes(detected: dict[str, Any], protected_area: Polygon | MultiPolygon) -> tuple[np.ndarray, np.ndarray]:
    axes = detected.get("candidate_axes")
    if axes and axes.get("main_axis") and axes.get("branch_axis"):
        main = np.array(axes["main_axis"]["unit_vector_xy"], dtype=float)
        branch = np.array(axes["branch_axis"]["unit_vector_xy"], dtype=float)
    else:
        mrr = protected_area.minimum_rotated_rectangle
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
    main = main / np.linalg.norm(main)
    branch = branch / np.linalg.norm(branch)
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

    parts = []
    if cols is not None and not cols.is_empty and column_buffer > 0:
        parts.append(cols.buffer(column_buffer))
    if stairs is not None and not stairs.is_empty and stair_buffer > 0:
        parts.append(stairs.buffer(stair_buffer))
    if walls is not None and not walls.is_empty and wall_clearance > 0:
        parts.append(walls.buffer(wall_clearance))
    if not parts:
        return None
    return normalize_polygon(unary_union(parts))


def point_and_proj(point: np.ndarray, origin: np.ndarray, axis: np.ndarray) -> float:
    return float(np.dot(point - origin, axis))


def candidate_main_lines(valid_area: Polygon | MultiPolygon, main_vec: np.ndarray, branch_vec: np.ndarray, scan_step: float) -> list[LineString]:
    centroid = np.array([valid_area.centroid.x, valid_area.centroid.y], dtype=float)
    hull = np.array(list(valid_area.convex_hull.exterior.coords), dtype=float)
    rel = hull - centroid
    branch_proj = rel @ branch_vec
    span = float(max(abs(np.min(branch_proj)), abs(np.max(branch_proj)))) + 10.0
    line_span = math.hypot(valid_area.bounds[2] - valid_area.bounds[0], valid_area.bounds[3] - valid_area.bounds[1]) + 10.0

    offsets = np.arange(float(np.min(branch_proj)), float(np.max(branch_proj)) + scan_step, scan_step)
    candidates: list[LineString] = []
    for off in offsets:
        base = centroid + branch_vec * off
        raw = LineString([tuple(base - main_vec * line_span), tuple(base + main_vec * line_span)])
        clipped = lines_from_intersection(raw.intersection(valid_area), min_length=2.0)
        candidates.extend(clipped)
    return candidates


def score_main_line(line: LineString, valid_area: Polygon | MultiPolygon, origin: np.ndarray, branch_vec: np.ndarray) -> float:
    coords = [np.array(c, dtype=float) for c in line.coords]
    if len(coords) < 2:
        return -1e9
    length = line.length
    bends = max(0, len(coords) - 2)
    midpoint = np.array(line.interpolate(0.5, normalized=True).coords[0], dtype=float)
    off_center = abs(point_and_proj(midpoint, origin, branch_vec))

    # prefer long, central, simple lines
    score = 0.0
    score += length * 10.0
    score -= bends * 8.0
    score -= off_center * 1.2
    # penalize lines too close to boundary everywhere
    sample_ds = np.linspace(0.1, max(0.11, line.length - 0.1), num=max(3, int(line.length / 4.0)))
    boundary_penalty = 0.0
    for d in sample_ds:
        p = line.interpolate(float(min(d, line.length)))
        boundary_penalty += max(0.0, 0.6 - p.distance(valid_area.boundary))
    score -= boundary_penalty * 5.0
    return score


def choose_main_line(valid_area: Polygon | MultiPolygon, main_vec: np.ndarray, branch_vec: np.ndarray, scan_step: float) -> LineString | None:
    centroid = np.array([valid_area.centroid.x, valid_area.centroid.y], dtype=float)
    cands = candidate_main_lines(valid_area, main_vec, branch_vec, scan_step)
    if not cands:
        return None
    best = max(cands, key=lambda ln: score_main_line(ln, valid_area, centroid, branch_vec))
    return best


def sample_positions_along_line(line: LineString, step: float, end_margin: float) -> list[tuple[np.ndarray, float]]:
    usable = max(0.0, line.length - 2.0 * end_margin)
    if usable <= 0.01:
        return []
    count = max(1, int(math.floor(usable / step)) + 1)
    out: list[tuple[np.ndarray, float]] = []
    for i in range(count):
        d = end_margin + i * step
        if d > line.length - end_margin + 1e-9:
            break
        p = line.interpolate(d)
        out.append((np.array([p.x, p.y], dtype=float), d))
    return out


def trim_segment_from_base(seg: LineString, base: np.ndarray, outward_vec: np.ndarray, trim: float) -> LineString:
    a = np.array(seg.coords[0], dtype=float)
    b = np.array(seg.coords[-1], dtype=float)
    if np.dot(b - a, outward_vec) < 0:
        a, b = b, a
    vec = b - a
    length = float(np.linalg.norm(vec))
    if length <= trim + 0.05:
        return LineString([tuple(a), tuple(b)])
    new_a = a + (vec / length) * trim
    return LineString([tuple(new_a), tuple(b)])


def dedupe_similar_lines(lines: Iterable[LineString], tol: float) -> list[LineString]:
    kept: list[LineString] = []
    seen: set[tuple[int, int, int, int]] = set()
    for line in lines:
        a = np.array(line.coords[0], dtype=float)
        b = np.array(line.coords[-1], dtype=float)
        sig = (
            int(round(a[0] / tol)),
            int(round(a[1] / tol)),
            int(round(b[0] / tol)),
            int(round(b[1] / tol)),
        )
        rev = (sig[2], sig[3], sig[0], sig[1])
        if sig in seen or rev in seen:
            continue
        seen.add(sig)
        kept.append(line)
    return kept


def build_branch_lines(
    valid_area: Polygon | MultiPolygon,
    main_line: LineString | None,
    branch_vec: np.ndarray,
    branch_spacing: float,
    head_spacing: float,
    branch_end_margin: float,
) -> list[LineString]:
    if main_line is None:
        return []

    minx, miny, maxx, maxy = valid_area.bounds
    cap = math.hypot(maxx - minx, maxy - miny) + 5.0
    sample_margin = min(1.2, branch_spacing * 0.5)
    min_branch_length = max(1.2, head_spacing * 0.9)
    branches: list[LineString] = []

    for base, _ in sample_positions_along_line(main_line, branch_spacing, sample_margin):
        raw = LineString([tuple(base - branch_vec * cap), tuple(base + branch_vec * cap)])
        clipped = lines_from_intersection(raw.intersection(valid_area), min_length=min_branch_length)
        if not clipped:
            continue

        pos_best: LineString | None = None
        neg_best: LineString | None = None
        pos_len = 0.0
        neg_len = 0.0
        for seg in clipped:
            mid = np.array(seg.interpolate(0.5, normalized=True).coords[0], dtype=float)
            sign = float(np.dot(mid - base, branch_vec))
            if sign >= 0:
                if seg.length > pos_len:
                    pos_best = seg
                    pos_len = seg.length
            else:
                if seg.length > neg_len:
                    neg_best = seg
                    neg_len = seg.length

        # default: use both if both are substantial and balanced enough, else use dominant side
        if pos_len >= min_branch_length and neg_len >= min_branch_length:
            ratio = min(pos_len, neg_len) / max(pos_len, neg_len)
            if ratio >= 0.68:
                branches.append(trim_segment_from_base(pos_best, base, branch_vec, 0.18))
                branches.append(trim_segment_from_base(neg_best, base, -branch_vec, 0.18))
            elif pos_len > neg_len:
                branches.append(trim_segment_from_base(pos_best, base, branch_vec, 0.18))
            else:
                branches.append(trim_segment_from_base(neg_best, base, -branch_vec, 0.18))
        elif pos_len >= min_branch_length:
            branches.append(trim_segment_from_base(pos_best, base, branch_vec, 0.18))
        elif neg_len >= min_branch_length:
            branches.append(trim_segment_from_base(neg_best, base, -branch_vec, 0.18))

    # keep only meaningful branches
    filtered = [bl for bl in branches if bl.length >= max(1.2, head_spacing + branch_end_margin * 0.4)]
    return dedupe_similar_lines(filtered, tol=0.25)


def points_along_branch(line: LineString, spacing: float, endpoint_margin: float) -> list[Point]:
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


def save_preview(
    out_png: Path,
    protected_area: Polygon | MultiPolygon | None,
    exclusion: Polygon | MultiPolygon | None,
    valid_area: Polygon | MultiPolygon | None,
    main_line: LineString | None,
    branches: list[LineString],
    heads: list[Point],
) -> None:
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.set_facecolor("#f7f7f7")
    draw_polygon(ax, protected_area, color="#cbd5e1", alpha=0.22, label="Floor boundary (slab union)")
    draw_polygon(ax, valid_area, color="#dbeafe", alpha=0.35, label="Head placement domain")
    draw_polygon(ax, exclusion, color="#fecaca", alpha=0.18, label="Obstacle buffer")

    if main_line is not None:
        x, y = main_line.xy
        ax.plot(x, y, color="#dc2626", linewidth=2.0, label="Main", zorder=4)

    for idx, bl in enumerate(branches):
        bx, by = bl.xy
        ax.plot(bx, by, color="#ef4444", linewidth=1.2, alpha=0.95, label="Branches" if idx == 0 else None, zorder=4)

    if heads:
        hx = [p.x for p in heads]
        hy = [p.y for p in heads]
        ax.scatter(hx, hy, facecolors="none", edgecolors="#ea580c", linewidths=1.0, s=36, label="Sprinklers (r≈250)", zorder=6)

    notes = (
        "Draft sprinkler grid from IFC geometry\n"
        "Single-main heuristic coordination layout\n"
        "Not code-compliant or hydraulic design\n"
        "Peer review required before construction"
    )
    ax.text(0.77, 0.94, notes, transform=ax.transAxes, ha="left", va="top", fontsize=8.5, color="#111827")

    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        if l and l not in seen:
            seen[l] = h
    if seen:
        ax.legend(seen.values(), seen.keys(), loc="upper left")

    ax.set_title("Sprinkler layout — draft coordination drawing")
    ax.set_xlabel("Easting / IFC X (m)")
    ax.set_ylabel("Northing / IFC Y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.18)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def write_dxf_fallback(out_dxf: Path, main_line: LineString | None, branches: list[LineString], heads: list[Point]) -> None:
    lines = ["0", "SECTION", "2", "ENTITIES"]

    def add_line(x1: float, y1: float, x2: float, y2: float, layer: str) -> None:
        lines.extend([
            "0", "LINE", "8", layer,
            "10", f"{x1}", "20", f"{y1}", "30", "0.0",
            "11", f"{x2}", "21", f"{y2}", "31", "0.0",
        ])

    if main_line is not None and not main_line.is_empty:
        coords = list(main_line.coords)
        for i in range(len(coords) - 1):
            add_line(coords[i][0], coords[i][1], coords[i + 1][0], coords[i + 1][1], "MAIN")

    for bl in branches:
        coords = list(bl.coords)
        for i in range(len(coords) - 1):
            add_line(coords[i][0], coords[i][1], coords[i + 1][0], coords[i + 1][1], "BRANCH")

    for p in heads:
        lines.extend(["0", "POINT", "8", "SPRINKLER", "10", f"{p.x}", "20", f"{p.y}", "30", "0.0"])

    lines.extend(["0", "ENDSEC", "0", "EOF"])
    out_dxf.parent.mkdir(parents=True, exist_ok=True)
    out_dxf.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate simpler single-main draft sprinkler layout from detected geometry.")
    parser.add_argument("--input-json", default="outputs/output/detected_geometry.json", help="Detected geometry JSON path")
    parser.add_argument("--output-dir", default="outputs/output", help="Output directory")
    parser.add_argument("--branch-spacing", type=float, default=3.8, help="Spacing along main between branch takeoffs (m)")
    parser.add_argument("--head-spacing", type=float, default=3.2, help="Spacing between sprinkler heads on each branch (m)")
    parser.add_argument("--column-clearance", type=float, default=0.55, help="Column buffer clearance (m)")
    parser.add_argument("--stair-clearance", type=float, default=0.8, help="Stair exclusion clearance (m)")
    parser.add_argument("--wall-clearance", type=float, default=0.3, help="Wall clearance buffer (m)")
    parser.add_argument("--branch-end-margin", type=float, default=0.8, help="Trim heads near branch ends (m)")
    parser.add_argument("--min-obstacle-clearance", type=float, default=0.2, help="Extra min clearance from exclusion edges")
    parser.add_argument("--main-scan-step", type=float, default=1.2, help="Perpendicular scan step when selecting the main line (m)")
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

    main_vec, branch_vec = choose_axes(detected, protected_area)
    main_line = choose_main_line(valid_area, main_vec, branch_vec, args.main_scan_step)
    branches = build_branch_lines(valid_area, main_line, branch_vec, args.branch_spacing, args.head_spacing, args.branch_end_margin)

    heads: list[Point] = []
    for bl in branches:
        for p in points_along_branch(bl, args.head_spacing, args.branch_end_margin):
            if point_clear(p, valid_area, exclusion, args.min_obstacle_clearance):
                heads.append(p)

    result = {
        "meta": {
            "status": "draft_single_main_layout_non_hydraulic",
            "note": "Single-main coordination layout from IFC geometry; engineer review required.",
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
            "main_scan_step": args.main_scan_step,
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
            "main_line": list(main_line.coords) if main_line is not None else [],
            "branch_lines": [list(bl.coords) for bl in branches],
            "sprinkler_heads": [{"x": float(p.x), "y": float(p.y)} for p in heads],
        },
        "counts": {
            "branch_lines": len(branches),
            "sprinkler_heads": len(heads),
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    save_preview(out_png, protected_area, exclusion, valid_area, main_line, branches, heads)
    write_dxf_fallback(out_dxf, main_line, branches, heads)

    print("Draft layout stage complete.")
    print(f"- Branch lines: {len(branches)}")
    print(f"- Sprinkler heads: {len(heads)}")
    print(f"- JSON: {out_json}")
    print(f"- Preview: {out_png}")
    print(f"- DXF: {out_dxf}")


if __name__ == "__main__":
    main()
