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
        line = LineString([tuple(suggested[0]), tuple(suggested[-1])])
        clipped = line.intersection(protected_area)
        if isinstance(clipped, LineString) and not clipped.is_empty:
            return clipped
        if isinstance(clipped, MultiLineString):
            segments = sorted(clipped.geoms, key=lambda g: g.length, reverse=True)
            if segments:
                return segments[0]

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


def build_branch_lines(
    protected_area: Polygon | MultiPolygon,
    trunk: LineString | None,
    main_vec: np.ndarray,
    branch_vec: np.ndarray,
    branch_spacing: float,
) -> list[LineString]:
    if trunk is None:
        return []

    c = np.array([protected_area.centroid.x, protected_area.centroid.y], dtype=float)
    hull = np.array(list(protected_area.convex_hull.exterior.coords), dtype=float)
    rel = hull - c
    along_main = rel @ main_vec
    along_branch = rel @ branch_vec
    min_main, max_main = float(np.min(along_main)), float(np.max(along_main))
    span_branch = float(max(abs(np.min(along_branch)), abs(np.max(along_branch)))) + 20.0

    branch_lines: list[LineString] = []
    t_values = np.arange(min_main, max_main + branch_spacing, branch_spacing)
    for t in t_values:
        origin = c + main_vec * t
        p0 = origin - branch_vec * span_branch
        p1 = origin + branch_vec * span_branch
        raw_line = LineString([tuple(p0), tuple(p1)])
        clipped = raw_line.intersection(protected_area)
        for seg in lines_from_intersection(clipped, min_length=1.0):
            branch_lines.append(seg)

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

    ax.plot(
        [leader_start[0], leader_end[0]],
        [leader_start[1], leader_end[1]],
        color="#444444",
        linewidth=0.8,
        alpha=0.85,
        zorder=7,
    )

    label = f"Ø{diameter_label}\nL={length:.2f}M"
    ax.text(
        text_pos[0],
        text_pos[1],
        label,
        fontsize=8.0,
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
        fontsize=7.9,
        color="#111111",
        bbox={"boxstyle": "square,pad=0.30", "fc": "white", "ec": "#9ca3af", "alpha": 0.92},
        zorder=10,
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
) -> None:
    fig, ax = plt.subplots(figsize=(13, 8))
    ax.set_facecolor("#f7f7f7")
    draw_polygon(ax, protected_area, color="#d1d5db", alpha=0.12, label="Protected area")
    draw_polygon(ax, exclusion, color="#fecaca", alpha=0.15, label="Exclusion zones")

    if trunk is not None:
        tx, ty = trunk.xy
        ax.plot(tx, ty, color="#ff2d2d", linewidth=1.4, label="Main trunk", zorder=3)
        trunk_coords = [np.array(c, dtype=float) for c in trunk.coords]
        for i in range(len(trunk_coords) - 1):
            annotate_pipe_segment(
                ax,
                trunk_coords[i],
                trunk_coords[i + 1],
                trunk_diameter_label,
                color="#111111",
                offset_scale=0.5,
            )

    for idx, bl in enumerate(branches):
        bx, by = bl.xy
        ax.plot(bx, by, color="#ff2d2d", linewidth=1.05, alpha=0.95, label="Branch lines" if idx == 0 else None, zorder=3)
        seg_coords = [np.array(c, dtype=float) for c in bl.coords]
        for i in range(len(seg_coords) - 1):
            annotate_pipe_segment(
                ax,
                seg_coords[i],
                seg_coords[i + 1],
                branch_diameter_label,
                color="#111111",
                offset_scale=0.42,
            )

    if heads:
        hx = [p.x for p in heads]
        hy = [p.y for p in heads]
        # Hollow red circles to match conventional fire-system drafting style.
        ax.scatter(hx, hy, facecolors="none", edgecolors="#ff2d2d", linewidths=1.0, s=42, label="Sprinkler heads", zorder=6)

    add_equipment_notes(ax)
    ax.set_title("Fire Suppression Layout")
    ax.set_xlabel("X (world)")
    ax.set_ylabel("Y (world)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)

    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        if l and l not in seen:
            seen[l] = h
    if seen:
        ax.legend(seen.values(), seen.keys(), loc="best")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220)
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
    parser.add_argument("--branch-spacing", type=float, default=3.8, help="Spacing between branch lines (m)")
    parser.add_argument("--head-spacing", type=float, default=3.2, help="Spacing between sprinkler heads on branch (m)")
    parser.add_argument("--column-clearance", type=float, default=0.55, help="Column buffer clearance (m)")
    parser.add_argument("--stair-clearance", type=float, default=0.8, help="Stair exclusion clearance (m)")
    parser.add_argument("--wall-clearance", type=float, default=0.3, help="Optional wall clearance buffer (m, 0 to disable)")
    parser.add_argument("--branch-end-margin", type=float, default=0.8, help="Trim heads near branch ends (m)")
    parser.add_argument("--min-obstacle-clearance", type=float, default=0.2, help="Extra min clearance from exclusion edges")
    parser.add_argument("--trunk-diameter", default="DN100", help="Main trunk diameter label")
    parser.add_argument("--branch-diameter", default="DN65", help="Branch line diameter label")
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
    branches = build_branch_lines(protected_area, trunk, main_vec, branch_vec, args.branch_spacing)

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

    candidate_heads: list[Point] = []
    for bl in branches:
        candidate_heads.extend(points_along_line(bl, args.head_spacing, endpoint_margin=args.branch_end_margin))

    kept_heads: list[Point] = []
    for p in candidate_heads:
        if valid_area is None or valid_area.is_empty:
            continue
        if not (valid_area.contains(p) or valid_area.buffer(1e-6).contains(p)):
            continue
        if exclusion is not None and not exclusion.is_empty and p.distance(exclusion) < args.min_obstacle_clearance:
            continue
        kept_heads.append(p)

    result = {
        "meta": {
            "status": "draft_deterministic_layout_non_hydraulic",
            "note": "NFPA 13-style spacing inspiration only; engineer review required.",
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
            "sprinkler_heads": len(kept_heads),
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
    )
    write_dxf_fallback(out_dxf, trunk, branches, kept_heads)

    print("Draft layout stage complete.")
    print(f"- Branch lines: {len(branches)}")
    print(f"- Sprinkler heads: {len(kept_heads)}")
    print(f"- JSON: {out_json}")
    print(f"- Preview: {out_png}")
    print(f"- DXF: {out_dxf}")


if __name__ == "__main__":
    main()
