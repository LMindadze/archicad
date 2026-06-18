"""
Interactive main-pipe line: click START on one wall, then END on another wall.

Run after detect_parking_geometry.py has produced detected_geometry.json.

  python sprinkler2/pick_trunk_line.py --json outputs/output/detected_geometry.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Point, Polygon
from shapely.ops import nearest_points


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
        if not polys:
            return None
        return MultiPolygon(polys)
    return None


def longest_line_in_intersection(geom: Any) -> LineString | None:
    if geom is None or geom.is_empty:
        return None
    if isinstance(geom, LineString) and geom.length > 1e-6:
        return geom
    if isinstance(geom, MultiLineString):
        segs = [g for g in geom.geoms if g.length > 1e-6]
        return max(segs, key=lambda g: g.length) if segs else None
    if hasattr(geom, "geoms"):
        best: LineString | None = None
        for g in geom.geoms:
            found = longest_line_in_intersection(g)
            if found is not None and (best is None or found.length > best.length):
                best = found
        return best
    return None


def snap_to_wall(pt: tuple[float, float], walls: Polygon | MultiPolygon | None, max_dist_m: float) -> tuple[float, float]:
    if walls is None or walls.is_empty:
        return pt
    p = Point(pt[0], pt[1])
    if p.distance(walls) > max_dist_m:
        return pt
    nw, _ = nearest_points(walls.boundary, p)
    return (float(nw.x), float(nw.y))


def draw_floor(ax: Any, slab: Polygon | MultiPolygon | None, walls: Polygon | MultiPolygon | None) -> None:
    if slab is not None and not slab.is_empty:
        for g in slab.geoms if isinstance(slab, MultiPolygon) else [slab]:
            xy = np.array(g.exterior.coords)
            ax.add_patch(
                MplPolygon(
                    xy,
                    closed=True,
                    facecolor="#d1e7f5",
                    edgecolor="#24577a",
                    linewidth=1.0,
                    alpha=0.35,
                )
            )
    if walls is not None and not walls.is_empty:
        for g in walls.geoms if isinstance(walls, MultiPolygon) else [walls]:
            xy = np.array(g.exterior.coords)
            ax.add_patch(
                MplPolygon(
                    xy,
                    closed=True,
                    facecolor="#e5e5e5",
                    edgecolor="#555555",
                    linewidth=0.8,
                    alpha=0.45,
                )
            )
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("X (world)")
    ax.set_ylabel("Y (world)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pick main trunk line: two clicks (start wall, end wall).")
    parser.add_argument(
        "--json",
        type=str,
        default="outputs/output/detected_geometry.json",
        help="Path to detected_geometry.json from detect_parking_geometry.py",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Write result here (default: overwrite --json).",
    )
    parser.add_argument(
        "--snap-to-walls",
        action="store_true",
        help="Snap each click to the nearest point on walls_all_union (within --snap-max-m).",
    )
    parser.add_argument(
        "--snap-max-m",
        type=float,
        default=3.0,
        help="Maximum distance (m) for wall snap.",
    )
    parser.add_argument(
        "--save-preview",
        type=str,
        default=None,
        help="Optional PNG path to save the figure with the chosen line.",
    )
    args = parser.parse_args()

    path = Path(args.json)
    if not path.is_file():
        print(f"File not found: {path}", file=sys.stderr)
        sys.exit(1)

    data = json.loads(path.read_text(encoding="utf-8"))
    slab = geometry_from_json(data.get("unified_protected_floor_area"))
    walls = geometry_from_json(data.get("walls_all_union"))
    if slab is None or slab.is_empty:
        print("No unified_protected_floor_area in JSON.", file=sys.stderr)
        sys.exit(1)

    fig, ax = plt.subplots(figsize=(13, 9))
    draw_floor(ax, slab, walls)
    ax.set_title("Click 1: START on wall  →  Click 2: END on wall  (close window to cancel)")
    fig.tight_layout()

    try:
        pts = plt.ginput(2, timeout=0, show_clicks=True)
    except Exception as exc:  # noqa: BLE001
        print(f"Input cancelled: {exc}", file=sys.stderr)
        sys.exit(1)
    plt.close(fig)

    if len(pts) < 2:
        print("Need exactly two clicks.", file=sys.stderr)
        sys.exit(1)

    p0 = (float(pts[0][0]), float(pts[0][1]))
    p1 = (float(pts[1][0]), float(pts[1][1]))
    if args.snap_to_walls:
        p0 = snap_to_wall(p0, walls, args.snap_max_m)
        p1 = snap_to_wall(p1, walls, args.snap_max_m)

    raw = LineString([p0, p1])
    clipped = longest_line_in_intersection(raw.intersection(slab.buffer(0)))
    if clipped is None or clipped.is_empty:
        print("Clipped trunk is empty (line must cross the protected floor area).", file=sys.stderr)
        sys.exit(1)

    coords = [list(c) for c in clipped.coords]
    meta = data.get("detection_meta") or {}
    meta["trunk_source"] = "manual_gui_pick_trunk_line"
    meta["trunk_snap_to_walls"] = bool(args.snap_to_walls)
    data["detection_meta"] = meta
    data["suggested_trunk_line"] = coords
    data["trunk_endpoints"] = {
        "start_xy": [p0[0], p0[1]],
        "end_xy": [p1[0], p1[1]],
        "clipped_to_slab": True,
    }

    out_path = Path(args.output_json) if args.output_json else path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.save_preview:
        fig2, ax2 = plt.subplots(figsize=(13, 9))
        draw_floor(ax2, slab, walls)
        xs, ys = clipped.xy
        ax2.plot(xs, ys, color="#c41e3a", linewidth=3.0, label="Main trunk (manual)")
        ax2.scatter([p0[0], p1[0]], [p0[1], p1[1]], c="red", s=80, zorder=5, marker="o")
        ax2.legend(loc="best")
        ax2.set_title("Manual trunk line")
        fig2.tight_layout()
        Path(args.save_preview).parent.mkdir(parents=True, exist_ok=True)
        fig2.savefig(args.save_preview, dpi=200)
        plt.close(fig2)
        print(f"Preview: {args.save_preview}")

    print(f"Updated: {out_path}")
    print(f"Trunk length (clipped): {clipped.length:.3f} m")


if __name__ == "__main__":
    main()
