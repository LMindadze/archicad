from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from shapely.geometry import GeometryCollection, LineString, MultiLineString
from shapely.ops import linemerge, unary_union


def _line_to_coords(line: LineString) -> list[list[float]]:
    return [[float(x), float(y)] for x, y in line.coords]


def _lines_from_coords(items: list[Any]) -> list[LineString]:
    lines: list[LineString] = []
    for coords in items:
        if len(coords) >= 2:
            line = LineString([(float(x), float(y)) for x, y in coords])
            if line.length > 1e-9:
                lines.append(line)
    return lines


def _collect_lines(geom: Any, min_line_length: float) -> list[LineString]:
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, LineString):
        return [geom] if geom.length >= min_line_length else []
    if isinstance(geom, MultiLineString):
        return [line for line in geom.geoms if line.length >= min_line_length]
    if isinstance(geom, GeometryCollection):
        out: list[LineString] = []
        for part in geom.geoms:
            out.extend(_collect_lines(part, min_line_length=min_line_length))
        return out
    return []


def simplify_layout_branches(layout: dict[str, Any], *, min_line_length: float) -> tuple[dict[str, Any], dict[str, Any]]:
    geoms = layout.setdefault("geometries", {})
    original_branch_coords = geoms.get("branch_lines") or []
    original_lines = _lines_from_coords(original_branch_coords)
    if not original_lines:
        return layout, {"status": "no_branch_lines"}

    dissolved = unary_union(original_lines)
    merged = linemerge(dissolved)
    simplified_lines = _collect_lines(merged, min_line_length=min_line_length)

    before_length = float(sum(line.length for line in original_lines))
    after_length = float(sum(line.length for line in simplified_lines))
    diagnostics = {
        "status": "ok",
        "method": "shapely_unary_union_linemerge",
        "branch_lines_before": len(original_lines),
        "branch_lines_after": len(simplified_lines),
        "total_branch_length_before_m": before_length,
        "total_branch_length_after_m": after_length,
        "removed_duplicate_length_m": max(0.0, before_length - after_length),
        "min_line_length_m": float(min_line_length),
    }

    if "coverage_added_branch_lines" in geoms:
        geoms["coverage_added_branch_lines_before_simplify"] = geoms.get("coverage_added_branch_lines")
        geoms.pop("coverage_added_branch_lines", None)
    geoms["branch_lines_before_simplify"] = original_branch_coords
    geoms["branch_lines"] = [_line_to_coords(line) for line in simplified_lines]

    counts = layout.setdefault("counts", {})
    counts["branch_lines_before_simplify"] = len(original_lines)
    counts["branch_lines"] = len(simplified_lines)

    meta = layout.setdefault("meta", {})
    meta["branch_network_simplification"] = diagnostics
    return layout, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description="Dissolve overlapping branch pipe geometry while preserving heads/trunk.")
    parser.add_argument("--layout-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-line-length", type=float, default=0.05)
    args = parser.parse_args()

    layout = json.loads(Path(args.layout_json).read_text(encoding="utf-8"))
    simplified, diagnostics = simplify_layout_branches(layout, min_line_length=float(args.min_line_length))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "layout_result.json"
    out_json.write_text(json.dumps(simplified, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Branch network simplification complete.")
    print(f"- Branch lines before: {diagnostics.get('branch_lines_before')}")
    print(f"- Branch lines after: {diagnostics.get('branch_lines_after')}")
    print(f"- Branch length before: {diagnostics.get('total_branch_length_before_m'):.1f} m")
    print(f"- Branch length after: {diagnostics.get('total_branch_length_after_m'):.1f} m")
    print(f"- JSON: {out_json}")


if __name__ == "__main__":
    main()
