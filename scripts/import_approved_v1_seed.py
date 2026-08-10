from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET = REPO_ROOT / "input" / "approved_v1_layout_result.json"
EXPECTED_HEADS = 176
EXPECTED_BRANCHES = 189
EXPECTED_TRUNK_SEGMENTS = 6


def geometry_bounds(geom: dict[str, Any] | None) -> list[float] | None:
    if not isinstance(geom, dict):
        return None
    points: list[list[float]] = []
    if geom.get("type") == "Polygon":
        rings = [geom.get("exterior") or []] + (geom.get("holes") or [])
        for ring in rings:
            for point in ring:
                if len(point) >= 2:
                    points.append([float(point[0]), float(point[1])])
    elif geom.get("type") == "MultiPolygon":
        for part in geom.get("parts") or []:
            bounds = geometry_bounds(part)
            if bounds:
                points.extend([[bounds[0], bounds[1]], [bounds[2], bounds[3]]])
    if not points:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Install the approved v1 sample layout seed used to reproduce the main-PC sample garage layout."
    )
    parser.add_argument("source", type=Path, help="Path to approved layout_result.json from the main PC.")
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET, help=f"Destination path. Defaults to {DEFAULT_TARGET}.")
    parser.add_argument("--force", action="store_true", help="Copy even when the source does not match the approved sample final-layout shape.")
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    target = args.target.expanduser().resolve()
    if not source.exists():
        raise SystemExit(f"Source does not exist: {source}")
    data = json.loads(source.read_text(encoding="utf-8"))
    geoms = data.get("geometries") or {}
    heads = geoms.get("sprinkler_heads") or []
    branches = (geoms.get("branch_lines") or []) + (geoms.get("secondary_branch_lines") or [])
    trunk_segments = geoms.get("trunk_segments") or []
    bounds = geometry_bounds(geoms.get("protected_floor_area"))
    if not bounds:
        raise SystemExit("Source does not look like a layout_result.json with geometries.protected_floor_area.")
    problems = []
    if len(heads) != EXPECTED_HEADS:
        problems.append(f"expected {EXPECTED_HEADS} heads, found {len(heads)}")
    if len(geoms.get("branch_lines") or []) != EXPECTED_BRANCHES:
        problems.append(f"expected {EXPECTED_BRANCHES} branches, found {len(geoms.get('branch_lines') or [])}")
    if geoms.get("secondary_branch_lines"):
        problems.append(f"expected no secondary branches, found {len(geoms.get('secondary_branch_lines') or [])}")
    if len(trunk_segments) != EXPECTED_TRUNK_SEGMENTS:
        problems.append(f"expected {EXPECTED_TRUNK_SEGMENTS} trunk_segments, found {len(trunk_segments)}")
    if problems and not args.force:
        raise SystemExit(
            "Refusing to install this approved v1 seed: "
            + "; ".join(problems)
            + ". Export/copy the approved 176-head layout_v1/layout_result.json, or pass --force only for a deliberate custom seed."
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    print(f"Installed approved v1 seed: {target}")
    print(f"Source heads: {len(heads)}")
    print(f"Source branches: {len(branches)}")
    print(f"Protected bounds: {bounds}")
    if trunk_segments:
        print(f"Source trunk segments: {len(trunk_segments)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
