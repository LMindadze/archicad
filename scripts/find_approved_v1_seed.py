from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET = REPO_ROOT / "input" / "approved_v1_layout_result.json"
SKIP_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    "__pycache__",
    "node_modules",
    "dist",
    "AppData",
}


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


def iter_layout_files(roots: list[Path]) -> Any:
    for root in roots:
        if root.is_file() and root.name == "layout_result.json":
            yield root
            continue
        if not root.exists():
            continue
        for current, dirs, files in os.walk(root):
            dirs[:] = [
                name
                for name in dirs
                if name not in SKIP_DIRS and not name.startswith(".venv") and not name.endswith(".egg-info")
            ]
            if "layout_result.json" in files:
                yield Path(current) / "layout_result.json"


def read_summary(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    geoms = data.get("geometries") or {}
    counts = data.get("counts") or {}
    heads = counts.get("sprinkler_heads")
    if heads is None:
        heads = len(geoms.get("sprinkler_heads") or [])
    branches = counts.get("branch_lines")
    if branches is None:
        branches = len((geoms.get("branch_lines") or []) + (geoms.get("secondary_branch_lines") or []))
    trunk_segments = geoms.get("trunk_segments") or []
    bounds = geometry_bounds(geoms.get("protected_floor_area"))
    if bounds is None:
        return None
    text_path = str(path)
    is_base = "layout_base" in text_path or text_path.endswith(str(Path("outputs") / "output" / "layout_result.json"))
    return {
        "path": path,
        "heads": int(heads),
        "branches": int(branches),
        "trunks": len(trunk_segments),
        "is_base": is_base,
        "bounds": bounds,
    }


def score_candidate(item: dict[str, Any], target_heads: int | None, target_branches: int | None) -> tuple[int, int, int, int]:
    score = 0
    if target_heads is None or item["heads"] == target_heads:
        score += 100
    if target_branches is None or item["branches"] == target_branches:
        score += 25
    if item["is_base"]:
        score += 10
    if item["trunks"] == 0:
        score += 5
    return (score, 1 if item["is_base"] else 0, -int(item["trunks"]), -int(item["branches"]))


def main() -> int:
    parser = argparse.ArgumentParser(description="Find a local layout_result.json that can be used as the approved v1 seed.")
    parser.add_argument("roots", nargs="*", type=Path, help="Folders or layout_result.json files to scan.")
    parser.add_argument("--heads", type=int, default=176, help="Preferred sprinkler head count. Defaults to 176.")
    parser.add_argument("--branches", type=int, default=None, help="Optional preferred branch count.")
    parser.add_argument("--limit", type=int, default=20, help="Maximum rows to print.")
    parser.add_argument("--install", action="store_true", help=f"Copy the best candidate to {DEFAULT_TARGET}.")
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET, help="Install destination.")
    args = parser.parse_args()

    roots = [path.expanduser().resolve() for path in args.roots] if args.roots else [REPO_ROOT]
    candidates = [item for path in iter_layout_files(roots) if (item := read_summary(path))]
    if args.heads is not None:
        candidates = [item for item in candidates if item["heads"] == args.heads]
    if args.branches is not None:
        candidates = [item for item in candidates if item["branches"] == args.branches]
    candidates.sort(key=lambda item: score_candidate(item, args.heads, args.branches), reverse=True)

    if not candidates:
        print("No matching layout_result.json files found.")
        return 1

    for item in candidates[: max(1, args.limit)]:
        kind = "base" if item["is_base"] else "final/other"
        print(f"{item['heads']:>4} heads  {item['branches']:>4} branches  {item['trunks']:>2} trunks  {kind}  {item['path']}")
        print(f"     bounds: {item['bounds']}")

    if args.install:
        target = args.target.expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidates[0]["path"], target)
        print(f"Installed seed: {target}")
        if candidates[0]["trunks"]:
            print("Warning: installed candidate contains trunk_segments; prefer a base layout if available.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
