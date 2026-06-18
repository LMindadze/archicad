from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import ifcopenshell
from ifcopenshell.util.unit import calculate_unit_scale
from shapely.geometry import LineString, Point

from sprinkler2.export_revit_handoff import (
    FT_PER_M,
    build_context,
    build_heads,
    build_pipe_runs,
    write_context_csv,
    write_csvs,
)
from sprinkler_app.storage import (
    DEFAULT_SETTINGS,
    OUTPUTS_ROOT,
    REPO_ROOT,
    file_url,
    find_floor,
    load_project,
    now_iso,
    project_dir,
    relative_project_path,
    run_dir,
    safe_slug,
    save_project,
    save_run,
    write_json,
)


PYREVIT_EXE = Path(os.environ.get("APPDATA", "")) / "pyRevit-Master" / "bin" / "pyrevit.exe"
TRACKED_TYPES = ("IfcSlab", "IfcColumn", "IfcStair", "IfcWallStandardCase", "IfcWall", "IfcSpace")
APPROVED_V1_BASE_LAYOUT = OUTPUTS_ROOT / "output" / "layout_result.json"


class PipelineCancelled(Exception):
    pass


class CommandRunner:
    def __init__(self, log: Callable[[str], None], should_cancel: Callable[[], bool] | None = None) -> None:
        self.log = log
        self.should_cancel = should_cancel or (lambda: False)
        self.current_process: subprocess.Popen[str] | None = None

    def run(self, cmd: list[str], *, env: dict[str, str] | None = None) -> None:
        if self.should_cancel():
            raise PipelineCancelled()
        self.log("> " + " ".join(cmd))
        process = subprocess.Popen(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        self.current_process = process
        assert process.stdout is not None
        for line in process.stdout:
            self.log(line.rstrip())
            if self.should_cancel() and process.poll() is None:
                process.terminate()
        code = process.wait()
        self.current_process = None
        if self.should_cancel():
            raise PipelineCancelled()
        if code != 0:
            raise RuntimeError("Command failed with exit {0}: {1}".format(code, " ".join(cmd)))

    def terminate(self) -> None:
        if self.current_process and self.current_process.poll() is None:
            self.current_process.terminate()


def storey_element_counts(model: Any) -> dict[int, dict[str, int]]:
    counts: dict[int, dict[str, int]] = {}
    for rel in model.by_type("IfcRelContainedInSpatialStructure"):
        storey = getattr(rel, "RelatingStructure", None)
        if not storey or not storey.is_a("IfcBuildingStorey"):
            continue
        bucket = counts.setdefault(storey.id(), {t: 0 for t in TRACKED_TYPES})
        for elem in getattr(rel, "RelatedElements", []) or []:
            for t in TRACKED_TYPES:
                if elem.is_a(t):
                    bucket[t] += 1
                    break
    return counts


def analyze_ifc(project: dict[str, Any]) -> dict[str, Any]:
    pdir = project_dir(project["id"])
    ifc_path = pdir / project["source_ifc"]
    model = ifcopenshell.open(str(ifc_path))
    unit_scale_to_m = float(calculate_unit_scale(model) or 1.0)
    counts = storey_element_counts(model)
    storeys: list[dict[str, Any]] = []
    for idx, storey in enumerate(sorted(model.by_type("IfcBuildingStorey"), key=lambda s: (float(getattr(s, "Elevation", 0.0) or 0.0), str(getattr(s, "Name", ""))))):
        name = getattr(storey, "Name", None) or f"Storey {idx + 1}"
        elevation = getattr(storey, "Elevation", 0.0)
        try:
            elevation_m = float(elevation or 0.0) * unit_scale_to_m
        except (TypeError, ValueError):
            elevation_m = 0.0
        floor_id = f"floor_{idx + 1:02d}_{safe_slug(name, 'storey')}"
        storeys.append(
            {
                "id": floor_id,
                "index": idx + 1,
                "ifc_id": storey.id(),
                "global_id": getattr(storey, "GlobalId", None),
                "name": name,
                "elevation_m": elevation_m,
                "counts": counts.get(storey.id(), {t: 0 for t in TRACKED_TYPES}),
                "selected": idx == 0,
                "status": "analyzed",
                "trunk_override": None,
            }
        )
    project["storeys"] = storeys
    project["status"] = "analyzed"
    save_project(project)
    return project


def floor_folder(run_path: Path, floor: dict[str, Any]) -> Path:
    return run_path / f"{floor['index']:02d}_{safe_slug(floor['name'], 'storey')}"


def geometry_summary(detected_path: Path) -> dict[str, Any]:
    data = json.loads(detected_path.read_text(encoding="utf-8"))
    return {
        "protected_floor_area": data.get("unified_protected_floor_area"),
        "columns": data.get("columns", []),
        "walls": (data.get("walls_standard_case") or []) + (data.get("walls_generic") or []),
        "stairs": data.get("stairs", []),
        "bounds": data.get("overall_floor_bounds"),
        "trunk_line": data.get("suggested_trunk_line"),
        "candidate_axes": data.get("candidate_axes"),
    }


def project_relative_path(project: dict[str, Any], value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return project_dir(project["id"]) / path


def geometry_points(geom: dict[str, Any] | None) -> list[list[float]]:
    if not isinstance(geom, dict):
        return []
    if geom.get("type") == "Polygon":
        points: list[list[float]] = []
        rings = [geom.get("exterior") or []] + (geom.get("holes") or [])
        for ring in rings:
            for point in ring:
                if len(point) >= 2:
                    points.append([float(point[0]), float(point[1])])
        return points
    if geom.get("type") == "MultiPolygon":
        points = []
        for part in geom.get("parts") or []:
            points.extend(geometry_points(part))
        return points
    return []


def geometry_bounds(geom: dict[str, Any] | None) -> list[float] | None:
    points = geometry_points(geom)
    if not points:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def bounds_match(left: list[float] | None, right: list[float] | None, tolerance: float = 0.05) -> bool:
    if not left or not right or len(left) != 4 or len(right) != 4:
        return False
    return all(abs(float(a) - float(b)) <= tolerance for a, b in zip(left, right))


def settings_match_approved_v1(settings: dict[str, Any]) -> bool:
    expected = {
        "branch_spacing": 3.8,
        "head_spacing": 3.2,
        "column_clearance": 0.55,
        "stair_clearance": 0.8,
        "wall_clearance": 0.3,
        "min_obstacle_clearance": 0.2,
        "demand_step": 1.0,
        "target_coverage": 0.96,
        "cpsat_min_head_spacing": 1.8288,
    }
    for key, value in expected.items():
        try:
            if abs(float(settings.get(key, DEFAULT_SETTINGS.get(key))) - float(value)) > 1e-6:
                return False
        except (TypeError, ValueError):
            return False
    return (
        str(settings.get("layout_model", DEFAULT_SETTINGS["layout_model"])) == "cpsat"
        and str(settings.get("routing_model", DEFAULT_SETTINGS["routing_model"])) == "direct"
        and not bool(settings.get("allow_secondary_branches", DEFAULT_SETTINGS["allow_secondary_branches"]))
    )


def approved_v1_seed_layout(detected_path: Path, settings: dict[str, Any]) -> Path | None:
    if not APPROVED_V1_BASE_LAYOUT.exists() or not settings_match_approved_v1(settings):
        return None
    try:
        detected = json.loads(detected_path.read_text(encoding="utf-8"))
        seed = json.loads(APPROVED_V1_BASE_LAYOUT.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    detected_bounds = geometry_bounds(detected.get("unified_protected_floor_area"))
    seed_bounds = geometry_bounds((seed.get("geometries") or {}).get("protected_floor_area"))
    if not bounds_match(detected_bounds, seed_bounds):
        return None
    return APPROVED_V1_BASE_LAYOUT


def build_base_layout(
    runner: CommandRunner,
    working_detected: Path,
    base_dir: Path,
    floor_name: str,
    settings: dict[str, Any],
    *,
    allow_approved_seed: bool,
) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    seed = approved_v1_seed_layout(working_detected, settings) if allow_approved_seed else None
    if seed:
        seeded_layout = json.loads(seed.read_text(encoding="utf-8"))
        meta = seeded_layout.setdefault("meta", {})
        meta["input_detected_json"] = str(working_detected)
        meta["approved_v1_seed_layout"] = str(seed)
        (base_dir / "layout_result.json").write_text(json.dumps(seeded_layout, indent=2, ensure_ascii=False), encoding="utf-8")
        runner.log(f"Using approved v1 base layout seed: {seed}")
        return base_dir / "layout_result.json"
    runner.run(
        [
            sys.executable,
            str(REPO_ROOT / "sprinkler2" / "generate_draft_layout.py"),
            "--input-json",
            str(working_detected),
            "--output-dir",
            str(base_dir),
            "--preview-floor-label",
            str(floor_name),
            *layout_args(settings),
        ]
    )
    return base_dir / "layout_result.json"


def layout_summary(layout_path: Path) -> dict[str, Any]:
    data = json.loads(layout_path.read_text(encoding="utf-8"))
    geoms = data.get("geometries") or {}
    heads = []
    for idx, head in enumerate(geoms.get("sprinkler_heads") or [], start=1):
        heads.append(
            {
                "id": f"H{idx:03d}",
                "x": float(head.get("x", 0.0)),
                "y": float(head.get("y", 0.0)),
                "source": head.get("source") or "generated",
            }
        )
    branches = []
    for idx, line in enumerate((geoms.get("branch_lines") or []) + (geoms.get("secondary_branch_lines") or []), start=1):
        if len(line) >= 2:
            branches.append({"id": f"B{idx:03d}", "points": [[float(x), float(y)] for x, y in line]})
    trunks = []
    for idx, segment in enumerate(geoms.get("trunk_segments") or [], start=1):
        start = segment.get("start") or []
        end = segment.get("end") or []
        if len(start) >= 2 and len(end) >= 2:
            trunks.append(
                {
                    "id": f"T{idx:03d}",
                    "start": [float(start[0]), float(start[1])],
                    "end": [float(end[0]), float(end[1])],
                    "kind": segment.get("kind") or "trunk",
                    "diameter": segment.get("diameter") or "DN100",
                }
            )
    return {
        "protected_floor_area": geoms.get("protected_floor_area"),
        "exclusion_area": geoms.get("exclusion_area"),
        "trunk_segments": trunks,
        "branch_lines": branches,
        "sprinkler_heads": heads,
        "counts": data.get("counts") or {},
        "parameters": data.get("parameters") or {},
    }


def refresh_floor_layout(project: dict[str, Any], floor: dict[str, Any]) -> None:
    layout_path = project_relative_path(project, floor.get("latest_layout_json"))
    if layout_path and layout_path.exists():
        floor["layout"] = layout_summary(layout_path)


def trunk_from_layout(layout_path: Path, diameter: str) -> dict[str, Any]:
    data = json.loads(layout_path.read_text(encoding="utf-8"))
    geoms = data.get("geometries") or {}
    trunk_line = geoms.get("main_trunk_line") or geoms.get("trunk_line") or []
    segments: list[dict[str, Any]] = []
    for idx, segment in enumerate(geoms.get("trunk_segments") or [], start=1):
        start = segment.get("start") or []
        end = segment.get("end") or []
        if len(start) >= 2 and len(end) >= 2:
            segments.append(
                {
                    "id": f"T{idx:03d}",
                    "start": [float(start[0]), float(start[1])],
                    "end": [float(end[0]), float(end[1])],
                    "kind": segment.get("kind") or "main",
                    "diameter": segment.get("diameter") or diameter,
                }
            )
    if not segments and len(trunk_line) >= 2:
        for idx, (start, end) in enumerate(zip(trunk_line, trunk_line[1:]), start=1):
            if len(start) >= 2 and len(end) >= 2:
                segments.append(
                    {
                        "id": f"T{idx:03d}",
                        "start": [float(start[0]), float(start[1])],
                        "end": [float(end[0]), float(end[1])],
                        "kind": "main",
                        "diameter": diameter,
                    }
                )
    if len(trunk_line) < 2 and segments:
        trunk_line = [segments[0]["start"], segments[-1]["end"]]
    if len(trunk_line) < 2:
        raise RuntimeError("V1 trunk generation did not produce a usable trunk line.")
    return {
        "main_trunk_line": [[float(point[0]), float(point[1])] for point in trunk_line],
        "segments": segments,
    }


def trunk_point_key(point: list[float] | tuple[float, float]) -> str:
    return f"{float(point[0]):.3f},{float(point[1]):.3f}"


def normalize_trunk_segments(raw_segments: list[dict[str, Any]] | None, diameter: str = "DN100") -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for idx, segment in enumerate(raw_segments or [], start=1):
        start = segment.get("start") or []
        end = segment.get("end") or []
        if len(start) < 2 or len(end) < 2:
            continue
        start_xy = [float(start[0]), float(start[1])]
        end_xy = [float(end[0]), float(end[1])]
        if ((start_xy[0] - end_xy[0]) ** 2 + (start_xy[1] - end_xy[1]) ** 2) ** 0.5 <= 0.001:
            continue
        segments.append(
            {
                "id": str(segment.get("id") or f"T{idx:03d}"),
                "start": start_xy,
                "end": end_xy,
                "kind": str(segment.get("kind") or "main"),
                "diameter": str(segment.get("diameter") or diameter),
            }
        )
    return segments


def ordered_trunk_points(
    segments: list[dict[str, Any]],
    fallback_start: list[float] | None = None,
    fallback_end: list[float] | None = None,
) -> list[list[float]]:
    if not segments:
        points: list[list[float]] = []
        if fallback_start and len(fallback_start) >= 2:
            points.append([float(fallback_start[0]), float(fallback_start[1])])
        if fallback_end and len(fallback_end) >= 2:
            points.append([float(fallback_end[0]), float(fallback_end[1])])
        return points
    points_by_key: dict[str, list[float]] = {}
    adjacency: dict[str, set[str]] = {}
    for segment in segments:
        start = [float(segment["start"][0]), float(segment["start"][1])]
        end = [float(segment["end"][0]), float(segment["end"][1])]
        start_key = trunk_point_key(start)
        end_key = trunk_point_key(end)
        points_by_key[start_key] = start
        points_by_key[end_key] = end
        adjacency.setdefault(start_key, set()).add(end_key)
        adjacency.setdefault(end_key, set()).add(start_key)
    end_keys = [key for key, neighbors in adjacency.items() if len(neighbors) == 1]
    if fallback_start and len(fallback_start) >= 2 and end_keys:
        start_key = min(
            end_keys,
            key=lambda key: (points_by_key[key][0] - float(fallback_start[0])) ** 2 + (points_by_key[key][1] - float(fallback_start[1])) ** 2,
        )
    else:
        start_key = end_keys[0] if end_keys else trunk_point_key(segments[0]["start"])
    line: list[list[float]] = []
    seen_edges: set[tuple[str, str]] = set()
    current = start_key
    previous = ""
    while current:
        if current in points_by_key:
            line.append(points_by_key[current])
        next_keys = []
        for neighbor in adjacency.get(current, set()):
            edge = tuple(sorted((current, neighbor)))
            if neighbor != previous and edge not in seen_edges:
                next_keys.append(neighbor)
        if not next_keys:
            break
        next_key = next_keys[0]
        seen_edges.add(tuple(sorted((current, next_key))))
        previous = current
        current = next_key
    if len(line) < 2 and fallback_end and len(fallback_end) >= 2:
        line.append([float(fallback_end[0]), float(fallback_end[1])])
    return line


def apply_trunk_segments_to_layout(layout_path: Path, override: dict[str, Any] | None, diameter: str) -> bool:
    if not override:
        return False
    segments = normalize_trunk_segments(override.get("segments") or [], diameter)
    if not segments:
        return False
    line = ordered_trunk_points(segments, override.get("start"), override.get("end"))
    if len(line) < 2:
        return False
    data = json.loads(layout_path.read_text(encoding="utf-8"))
    geoms = data.setdefault("geometries", {})
    geoms["trunk_segments"] = segments
    geoms["trunk_line"] = line
    geoms["main_trunk_line"] = line
    data.setdefault("meta", {})["app_trunk_override"] = {"mode": "segments", "segments": len(segments)}
    layout_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return True


def is_user_trunk_override(override: dict[str, Any] | None) -> bool:
    return bool(override and str(override.get("source") or "").lower() == "user")


def generate_floor_trunk(
    project: dict[str, Any],
    floor_id: str,
    settings: dict[str, Any] | None = None,
    log: Callable[[str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
    runner_ref: dict[str, CommandRunner | None] | None = None,
) -> dict[str, Any]:
    floor = find_floor(project, floor_id)
    if floor is None:
        raise ValueError(f"Unknown floor id: {floor_id}")
    detected_path = project_relative_path(project, floor.get("detected_json"))
    if not detected_path or not detected_path.exists():
        raise RuntimeError("Detect the floor before generating a trunk.")

    pdir = project_dir(project["id"])
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = pdir / "trunks" / floor_id / stamp
    detected_copy_dir = out_root / "detected"
    base_dir = out_root / "layout_base"
    v1_dir = out_root / "layout_v1"
    detected_copy_dir.mkdir(parents=True, exist_ok=True)
    working_detected = detected_copy_dir / "detected_geometry.json"
    shutil.copy2(detected_path, working_detected)

    merged_settings = {**DEFAULT_SETTINGS, **project.get("settings", {}), **(settings or {})}
    runner = CommandRunner(log or (lambda msg: None), should_cancel)
    if runner_ref is not None:
        runner_ref["runner"] = runner
    base_layout = build_base_layout(
        runner,
        working_detected,
        base_dir,
        str(floor["name"]),
        merged_settings,
        allow_approved_seed=True,
    )
    runner.run(
        [
            sys.executable,
            str(REPO_ROOT / "sprinkler2" / "auto_main_trunk.py"),
            "--detected-json",
            str(working_detected),
            "--layout-json",
            str(base_layout),
            "--output-dir",
            str(v1_dir),
            "--preview-floor-label",
            str(floor["name"]),
        ]
    )

    diameter = str(merged_settings.get("main_diameter") or DEFAULT_SETTINGS["main_diameter"])
    trunk = trunk_from_layout(v1_dir / "layout_result.json", diameter)
    line = trunk["main_trunk_line"]
    start = line[0]
    end = line[-1]
    geometry = geometry_summary(detected_path)
    floor["geometry"] = geometry
    floor["trunk_override"] = {"start": start, "end": end, "source": "auto_v1"}
    floor["trunk"] = {
        "segments": trunk["segments"],
        "main_trunk_line": line,
        "source": "v1_main_trunk",
        "candidate_layout_json": relative_project_path(project["id"], v1_dir / "layout_result.json"),
    }
    floor["status"] = "trunk_ready"
    trunk_json = out_root / "trunk.json"
    trunk_json.write_text(json.dumps(floor["trunk"], indent=2, ensure_ascii=False), encoding="utf-8")
    floor["trunk_json"] = relative_project_path(project["id"], trunk_json)
    floor["trunk_preview_url"] = file_url(project["id"], v1_dir / "layout_preview.png")
    if log:
        log(f"Generated v1 main trunk for {floor['name']}: {start} -> {end}")
    save_project(project)
    return floor


def apply_trunk_override(detected_path: Path, override: dict[str, Any] | None) -> None:
    if not override:
        return
    segments = normalize_trunk_segments(override.get("segments") or [], str(override.get("diameter") or "DN100"))
    line = ordered_trunk_points(segments, override.get("start"), override.get("end")) if segments else []
    if len(line) < 2:
        start = override.get("start")
        end = override.get("end")
        if not start or not end:
            return
        line = [[float(start[0]), float(start[1])], [float(end[0]), float(end[1])]]
    if len(line) < 2:
        return
    data = json.loads(detected_path.read_text(encoding="utf-8"))
    data["suggested_trunk_line"] = line
    data["trunk_endpoints"] = {"start_xy": data["suggested_trunk_line"][0], "end_xy": data["suggested_trunk_line"][1], "source": "app_override"}
    if segments:
        data["suggested_trunk_segments"] = segments
    data.setdefault("detection_meta", {})["trunk_source"] = "app_override"
    detected_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def detect_floor(project: dict[str, Any], floor_id: str, log: Callable[[str], None] | None = None) -> dict[str, Any]:
    pdir = project_dir(project["id"])
    floor = find_floor(project, floor_id)
    if floor is None:
        raise ValueError(f"Unknown floor id: {floor_id}")
    out_dir = pdir / "detected" / floor_id
    out_dir.mkdir(parents=True, exist_ok=True)
    log_fn = log or (lambda msg: None)
    runner = CommandRunner(log_fn)
    runner.run(
        [
            sys.executable,
            str(REPO_ROOT / "sprinkler2" / "detect_parking_geometry.py"),
            "--ifc",
            str(pdir / project["source_ifc"]),
            "--storey",
            str(floor["name"]),
            "--output-dir",
            str(out_dir),
            "--preview-floor-label",
            f"{floor['name']} | IfcBuildingStorey id={floor.get('ifc_id')}",
        ]
    )
    detected_json = out_dir / "detected_geometry.json"
    preview = out_dir / "detected_geometry_preview.png"
    floor["status"] = "detected"
    floor["detected_json"] = relative_project_path(project["id"], detected_json)
    floor["preview_url"] = file_url(project["id"], preview)
    floor["geometry"] = geometry_summary(detected_json)
    save_project(project)
    return floor


def layout_args(settings: dict[str, Any]) -> list[str]:
    args = [
        "--branch-spacing",
        str(settings.get("branch_spacing", DEFAULT_SETTINGS["branch_spacing"])),
        "--head-spacing",
        str(settings.get("head_spacing", DEFAULT_SETTINGS["head_spacing"])),
        "--column-clearance",
        str(settings.get("column_clearance", DEFAULT_SETTINGS["column_clearance"])),
        "--stair-clearance",
        str(settings.get("stair_clearance", DEFAULT_SETTINGS["stair_clearance"])),
        "--wall-clearance",
        str(settings.get("wall_clearance", DEFAULT_SETTINGS["wall_clearance"])),
        "--min-obstacle-clearance",
        str(settings.get("min_obstacle_clearance", DEFAULT_SETTINGS["min_obstacle_clearance"])),
        "--trunk-diameter",
        str(settings.get("main_diameter", DEFAULT_SETTINGS["main_diameter"])),
        "--branch-diameter",
        str(settings.get("branch_diameter", DEFAULT_SETTINGS["branch_diameter"])),
        "--layout-model",
        str(settings.get("layout_model", DEFAULT_SETTINGS["layout_model"])),
        "--routing-model",
        str(settings.get("routing_model", DEFAULT_SETTINGS["routing_model"])),
        "--demand-step",
        str(settings.get("demand_step", DEFAULT_SETTINGS["demand_step"])),
        "--target-coverage",
        str(settings.get("target_coverage", DEFAULT_SETTINGS["target_coverage"])),
        "--cpsat-time-limit",
        str(settings.get("cpsat_time_limit", DEFAULT_SETTINGS["cpsat_time_limit"])),
        "--cpsat-max-demand",
        str(settings.get("cpsat_max_demand", DEFAULT_SETTINGS["cpsat_max_demand"])),
        "--cpsat-min-head-spacing",
        str(settings.get("cpsat_min_head_spacing", DEFAULT_SETTINGS["cpsat_min_head_spacing"])),
    ]
    if settings.get("allow_secondary_branches"):
        args.append("--allow-secondary-branches")
    return args


def add_floor_metadata(
    items: list[dict[str, Any]],
    floor: dict[str, Any],
    prefix: str,
    *,
    level_elevation_m: float | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    level_name = str(floor.get("name") or floor["id"])
    source_elevation = float(floor.get("elevation_m") or 0.0)
    elevation = source_elevation if level_elevation_m is None else float(level_elevation_m)
    for idx, item in enumerate(items, start=1):
        copied = dict(item)
        copied["id"] = f"{prefix}{floor['index']:02d}_{idx:04d}"
        copied["floor_id"] = floor["id"]
        copied["storey_name"] = level_name
        copied["level_name"] = level_name
        copied["level_elevation_m"] = elevation
        copied["ifc_elevation_m"] = source_elevation
        out.append(copied)
    return out


def nearest_point_on_segments(point_xy: list[float], segments: list[dict[str, Any]]) -> list[float] | None:
    point = Point(float(point_xy[0]), float(point_xy[1]))
    best_point: Point | None = None
    best_distance: float | None = None
    for segment in segments:
        start = segment.get("start") or []
        end = segment.get("end") or []
        if len(start) < 2 or len(end) < 2:
            continue
        line = LineString([(float(start[0]), float(start[1])), (float(end[0]), float(end[1]))])
        if line.length <= 1e-9:
            continue
        candidate = line.interpolate(line.project(point))
        distance = point.distance(candidate)
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_point = candidate
    if best_point is None:
        return None
    return [float(best_point.x), float(best_point.y)]


def apply_layout_edits_to_file(layout_path: Path, edits: dict[str, Any] | None) -> dict[str, Any]:
    data = json.loads(layout_path.read_text(encoding="utf-8"))
    if not edits:
        return data
    geoms = data.setdefault("geometries", {})
    heads = geoms.setdefault("sprinkler_heads", [])
    head_edits = edits.get("heads") or {}
    trunk_segments = geoms.get("trunk_segments") or []
    branch_lines = geoms.setdefault("branch_lines", [])
    changed_heads = 0
    for head_id, point in head_edits.items():
        try:
            index = int(str(head_id).lstrip("Hh")) - 1
        except ValueError:
            continue
        if index < 0 or index >= len(heads) or len(point) < 2:
            continue
        old = [float(heads[index].get("x", 0.0)), float(heads[index].get("y", 0.0))]
        new = [float(point[0]), float(point[1])]
        heads[index]["x"] = new[0]
        heads[index]["y"] = new[1]
        heads[index]["source"] = "user_override"
        changed_heads += 1
        replaced_endpoint = False
        for line in branch_lines:
            for coord in line:
                if len(coord) >= 2 and ((float(coord[0]) - old[0]) ** 2 + (float(coord[1]) - old[1]) ** 2) ** 0.5 <= 0.45:
                    coord[0] = new[0]
                    coord[1] = new[1]
                    replaced_endpoint = True
        if not replaced_endpoint:
            trunk_point = nearest_point_on_segments(new, trunk_segments)
            if trunk_point:
                branch_lines.append([trunk_point, new])
    data.setdefault("meta", {})["user_overrides"] = {"heads": changed_heads}
    layout_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return data


def patch_floor_layout_edits(project: dict[str, Any], floor_id: str, edits: dict[str, Any]) -> dict[str, Any]:
    floor = find_floor(project, floor_id)
    if floor is None:
        raise ValueError(f"Unknown floor id: {floor_id}")
    layout_path = project_relative_path(project, floor.get("latest_layout_json"))
    if not layout_path or not layout_path.exists():
        raise RuntimeError("Generate sprinklers before editing sprinkler heads.")
    current = floor.get("layout_edits") or {}
    current_heads = dict(current.get("heads") or {})
    for key, value in (edits.get("heads") or {}).items():
        current_heads[str(key)] = [float(value[0]), float(value[1])]
    current["heads"] = current_heads
    floor["layout_edits"] = current
    apply_layout_edits_to_file(layout_path, current)
    floor["layout"] = layout_summary(layout_path)
    floor["status"] = "edited"
    save_project(project)
    return floor


def generate_floor_sprinklers(
    project: dict[str, Any],
    floor_id: str,
    settings: dict[str, Any] | None = None,
    log: Callable[[str], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
    runner_ref: dict[str, CommandRunner | None] | None = None,
) -> dict[str, Any]:
    floor = find_floor(project, floor_id)
    if floor is None:
        raise ValueError(f"Unknown floor id: {floor_id}")
    detected_path = project_relative_path(project, floor.get("detected_json"))
    if not detected_path or not detected_path.exists():
        raise RuntimeError("Detect the floor before generating sprinklers.")
    if not floor.get("trunk_override"):
        generate_floor_trunk(project, floor_id, settings=settings, log=log, should_cancel=should_cancel, runner_ref=runner_ref)
        floor = find_floor(project, floor_id)
        assert floor is not None

    pdir = project_dir(project["id"])
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = pdir / "layouts" / floor_id / stamp
    detected_copy_dir = out_root / "detected"
    base_dir = out_root / "layout_base"
    v1_dir = out_root / "layout_v1"
    detected_copy_dir.mkdir(parents=True, exist_ok=True)
    working_detected = detected_copy_dir / "detected_geometry.json"
    shutil.copy2(detected_path, working_detected)
    preview = project_relative_path(project, floor.get("preview_url", "").replace(f"/files/{project['id']}/", ""))
    if preview and preview.exists():
        shutil.copy2(preview, detected_copy_dir / preview.name)
    preserve_existing_trunk = is_user_trunk_override(floor.get("trunk_override"))
    if preserve_existing_trunk:
        apply_trunk_override(working_detected, floor.get("trunk_override"))

    merged_settings = {**DEFAULT_SETTINGS, **project.get("settings", {}), **(settings or {})}
    runner = CommandRunner(log or (lambda msg: None), should_cancel)
    if runner_ref is not None:
        runner_ref["runner"] = runner
    base_layout = build_base_layout(
        runner,
        working_detected,
        base_dir,
        str(floor["name"]),
        merged_settings,
        allow_approved_seed=not preserve_existing_trunk,
    )
    diameter = str(merged_settings.get("main_diameter") or DEFAULT_SETTINGS["main_diameter"])
    if preserve_existing_trunk:
        apply_trunk_segments_to_layout(base_layout, floor.get("trunk_override"), diameter)
    trunk_cmd = [
        sys.executable,
        str(REPO_ROOT / "sprinkler2" / "auto_main_trunk.py"),
        "--detected-json",
        str(working_detected),
        "--layout-json",
        str(base_layout),
        "--output-dir",
        str(v1_dir),
        "--preview-floor-label",
        str(floor["name"]),
    ]
    if preserve_existing_trunk:
        trunk_cmd.append("--preserve-existing-trunk")
    runner.run(trunk_cmd)
    score_json = v1_dir / "score_report.json"
    runner.run(
        [
            sys.executable,
            str(REPO_ROOT / "sprinkler2" / "score_layout.py"),
            "--detected-json",
            str(working_detected),
            "--layout-json",
            str(v1_dir / "layout_result.json"),
            "--out-json",
            str(score_json),
        ]
    )
    layout_json = v1_dir / "layout_result.json"
    if floor.get("layout_edits"):
        apply_layout_edits_to_file(layout_json, floor.get("layout_edits"))
    floor["status"] = "layout_ready"
    floor["latest_detected_json"] = relative_project_path(project["id"], working_detected)
    floor["latest_layout_json"] = relative_project_path(project["id"], layout_json)
    floor["latest_layout_preview_url"] = file_url(project["id"], v1_dir / "layout_preview.png")
    floor["latest_score_json"] = relative_project_path(project["id"], score_json)
    floor["latest_score"] = json.loads(score_json.read_text(encoding="utf-8")) if score_json.exists() else None
    floor["layout"] = layout_summary(layout_json)
    floor["settings_snapshot"] = merged_settings
    save_project(project)
    return floor


def build_combined_handoff(project: dict[str, Any], run: dict[str, Any], floor_results: list[dict[str, Any]]) -> dict[str, Any]:
    pdir = project_dir(project["id"])
    out_dir = run_dir(project["id"], run["id"]) / "revit"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_pipes: list[dict[str, Any]] = []
    all_heads: list[dict[str, Any]] = []
    all_context: list[dict[str, Any]] = []
    levels: list[dict[str, Any]] = []
    flatten_single_floor = len(floor_results) == 1

    for result in floor_results:
        floor = result["floor"]
        layout_path = Path(result["layout_json"])
        detected_path = Path(result["detected_json"])
        layout = json.loads(layout_path.read_text(encoding="utf-8"))
        detected = json.loads(detected_path.read_text(encoding="utf-8"))
        source_z_m = float(floor.get("elevation_m") or 0.0)
        placement_z_m = 0.0 if flatten_single_floor else source_z_m
        context, context_items = build_context(layout, detected, z_m=placement_z_m)
        pipes = build_pipe_runs(layout, z_m=placement_z_m, min_length_m=0.02)
        heads = build_heads(layout, z_m=placement_z_m)
        all_pipes.extend(add_floor_metadata(pipes, floor, "P", level_elevation_m=placement_z_m))
        all_heads.extend(add_floor_metadata(heads, floor, "H", level_elevation_m=placement_z_m))
        all_context.extend(add_floor_metadata(context_items, floor, "C", level_elevation_m=placement_z_m))
        levels.append(
            {
                "id": floor["id"],
                "name": floor["name"],
                "storey_name": floor["name"],
                "elevation_m": placement_z_m,
                "ifc_elevation_m": source_z_m,
                "placement_mode": "single_floor_v1_template_level" if flatten_single_floor else "ifc_storey_elevation",
            }
        )
        result["building_context"] = context

    handoff = {
        "schema": "sprinkler_layout.revit_handoff.multifloor.v1",
        "source": {
            "project_id": project["id"],
            "run_id": run["id"],
            "source_ifc": str((pdir / project["source_ifc"]).resolve()),
            "generated_utc": datetime.now(timezone.utc).isoformat(),
        },
        "units": {
            "coordinates": "m",
            "pipe_diameter": "mm",
            "revit_internal_length": "ft",
            "meters_to_revit_feet": FT_PER_M,
        },
        "levels": levels,
        "pipe_runs": all_pipes,
        "sprinkler_heads": all_heads,
        "context_elements": all_context,
        "counts": {
            "levels": len(levels),
            "context_elements": len(all_context),
            "pipe_runs": len(all_pipes),
            "sprinkler_heads": len(all_heads),
            "trunk_pipe_runs": sum(1 for p in all_pipes if p.get("kind") == "trunk"),
            "branch_pipe_runs": sum(1 for p in all_pipes if p.get("kind") != "trunk"),
        },
        "warnings": [
            "Architectural context imports as DirectShape references.",
            "Draft geometric sprinkler layout only; engineer review and hydraulic calculations are required.",
            *(
                [
                    "Single-floor RVT export is placed on the Revit template level for hosted sprinkler-family visibility; IFC storey elevation is retained as metadata.",
                ]
                if flatten_single_floor
                else []
            ),
        ],
    }

    json_path = out_dir / "revit_sprinkler_layout.json"
    json_path.write_text(json.dumps(handoff, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csvs(out_dir, all_pipes, all_heads)
    write_context_csv(out_dir, all_context)
    shutil.copy2(pdir / project["source_ifc"], out_dir / "source_model.ifc")
    return {"handoff_json": json_path, "out_dir": out_dir, "counts": handoff["counts"]}


def write_manifest_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["floor_id", "storey_name", "elevation_m", "detected_json", "layout_json", "score_json", "preview_png"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def export_revit_from_project(
    project: dict[str, Any],
    run: dict[str, Any],
    selected_floor_ids: list[str],
    log: Callable[[str], None],
    should_cancel: Callable[[], bool],
    runner_ref: dict[str, CommandRunner | None] | None = None,
) -> dict[str, Any]:
    floor_results: list[dict[str, Any]] = []
    for floor_id in selected_floor_ids:
        floor = find_floor(project, floor_id)
        if floor is None:
            raise ValueError(f"Unknown floor id: {floor_id}")
        layout_path = project_relative_path(project, floor.get("latest_layout_json"))
        detected_path = project_relative_path(project, floor.get("latest_detected_json") or floor.get("detected_json"))
        score_path = project_relative_path(project, floor.get("latest_score_json"))
        if not layout_path or not layout_path.exists():
            raise RuntimeError(f"Generate sprinklers before exporting {floor.get('name')}.")
        if not detected_path or not detected_path.exists():
            raise RuntimeError(f"Missing detected geometry for {floor.get('name')}.")
        floor_results.append(
            {
                "floor": floor,
                "floor_id": floor["id"],
                "storey_name": floor["name"],
                "elevation_m": floor.get("elevation_m"),
                "detected_json": str(detected_path),
                "layout_json": str(layout_path),
                "score_json": str(score_path) if score_path and score_path.exists() else "",
                "preview_png": str(project_relative_path(project, floor.get("latest_layout_preview_url", "").replace(f"/files/{project['id']}/", "")) or ""),
            }
        )
    if not floor_results:
        raise RuntimeError("Select at least one layout-ready floor before exporting.")

    command_runner = CommandRunner(log, should_cancel)
    if runner_ref is not None:
        runner_ref["runner"] = command_runner
    handoff = build_combined_handoff(project, run, floor_results)
    rvt_output = handoff["out_dir"] / "sprinkler_layout_combined.rvt"
    settings = {**DEFAULT_SETTINGS, **project.get("settings", {}), **run.get("settings", {})}
    env = os.environ.copy()
    env["SPRINKLER_REVIT_LAYOUT_JSON"] = str(handoff["handoff_json"])
    env["SPRINKLER_REVIT_OUTPUT_RVT"] = str(rvt_output)
    env["SPRINKLER_REVIT_TEMPLATE"] = str(settings.get("revit_template") or DEFAULT_SETTINGS["revit_template"])
    env["SPRINKLER_REVIT_LOG"] = str(handoff["out_dir"] / "make_rvt.log")
    pyrevit = Path(str(settings.get("pyrevit_exe") or PYREVIT_EXE))
    if not pyrevit.exists():
        raise FileNotFoundError(f"pyrevit.exe not found: {pyrevit}")
    command_runner.run(
        [
            str(pyrevit),
            "run",
            str(REPO_ROOT / "sprinkler2" / "make_rvt_pyrevit.py"),
            f"--revit={settings.get('revit_year', '2027')}",
            "--purge",
        ],
        env=env,
    )
    revit_log = handoff["out_dir"] / "make_rvt.log"
    text = revit_log.read_text(encoding="utf-8", errors="replace") if revit_log.exists() else ""
    if "Traceback" in text or "Failures: 0" not in text or "Saved RVT:" not in text:
        raise RuntimeError(f"Revit export did not complete cleanly. Check {revit_log}")

    rdir = run_dir(project["id"], run["id"])
    manifest_rows = [
        {
            "floor_id": item["floor_id"],
            "storey_name": item["storey_name"],
            "elevation_m": item["elevation_m"],
            "detected_json": item["detected_json"],
            "layout_json": item["layout_json"],
            "score_json": item["score_json"],
            "preview_png": item["preview_png"],
        }
        for item in floor_results
    ]
    write_manifest_csv(rdir / "floor_manifest.csv", manifest_rows)
    artifacts = [
        {"label": "Combined RVT", "path": relative_project_path(project["id"], rvt_output), "url": file_url(project["id"], rvt_output)},
        {"label": "Revit handoff JSON", "path": relative_project_path(project["id"], handoff["handoff_json"]), "url": file_url(project["id"], handoff["handoff_json"])},
        {"label": "Pipe schedule CSV", "path": relative_project_path(project["id"], handoff["out_dir"] / "pipe_runs.csv"), "url": file_url(project["id"], handoff["out_dir"] / "pipe_runs.csv")},
        {"label": "Head schedule CSV", "path": relative_project_path(project["id"], handoff["out_dir"] / "sprinkler_heads.csv"), "url": file_url(project["id"], handoff["out_dir"] / "sprinkler_heads.csv")},
        {"label": "Context schedule CSV", "path": relative_project_path(project["id"], handoff["out_dir"] / "context_elements.csv"), "url": file_url(project["id"], handoff["out_dir"] / "context_elements.csv")},
        {"label": "Revit log", "path": relative_project_path(project["id"], revit_log), "url": file_url(project["id"], revit_log)},
        {"label": "Floor manifest", "path": relative_project_path(project["id"], rdir / "floor_manifest.csv"), "url": file_url(project["id"], rdir / "floor_manifest.csv")},
    ]
    for item in floor_results:
        item["floor"]["status"] = "exported"
    run["status"] = "complete"
    run["finished_at"] = now_iso()
    run["artifacts"] = artifacts
    run["counts"] = handoff["counts"]
    run["floor_results"] = [
        {
            "floor_id": item["floor_id"],
            "storey_name": item["storey_name"],
            "preview_url": file_url(project["id"], Path(item["preview_png"])) if item["preview_png"] else "",
            "score": json.loads(Path(item["score_json"]).read_text(encoding="utf-8")) if item["score_json"] else {},
        }
        for item in floor_results
    ]
    save_run(run)
    save_project(project)
    return run


def run_project_pipeline(
    project: dict[str, Any],
    run: dict[str, Any],
    log: Callable[[str], None],
    should_cancel: Callable[[], bool],
    runner_ref: dict[str, CommandRunner | None] | None = None,
) -> dict[str, Any]:
    pdir = project_dir(project["id"])
    rdir = run_dir(project["id"], run["id"])
    selected = set(run.get("selected_floor_ids") or [])
    settings = {**DEFAULT_SETTINGS, **project.get("settings", {}), **run.get("settings", {})}
    command_runner = CommandRunner(log, should_cancel)
    if runner_ref is not None:
        runner_ref["runner"] = command_runner
    floor_results: list[dict[str, Any]] = []
    try:
        for floor in project.get("storeys", []):
            if floor.get("id") not in selected:
                continue
            log(f"=== Floor {floor['index']}: {floor['name']} ===")
            fdir = floor_folder(rdir, floor)
            detected_dir = fdir / "detected"
            base_dir = fdir / "layout_base"
            v1_dir = fdir / "layout_v1"
            detected_dir.mkdir(parents=True, exist_ok=True)
            command_runner.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "sprinkler2" / "detect_parking_geometry.py"),
                    "--ifc",
                    str(pdir / project["source_ifc"]),
                    "--storey",
                    str(floor["name"]),
                    "--output-dir",
                    str(detected_dir),
                    "--preview-floor-label",
                    f"{floor['name']} | IfcBuildingStorey id={floor.get('ifc_id')}",
                ]
            )
            detected_json = detected_dir / "detected_geometry.json"
            preserve_existing_trunk = is_user_trunk_override(floor.get("trunk_override"))
            if preserve_existing_trunk:
                apply_trunk_override(detected_json, floor.get("trunk_override"))
            base_layout = build_base_layout(
                command_runner,
                detected_json,
                base_dir,
                str(floor["name"]),
                settings,
                allow_approved_seed=not preserve_existing_trunk,
            )
            diameter = str(settings.get("main_diameter") or DEFAULT_SETTINGS["main_diameter"])
            if preserve_existing_trunk:
                apply_trunk_segments_to_layout(base_layout, floor.get("trunk_override"), diameter)
            trunk_cmd = [
                sys.executable,
                str(REPO_ROOT / "sprinkler2" / "auto_main_trunk.py"),
                "--detected-json",
                str(detected_json),
                "--layout-json",
                str(base_layout),
                "--output-dir",
                str(v1_dir),
                "--preview-floor-label",
                str(floor["name"]),
            ]
            if preserve_existing_trunk:
                trunk_cmd.append("--preserve-existing-trunk")
            command_runner.run(trunk_cmd)
            score_json = v1_dir / "score_report.json"
            command_runner.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "sprinkler2" / "score_layout.py"),
                    "--detected-json",
                    str(detected_json),
                    "--layout-json",
                    str(v1_dir / "layout_result.json"),
                    "--out-json",
                    str(score_json),
                ]
            )
            floor["status"] = "layout_ready"
            floor["detected_json"] = relative_project_path(project["id"], detected_json)
            floor["preview_url"] = file_url(project["id"], detected_dir / "detected_geometry_preview.png")
            floor["geometry"] = geometry_summary(detected_json)
            floor["latest_layout_preview_url"] = file_url(project["id"], v1_dir / "layout_preview.png")
            floor["latest_score"] = json.loads(score_json.read_text(encoding="utf-8")) if score_json.exists() else None
            floor_results.append(
                {
                    "floor": floor,
                    "floor_id": floor["id"],
                    "storey_name": floor["name"],
                    "elevation_m": floor.get("elevation_m"),
                    "detected_json": str(detected_json),
                    "layout_json": str(v1_dir / "layout_result.json"),
                    "score_json": str(score_json),
                    "preview_png": str(v1_dir / "layout_preview.png"),
                }
            )

        if not floor_results:
            raise RuntimeError("No selected floors to run.")

        handoff = build_combined_handoff(project, run, floor_results)
        rvt_output = handoff["out_dir"] / "sprinkler_layout_combined.rvt"
        env = os.environ.copy()
        env["SPRINKLER_REVIT_LAYOUT_JSON"] = str(handoff["handoff_json"])
        env["SPRINKLER_REVIT_OUTPUT_RVT"] = str(rvt_output)
        env["SPRINKLER_REVIT_TEMPLATE"] = str(settings.get("revit_template") or DEFAULT_SETTINGS["revit_template"])
        env["SPRINKLER_REVIT_LOG"] = str(handoff["out_dir"] / "make_rvt.log")
        pyrevit = Path(str(settings.get("pyrevit_exe") or PYREVIT_EXE))
        if not pyrevit.exists():
            raise FileNotFoundError(f"pyrevit.exe not found: {pyrevit}")
        command_runner.run(
            [
                str(pyrevit),
                "run",
                str(REPO_ROOT / "sprinkler2" / "make_rvt_pyrevit.py"),
                f"--revit={settings.get('revit_year', '2027')}",
                "--purge",
            ],
            env=env,
        )
        revit_log = handoff["out_dir"] / "make_rvt.log"
        text = revit_log.read_text(encoding="utf-8", errors="replace") if revit_log.exists() else ""
        if "Traceback" in text or "Failures: 0" not in text or "Saved RVT:" not in text:
            raise RuntimeError(f"Revit export did not complete cleanly. Check {revit_log}")

        manifest_rows = [
            {
                "floor_id": item["floor_id"],
                "storey_name": item["storey_name"],
                "elevation_m": item["elevation_m"],
                "detected_json": item["detected_json"],
                "layout_json": item["layout_json"],
                "score_json": item["score_json"],
                "preview_png": item["preview_png"],
            }
            for item in floor_results
        ]
        write_manifest_csv(rdir / "floor_manifest.csv", manifest_rows)
        artifacts = [
            {"label": "Combined RVT", "path": relative_project_path(project["id"], rvt_output), "url": file_url(project["id"], rvt_output)},
            {"label": "Revit handoff JSON", "path": relative_project_path(project["id"], handoff["handoff_json"]), "url": file_url(project["id"], handoff["handoff_json"])},
            {"label": "Pipe schedule CSV", "path": relative_project_path(project["id"], handoff["out_dir"] / "pipe_runs.csv"), "url": file_url(project["id"], handoff["out_dir"] / "pipe_runs.csv")},
            {"label": "Head schedule CSV", "path": relative_project_path(project["id"], handoff["out_dir"] / "sprinkler_heads.csv"), "url": file_url(project["id"], handoff["out_dir"] / "sprinkler_heads.csv")},
            {"label": "Context schedule CSV", "path": relative_project_path(project["id"], handoff["out_dir"] / "context_elements.csv"), "url": file_url(project["id"], handoff["out_dir"] / "context_elements.csv")},
            {"label": "Revit log", "path": relative_project_path(project["id"], revit_log), "url": file_url(project["id"], revit_log)},
            {"label": "Floor manifest", "path": relative_project_path(project["id"], rdir / "floor_manifest.csv"), "url": file_url(project["id"], rdir / "floor_manifest.csv")},
        ]
        run["status"] = "complete"
        run["finished_at"] = now_iso()
        run["artifacts"] = artifacts
        run["counts"] = handoff["counts"]
        run["floor_results"] = [
            {
                "floor_id": item["floor_id"],
                "storey_name": item["storey_name"],
                "preview_url": file_url(project["id"], Path(item["preview_png"])),
                "score": json.loads(Path(item["score_json"]).read_text(encoding="utf-8")),
            }
            for item in floor_results
        ]
        save_run(run)
        save_project(project)
        return run
    except PipelineCancelled:
        run["status"] = "cancelled"
        run["finished_at"] = now_iso()
        save_run(run)
        log("Run cancelled.")
        return run
    except Exception as exc:
        run["status"] = "failed"
        run["error"] = str(exc)
        run["finished_at"] = now_iso()
        save_run(run)
        log(f"ERROR: {exc}")
        raise
    finally:
        if runner_ref is not None:
            runner_ref["runner"] = None
