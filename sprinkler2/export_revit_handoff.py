from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FT_PER_M = 3.280839895013123
MM_PER_M = 1000.0
DEFAULT_MAIN_DIAMETER = "DN100"
DEFAULT_BRANCH_DIAMETER = "DN65"


PYREVIT_SCRIPT = r'''# pyRevit script: import sprinkler layout JSON as editable Revit elements.
# Copy this extension folder into pyRevit, reload pyRevit, then run the button.
from __future__ import print_function

import json
import math
import os

from pyrevit import DB, forms, revit
from System.Collections.Generic import List


SAVE_RVT_AFTER_IMPORT = False
FT_PER_M = 3.280839895013123
FT_PER_MM = FT_PER_M / 1000.0


doc = revit.doc
uidoc = revit.uidoc


def m_to_ft(value):
    return float(value) * FT_PER_M


def mm_to_ft(value):
    return float(value) * FT_PER_MM


def xyz_from_m(coords):
    return DB.XYZ(m_to_ft(coords[0]), m_to_ft(coords[1]), m_to_ft(coords[2]))


def collect_first(cls):
    items = list(DB.FilteredElementCollector(doc).OfClass(cls).WhereElementIsElementType())
    return items[0] if items else None


def element_id_value(element_id):
    if hasattr(element_id, "IntegerValue"):
        return element_id.IntegerValue
    if hasattr(element_id, "Value"):
        return element_id.Value
    return int(element_id)


def built_in_category_id(category):
    try:
        return DB.ElementId(category)
    except Exception:
        return DB.ElementId(int(category))


def safe_element_name(element):
    if element is None:
        return "<missing>"
    try:
        return element.Name
    except Exception:
        pass
    try:
        return DB.Element.Name.GetValue(element)
    except Exception:
        return "<unnamed>"


def active_level():
    view = doc.ActiveView
    level = getattr(view, "GenLevel", None)
    if level:
        return level
    levels = list(DB.FilteredElementCollector(doc).OfClass(DB.Level))
    if levels:
        return sorted(levels, key=lambda x: x.Elevation)[0]
    return None


def first_pipe_type():
    try:
        return collect_first(DB.Plumbing.PipeType)
    except Exception:
        return None


def first_piping_system_type():
    try:
        systems = list(
            DB.FilteredElementCollector(doc)
            .OfClass(DB.Plumbing.PipingSystemType)
            .WhereElementIsElementType()
        )
        for system in systems:
            name = safe_element_name(system).lower()
            if "fire" in name or "sprink" in name:
                return system
        return systems[0] if systems else None
    except Exception:
        return None


def sprinkler_symbol():
    symbols = list(
        DB.FilteredElementCollector(doc)
        .OfClass(DB.FamilySymbol)
        .WhereElementIsElementType()
    )
    sprinkler_cat = int(DB.BuiltInCategory.OST_Sprinklers)
    for symbol in symbols:
        cat = getattr(symbol, "Category", None)
        if cat and element_id_value(cat.Id) == sprinkler_cat:
            return symbol
    return None


def set_comment(element, text):
    try:
        param = element.get_Parameter(DB.BuiltInParameter.ALL_MODEL_INSTANCE_COMMENTS)
        if param and not param.IsReadOnly:
            param.Set(text)
    except Exception:
        pass


def set_pipe_diameter(pipe, diameter_mm):
    try:
        param = pipe.get_Parameter(DB.BuiltInParameter.RBS_PIPE_DIAMETER_PARAM)
        if param and not param.IsReadOnly:
            param.Set(mm_to_ft(diameter_mm))
    except Exception:
        pass


def create_pipe(pipe_run, pipe_type, system_type, level):
    start = xyz_from_m(pipe_run["start_m"])
    end = xyz_from_m(pipe_run["end_m"])
    pipe = DB.Plumbing.Pipe.Create(
        doc,
        system_type.Id,
        pipe_type.Id,
        level.Id,
        start,
        end,
    )
    set_pipe_diameter(pipe, pipe_run.get("diameter_mm", 65.0))
    set_comment(pipe, "SprinklerLayout {0} {1}".format(pipe_run.get("id"), pipe_run.get("kind")))
    return pipe


def make_curve_loop(points_m, z_ft):
    loop = DB.CurveLoop()
    pts = []
    for coords in points_m:
        if len(coords) < 2:
            continue
        pts.append(DB.XYZ(m_to_ft(coords[0]), m_to_ft(coords[1]), z_ft))
    if len(pts) > 1 and pts[0].DistanceTo(pts[-1]) < 0.001:
        pts = pts[:-1]
    if len(pts) < 3:
        return None
    for i in range(len(pts)):
        if pts[i].DistanceTo(pts[(i + 1) % len(pts)]) > 0.001:
            loop.Append(DB.Line.CreateBound(pts[i], pts[(i + 1) % len(pts)]))
    return loop


def solids_from_footprint(footprint, z_base_m, height_m):
    solids = []
    if not footprint:
        return solids
    z_ft = m_to_ft(z_base_m)
    height_ft = max(mm_to_ft(20.0), m_to_ft(height_m))
    geom_type = footprint.get("type")
    polygons = []
    if geom_type == "Polygon":
        polygons = [footprint]
    elif geom_type == "MultiPolygon":
        polygons = [p for p in footprint.get("parts", []) if p and p.get("type") == "Polygon"]
    for poly in polygons:
        loops = List[DB.CurveLoop]()
        exterior = make_curve_loop(poly.get("exterior", []), z_ft)
        if exterior is None:
            continue
        loops.Add(exterior)
        for hole in poly.get("holes", []) or []:
            inner = make_curve_loop(hole, z_ft)
            if inner is not None:
                loops.Add(inner)
        try:
            solids.append(DB.GeometryCreationUtilities.CreateExtrusionGeometry(loops, DB.XYZ.BasisZ, height_ft))
        except Exception:
            pass
    return solids


def create_context_element(item):
    solids = solids_from_footprint(
        item.get("footprint"),
        item.get("z_base_m", 0.0),
        item.get("height_m", 0.05),
    )
    if not solids:
        return None
    ds = DB.DirectShape.CreateElement(doc, built_in_category_id(DB.BuiltInCategory.OST_GenericModel))
    ds.ApplicationId = "SprinklerLayoutContext"
    ds.ApplicationDataId = item.get("id", "context")
    shape = List[DB.GeometryObject]()
    for solid in solids:
        shape.Add(solid)
    ds.SetShape(shape)
    set_comment(
        ds,
        "SprinklerLayout context {0} {1} {2}".format(
            item.get("id"),
            item.get("ifc_class"),
            item.get("name") or "",
        ),
    )
    return ds


def make_circle_loop(center, radius_ft, z_ft):
    loop = DB.CurveLoop()
    pts = []
    for i in range(16):
        a = 2.0 * 3.141592653589793 * float(i) / 16.0
        pts.append(DB.XYZ(center.X + radius_ft * math.cos(a), center.Y + radius_ft * math.sin(a), z_ft))
    for i in range(len(pts)):
        loop.Append(DB.Line.CreateBound(pts[i], pts[(i + 1) % len(pts)]))
    return loop


def create_directshape_head(head):
    # Fallback when no sprinkler family is loaded. This is visible and movable,
    # but a real sprinkler family should be loaded and re-imported for final MEP editing.
    center = xyz_from_m(head["point_m"])
    radius = mm_to_ft(120.0)
    height = mm_to_ft(60.0)
    z0 = center.Z - height / 2.0
    loop = make_circle_loop(center, radius, z0)
    loops = List[DB.CurveLoop]()
    loops.Add(loop)
    solid = DB.GeometryCreationUtilities.CreateExtrusionGeometry(loops, DB.XYZ.BasisZ, height)
    ds = DB.DirectShape.CreateElement(doc, built_in_category_id(DB.BuiltInCategory.OST_GenericModel))
    ds.ApplicationId = "SprinklerLayoutImporter"
    ds.ApplicationDataId = head.get("id", "head")
    shape = List[DB.GeometryObject]()
    shape.Add(solid)
    ds.SetShape(shape)
    set_comment(ds, "SprinklerLayout head marker {0}; load a sprinkler family for native heads.".format(head.get("id")))
    return ds


def create_head(head, symbol, level):
    point = xyz_from_m(head["point_m"])
    if symbol is None:
        return create_directshape_head(head)
    if not symbol.IsActive:
        symbol.Activate()
        doc.Regenerate()
    try:
        inst = doc.Create.NewFamilyInstance(point, symbol, level, DB.Structure.StructuralType.NonStructural)
        set_comment(inst, "SprinklerLayout head {0}".format(head.get("id")))
        return inst
    except Exception as exc:
        marker = create_directshape_head(head)
        set_comment(
            marker,
            "SprinklerLayout head marker {0}; sprinkler family placement failed: {1}".format(head.get("id"), exc),
        )
        return marker


def save_as_rvt(layout_path):
    out_path = forms.save_file(
        file_ext="rvt",
        init_dir=os.path.dirname(layout_path),
        default_name="sprinkler_layout_from_export.rvt",
    )
    if not out_path:
        return None
    opts = DB.SaveAsOptions()
    opts.OverwriteExistingFile = True
    doc.SaveAs(out_path, opts)
    return out_path


button_dir = os.path.dirname(__file__)
panel_dir = os.path.dirname(button_dir)
tab_dir = os.path.dirname(panel_dir)
extension_dir = os.path.dirname(tab_dir)
default_json = os.path.join(extension_dir, "revit_sprinkler_layout.json")
layout_path = forms.pick_file(file_ext="json", init_dir=os.path.dirname(default_json), default_name=os.path.basename(default_json))
if not layout_path:
    forms.alert("No layout JSON selected.", exitscript=True)

with open(layout_path, "r") as f:
    layout = json.load(f)

level = active_level()
if level is None:
    forms.alert("No Revit Level found in this model.", exitscript=True)

pipe_type = first_pipe_type()
system_type = first_piping_system_type()
if pipe_type is None or system_type is None:
    forms.alert("No PipeType or MEPSystemType found. Open an MEP template or load pipe settings first.", exitscript=True)

symbol = sprinkler_symbol()
pipe_runs = layout.get("pipe_runs", [])
heads = layout.get("sprinkler_heads", [])
context_elements = layout.get("context_elements", [])

created_pipes = 0
created_heads = 0
created_context = 0
failed = []

with revit.Transaction("Import sprinkler layout"):
    for item in context_elements:
        try:
            created = create_context_element(item)
            if created is not None:
                created_context += 1
        except Exception as exc:
            failed.append("context {0}: {1}".format(item.get("id"), exc))

    for pipe_run in pipe_runs:
        try:
            create_pipe(pipe_run, pipe_type, system_type, level)
            created_pipes += 1
        except Exception as exc:
            failed.append("pipe {0}: {1}".format(pipe_run.get("id"), exc))

    for head in heads:
        try:
            create_head(head, symbol, level)
            created_heads += 1
        except Exception as exc:
            failed.append("head {0}: {1}".format(head.get("id"), exc))

rvt_path = None
if SAVE_RVT_AFTER_IMPORT:
    try:
        rvt_path = save_as_rvt(layout_path)
    except Exception as exc:
        failed.append("save rvt: {0}".format(exc))

msg = "Created {0} context references, {1} pipe runs, and {2} sprinkler heads/markers.".format(
    created_context,
    created_pipes,
    created_heads,
)
if rvt_path:
    msg += "\n\nSaved RVT:\n{0}".format(rvt_path)
if symbol is None:
    msg += "\n\nNo OST_Sprinklers family symbol was loaded, so heads were created as Generic Model markers."
if failed:
    msg += "\n\nFailures:\n" + "\n".join(failed[:20])
    if len(failed) > 20:
        msg += "\n... {0} more".format(len(failed) - 20)
forms.alert(msg, title="Sprinkler Layout Import")
'''


def parse_dn_mm(label: str | None, default_mm: float) -> float:
    if not label:
        return default_mm
    text = str(label).upper().strip()
    if text.startswith("DN"):
        text = text[2:]
    digits = "".join(ch for ch in text if ch.isdigit() or ch == ".")
    if not digits:
        return default_mm
    try:
        return float(digits)
    except ValueError:
        return default_mm


def point3(raw: Any, z_m: float) -> list[float] | None:
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        return None
    try:
        return [float(raw[0]), float(raw[1]), float(raw[2]) if len(raw) > 2 else float(z_m)]
    except (TypeError, ValueError):
        return None


def segment_length_m(start: list[float], end: list[float]) -> float:
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(start, end)))


def segment_key(start: list[float], end: list[float], kind: str, diameter_mm: float) -> tuple[Any, ...]:
    a = tuple(round(v, 5) for v in start)
    b = tuple(round(v, 5) for v in end)
    ordered = tuple(sorted([a, b]))
    return (ordered, kind, round(float(diameter_mm), 3))


def append_segment(
    pipe_runs: list[dict[str, Any]],
    seen: set[tuple[Any, ...]],
    start: list[float],
    end: list[float],
    *,
    kind: str,
    diameter_label: str,
    source: str,
    min_length_m: float,
) -> None:
    diameter_mm = parse_dn_mm(diameter_label, 100.0 if kind == "trunk" else 65.0)
    if segment_length_m(start, end) < min_length_m:
        return
    key = segment_key(start, end, kind, diameter_mm)
    if key in seen:
        return
    seen.add(key)
    pipe_id = "P{0:04d}".format(len(pipe_runs) + 1)
    pipe_runs.append(
        {
            "id": pipe_id,
            "kind": kind,
            "diameter_label": diameter_label,
            "diameter_mm": diameter_mm,
            "start_m": start,
            "end_m": end,
            "length_m": round(segment_length_m(start, end), 4),
            "source": source,
        }
    )


def append_polyline(
    pipe_runs: list[dict[str, Any]],
    seen: set[tuple[Any, ...]],
    coords: Any,
    *,
    z_m: float,
    kind: str,
    diameter_label: str,
    source: str,
    min_length_m: float,
) -> None:
    if not isinstance(coords, list) or len(coords) < 2:
        return
    points = [point3(pt, z_m) for pt in coords]
    clean_points = [pt for pt in points if pt is not None]
    for idx in range(len(clean_points) - 1):
        append_segment(
            pipe_runs,
            seen,
            clean_points[idx],
            clean_points[idx + 1],
            kind=kind,
            diameter_label=diameter_label,
            source="{0}[segment:{1}]".format(source, idx),
            min_length_m=min_length_m,
        )


def build_pipe_runs(layout: dict[str, Any], z_m: float, min_length_m: float) -> list[dict[str, Any]]:
    geoms = layout.get("geometries") or {}
    pipe_runs: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()

    trunk_segments = geoms.get("trunk_segments") or []
    if trunk_segments:
        for idx, item in enumerate(trunk_segments):
            start = point3(item.get("start"), z_m)
            end = point3(item.get("end"), z_m)
            if start is None or end is None:
                continue
            append_segment(
                pipe_runs,
                seen,
                start,
                end,
                kind=str(item.get("kind") or "trunk"),
                diameter_label=str(item.get("diameter") or DEFAULT_MAIN_DIAMETER),
                source="geometries.trunk_segments[{0}]".format(idx),
                min_length_m=min_length_m,
            )
    else:
        trunk = geoms.get("main_trunk_line") or geoms.get("trunk_line") or []
        append_polyline(
            pipe_runs,
            seen,
            trunk,
            z_m=z_m,
            kind="trunk",
            diameter_label=DEFAULT_MAIN_DIAMETER,
            source="geometries.main_trunk_line",
            min_length_m=min_length_m,
        )

    for idx, coords in enumerate(geoms.get("main_trunk_connectors") or []):
        append_polyline(
            pipe_runs,
            seen,
            coords,
            z_m=z_m,
            kind="branch",
            diameter_label=DEFAULT_BRANCH_DIAMETER,
            source="geometries.main_trunk_connectors[{0}]".format(idx),
            min_length_m=min_length_m,
        )

    for idx, coords in enumerate(geoms.get("branch_lines") or []):
        append_polyline(
            pipe_runs,
            seen,
            coords,
            z_m=z_m,
            kind="branch",
            diameter_label=DEFAULT_BRANCH_DIAMETER,
            source="geometries.branch_lines[{0}]".format(idx),
            min_length_m=min_length_m,
        )

    return pipe_runs


def build_heads(layout: dict[str, Any], z_m: float) -> list[dict[str, Any]]:
    geoms = layout.get("geometries") or {}
    heads = []
    for idx, item in enumerate(geoms.get("sprinkler_heads") or []):
        try:
            x = float(item["x"])
            y = float(item["y"])
        except (KeyError, TypeError, ValueError):
            continue
        heads.append(
            {
                "id": "H{0:04d}".format(idx + 1),
                "point_m": [x, y, float(z_m)],
                "family_category": "OST_Sprinklers",
                "source": "geometries.sprinkler_heads[{0}]".format(idx),
            }
        )
    return heads


def normalize_path_from_layout(layout_json: Path, raw_path: str | None) -> Path | None:
    if not raw_path:
        return None
    p = Path(raw_path)
    candidates = [p]
    if not p.is_absolute():
        candidates.append(Path.cwd() / p)
        candidates.append(layout_json.parent / p)
        candidates.append(layout_json.parent.parent / p)
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved.exists():
            return resolved
    return None


def resolve_detected_json(layout: dict[str, Any], layout_json: Path, explicit_path: Path | None) -> Path | None:
    if explicit_path is not None:
        return explicit_path if explicit_path.exists() else None
    meta = layout.get("meta") or {}
    raw = meta.get("input_detected_json")
    found = normalize_path_from_layout(layout_json, raw)
    if found is not None:
        return found
    fallback = Path("outputs/output/detected_geometry.json")
    return fallback if fallback.exists() else None


def source_element_list(detected: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = detected.get(key)
    return value if isinstance(value, list) else []


def context_element(
    item: dict[str, Any],
    *,
    prefix: str,
    idx: int,
    category: str,
    z_base_m: float,
    height_m: float,
) -> dict[str, Any] | None:
    footprint = item.get("footprint")
    if not isinstance(footprint, dict):
        return None
    return {
        "id": "{0}{1:04d}".format(prefix, idx + 1),
        "category": category,
        "ifc_id": item.get("ifc_id"),
        "global_id": item.get("global_id"),
        "ifc_class": item.get("ifc_class"),
        "name": item.get("name"),
        "storey": item.get("storey"),
        "z_base_m": float(z_base_m),
        "height_m": float(height_m),
        "footprint": footprint,
        "source": category,
    }


def build_context(layout: dict[str, Any], detected: dict[str, Any] | None, z_m: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    geoms = layout.get("geometries") or {}
    context = {
        "protected_floor_area": geoms.get("protected_floor_area"),
        "valid_coverage_area": geoms.get("valid_coverage_area"),
        "exclusion_area": geoms.get("exclusion_area"),
    }
    elements: list[dict[str, Any]] = []
    if detected is None:
        return context, elements

    context.update(
        {
            "input_ifc": detected.get("input_ifc"),
            "target_storey": detected.get("target_storey"),
            "detected_counts_on_target_storey": detected.get("detected_counts_on_target_storey"),
            "overall_floor_bounds": detected.get("overall_floor_bounds"),
            "candidate_axes": detected.get("candidate_axes"),
            "trunk_endpoints": detected.get("trunk_endpoints"),
        }
    )

    specs = [
        ("slab_footprints", "SLAB", "slab", z_m - 0.05, 0.05),
        ("columns", "COL", "column", z_m, 2.7),
        ("walls_standard_case", "WALL", "wall", z_m, 2.7),
        ("walls_generic", "GWALL", "generic_wall", z_m, 2.7),
        ("stairs", "STAIR", "stair", z_m, 0.3),
        ("spaces", "SPACE", "space", z_m, 0.02),
    ]
    for key, prefix, category, z_base, height in specs:
        for idx, item in enumerate(source_element_list(detected, key)):
            element = context_element(
                item,
                prefix=prefix,
                idx=idx,
                category=category,
                z_base_m=z_base,
                height_m=height,
            )
            if element is not None:
                elements.append(element)
    return context, elements


def write_csvs(out_dir: Path, pipe_runs: list[dict[str, Any]], heads: list[dict[str, Any]]) -> None:
    with (out_dir / "pipe_runs.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "kind",
                "diameter_label",
                "diameter_mm",
                "start_x_m",
                "start_y_m",
                "start_z_m",
                "end_x_m",
                "end_y_m",
                "end_z_m",
                "length_m",
                "source",
            ],
        )
        writer.writeheader()
        for pipe in pipe_runs:
            writer.writerow(
                {
                    "id": pipe["id"],
                    "kind": pipe["kind"],
                    "diameter_label": pipe["diameter_label"],
                    "diameter_mm": pipe["diameter_mm"],
                    "start_x_m": pipe["start_m"][0],
                    "start_y_m": pipe["start_m"][1],
                    "start_z_m": pipe["start_m"][2],
                    "end_x_m": pipe["end_m"][0],
                    "end_y_m": pipe["end_m"][1],
                    "end_z_m": pipe["end_m"][2],
                    "length_m": pipe["length_m"],
                    "source": pipe["source"],
                }
            )

    with (out_dir / "sprinkler_heads.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "x_m", "y_m", "z_m", "family_category", "source"])
        writer.writeheader()
        for head in heads:
            writer.writerow(
                {
                    "id": head["id"],
                    "x_m": head["point_m"][0],
                    "y_m": head["point_m"][1],
                    "z_m": head["point_m"][2],
                    "family_category": head["family_category"],
                    "source": head["source"],
                }
            )


def write_context_csv(out_dir: Path, context_elements: list[dict[str, Any]]) -> None:
    with (out_dir / "context_elements.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "category",
                "ifc_id",
                "global_id",
                "ifc_class",
                "name",
                "storey",
                "z_base_m",
                "height_m",
                "area_m2",
            ],
        )
        writer.writeheader()
        for item in context_elements:
            footprint = item.get("footprint") or {}
            writer.writerow(
                {
                    "id": item.get("id"),
                    "category": item.get("category"),
                    "ifc_id": item.get("ifc_id"),
                    "global_id": item.get("global_id"),
                    "ifc_class": item.get("ifc_class"),
                    "name": item.get("name"),
                    "storey": item.get("storey"),
                    "z_base_m": item.get("z_base_m"),
                    "height_m": item.get("height_m"),
                    "area_m2": footprint.get("area"),
                }
            )


def write_pyrevit_extension(out_dir: Path) -> Path:
    import_script_path = (
        out_dir
        / "SprinklerLayout.extension"
        / "Sprinkler Layout.tab"
        / "Import.panel"
        / "Import Layout.pushbutton"
        / "script.py"
    )
    import_script_path.parent.mkdir(parents=True, exist_ok=True)
    import_script_path.write_text(PYREVIT_SCRIPT, encoding="utf-8")

    save_script_path = (
        out_dir
        / "SprinklerLayout.extension"
        / "Sprinkler Layout.tab"
        / "Import.panel"
        / "Import and Save RVT.pushbutton"
        / "script.py"
    )
    save_script_path.parent.mkdir(parents=True, exist_ok=True)
    save_script_path.write_text(
        PYREVIT_SCRIPT.replace("SAVE_RVT_AFTER_IMPORT = False", "SAVE_RVT_AFTER_IMPORT = True"),
        encoding="utf-8",
    )
    return import_script_path


def write_readme(out_dir: Path, source_layout: Path, pipe_count: int, head_count: int, context_count: int) -> None:
    readme = """# Revit handoff

This folder was generated from `{source}`.

## Contents

- `revit_sprinkler_layout.json`: normalized meter-based geometry for the Revit importer.
- `pipe_runs.csv`: pipe schedule for review.
- `sprinkler_heads.csv`: head schedule for review.
- `context_elements.csv`: exported source IFC slab/column/wall/stair reference elements.
- `source_model.ifc`: copy of the source IFC when the file is available.
- `SprinklerLayout.extension`: self-contained pyRevit extension with its own copy of `revit_sprinkler_layout.json`.

## Import in Revit

1. Install pyRevit if it is not already available.
2. Copy `SprinklerLayout.extension` to a pyRevit extensions folder.
3. In Revit, open an MEP project/template with pipe types and a piping system type.
4. Load a fire sprinkler family before import if you want native sprinkler family instances.
5. Reload pyRevit, run `Sprinkler Layout > Import > Import Layout`, and use the default selected JSON inside the extension.

To create a real `.rvt`, open a new Revit MEP project/template and run `Sprinkler Layout > Import > Import and Save RVT`. That command imports the same geometry and then calls Revit API `Document.SaveAs()` so Revit writes the `.rvt` file.

The importer creates source-building context as Generic Model DirectShape references, then creates pipes on the active view level. Coordinates are passed as meters from the layout JSON and converted to Revit internal feet. If no `OST_Sprinklers` family is loaded, sprinkler heads are created as Generic Model markers so their positions are still visible; load a real sprinkler family and re-run for editable MEP heads.

Generated counts: {context_count} context references, {pipe_count} pipe runs, {head_count} sprinkler heads.

Engineering note: pipes are native Revit pipe elements, but the architectural context is reference geometry. Link/import `source_model.ifc` in Revit when the engineer needs the original building model. This is a draft geometric layout, not a hydraulic calculation or authority-approved design.
""".format(
        source=source_layout,
        pipe_count=pipe_count,
        head_count=head_count,
        context_count=context_count,
    )
    (out_dir / "README.md").write_text(readme, encoding="utf-8")


def export(
    layout_json: Path,
    out_dir: Path,
    z_m: float,
    min_pipe_length_m: float,
    detected_json: Path | None = None,
) -> dict[str, Any]:
    layout = json.loads(layout_json.read_text(encoding="utf-8"))
    pipe_runs = build_pipe_runs(layout, z_m=z_m, min_length_m=min_pipe_length_m)
    heads = build_heads(layout, z_m=z_m)
    detected_path = resolve_detected_json(layout, layout_json, detected_json)
    detected = json.loads(detected_path.read_text(encoding="utf-8")) if detected_path is not None else None
    context, context_elements = build_context(layout, detected, z_m=z_m)

    source_ifc = normalize_path_from_layout(layout_json, context.get("input_ifc")) if context.get("input_ifc") else None
    packaged_ifc = None
    if source_ifc is not None and source_ifc.is_file():
        packaged_ifc = out_dir / "source_model.ifc"

    handoff = {
        "schema": "sprinkler_layout.revit_handoff.v2",
        "source": {
            "layout_result_json": str(layout_json),
            "detected_geometry_json": str(detected_path) if detected_path is not None else None,
            "source_ifc": str(source_ifc) if source_ifc is not None else context.get("input_ifc"),
            "packaged_source_ifc": str(packaged_ifc.name) if packaged_ifc is not None else None,
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "layout_status": (layout.get("meta") or {}).get("status"),
            "layout_note": (layout.get("meta") or {}).get("note"),
        },
        "units": {
            "coordinates": "m",
            "pipe_diameter": "mm",
            "revit_internal_length": "ft",
            "meters_to_revit_feet": FT_PER_M,
        },
        "coordinate_system": {
            "source": "layout_result_json.plan_xy",
            "mapping": "x_m_to_revit_x_ft, y_m_to_revit_y_ft, z_m_to_revit_z_ft",
            "default_z_m": float(z_m),
        },
        "defaults": {
            "main_pipe_diameter_label": DEFAULT_MAIN_DIAMETER,
            "branch_pipe_diameter_label": DEFAULT_BRANCH_DIAMETER,
            "sprinkler_family_category": "OST_Sprinklers",
        },
        "building_context": context,
        "context_elements": context_elements,
        "pipe_runs": pipe_runs,
        "sprinkler_heads": heads,
        "counts": {
            "context_elements": len(context_elements),
            "slabs": sum(1 for item in context_elements if item.get("category") == "slab"),
            "columns": sum(1 for item in context_elements if item.get("category") == "column"),
            "walls": sum(1 for item in context_elements if "wall" in str(item.get("category"))),
            "stairs": sum(1 for item in context_elements if item.get("category") == "stair"),
            "spaces": sum(1 for item in context_elements if item.get("category") == "space"),
            "pipe_runs": len(pipe_runs),
            "sprinkler_heads": len(heads),
            "trunk_pipe_runs": sum(1 for p in pipe_runs if p.get("kind") == "trunk"),
            "branch_pipe_runs": sum(1 for p in pipe_runs if p.get("kind") != "trunk"),
        },
        "warnings": [
            "Revit API import must run inside Revit; this package generates the JSON and pyRevit importer only.",
            "Architectural context imports as DirectShape reference geometry; link/import the included source IFC for full building-model fidelity.",
            "Load a real OST_Sprinklers family before import for editable sprinkler family instances.",
            "Draft geometric layout only; engineer review and hydraulic calculations are required.",
        ],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    handoff_json = json.dumps(handoff, indent=2)
    (out_dir / "revit_sprinkler_layout.json").write_text(handoff_json, encoding="utf-8")
    write_csvs(out_dir, pipe_runs, heads)
    write_context_csv(out_dir, context_elements)
    if packaged_ifc is not None:
        shutil.copy2(source_ifc, packaged_ifc)
    script_path = write_pyrevit_extension(out_dir)
    (out_dir / "SprinklerLayout.extension" / "revit_sprinkler_layout.json").write_text(handoff_json, encoding="utf-8")
    write_readme(out_dir, layout_json, len(pipe_runs), len(heads), len(context_elements))
    return {
        "out_dir": str(out_dir),
        "json": str(out_dir / "revit_sprinkler_layout.json"),
        "pipe_csv": str(out_dir / "pipe_runs.csv"),
        "heads_csv": str(out_dir / "sprinkler_heads.csv"),
        "context_csv": str(out_dir / "context_elements.csv"),
        "source_ifc": str(packaged_ifc) if packaged_ifc is not None else None,
        "pyrevit_script": str(script_path),
        "context_elements": len(context_elements),
        "pipe_runs": len(pipe_runs),
        "sprinkler_heads": len(heads),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a Revit-ready handoff package from a sprinkler layout_result.json.")
    parser.add_argument("--layout-json", default="outputs/output_v1_main_trunk_connected/layout_result.json")
    parser.add_argument("--output-dir", default="outputs/output_v1_revit_ready")
    parser.add_argument("--detected-json", default=None, help="Optional detected_geometry.json path for architectural context.")
    parser.add_argument("--z-m", type=float, default=0.0, help="Pipe/head elevation in meters for the Revit import JSON.")
    parser.add_argument("--min-pipe-length-m", type=float, default=0.02, help="Skip tiny pipe segments below this length.")
    args = parser.parse_args()

    result = export(
        layout_json=Path(args.layout_json),
        out_dir=Path(args.output_dir),
        z_m=args.z_m,
        min_pipe_length_m=args.min_pipe_length_m,
        detected_json=Path(args.detected_json) if args.detected_json else None,
    )
    print("Revit handoff package complete.")
    for key, value in result.items():
        print("- {0}: {1}".format(key, value))


if __name__ == "__main__":
    main()
