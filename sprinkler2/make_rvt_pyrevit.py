from __future__ import print_function

import json
import math
import os
import traceback

from Autodesk.Revit import DB
from System.Collections.Generic import List


LAYOUT_JSON = os.environ.get("SPRINKLER_REVIT_LAYOUT_JSON")
OUTPUT_RVT = os.environ.get("SPRINKLER_REVIT_OUTPUT_RVT")
TEMPLATE_PATH = os.environ.get("SPRINKLER_REVIT_TEMPLATE", r"F:\autodesk\RVT 2027\Templates\English\Systems-Default_Metric.rte")
LOG_PATH = os.environ.get("SPRINKLER_REVIT_LOG", r"F:\unified\archicad\projects\make_rvt.log")

FT_PER_M = 3.280839895013123
FT_PER_MM = FT_PER_M / 1000.0


def log(message):
    parent = os.path.dirname(LOG_PATH)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)
    with open(LOG_PATH, "a") as f:
        f.write(str(message) + "\n")
    print(message)


def m_to_ft(value):
    return float(value) * FT_PER_M


def mm_to_ft(value):
    return float(value) * FT_PER_MM


def xyz_from_m(coords):
    return DB.XYZ(m_to_ft(coords[0]), m_to_ft(coords[1]), m_to_ft(coords[2]))


def level_relative_xy_from_m(coords):
    return DB.XYZ(m_to_ft(coords[0]), m_to_ft(coords[1]), 0.0)


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


def first_or_none(items):
    for item in items:
        return item
    return None


def first_pipe_type(doc):
    try:
        return first_or_none(DB.FilteredElementCollector(doc).OfClass(DB.Plumbing.PipeType).WhereElementIsElementType())
    except Exception:
        return None


def first_piping_system_type(doc):
    try:
        systems = list(DB.FilteredElementCollector(doc).OfClass(DB.Plumbing.PipingSystemType).WhereElementIsElementType())
        for system in systems:
            name = safe_element_name(system).lower()
            if "fire" in name or "sprink" in name or "wet" in name:
                return system
        return systems[0] if systems else None
    except Exception:
        return None


def levels_by_name(doc):
    return {safe_element_name(level): level for level in DB.FilteredElementCollector(doc).OfClass(DB.Level)}


def get_or_create_level(doc, name, elevation_m):
    existing = levels_by_name(doc)
    if name in existing:
        return existing[name]
    close = sorted(existing.values(), key=lambda level: abs(level.Elevation - m_to_ft(elevation_m)))
    if close and abs(close[0].Elevation - m_to_ft(elevation_m)) < mm_to_ft(5.0):
        return close[0]
    level = DB.Level.Create(doc, m_to_ft(elevation_m))
    try:
        level.Name = name
    except Exception:
        pass
    return level


def sprinkler_symbol(doc):
    symbols = list(DB.FilteredElementCollector(doc).OfClass(DB.FamilySymbol).WhereElementIsElementType())
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


def set_element_id_parameter(element, built_in_name, element_id):
    try:
        built_in = getattr(DB.BuiltInParameter, built_in_name)
    except Exception:
        return False
    try:
        param = element.get_Parameter(built_in)
        if param and not param.IsReadOnly:
            param.Set(element_id)
            return True
    except Exception:
        pass
    return False


def set_head_level_metadata(element, level):
    if level is None:
        return
    for name in (
        "FAMILY_LEVEL_PARAM",
        "INSTANCE_SCHEDULE_ONLY_LEVEL_PARAM",
        "INSTANCE_REFERENCE_LEVEL_PARAM",
    ):
        set_element_id_parameter(element, name, level.Id)


def item_level(item, level_cache, default_level):
    name = item.get("level_name") or item.get("storey_name")
    if name and name in level_cache:
        return level_cache[name]
    return default_level


def create_pipe(doc, pipe_run, pipe_type, system_type, level):
    pipe = DB.Plumbing.Pipe.Create(doc, system_type.Id, pipe_type.Id, level.Id, xyz_from_m(pipe_run["start_m"]), xyz_from_m(pipe_run["end_m"]))
    set_pipe_diameter(pipe, pipe_run.get("diameter_mm", 65.0))
    set_comment(pipe, "SprinklerLayout {0} {1} {2}".format(pipe_run.get("id"), pipe_run.get("kind"), pipe_run.get("storey_name") or ""))
    return pipe


def xyz_delta(start, end):
    return DB.XYZ(end.X - start.X, end.Y - start.Y, end.Z - start.Z)


def xyz_length(vector):
    return math.sqrt(vector.X * vector.X + vector.Y * vector.Y + vector.Z * vector.Z)


def sketch_plane_for_line(doc, start, end):
    delta = xyz_delta(start, end)
    if abs(start.Z - end.Z) < 0.001:
        normal = DB.XYZ.BasisZ
    else:
        normal = delta.CrossProduct(DB.XYZ.BasisZ)
        if xyz_length(normal) < 0.001:
            normal = DB.XYZ.BasisX
        else:
            normal = normal.Normalize()
    plane = DB.Plane.CreateByNormalAndOrigin(normal, start)
    return DB.SketchPlane.Create(doc, plane)


def create_pipe_centerline(doc, pipe_run):
    start = xyz_from_m(pipe_run["start_m"])
    end = xyz_from_m(pipe_run["end_m"])
    if start.DistanceTo(end) < mm_to_ft(20.0):
        return None
    sketch_plane = sketch_plane_for_line(doc, start, end)
    curve = DB.Line.CreateBound(start, end)
    model_curve = doc.Create.NewModelCurve(curve, sketch_plane)
    set_comment(
        model_curve,
        "SprinklerLayout visible pipe centerline {0} {1} {2}; native pipe is also created.".format(
            pipe_run.get("id"),
            pipe_run.get("kind"),
            pipe_run.get("storey_name") or "",
        ),
    )
    return model_curve


def make_curve_loop(points_m, z_ft):
    loop = DB.CurveLoop()
    pts = []
    for coords in points_m:
        if len(coords) >= 2:
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
    if geom_type == "Polygon":
        polygons = [footprint]
    elif geom_type == "MultiPolygon":
        polygons = [p for p in footprint.get("parts", []) if p and p.get("type") == "Polygon"]
    else:
        polygons = []
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


def create_context_element(doc, item):
    solids = solids_from_footprint(item.get("footprint"), item.get("z_base_m", 0.0), item.get("height_m", 0.05))
    if not solids:
        return None
    ds = DB.DirectShape.CreateElement(doc, built_in_category_id(DB.BuiltInCategory.OST_GenericModel))
    ds.ApplicationId = "SprinklerLayoutContext"
    ds.ApplicationDataId = item.get("id", "context")
    shape = List[DB.GeometryObject]()
    for solid in solids:
        shape.Add(solid)
    ds.SetShape(shape)
    set_comment(ds, "SprinklerLayout context {0} {1} {2}".format(item.get("id"), item.get("ifc_class"), item.get("storey_name") or ""))
    return ds


def make_circle_loop(center, radius_ft, z_ft):
    loop = DB.CurveLoop()
    pts = []
    for i in range(16):
        a = 2.0 * math.pi * float(i) / 16.0
        pts.append(DB.XYZ(center.X + radius_ft * math.cos(a), center.Y + radius_ft * math.sin(a), z_ft))
    for i in range(len(pts)):
        loop.Append(DB.Line.CreateBound(pts[i], pts[(i + 1) % len(pts)]))
    return loop


def create_directshape_head(doc, head):
    center = xyz_from_m(head["point_m"])
    height = mm_to_ft(60.0)
    loop = make_circle_loop(center, mm_to_ft(180.0), center.Z - height / 2.0)
    loops = List[DB.CurveLoop]()
    loops.Add(loop)
    solid = DB.GeometryCreationUtilities.CreateExtrusionGeometry(loops, DB.XYZ.BasisZ, height)
    ds = DB.DirectShape.CreateElement(doc, built_in_category_id(DB.BuiltInCategory.OST_GenericModel))
    ds.ApplicationId = "SprinklerLayoutImporter"
    ds.ApplicationDataId = head.get("id", "head")
    shape = List[DB.GeometryObject]()
    shape.Add(solid)
    ds.SetShape(shape)
    set_comment(ds, "SprinklerLayout head marker {0} {1}".format(head.get("id"), head.get("storey_name") or ""))
    return ds


def create_head(doc, head, symbol, level):
    if symbol is None:
        return create_directshape_head(doc, head), "marker"
    if not symbol.IsActive:
        symbol.Activate()
        doc.Regenerate()
    try:
        # This sidewall sprinkler family is level-based: the Z passed to
        # NewFamilyInstance becomes an elevation offset from the supplied level.
        # Use zero offset so the symbol belongs to and displays on that floor.
        inst = doc.Create.NewFamilyInstance(level_relative_xy_from_m(head["point_m"]), symbol, level, DB.Structure.StructuralType.NonStructural)
        set_head_level_metadata(inst, level)
        set_comment(inst, "SprinklerLayout native head {0} {1}".format(head.get("id"), head.get("storey_name") or ""))
        return inst, "native"
    except Exception:
        return create_directshape_head(doc, head), "marker"


def create_project_document(app):
    if TEMPLATE_PATH and os.path.exists(TEMPLATE_PATH):
        return app.NewProjectDocument(TEMPLATE_PATH)
    return app.NewProjectDocument(DB.UnitSystem.Metric)


def view_family_type(doc, family):
    try:
        for item in DB.FilteredElementCollector(doc).OfClass(DB.ViewFamilyType):
            if item.ViewFamily == family:
                return item
    except Exception:
        pass
    return None


def unique_view_name(doc, base_name):
    names = set()
    try:
        for view in DB.FilteredElementCollector(doc).OfClass(DB.View):
            names.add(safe_element_name(view))
    except Exception:
        pass
    if base_name not in names:
        return base_name
    index = 2
    while "{0} {1}".format(base_name, index) in names:
        index += 1
    return "{0} {1}".format(base_name, index)


def unhide_category(view, category):
    try:
        category_id = built_in_category_id(category)
        if view.CanCategoryBeHidden(category_id):
            view.SetCategoryHidden(category_id, False)
    except Exception:
        pass


def configure_sprinkler_view(view):
    try:
        view.DetailLevel = DB.ViewDetailLevel.Fine
    except Exception:
        pass
    try:
        view.DisplayStyle = DB.DisplayStyle.Wireframe
    except Exception:
        pass
    for category in (
        DB.BuiltInCategory.OST_PipeCurves,
        DB.BuiltInCategory.OST_PipeFitting,
        DB.BuiltInCategory.OST_PipeAccessory,
        DB.BuiltInCategory.OST_Sprinklers,
        DB.BuiltInCategory.OST_GenericModel,
        DB.BuiltInCategory.OST_Lines,
    ):
        unhide_category(view, category)


def add_bounds_point(bounds, coords):
    if not coords or len(coords) < 3:
        return
    bounds["xs"].append(float(coords[0]))
    bounds["ys"].append(float(coords[1]))
    bounds["zs"].append(float(coords[2]))


def add_footprint_points(bounds, footprint, z_m):
    if not footprint:
        return
    geom_type = footprint.get("type")
    polygons = []
    if geom_type == "Polygon":
        polygons = [footprint]
    elif geom_type == "MultiPolygon":
        polygons = [p for p in footprint.get("parts", []) if p and p.get("type") == "Polygon"]
    for polygon in polygons:
        for point in polygon.get("exterior", []) or []:
            if len(point) >= 2:
                bounds["xs"].append(float(point[0]))
                bounds["ys"].append(float(point[1]))
                bounds["zs"].append(float(z_m))


def layout_bounds(layout):
    bounds = {"xs": [], "ys": [], "zs": []}
    for pipe_run in layout.get("pipe_runs", []) or []:
        add_bounds_point(bounds, pipe_run.get("start_m"))
        add_bounds_point(bounds, pipe_run.get("end_m"))
    for head in layout.get("sprinkler_heads", []) or []:
        add_bounds_point(bounds, head.get("point_m"))
    for item in layout.get("context_elements", []) or []:
        add_footprint_points(bounds, item.get("footprint"), item.get("z_base_m", 0.0))
    if not bounds["xs"]:
        return None
    return {
        "min_x": min(bounds["xs"]),
        "max_x": max(bounds["xs"]),
        "min_y": min(bounds["ys"]),
        "max_y": max(bounds["ys"]),
        "min_z": min(bounds["zs"]),
        "max_z": max(bounds["zs"]),
    }


def set_3d_section_box(view, bounds):
    if not bounds:
        return
    try:
        pad_xy = 3.0
        pad_z_low = 1.0
        pad_z_high = 4.0
        box = DB.BoundingBoxXYZ()
        box.Min = DB.XYZ(m_to_ft(bounds["min_x"] - pad_xy), m_to_ft(bounds["min_y"] - pad_xy), m_to_ft(bounds["min_z"] - pad_z_low))
        box.Max = DB.XYZ(m_to_ft(bounds["max_x"] + pad_xy), m_to_ft(bounds["max_y"] + pad_xy), m_to_ft(bounds["max_z"] + pad_z_high))
        view.IsSectionBoxActive = True
        view.SetSectionBox(box)
    except Exception:
        pass


def create_review_views(doc, levels, layout):
    created = []
    floor_plan_type = view_family_type(doc, DB.ViewFamily.FloorPlan)
    for level_data in levels:
        level = level_data.get("element")
        if floor_plan_type is None or level is None:
            continue
        try:
            view = DB.ViewPlan.Create(doc, floor_plan_type.Id, level.Id)
            view.Name = unique_view_name(doc, "Sprinkler Layout - {0}".format(level_data.get("name") or safe_element_name(level)))
            configure_sprinkler_view(view)
            created.append(view)
        except Exception as exc:
            log("Warning: could not create plan view for {0}: {1}".format(level_data.get("name"), exc))
    three_d_type = view_family_type(doc, DB.ViewFamily.ThreeDimensional)
    if three_d_type is not None:
        try:
            view_3d = DB.View3D.CreateIsometric(doc, three_d_type.Id)
            view_3d.Name = unique_view_name(doc, "Sprinkler Layout - 3D All")
            configure_sprinkler_view(view_3d)
            set_3d_section_box(view_3d, layout_bounds(layout))
            created.append(view_3d)
        except Exception as exc:
            log("Warning: could not create 3D review view: {0}".format(exc))
    return created


def set_starting_view(doc, view):
    if view is None:
        return False
    try:
        settings = DB.StartingViewSettings.GetStartingViewSettings(doc)
        settings.ViewId = view.Id
        return True
    except Exception as exc:
        log("Warning: could not set starting view: {0}".format(exc))
        return False


def save_document(doc):
    out_dir = os.path.dirname(OUTPUT_RVT)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    if os.path.exists(OUTPUT_RVT):
        os.remove(OUTPUT_RVT)
    opts = DB.SaveAsOptions()
    opts.OverwriteExistingFile = True
    doc.SaveAs(OUTPUT_RVT, opts)


def main():
    if not LAYOUT_JSON or not OUTPUT_RVT:
        raise RuntimeError("SPRINKLER_REVIT_LAYOUT_JSON and SPRINKLER_REVIT_OUTPUT_RVT are required.")
    open(LOG_PATH, "w").close()
    log("Starting Revit RVT creation")
    log("Layout JSON: {0}".format(LAYOUT_JSON))
    log("Output RVT: {0}".format(OUTPUT_RVT))
    doc = __revit__.Application.NewProjectDocument(TEMPLATE_PATH) if TEMPLATE_PATH and os.path.exists(TEMPLATE_PATH) else create_project_document(__revit__.Application)
    with open(LAYOUT_JSON, "r") as f:
        layout = json.load(f)

    created_context = 0
    created_pipes = 0
    created_pipe_centerlines = 0
    created_native_heads = 0
    created_head_markers = 0
    created_views = []
    failed = []
    trans = DB.Transaction(doc, "Import sprinkler layout")
    trans.Start()
    try:
        level_cache = {}
        level_records = []
        levels = layout.get("levels") or [{"name": "Sprinkler Level", "elevation_m": 0.0}]
        for level_data in levels:
            name = level_data.get("name") or level_data.get("storey_name") or "Sprinkler Level"
            level = get_or_create_level(doc, name, float(level_data.get("elevation_m") or 0.0))
            level_cache[name] = level
            level_records.append({"name": name, "element": level})
        default_level = list(level_cache.values())[0]
        pipe_type = first_pipe_type(doc)
        system_type = first_piping_system_type(doc)
        symbol = sprinkler_symbol(doc)
        log("PipeType: {0}".format(safe_element_name(pipe_type)))
        log("SystemType: {0}".format(safe_element_name(system_type)))
        log("Sprinkler symbol: {0}".format(safe_element_name(symbol) if symbol else "<missing; markers will be used>"))

        for item in layout.get("context_elements", []):
            try:
                if create_context_element(doc, item) is not None:
                    created_context += 1
            except Exception as exc:
                failed.append("context {0}: {1}".format(item.get("id"), exc))
        for pipe_run in layout.get("pipe_runs", []):
            try:
                if pipe_type is None or system_type is None:
                    failed.append("pipe {0}: missing pipe type or system type".format(pipe_run.get("id")))
                    continue
                create_pipe(doc, pipe_run, pipe_type, system_type, item_level(pipe_run, level_cache, default_level))
                created_pipes += 1
            except Exception as exc:
                failed.append("pipe {0}: {1}".format(pipe_run.get("id"), exc))
            try:
                if create_pipe_centerline(doc, pipe_run) is not None:
                    created_pipe_centerlines += 1
            except Exception as exc:
                failed.append("pipe centerline {0}: {1}".format(pipe_run.get("id"), exc))
        for head in layout.get("sprinkler_heads", []):
            try:
                created, created_kind = create_head(doc, head, symbol, item_level(head, level_cache, default_level))
                if created is not None and created_kind == "native":
                    created_native_heads += 1
                elif created is not None:
                    created_head_markers += 1
            except Exception as exc:
                failed.append("head {0}: {1}".format(head.get("id"), exc))
        created_views = create_review_views(doc, level_records, layout)
        if created_views:
            set_starting_view(doc, created_views[0])
        trans.Commit()
    except Exception:
        trans.RollBack()
        raise

    log("Created context refs: {0}".format(created_context))
    log("Created native pipes: {0}".format(created_pipes))
    log("Created visible pipe centerlines: {0}".format(created_pipe_centerlines))
    log("Created native sprinkler heads: {0}".format(created_native_heads))
    log("Created fallback head markers: {0}".format(created_head_markers))
    log("Created heads/markers: {0}".format(created_native_heads + created_head_markers))
    log("Created review views: {0}".format(len(created_views)))
    if created_views:
        log("Starting/review view: {0}".format(safe_element_name(created_views[0])))
    log("Failures: {0}".format(len(failed)))
    for entry in failed[:120]:
        log(entry)
    save_document(doc)
    log("Saved RVT: {0}".format(OUTPUT_RVT))
    try:
        doc.Close(False)
        log("Closed generated document.")
    except Exception as exc:
        log("Warning: could not close generated document: {0}".format(exc))


try:
    main()
except Exception:
    text = traceback.format_exc()
    with open(LOG_PATH, "a") as f:
        f.write(text)
    raise
