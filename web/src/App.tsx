import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  AlertTriangle,
  Boxes,
  Building2,
  CheckCircle2,
  ChevronDown,
  CircleStop,
  Download,
  FileUp,
  Flame,
  FlipHorizontal,
  FlipVertical,
  Layers,
  Loader2,
  Maximize2,
  MousePointer2,
  Move,
  Play,
  RefreshCw,
  RotateCcw,
  Route,
  Save,
  Settings2,
  SlidersHorizontal,
  Sprout,
  Upload,
  ZoomIn,
  ZoomOut
} from "lucide-react";
import {
  analyzeProject,
  cancelRun,
  createSampleProject,
  getProject,
  getRun,
  listProjects,
  patchLayoutEdits,
  patchSettings,
  patchTrunk,
  startDetectFloor,
  startExportRevit,
  startGenerateSprinklers,
  startGenerateTrunk,
  uploadProject
} from "./api";
import type { AnyGeometry, FloorGeometry, LayoutGeometry, LayoutTrunkSegment, Project, Run, Settings, Storey, TrunkOverride } from "./types";

type Stage = "Input" | "Analyze" | "Detect" | "Trunk" | "Sprinklers" | "Review" | "Export";
type StageState = "locked" | "ready" | "active" | "done" | "running" | "failed";
type CanvasMode = "edit" | "pan";
type DragTarget =
  | { type: "trunk-start" | "trunk-end" | "head"; id?: string }
  | { type: "trunk-vertex"; key: string }
  | { type: "pan"; clientX: number; clientY: number };
type CanvasEvent = React.PointerEvent<Element>;
type TrunkGrip = { key: string; point: number[]; reason: "end" | "angle" | "junction" };

const stages: Stage[] = ["Input", "Analyze", "Detect", "Trunk", "Sprinklers", "Review", "Export"];

const stageIcon = {
  Input: FileUp,
  Analyze: Activity,
  Detect: Building2,
  Trunk: Route,
  Sprinklers: Sprout,
  Review: SlidersHorizontal,
  Export: Download
};

const defaultLayerState = {
  slab: true,
  walls: true,
  columns: true,
  trunk: true,
  branches: true,
  heads: true,
  warnings: true
};

const emptySettings: Settings = {
  hazard_preset: "ordinary_group_1",
  head_type: "dry_horizontal_sidewall",
  head_spacing: 3.2,
  branch_spacing: 4.0,
  main_diameter: "DN80",
  branch_diameter: "DN32",
  column_clearance: 0.75,
  stair_clearance: 0.6,
  wall_clearance: 0.25,
  min_obstacle_clearance: 0.4,
  routing_model: "direct",
  layout_model: "grid",
  allow_secondary_branches: true,
  cpsat_time_limit: 20,
  cpsat_max_demand: 12,
  cpsat_min_head_spacing: 2.0,
  demand_step: 0.25,
  target_coverage: 0.92,
  revit_year: "2027",
  revit_template: ""
};

function fmtNumber(value: number | undefined, digits = 2) {
  if (value === undefined || Number.isNaN(value)) return "-";
  return value.toFixed(digits);
}

function countSum(counts: Record<string, number> | undefined) {
  if (!counts) return 0;
  return Object.values(counts).reduce((sum, value) => sum + Number(value || 0), 0);
}

function geometryParts(geom: AnyGeometry): number[][][] {
  if (!geom) return [];
  if (geom.type === "Polygon") return [geom.exterior];
  if (geom.type === "MultiPolygon") return geom.parts.map((part) => part.exterior);
  return [];
}

function pushGeomPoints(points: number[][], geom: AnyGeometry) {
  geometryParts(geom).forEach((ring) => points.push(...ring));
}

function floorBounds(floor: Storey | null) {
  const geometry = floor?.geometry;
  if (geometry?.bounds) {
    const { min_x, min_y, max_x, max_y } = geometry.bounds;
    return { minX: min_x, minY: min_y, maxX: max_x, maxY: max_y };
  }
  const points: number[][] = [];
  pushGeomPoints(points, geometry?.protected_floor_area);
  geometry?.columns?.forEach((item) => pushGeomPoints(points, item.footprint));
  geometry?.walls?.forEach((item) => pushGeomPoints(points, item.footprint));
  const layout = floor?.layout;
  const generatedTrunkSegments = layout?.trunk_segments?.length ? layout.trunk_segments : floor?.trunk?.segments || [];
  generatedTrunkSegments.forEach((segment) => points.push(segment.start, segment.end));
  layout?.branch_lines?.forEach((line) => points.push(...line.points));
  layout?.sprinkler_heads?.forEach((head) => points.push([head.x, head.y]));
  const trunk = floor?.trunk_override;
  if (trunk) points.push(trunk.start, trunk.end);
  if (!points.length) return { minX: -25, minY: -45, maxX: 82, maxY: 4 };
  const xs = points.map((p) => p[0]);
  const ys = points.map((p) => p[1]);
  return { minX: Math.min(...xs), minY: Math.min(...ys), maxX: Math.max(...xs), maxY: Math.max(...ys) };
}

function transformPoint(point: number[], bounds: ReturnType<typeof floorBounds>, mirrorX: boolean, mirrorY: boolean) {
  const cx = (bounds.minX + bounds.maxX) / 2;
  const cy = (bounds.minY + bounds.maxY) / 2;
  return [mirrorX ? cx - (point[0] - cx) : point[0], mirrorY ? cy - (point[1] - cy) : point[1]];
}

function inverseTransformPoint(point: number[], bounds: ReturnType<typeof floorBounds>, mirrorX: boolean, mirrorY: boolean) {
  return transformPoint(point, bounds, mirrorX, mirrorY);
}

function trunkPointKey(point: number[]) {
  return `${Number(point[0]).toFixed(3)},${Number(point[1]).toFixed(3)}`;
}

function trunkSegmentLength(segment: LayoutTrunkSegment) {
  const dx = Number(segment.end[0]) - Number(segment.start[0]);
  const dy = Number(segment.end[1]) - Number(segment.start[1]);
  return Math.hypot(dx, dy);
}

function orderedTrunkPoints(segments: LayoutTrunkSegment[]) {
  const valid = segments.filter((segment) => trunkSegmentLength(segment) > 0.001);
  if (!valid.length) return [];
  const points = new Map<string, number[]>();
  const adjacency = new Map<string, Set<string>>();
  valid.forEach((segment) => {
    const startKey = trunkPointKey(segment.start);
    const endKey = trunkPointKey(segment.end);
    points.set(startKey, segment.start);
    points.set(endKey, segment.end);
    if (!adjacency.has(startKey)) adjacency.set(startKey, new Set());
    if (!adjacency.has(endKey)) adjacency.set(endKey, new Set());
    adjacency.get(startKey)?.add(endKey);
    adjacency.get(endKey)?.add(startKey);
  });
  const endKeys = [...adjacency.entries()].filter(([, next]) => next.size === 1).map(([key]) => key);
  const startKey = endKeys[0] || trunkPointKey(valid[0].start);
  const line: number[][] = [];
  const seenEdges = new Set<string>();
  let current = startKey;
  let previous = "";
  while (current) {
    const point = points.get(current);
    if (point) line.push(point);
    const nextKeys = [...(adjacency.get(current) || [])].filter((key) => {
      const edgeKey = [current, key].sort().join("|");
      return key !== previous && !seenEdges.has(edgeKey);
    });
    if (!nextKeys.length) break;
    const next = nextKeys[0];
    seenEdges.add([current, next].sort().join("|"));
    previous = current;
    current = next;
  }
  return line;
}

function trunkGripPoints(segments: LayoutTrunkSegment[]): TrunkGrip[] {
  const refs = new Map<string, { point: number[]; vectors: number[][] }>();
  segments.forEach((segment) => {
    if (trunkSegmentLength(segment) <= 0.001) return;
    [
      { point: segment.start, other: segment.end },
      { point: segment.end, other: segment.start }
    ].forEach(({ point, other }) => {
      const key = trunkPointKey(point);
      const entry = refs.get(key) || { point, vectors: [] };
      entry.vectors.push([Number(other[0]) - Number(point[0]), Number(other[1]) - Number(point[1])]);
      refs.set(key, entry);
    });
  });
  const grips: TrunkGrip[] = [];
  refs.forEach((entry, key) => {
    if (entry.vectors.length === 1) {
      grips.push({ key, point: entry.point, reason: "end" });
      return;
    }
    if (entry.vectors.length > 2) {
      grips.push({ key, point: entry.point, reason: "junction" });
      return;
    }
    const [a, b] = entry.vectors;
    const aLen = Math.hypot(a[0], a[1]);
    const bLen = Math.hypot(b[0], b[1]);
    if (!aLen || !bLen) return;
    const cross = Math.abs(a[0] * b[1] - a[1] * b[0]) / (aLen * bLen);
    const dot = (a[0] * b[0] + a[1] * b[1]) / (aLen * bLen);
    if (cross > 0.015 || dot > -0.985) {
      grips.push({ key, point: entry.point, reason: "angle" });
    }
  });
  return grips;
}

function moveTrunkVertex(segments: LayoutTrunkSegment[], key: string, point: number[]) {
  return segments.map((segment) => ({
    ...segment,
    start: trunkPointKey(segment.start) === key ? point : segment.start,
    end: trunkPointKey(segment.end) === key ? point : segment.end
  }));
}

function statusLabel(status?: string) {
  return (status || "idle").replace(/_/g, " ");
}

function stageState(stage: Stage, project: Project | null, floor: Storey | null, activeStage: Stage, run: Run | null): StageState {
  if (run && ["queued", "running"].includes(run.status)) {
    if ((run.stage === "detect" && stage === "Detect") || (run.stage === "trunk" && stage === "Trunk") || (run.stage === "sprinklers" && stage === "Sprinklers") || (run.stage === "export" && stage === "Export")) {
      return "running";
    }
  }
  if (run?.status === "failed" && ((run.stage === "detect" && stage === "Detect") || (run.stage === "trunk" && stage === "Trunk") || (run.stage === "sprinklers" && stage === "Sprinklers") || (run.stage === "export" && stage === "Export"))) {
    return "failed";
  }
  const done = {
    Input: Boolean(project),
    Analyze: Boolean(project?.storeys?.length),
    Detect: Boolean(floor?.detected_json),
    Trunk: Boolean(floor?.trunk_override),
    Sprinklers: Boolean(floor?.layout),
    Review: Boolean(floor?.layout),
    Export: Boolean(project?.runs?.some((item) => item.status === "complete" && item.artifact_count))
  } satisfies Record<Stage, boolean>;
  const prerequisites = {
    Input: true,
    Analyze: Boolean(project),
    Detect: Boolean(project?.storeys?.length && floor),
    Trunk: Boolean(floor?.detected_json),
    Sprinklers: Boolean(floor?.trunk_override),
    Review: Boolean(floor?.layout),
    Export: Boolean(project?.storeys?.some((item) => item.latest_layout_json))
  } satisfies Record<Stage, boolean>;
  if (done[stage]) return activeStage === stage ? "active" : "done";
  if (!prerequisites[stage]) return "locked";
  return activeStage === stage ? "active" : "ready";
}

function SettingsField({
  label,
  value,
  onChange,
  suffix,
  type = "number"
}: {
  label: string;
  value: number | string;
  onChange: (value: string) => void;
  suffix?: string;
  type?: "number" | "text";
}) {
  return (
    <label className="field">
      <span>{label}</span>
      <div className="inputShell">
        <input type={type} value={value} onChange={(event) => onChange(event.target.value)} />
        {suffix ? <b>{suffix}</b> : null}
      </div>
    </label>
  );
}

function SelectField({
  label,
  value,
  onChange,
  options
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  options: Array<{ label: string; value: string }>;
}) {
  return (
    <label className="field">
      <span>{label}</span>
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        {options.map((item) => (
          <option key={item.value} value={item.value}>
            {item.label}
          </option>
        ))}
      </select>
    </label>
  );
}

function StageRail({
  activeStage,
  project,
  activeFloor,
  run,
  onStage
}: {
  activeStage: Stage;
  project: Project | null;
  activeFloor: Storey | null;
  run: Run | null;
  onStage: (stage: Stage) => void;
}) {
  return (
    <nav className="stageRail">
      <div className="brandMark">
        <Flame size={20} />
        <span>
          IFC
          <br />
          RVT
        </span>
      </div>
      {stages.map((stage) => {
        const Icon = stageIcon[stage];
        const state = stageState(stage, project, activeFloor, activeStage, run);
        return (
          <button key={stage} className={`${activeStage === stage ? "active" : ""} ${state}`} type="button" onClick={() => onStage(stage)} title={`${stage}: ${state}`}>
            <Icon size={18} />
            <span>{stage}</span>
            <i>{state}</i>
          </button>
        );
      })}
    </nav>
  );
}

function ProjectIntake({
  project,
  busy,
  error,
  onUpload,
  onSample,
  onAnalyze
}: {
  project: Project | null;
  busy: boolean;
  error: string | null;
  onUpload: (file: File) => void;
  onSample: () => void;
  onAnalyze: () => void;
}) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  return (
    <section className="intakePanel">
      <div className="sectionTitle">
        <FileUp size={18} />
        <div>
          <h2>IFC Intake</h2>
          <p>Upload IFC, then analyze storeys before detection and layout generation.</p>
        </div>
      </div>
      <div className="uploadZone" onClick={() => inputRef.current?.click()}>
        <Upload size={24} />
        <strong>{project ? project.original_filename : "Upload IFC"}</strong>
        <span>{project ? `Project ${project.id}` : "Drop/select a .ifc file for local analysis"}</span>
        <input
          ref={inputRef}
          type="file"
          accept=".ifc"
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (file) onUpload(file);
          }}
        />
      </div>
      <div className="buttonRow">
        <button className="secondaryButton devButton" type="button" onClick={onSample} disabled={busy}>
          <Boxes size={16} /> Sample
        </button>
        <button className="primaryButton" type="button" onClick={onAnalyze} disabled={!project || busy}>
          {busy ? <Loader2 className="spin" size={16} /> : <Activity size={16} />}
          Analyze
        </button>
      </div>
      {error ? <div className="errorText">{error}</div> : null}
    </section>
  );
}

function FloorList({
  project,
  selectedFloorIds,
  activeFloorId,
  onSelect,
  onActivate
}: {
  project: Project | null;
  selectedFloorIds: Set<string>;
  activeFloorId: string | null;
  onSelect: (floorId: string, selected: boolean) => void;
  onActivate: (floorId: string) => void;
}) {
  return (
    <aside className="floorPanel">
      <div className="panelHeader">
        <div>
          <h3>Floors</h3>
          <span>{project?.storeys?.length || 0} analyzed storeys</span>
        </div>
        <Layers size={17} />
      </div>
      <div className="floorList">
        {project?.storeys?.map((floor) => {
          const selected = selectedFloorIds.has(floor.id);
          return (
            <button key={floor.id} type="button" className={`floorItem ${activeFloorId === floor.id ? "active" : ""}`} onClick={() => onActivate(floor.id)}>
              <span className="floorCheck" onClick={(event) => event.stopPropagation()}>
                <input type="checkbox" checked={selected} onChange={(event) => onSelect(floor.id, event.target.checked)} />
              </span>
              <span className="floorMain">
                <strong>{floor.name}</strong>
                <small>Elevation {fmtNumber(floor.elevation_m)} m</small>
                <em>{countSum(floor.counts)} IFC elements</em>
              </span>
              <span className={`statusDot ${floor.status || "idle"}`} title={floor.status || "idle"} />
              <span className="floorStatus">{statusLabel(floor.status)}</span>
            </button>
          );
        })}
        {!project?.storeys?.length ? <div className="emptyState">Analyze an IFC to populate floors.</div> : null}
      </div>
    </aside>
  );
}

function WorkflowActions({
  project,
  floor,
  run,
  selectedCount,
  onAnalyze,
  onDetect,
  onTrunk,
  onSprinklers,
  onExport,
  onCancel
}: {
  project: Project | null;
  floor: Storey | null;
  run: Run | null;
  selectedCount: number;
  onAnalyze: () => void;
  onDetect: () => void;
  onTrunk: () => void;
  onSprinklers: () => void;
  onExport: () => void;
  onCancel: () => void;
}) {
  const running = Boolean(run && ["queued", "running"].includes(run.status));
  return (
    <section className="workflowPanel">
      <div>
        <strong>{floor ? `${floor.name} / ${statusLabel(floor.status)}` : "No floor selected"}</strong>
        <span>{selectedCount} floors selected for combined RVT</span>
      </div>
      <div className="workflowActions">
        <button className="secondaryButton" type="button" onClick={onAnalyze} disabled={!project || running}>
          <Activity size={15} /> Analyze
        </button>
        <button className="secondaryButton" type="button" onClick={onDetect} disabled={!floor || !project?.storeys?.length || running}>
          <Building2 size={15} /> Detect
        </button>
        <button className="secondaryButton" type="button" onClick={onTrunk} disabled={!floor?.detected_json || running}>
          <Route size={15} /> Generate trunk
        </button>
        <button className="primaryButton" type="button" onClick={onSprinklers} disabled={!floor?.trunk_override || running}>
          <Sprout size={15} /> {floor?.layout ? "Regenerate" : "Generate sprinklers"}
        </button>
        <button className="primaryButton" type="button" onClick={onExport} disabled={!project?.storeys?.some((item) => item.latest_layout_json) || !selectedCount || running}>
          <Save size={15} /> Save RVT
        </button>
        {running ? (
          <button className="dangerButton" type="button" onClick={onCancel}>
            <CircleStop size={15} /> Cancel
          </button>
        ) : null}
      </div>
    </section>
  );
}

function PreviewCanvas({
  floor,
  onTrunkChange,
  onHeadMove
}: {
  floor: Storey | null;
  onTrunkChange: (trunk: TrunkOverride) => void;
  onHeadMove: (headId: string, point: number[]) => void;
}) {
  const [layers, setLayers] = useState(defaultLayerState);
  const [mode, setMode] = useState<CanvasMode>("edit");
  const [mirrorX, setMirrorX] = useState(false);
  const [mirrorY, setMirrorY] = useState(true);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState<[number, number]>([0, 0]);
  const [hoverWorld, setHoverWorld] = useState<number[] | null>(null);
  const [dragging, setDragging] = useState<DragTarget | null>(null);
  const svgRef = useRef<SVGSVGElement | null>(null);
  const bounds = floorBounds(floor);
  const pad = 3;
  const fullWidth = Math.max(1, bounds.maxX - bounds.minX + pad * 2);
  const fullHeight = Math.max(1, bounds.maxY - bounds.minY + pad * 2);
  const centerX = (bounds.minX + bounds.maxX) / 2;
  const centerY = (bounds.minY + bounds.maxY) / 2;
  const viewWidth = fullWidth / zoom;
  const viewHeight = fullHeight / zoom;
  const view = {
    minX: centerX - viewWidth / 2 + pan[0],
    minY: centerY - viewHeight / 2 + pan[1],
    width: viewWidth,
    height: viewHeight
  };
  const viewBox = `${view.minX} ${view.minY} ${view.width} ${view.height}`;
  const geometry = floor?.geometry;
  const layout = floor?.layout;
  const trunk = floor?.trunk_override || (geometry?.trunk_line ? { start: geometry.trunk_line[0], end: geometry.trunk_line[geometry.trunk_line.length - 1], source: "auto" } : null);
  const layoutTrunks = trunk?.segments?.length ? trunk.segments : layout?.trunk_segments?.length ? layout.trunk_segments : floor?.trunk?.segments || [];
  const trunkGrips = useMemo(() => trunkGripPoints(layoutTrunks), [layoutTrunks]);
  const showStraightTrunk = Boolean(trunk && (!layoutTrunks.length || (trunk.source === "user" && !trunk.segments?.length)));

  const toScreen = useCallback((point: number[]) => transformPoint(point, bounds, mirrorX, mirrorY), [bounds, mirrorX, mirrorY]);
  const toWorld = useCallback((point: number[]) => inverseTransformPoint(point, bounds, mirrorX, mirrorY), [bounds, mirrorX, mirrorY]);

  const clientToScreen = useCallback(
    (event: CanvasEvent) => {
      const svg = svgRef.current;
      if (!svg) return [0, 0];
      const rect = svg.getBoundingClientRect();
      const x = view.minX + ((event.clientX - rect.left) / rect.width) * view.width;
      const y = view.minY + ((event.clientY - rect.top) / rect.height) * view.height;
      return [Number(x.toFixed(3)), Number(y.toFixed(3))];
    },
    [view.height, view.minX, view.minY, view.width]
  );

  const clientToWorld = useCallback((event: CanvasEvent) => toWorld(clientToScreen(event)).map((value) => Number(value.toFixed(3))), [clientToScreen, toWorld]);

  function capturePointer(event: CanvasEvent) {
    if ("pointerId" in event) {
      event.currentTarget.setPointerCapture(event.pointerId);
    }
  }

  function beginCanvasDrag(event: CanvasEvent) {
    if (mode === "pan") {
      setDragging({ type: "pan", clientX: event.clientX, clientY: event.clientY });
      capturePointer(event);
    }
  }

  function movePointer(event: CanvasEvent) {
    const world = clientToWorld(event);
    setHoverWorld(world);
    if (dragging?.type === "pan") {
      const svg = svgRef.current;
      if (!svg) return;
      const rect = svg.getBoundingClientRect();
      const dx = ((event.clientX - dragging.clientX) / rect.width) * view.width;
      const dy = ((event.clientY - dragging.clientY) / rect.height) * view.height;
      setPan((prev) => [Number((prev[0] - dx).toFixed(3)), Number((prev[1] - dy).toFixed(3))]);
      setDragging({ ...dragging, clientX: event.clientX, clientY: event.clientY });
    }
  }

  function finishDrag(event: CanvasEvent) {
    if (!dragging) return;
    if (dragging.type === "trunk-start" || dragging.type === "trunk-end") {
      if (!floor || !trunk) return;
      const point = clientToWorld(event);
      onTrunkChange(dragging.type === "trunk-start" ? { ...trunk, start: point, source: "user" } : { ...trunk, end: point, source: "user" });
    }
    if (dragging.type === "trunk-vertex") {
      if (!floor || !trunk || !layoutTrunks.length) return;
      const point = clientToWorld(event);
      const segments = moveTrunkVertex(layoutTrunks, dragging.key, point).filter((segment) => trunkSegmentLength(segment) > 0.001);
      const line = orderedTrunkPoints(segments);
      const start = line[0] || trunk.start;
      const end = line[line.length - 1] || trunk.end;
      onTrunkChange({ ...trunk, start, end, segments, main_trunk_line: line, source: "user" });
    }
    if (dragging.type === "head" && dragging.id) {
      onHeadMove(dragging.id, clientToWorld(event));
    }
    setDragging(null);
  }

  function polygonPoints(ring: number[][]) {
    return ring.map((p) => toScreen(p).join(",")).join(" ");
  }

  return (
    <section className="previewPanel">
      <div className="previewToolbar">
        <div>
          <h2>{floor ? floor.name : "Floor preview"}</h2>
          <span>{floor ? `IfcBuildingStorey ${floor.ifc_id || "-"} / ${fmtNumber(floor.elevation_m)} m / ${statusLabel(floor.status)}` : "Select a floor and detect geometry"}</span>
        </div>
        <div className="canvasTools">
          <button type="button" className={mode === "edit" ? "selected" : ""} onClick={() => setMode("edit")} title="Edit">
            <MousePointer2 size={14} />
          </button>
          <button type="button" className={mode === "pan" ? "selected" : ""} onClick={() => setMode("pan")} title="Pan">
            <Move size={14} />
          </button>
          <button type="button" onClick={() => setZoom((value) => Math.min(8, Number((value * 1.2).toFixed(2))))} title="Zoom in">
            <ZoomIn size={14} />
          </button>
          <button type="button" onClick={() => setZoom((value) => Math.max(0.4, Number((value / 1.2).toFixed(2))))} title="Zoom out">
            <ZoomOut size={14} />
          </button>
          <button type="button" onClick={() => { setZoom(1); setPan([0, 0]); }} title="Fit">
            <Maximize2 size={14} />
          </button>
          <button type="button" className={mirrorX ? "selected" : ""} onClick={() => setMirrorX((value) => !value)} title="Mirror X">
            <FlipHorizontal size={14} />
          </button>
          <button type="button" className={mirrorY ? "selected" : ""} onClick={() => setMirrorY((value) => !value)} title="Mirror Y / CAD Y-up">
            <FlipVertical size={14} />
          </button>
          <button type="button" onClick={() => { setMirrorX(false); setMirrorY(true); setZoom(1); setPan([0, 0]); }} title="Reset orientation">
            <RotateCcw size={14} />
          </button>
        </div>
      </div>
      <div className="layerToggles">
        {Object.entries(layers).map(([key, value]) => (
          <button key={key} type="button" className={value ? "selected" : ""} onClick={() => setLayers((prev) => ({ ...prev, [key]: !prev[key as keyof typeof prev] }))}>
            {key}
          </button>
        ))}
      </div>
      <div className="canvasFrame">
        <svg
          ref={svgRef}
          viewBox={viewBox}
          className={`planSvg ${mode === "pan" ? "panMode" : ""}`}
          onPointerDown={beginCanvasDrag}
          onPointerMove={movePointer}
          onPointerUp={finishDrag}
          onPointerLeave={() => setHoverWorld(null)}
        >
          <rect x={view.minX} y={view.minY} width={view.width} height={view.height} className="svgBg" />
          {layers.slab &&
            geometryParts(geometry?.protected_floor_area || layout?.protected_floor_area).map((ring, idx) => <polygon key={`slab-${idx}`} points={polygonPoints(ring)} className="slabShape" />)}
          {layers.walls &&
            geometry?.walls?.flatMap((item, i) =>
              geometryParts(item.footprint).map((ring, idx) => <polygon key={`wall-${i}-${idx}`} points={polygonPoints(ring)} className="wallShape" />)
            )}
          {layers.columns &&
            geometry?.columns?.flatMap((item, i) =>
              geometryParts(item.footprint).map((ring, idx) => <polygon key={`col-${i}-${idx}`} points={polygonPoints(ring)} className="columnShape" />)
            )}
          {layers.branches &&
            layout?.branch_lines?.map((line) => {
              const pts = line.points.map(toScreen);
              return <polyline key={line.id} points={pts.map((p) => p.join(",")).join(" ")} className="branchLine" />;
            })}
          {layers.trunk &&
            layoutTrunks.map((segment) => {
              const a = toScreen(segment.start);
              const b = toScreen(segment.end);
              return <line key={segment.id} x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]} className="trunkLine generated" />;
            })}
          {layers.trunk && trunk ? (
            <>
              {showStraightTrunk
                ? (() => {
                    const a = toScreen(trunk.start);
                    const b = toScreen(trunk.end);
                    return <line x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]} className="trunkLine override" />;
                  })()
                : null}
            </>
          ) : null}
          {layers.heads &&
            layout?.sprinkler_heads?.map((head) => {
              const p = toScreen([head.x, head.y]);
              return (
                <g key={head.id} className={head.source === "user_override" ? "headGroup edited" : "headGroup"}>
                  <circle
                    cx={p[0]}
                    cy={p[1]}
                    r={1.75 / Math.sqrt(zoom)}
                    className="headHit"
                    onPointerDown={(event) => {
                      if (mode !== "edit") return;
                      event.stopPropagation();
                      setDragging({ type: "head", id: head.id });
                      capturePointer(event);
                    }}
                    onPointerUp={(event) => {
                      event.stopPropagation();
                      finishDrag(event);
                    }}
                  />
                  <circle
                    cx={p[0]}
                    cy={p[1]}
                    r={0.62 / Math.sqrt(zoom)}
                    className="headRing"
                    onPointerDown={(event) => {
                      if (mode !== "edit") return;
                      event.stopPropagation();
                      setDragging({ type: "head", id: head.id });
                      capturePointer(event);
                    }}
                    onPointerUp={(event) => {
                      event.stopPropagation();
                      finishDrag(event);
                    }}
                  />
                  <circle cx={p[0]} cy={p[1]} r={0.12 / Math.sqrt(zoom)} className="headDot" />
                </g>
              );
            })}
          {layers.trunk && trunk ? (
            <g className="trunkHandleLayer">
              {showStraightTrunk
                ? (["start", "end"] as const).map((key) => {
                    const p = toScreen(trunk[key]);
                    const size = 1.15 / Math.sqrt(zoom);
                    return (
                      <rect
                        key={key}
                        x={p[0] - size / 2}
                        y={p[1] - size / 2}
                        width={size}
                        height={size}
                        className="trunkHandle"
                        onPointerDown={(event) => {
                          if (mode !== "edit") return;
                          event.stopPropagation();
                          setDragging({ type: key === "start" ? "trunk-start" : "trunk-end" });
                          capturePointer(event);
                        }}
                        onPointerUp={(event) => {
                          event.stopPropagation();
                          finishDrag(event);
                        }}
                      />
                    );
                  })
                : null}
              {!showStraightTrunk
                ? trunkGrips.map((grip) => {
                    const p = toScreen(grip.point);
                    const size = 1.15 / Math.sqrt(zoom);
                    return (
                      <rect
                        key={grip.key}
                        x={p[0] - size / 2}
                        y={p[1] - size / 2}
                        width={size}
                        height={size}
                        className={`trunkHandle ${grip.reason}`}
                        onPointerDown={(event) => {
                          if (mode !== "edit") return;
                          event.stopPropagation();
                          setDragging({ type: "trunk-vertex", key: grip.key });
                          capturePointer(event);
                        }}
                        onPointerUp={(event) => {
                          event.stopPropagation();
                          finishDrag(event);
                        }}
                      />
                    );
                  })
                : null}
            </g>
          ) : null}
        </svg>
        {!geometry ? (
          <div className="canvasEmpty">
            <Route size={26} />
            <strong>Detect floor geometry</strong>
            <span>Use the staged actions to detect, generate trunk, then generate sprinklers.</span>
          </div>
        ) : null}
      </div>
      <div className="canvasStatus">
        <span>{hoverWorld ? `x ${fmtNumber(hoverWorld[0], 3)} / y ${fmtNumber(hoverWorld[1], 3)}` : "Move over canvas for coordinates"}</span>
        <span>zoom {fmtNumber(zoom, 2)} / {mirrorX ? "mirror X" : "normal X"} / {mirrorY ? "Y up" : "SVG Y down"}</span>
        <span>{layout?.sprinkler_heads?.length || 0} heads / {layout?.branch_lines?.length || 0} branches</span>
      </div>
    </section>
  );
}

function SettingsInspector({
  settings,
  onChange,
  selectedCount
}: {
  settings: Settings;
  onChange: (settings: Partial<Settings>) => void;
  selectedCount: number;
}) {
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const setNumber = (key: keyof Settings) => (value: string) => onChange({ [key]: Number(value) } as Partial<Settings>);
  const setString = (key: keyof Settings) => (value: string) => onChange({ [key]: value } as Partial<Settings>);
  return (
    <aside className="settingsPanel">
      <div className="panelHeader">
        <div>
          <h3>Sprinkler Setup</h3>
          <span>Editable before and after generation / {selectedCount} selected floors</span>
        </div>
        <Settings2 size={17} />
      </div>
      <div className="settingGroup">
        <SelectField
          label="Hazard preset"
          value={settings.hazard_preset}
          onChange={setString("hazard_preset")}
          options={[
            { label: "Ordinary Group 1", value: "ordinary_group_1" },
            { label: "Light Hazard", value: "light" },
            { label: "Ordinary Group 2", value: "ordinary_group_2" },
            { label: "Custom", value: "custom" }
          ]}
        />
        <SelectField
          label="Head type"
          value={settings.head_type}
          onChange={setString("head_type")}
          options={[
            { label: "Dry horizontal sidewall", value: "dry_horizontal_sidewall" },
            { label: "Pendent", value: "pendent" },
            { label: "Upright", value: "upright" },
            { label: "Sidewall", value: "sidewall" }
          ]}
        />
        <div className="dualFields">
          <SettingsField label="Head spacing" value={settings.head_spacing} onChange={setNumber("head_spacing")} suffix="m" />
          <SettingsField label="Branch spacing" value={settings.branch_spacing} onChange={setNumber("branch_spacing")} suffix="m" />
        </div>
        <div className="dualFields">
          <SettingsField label="Main pipe" value={settings.main_diameter} onChange={setString("main_diameter")} type="text" />
          <SettingsField label="Branch pipe" value={settings.branch_diameter} onChange={setString("branch_diameter")} type="text" />
        </div>
        <SelectField
          label="Routing model"
          value={settings.routing_model}
          onChange={setString("routing_model")}
          options={[
            { label: "Direct from trunk", value: "direct" },
            { label: "Steiner/shared", value: "steiner" },
            { label: "Legacy", value: "legacy" }
          ]}
        />
        <label className="toggleLine">
          <input type="checkbox" checked={settings.allow_secondary_branches} onChange={(event) => onChange({ allow_secondary_branches: event.target.checked })} />
          <span>Allow secondary branches</span>
        </label>
      </div>
      <button className="advancedToggle" type="button" onClick={() => setAdvancedOpen((open) => !open)}>
        <ChevronDown size={16} className={advancedOpen ? "open" : ""} />
        Advanced
      </button>
      {advancedOpen ? (
        <div className="settingGroup advanced">
          <div className="dualFields">
            <SettingsField label="Wall clearance" value={settings.wall_clearance} onChange={setNumber("wall_clearance")} suffix="m" />
            <SettingsField label="Column clearance" value={settings.column_clearance} onChange={setNumber("column_clearance")} suffix="m" />
          </div>
          <div className="dualFields">
            <SettingsField label="CP-SAT limit" value={settings.cpsat_time_limit} onChange={setNumber("cpsat_time_limit")} suffix="s" />
            <SettingsField label="Max demand" value={settings.cpsat_max_demand} onChange={setNumber("cpsat_max_demand")} />
          </div>
          <SettingsField label="Revit template" value={settings.revit_template} onChange={setString("revit_template")} type="text" />
        </div>
      ) : null}
    </aside>
  );
}

function RunConsole({ run, onCancel }: { run: Run | null; onCancel: () => void }) {
  const running = run?.status === "running" || run?.status === "queued";
  return (
    <section className="runConsole">
      <div className="runHeader">
        <div>
          <strong>{run ? `${run.stage || "run"} / ${run.status}` : "Stage log"}</strong>
          <span>{run ? run.id : "Logs and artifacts appear here after each stage."}</span>
        </div>
        {running ? (
          <button className="dangerButton" type="button" onClick={onCancel}>
            <CircleStop size={15} /> Cancel
          </button>
        ) : null}
      </div>
      <pre>{run?.log || "No active stage."}</pre>
      {run?.artifacts?.length ? (
        <div className="artifactRail">
          {run.artifacts.map((artifact) => (
            <a key={artifact.path} href={artifact.url} target="_blank" rel="noreferrer">
              <Download size={14} />
              {artifact.label}
            </a>
          ))}
        </div>
      ) : null}
    </section>
  );
}

export function App() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [project, setProject] = useState<Project | null>(null);
  const [activeStage, setActiveStage] = useState<Stage>("Input");
  const [activeFloorId, setActiveFloorId] = useState<string | null>(null);
  const [selectedFloorIds, setSelectedFloorIds] = useState<Set<string>>(new Set());
  const [settingsDraft, setSettingsDraft] = useState<Partial<Settings>>({});
  const [run, setRun] = useState<Run | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const settings = useMemo(() => ({ ...emptySettings, ...(project?.settings || {}), ...settingsDraft }) as Settings, [project?.settings, settingsDraft]);
  const activeFloor = project?.storeys?.find((floor) => floor.id === activeFloorId) || project?.storeys?.[0] || null;

  const refreshProject = useCallback(
    async (projectId: string) => {
      const next = await getProject(projectId);
      setProject(next);
      setSelectedFloorIds((prev) => {
        if (prev.size) return new Set([...prev].filter((id) => next.storeys.some((floor) => floor.id === id)));
        return new Set(next.storeys.filter((floor) => floor.selected).map((floor) => floor.id));
      });
      if (!activeFloorId && next.storeys[0]) setActiveFloorId(next.storeys[0].id);
    },
    [activeFloorId]
  );

  useEffect(() => {
    listProjects()
      .then((items) => {
        setProjects(items);
        if (items[0]) {
          setProject(items[0]);
          setSelectedFloorIds(new Set(items[0].storeys.filter((floor) => floor.selected).map((floor) => floor.id)));
          setActiveFloorId(items[0].storeys[0]?.id || null);
        }
      })
      .catch((err) => setError(err.message));
  }, []);

  useEffect(() => {
    if (!run || (run.status !== "running" && run.status !== "queued")) return;
    const timer = window.setInterval(async () => {
      const next = await getRun(run.id);
      setRun(next);
      if (project) await refreshProject(project.id);
    }, 1600);
    return () => window.clearInterval(timer);
  }, [project, refreshProject, run]);

  async function withBusy(action: () => Promise<void>) {
    setBusy(true);
    setError(null);
    try {
      await action();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  async function startStage(stage: Stage, action: () => Promise<Run>) {
    setError(null);
    const next = await action();
    setRun(next);
    setActiveStage(stage);
  }

  const selectFloor = (floorId: string, selected: boolean) => {
    setSelectedFloorIds((prev) => {
      const next = new Set(prev);
      if (selected) next.add(floorId);
      else next.delete(floorId);
      return next;
    });
  };

  const selectedFloorArray = Array.from(selectedFloorIds);

  return (
    <main className="appShell">
      <StageRail activeStage={activeStage} project={project} activeFloor={activeFloor} run={run} onStage={setActiveStage} />

      <div className="workspace">
        <header className="topBar">
          <div>
            <h1>{project?.name || "Local IFC-to-Revit Sprinkler App"}</h1>
            <span>{project ? `${project.original_filename} / ${project.status}` : "Input, analyze, detect, trunk, sprinklers, review, save RVT"}</span>
          </div>
          <div className="statusCluster">
            <span className="statusPill"><CheckCircle2 size={14} /> Revit 2027</span>
            <span className="statusPill"><Sprout size={14} /> Staged</span>
            {run ? <span className={`statusPill runPill ${run.status}`}>{run.stage || "run"} / {run.status}</span> : null}
          </div>
        </header>

        <WorkflowActions
          project={project}
          floor={activeFloor}
          run={run}
          selectedCount={selectedFloorIds.size}
          onAnalyze={() =>
            project &&
            withBusy(async () => {
              const next = await analyzeProject(project.id);
              setProject(next);
              setSelectedFloorIds(new Set(next.storeys.filter((floor) => floor.selected).map((floor) => floor.id)));
              setActiveFloorId(next.storeys[0]?.id || null);
              setActiveStage("Detect");
            })
          }
          onDetect={() =>
            project &&
            activeFloor &&
            startStage("Detect", async () => startDetectFloor(project.id, activeFloor.id))
          }
          onTrunk={() =>
            project &&
            activeFloor &&
            startStage("Trunk", async () => startGenerateTrunk(project.id, activeFloor.id, settingsDraft))
          }
          onSprinklers={() =>
            project &&
            activeFloor &&
            startStage("Sprinklers", async () => startGenerateSprinklers(project.id, activeFloor.id, settingsDraft))
          }
          onExport={() =>
            project &&
            startStage("Export", async () => startExportRevit(project.id, selectedFloorArray, settingsDraft))
          }
          onCancel={() => run && cancelRun(run.id).then(setRun).catch((err) => setError(err.message))}
        />

        <div className="mainGrid">
          <div className="leftStack">
            <ProjectIntake
              project={project}
              busy={busy}
              error={error}
              onSample={() =>
                withBusy(async () => {
                  const next = await createSampleProject();
                  setProject(next);
                  setProjects([next, ...projects]);
                  setActiveStage("Analyze");
                })
              }
              onUpload={(file) =>
                withBusy(async () => {
                  const next = await uploadProject(file);
                  setProject(next);
                  setProjects([next, ...projects]);
                  setActiveStage("Analyze");
                })
              }
              onAnalyze={() =>
                project &&
                withBusy(async () => {
                  const next = await analyzeProject(project.id);
                  setProject(next);
                  setSelectedFloorIds(new Set(next.storeys.filter((floor) => floor.selected).map((floor) => floor.id)));
                  setActiveFloorId(next.storeys[0]?.id || null);
                  setActiveStage("Detect");
                })
              }
            />
            <FloorList project={project} selectedFloorIds={selectedFloorIds} activeFloorId={activeFloorId} onSelect={selectFloor} onActivate={setActiveFloorId} />
          </div>

          <PreviewCanvas
            floor={activeFloor}
            onTrunkChange={(trunk) =>
              project &&
              activeFloor &&
              withBusy(async () => {
                const next = await patchTrunk(project.id, activeFloor.id, trunk);
                setProject(next);
                setActiveStage("Trunk");
              })
            }
            onHeadMove={(headId, point) =>
              project &&
              activeFloor &&
              withBusy(async () => {
                const next = await patchLayoutEdits(project.id, activeFloor.id, { heads: { [headId]: point } });
                setProject(next);
                setActiveStage("Review");
              })
            }
          />

          <SettingsInspector
            settings={settings}
            selectedCount={selectedFloorIds.size}
            onChange={(patch) => {
              setSettingsDraft((prev) => ({ ...prev, ...patch }));
              if (project) {
                patchSettings(project.id, { ...settingsDraft, ...patch }).then(setProject).catch((err) => setError(err.message));
              }
            }}
          />
        </div>

        <RunConsole run={run} onCancel={() => run && cancelRun(run.id).then(setRun).catch((err) => setError(err.message))} />
        {error ? (
          <div className="globalError">
            <AlertTriangle size={16} />
            {error}
          </div>
        ) : null}
      </div>
    </main>
  );
}
