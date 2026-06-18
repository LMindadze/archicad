export type Counts = Record<string, number>;

export type PolygonGeometry = {
  type: "Polygon";
  exterior: number[][];
  holes?: number[][][];
  area?: number;
};

export type MultiPolygonGeometry = {
  type: "MultiPolygon";
  parts: PolygonGeometry[];
  area?: number;
};

export type AnyGeometry = PolygonGeometry | MultiPolygonGeometry | null | undefined;

export type TrunkOverride = {
  start: number[];
  end: number[];
  source?: string;
  segments?: LayoutTrunkSegment[];
  main_trunk_line?: number[][];
};

export type FloorGeometry = {
  protected_floor_area?: AnyGeometry;
  columns?: Array<{ footprint?: AnyGeometry }>;
  walls?: Array<{ footprint?: AnyGeometry }>;
  stairs?: Array<{ footprint?: AnyGeometry }>;
  bounds?: { min_x: number; min_y: number; max_x: number; max_y: number };
  trunk_line?: number[][];
};

export type LayoutHead = {
  id: string;
  x: number;
  y: number;
  source?: string;
};

export type LayoutBranch = {
  id: string;
  points: number[][];
};

export type LayoutTrunkSegment = {
  id: string;
  start: number[];
  end: number[];
  kind?: string;
  diameter?: string;
};

export type LayoutGeometry = {
  protected_floor_area?: AnyGeometry;
  exclusion_area?: AnyGeometry;
  trunk_segments?: LayoutTrunkSegment[];
  branch_lines?: LayoutBranch[];
  sprinkler_heads?: LayoutHead[];
  counts?: Counts;
  parameters?: Record<string, unknown>;
};

export type Storey = {
  id: string;
  index: number;
  ifc_id?: number;
  global_id?: string;
  name: string;
  elevation_m: number;
  counts: Counts;
  selected?: boolean;
  status?: string;
  detected_json?: string;
  preview_url?: string;
  latest_detected_json?: string;
  latest_layout_preview_url?: string;
  latest_layout_json?: string;
  latest_score?: Record<string, unknown>;
  latest_score_json?: string;
  geometry?: FloorGeometry;
  layout?: LayoutGeometry;
  trunk?: { segments: LayoutTrunkSegment[] };
  trunk_override?: TrunkOverride | null;
  layout_edits?: { heads?: Record<string, number[]> };
};

export type Settings = {
  hazard_preset: string;
  head_type: string;
  head_spacing: number;
  branch_spacing: number;
  main_diameter: string;
  branch_diameter: string;
  column_clearance: number;
  stair_clearance: number;
  wall_clearance: number;
  min_obstacle_clearance: number;
  routing_model: string;
  layout_model: string;
  allow_secondary_branches: boolean;
  cpsat_time_limit: number;
  cpsat_max_demand: number;
  cpsat_min_head_spacing: number;
  demand_step: number;
  target_coverage: number;
  revit_year: string;
  revit_template: string;
};

export type Project = {
  id: string;
  name: string;
  original_filename: string;
  status: string;
  storeys: Storey[];
  settings: Settings;
  runs: Array<{ id: string; status: string; created_at: string; finished_at?: string; artifact_count?: number }>;
};

export type Artifact = {
  label: string;
  path: string;
  url: string;
};

export type Run = {
  id: string;
  project_id: string;
  stage?: string;
  floor_id?: string;
  status: "queued" | "running" | "complete" | "failed" | "cancelled";
  selected_floor_ids: string[];
  created_at: string;
  started_at?: string;
  finished_at?: string;
  artifacts: Artifact[];
  counts: Record<string, number>;
  floor_results: Array<{ floor_id: string; storey_name: string; preview_url: string; score?: Record<string, unknown> }>;
  log?: string;
  error?: string;
};
