from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy import ndimage as ndi
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.ops import unary_union
from shapely import contains_xy
from skimage.draw import line as sk_line
from skimage.morphology import skeletonize

try:
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    _HAS_SAM2 = True
except Exception:
    _HAS_SAM2 = False


class ProgressTracker:
    def __init__(self, total_steps: int) -> None:
        self.total_steps = max(1, int(total_steps))
        self.current = 0
        self.step_start = 0.0
        self.durations: list[float] = []
        self.global_start = time.perf_counter()

    def _fmt(self, seconds: float) -> str:
        s = max(0, int(round(seconds)))
        mm = s // 60
        ss = s % 60
        return f"{mm:02d}:{ss:02d}"

    def start(self, label: str) -> None:
        self.current += 1
        pct = int(round((self.current - 1) / self.total_steps * 100))
        filled = int(round((self.current - 1) / self.total_steps * 24))
        bar = "#" * filled + "-" * (24 - filled)
        eta = 0.0
        if self.durations:
            avg = sum(self.durations) / len(self.durations)
            eta = avg * (self.total_steps - self.current + 1)
        print(f"[{self.current}/{self.total_steps}] [{bar}] {pct:3d}% | ETA ~{self._fmt(eta)} | {label}")
        self.step_start = time.perf_counter()

    def done(self, extra: str | None = None) -> None:
        elapsed = time.perf_counter() - self.step_start
        self.durations.append(elapsed)
        pct = int(round(self.current / self.total_steps * 100))
        total_elapsed = time.perf_counter() - self.global_start
        msg = f"      done in {self._fmt(elapsed)} | total {self._fmt(total_elapsed)} | progress {pct}%"
        if extra:
            msg += f" | {extra}"
        print(msg)

    def finish(self) -> None:
        total_elapsed = time.perf_counter() - self.global_start
        print(f"[✓] Completed all steps in {self._fmt(total_elapsed)}")


class StepBar:
    def __init__(self, label: str, total: int, width: int = 28) -> None:
        self.label = label
        self.total = max(1, int(total))
        self.width = max(10, int(width))
        self.start = time.perf_counter()
        self.last_print = -1

    def _fmt(self, seconds: float) -> str:
        s = max(0, int(round(seconds)))
        mm = s // 60
        ss = s % 60
        return f"{mm:02d}:{ss:02d}"

    def update(self, current: int) -> None:
        cur = max(0, min(self.total, int(current)))
        pct = int(round((cur / self.total) * 100))
        # Avoid printing too frequently if percentage did not change.
        if pct == self.last_print and cur < self.total:
            return
        self.last_print = pct
        filled = int(round((cur / self.total) * self.width))
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = time.perf_counter() - self.start
        eta = 0.0
        if cur > 0:
            eta = elapsed * (self.total - cur) / cur
        print(
            f"\r      {self.label} [{bar}] {pct:3d}% | elapsed {self._fmt(elapsed)} | ETA {self._fmt(eta)}",
            end="",
            flush=True,
        )
        if cur >= self.total:
            print("")

    def done(self, extra: str | None = None) -> None:
        self.update(self.total)
        if extra:
            print(f"      {extra}")


def geometry_from_json(data: dict[str, Any] | None) -> Polygon | MultiPolygon | None:
    if data is None:
        return None
    t = data.get("type")
    if t == "Polygon":
        ext = data.get("exterior", [])
        holes = data.get("holes", [])
        if len(ext) < 3:
            return None
        return Polygon(ext, holes=holes)
    if t == "MultiPolygon":
        polys: list[Polygon] = []
        for part in data.get("parts", []):
            g = geometry_from_json(part)
            if isinstance(g, Polygon):
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
    if hasattr(cleaned, "geoms"):
        polys = [g for g in cleaned.geoms if isinstance(g, (Polygon, MultiPolygon)) and not g.is_empty]
        if not polys:
            return None
        return normalize_polygon(unary_union(polys))
    return None


def build_canvas_bounds(geom: Polygon | MultiPolygon, margin_m: float) -> tuple[float, float, float, float]:
    minx, miny, maxx, maxy = geom.bounds
    return (minx - margin_m, miny - margin_m, maxx + margin_m, maxy + margin_m)


def world_to_pixel(x: float, y: float, bounds: tuple[float, float, float, float], ppm: float) -> tuple[int, int]:
    minx, miny, _, maxy = bounds
    col = int(round((x - minx) * ppm))
    row = int(round((maxy - y) * ppm))
    return row, col


def pixel_to_world(row: int, col: int, bounds: tuple[float, float, float, float], ppm: float) -> tuple[float, float]:
    minx, _, _, maxy = bounds
    x = minx + (float(col) / ppm)
    y = maxy - (float(row) / ppm)
    return (x, y)


def rasterize_polygon_mask(
    geom: Polygon | MultiPolygon | None,
    bounds: tuple[float, float, float, float],
    shape_hw: tuple[int, int],
    ppm: float,
    bar_label: str | None = None,
) -> np.ndarray:
    h, w = shape_hw
    if geom is None or geom.is_empty:
        return np.zeros((h, w), dtype=bool)
    xs = bounds[0] + (np.arange(w, dtype=float) + 0.5) / ppm
    ys = bounds[3] - (np.arange(h, dtype=float) + 0.5) / ppm
    out = np.zeros((h, w), dtype=bool)
    polys = [geom] if isinstance(geom, Polygon) else list(geom.geoms)
    bar = StepBar(bar_label, h) if bar_label else None
    # Vectorized point-in-polygon over each row: vastly faster than per-point Point() loops.
    for r in range(h):
        row_mask = np.zeros(w, dtype=bool)
        y_row = np.full(w, ys[r], dtype=float)
        for poly in polys:
            row_mask |= contains_xy(poly, xs, y_row)
        out[r] = row_mask
        if bar:
            bar.update(r + 1)
    return out


def make_prompt_image(
    slab_mask: np.ndarray,
    wall_mask: np.ndarray,
    column_mask: np.ndarray,
) -> np.ndarray:
    h, w = slab_mask.shape
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    img[slab_mask] = np.array([230, 240, 248], dtype=np.uint8)
    img[wall_mask] = np.array([80, 80, 80], dtype=np.uint8)
    img[column_mask] = np.array([120, 100, 90], dtype=np.uint8)
    return img


def nearest_true(mask: np.ndarray, seed: tuple[int, int]) -> tuple[int, int] | None:
    r0, c0 = seed
    if r0 < 0 or c0 < 0 or r0 >= mask.shape[0] or c0 >= mask.shape[1]:
        return None
    if mask[r0, c0]:
        return seed
    idx = np.argwhere(mask)
    if len(idx) == 0:
        return None
    dr = idx[:, 0] - r0
    dc = idx[:, 1] - c0
    k = int(np.argmin(dr * dr + dc * dc))
    return int(idx[k, 0]), int(idx[k, 1])


def skeleton_graph(skel: np.ndarray, bar_label: str | None = None) -> nx.Graph:
    g = nx.Graph()
    rr, cc = np.where(skel)
    active_list = list(zip(rr.tolist(), cc.tolist()))
    active = set(active_list)
    nbrs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    bar = StepBar(bar_label, len(active_list)) if bar_label and active_list else None
    for idx, (r, c) in enumerate(active_list):
        g.add_node((r, c))
        for dr, dc in nbrs:
            nr, nc = r + dr, c + dc
            if (nr, nc) in active:
                g.add_edge((r, c), (nr, nc), weight=math.hypot(dr, dc))
        if bar:
            bar.update(idx + 1)
    return g


def grid_shortest_path(mask: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]] | None:
    """Shortest path on a binary mask using 8-neighbor Dijkstra."""
    rr, cc = np.where(mask)
    if len(rr) == 0:
        return None
    active = set(zip(rr.tolist(), cc.tolist()))
    if start not in active or goal not in active:
        return None

    g = nx.Graph()
    nbrs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for r, c in active:
        g.add_node((r, c))
        for dr, dc in nbrs:
            n = (r + dr, c + dc)
            if n in active:
                g.add_edge((r, c), n, weight=math.hypot(dr, dc))
    try:
        return nx.shortest_path(g, start, goal, weight="weight")
    except Exception:
        return None


def largest_component(mask: np.ndarray) -> np.ndarray:
    labeled, n = ndi.label(mask.astype(bool))
    if n <= 1:
        return mask.astype(bool)
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    keep = int(np.argmax(counts))
    return labeled == keep


def farthest_pair_on_mask(mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]] | None:
    rr, cc = np.where(mask)
    if len(rr) < 2:
        return None
    active = set(zip(rr.tolist(), cc.tolist()))
    g = nx.Graph()
    nbrs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for r, c in active:
        g.add_node((r, c))
        for dr, dc in nbrs:
            n = (r + dr, c + dc)
            if n in active:
                g.add_edge((r, c), n, weight=math.hypot(dr, dc))
    if g.number_of_nodes() < 2:
        return None
    start = next(iter(g.nodes))
    d0 = nx.single_source_dijkstra_path_length(g, start, weight="weight")
    a = max(d0, key=d0.get)
    d1 = nx.single_source_dijkstra_path_length(g, a, weight="weight")
    b = max(d1, key=d1.get)
    return a, b


def simplify_path_pixels(path: list[tuple[int, int]], min_step_px: float = 3.0) -> list[tuple[int, int]]:
    if len(path) < 2:
        return path
    out = [path[0]]
    last = np.array(path[0], dtype=float)
    for p in path[1:]:
        cur = np.array(p, dtype=float)
        if float(np.linalg.norm(cur - last)) >= min_step_px:
            out.append((int(cur[0]), int(cur[1])))
            last = cur
    if out[-1] != path[-1]:
        out.append(path[-1])
    return out


def segment_bresenham_in_mask(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> bool:
    h, w = mask.shape
    rr, cc = sk_line(int(r0), int(c0), int(r1), int(c1))
    if rr.size == 0:
        return False
    ok = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
    return bool(np.all(ok) and np.all(mask[rr, cc]))


def segment_h_in_mask(mask: np.ndarray, r: int, c0: int, c1: int) -> bool:
    r = int(r)
    c_lo, c_hi = int(min(c0, c1)), int(max(c0, c1))
    h, w = mask.shape
    if r < 0 or r >= h or c_lo < 0 or c_hi >= w:
        return False
    return bool(np.all(mask[r, c_lo : c_hi + 1]))


def segment_v_in_mask(mask: np.ndarray, c: int, r0: int, r1: int) -> bool:
    c = int(c)
    r_lo, r_hi = int(min(r0, r1)), int(max(r0, r1))
    h, w = mask.shape
    if c < 0 or c >= w or r_lo < 0 or r_hi >= h:
        return False
    return bool(np.all(mask[r_lo : r_hi + 1, c]))


def manhattan_l_length(sr: int, sc: int, tr: int, tc: int, corner_r: int, corner_c: int) -> float:
    return float(abs(sr - corner_r) + abs(sc - corner_c) + abs(tr - corner_r) + abs(tc - corner_c))


def minimal_turn_trunk_pixels(
    corridor: np.ndarray,
    s: tuple[int, int],
    t: tuple[int, int],
) -> tuple[list[tuple[int, int]], int] | None:
    """0 turns = straight (Bresenham), else one of two L-paths (axis-aligned). Pick fewest turns, then shortest."""
    sr, sc = int(s[0]), int(s[1])
    tr, tc = int(t[0]), int(t[1])
    candidates: list[tuple[int, float, list[tuple[int, int]]]] = []

    if segment_bresenham_in_mask(corridor, sr, sc, tr, tc):
        d = float(math.hypot(tr - sr, tc - sc))
        candidates.append((0, d, [(sr, sc), (tr, tc)]))

    # L1: corner (sr, tc) — horizontal along row sr, then vertical along col tc
    if segment_h_in_mask(corridor, sr, sc, tc) and segment_v_in_mask(corridor, tc, sr, tr):
        if (sr, tc) != (sr, sc) and (tr, tc) != (sr, tc):
            d = manhattan_l_length(sr, sc, tr, tc, sr, tc)
            candidates.append((1, d, [(sr, sc), (sr, tc), (tr, tc)]))

    # L2: corner (tr, sc) — vertical along col sc, then horizontal along row tr
    if segment_v_in_mask(corridor, sc, sr, tr) and segment_h_in_mask(corridor, tr, sc, tc):
        if (tr, sc) != (sr, sc) and (tr, tc) != (tr, sc):
            d = manhattan_l_length(sr, sc, tr, tc, tr, sc)
            candidates.append((1, d, [(sr, sc), (tr, sc), (tr, tc)]))

    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], x[1]))
    turns, _, pts = candidates[0]
    return pts, turns


def principal_axis_unit(slab: Polygon | MultiPolygon) -> np.ndarray | None:
    if slab is None or slab.is_empty:
        return None
    mrr = slab.minimum_rotated_rectangle
    pts = list(mrr.exterior.coords)
    if len(pts) < 5:
        return None
    edges: list[tuple[float, np.ndarray]] = []
    for i in range(4):
        p0 = np.array(pts[i], dtype=float)
        p1 = np.array(pts[i + 1], dtype=float)
        v = p1 - p0
        ln = float(np.linalg.norm(v))
        if ln > 1e-9:
            edges.append((ln, v / ln))
    if not edges:
        return None
    edges.sort(key=lambda x: x[0], reverse=True)
    return edges[0][1]


def snap_world_polyline_to_slab_axes(
    line: LineString,
    main: np.ndarray,
) -> LineString:
    """Manhattan snapping in slab (main, perpendicular) — fewer jagged diagonals."""
    if line is None or line.is_empty or len(list(line.coords)) < 2:
        return line
    mx = main / np.linalg.norm(main)
    bx = np.array([-mx[1], mx[0]], dtype=float)

    pts = [np.array(c, dtype=float) for c in line.coords]
    out: list[tuple[float, float]] = [tuple(pts[0])]
    min_step = 0.05

    for i in range(1, len(pts)):
        target = pts[i]
        cur = np.array(out[-1], dtype=float)
        delta = target - cur
        u = float(delta @ mx)
        v = float(delta @ bx)
        if abs(u) < min_step and abs(v) < min_step:
            continue
        if abs(u) >= abs(v):
            nxt = cur + mx * u
        else:
            nxt = cur + bx * v
        if float(np.linalg.norm(nxt - cur)) < min_step:
            continue
        if math.hypot(nxt[0] - out[-1][0], nxt[1] - out[-1][1]) >= min_step:
            out.append((float(nxt[0]), float(nxt[1])))

    if math.hypot(pts[-1][0] - out[-1][0], pts[-1][1] - out[-1][1]) >= min_step:
        out.append((float(pts[-1][0]), float(pts[-1][1])))

    # Merge collinear consecutive segments in (main, branch) sense
    reduced: list[tuple[float, float]] = []
    for p in out:
        p_arr = np.array(p, dtype=float)
        if len(reduced) < 2:
            reduced.append(p)
            continue
        p0 = np.array(reduced[-2], dtype=float)
        p1 = np.array(reduced[-1], dtype=float)
        u0, v0 = float((p1 - p0) @ mx), float((p1 - p0) @ bx)
        u1, v1 = float((p_arr - p1) @ mx), float((p_arr - p1) @ bx)
        dom0 = "u" if abs(u0) >= abs(v0) else "v"
        dom1 = "u" if abs(u1) >= abs(v1) else "v"
        if dom0 == dom1 and (abs(u1) + abs(v1) > 1e-9):
            reduced[-1] = p
        else:
            reduced.append(p)

    if len(reduced) < 2:
        return line
    return LineString(reduced)


def auto_reduce_segments(line: LineString) -> LineString:
    """Shrink vertex count with increasing Douglas–Peucker tolerance (no user param)."""
    if line is None or line.is_empty:
        return line
    coords0 = list(line.coords)
    if len(coords0) <= 2:
        return line
    for tol in np.linspace(0.15, 12.0, 28):
        c = line.simplify(float(tol), preserve_topology=False)
        cr = list(c.coords)
        if len(cr) < 2:
            continue
        cr[0] = coords0[0]
        cr[-1] = coords0[-1]
        cand = LineString(cr)
        nseg = len(cr) - 1
        if nseg <= 3:
            return cand
    return LineString([coords0[0], coords0[-1]])


def sample_line_to_rc(
    line: LineString,
    bounds: tuple[float, float, float, float],
    ppm: float,
    step_m: float = 0.35,
) -> list[tuple[int, int]]:
    if line is None or line.is_empty:
        return []
    n = max(2, int(math.ceil(float(line.length) / step_m)) + 1)
    out: list[tuple[int, int]] = []
    for i in range(n):
        d = min(float(line.length), float(line.length) * i / max(1, n - 1))
        p = line.interpolate(d)
        out.append(world_to_pixel(float(p.x), float(p.y), bounds, ppm))
    return out


def auto_endpoint_pixels_from_slab(
    slab: Polygon | MultiPolygon,
    slab_mask: np.ndarray,
    bounds: tuple[float, float, float, float],
    ppm: float,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Pick two far endpoints automatically from slab long axis."""
    mrr = slab.minimum_rotated_rectangle
    pts = list(mrr.exterior.coords)
    edges: list[tuple[float, np.ndarray]] = []
    for i in range(4):
        p0 = np.array(pts[i], dtype=float)
        p1 = np.array(pts[i + 1], dtype=float)
        v = p1 - p0
        ln = float(np.linalg.norm(v))
        if ln > 1e-9:
            edges.append((ln, v / ln))
    if not edges:
        # Fallback to mask corners if MRR fails for some reason.
        h, w = slab_mask.shape
        return (h // 2, max(0, int(w * 0.1))), (h // 2, min(w - 1, int(w * 0.9)))

    edges.sort(key=lambda x: x[0], reverse=True)
    main = edges[0][1]
    c = np.array([slab.centroid.x, slab.centroid.y], dtype=float)
    hull = np.array(list(slab.convex_hull.exterior.coords), dtype=float)
    rel = hull - c
    proj = rel @ main
    p0w = c + main * float(np.min(proj))
    p1w = c + main * float(np.max(proj))
    p0 = world_to_pixel(float(p0w[0]), float(p0w[1]), bounds, ppm)
    p1 = world_to_pixel(float(p1w[0]), float(p1w[1]), bounds, ppm)
    s = nearest_true(slab_mask, p0) or p0
    t = nearest_true(slab_mask, p1) or p1
    return s, t


def run() -> None:
    parser = argparse.ArgumentParser(description="Use SAM2 to segment corridor and build trunk between two user points.")
    parser.add_argument("--json", type=str, default="outputs/output/detected_geometry.json")
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--sam2-config", type=str, required=True, help="SAM2 config YAML path.")
    parser.add_argument("--sam2-checkpoint", type=str, required=True, help="SAM2 checkpoint .pt path.")
    parser.add_argument("--pixels-per-meter", type=float, default=14.0)
    parser.add_argument("--wall-buffer-m", type=float, default=0.25)
    parser.add_argument("--column-buffer-m", type=float, default=0.3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-preview", type=str, default=None)
    parser.add_argument(
        "--auto-points",
        action="store_true",
        help="Auto-select start/end points from slab geometry (no manual clicks).",
    )
    parser.add_argument(
        "--preview-floor-label",
        type=str,
        default="",
        help="If set, draws this text on --save-preview PNG",
    )
    args = parser.parse_args()

    if not _HAS_SAM2:
        print("SAM2 not importable. Install/clone segment-anything-2 and dependencies.", file=sys.stderr)
        sys.exit(2)

    # Print runtime backend first, before any step logs.
    resolved_device = args.device
    if resolved_device == "cuda" and not torch.cuda.is_available():
        resolved_device = "cpu"
    print(f"Runtime device requested: {args.device}")
    if resolved_device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Runtime device active   : GPU (CUDA) | {gpu_name}")
    else:
        print("Runtime device active   : CPU")
        if args.device == "cuda":
            print("Note: CUDA requested but not available; using CPU.")

    progress = ProgressTracker(total_steps=8)
    progress.start("Loading detected geometry JSON")
    inp = Path(args.json)
    if not inp.exists():
        print(f"Missing JSON: {inp}", file=sys.stderr)
        sys.exit(1)
    data = json.loads(inp.read_text(encoding="utf-8"))

    slab = normalize_polygon(geometry_from_json(data.get("unified_protected_floor_area")))
    walls = normalize_polygon(geometry_from_json(data.get("walls_all_union")))
    cols = normalize_polygon(geometry_from_json(data.get("columns_union")))
    if slab is None:
        print("Missing unified_protected_floor_area.", file=sys.stderr)
        sys.exit(1)
    progress.done(extra=f"json={inp}")

    progress.start("Rasterizing slab/walls/columns into prompt canvas")
    wall_geom = walls.buffer(args.wall_buffer_m) if walls is not None and not walls.is_empty else None
    col_geom = cols.buffer(args.column_buffer_m) if cols is not None and not cols.is_empty else None

    bounds = build_canvas_bounds(slab, margin_m=1.2)
    w = int(math.ceil((bounds[2] - bounds[0]) * args.pixels_per_meter))
    h = int(math.ceil((bounds[3] - bounds[1]) * args.pixels_per_meter))
    shape = (max(32, h), max(32, w))

    slab_mask = rasterize_polygon_mask(
        slab, bounds, shape, args.pixels_per_meter, bar_label="Rasterizing slab"
    )
    wall_mask = rasterize_polygon_mask(
        wall_geom, bounds, shape, args.pixels_per_meter, bar_label="Rasterizing walls"
    )
    col_mask = rasterize_polygon_mask(
        col_geom, bounds, shape, args.pixels_per_meter, bar_label="Rasterizing columns"
    )
    prompt_img = make_prompt_image(slab_mask, wall_mask, col_mask)
    progress.done(extra=f"canvas={shape[1]}x{shape[0]} px, slab_pixels={int(slab_mask.sum())}")

    progress.start("Selecting trunk endpoints")
    if args.auto_points:
        s_auto, t_auto = auto_endpoint_pixels_from_slab(slab, slab_mask, bounds, args.pixels_per_meter)
        p0_px = np.array([[float(s_auto[1]), float(s_auto[0])]], dtype=np.float32)
        p1_px = np.array([[float(t_auto[1]), float(t_auto[0])]], dtype=np.float32)
        progress.done(extra="auto points selected from slab long axis")
    else:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(prompt_img)
        ax.set_title("Click START wall, then END wall for trunk")
        ax.set_axis_off()
        pts = plt.ginput(2, timeout=0, show_clicks=True)
        plt.close(fig)
        if len(pts) != 2:
            print("Need exactly 2 clicks.", file=sys.stderr)
            sys.exit(1)
        p0_px = np.array([[float(pts[0][0]), float(pts[0][1])]], dtype=np.float32)
        p1_px = np.array([[float(pts[1][0]), float(pts[1][1])]], dtype=np.float32)
        progress.done(extra="received 2 clicks")

    point_coords = np.concatenate([p0_px, p1_px], axis=0)
    point_labels = np.array([1, 1], dtype=np.int32)

    device = resolved_device
    progress.start("Loading SAM2 model")
    model = build_sam2(args.sam2_config, args.sam2_checkpoint, device=device)
    predictor = SAM2ImagePredictor(model)
    predictor.set_image(prompt_img)
    progress.done(extra=f"config={args.sam2_config}")

    progress.start("Running SAM2 point-prompt inference")
    masks, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True,
    )

    if masks is None or len(masks) == 0:
        print("SAM2 returned no masks.", file=sys.stderr)
        sys.exit(1)

    best_idx = 0
    best_score = -1.0
    score_bar = StepBar("Scoring candidate masks", len(masks))
    for i, m in enumerate(masks):
        mbool = m.astype(bool) & slab_mask
        if mbool.sum() < 50:
            score_bar.update(i + 1)
            continue
        if args.auto_points:
            # In auto mode, prefer larger connected corridor masks.
            comp = largest_component(mbool)
            score = float(scores[i]) + 0.00001 * float(comp.sum())
        else:
            r0, c0 = int(round(p0_px[0, 1])), int(round(p0_px[0, 0]))
            r1, c1 = int(round(p1_px[0, 1])), int(round(p1_px[0, 0]))
            contains = int(mbool[r0, c0]) + int(mbool[r1, c1])
            score = float(scores[i]) + 0.25 * contains
        if score > best_score:
            best_score = score
            best_idx = i
        score_bar.update(i + 1)
    progress.done(
        extra=(
            f"chosen_mask={best_idx}, score={best_score:.4f}, raw={float(scores[best_idx]):.4f}"
        )
    )

    progress.start("Cleaning corridor mask and building skeleton graph")
    corridor = masks[best_idx].astype(bool) & slab_mask
    corridor &= ~wall_mask
    corridor = ndi.binary_opening(corridor, structure=np.ones((3, 3), dtype=bool))
    corridor = ndi.binary_closing(corridor, structure=np.ones((3, 3), dtype=bool))
    corridor = largest_component(corridor)
    if corridor.sum() == 0:
        print("Corridor mask empty after cleanup.", file=sys.stderr)
        sys.exit(1)

    if args.auto_points:
        pair = farthest_pair_on_mask(corridor)
        if pair is None:
            print("Auto endpoint detection failed on corridor mask.", file=sys.stderr)
            sys.exit(1)
        p0_rc, p1_rc = pair
        # keep predictor points in sync for metadata/debug output
        p0_px = np.array([[float(p0_rc[1]), float(p0_rc[0])]], dtype=np.float32)
        p1_px = np.array([[float(p1_rc[1]), float(p1_rc[0])]], dtype=np.float32)
    else:
        p0_rc = (int(round(p0_px[0, 1])), int(round(p0_px[0, 0])))
        p1_rc = (int(round(p1_px[0, 1])), int(round(p1_px[0, 0])))

    s = nearest_true(corridor, p0_rc)
    t = nearest_true(corridor, p1_rc)
    if s is None or t is None:
        print("Cannot snap endpoints to corridor mask.", file=sys.stderr)
        sys.exit(1)

    skel = skeletonize(corridor)
    g = skeleton_graph(skel, bar_label="Building skeleton graph")
    if g.number_of_nodes() < 2:
        print("Skeleton graph too small.", file=sys.stderr)
        sys.exit(1)

    s2 = nearest_true(skel, s)
    t2 = nearest_true(skel, t)
    if s2 is None or t2 is None:
        print("Cannot snap endpoints to skeleton.", file=sys.stderr)
        sys.exit(1)

    try:
        path = nx.shortest_path(g, s2, t2, weight="weight")
    except Exception:
        print("      No skeleton path; falling back to corridor-mask routing.")
        path = grid_shortest_path(corridor, s, t)
        if not path:
            print("No corridor path between selected endpoints.", file=sys.stderr)
            if args.auto_points:
                print("Tip: SAM2 selected a fragmented corridor mask; try a larger checkpoint/model.", file=sys.stderr)
            else:
                print("Tip: click both points inside the same connected corridor region.", file=sys.stderr)
            sys.exit(1)
    progress.done(
        extra=f"corridor_pixels={int(corridor.sum())}, skel_nodes={g.number_of_nodes()}, skel_edges={g.number_of_edges()}"
    )

    progress.start("Converting corridor path to minimal-turn trunk polyline")
    geom_route = minimal_turn_trunk_pixels(corridor, s, t)
    if geom_route is not None:
        pix_waypoints, turn_count = geom_route
        world_pts = [pixel_to_world(r, c, bounds, args.pixels_per_meter) for r, c in pix_waypoints]
        deduped: list[tuple[float, float]] = [world_pts[0]]
        for p in world_pts[1:]:
            if math.hypot(p[0] - deduped[-1][0], p[1] - deduped[-1][1]) > 1e-4:
                deduped.append(p)
        line = LineString(deduped)
        path = sample_line_to_rc(line, bounds, args.pixels_per_meter)
        route_note = f"straight_or_L_turns={turn_count}"
    else:
        original_path = list(path)
        path = simplify_path_pixels(path, min_step_px=3.0)
        if len(path) < 2:
            path = original_path if len(original_path) >= 2 else path
        if len(path) < 2:
            print("Path collapsed to a single point; try different settings.", file=sys.stderr)
            sys.exit(1)
        world = [pixel_to_world(r, c, bounds, args.pixels_per_meter) for r, c in path]
        line = LineString(world)
        main_ax = principal_axis_unit(slab)
        if main_ax is not None:
            line = snap_world_polyline_to_slab_axes(line, main_ax)
        line = auto_reduce_segments(line)
        route_note = "skeleton_snapped_to_axes"
        path = sample_line_to_rc(line, bounds, args.pixels_per_meter)
    if line.length <= 0.25:
        print("Generated trunk too short.", file=sys.stderr)
        sys.exit(1)

    line = line.simplify(0.02, preserve_topology=False)
    progress.done(
        extra=f"vertices={len(list(line.coords))}, length={line.length:.2f} m | {route_note}"
    )

    progress.start("Writing output JSON and preview")
    coords_final = list(line.coords)
    data["suggested_trunk_line"] = [list(c) for c in coords_final]
    data["trunk_endpoints"] = {
        "start_xy": list(coords_final[0]),
        "end_xy": list(coords_final[-1]),
        "source": "sam2_corridor_auto" if args.auto_points else "sam2_corridor_clicks",
    }
    meta = data.get("detection_meta") or {}
    meta["trunk_source"] = "sam2_corridor"
    meta["trunk_route_strategy"] = route_note
    meta["sam2_config"] = str(args.sam2_config)
    meta["sam2_checkpoint"] = str(args.sam2_checkpoint)
    data["detection_meta"] = meta

    outp = Path(args.output_json) if args.output_json else inp
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.save_preview:
        fig2, ax2 = plt.subplots(figsize=(12, 8))
        ax2.imshow(prompt_img)
        ax2.imshow(np.where(corridor, 1.0, np.nan), alpha=0.25, cmap="Greens")
        ax2.plot([p[1] for p in path], [p[0] for p in path], color="red", linewidth=2.0)
        ax2.set_title("SAM2 corridor + selected trunk")
        ax2.set_axis_off()
        lbl = (args.preview_floor_label or "").strip()
        if lbl:
            ax2.text(
                0.01,
                0.98,
                lbl,
                transform=ax2.transAxes,
                fontsize=9,
                fontweight="bold",
                color="#111111",
                va="top",
                ha="left",
                bbox={
                    "boxstyle": "round,pad=0.4",
                    "fc": "white",
                    "ec": "#ff2d2d",
                    "linewidth": 1.0,
                    "alpha": 0.95,
                },
                zorder=100,
            )
        Path(args.save_preview).parent.mkdir(parents=True, exist_ok=True)
        fig2.savefig(args.save_preview, dpi=180, bbox_inches="tight")
        plt.close(fig2)
    progress.done(extra=f"json={outp}" + (f", preview={args.save_preview}" if args.save_preview else ""))

    progress.finish()
    print(f"Updated trunk in: {outp}")
    print(f"Trunk length: {line.length:.2f} m")


if __name__ == "__main__":
    run()
