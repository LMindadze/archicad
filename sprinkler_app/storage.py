from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_ROOT = REPO_ROOT / "outputs"
PROJECTS_ROOT = OUTPUTS_ROOT / "projects"


DEFAULT_SETTINGS: dict[str, Any] = {
    "hazard_preset": "ordinary_group_1",
    "head_type": "dry_horizontal_sidewall",
    "head_spacing": 3.2,
    "branch_spacing": 3.8,
    "main_diameter": "DN100",
    "branch_diameter": "DN65",
    "column_clearance": 0.55,
    "stair_clearance": 0.8,
    "wall_clearance": 0.3,
    "min_obstacle_clearance": 0.2,
    "routing_model": "direct",
    "layout_model": "cpsat",
    "allow_secondary_branches": False,
    "cpsat_time_limit": 60.0,
    "cpsat_max_demand": 4000,
    "cpsat_min_head_spacing": 1.8288,
    "demand_step": 1.0,
    "target_coverage": 0.96,
    "revit_year": "2027",
    "revit_template": r"F:\autodesk\RVT 2027\Templates\English\Systems-Default_Metric.rte",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_slug(value: str, fallback: str = "project") -> str:
    raw = (value or "").strip() or fallback
    raw = re.sub(r'[<>:"/\\|?*\n\r\t]+', "_", raw)
    raw = re.sub(r"\s+", "_", raw)
    raw = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)
    return raw[:80].strip("._-") or fallback


def new_id(prefix: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{stamp}_{uuid4().hex[:8]}"


def ensure_projects_root() -> None:
    PROJECTS_ROOT.mkdir(parents=True, exist_ok=True)


def project_dir(project_id: str) -> Path:
    return PROJECTS_ROOT / project_id


def project_file(project_id: str) -> Path:
    return project_dir(project_id) / "project.json"


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_project(project_id: str) -> dict[str, Any]:
    data = read_json(project_file(project_id))
    if not data:
        raise FileNotFoundError(f"Project not found: {project_id}")
    return data


def save_project(project: dict[str, Any]) -> None:
    project["updated_at"] = now_iso()
    write_json(project_file(project["id"]), project)


def list_projects() -> list[dict[str, Any]]:
    ensure_projects_root()
    out: list[dict[str, Any]] = []
    for path in sorted(PROJECTS_ROOT.glob("*/project.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        data = read_json(path)
        if isinstance(data, dict):
            out.append(data)
    return out


def create_project_record(name: str, original_filename: str, source_path: Path) -> dict[str, Any]:
    project_id = new_id(safe_slug(Path(original_filename).stem, fallback="ifc"))
    pdir = project_dir(project_id)
    pdir.mkdir(parents=True, exist_ok=True)
    try:
        rel_source = str(source_path.relative_to(pdir))
    except ValueError:
        rel_source = source_path.name
    project = {
        "id": project_id,
        "name": name or Path(original_filename).stem,
        "original_filename": original_filename,
        "source_ifc": rel_source,
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "status": "created",
        "storeys": [],
        "settings": DEFAULT_SETTINGS.copy(),
        "runs": [],
    }
    save_project(project)
    return project


def run_dir(project_id: str, run_id: str) -> Path:
    return project_dir(project_id) / "runs" / run_id


def run_file(project_id: str, run_id: str) -> Path:
    return run_dir(project_id, run_id) / "run.json"


def save_run(run: dict[str, Any]) -> None:
    write_json(run_file(run["project_id"], run["id"]), run)


def load_run(run_id: str) -> dict[str, Any]:
    ensure_projects_root()
    matches = list(PROJECTS_ROOT.glob(f"*/runs/{run_id}/run.json"))
    if not matches:
        raise FileNotFoundError(f"Run not found: {run_id}")
    return read_json(matches[0])


def relative_project_path(project_id: str, path: Path) -> str:
    return str(path.resolve().relative_to(project_dir(project_id).resolve())).replace("\\", "/")


def file_url(project_id: str, path: Path) -> str:
    rel = relative_project_path(project_id, path)
    return f"/files/{project_id}/{rel}"


def find_floor(project: dict[str, Any], floor_id: str) -> dict[str, Any] | None:
    for floor in project.get("storeys", []):
        if floor.get("id") == floor_id:
            return floor
    return None
