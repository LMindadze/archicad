from __future__ import annotations

import shutil
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from sprinkler_app.pipeline import (
    PipelineCancelled,
    analyze_ifc,
    approved_v1_seed_candidates,
    approved_v1_seed_quality,
    detect_floor,
    export_revit_from_project,
    generate_floor_sprinklers,
    generate_floor_trunk,
    normalize_trunk_segments,
    ordered_trunk_points,
    patch_floor_layout_edits,
    run_project_pipeline,
)
from sprinkler_app.storage import (
    DEFAULT_SETTINGS,
    PROJECTS_ROOT,
    REPO_ROOT,
    create_project_record,
    file_url,
    find_floor,
    list_projects,
    load_project,
    load_run,
    new_id,
    now_iso,
    project_dir,
    run_dir,
    save_project,
    save_run,
    write_json,
)


app = FastAPI(title="Local IFC-to-Revit Sprinkler App")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:5174", "http://127.0.0.1:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
PROJECTS_ROOT.mkdir(parents=True, exist_ok=True)
app.mount("/files", StaticFiles(directory=str(PROJECTS_ROOT)), name="files")


jobs: dict[str, dict[str, Any]] = {}


def public_project(project: dict[str, Any]) -> dict[str, Any]:
    return project


def public_run(run: dict[str, Any]) -> dict[str, Any]:
    try:
        log_path = run_dir(run["project_id"], run["id"]) / "pipeline.log"
        run = dict(run)
        run["log"] = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else run.get("log", "")
    except Exception:
        pass
    return run


def append_log(project_id: str, run_id: str, message: str) -> None:
    path = run_dir(project_id, run_id) / "pipeline.log"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(message + "\n")


def update_project_run_summary(project_id: str, run_id: str) -> None:
    try:
        updated_project = load_project(project_id)
        updated_run = load_run(run_id)
        for item in updated_project.get("runs", []):
            if item.get("id") == run_id:
                item["status"] = updated_run.get("status")
                item["finished_at"] = updated_run.get("finished_at")
                item["artifact_count"] = len(updated_run.get("artifacts", []))
                break
        save_project(updated_project)
    except Exception:
        pass


def start_stage_run(
    project: dict[str, Any],
    *,
    stage: str,
    action: Any,
    floor_id: str | None = None,
    selected_floor_ids: list[str] | None = None,
    settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    run_id = new_id("run")
    selected = selected_floor_ids or ([floor_id] if floor_id else [])
    run = {
        "id": run_id,
        "project_id": project["id"],
        "stage": stage,
        "floor_id": floor_id,
        "status": "queued",
        "selected_floor_ids": selected,
        "settings": settings or {},
        "created_at": now_iso(),
        "started_at": None,
        "finished_at": None,
        "artifacts": [],
        "counts": {},
        "floor_results": [],
        "error": None,
    }
    rdir = run_dir(project["id"], run_id)
    rdir.mkdir(parents=True, exist_ok=True)
    write_json(rdir / "settings.json", {"project_settings": project.get("settings", {}), "run_settings": run["settings"]})
    save_run(run)
    project.setdefault("runs", []).insert(0, {"id": run_id, "stage": stage, "status": run["status"], "created_at": run["created_at"]})
    save_project(project)
    cancel_flag = {"cancel": False}
    runner_ref: dict[str, Any] = {"runner": None}
    jobs[run_id] = {"cancel": cancel_flag, "runner_ref": runner_ref}

    def worker() -> None:
        active_project = load_project(project["id"])
        active_run = load_run(run_id)
        active_run["status"] = "running"
        active_run["started_at"] = now_iso()
        save_run(active_run)
        append_log(project["id"], run_id, f"Stage started: {stage}")
        try:
            action(active_project, active_run, lambda msg: append_log(project["id"], run_id, msg), lambda: bool(cancel_flag["cancel"]), runner_ref)
            active_run = load_run(run_id)
            if active_run.get("status") == "running":
                active_run["status"] = "complete"
                active_run["finished_at"] = now_iso()
                save_run(active_run)
            append_log(project["id"], run_id, f"Stage finished: {load_run(run_id).get('status')}")
        except PipelineCancelled:
            active_run = load_run(run_id)
            active_run["status"] = "cancelled"
            active_run["finished_at"] = now_iso()
            save_run(active_run)
            append_log(project["id"], run_id, "Stage cancelled.")
        except Exception as exc:
            active_run = load_run(run_id)
            active_run["status"] = "failed"
            active_run["error"] = str(exc)
            active_run["finished_at"] = now_iso()
            save_run(active_run)
            append_log(project["id"], run_id, f"ERROR: {exc}")
        finally:
            jobs.pop(run_id, None)
            update_project_run_summary(project["id"], run_id)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return run


@app.get("/api/health")
def health() -> dict[str, Any]:
    approved_v1_seeds = [
        approved_v1_seed_quality(path)
        for path in approved_v1_seed_candidates()
    ]
    return {
        "ok": True,
        "projects_root": str(PROJECTS_ROOT),
        "revit_runner": str(REPO_ROOT / "sprinkler2" / "make_rvt_pyrevit.py"),
        "approved_v1_seeds": approved_v1_seeds,
    }


@app.get("/api/projects")
def get_projects() -> dict[str, Any]:
    return {"projects": [public_project(p) for p in list_projects()]}


@app.post("/api/projects")
def create_project(file: UploadFile = File(...), name: str | None = None) -> dict[str, Any]:
    if not file.filename or not file.filename.lower().endswith(".ifc"):
        raise HTTPException(status_code=400, detail="Upload an .ifc file.")
    temp_id = new_id("upload")
    pdir = project_dir(temp_id)
    pdir.mkdir(parents=True, exist_ok=True)
    source_path = pdir / "source.ifc"
    with source_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    project = create_project_record(name or Path(file.filename).stem, file.filename, source_path)
    final_dir = project_dir(project["id"])
    if final_dir != pdir:
        final_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source_path), str(final_dir / "source.ifc"))
        try:
            pdir.rmdir()
        except OSError:
            pass
        project["source_ifc"] = "source.ifc"
        save_project(project)
    return {"project": public_project(project)}


@app.post("/api/projects/sample")
def create_sample_project() -> dict[str, Any]:
    source = REPO_ROOT / "archicad" / "გარემო.ifc"
    if not source.exists():
        raise HTTPException(status_code=404, detail=f"Sample IFC not found: {source}")
    project = create_project_record("Sample Garage", source.name, project_dir("tmp") / "unused.ifc")
    pdir = project_dir(project["id"])
    shutil.copy2(source, pdir / "source.ifc")
    project["source_ifc"] = "source.ifc"
    save_project(project)
    return {"project": public_project(project)}


@app.get("/api/projects/{project_id}")
def get_project(project_id: str) -> dict[str, Any]:
    try:
        return {"project": public_project(load_project(project_id))}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/projects/{project_id}/analyze")
def analyze_project(project_id: str) -> dict[str, Any]:
    try:
        project = load_project(project_id)
        return {"project": public_project(analyze_ifc(project))}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.patch("/api/projects/{project_id}/settings")
def patch_settings(project_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        project = load_project(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    settings = {**DEFAULT_SETTINGS, **project.get("settings", {})}
    settings.update(payload.get("settings", payload))
    project["settings"] = settings
    save_project(project)
    return {"project": public_project(project)}


@app.post("/api/projects/{project_id}/floors/{floor_id}/detect")
def detect_project_floor(project_id: str, floor_id: str) -> dict[str, Any]:
    try:
        project = load_project(project_id)
        run = start_stage_run(
            project,
            stage="detect",
            floor_id=floor_id,
            action=lambda active_project, active_run, log, should_cancel, runner_ref: detect_floor(active_project, floor_id, log=log),
        )
        return {"run": public_run(run)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/projects/{project_id}/floors/{floor_id}/trunk/generate")
def generate_project_floor_trunk(project_id: str, floor_id: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    try:
        project = load_project(project_id)
        settings = (payload or {}).get("settings") or {}
        run = start_stage_run(
            project,
            stage="trunk",
            floor_id=floor_id,
            settings=settings,
            action=lambda active_project, active_run, log, should_cancel, runner_ref: generate_floor_trunk(
                active_project,
                floor_id,
                settings=active_run.get("settings") or {},
                log=log,
                should_cancel=should_cancel,
                runner_ref=runner_ref,
            ),
        )
        return {"run": public_run(run)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.patch("/api/projects/{project_id}/floors/{floor_id}/trunk")
def patch_trunk(project_id: str, floor_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        project = load_project(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    floor = find_floor(project, floor_id)
    if floor is None:
        raise HTTPException(status_code=404, detail=f"Unknown floor: {floor_id}")
    override = payload.get("trunk_override") or payload
    diameter = str((project.get("settings") or {}).get("main_diameter") or DEFAULT_SETTINGS["main_diameter"])
    segments = normalize_trunk_segments(override.get("segments") or [], diameter)
    line = ordered_trunk_points(segments, override.get("start"), override.get("end")) if segments else []
    if len(line) < 2:
        line = [[float(override["start"][0]), float(override["start"][1])], [float(override["end"][0]), float(override["end"][1])]]
    floor["trunk_override"] = {
        "start": line[0],
        "end": line[-1],
        "source": "user",
    }
    if segments:
        floor["trunk_override"]["segments"] = segments
        floor["trunk_override"]["main_trunk_line"] = line
        trunk_state = dict(floor.get("trunk") or {})
        trunk_state["segments"] = segments
        trunk_state["main_trunk_line"] = line
        trunk_state["source"] = "user_override"
        floor["trunk"] = trunk_state
    if floor.get("latest_layout_json"):
        floor["status"] = "edited"
    elif floor.get("detected_json"):
        floor["status"] = "trunk_ready"
    save_project(project)
    return {"floor": floor, "project": public_project(project)}


@app.post("/api/projects/{project_id}/floors/{floor_id}/sprinklers/generate")
def generate_project_floor_sprinklers(project_id: str, floor_id: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    try:
        project = load_project(project_id)
        settings = (payload or {}).get("settings") or {}
        run = start_stage_run(
            project,
            stage="sprinklers",
            floor_id=floor_id,
            settings=settings,
            action=lambda active_project, active_run, log, should_cancel, runner_ref: generate_floor_sprinklers(
                active_project,
                floor_id,
                settings=active_run.get("settings") or {},
                log=log,
                should_cancel=should_cancel,
                runner_ref=runner_ref,
            ),
        )
        return {"run": public_run(run)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.patch("/api/projects/{project_id}/floors/{floor_id}/layout-edits")
def patch_layout_edits(project_id: str, floor_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        project = load_project(project_id)
        floor = patch_floor_layout_edits(project, floor_id, payload.get("edits") or payload)
        return {"floor": floor, "project": public_project(project)}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/projects/{project_id}/exports/revit")
def export_project_revit(project_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        project = load_project(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    selected_floor_ids = payload.get("selected_floor_ids") or [f["id"] for f in project.get("storeys", []) if f.get("selected")]
    if not selected_floor_ids:
        raise HTTPException(status_code=400, detail="Select at least one floor.")
    run = start_stage_run(
        project,
        stage="export",
        selected_floor_ids=selected_floor_ids,
        settings=payload.get("settings") or {},
        action=lambda active_project, active_run, log, should_cancel, runner_ref: export_revit_from_project(
            active_project,
            active_run,
            active_run.get("selected_floor_ids") or [],
            log,
            should_cancel,
            runner_ref=runner_ref,
        ),
    )
    return {"run": public_run(run)}


@app.post("/api/projects/{project_id}/runs")
def create_run(project_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        project = load_project(project_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    selected_floor_ids = payload.get("selected_floor_ids") or [f["id"] for f in project.get("storeys", []) if f.get("selected")]
    if not selected_floor_ids:
        raise HTTPException(status_code=400, detail="Select at least one floor.")
    run_id = new_id("run")
    run = {
        "id": run_id,
        "project_id": project_id,
        "status": "queued",
        "selected_floor_ids": selected_floor_ids,
        "settings": payload.get("settings") or {},
        "created_at": now_iso(),
        "started_at": None,
        "finished_at": None,
        "artifacts": [],
        "counts": {},
        "floor_results": [],
        "error": None,
    }
    rdir = run_dir(project_id, run_id)
    rdir.mkdir(parents=True, exist_ok=True)
    write_json(rdir / "settings.json", {"project_settings": project.get("settings", {}), "run_settings": run["settings"]})
    save_run(run)
    project.setdefault("runs", []).insert(0, {"id": run_id, "status": run["status"], "created_at": run["created_at"]})
    save_project(project)
    cancel_flag = {"cancel": False}
    runner_ref: dict[str, Any] = {"runner": None}
    jobs[run_id] = {"cancel": cancel_flag, "runner_ref": runner_ref}

    def worker() -> None:
        active_project = load_project(project_id)
        active_run = load_run(run_id)
        active_run["status"] = "running"
        active_run["started_at"] = now_iso()
        save_run(active_run)
        append_log(project_id, run_id, "Run started.")
        try:
            result = run_project_pipeline(
                active_project,
                active_run,
                lambda msg: append_log(project_id, run_id, msg),
                lambda: bool(cancel_flag["cancel"]),
                runner_ref=runner_ref,
            )
            append_log(project_id, run_id, f"Run finished: {result['status']}")
        except Exception as exc:
            append_log(project_id, run_id, f"Run failed: {exc}")
        finally:
            jobs.pop(run_id, None)
            try:
                updated_project = load_project(project_id)
                updated_run = load_run(run_id)
                for item in updated_project.get("runs", []):
                    if item.get("id") == run_id:
                        item["status"] = updated_run.get("status")
                        item["finished_at"] = updated_run.get("finished_at")
                        item["artifact_count"] = len(updated_run.get("artifacts", []))
                        break
                save_project(updated_project)
            except Exception:
                pass

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return {"run": public_run(run)}


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    try:
        return {"run": public_run(load_run(run_id))}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/runs/{run_id}/cancel")
def cancel_run(run_id: str) -> dict[str, Any]:
    job = jobs.get(run_id)
    if not job:
        try:
            run = load_run(run_id)
            return {"run": public_run(run), "cancelled": False}
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
    job["cancel"]["cancel"] = True
    runner = job["runner_ref"].get("runner")
    if runner is not None:
        runner.terminate()
    append_log(load_run(run_id)["project_id"], run_id, "Cancel requested.")
    return {"run": public_run(load_run(run_id)), "cancelled": True}
