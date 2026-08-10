# Full Project Guide

This guide explains how to set up, run, and maintain the Archicad sprinkler layout project. It covers the local web app, command-line scripts, output folders, Revit export, the experimental GAN package, and the Git workflow used for publishing.

## What This Project Does

The repository contains local tools for turning IFC building data into sprinkler layout artifacts:

1. Read and analyze IFC storeys.
2. Detect floor geometry, walls, columns, stairs, and protected floor areas.
3. Generate or edit a main trunk route.
4. Generate sprinkler heads, branch lines, trunk segments, score reports, previews, and DXF overlays.
5. Export a Revit handoff package and optionally create an RVT through pyRevit/Revit.
6. Review and manage the workflow through a local FastAPI and React app.

The main user-facing path is the web app started by `run_sprinkler_app.py`. The lower-level CLI scripts in `sprinkler2/` are useful for debugging, reproducible runs, and batch work.

## Repository Layout

- `README.md` - short project overview and quick start.
- `docs/FULL_GUIDE.md` - this guide.
- `run_sprinkler_app.py` - starts the backend and frontend together.
- `sprinkler_app/` - FastAPI backend, project storage, job runner, and pipeline orchestration.
- `web/` - React/Vite frontend.
- `sprinkler2/` - current geometry detection, layout generation, scoring, trunk, SAM2, and Revit export scripts.
- `archicad/` - older/legacy IFC extraction and layout scripts plus a small sample IFC.
- `sprinkler_hd_gan/` - experimental pix2pixHD-style raster/GAN workflow.
- `src/` - older package/test surface kept for regression work.
- `outputs/` - generated artifacts and app project data. This folder is ignored by git.
- `input/` - local BIM inputs. This folder is ignored by git.
- `segment-anything-2/` - optional local third-party SAM2 checkout. This folder is ignored by git.

## Git And Output Policy

The repository is designed so source code is tracked and generated project data stays local.

Tracked:

- Python source files.
- React/Vite source files.
- Small sample IFC files already in source folders.
- Documentation.
- `web/package-lock.json`.

Ignored:

- `outputs/`
- `input/`
- `projects/`
- `output/`
- `output_*/`
- `.venv/`
- `.venv*/`
- `web/node_modules/`
- `web/dist/`
- `segment-anything-2/`
- Python caches and egg-info folders.
- GAN generated data in `sprinkler_hd_gan/data/`, `sprinkler_hd_gan/out/`, and `sprinkler_hd_gan/runs/`.

New generated work should go under `outputs/`. Do not commit large IFC, RVT, checkpoint, generated PNG, generated DXF, or run-output folders unless there is a specific reason and the file size is safe for GitHub.

The current remote is:

```powershell
git remote -v
```

Expected remote:

```text
origin  https://github.com/LMindadze/archicad.git
```

## Prerequisites

Recommended platform:

- Windows PowerShell.
- Python 3.11.
- Node.js and npm.
- Git.
- Revit 2027 and pyRevit only if you want RVT generation.
- NVIDIA/CUDA only for SAM2/GAN GPU paths.

Python packages used by the app and CLI include:

- `fastapi`
- `uvicorn`
- `python-multipart`
- `ifcopenshell`
- `shapely`
- `matplotlib`
- `numpy`
- `scipy`
- `scikit-image`
- `networkx`

The GAN package has its own `sprinkler_hd_gan/pyproject.toml` and includes PyTorch dependencies.

## Fresh Setup

From the repository root:

```powershell
cd F:\unified\archicad
```

Create and activate a Python environment:

```powershell
python -m venv .venv311
.\.venv311\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install backend and CLI dependencies:

```powershell
python -m pip install fastapi uvicorn python-multipart ifcopenshell shapely matplotlib numpy scipy scikit-image networkx
```

Install frontend dependencies:

```powershell
npm --prefix web install
```

Optional: install the experimental GAN package:

```powershell
python -m pip install -e .\sprinkler_hd_gan
```

Optional: install the GAN package with IFC extras:

```powershell
python -m pip install -e ".\sprinkler_hd_gan[ifc]"
```

## Run The Web App

Start both backend and frontend:

```powershell
python run_sprinkler_app.py
```

By default:

- Backend starts on `http://127.0.0.1:8000`.
- Frontend starts on `http://127.0.0.1:5173`, or the next free port if 5173 is busy.

The runner prints the exact URLs. Press `Ctrl+C` in the terminal to stop both processes.

Custom ports:

```powershell
python run_sprinkler_app.py --backend-port 8001 --frontend-port 5174
```

## Web App Workflow

The app workflow follows these stages:

1. Input
2. Analyze
3. Detect
4. Trunk
5. Sprinklers
6. Review
7. Export

### Approved V1 Sample Layout Seed

For the sample garage, the app can preserve a previously approved baseline sprinkler layout before it runs main trunk post-processing. This is how two machines stay visually consistent for the same IFC and default settings.

The preferred local seed path is:

```text
input/approved_v1_layout_result.json
```

This file is intentionally local-only because `input/` is ignored by git. To make a second PC match the original approved sample layout, copy the approved final layout JSON from the original PC to that path.

The approved sample seed is the final post-trunk layout from the main PC: 176 sprinkler heads, 189 branch lines, no secondary branches, and 6 `trunk_segments`.

Use the helper script to copy and validate the seed:

```powershell
python scripts\import_approved_v1_seed.py "D:\path\from\main-pc\layout_result.json"
```

On the original/main PC, this helper can search for the approved sample seed and install the best 176-head candidate:

```powershell
python scripts\find_approved_v1_seed.py F:\unified\archicad --heads 176 --branches 189 --install
```

Then copy `input\approved_v1_layout_result.json` from the main PC into the same path on the second PC.

You can also point to a seed explicitly for one session:

```powershell
$env:SPRINKLER_APPROVED_V1_LAYOUT = "C:\path\to\layout_result.json"
python run_sprinkler_app.py
```

When the seed is used, the run log contains `Using approved v1 final layout seed`. If the seed is missing, has the wrong counts/topology, or does not match the detected sample bounds, the app falls back to CP-SAT generation.

### 1. Input

Use one of these options:

- Upload an IFC file from your machine.
- Use the sample IFC through the `Sample` button.

Uploaded projects are stored under:

```text
outputs/projects/<project_id>/
```

### 2. Analyze

Click `Analyze` after selecting or uploading an IFC. The backend reads the IFC storeys and creates floor records with counts, elevations, names, and selection state.

The relevant API endpoint is:

```text
POST /api/projects/{project_id}/analyze
```

### 3. Detect

Select a floor and click `Detect`. The backend runs:

```text
sprinkler2/detect_parking_geometry.py
```

It creates:

- `detected_geometry.json`
- `detected_geometry_preview.png`

These are stored under the project folder in `outputs/projects/`.

### 4. Generate Or Edit Trunk

Click `Generate trunk` after detection. The backend builds a base layout, runs the automatic trunk tool, and stores a trunk override for the selected floor.

The main script is:

```text
sprinkler2/auto_main_trunk.py
```

You can edit the trunk visually in the preview. Trunk endpoint and segment edits are persisted through:

```text
PATCH /api/projects/{project_id}/floors/{floor_id}/trunk
```

### 5. Generate Sprinklers

Click `Generate sprinklers` after the trunk is ready. The backend runs the layout generation and scoring flow.

The main scripts are:

- `sprinkler2/generate_draft_layout.py`
- `sprinkler2/auto_main_trunk.py`
- `sprinkler2/score_layout.py`

Typical outputs include:

- `layout_result.json`
- `layout_preview.png`
- `layout_overlay.dxf`
- `score_report.json`

### 6. Review

Use the map preview to inspect:

- Protected floor area.
- Exclusions.
- Walls.
- Columns.
- Stairs.
- Trunk.
- Branches.
- Sprinkler heads.

The UI includes controls for edit/pan mode, zoom, fit, mirror X, mirror Y, orientation reset, and layer visibility.

### 7. Export Revit

Select the floors with completed layouts and click the export action. The backend creates a Revit handoff package and, when Revit/pyRevit are available, attempts RVT creation.

The relevant backend scripts are:

- `sprinkler2/export_revit_handoff.py`
- `sprinkler2/make_rvt_pyrevit.py`

Default Revit settings live in `sprinkler_app/storage.py`:

```text
revit_year = 2027
revit_template = F:\autodesk\RVT 2027\Templates\English\Systems-Default_Metric.rte
```

Update the Revit template path in the UI settings if your local template differs.

## Settings Reference

Default app settings are defined in `sprinkler_app/storage.py`.

Common settings:

- `hazard_preset` - hazard preset label used by the workflow.
- `head_type` - sprinkler head type label.
- `head_spacing` - target head spacing in meters.
- `branch_spacing` - target branch spacing in meters.
- `main_diameter` - main pipe diameter label.
- `branch_diameter` - branch pipe diameter label.
- `column_clearance` - clearance from columns in meters.
- `stair_clearance` - clearance from stairs in meters.
- `wall_clearance` - clearance from walls in meters.
- `min_obstacle_clearance` - extra clearance from exclusion boundaries.
- `routing_model` - `direct`, `steiner`, or `legacy`.
- `layout_model` - default is `cpsat`.
- `allow_secondary_branches` - enables secondary branch stubs.
- `cpsat_time_limit` - CP-SAT solve time limit in seconds.
- `cpsat_max_demand` - maximum demand sample count.
- `cpsat_min_head_spacing` - minimum head spacing for CP-SAT.
- `demand_step` - demand grid step in meters.
- `target_coverage` - target coverage ratio.
- `revit_year` - Revit version used by pyRevit.
- `revit_template` - local Revit template path.

## CLI Workflow: Single Floor

The command-line scripts are useful when you want direct artifact generation without the web app.

Start from the repo root:

```powershell
cd F:\unified\archicad
```

Detect IFC geometry:

```powershell
python sprinkler2\detect_parking_geometry.py --ifc input\your_model.ifc --storey "2. Story" --output-dir outputs\output
```

Generate a draft layout:

```powershell
python sprinkler2\generate_draft_layout.py --input-json outputs\output\detected_geometry.json --output-dir outputs\output
```

Generate an automatic main trunk from the draft layout:

```powershell
python sprinkler2\auto_main_trunk.py --detected-json outputs\output\detected_geometry.json --layout-json outputs\output\layout_result.json --output-dir outputs\output_main_trunk_only
```

Score a layout:

```powershell
python sprinkler2\score_layout.py --detected-json outputs\output\detected_geometry.json --layout-json outputs\output_main_trunk_only\layout_result.json --out-json outputs\output_main_trunk_only\score_report.json
```

Create a Revit handoff package without launching Revit:

```powershell
python sprinkler2\run_revit_export_pipeline.py --no-revit
```

Run the full Revit export pipeline when pyRevit/Revit are configured:

```powershell
python sprinkler2\run_revit_export_pipeline.py
```

## CLI Workflow: Multi-Floor With SAM2

The multi-floor script processes IFC storeys and can use SAM2 for corridor/trunk segmentation.

Dry-run to list storeys and output folders:

```powershell
python sprinkler2\run_multifloor_sprinkler_pipeline.py --ifc input\your_model.ifc --sam2-config configs\sam2.1\sam2.1_hiera_t.yaml --sam2-checkpoint segment-anything-2\checkpoints\sam2.1_hiera_tiny.pt --dry-run
```

Run selected storeys:

```powershell
python sprinkler2\run_multifloor_sprinkler_pipeline.py --ifc input\your_model.ifc --only-storeys "2. Story,3. Story" --sam2-config configs\sam2.1\sam2.1_hiera_t.yaml --sam2-checkpoint segment-anything-2\checkpoints\sam2.1_hiera_tiny.pt
```

Outputs default to:

```text
outputs/<project_name>/
```

Use `--out-root` to choose another local output parent.

## Revit Export Details

The handoff exporter writes CSV/JSON files and copies a pyRevit runner into the output package.

Typical handoff files:

- `revit_sprinkler_layout.json`
- `sprinkler_heads.csv`
- `pipe_runs.csv`
- `context_elements.csv`
- `make_rvt_pyrevit.py`
- `make_rvt.log`
- `sprinkler_layout_combined.rvt` when RVT creation succeeds.

The app sets these environment variables for the Revit run:

- `SPRINKLER_REVIT_LAYOUT_JSON`
- `SPRINKLER_REVIT_OUTPUT_RVT`
- `SPRINKLER_REVIT_TEMPLATE`
- `SPRINKLER_REVIT_LOG`

If RVT generation fails, inspect `make_rvt.log` in the relevant output folder.

## API Reference

Important backend endpoints:

- `GET /api/health` - backend health and project root.
- `GET /api/projects` - list local app projects.
- `POST /api/projects` - upload an IFC.
- `POST /api/projects/sample` - create a project from the bundled sample.
- `GET /api/projects/{project_id}` - load a project.
- `POST /api/projects/{project_id}/analyze` - analyze IFC storeys.
- `PATCH /api/projects/{project_id}/settings` - update project settings.
- `POST /api/projects/{project_id}/floors/{floor_id}/detect` - start floor detection.
- `POST /api/projects/{project_id}/floors/{floor_id}/trunk/generate` - start trunk generation.
- `PATCH /api/projects/{project_id}/floors/{floor_id}/trunk` - save a trunk override.
- `POST /api/projects/{project_id}/floors/{floor_id}/sprinklers/generate` - start sprinkler layout generation.
- `PATCH /api/projects/{project_id}/floors/{floor_id}/layout-edits` - save head edits.
- `POST /api/projects/{project_id}/exports/revit` - start Revit export.
- `GET /api/runs/{run_id}` - get run status and log.
- `POST /api/runs/{run_id}/cancel` - cancel a running job.

Run records are stored inside the app project folder under:

```text
outputs/projects/<project_id>/runs/<run_id>/
```

## Experimental GAN Workflow

The GAN package is under `sprinkler_hd_gan/`. It is separate from the main app workflow.

Install:

```powershell
python -m pip install -e .\sprinkler_hd_gan
```

Generate synthetic training pairs:

```powershell
cd sprinkler_hd_gan
sprinkler-hd-synth --out data\synthetic --train 200 --val 20
```

Train:

```powershell
sprinkler-hd-train --data data\synthetic --out runs\demo --epochs 5
```

Infer:

```powershell
sprinkler-hd-infer --checkpoint runs\demo\epoch_0005.pt --input data\synthetic\val\input\00000.png --meta data\synthetic\val\meta\00000.yaml --out out\pred.png
```

The generated `data/`, `runs/`, and `out/` folders are ignored by git.

## Useful Maintenance Commands

Check repository status:

```powershell
git status --short --branch
```

Run a Python syntax check:

```powershell
python -m compileall archicad sprinkler2 sprinkler_app run_sprinkler_app.py -q
```

Build the frontend:

```powershell
npm --prefix web run build
```

Push committed changes:

```powershell
git push
```

## Troubleshooting

### Frontend does not start

Run:

```powershell
npm --prefix web install
npm --prefix web run dev
```

If port 5173 is busy, `run_sprinkler_app.py` automatically tries the next free port.

### Backend import errors

Activate the Python environment and install the core dependencies:

```powershell
.\.venv311\Scripts\Activate.ps1
python -m pip install fastapi uvicorn python-multipart ifcopenshell shapely matplotlib numpy scipy scikit-image networkx
```

### IFC upload fails

Confirm the file has an `.ifc` extension and is accessible on disk. Large IFC files should stay in `input/` or another local-only folder.

### Detection fails

Try the CLI directly to see the full traceback:

```powershell
python sprinkler2\detect_parking_geometry.py --ifc input\your_model.ifc --storey "2. Story" --output-dir outputs\debug_detect
```

If the storey name is wrong, use the web app Analyze stage to inspect detected storey names.

### Layout generation is slow

Lower `cpsat_time_limit`, increase `demand_step`, or reduce the selected floor scope. Keep `outputs/` local because repeated tuning creates many artifacts.

### Revit export fails

Check:

- Revit version matches `revit_year`.
- `revit_template` points to an existing local `.rte`.
- pyRevit is installed and available under `%APPDATA%\pyRevit-Master\bin\pyrevit.exe` or on `PATH`.
- `make_rvt.log` in the output folder does not contain a traceback.

### GitHub push fails because of large files

Check for staged large files:

```powershell
git status --short
```

Generated folders should stay ignored. If a large generated file is staged by mistake, unstage it and add an ignore rule before committing.

## Recommended Daily Workflow

1. Pull latest source:

```powershell
git pull
```

2. Start the app:

```powershell
python run_sprinkler_app.py
```

3. Work through Input, Analyze, Detect, Trunk, Sprinklers, Review, and Export.

4. Keep generated files in `outputs/`.

5. Commit only source or documentation changes:

```powershell
git status --short
git add README.md docs\FULL_GUIDE.md
git commit -m "update guide"
git push
```
