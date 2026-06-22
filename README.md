# Archicad Sprinkler Layout Tools

Local tooling for extracting IFC geometry, generating sprinkler layouts, reviewing them in a web UI, and exporting Revit handoff artifacts.

## Project Layout

- `archicad/` - IFC geometry extraction and legacy layout scripts.
- `sprinkler2/` - layout generation, scoring, trunk selection, and Revit export scripts.
- `sprinkler_app/` - FastAPI backend for the local review/generation app.
- `web/` - React/Vite frontend for the app.
- `sprinkler_hd_gan/` - experimental raster/GAN sprinkler layout tooling.
- `outputs/` - generated artifacts and app project runs. This folder is local-only and ignored by git.

Large local BIM inputs are kept in `input/`, which is also ignored by git. Keep reusable small samples in source folders only when they are safe to publish.

## Run The App

```powershell
python run_sprinkler_app.py
```

The runner starts the backend and frontend, then prints the local URLs.

For setup, app usage, CLI workflows, Revit export, output handling, and troubleshooting, see [docs/FULL_GUIDE.md](docs/FULL_GUIDE.md).

## Output Policy

Generated data should go under `outputs/`. The app stores project runs under `outputs/projects/`, and CLI defaults now write to `outputs/output*` paths instead of top-level `output*` folders.
