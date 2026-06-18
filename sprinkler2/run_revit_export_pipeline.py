from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PYREVIT = Path(os.environ.get("APPDATA", "")) / "pyRevit-Master" / "bin" / "pyrevit.exe"
DEFAULT_TEMPLATE = Path(r"F:\autodesk\RVT 2027\Templates\English\Systems-Default_Metric.rte")


def run(cmd: list[str], *, cwd: Path = REPO_ROOT, env: dict[str, str] | None = None) -> None:
    print("\n> " + " ".join(cmd), flush=True)
    result = subprocess.run(cmd, cwd=str(cwd), env=env)
    if result.returncode != 0:
        raise RuntimeError("Command failed with exit {0}: {1}".format(result.returncode, " ".join(cmd)))


def existing_path(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError("{0} not found: {1}".format(label, path))
    return path


def resolve_pyrevit(explicit: str | None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    candidates.append(DEFAULT_PYREVIT)
    path_env = os.environ.get("PATH", "")
    for entry in path_env.split(os.pathsep):
        if entry.strip():
            candidates.append(Path(entry) / "pyrevit.exe")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("pyrevit.exe not found. Pass --pyrevit-exe or install pyRevit.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the current sprinkler v1 -> Revit handoff -> RVT creation pipeline.",
    )
    parser.add_argument("--detected-json", default="outputs/output/detected_geometry.json")
    parser.add_argument("--base-layout-json", default="outputs/output/layout_result.json")
    parser.add_argument("--v1-layout-dir", default="outputs/output_v1_main_trunk_connected")
    parser.add_argument("--v1-layout-json", default=None, help="Defaults to <v1-layout-dir>/layout_result.json.")
    parser.add_argument("--output-dir", default="outputs/output_v1_revit_ready")
    parser.add_argument("--rvt-output", default=None, help="Defaults to <output-dir>/sprinkler_layout_from_export_pipeline.rvt.")
    parser.add_argument("--template", default=str(DEFAULT_TEMPLATE))
    parser.add_argument("--revit-year", default="2027")
    parser.add_argument("--pyrevit-exe", default=None)
    parser.add_argument("--refresh-base-layout", action="store_true", help="Regenerate outputs/output/layout_result.json from detected JSON.")
    parser.add_argument("--refresh-v1-layout", action="store_true", help="Regenerate the v1 main-trunk-connected layout.")
    parser.add_argument("--no-revit", action="store_true", help="Stop after Revit handoff export; do not create RVT.")
    args = parser.parse_args()

    detected_json = existing_path((REPO_ROOT / args.detected_json).resolve(), "Detected geometry JSON")
    base_layout_json = (REPO_ROOT / args.base_layout_json).resolve()
    v1_layout_dir = (REPO_ROOT / args.v1_layout_dir).resolve()
    v1_layout_json = (REPO_ROOT / args.v1_layout_json).resolve() if args.v1_layout_json else v1_layout_dir / "layout_result.json"
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    rvt_output = (REPO_ROOT / args.rvt_output).resolve() if args.rvt_output else output_dir / "sprinkler_layout_from_export_pipeline.rvt"
    template = Path(args.template).resolve()

    if args.refresh_base_layout:
        run(
            [
                sys.executable,
                str(REPO_ROOT / "sprinkler2" / "generate_draft_layout.py"),
                "--input-json",
                str(detected_json),
                "--output-dir",
                str(base_layout_json.parent),
            ]
        )
    else:
        existing_path(base_layout_json, "Base layout JSON")

    if args.refresh_v1_layout:
        run(
            [
                sys.executable,
                str(REPO_ROOT / "sprinkler2" / "auto_main_trunk.py"),
                "--detected-json",
                str(detected_json),
                "--layout-json",
                str(base_layout_json),
                "--output-dir",
                str(v1_layout_dir),
            ]
        )
    existing_path(v1_layout_json, "V1 layout JSON")

    run(
        [
            sys.executable,
            str(REPO_ROOT / "sprinkler2" / "export_revit_handoff.py"),
            "--layout-json",
            str(v1_layout_json),
            "--detected-json",
            str(detected_json),
            "--output-dir",
            str(output_dir),
        ]
    )

    if args.no_revit:
        print("\nRevit handoff generated: {0}".format(output_dir))
        return

    pyrevit = resolve_pyrevit(args.pyrevit_exe)
    runner = existing_path(output_dir / "make_rvt_pyrevit.py", "Revit runner script")
    if not template.exists():
        print("Warning: template not found; runner will fall back only if its default exists: {0}".format(template), flush=True)

    env = os.environ.copy()
    env["SPRINKLER_REVIT_LAYOUT_JSON"] = str(output_dir / "revit_sprinkler_layout.json")
    env["SPRINKLER_REVIT_OUTPUT_RVT"] = str(rvt_output)
    env["SPRINKLER_REVIT_TEMPLATE"] = str(template)
    env["SPRINKLER_REVIT_LOG"] = str(output_dir / "make_rvt.log")
    run(
        [
            str(pyrevit),
            "run",
            str(runner),
            "--revit={0}".format(args.revit_year),
            "--purge",
        ],
        env=env,
    )

    log_path = output_dir / "make_rvt.log"
    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    if "Traceback" in log_text or "Failures: 0" not in log_text or "Saved RVT:" not in log_text:
        raise RuntimeError("Revit run did not complete cleanly. Check log: {0}".format(log_path))

    existing_path(rvt_output, "Output RVT")
    print("\nPipeline complete.")
    print("- RVT: {0}".format(rvt_output))
    print("- Log: {0}".format(output_dir / "make_rvt.log"))


if __name__ == "__main__":
    main()
