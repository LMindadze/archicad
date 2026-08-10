from __future__ import annotations

import argparse
import os
import signal
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REQUIRED_BACKEND_MODULES = ("ifcopenshell", "shapely", "ortools", "uvicorn")


def start_process(command: list[str], cwd: Path) -> subprocess.Popen[bytes]:
    return subprocess.Popen(command, cwd=str(cwd))


def venv_python(venv_name: str) -> Path:
    scripts = "Scripts" if sys.platform.startswith("win") else "bin"
    exe = "python.exe" if sys.platform.startswith("win") else "python"
    return ROOT / venv_name / scripts / exe


def backend_python_candidates() -> list[Path]:
    candidates: list[Path] = []
    env_python = os.environ.get("SPRINKLER_BACKEND_PYTHON")
    if env_python:
        candidates.append(Path(env_python).expanduser())
    candidates.extend([venv_python(".venv311"), venv_python(".venv"), venv_python(".venv312"), Path(sys.executable)])

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        try:
            key = str(candidate.resolve())
        except OSError:
            key = str(candidate)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def python_has_backend_deps(python_exe: Path) -> bool:
    if not python_exe.exists():
        return False
    probe = "; ".join(f"import {module}" for module in REQUIRED_BACKEND_MODULES)
    result = subprocess.run(
        [str(python_exe), "-c", probe],
        cwd=str(ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def find_backend_python() -> Path:
    checked: list[str] = []
    for candidate in backend_python_candidates():
        checked.append(str(candidate))
        if python_has_backend_deps(candidate):
            return candidate
    modules = ", ".join(REQUIRED_BACKEND_MODULES)
    raise RuntimeError(f"No Python interpreter with backend modules ({modules}) was found. Checked: {', '.join(checked)}")


def find_npm_command() -> str:
    command = "npm.cmd" if sys.platform.startswith("win") else "npm"
    found = shutil.which(command)
    if found:
        return found
    if sys.platform.startswith("win"):
        common = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "nodejs" / "npm.cmd"
        if common.exists():
            return str(common)
    return command


def find_free_port(start: int) -> int:
    port = start
    while port < start + 100:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
        port += 1
    raise RuntimeError(f"No free local port found from {start} to {start + 99}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the local IFC-to-Revit sprinkler app.")
    parser.add_argument("--backend-port", type=int, default=8000)
    parser.add_argument("--frontend-port", type=int, default=5173)
    args = parser.parse_args()

    frontend_port = find_free_port(args.frontend_port)

    backend_python = find_backend_python()
    backend = start_process(
        [
            str(backend_python),
            "-m",
            "uvicorn",
            "sprinkler_app.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(args.backend_port),
        ],
        ROOT,
    )
    npm_command = find_npm_command()
    frontend = start_process(
        [
            npm_command,
            "--prefix",
            "web",
            "run",
            "dev",
            "--",
            "--host",
            "127.0.0.1",
            "--port",
            str(frontend_port),
        ],
        ROOT,
    )

    children = [backend, frontend]

    def stop_children(*_: object) -> None:
        for child in children:
            if child.poll() is None:
                child.terminate()
        time.sleep(0.8)
        for child in children:
            if child.poll() is None:
                child.kill()

    signal.signal(signal.SIGINT, stop_children)
    signal.signal(signal.SIGTERM, stop_children)

    print(f"Frontend: http://127.0.0.1:{frontend_port}")
    print(f"Backend:  http://127.0.0.1:{args.backend_port}")
    print("Press Ctrl+C to stop both servers.")

    try:
        while any(child.poll() is None for child in children):
            time.sleep(0.25)
    finally:
        stop_children()

    return max((child.returncode or 0 for child in children), default=0)


if __name__ == "__main__":
    raise SystemExit(main())
