from __future__ import annotations

import argparse
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def start_process(command: list[str], cwd: Path) -> subprocess.Popen[bytes]:
    return subprocess.Popen(command, cwd=str(cwd))


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

    backend = start_process(
        [
            sys.executable,
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
    frontend = start_process(
        [
            "npm",
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
