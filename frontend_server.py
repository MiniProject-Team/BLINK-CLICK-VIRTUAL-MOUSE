from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT_DIR / "frontend"
DIST_DIR = FRONTEND_DIR / "dist"
STATIC_DIR = DIST_DIR if DIST_DIR.exists() else FRONTEND_DIR
PORT = int(os.environ.get("FRONTEND_PORT", "3000"))
STDOUT_LOG = ROOT_DIR / "frontend_server.out"
STDERR_LOG = ROOT_DIR / "frontend_server.err"
DEFAULT_STATUS_MESSAGE = "Launcher ready. Press Start Project to begin."

_PROCESS_LOCK = threading.Lock()
_PROJECT_PROCESS: subprocess.Popen | None = None
_LAST_STATUS_MESSAGE = DEFAULT_STATUS_MESSAGE


def _is_running() -> bool:
    return _PROJECT_PROCESS is not None and _PROJECT_PROCESS.poll() is None


def _tail_log(path: Path, max_lines: int = 12) -> str:
    try:
        content = path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""

    if not content:
        return ""

    return "\n".join(content.splitlines()[-max_lines:])


def _status_message() -> str:
    global _LAST_STATUS_MESSAGE

    if _is_running():
        return "Project is running."

    if _PROJECT_PROCESS is None:
        return _LAST_STATUS_MESSAGE

    returncode = _PROJECT_PROCESS.poll()
    if returncode is None:
        return "Project is running."

    error_tail = _tail_log(STDERR_LOG)
    output_tail = _tail_log(STDOUT_LOG)
    details = error_tail or output_tail

    if details:
        last_line = details.splitlines()[-1].strip()
        _LAST_STATUS_MESSAGE = f"Project exited with code {returncode}: {last_line}"
    else:
        _LAST_STATUS_MESSAGE = (
            f"Project exited with code {returncode}. "
            f"Check {STDERR_LOG.name} and {STDOUT_LOG.name}."
        )

    return _LAST_STATUS_MESSAGE


def _start_project() -> tuple[bool, str]:
    global _PROJECT_PROCESS, _LAST_STATUS_MESSAGE

    with _PROCESS_LOCK:
        if _is_running():
            return False, "Project is already running."

        creationflags = 0
        if os.name == "nt":
            creationflags = (
                subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
            )

        python_executable = sys.executable
        if not python_executable:
            python_executable = "python"

        child_env = os.environ.copy()
        # Force UTF-8 stdio for detached child processes on Windows.
        # Without this, Unicode banner output in main.py can crash with
        # UnicodeEncodeError when no interactive UTF-8 console is attached.
        child_env.setdefault("PYTHONUTF8", "1")
        child_env.setdefault("PYTHONIOENCODING", "utf-8")

        stdout_log = open(STDOUT_LOG, "w", encoding="utf-8", errors="replace")
        stderr_log = open(STDERR_LOG, "w", encoding="utf-8", errors="replace")
        try:
            _PROJECT_PROCESS = subprocess.Popen(
                [python_executable, "main.py"],
                cwd=str(ROOT_DIR),
                stdin=subprocess.DEVNULL,
                stdout=stdout_log,
                stderr=stderr_log,
                env=child_env,
                creationflags=creationflags,
                close_fds=True,
            )
        except OSError as exc:
            _PROJECT_PROCESS = None
            _LAST_STATUS_MESSAGE = f"Could not start project: {exc}"
            return False, _LAST_STATUS_MESSAGE
        finally:
            stdout_log.close()
            stderr_log.close()

        # Give the child a moment to fail fast so the UI can show a useful error.
        time.sleep(1.0)
        if not _is_running():
            return False, _status_message()

        _LAST_STATUS_MESSAGE = (
            "Project started. The camera window should appear shortly."
        )
        return True, _LAST_STATUS_MESSAGE


class FrontendHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def _write_json(self, payload: dict, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/api/status":
            self._write_json(
                {
                    "running": _is_running(),
                    "message": _status_message(),
                }
            )
            return

        if self.path in {"/", "/index.html"}:
            self.path = "/index.html"
        return super().do_GET()

    def do_POST(self) -> None:
        if self.path != "/api/start":
            self._write_json(
                {"message": "Not found."},
                status=HTTPStatus.NOT_FOUND,
            )
            return

        started, message = _start_project()
        self._write_json(
            {
                "running": _is_running(),
                "started": started,
                "message": message,
            }
        )

    def log_message(self, format: str, *args) -> None:
        print(f"[frontend] {format % args}")


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", PORT), FrontendHandler)
    print(f"Frontend launcher running at http://127.0.0.1:{PORT}")
    print(f"Serving files from: {STATIC_DIR}")
    server.serve_forever()


if __name__ == "__main__":
    main()
