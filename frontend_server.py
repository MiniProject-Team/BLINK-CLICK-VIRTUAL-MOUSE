from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
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
DEFAULT_CHAT_SYSTEM_PROMPT = (
    "You are a helpful assistant for the Blink-Click Virtual Mouse web app. "
    "Answer in 1-3 short sentences, plain text only."
)

_PROCESS_LOCK = threading.Lock()
_PROJECT_PROCESS: subprocess.Popen | None = None
_LAST_STATUS_MESSAGE = DEFAULT_STATUS_MESSAGE

_FAQ: tuple[tuple[tuple[str, ...], str], ...] = (
    (
        ("what is", "about", "project", "blink click", "virtual mouse"),
        "Blink-Click Virtual Mouse is a hands-free accessibility system that uses face tracking, blink clicks, and voice commands to control the computer.",
    ),
    (
        ("how it works", "how does", "working"),
        "The webcam tracks facial landmarks, maps head movement to cursor motion, and uses eye blinks for clicks while voice commands handle actions like open, type, and scroll.",
    ),
    (
        ("features", "capabilities", "what can you do"),
        "It supports head-tracked cursor control, blink-based clicking, voice commands with a wake word, and optional local AI planning through Ollama.",
    ),
    (
        ("voice", "wake word", "ashu"),
        "Say the wake word 'Ashu' before a command. You can change it with the WAKE_WORD environment variable.",
    ),
    (
        ("run", "start", "launch", "how to run"),
        "Run python main.py to start the system. Use python frontend_server.py for the launcher page.",
    ),
)


def _clean_text(text: str) -> str:
    cleaned = text.strip().lower()
    cleaned = re.sub(r"[^a-z0-9:/?&=._+\-\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _answer_faq(text: str) -> str | None:
    cleaned = _clean_text(text)
    if not cleaned:
        return None
    for keywords, answer in _FAQ:
        if any(keyword in cleaned for keyword in keywords):
            return answer
    if "blink" in cleaned and "click" in cleaned:
        return "Blink clicks use the eye aspect ratio to detect intentional blinks and translate them into left or right clicks."
    if "head" in cleaned or "cursor" in cleaned:
        return "Head movement is mapped to the screen using the nose tip landmark and smoothed to reduce jitter."
    if "voice command" in cleaned or "command" in cleaned:
        return "Voice commands start with the wake word, then you can say open, type, scroll, press keys, or ask a question."
    return None


def _ollama_chat(message: str) -> tuple[bool, str]:
    host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    model = os.environ.get("OLLAMA_MODEL", "phi3")
    timeout_s = float(os.environ.get("OLLAMA_TIMEOUT", "15"))
    prompt = f"{DEFAULT_CHAT_SYSTEM_PROMPT}\n\nUser: {message}\nAssistant:"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2},
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url=f"{host}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8", errors="ignore")
    except urllib.error.URLError:
        return False, "Ollama is not reachable. Start it with `ollama serve`."
    except Exception:
        return False, "Ollama did not respond in time. Try again or increase OLLAMA_TIMEOUT."

    try:
        wrapper = json.loads(body)
    except json.JSONDecodeError:
        return False, "The assistant returned an unreadable response."

    response_text = wrapper.get("response")
    if isinstance(response_text, str) and response_text.strip():
        cleaned = re.sub(r"\s+", " ", response_text.strip())
        return True, cleaned[:600]

    return False, "The assistant returned an empty response."


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


def _extract_voice_commands(max_items: int = 8) -> list[dict]:
    content = "\n".join(
        part
        for part in (
            _tail_log(STDERR_LOG, max_lines=200),
            _tail_log(STDOUT_LOG, max_lines=200),
        )
        if part
    )
    if not content:
        return []

    keywords = (
        "user",
        "command",
        "recognized",
        "transcript",
        "[voice]",
        "intent",
        "utterance",
        "said",
        "request",
    )
    candidates: list[dict] = []
    for line in content.splitlines():
        lower = line.lower()
        if not any(key in lower for key in keywords):
            continue

        time_match = re.match(r"^(?P<time>\d{2}:\d{2}:\d{2})\s+(?P<rest>.*)$", line)
        timestamp = ""
        message = line
        if time_match:
            timestamp = time_match.group("time")
            message = time_match.group("rest")

        for token in ("INFO", "WARNING", "ERROR", "DEBUG"):
            if token in message:
                message = message.split(token, 1)[1].strip()
                break

        message = (
            message.replace("[Assistant]", "")
            .replace("[User]", "")
            .replace("[Voice]", "")
            .strip(" -")
        )
        if not message:
            continue

        candidates.append({"time": timestamp, "text": message})

    if not candidates:
        return []

    seen: set[tuple[str, str]] = set()
    deduped: list[dict] = []
    for entry in reversed(candidates):
        key = (entry["time"], entry["text"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(entry)

    deduped.reverse()
    return deduped[-max_items:]


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

    def _read_json(self) -> dict:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {}

    def do_GET(self) -> None:
        if self.path == "/api/status":
            self._write_json(
                {
                    "running": _is_running(),
                    "message": _status_message(),
                }
            )
            return

        if self.path == "/api/voice-commands":
            if _is_running():
                commands = _extract_voice_commands()
                message = "Commands sourced from frontend_server.err"
            else:
                commands = []
                message = "Project is not running. Recent commands cleared."
            self._write_json(
                {
                    "commands": commands,
                    "message": message,
                }
            )
            return

        if self.path in {"/", "/index.html"}:
            self.path = "/index.html"
        return super().do_GET()

    def do_POST(self) -> None:
        if self.path == "/api/start":
            started, message = _start_project()
            self._write_json(
                {
                    "running": _is_running(),
                    "started": started,
                    "message": message,
                }
            )
            return

        if self.path == "/api/chat":
            payload = self._read_json()
            message = str(payload.get("message", "")).strip()
            if not message:
                self._write_json({"reply": "", "error": "Message is required."}, status=HTTPStatus.BAD_REQUEST)
                return

            faq_answer = _answer_faq(message)
            if faq_answer:
                self._write_json({"reply": faq_answer, "source": "faq"})
                return

            ok, reply = _ollama_chat(message)
            if ok:
                self._write_json({"reply": reply, "source": "ollama"})
                return

            self._write_json({"reply": "", "error": reply}, status=HTTPStatus.SERVICE_UNAVAILABLE)
            return

        self._write_json(
            {"message": "Not found."},
            status=HTTPStatus.NOT_FOUND,
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
