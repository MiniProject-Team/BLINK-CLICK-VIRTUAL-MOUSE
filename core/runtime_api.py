"""Local control bridge between the launcher page and the desktop engine."""

from __future__ import annotations

import json
import logging
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable

logger = logging.getLogger(__name__)


class RuntimeCommandServer:
    """Expose the running engine on localhost for the bundled frontend only."""

    def __init__(
        self,
        *,
        port: int,
        submit_voice_command: Callable[[str], bool],
        set_microphone_enabled: Callable[[bool], bool],
        get_status: Callable[[], dict],
    ) -> None:
        self.port = port
        self._submit_voice_command = submit_voice_command
        self._set_microphone_enabled = set_microphone_enabled
        self._get_status = get_status
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> bool:
        bridge = self

        class Handler(BaseHTTPRequestHandler):
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
                    return {}
                if length <= 0 or length > 4096:
                    return {}
                try:
                    payload = json.loads(self.rfile.read(length).decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    return {}
                return payload if isinstance(payload, dict) else {}

            def do_GET(self) -> None:
                if self.path == "/health":
                    self._write_json(bridge._get_status())
                    return
                self._write_json({"message": "Not found."}, HTTPStatus.NOT_FOUND)

            def do_POST(self) -> None:
                payload = self._read_json()
                if self.path == "/voice-command":
                    text = str(payload.get("text", "")).strip()
                    if not text or len(text) > 600:
                        self._write_json(
                            {"accepted": False, "message": "A short voice command is required."},
                            HTTPStatus.BAD_REQUEST,
                        )
                        return
                    accepted = bridge._submit_voice_command(text)
                    self._write_json(
                        {
                            "accepted": accepted,
                            "message": (
                                "Voice command queued."
                                if accepted
                                else "Wake word required, or the voice engine is unavailable."
                            ),
                        },
                        HTTPStatus.ACCEPTED if accepted else HTTPStatus.SERVICE_UNAVAILABLE,
                    )
                    return

                if self.path == "/microphone":
                    enabled = payload.get("enabled")
                    if not isinstance(enabled, bool):
                        self._write_json(
                            {"message": "The enabled field must be true or false."},
                            HTTPStatus.BAD_REQUEST,
                        )
                        return
                    available = bridge._set_microphone_enabled(enabled)
                    self._write_json(
                        {
                            "available": available,
                            "enabled": enabled if available else False,
                            "message": "Microphone updated." if available else "Voice engine is unavailable.",
                        },
                        HTTPStatus.OK if available else HTTPStatus.SERVICE_UNAVAILABLE,
                    )
                    return

                self._write_json({"message": "Not found."}, HTTPStatus.NOT_FOUND)

            def log_message(self, format: str, *args) -> None:
                logger.debug("Runtime API: %s", format % args)

        try:
            self._server = ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        except OSError as exc:
            logger.warning("Runtime control bridge unavailable on port %d: %s", self.port, exc)
            return False

        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        logger.info("Runtime control bridge listening on http://127.0.0.1:%d", self.port)
        return True

    def stop(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._server = None
        self._thread = None
