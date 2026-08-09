from __future__ import annotations

import logging
import os
import queue
from typing import Optional

from speech_controller import CloudBrain, OllamaBrain, VoiceCommandProcessor, VoiceController

logger = logging.getLogger(__name__)

def _env_bool(name: str, default: bool) -> bool:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError:
        logger.warning("Invalid %s value '%s'; using %s", name, raw_value, default)
        return default


def _env_float(name: str, default: float) -> float:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        return float(raw_value)
    except ValueError:
        logger.warning("Invalid %s value '%s'; using %.3f", name, raw_value, default)
        return default


def _env_optional_int(name: str) -> Optional[int]:
    raw_value = os.environ.get(name)
    if not raw_value:
        return None
    try:
        return int(raw_value)
    except ValueError:
        logger.warning("Invalid %s value '%s'; using automatic microphone selection", name, raw_value)
        return None


class VoiceModule:
    def __init__(self, assistant: Optional[object]) -> None:
        self.assistant = assistant
        self.voice_ready = False
        self.brain: Optional[OllamaBrain] = None
        self.cloud_brain: Optional[CloudBrain] = None
        self.voice_processor: Optional[VoiceCommandProcessor] = None

        self.wake_word = os.environ.get("WAKE_WORD", "jarvis").strip() or "jarvis"
        if os.environ.get("OLLAMA_BRAIN", "1").lower() not in ("0", "false", "off"):
            timeout_s = 25.0
            try:
                timeout_s = float(os.environ.get("OLLAMA_TIMEOUT", "25"))
            except ValueError:
                logger.warning("Invalid OLLAMA_TIMEOUT value; using 25 seconds")
            self.brain = OllamaBrain(
                model=os.environ.get("OLLAMA_MODEL", "phi3"),
                host=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"),
                timeout_s=timeout_s,
            )
            logger.info("Ollama brain enabled (model=%s)", self.brain.model)

        cloud_api_key = (
            os.environ.get("CLOUD_BRAIN_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or ""
        ).strip()
        cloud_requested = os.environ.get("CLOUD_BRAIN")
        cloud_enabled = (cloud_requested or "1").lower() not in ("0", "false", "off")
        if cloud_enabled and cloud_api_key:
            cloud_timeout_s = 18.0
            try:
                cloud_timeout_s = float(os.environ.get("CLOUD_BRAIN_TIMEOUT", "18"))
            except ValueError:
                logger.warning("Invalid CLOUD_BRAIN_TIMEOUT value; using 18 seconds")
            self.cloud_brain = CloudBrain(
                model=os.environ.get("CLOUD_BRAIN_MODEL", "gpt-4o-mini"),
                api_key=cloud_api_key,
                base_url=os.environ.get(
                    "CLOUD_BRAIN_BASE_URL",
                    "https://api.openai.com/v1",
                ),
                timeout_s=cloud_timeout_s,
            )
            logger.info("Cloud brain enabled (model=%s)", self.cloud_brain.model)
        elif cloud_requested and cloud_enabled and not cloud_api_key:
            logger.warning(
                "CLOUD_BRAIN requested but no CLOUD_BRAIN_API_KEY or OPENAI_API_KEY "
                "was provided."
            )

        self.voice_controller: Optional[VoiceController] = None
        self.mic_label: Optional[str] = None
        self._control_mode_requests: queue.Queue[str] = queue.Queue()

        if _env_bool("VOICE_ENABLED", True):
            try:
                self.voice_controller = VoiceController(
                    assistant=self.assistant,
                    energy_threshold=_env_int("VOICE_ENERGY_THRESHOLD", 350),
                    pause_threshold=_env_float("VOICE_PAUSE_THRESHOLD", 0.55),
                    phrase_threshold=_env_float("VOICE_PHRASE_THRESHOLD", 0.25),
                    calibration_duration=_env_float("VOICE_CALIBRATION_S", 1.5),
                    microphone_index=_env_optional_int("MIC_INDEX"),
                    microphone_name=os.environ.get("MIC_NAME") or None,
                    debug_raw_recognition=_env_bool("VOICE_DEBUG", False),
                    wake_word=self.wake_word,
                    command_window_s=_env_float("VOICE_COMMAND_WINDOW_S", 10.0),
                    acknowledge_wake=_env_bool("VOICE_ACKNOWLEDGE_WAKE", True),
                )
                self.voice_processor = VoiceCommandProcessor(
                    assistant=self.assistant,
                    voice=self.voice_controller,
                    brain=self.brain,
                    cloud_brain=self.cloud_brain,
                    control_mode_handler=self.request_control_mode,
                )
                self.voice_ready = True
                self.mic_label = self.voice_controller.mic_name
                logger.info("Adaptive voice controller active (microphone=%s)", self.mic_label or "default")
            except Exception as exc:
                logger.error("Cannot start voice controller: %s", exc)
        else:
            logger.info("Voice input disabled via VOICE_ENABLED.")

    def toggle_microphone(self, source: str) -> None:
        if self.voice_ready and self.voice_controller:
            next_state = not self.voice_controller.mic_enabled
            self.voice_controller.set_mic_enabled(next_state)
            state_label = "ON" if next_state else "OFF"
            logger.info("Microphone toggled %s by %s.", state_label, source)
            print(f"[Voice] Microphone toggled {state_label} ({source})")
            if self.assistant:
                self.assistant.say("Microphone on" if next_state else "Microphone off")
        else:
            logger.info(
                "Mic toggle requested by %s but voice input is unavailable.",
                source,
            )
            print(
                "[Voice] Microphone control unavailable because voice input "
                "is disabled."
            )

    def is_mic_enabled(self) -> bool:
        return bool(self.voice_ready and self.voice_controller and self.voice_controller.mic_enabled)

    def get_wake_status_text(self) -> str:
        if not self.voice_ready:
            return "Voice: off"
        return self.voice_controller.get_status_text() if self.voice_controller else "Voice: off"

    def get_voice_overlay(self) -> tuple[bool, str, float]:
        if not self.voice_controller:
            return False, "", 0.0
        return (
            self.voice_controller.listening,
            self.voice_controller.last_heard,
            self.voice_controller.last_heard_time,
        )

    def update(self) -> Optional[str]:
        if self.voice_controller and self.voice_processor:
            command = self.voice_controller.get_command()
            if command:
                self.voice_processor.submit(command)
        if self.voice_processor and self.voice_processor.poll_should_exit():
            return "exit"
        return None

    def request_control_mode(self, mode: str) -> None:
        normalized_mode = mode.strip().lower()
        if normalized_mode in {"head", "hand"}:
            self._control_mode_requests.put(normalized_mode)

    def get_requested_control_mode(self) -> Optional[str]:
        try:
            return self._control_mode_requests.get_nowait()
        except queue.Empty:
            return None

    def set_microphone_enabled(self, enabled: bool) -> bool:
        if not self.voice_ready or not self.voice_controller:
            return False
        self.voice_controller.set_mic_enabled(enabled)
        return True

    def submit_external_command(self, command: str) -> bool:
        if not self.voice_ready or not self.voice_processor or not self.voice_controller:
            return False
        authorized_command = self.voice_controller.authorize_transcript(command)
        if not authorized_command:
            return False
        self.voice_processor.submit(authorized_command)
        return True

    def get_runtime_status(self) -> dict:
        return {
            "ready": self.voice_ready,
            "microphone_enabled": self.is_mic_enabled(),
            "wake_word": self.wake_word,
            "voice_status": self.get_wake_status_text(),
            "task_status": self.voice_processor.get_status_text() if self.voice_processor else "Voice: off",
            "microphone": self.mic_label or "System default",
            "listening": bool(self.voice_controller and self.voice_controller.listening),
        }

    def stop(self) -> None:
        if self.voice_controller:
            self.voice_controller.stop()
        if self.voice_processor:
            self.voice_processor.stop()
