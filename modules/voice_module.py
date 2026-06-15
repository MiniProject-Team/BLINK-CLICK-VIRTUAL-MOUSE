from __future__ import annotations

import logging
import os
import threading
import time
from typing import Optional

import sounddevice as sd

from speech_controller import CloudBrain, OllamaBrain, VoiceCommandProcessor
from stt_google import GoogleSTT
from utils.helpers import normalize
from wake_vosk import VoskWake

logger = logging.getLogger(__name__)

DEFAULT_MIC_PREFERENCES = (
    "headset (wings phantom)",
    "microphone array",
)


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


def _pick_microphone_index() -> tuple[Optional[int], Optional[str]]:
    mic_index_raw = os.environ.get("MIC_INDEX")
    if mic_index_raw:
        try:
            idx = int(mic_index_raw)
        except ValueError:
            logger.warning("Invalid MIC_INDEX value '%s'; using auto-detect", mic_index_raw)
        else:
            try:
                info = sd.query_devices(idx)
                return idx, info.get("name") if isinstance(info, dict) else None
            except Exception as exc:
                logger.warning("MIC_INDEX %s not available: %s", idx, exc)

    mic_name_raw = os.environ.get("MIC_NAME", "").strip().lower()
    try:
        devices = sd.query_devices()
    except Exception as exc:
        logger.warning("Unable to query microphone devices: %s", exc)
        return None, None

    preferred_names = [
        part.strip().lower()
        for part in mic_name_raw.split(",")
        if part.strip()
    ]
    if not preferred_names:
        preferred_names = list(DEFAULT_MIC_PREFERENCES)

    for preferred in preferred_names:
        for idx, info in enumerate(devices):
            if not isinstance(info, dict):
                continue
            if info.get("max_input_channels", 0) <= 0:
                continue
            name = str(info.get("name", "")).lower()
            if preferred in name:
                return idx, info.get("name")

    default_device = sd.default.device[0] if sd.default.device else None
    if isinstance(default_device, int) and default_device >= 0:
        info = sd.query_devices(default_device)
        if isinstance(info, dict) and info.get("max_input_channels", 0) > 0:
            return default_device, info.get("name")

    input_keywords = ("microphone", "mic", "headset", "array", "input", "hands-free")
    output_keywords = ("speaker", "output", "stereo mix", "mapper - output")
    best_idx: Optional[int] = None
    best_name: Optional[str] = None
    best_score = float("-inf")

    for idx, info in enumerate(devices):
        if not isinstance(info, dict):
            continue
        if info.get("max_input_channels", 0) <= 0:
            continue
        name = str(info.get("name", "")).lower()
        score = 0
        if any(keyword in name for keyword in input_keywords):
            score += 10
        if any(keyword in name for keyword in output_keywords):
            score -= 12
        if "stereo mix" in name:
            score -= 8
        if score > best_score:
            best_score = score
            best_idx = idx
            best_name = info.get("name")

    if best_idx is not None:
        return best_idx, best_name

    for idx, info in enumerate(devices):
        if isinstance(info, dict) and info.get("max_input_channels", 0) > 0:
            return idx, info.get("name")
    return None, None


class VoiceModule:
    def __init__(self, assistant: Optional[object]) -> None:
        self.assistant = assistant
        self.voice_ready = False
        self.brain: Optional[OllamaBrain] = None
        self.cloud_brain: Optional[CloudBrain] = None
        self.voice_processor: Optional[VoiceCommandProcessor] = None

        self.wake_word = os.environ.get("WAKE_WORD", "jarvis").strip() or "jarvis"
        vosk_model_path = (
            os.environ.get("VOSK_MODEL_PATH", "vosk-model-small-en-in-0.4").strip()
            or "vosk-model-small-en-in-0.4"
        )

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

        mic_index, mic_label = _pick_microphone_index()
        self.mic_label = mic_label
        if mic_index is not None:
            logger.info("Using microphone index %d (%s)", mic_index, mic_label or "unknown")
        else:
            logger.warning("No input microphone detected; voice input may be unavailable.")

        self.wake_listener: Optional[VoskWake] = None
        self.stt: Optional[GoogleSTT] = None
        self.wake_thread: Optional[threading.Thread] = None
        self.wake_stop_event = threading.Event()
        self.wake_enabled_event = threading.Event()
        self.wake_status_lock = threading.Lock()
        self.wake_status = {
            "listening": False,
            "last_text": "",
            "last_time": 0.0,
            "last_error": "",
        }

        if _env_bool("VOICE_ENABLED", True):
            try:
                self.wake_listener = VoskWake(
                    vosk_model_path,
                    wake_word=self.wake_word,
                    device=mic_index,
                )
                self.stt = GoogleSTT(
                    language=os.environ.get("VOICE_LANGUAGE", "en-IN").strip() or "en-IN",
                    mic_index=mic_index,
                    energy_threshold=_env_int("VOICE_ENERGY_THRESHOLD", 300),
                    dynamic_energy_threshold=_env_bool("VOICE_DYNAMIC_THRESHOLD", True),
                    listen_timeout=_env_float("VOICE_LISTEN_TIMEOUT_S", 2.0),
                    phrase_time_limit=_env_float("VOICE_PHRASE_LIMIT_S", 5.0),
                )
                self.voice_processor = VoiceCommandProcessor(
                    assistant=self.assistant,
                    voice=None,
                    brain=self.brain,
                    cloud_brain=self.cloud_brain,
                )
                if _env_bool("VOICE_MIC_ENABLED", True):
                    self.wake_enabled_event.set()
                self.voice_ready = True
                self.wake_thread = threading.Thread(target=self._wake_command_loop, daemon=True)
                self.wake_thread.start()
                logger.info("Vosk wake listener active (model=%s)", vosk_model_path)
            except Exception as exc:
                logger.error("Cannot start Vosk wake or Google STT: %s", exc)
        else:
            logger.info("Voice input disabled via VOICE_ENABLED.")

    def _set_wake_status(
        self,
        *,
        listening: Optional[bool] = None,
        last_text: Optional[str] = None,
        last_time: Optional[float] = None,
        last_error: Optional[str] = None,
    ) -> None:
        with self.wake_status_lock:
            if listening is not None:
                self.wake_status["listening"] = listening
            if last_text is not None:
                self.wake_status["last_text"] = last_text
            if last_time is not None:
                self.wake_status["last_time"] = last_time
            if last_error is not None:
                self.wake_status["last_error"] = last_error

    def _wake_command_loop(self) -> None:
        if not self.wake_listener or not self.stt or not self.voice_processor:
            return

        while not self.wake_stop_event.is_set():
            if not self.wake_enabled_event.is_set():
                self._set_wake_status(listening=False)
                time.sleep(0.1)
                continue

            self._set_wake_status(listening=True)
            try:
                detected = self.wake_listener.listen_wake(stop_event=self.wake_stop_event)
            except Exception as exc:
                self._set_wake_status(listening=False, last_error=str(exc))
                logger.error("Wake listener failed: %s", exc)
                time.sleep(1.0)
                continue

            self._set_wake_status(listening=False)
            if not detected or self.wake_stop_event.is_set():
                continue

            print("Wake detected")

            if self.assistant and _env_bool("VOICE_ACKNOWLEDGE_WAKE", True):
                self.assistant.say("Yes?")

            command = self.stt.listen_command()
            if not command:
                continue

            command = normalize(command)
            self._set_wake_status(
                last_text=command,
                last_time=time.time(),
                last_error="",
            )
            self.voice_processor.submit(command)

    def toggle_microphone(self, source: str) -> None:
        if self.voice_ready:
            next_state = not self.wake_enabled_event.is_set()
            if next_state:
                self.wake_enabled_event.set()
            else:
                self.wake_enabled_event.clear()
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
        return bool(self.voice_ready and self.wake_enabled_event.is_set())

    def get_wake_status_text(self) -> str:
        if not self.voice_ready:
            return "Voice: off"
        if not self.wake_enabled_event.is_set():
            return "Mic: off"
        with self.wake_status_lock:
            if self.wake_status.get("listening"):
                return f"Wake: listening ({self.wake_word})"
            if self.wake_status.get("last_error"):
                return "Wake: error"
        return f"Wake: say {self.wake_word}"

    def get_voice_overlay(self) -> tuple[bool, str, float]:
        with self.wake_status_lock:
            listening = bool(self.wake_status.get("listening"))
            last_text = str(self.wake_status.get("last_text", ""))
            last_time = float(self.wake_status.get("last_time", 0.0))
        return listening, last_text, last_time

    def update(self) -> Optional[str]:
        if self.voice_processor and self.voice_processor.poll_should_exit():
            return "exit"
        return None

    def stop(self) -> None:
        self.wake_stop_event.set()
        if self.wake_thread:
            self.wake_thread.join(timeout=1.5)
        if self.voice_processor:
            self.voice_processor.stop()
