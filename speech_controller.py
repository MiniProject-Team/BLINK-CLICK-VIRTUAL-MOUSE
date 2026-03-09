# -*- coding: utf-8 -*-
"""
speech_controller.py
════════════════════
Speech recognition and voice assistant for Blink-Click Virtual Mouse.

This module provides:
  • AssistantVoice — non-blocking TTS engine (speaks responses in background)
  • VoiceController — continuous speech recognition via Google Speech API
  • process_voice_command() — maps recognised text to mouse/system actions

Supported voice commands:
    "click"           → left click
    "right click"     → right click
    "double click"    → double click
    "scroll up"       → scroll up
    "scroll down"     → scroll down
    "drag"            → toggle drag mode
    "open …"          → open a website / application
    "type …"          → type whatever follows
    "help"            → read available commands
    "stop" / "exit"   → quit the program

Only exact / close matches from a strict command whitelist are
executed – background noise and unrecognised phrases are silently
ignored so actions never fire accidentally.

Author  : Blink-Click Virtual Mouse Team
"""

from __future__ import annotations

import logging
import queue
import threading
import time
import webbrowser
from typing import Optional, Tuple

import difflib

import pyautogui

# ── Module logger ────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ================================================================
#  STRICT COMMAND WHITELIST
# ================================================================
# Only recognised text that matches (or closely matches) one of these
# entries will ever be executed.  Everything else is silently dropped.
VALID_COMMANDS: set[str] = {
    "click",
    "right click",
    "double click",
    "scroll up",
    "scroll down",
    "up",
    "down",
    "page up",
    "page down",
    "drag",
    "dwell",
    "open google",
    "open chrome",
    "open youtube",
    "help",
    "stop",
    "exit",
    "quit",
    "close",
}
# Prefix commands (e.g. "type hello world") – matched by startswith
VALID_PREFIXES: tuple[str, ...] = ("type ", "open ")

# Minimum similarity ratio (0-1) for fuzzy matching via difflib
_FUZZY_THRESHOLD: float = 0.72
INPUT_DEVICE_KEYWORDS: tuple[str, ...] = (
    "microphone",
    "mic",
    "headset",
    "array",
    "input",
    "hands-free",
)
OUTPUT_DEVICE_KEYWORDS: tuple[str, ...] = (
    "output",
    "speaker",
    "stereo mix",
    "mapper - output",
)


def match_command(text: str) -> Optional[str]:
    """Return the canonical command if *text* matches whitelist, else None.

    Matching strategy:
      1. Exact match against VALID_COMMANDS.
      2. Prefix match against VALID_PREFIXES.
      3. Fuzzy match (difflib) against VALID_COMMANDS (≥ 72 % similarity).
    """
    text = text.strip().lower()
    if not text:
        return None

    # 1. Exact
    if text in VALID_COMMANDS:
        return text

    # 2. Prefix
    for pfx in VALID_PREFIXES:
        if text.startswith(pfx):
            return text

    # 3. Fuzzy – find the closest command
    best = difflib.get_close_matches(
        text, VALID_COMMANDS, n=1, cutoff=_FUZZY_THRESHOLD
    )
    if best:
        logger.info("Fuzzy matched '%s' → '%s'", text, best[0])
        return best[0]

    return None

# ── Optional dependency flags ────────────────────────────────────
try:
    import speech_recognition as sr

    SR_AVAILABLE = True
except ImportError:
    SR_AVAILABLE = False
    logger.warning("speech_recognition not installed – voice input disabled.")

try:
    import pyttsx3

    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    logger.warning("pyttsx3 not installed – TTS disabled.")


# ================================================================
#  ASSISTANT VOICE  – non-blocking text-to-speech
# ================================================================
def list_microphone_names() -> list[str]:
    """Return available microphone device names or an empty list."""
    if not SR_AVAILABLE:
        return []
    try:
        return sr.Microphone.list_microphone_names()
    except Exception as exc:
        logger.error("Unable to list microphone devices: %s", exc)
        return []


def pick_input_microphone(
    preferred_name: Optional[str] = None,
) -> Tuple[Optional[int], Optional[str]]:
    """Choose the most likely usable input microphone."""
    names = list_microphone_names()
    if not names:
        return None, None

    if preferred_name:
        preferred = preferred_name.strip().lower()
        for idx, name in enumerate(names):
            if preferred in name.lower():
                return idx, name

    best_idx: Optional[int] = None
    best_name: Optional[str] = None
    best_score = float("-inf")

    for idx, raw_name in enumerate(names):
        name = raw_name.lower()
        score = 0
        if any(keyword in name for keyword in INPUT_DEVICE_KEYWORDS):
            score += 10
        if "microphone array" in name:
            score += 4
        if "realtek" in name:
            score += 2
        if any(keyword in name for keyword in OUTPUT_DEVICE_KEYWORDS):
            score -= 12
        if "mapper - input" in name or "primary sound capture" in name:
            score -= 3

        if score > best_score:
            best_score = score
            best_idx = idx
            best_name = raw_name

    if best_score <= 0:
        return None, None
    return best_idx, best_name


class AssistantVoice:
    """
    Google-Assistant-style TTS that queues text and speaks in a
    background thread so the main loop is never blocked.
    """

    GREETING = (
        "Hello! I am your voice assistant virtual mouse. "
        "I am ready. Move your head to control the cursor. "
        "Blink to click. Say help for available voice commands."
    )

    def __init__(self, rate: int = 165, volume: float = 1.0) -> None:
        self._q: queue.Queue[str] = queue.Queue()
        self._stopped = False
        self._rate = rate
        self._volume = volume
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        logger.info("AssistantVoice started (rate=%d).", rate)

    # ── background worker ────────────────────────────────────────
    def _worker(self) -> None:
        """Each message creates a fresh pyttsx3 engine to avoid thread issues."""
        while not self._stopped:
            try:
                text = self._q.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                engine = pyttsx3.init()
                engine.setProperty("rate", self._rate)
                engine.setProperty("volume", self._volume)

                # Prefer a female voice for friendlier tone
                voices = engine.getProperty("voices")
                for v in voices:
                    if "female" in v.name.lower() or "zira" in v.name.lower():
                        engine.setProperty("voice", v.id)
                        break

                engine.say(text)
                engine.runAndWait()
                engine.stop()
            except Exception as exc:
                logger.error("TTS error: %s", exc)
            finally:
                self._q.task_done()

    # ── public API ───────────────────────────────────────────────
    def say(self, text: str) -> None:
        """Add *text* to the speech queue (non-blocking)."""
        logger.info("[Assistant] ▶ %s", text)
        print(f"[Assistant] ▶ {text}")
        self._q.put(text)

    def greet(self) -> None:
        """Speak the startup greeting."""
        self.say(self.GREETING)

    def stop(self) -> None:
        """Signal the worker thread to exit."""
        self._stopped = True
        logger.info("AssistantVoice stopped.")


# ================================================================
#  VOICE CONTROLLER  – continuous speech recognition
# ================================================================
class VoiceController:
    """
    Runs in a background thread, continuously listens via the
    microphone, and feeds recognised text into a thread-safe queue.

    Uses Google Speech Recognition API (online, high accuracy).
    """

    HELP_TEXT = (
        "Available voice commands: "
        "click. right click. double click. "
        "scroll up. scroll down. "
        "drag. type then your text. "
        "open Google or open YouTube. "
        "help. stop or exit."
    )

    def __init__(
        self,
        assistant: Optional[AssistantVoice] = None,
        energy_threshold: int = 350,
        pause_threshold: float = 0.6,
        phrase_threshold: float = 0.3,
        calibration_duration: float = 2.5,
        microphone_index: Optional[int] = None,
        microphone_name: Optional[str] = None,
        debug_raw_recognition: bool = False,
    ) -> None:
        if not SR_AVAILABLE:
            raise RuntimeError(
                "speech_recognition is not installed. "
                "Install it with: pip install SpeechRecognition"
            )

        self.recognizer = sr.Recognizer()
        self.mic_index = microphone_index
        self.mic_name: Optional[str] = None
        self.debug_raw = debug_raw_recognition

        available_names = list_microphone_names()
        if self.mic_index is None:
            picked_index, picked_name = pick_input_microphone(microphone_name)
            if picked_index is not None:
                self.mic_index = picked_index
                self.mic_name = picked_name
            elif available_names:
                self.mic_name = available_names[0]
        elif 0 <= self.mic_index < len(available_names):
            self.mic_name = available_names[self.mic_index]

        if self.mic_index is None:
            self.mic = sr.Microphone()
        else:
            self.mic = sr.Microphone(device_index=self.mic_index)
        self.assistant = assistant
        self._cmd_q: queue.Queue[str] = queue.Queue()
        self.stopped = False
        self.last_error: str = ""

        # Voice status tracking (read by main loop for HUD)
        self.listening: bool = False
        self.last_heard: str = ""
        self.last_matched: str = ""
        self.last_heard_time: float = 0.0

        # Tuning
        self.recognizer.energy_threshold = energy_threshold
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.dynamic_energy_adjustment_damping = 0.2
        self.recognizer.dynamic_energy_adjustment_ratio = 1.5
        self.recognizer.pause_threshold = pause_threshold
        self.recognizer.phrase_threshold = phrase_threshold
        self.recognizer.non_speaking_duration = pause_threshold

        # Calibrate
        logger.info("Calibrating microphone …")
        print("[Voice] Calibrating microphone …")
        if available_names:
            print(f"[Voice] Using microphone: {self.mic_name or 'System default'}")
        try:
            with self.mic as source:
                self.recognizer.adjust_for_ambient_noise(
                    source, duration=calibration_duration
                )
            logger.info("Microphone calibrated.")
            logger.info(
                "Voice recognizer ready (device_index=%s, device_name=%s, energy_threshold=%.2f)",
                self.mic_index,
                self.mic_name or "default",
                self.recognizer.energy_threshold,
            )
            print("[Voice] Microphone ready.")
        except Exception as exc:
            self.last_error = str(exc)
            logger.error("Microphone calibration failed: %s", exc)
            print(f"[Voice] Microphone error: {exc}")

        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()

    # ── listener loop ────────────────────────────────────────────
    def _listen_loop(self) -> None:
        while not self.stopped:
            try:
                self.listening = True
                with self.mic as source:
                    audio = self.recognizer.listen(
                        source, timeout=4, phrase_time_limit=6
                    )
                self.listening = False
                try:
                    raw_text = self.recognizer.recognize_google(audio, language="en-US")
                    self.last_error = ""
                except sr.UnknownValueError:
                    if self.debug_raw:
                        print("[Voice DEBUG] UnknownValueError: could not parse audio")
                    continue
                except sr.RequestError as exc:
                    self.last_error = f"Speech API error: {exc}"
                    logger.error("Google Speech API error during recognition: %s", exc)
                    time.sleep(2)
                    continue

                text = raw_text.lower().strip()
                self.last_heard = text
                self.last_heard_time = time.time()
                if self.debug_raw:
                    print(f"[Voice DEBUG] Raw recognised: '{raw_text}'")

                # ── Whitelist gate ────────────────────────────────
                matched = match_command(text)
                if matched is None:
                    logger.info("Ignored (no match): '%s'", text)
                    self.last_matched = ""
                    continue

                self.last_matched = matched
                logger.info("Matched command: '%s' → '%s'", text, matched)
                print(f"[Voice] Command: '{matched}'")
                self._cmd_q.put(matched)

            except (sr.WaitTimeoutError, sr.UnknownValueError):
                self.listening = False
                pass
            except sr.RequestError as exc:
                self.listening = False
                self.last_error = f"Speech API error: {exc}"
                logger.error("Google Speech API error: %s", exc)
                time.sleep(2)
            except Exception as exc:
                self.listening = False
                self.last_error = str(exc)
                logger.exception("Unexpected voice recognition failure")

    # ── public API ───────────────────────────────────────────────
    def get_command(self) -> Optional[str]:
        """Return the next recognised command, or ``None``."""
        try:
            return self._cmd_q.get_nowait()
        except queue.Empty:
            return None

    def get_status_text(self) -> str:
        """Short device or error summary for logs or HUD."""
        if self.last_error:
            return self.last_error
        if self.mic_name:
            return f"Mic: {self.mic_name}"
        return "Mic: default"

    def stop(self) -> None:
        """Signal the listener thread to shut down."""
        self.stopped = True
        logger.info("VoiceController stopped.")


# ================================================================
#  VOICE COMMAND PROCESSOR
# ================================================================
def process_voice_command(
    cmd: str,
    drag_mode: bool,
    assistant: Optional[AssistantVoice],
    voice: Optional[VoiceController],
) -> Tuple[bool, bool]:
    """
    Execute the action corresponding to the recognised command.

    Parameters
    ----------
    cmd : str
        Lowercased recognised text.
    drag_mode : bool
        Current drag-mode state.
    assistant : AssistantVoice or None
        TTS engine for spoken feedback.
    voice : VoiceController or None
        Voice controller (used for help text).

    Returns
    -------
    (drag_mode, should_exit) : Tuple[bool, bool]
    """
    if not cmd:
        return drag_mode, False

    def speak(text: str) -> None:
        if assistant:
            assistant.say(text)

    should_exit = False

    # ── Click commands ───────────────────────────────────────────
    if "right" in cmd and "click" in cmd:
        pyautogui.rightClick()
        speak("Right click")

    elif "double" in cmd and "click" in cmd:
        pyautogui.doubleClick()
        speak("Double click")

    elif "click" in cmd:
        pyautogui.click()
        speak("Click")

    # ── Scroll commands ──────────────────────────────────────────
    elif "scroll up" in cmd or cmd in ("up", "page up"):
        pyautogui.scroll(8)
        speak("Scrolling up")

    elif "scroll down" in cmd or cmd in ("down", "page down"):
        pyautogui.scroll(-8)
        speak("Scrolling down")

    # ── Drag toggle ──────────────────────────────────────────────
    elif "drag" in cmd:
        drag_mode = not drag_mode
        if drag_mode:
            pyautogui.mouseDown()
            speak("Drag mode on. Move your head to drag. Say drag again to release.")
        else:
            pyautogui.mouseUp()
            speak("Drag released")

    # ── Type text ────────────────────────────────────────────────
    elif cmd.startswith("type "):
        typed_text = cmd[5:]
        pyautogui.typewrite(typed_text, interval=0.06)
        speak(f"Typed: {typed_text}")

    # ── Open websites / apps ─────────────────────────────────────
    elif "open" in cmd:
        if "google" in cmd or "chrome" in cmd:
            webbrowser.open("https://www.google.com")
            speak("Opening Google")
        elif "youtube" in cmd:
            webbrowser.open("https://www.youtube.com")
            speak("Opening YouTube")
        else:
            speak(
                "I can open Google or YouTube. "
                "Say open Google or open YouTube."
            )

    # ── Help ─────────────────────────────────────────────────────
    elif "help" in cmd:
        help_msg = (
            VoiceController.HELP_TEXT
            if voice
            else (
                "Voice commands: click, right click, double click, "
                "scroll up, scroll down, drag, type, help, stop."
            )
        )
        speak(help_msg)

    # ── Exit commands ────────────────────────────────────────────
    elif any(w in cmd for w in ("stop", "exit", "quit", "close")):
        speak("Goodbye! Closing virtual mouse.")
        time.sleep(2)
        should_exit = True

    # ── Unrecognised – silently ignore (whitelist already filtered) ─
    else:
        logger.info("Unrecognised command bypassed: '%s'", cmd)

    return drag_mode, should_exit
