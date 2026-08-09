# -*- coding: utf-8 -*-
"""
speech_controller.py

Speech recognition and voice assistant support for Blink-Click Virtual Mouse.

This module provides:
    - AssistantVoice: non-blocking TTS engine
    - VoiceController: continuous speech recognition with a wake word
    - OllamaBrain: local LLM planner for free-form task requests
    - VoiceCommandProcessor: safe desktop-action executor with confirmation flow
"""

from __future__ import annotations

import difflib
import json
import logging
import os
import queue
import re
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from dataclasses import dataclass
from typing import Any, Callable, Optional

import pyautogui

try:
    import pygetwindow as gw
except ImportError:  # pragma: no cover - optional dependency
    gw = None

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional dependency
    np = None

try:
    import noisereduce as nr
except ImportError:  # pragma: no cover - optional dependency
    nr = None

logger = logging.getLogger(__name__)

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

SUPPORTED_DECISIONS: tuple[str, ...] = ("allow", "confirm", "block", "noop")
SUPPORTED_STEP_ACTIONS: tuple[str, ...] = (
    "left_click",
    "right_click",
    "double_click",
    "minimize",
    "maximize",
    "restore",
    "scroll",
    "drag_toggle",
    "type_text",
    "press_key",
    "hotkey",
    "open_url",
    "launch_app",
    "system_search",
    "close_app",
    "control_mode",
    "wait",
    "help",
    "stop",
    "noop",
)

DEFAULT_WAKE_WORD = "jarvis"
DEFAULT_COMMAND_WINDOW_S = 10.0
DEFAULT_CONFIRM_WINDOW_S = 15.0
MAX_PLAN_STEPS = 6
MAX_TYPE_CHARS = 400
MAX_SCROLL_AMOUNT = 1200
MAX_WAIT_SECONDS = 5.0
DEFAULT_WAKE_WORD_MATCH_RATIO = 0.86
DEFAULT_WAKE_WORD_PREFIX_TOKENS = 2
DEFAULT_WAKE_WORD_COOLDOWN_S = 1.0
DEFAULT_WAKE_LISTEN_TIMEOUT_S = 1.8
DEFAULT_WAKE_PHRASE_LIMIT_S = 2.1
DEFAULT_COMMAND_PHRASE_LIMIT_S = 5.2
DEFAULT_ERROR_FEEDBACK_COOLDOWN_S = 6.0
LOCAL_STEP_WAIT_SECONDS = 0.8
DEFAULT_VOICE_COMMAND_STALE_S = 4.0
DEFAULT_VOICE_DUPLICATE_WINDOW_S = 1.4
DEFAULT_VOICE_MAX_PENDING_COMMANDS = 2
DEFAULT_MIC_LISTEN_WINDOW_S = 10.0
DEFAULT_MIC_CYCLE_PAUSE_S = 0.35
DEFAULT_CONVERSATION_TIMEOUT_S = 9.0
DEFAULT_AUDIO_TARGET_RMS = 0.085
DEFAULT_AUDIO_MAX_GAIN = 4.0
DEFAULT_NOISE_GATE_LEVEL = 0.01
DEFAULT_AMBIENT_REFRESH_S = 45.0
DEFAULT_AMBIENT_SAMPLE_S = 0.35
DEFAULT_MIN_NR_AUDIO_S = 0.45
DEFAULT_NR_SKIP_RMS_FACTOR = 1.8
DEFAULT_CLOUD_TIMEOUT_S = 18.0

VOICE_TEXT_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("note pad", "notepad"),
    ("google chrome", "chrome"),
    ("you tube", "youtube"),
    ("u tube", "youtube"),
    ("micro soft edge", "microsoft edge"),
    ("minimise", "minimize"),
    ("maximise", "maximize"),
    ("आशु", "jarvis"),
    ("अशु", "jarvis"),
    ("हाँ", "yes"),
    ("हां", "yes"),
    ("नहीं", "no"),
    ("मत", "no"),
    ("रद्द", "cancel"),
    ("राइट क्लिक", "right click"),
    ("लेफ्ट क्लिक", "left click"),
    ("स्क्रॉल नीचे", "scroll down"),
    ("नीचे स्क्रॉल", "scroll down"),
    ("स्क्रॉल ऊपर", "scroll up"),
    ("ऊपर स्क्रॉल", "scroll up"),
    ("खोल दो", "open"),
    ("खोलो", "open"),
    ("ओपन", "open"),
    ("बंद करो", "close"),
    ("रोक दो", "stop"),
    ("रुको", "stop"),
    ("लिखो", "type"),
    ("टाइप करो", "type"),
    ("टाइप", "type"),
    ("सर्च", "search"),
    ("खोजो", "search"),
    ("पूछो", "ask"),
    ("सवाल", "question"),
    ("नीचे", "down"),
    ("ऊपर", "up"),
    ("दायें", "right"),
    ("दाएं", "right"),
    ("बायें", "left"),
    ("बाएं", "left"),
    ("क्लिक", "click"),
    ("यूट्यूब", "youtube"),
    ("क्रोम", "chrome"),
    ("गूगल", "google"),
    ("नोटपैड", "notepad"),
    ("कैलकुलेटर", "calculator"),
    ("सेटिंग्स", "settings"),
    ("haan ji", "yes"),
    ("haan", "yes"),
    ("nahin", "no"),
    ("nahi", "no"),
    ("right click", "right click"),
    ("left click", "left click"),
    ("scroll neeche", "scroll down"),
    ("scroll niche", "scroll down"),
    ("scroll upar", "scroll up"),
    ("khol do", "open"),
    ("band karo", "close"),
    ("rok do", "stop"),
    ("likho", "type"),
    ("type karo", "type"),
    ("khojo", "search"),
    ("poocho", "ask"),
    ("sawal", "question"),
    ("neeche", "down"),
    ("niche", "down"),
    ("upar", "up"),
    ("daaye", "right"),
    ("daayen", "right"),
    ("baaye", "left"),
    ("baayen", "left"),
    ("ruko", "stop"),
)

KEY_ALIASES: dict[str, str] = {
    "control": "ctrl",
    "escape": "esc",
    "return": "enter",
    "spacebar": "space",
    "pageup": "pgup",
    "pagedown": "pgdn",
    "page-up": "pgup",
    "page-down": "pgdn",
    "space bar": "space",
    "page up": "pgup",
    "page down": "pgdn",
    "del": "delete",
    "windows": "win",
    "command": "win",
}

SAFE_APP_ALIASES: dict[str, str] = {
    "chrome": "chrome.exe",
    "google chrome": "chrome.exe",
    "edge": "msedge.exe",
    "microsoft edge": "msedge.exe",
    "notepad": "notepad.exe",
    "calculator": "calc.exe",
    "calc": "calc.exe",
    "paint": "mspaint.exe",
    "mspaint": "mspaint.exe",
    "explorer": "explorer.exe",
    "file explorer": "explorer.exe",
    "settings": "ms-settings:",
    "camera": "microsoft.windows.camera:",
    "photos": "ms-photos:",
}
APP_DISPLAY_NAMES: dict[str, str] = {
    "chrome": "Chrome",
    "google chrome": "Chrome",
    "edge": "Edge",
    "microsoft edge": "Edge",
    "notepad": "Notepad",
    "calculator": "Calculator",
    "calc": "Calculator",
    "paint": "Paint",
    "mspaint": "Paint",
    "explorer": "File Explorer",
    "file explorer": "File Explorer",
    "settings": "Settings",
    "camera": "Camera",
    "photos": "Photos",
}

BLOCKED_APP_TERMS: tuple[str, ...] = (
    "cmd",
    "command prompt",
    "powershell",
    "terminal",
    "bash",
    "wsl",
    "regedit",
    "registry",
    "task scheduler",
)

BLOCKED_REQUEST_TERMS: tuple[str, ...] = (
    "delete",
    "remove file",
    "erase",
    "format",
    "factory reset",
    "wipe",
    "shutdown",
    "restart computer",
    "reboot",
    "powershell",
    "command prompt",
    "terminal",
    "shell command",
    "run script",
    "disable antivirus",
    "disable defender",
    "disable firewall",
    "registry",
    "regedit",
    "password",
    "otp",
    "token",
    "secret",
    "credential",
    "hack",
    "exploit",
    "bypass security",
    "steal",
    "malware",
    "ransomware",
)

CONFIRM_REQUEST_TERMS: tuple[str, ...] = (
    "close window",
    "close tab",
    "quit app",
    "save file",
    "submit",
    "send",
    "purchase",
    "pay",
)

SAFE_HOTKEY_COMBOS: set[tuple[str, ...]] = {
    ("alt", "tab"),
    ("ctrl", "a"),
    ("ctrl", "c"),
    ("ctrl", "f"),
    ("ctrl", "l"),
    ("ctrl", "n"),
    ("ctrl", "s"),
    ("ctrl", "t"),
    ("ctrl", "v"),
    ("ctrl", "w"),
    ("ctrl", "x"),
    ("ctrl", "y"),
    ("ctrl", "z"),
}
CONFIRM_HOTKEY_COMBOS: set[tuple[str, ...]] = {
    ("alt", "f4"),
}
BLOCKED_HOTKEY_KEYS: set[str] = {"win", "winleft", "winright"}
CONFIRM_KEYS: set[str] = {"delete"}
YES_WORDS: set[str] = {"yes", "confirm", "do it", "go ahead", "continue", "ok", "okay"}
NO_WORDS: set[str] = {"no", "cancel", "stop", "dont", "don't", "never mind"}

LOCAL_HOTKEY_COMMANDS: tuple[tuple[tuple[str, ...], tuple[str, ...], str, str], ...] = (
    (("copy", "copy this", "copy that"), ("ctrl", "c"), "Copy", "Copying."),
    (("paste", "paste here"), ("ctrl", "v"), "Paste", "Pasting."),
    (("cut", "cut this", "cut that"), ("ctrl", "x"), "Cut", "Cutting."),
    (("undo",), ("ctrl", "z"), "Undo", "Undoing."),
    (("redo",), ("ctrl", "y"), "Redo", "Redoing."),
    (("select all",), ("ctrl", "a"), "Select all", "Selecting everything."),
    (("save", "save file"), ("ctrl", "s"), "Save", "Saving."),
    (("new tab",), ("ctrl", "t"), "New tab", "Opening a new tab."),
    (("close tab",), ("ctrl", "w"), "Close tab", "Closing the tab."),
    (("find", "find text"), ("ctrl", "f"), "Find", "Opening find."),
)

BROWSER_APPS: set[str] = {"chrome", "google chrome", "edge", "microsoft edge"}
COMPOUND_COMMAND_SEPARATORS: tuple[str, ...] = (
    " and then ",
    " then ",
    ", then ",
    ";",
    ",",
    " and ",
)
WEB_APP_FALLBACK_URLS: dict[str, str] = {
    "google": "https://www.google.com",
    "youtube": "https://www.youtube.com",
}
GENERIC_PROGRESS_REPLIES: set[str] = {"", "working on it."}
NO_MATCH_REPLY = (
    "I didn't understand that. Try something like open chrome, type hello, "
    "or search AI tools."
)
QUESTION_THINKING_REPLY = "Let me think."
QUESTION_UNAVAILABLE_REPLY = "I could not get an answer right now. Please try again."
CHAT_PREFIXES: tuple[str, ...] = (
    "chat",
    "chatbot",
    "assistant",
    "ask",
    "question",
)

PROJECT_FAQ: tuple[tuple[tuple[str, ...], str], ...] = (
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
        ("voice", "wake word", "jarvis"),
        "Say the wake word 'Jarvis' before a command. You can change it with the WAKE_WORD environment variable.",
    ),
    (
        ("run", "start", "launch", "how to run"),
        "Run python main.py to start the system. Use python frontend_server.py for the launcher page.",
    ),
    (
        ("exit", "quit", "stop"),
        "Press ESC twice within about 1.5 seconds or say 'Jarvis stop' to exit.",
    ),
    (
        ("camera", "webcam", "not opening"),
        "If the camera is not opening, close other apps using the webcam and try CAMERA_INDEX=1 before running.",
    ),
    (
        ("requirements", "prerequisites"),
        "You need Python 3.10+, a webcam, a microphone, and internet for speech recognition. Ollama is optional for natural language planning.",
    ),
)


def _extract_json_object(text: str) -> Optional[dict[str, Any]]:
    """Extract and parse the first JSON object in *text*."""
    if not text:
        return None

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, dict):
        return parsed
    return None


def _clean_voice_text(text: str) -> str:
    text = text.lower().strip()
    text = text.replace("_", " ")
    text = re.sub(r"[^\w\s:/?&=._+\-]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    for source, target in VOICE_TEXT_REPLACEMENTS:
        text = re.sub(
            rf"(?<!\w){re.escape(source)}(?!\w)",
            target,
            text,
            flags=re.UNICODE,
        )
    return text.strip()


def _clamp_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(parsed, maximum))


def _clamp_float(value: Any, default: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(parsed, maximum))


def _normalize_key_name(key: str) -> str:
    normalized = key.strip().lower().replace("_", " ").replace("-", " ")
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = KEY_ALIASES.get(normalized, normalized)
    return normalized.replace(" ", "")


def _is_safe_http_url(url: str) -> bool:
    parsed = urllib.parse.urlparse(url.strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _make_search_url(query: str, engine: str = "google") -> str:
    encoded = urllib.parse.quote_plus(query.strip())
    if engine == "youtube":
        return f"https://www.youtube.com/results?search_query={encoded}"
    return f"https://www.google.com/search?q={encoded}"


def _coerce_safe_url(target: str) -> Optional[str]:
    candidate = target.strip()
    if not candidate:
        return None
    if candidate.startswith(("http://", "https://")):
        return candidate if _is_safe_http_url(candidate) else None

    candidate = candidate.strip().lstrip("/")
    if re.fullmatch(r"(?:www\.)?[a-z0-9-]+(?:\.[a-z0-9-]+)+(?:/[^\s]*)?", candidate):
        return f"https://{candidate}"
    return None


def _matches_phrase(text: str, options: set[str]) -> bool:
    if text in options:
        return True
    for option in options:
        if text.startswith(f"{option} ") or text.endswith(f" {option}"):
            return True
        if f" {option} " in text:
            return True
    return False


def _parse_phrase_list(value: Optional[str]) -> set[str]:
    if not value:
        return set()
    phrases: set[str] = set()
    for raw_part in value.split(","):
        cleaned = _clean_voice_text(raw_part)
        if cleaned:
            phrases.add(cleaned)
    return phrases


def _parse_bool_env(value: Optional[str], default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_language_codes(value: Optional[str], fallback: str) -> list[str]:
    raw_parts = value.split(",") if value else [fallback]
    languages: list[str] = []
    for raw_part in raw_parts:
        cleaned = raw_part.strip()
        if cleaned and cleaned not in languages:
            languages.append(cleaned)
    if not languages:
        languages.append(fallback)
    return languages


def _extract_transcript_candidates(result: Any) -> list[str]:
    if isinstance(result, str):
        cleaned = result.strip()
        return [cleaned] if cleaned else []

    if isinstance(result, dict):
        alternatives = result.get("alternative", [])
        if isinstance(alternatives, list):
            ranked: list[tuple[float, int, str]] = []
            for index, item in enumerate(alternatives):
                if not isinstance(item, dict):
                    continue
                transcript = item.get("transcript")
                if not isinstance(transcript, str):
                    continue
                cleaned = transcript.strip()
                if not cleaned:
                    continue
                confidence_raw = item.get("confidence", 0.0)
                try:
                    confidence = float(confidence_raw)
                except (TypeError, ValueError):
                    confidence = 0.0
                ranked.append((confidence, index, cleaned))

            if ranked:
                ranked.sort(key=lambda entry: (-entry[0], entry[1]))
                return [entry[2] for entry in ranked]

    return []


def _resolve_app_target(app_name: str) -> Optional[str]:
    cleaned = app_name.strip().lower()
    if cleaned in SAFE_APP_ALIASES:
        return SAFE_APP_ALIASES[cleaned]
    return None


def _close_app_by_name(app_name: str) -> bool:
    cleaned = app_name.strip().lower()
    app_map = {
        "notepad": "notepad.exe",
        "chrome": "chrome.exe",
        "edge": "msedge.exe",
        "vlc": "vlc.exe",
        "word": "winword.exe",
        "winword": "winword.exe",
        "excel": "excel.exe",
    }

    exe = app_map.get(cleaned)
    if not exe:
        return False

    subprocess.run(
        ["taskkill", "/f", "/im", exe],
        capture_output=True,
        text=True,
        check=False,
    )
    return True


def control_window(action: str) -> bool:
    """Control the active window directly, with a keyboard fallback."""
    requested_action = _clean_voice_text(action)
    if requested_action not in {"minimize", "maximize", "restore"}:
        return False

    active_window = None
    if gw is not None:
        try:
            active_window = gw.getActiveWindow()
        except Exception as exc:
            logger.debug("Unable to access active window via pygetwindow: %s", exc)

    if active_window is not None:
        try:
            if requested_action == "minimize":
                active_window.minimize()
            elif requested_action == "maximize":
                active_window.maximize()
            else:
                active_window.restore()
            return True
        except Exception as exc:
            logger.debug(
                "Direct window control failed for '%s': %s",
                requested_action,
                exc,
            )

    # Keyboard fallback when direct window control is unavailable.
    if requested_action == "minimize":
        pyautogui.hotkey("win", "down")
        time.sleep(0.1)
        pyautogui.hotkey("win", "down")
        return True
    if requested_action == "maximize":
        pyautogui.hotkey("win", "up")
        return True
    if requested_action == "restore":
        pyautogui.hotkey("alt", "space")
        time.sleep(0.1)
        pyautogui.press("r")
        return True
    return False


def _scroll_amount_from_text(text: str) -> int:
    if any(word in text for word in ("little", "slightly", "bit")):
        return 250
    if any(word in text for word in ("fast", "far", "lot", "more")):
        return 700
    return 450


def _extract_text_after_command(text: str, commands: tuple[str, ...]) -> str:
    for command in commands:
        if text.startswith(command):
            return text[len(command) :].strip(" .")
        marker = f" {command}"
        if marker in text:
            return text.split(marker, 1)[1].strip(" .")
    return ""


def _normalize_request_text(text: str) -> str:
    cleaned = _clean_voice_text(text)
    if not cleaned:
        return ""

    original = cleaned
    previous = None
    while cleaned != previous:
        previous = cleaned
        cleaned = re.sub(r"^(?:please|hey|ok|okay)\s+", "", cleaned).strip()
        cleaned = re.sub(r"^(?:can|could|would|will)\s+you\s+", "", cleaned).strip()
        cleaned = re.sub(r"^(?:i want you to|i need you to|help me(?: to)?)\s+", "", cleaned).strip()

    if any(
        cleaned.startswith(prefix)
        for prefix in ("type ", "write ", "input ", "enter text ")
    ):
        return cleaned

    cleaned = re.sub(r"\s+(?:for me|please)$", "", cleaned).strip()

    cleaned = re.sub(r"\blook up\b", "search", cleaned)
    cleaned = re.sub(r"\bsearch up\b", "search", cleaned)
    cleaned = re.sub(r"\bgo to\b", "open", cleaned)
    cleaned = re.sub(r"\bopen up\b", "open", cleaned)
    cleaned = re.sub(r"\bbring up\b", "open", cleaned)
    cleaned = re.sub(r"\blaunch\b", "open", cleaned)
    cleaned = re.sub(
        r"\bstart\b(?=\s+(?:chrome|google chrome|edge|microsoft edge|notepad|calculator|calc|paint|mspaint|explorer|file explorer|settings|camera|photos)\b)",
        "open",
        cleaned,
    )
    cleaned = re.sub(r"\bwrite down\b", "type", cleaned)

    youtube_match = re.fullmatch(r"search\s+(.+?)\s+on youtube", cleaned)
    if youtube_match:
        cleaned = f"open youtube and search {youtube_match.group(1).strip()}"

    google_match = re.fullmatch(r"search\s+(.+?)\s+on google", cleaned)
    if google_match:
        cleaned = f"search {google_match.group(1).strip()}"

    if cleaned != original:
        logger.debug("Normalized request text '%s' -> '%s'", original, cleaned)

    return cleaned


def _step_needs_focus_wait(step: dict[str, Any]) -> bool:
    return step.get("action") in {"type_text", "press_key", "hotkey"}


def _merge_local_plans(
    parts: list[str],
    planned_parts: list[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    merged_steps: list[dict[str, Any]] = []
    summaries: list[str] = []

    for index, plan in enumerate(planned_parts):
        steps = list(plan.get("steps", []))
        if not steps:
            return None

        if merged_steps:
            previous_step = merged_steps[-1]
            next_step = steps[0]

            # Avoid launching a browser and then opening a URL in a separate app.
            if (
                previous_step.get("action") == "launch_app"
                and previous_step.get("app") in BROWSER_APPS
                and next_step.get("action") == "open_url"
            ):
                merged_steps.pop()
                previous_step = merged_steps[-1] if merged_steps else None

            if previous_step and _step_needs_focus_wait(next_step):
                if previous_step.get("action") in {"launch_app", "open_url"}:
                    merged_steps.append(
                        {"action": "wait", "seconds": LOCAL_STEP_WAIT_SECONDS}
                    )

        merged_steps.extend(steps)
        summary = str(plan.get("summary", "")).strip()
        if summary:
            summaries.append(summary)

        if len(merged_steps) > MAX_PLAN_STEPS:
            logger.debug(
                "Local compound plan exceeded max steps for parts: %s",
                parts,
            )
            return None

    if not merged_steps:
        return None

    summary_text = " + ".join(summaries[:3]).strip() or "Combined action"
    if len(summaries) > 3:
        summary_text = f"{summary_text} + more"

    return _default_plan(
        decision="allow",
        summary=summary_text,
        reply="Working on it.",
        steps=merged_steps[:MAX_PLAN_STEPS],
    )


def _should_prefer_direct_local_plan(text: str, direct_plan: dict[str, Any]) -> bool:
    if not any(separator in text for separator in COMPOUND_COMMAND_SEPARATORS):
        return True

    if re.search(r"open (?:google|youtube)(?: and)? search(?: for)? .+", text):
        return True

    steps = direct_plan.get("steps", [])
    if len(steps) == 1 and steps[0].get("action") == "open_url":
        summary = str(direct_plan.get("summary", "")).lower()
        if "search" in summary:
            return True

    return False


def _plan_single_task_locally(cmd: str) -> Optional[dict[str, Any]]:
    text = _clean_voice_text(cmd)
    if not text:
        return None

    switch_words = ("switch", "change", "activate", "enable", "use")
    if "hand" in text and (
        "hand mode" in text
        or "hand control" in text
        or "hand tracking" in text
        or any(word in text for word in switch_words)
    ):
        return _default_plan(
            decision="allow",
            summary="Switch to hand mode",
            reply="Switching to hand mode.",
            steps=[{"action": "control_mode", "mode": "hand"}],
        )

    if "head" in text and (
        "head mode" in text
        or "head control" in text
        or "head tracking" in text
        or "blink mode" in text
        or any(word in text for word in switch_words)
    ):
        return _default_plan(
            decision="allow",
            summary="Switch to head mode",
            reply="Switching to head and blink mode.",
            steps=[{"action": "control_mode", "mode": "head"}],
        )

    # SMART CLOSE SYSTEM
    if "close youtube" in text:
        return _default_plan(
            decision="allow",
            summary="Close YouTube tab",
            reply="Closing YouTube tab",
            steps=[{"action": "hotkey", "keys": ["ctrl", "w"]}],
        )

    if "close" in text:
        # Close tab (browser)
        if "tab" in text or "youtube" in text:
            return _default_plan(
                decision="allow",
                summary="Close tab",
                reply="Closing tab",
                steps=[{"action": "hotkey", "keys": ["ctrl", "w"]}],
            )

        # Close specific app
        for app in ["notepad", "chrome", "edge", "vlc", "word", "excel"]:
            if app in text:
                return _default_plan(
                    decision="allow",
                    summary=f"Close {app}",
                    reply=f"Closing {app}",
                    steps=[{"action": "close_app", "app": app}],
                )

        # Fallback close current window
        return _default_plan(
            decision="allow",
            summary="Close window",
            reply="Closing current window",
            steps=[{"action": "hotkey", "keys": ["alt", "f4"]}],
        )

    if any(word in text for word in ("help", "what can you do")):
        return _default_plan(
            decision="allow",
            summary="Help",
            reply="Here is what I can do.",
            steps=[{"action": "help"}],
        )

    if any(
        phrase in text
        for phrase in ("stop", "stop app", "stop assistant", "quit assistant", "exit assistant")
    ):
        return _default_plan(
            decision="allow",
            summary="Stop app",
            reply="Stopping the app.",
            steps=[{"action": "stop"}],
        )

    if "minimize" in text:
        return _default_plan(
            decision="allow",
            summary="Minimize window",
            reply="Minimizing window.",
            steps=[{"action": "minimize"}],
        )

    if "maximize" in text:
        return _default_plan(
            decision="allow",
            summary="Maximize window",
            reply="Maximizing window.",
            steps=[{"action": "maximize"}],
        )

    if "restore" in text:
        return _default_plan(
            decision="allow",
            summary="Restore window",
            reply="Restoring window.",
            steps=[{"action": "restore"}],
        )

    if "double click" in text:
        return _default_plan(
            decision="allow",
            summary="Double click",
            reply="Double clicking.",
            steps=[{"action": "double_click"}],
        )

    if "right click" in text:
        return _default_plan(
            decision="allow",
            summary="Right click",
            reply="Right clicking.",
            steps=[{"action": "right_click"}],
        )

    if "click" in text:
        return _default_plan(
            decision="allow",
            summary="Click",
            reply="Clicking.",
            steps=[{"action": "left_click"}],
        )

    if "scroll" in text and ("up" in text or "down" in text):
        direction = "up" if "up" in text else "down"
        return _default_plan(
            decision="allow",
            summary=f"Scroll {direction}",
            reply=f"Scrolling {direction}.",
            steps=[
                {
                    "action": "scroll",
                    "direction": direction,
                    "amount": _scroll_amount_from_text(text),
                }
            ],
        )

    if "drag" in text:
        return _default_plan(
            decision="allow",
            summary="Toggle drag",
            reply="Toggling drag mode.",
            steps=[{"action": "drag_toggle"}],
        )

    typed_text = _extract_text_after_command(
        text,
        ("type ", "write ", "input ", "enter text "),
    )
    if typed_text:
        return _default_plan(
            decision="allow",
            summary="Type text",
            reply="Typing your text.",
            steps=[{"action": "type_text", "text": typed_text}],
        )

    for phrases, keys, summary, reply in LOCAL_HOTKEY_COMMANDS:
        if any(text == phrase for phrase in phrases):
            return _default_plan(
                decision="allow",
                summary=summary,
                reply=reply,
                steps=[{"action": "hotkey", "keys": list(keys)}],
            )

    hotkey_match = re.search(
        r"\b(?:press|hit|use)\s+((?:ctrl|control|alt|shift|win|windows|command)(?:\s*(?:\+|and)?\s*(?:ctrl|control|alt|shift|win|windows|command|[a-z0-9]|f\d{1,2}|enter|tab|esc|escape|space))+)",
        text,
    )
    if hotkey_match:
        keys = [
            _normalize_key_name(part)
            for part in re.split(r"\s*(?:\+|and)\s*|\s+", hotkey_match.group(1).strip())
            if part.strip()
        ]
        keys = [key for key in keys if key]
        if len(keys) >= 2 and all(key in pyautogui.KEYBOARD_KEYS for key in keys):
            return _default_plan(
                decision="allow",
                summary="Keyboard shortcut",
                reply="Using that keyboard shortcut.",
                steps=[{"action": "hotkey", "keys": keys[:4]}],
            )

    key_match = re.search(
        r"\b(?:press|hit|tap)\s+(?:the\s+)?(enter|tab|space|escape|esc|delete|backspace|up|down|left|right|home|end|page up|page down|f\d{1,2}|[a-z0-9])\b",
        text,
    )
    if key_match:
        key = _normalize_key_name(key_match.group(1))
        if key in pyautogui.KEYBOARD_KEYS:
            return _default_plan(
                decision="allow",
                summary="Press key",
                reply=f"Pressing {key}.",
                steps=[{"action": "press_key", "key": key}],
            )

    for app_name in SAFE_APP_ALIASES:
        if app_name in text:
            return _default_plan(
                decision="allow",
                summary=f"Open {app_name}",
                reply=f"Opening {app_name}.",
                steps=[{"action": "launch_app", "app": app_name}],
            )

    if "word" in text:
        return _default_plan(
            decision="allow",
            summary="Open Word",
            reply="Opening Microsoft Word",
            steps=[{"action": "launch_app", "app": "winword"}],
        )

    # PERFECT PLAY SYSTEM (FINAL)
    if text.startswith("play "):
        query = text.replace("play", "", 1).strip()

        if query:
            return _default_plan(
                decision="allow",
                summary="Play on YouTube",
                reply=f"Playing {query}",
                steps=[
                    {
                        "action": "open_url",
                        "url": f"https://www.youtube.com/results?search_query={urllib.parse.quote_plus(query)}",
                    },
                    {"action": "wait", "seconds": 2},
                    {"action": "press_key", "key": "tab"},
                    {"action": "press_key", "key": "tab"},
                    {"action": "press_key", "key": "enter"},
                ],
            )

    if text.startswith("open youtube and play "):
        query = text.replace("open youtube and play", "", 1).strip()
        if query:
            return _default_plan(
                decision="allow",
                summary="Play on YouTube",
                reply=f"Playing {query}",
                steps=[
                    {
                        "action": "open_url",
                        "url": f"https://www.youtube.com/results?search_query={urllib.parse.quote_plus(query)}",
                    },
                    {"action": "wait", "seconds": 2},
                    {"action": "press_key", "key": "tab"},
                    {"action": "press_key", "key": "tab"},
                    {"action": "press_key", "key": "enter"},
                ],
            )

    if "youtube" in text and "search" in text:
        query = _extract_text_after_command(text, ("search for ", "search "))
        if query:
            return _default_plan(
                decision="allow",
                summary="Open YouTube search",
                reply=f"Searching YouTube for {query}.",
                steps=[{"action": "open_url", "url": _make_search_url(query, "youtube")}],
            )

    if "youtube" in text:
        query = ""
        search_match = re.search(r"open youtube(?: and)? search(?: for)? (.+)", text)
        if search_match:
            query = search_match.group(1).strip(" .")
        if query:
            return _default_plan(
                decision="allow",
                summary="Open YouTube search",
                reply=f"Searching YouTube for {query}.",
                steps=[{"action": "open_url", "url": _make_search_url(query, "youtube")}],
            )
        return _default_plan(
            decision="allow",
            summary="Open YouTube",
            reply="Opening YouTube.",
            steps=[{"action": "open_url", "url": "https://www.youtube.com"}],
        )

    if text.startswith("search ") or "search for " in text:
        query = _extract_text_after_command(text, ("search for ", "search "))
        if query:
            return _default_plan(
                decision="allow",
                summary="Google search",
                reply=f"Searching Google for {query}.",
                steps=[{"action": "open_url", "url": _make_search_url(query)}],
            )

    if "open google" in text:
        query = ""
        search_match = re.search(r"open google(?: and)? search(?: for)? (.+)", text)
        if search_match:
            query = search_match.group(1).strip(" .")
        if query:
            return _default_plan(
                decision="allow",
                summary="Google search",
                reply=f"Searching Google for {query}.",
                steps=[{"action": "open_url", "url": _make_search_url(query)}],
            )
        return _default_plan(
            decision="allow",
            summary="Open Google",
            reply="Opening Google.",
            steps=[{"action": "open_url", "url": "https://www.google.com"}],
        )

    if text.startswith("open "):
        app_name = text.replace("open", "", 1).strip()
        url = _coerce_safe_url(text[5:].strip())
        if url:
            return _default_plan(
                decision="allow",
                summary="Open website",
                reply="Opening that website.",
                steps=[{"action": "open_url", "url": url}],
            )

        # UNIVERSAL APP OPEN (WINDOWS SEARCH)
        if app_name and len(app_name) > 2:
            return _default_plan(
                decision="allow",
                summary=f"Open {app_name}",
                reply=f"Opening {app_name}",
                steps=[{"action": "system_search", "query": app_name}],
            )

    # FINAL FALLBACK (guaranteed execution)
    if "youtube" in text:
        return _default_plan(
            decision="allow",
            summary="Open YouTube",
            reply="Opening YouTube",
            steps=[{"action": "open_url", "url": "https://www.youtube.com"}],
        )

    if "chrome" in text:
        return _default_plan(
            decision="allow",
            summary="Open Chrome",
            reply="Opening Chrome",
            steps=[{"action": "launch_app", "app": "chrome"}],
        )

    return None


def _plan_task_locally(cmd: str) -> Optional[dict[str, Any]]:
    text = _clean_voice_text(cmd)
    if not text:
        return None

    direct_plan = _plan_single_task_locally(text)
    if direct_plan is not None and _should_prefer_direct_local_plan(text, direct_plan):
        return direct_plan

    for separator in COMPOUND_COMMAND_SEPARATORS:
        if separator not in text:
            continue

        parts = [part.strip(" .") for part in text.split(separator)]
        if len(parts) < 2 or any(not part for part in parts):
            continue

        planned_parts: list[dict[str, Any]] = []
        for part in parts:
            part_plan = _plan_single_task_locally(part)
            if part_plan is None or part_plan.get("decision") != "allow":
                planned_parts = []
                break
            planned_parts.append(part_plan)

        if planned_parts:
            merged_plan = _merge_local_plans(parts, planned_parts)
            if merged_plan is not None:
                logger.debug(
                    "Using local compound plan for '%s' with separator '%s'",
                    text,
                    separator.strip(),
                )
                return merged_plan

    return direct_plan


def _default_plan(
    decision: str = "noop",
    reply: str = "",
    summary: str = "",
    steps: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    return {
        "decision": decision if decision in SUPPORTED_DECISIONS else "noop",
        "summary": summary.strip(),
        "reply": reply.strip(),
        "steps": steps or [],
    }


def _normalize_step(step: Any) -> Optional[dict[str, Any]]:
    if not isinstance(step, dict):
        return None

    action = str(step.get("action", "")).strip().lower()
    if action not in SUPPORTED_STEP_ACTIONS:
        return None

    normalized: dict[str, Any] = {"action": action}

    if action in {"minimize", "maximize", "restore"}:
        pass

    elif action == "control_mode":
        mode = str(step.get("mode", "")).strip().lower()
        if mode not in {"head", "hand"}:
            return None
        normalized["mode"] = mode

    elif action == "scroll":
        direction = str(step.get("direction", "down")).strip().lower()
        if direction not in {"up", "down"}:
            direction = "down"
        normalized["direction"] = direction
        normalized["amount"] = _clamp_int(
            step.get("amount", 300),
            default=300,
            minimum=1,
            maximum=MAX_SCROLL_AMOUNT,
        )

    elif action == "type_text":
        text = str(step.get("text", "")).strip()
        if not text:
            return None
        normalized["text"] = text[:MAX_TYPE_CHARS]

    elif action == "press_key":
        key = _normalize_key_name(str(step.get("key", "")))
        if key not in pyautogui.KEYBOARD_KEYS:
            return None
        normalized["key"] = key

    elif action == "hotkey":
        raw_keys = step.get("keys", [])
        if not isinstance(raw_keys, list):
            return None
        keys = []
        for raw_key in raw_keys[:4]:
            key = _normalize_key_name(str(raw_key))
            if key not in pyautogui.KEYBOARD_KEYS:
                return None
            keys.append(key)
        if len(keys) < 2:
            return None
        normalized["keys"] = keys

    elif action == "open_url":
        url = str(step.get("url", "")).strip()
        if not _is_safe_http_url(url):
            return None
        normalized["url"] = url

    elif action == "system_search":
        query = str(step.get("query", "")).strip()
        if len(query) <= 2:
            return None
        normalized["query"] = query

    elif action == "launch_app":
        app = str(step.get("app", "")).strip().lower()
        if not app:
            return None
        normalized["app"] = app

    elif action == "close_app":
        app = str(step.get("app", "")).strip().lower()
        if not app:
            return None
        normalized["app"] = app

    elif action == "wait":
        normalized["seconds"] = _clamp_float(
            step.get("seconds", 1.0),
            default=1.0,
            minimum=0.2,
            maximum=MAX_WAIT_SECONDS,
        )

    return normalized


def _postprocess_plan_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    processed: list[dict[str, Any]] = []

    for step in steps:
        current = dict(step)

        if current.get("action") == "launch_app":
            app_name = current.get("app", "")
            if app_name in WEB_APP_FALLBACK_URLS:
                current = {"action": "open_url", "url": WEB_APP_FALLBACK_URLS[app_name]}

        if processed:
            previous = processed[-1]

            if (
                previous.get("action") == "launch_app"
                and previous.get("app") in BROWSER_APPS
                and current.get("action") == "open_url"
            ):
                processed.pop()
                previous = processed[-1] if processed else None

            if previous and _step_needs_focus_wait(current):
                if previous.get("action") in {"launch_app", "open_url"}:
                    if previous.get("action") != "wait":
                        processed.append(
                            {"action": "wait", "seconds": LOCAL_STEP_WAIT_SECONDS}
                        )

        processed.append(current)
        if len(processed) >= MAX_PLAN_STEPS:
            break

    return processed[:MAX_PLAN_STEPS]


def _normalize_plan(plan: Optional[dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(plan, dict):
        return _default_plan()

    decision = str(plan.get("decision", "noop")).strip().lower()
    if decision not in SUPPORTED_DECISIONS:
        decision = "noop"

    reply = str(plan.get("reply", "")).strip()
    summary = str(plan.get("summary", "")).strip()
    raw_steps = plan.get("steps", [])
    if not isinstance(raw_steps, list):
        raw_steps = []

    steps: list[dict[str, Any]] = []
    for raw_step in raw_steps[:MAX_PLAN_STEPS]:
        normalized = _normalize_step(raw_step)
        if normalized:
            steps.append(normalized)

    steps = _postprocess_plan_steps(steps)

    if decision == "allow" and not steps:
        decision = "noop"

    return _default_plan(decision=decision, reply=reply, summary=summary, steps=steps)


class OllamaBrain:
    """Local LLM planner that maps speech into safe desktop-action plans."""

    def __init__(
        self,
        model: str = "phi3",
        host: str = "http://127.0.0.1:11434",
        timeout_s: float = 8.0,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.timeout_s = timeout_s
        self.last_error: str = ""

    def _build_prompt(self, utterance: str, drag_mode: bool) -> str:
        return f"""
You are the brain of a Windows voice-controlled assistant.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎯 CORE GOAL
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Convert user voice commands into safe desktop actions.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💻 APPLICATION CONTROL (IMPORTANT)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- The assistant can open ANY application installed on the system.
- Do NOT restrict to predefined apps.
- Use "launch_app" for all applications.

Examples:
- "open word" -> app = "winword"
- "open excel" -> app = "excel"
- "open chrome" -> app = "chrome"
- "open vlc" -> app = "vlc"
- "open photoshop" -> app = "photoshop"

If unsure:
-> use the spoken name directly (e.g., "spotify", "telegram")

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🚫 BLOCKED ACTIONS (VERY IMPORTANT)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DO NOT allow:
- command prompt
- powershell
- terminal
- registry (regedit)
- hacking / bypass / malware
- deleting system files
- disabling security

If detected:
-> decision = "block"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎵 PLAY COMMAND (CRITICAL)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

If user says:
- "play <song>"
- "play <anything>"

-> ALWAYS return:
open_url with YouTube search

Example:
"play tum hi ho"
-> https://www.youtube.com/results?search_query=tum+hi+ho

DO NOT:
- type text
- open google
- launch browser manually

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚡ EXECUTION RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- Prefer DIRECT actions (open_url, launch_app)
- Window control actions available: minimize, maximize, restore
- NEVER use type_text for search
- NEVER mix launch_app + type_text for search
- Keep steps minimal (1-3 steps max)
- No unnecessary steps

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📦 OUTPUT FORMAT (STRICT JSON ONLY)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{{
    "decision": "allow" | "confirm" | "block" | "noop",
    "summary": "short summary",
    "reply": "short reply",
    "steps": [
        {{ "action": "open_url", "url": "..." }},
        {{ "action": "launch_app", "app": "..." }},
        {{ "action": "press_key", "key": "enter" }},
        {{ "action": "hotkey", "keys": ["ctrl", "c"] }},
        {{ "action": "minimize" }},
        {{ "action": "maximize" }},
        {{ "action": "restore" }}
    ]
}}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

User request: {utterance}
"""

    def plan(self, utterance: str, drag_mode: bool = False) -> Optional[dict[str, Any]]:
        self.last_error = ""
        payload = {
            "model": self.model,
            "prompt": self._build_prompt(utterance, drag_mode=drag_mode),
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.0},
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url=f"{self.host}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                body = resp.read().decode("utf-8", errors="ignore")
        except urllib.error.URLError as exc:
            logger.warning("Ollama not reachable: %s", exc)
            self.last_error = (
                "The local brain is not reachable. Start Ollama first with "
                "ollama serve and keep the model available."
            )
            return None
        except Exception as exc:
            logger.warning("Ollama planning failed: %s", exc)
            if "timed out" in str(exc).lower():
                self.last_error = (
                    "The local brain took too long to answer. Start Ollama first "
                    "or increase OLLAMA_TIMEOUT."
                )
            else:
                self.last_error = "The local brain could not finish that request."
            return None

        wrapper = _extract_json_object(body)
        if not wrapper:
            self.last_error = "The local brain returned an unreadable response."
            return None

        response_text = wrapper.get("response", "")
        plan = _extract_json_object(response_text)
        if plan:
            return plan

        if isinstance(wrapper.get("response"), dict):
            response_dict = wrapper["response"]
            if isinstance(response_dict, dict):
                return response_dict

        self.last_error = "The local brain returned an invalid plan."
        return None

    def answer_question(self, utterance: str) -> Optional[str]:
        """Return a short spoken answer for general user questions."""
        self.last_error = ""
        prompt = f"""
You are a helpful voice assistant for accessibility software.
Answer the user question in plain English, short and clear.
Keep it to 1-2 sentences and avoid markdown or bullet points.
If unsure, say you are not fully sure instead of inventing details.

User question: {utterance}
""".strip()

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2},
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url=f"{self.host}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                body = resp.read().decode("utf-8", errors="ignore")
        except urllib.error.URLError as exc:
            logger.warning("Ollama Q&A not reachable: %s", exc)
            self.last_error = (
                "The local brain is not reachable. Start Ollama first with "
                "ollama serve and keep the model available."
            )
            return None
        except Exception as exc:
            logger.warning("Ollama Q&A failed: %s", exc)
            if "timed out" in str(exc).lower():
                self.last_error = (
                    "The local brain took too long to answer. Start Ollama first "
                    "or increase OLLAMA_TIMEOUT."
                )
            else:
                self.last_error = "The local brain could not answer that right now."
            return None

        wrapper = _extract_json_object(body)
        if not wrapper:
            self.last_error = "The local brain returned an unreadable response."
            return None

        response_text = wrapper.get("response", "")
        if not isinstance(response_text, str):
            self.last_error = "The local brain returned an invalid answer."
            return None

        cleaned = response_text.strip().replace("\n", " ")
        cleaned = re.sub(r"\s+", " ", cleaned)
        if not cleaned:
            self.last_error = "The local brain returned an empty answer."
            return None
        return cleaned[:320]


class CloudBrain:
    """Optional OpenAI-compatible cloud planner and Q&A fallback."""

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        timeout_s: float = DEFAULT_CLOUD_TIMEOUT_S,
    ) -> None:
        api_key = api_key.strip()
        if not api_key:
            raise ValueError("Cloud brain requires a non-empty API key.")
        self.model = model.strip() or "gpt-4o-mini"
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.last_error: str = ""

    def _request_chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
    ) -> Optional[str]:
        self.last_error = ""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url=f"{self.base_url}/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                body = resp.read().decode("utf-8", errors="ignore")
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="ignore")
            except Exception:
                detail = str(exc)
            logger.warning("Cloud brain HTTP error: %s", detail or exc)
            self.last_error = (
                "The cloud brain rejected the request. Check the API key, "
                "model name, and endpoint configuration."
            )
            return None
        except urllib.error.URLError as exc:
            logger.warning("Cloud brain not reachable: %s", exc)
            self.last_error = (
                "The cloud brain is not reachable right now. Check the internet "
                "connection and the configured endpoint."
            )
            return None
        except Exception as exc:
            logger.warning("Cloud brain request failed: %s", exc)
            if "timed out" in str(exc).lower():
                self.last_error = (
                    "The cloud brain took too long to answer. Try again or "
                    "increase CLOUD_BRAIN_TIMEOUT."
                )
            else:
                self.last_error = "The cloud brain could not finish that request."
            return None

        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            self.last_error = "The cloud brain returned an unreadable response."
            return None

        choices = parsed.get("choices", [])
        if not isinstance(choices, list) or not choices:
            self.last_error = "The cloud brain returned no answer."
            return None

        message = choices[0].get("message", {})
        if not isinstance(message, dict):
            self.last_error = "The cloud brain returned an invalid answer."
            return None

        content = message.get("content", "")
        if not isinstance(content, str) or not content.strip():
            self.last_error = "The cloud brain returned an empty answer."
            return None
        return content.strip()

    def plan(self, utterance: str, drag_mode: bool = False) -> Optional[dict[str, Any]]:
        system_prompt = """
You are the cloud fallback planner for a Windows voice-controlled accessibility assistant.
Convert user voice commands into safe desktop actions.

Rules:
- Keep actions minimal and safe.
- Block dangerous requests involving shells, deleting files, security bypasses, or secrets.
- Use direct actions when possible.
- Window control actions available: minimize, maximize, restore.
- For play/search requests, prefer a direct open_url result instead of typing into an app.
- Return strict JSON only with this schema:
{
  "decision": "allow" | "confirm" | "block" | "noop",
  "summary": "short summary",
  "reply": "short reply",
  "steps": [
    { "action": "open_url", "url": "..." },
    { "action": "launch_app", "app": "..." },
    { "action": "press_key", "key": "enter" },
    { "action": "hotkey", "keys": ["ctrl", "c"] },
    { "action": "minimize" },
    { "action": "maximize" },
    { "action": "restore" }
  ]
}
""".strip()
        user_prompt = (
            f"User request: {utterance}\n"
            f"Drag mode active: {'yes' if drag_mode else 'no'}"
        )

        response_text = self._request_chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
        )
        if not response_text:
            return None

        plan = _extract_json_object(response_text)
        if plan:
            return plan

        self.last_error = "The cloud brain returned an invalid plan."
        return None

    def answer_question(self, utterance: str) -> Optional[str]:
        response_text = self._request_chat(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful accessibility voice assistant. "
                        "Answer in plain English, short and clear, in 1 or 2 sentences."
                    ),
                },
                {"role": "user", "content": utterance},
            ],
            temperature=0.2,
        )
        if not response_text:
            return None

        cleaned = response_text.strip().replace("\n", " ")
        cleaned = re.sub(r"\s+", " ", cleaned)
        if not cleaned:
            self.last_error = "The cloud brain returned an empty answer."
            return None
        return cleaned[:320]


def _looks_like_question(text: str) -> bool:
    """Heuristic to route general knowledge questions to the local brain."""
    cleaned = _clean_voice_text(text)
    if not cleaned:
        return False

    question_prefixes = (
        "what ",
        "why ",
        "how ",
        "who ",
        "when ",
        "where ",
        "which ",
        "whom ",
        "whose ",
        "is ",
        "are ",
        "can ",
        "could ",
        "do ",
        "does ",
        "did ",
        "will ",
        "would ",
        "should ",
        "tell me ",
        "explain ",
    )
    return cleaned.startswith(question_prefixes)


def _strip_chat_prefix(text: str) -> str:
    cleaned = _clean_voice_text(text)
    if not cleaned:
        return ""

    for prefix in CHAT_PREFIXES:
        if cleaned == prefix:
            return ""
        if cleaned.startswith(f"{prefix} "):
            return cleaned[len(prefix) :].strip()
    return cleaned


def _should_route_to_chat(text: str) -> bool:
    cleaned = _clean_voice_text(text)
    if not cleaned:
        return False
    if _looks_like_desktop_action_request(cleaned):
        return False
    if _answer_project_faq(cleaned):
        return True
    if _looks_like_question(cleaned):
        return True
    if "chat bot" in cleaned or "chatbot" in cleaned:
        return True
    return any(cleaned.startswith(prefix) for prefix in CHAT_PREFIXES)


def _looks_like_desktop_action_request(text: str) -> bool:
    """Return True when a question-shaped phrase is really a command."""
    cleaned = _clean_voice_text(text)
    if not cleaned:
        return False

    normalized = _normalize_request_text(cleaned)
    action_prefixes = (
        "open ",
        "close ",
        "type ",
        "write ",
        "input ",
        "enter text ",
        "press ",
        "hit ",
        "tap ",
        "use ctrl",
        "use control",
        "use alt",
        "click",
        "double click",
        "right click",
        "left click",
        "scroll ",
        "drag",
        "minimize",
        "maximize",
        "restore",
        "search ",
        "search for ",
        "play ",
    )
    exact_actions = {
        "copy",
        "paste",
        "cut",
        "undo",
        "redo",
        "select all",
        "save",
        "new tab",
        "close tab",
        "find",
        "stop",
        "stop app",
        "stop assistant",
        "quit assistant",
        "exit assistant",
    }
    return normalized in exact_actions or normalized.startswith(action_prefixes)


def _answer_project_faq(text: str) -> Optional[str]:
    cleaned = _clean_voice_text(text)
    if not cleaned:
        return None

    project_terms = (
        "blink click",
        "virtual mouse",
        "project",
        "this app",
        "software",
        "system",
        "assistant",
    )
    has_project_context = any(term in cleaned for term in project_terms)

    if ("what is" in cleaned or "about" in cleaned) and has_project_context:
        return PROJECT_FAQ[0][1]
    if (
        "how it works" in cleaned
        or "how does it work" in cleaned
        or ("working" in cleaned and has_project_context)
    ):
        return PROJECT_FAQ[1][1]
    if "what can you do" in cleaned or (
        any(term in cleaned for term in ("features", "capabilities"))
        and has_project_context
    ):
        return PROJECT_FAQ[2][1]
    if (
        "wake word" in cleaned
        or "voice command" in cleaned
        or ("voice" in cleaned and has_project_context)
    ):
        return PROJECT_FAQ[3][1]
    if (
        "how to run" in cleaned
        or "run project" in cleaned
        or "start project" in cleaned
        or ("launch" in cleaned and has_project_context)
    ):
        return PROJECT_FAQ[4][1]
    if "exit" in cleaned or "quit" in cleaned or ("stop" in cleaned and has_project_context):
        return PROJECT_FAQ[5][1]
    if (
        "camera" in cleaned
        or "webcam" in cleaned
        or ("not opening" in cleaned and has_project_context)
    ):
        return PROJECT_FAQ[6][1]
    if "requirements" in cleaned or "prerequisites" in cleaned:
        return PROJECT_FAQ[7][1]

    if (
        "head tracking" in cleaned
        or "cursor control" in cleaned
        or (("head" in cleaned or "cursor" in cleaned) and has_project_context)
    ):
        return "Head movement is mapped to the screen using the nose tip landmark and smoothed to reduce jitter."
    if "voice command" in cleaned or ("command" in cleaned and has_project_context):
        return "Voice commands start with the wake word, then you can say open, type, scroll, press keys, or ask a question."

    return None


def plan_task(
    cmd: str,
    brain: Optional[OllamaBrain],
    cloud_brain: Optional[CloudBrain] = None,
    drag_mode: bool = False,
) -> dict[str, Any]:
    """Plan a free-form command via the local brain."""
    normalized_cmd = _normalize_request_text(cmd)
    local_plan = _plan_task_locally(normalized_cmd)
    if local_plan is not None:
        logger.debug(
            "Using local voice plan for '%s' -> %s",
            normalized_cmd or cmd,
            local_plan.get("summary", "local"),
        )
        return local_plan

    if brain is None and cloud_brain is None:
        return _default_plan(
            reply=(
                "I didn't understand that in rule mode. Try a simpler command like "
                "open chrome or type hello, or enable Ollama or a cloud brain "
                "for more natural requests."
            )
        )

    last_error = ""

    if brain is not None:
        raw_plan = brain.plan(normalized_cmd or cmd, drag_mode=drag_mode)
        if raw_plan:
            logger.debug("Using Ollama voice plan for '%s'", normalized_cmd or cmd)
            return _normalize_plan(raw_plan)
        last_error = brain.last_error

    if cloud_brain is not None:
        raw_plan = cloud_brain.plan(normalized_cmd or cmd, drag_mode=drag_mode)
        if raw_plan:
            logger.debug("Using cloud voice plan for '%s'", normalized_cmd or cmd)
            return _normalize_plan(raw_plan)
        last_error = cloud_brain.last_error or last_error

    return _default_plan(
        reply=last_error or "I heard you, but I could not understand that request safely."
    )


try:
    import speech_recognition as sr

    SR_AVAILABLE = True
except ImportError:
    SR_AVAILABLE = False
    logger.warning("speech_recognition not installed - voice input disabled.")

try:
    import pyttsx3

    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    logger.warning("pyttsx3 not installed - TTS disabled.")


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
) -> tuple[Optional[int], Optional[str]]:
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
    """Non-blocking text-to-speech assistant."""

    GREETING = (
        "Hello. I am ready. Say Jarvis, then tell me what you want to do. "
        "I can help with safe desktop actions like typing, scrolling, clicks, "
        "copy and paste, opening apps, and opening websites."
    )

    def __init__(self, rate: int = 165, volume: float = 1.0) -> None:
        self._q: queue.Queue[str] = queue.Queue()
        self._stopped = False
        self._rate = rate
        self._volume = volume
        self._speaking = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        logger.info("AssistantVoice started (rate=%d).", rate)

    def _worker(self) -> None:
        """Each message uses a fresh pyttsx3 engine to avoid thread issues."""
        while not self._stopped:
            try:
                text = self._q.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                self._speaking.set()
                engine = pyttsx3.init()
                engine.setProperty("rate", self._rate)
                engine.setProperty("volume", self._volume)

                voices = engine.getProperty("voices")
                for voice in voices:
                    if "female" in voice.name.lower() or "zira" in voice.name.lower():
                        engine.setProperty("voice", voice.id)
                        break

                engine.say(text)
                engine.runAndWait()
                engine.stop()
            except Exception as exc:
                logger.error("TTS error: %s", exc)
            finally:
                self._speaking.clear()
                self._q.task_done()

    def say(self, text: str) -> None:
        """Add *text* to the speech queue."""
        logger.info("[Assistant] %s", text)
        print(f"[Assistant] {text}")
        self._q.put(text)

    def greet(self) -> None:
        self.say(self.GREETING)

    @property
    def is_speaking(self) -> bool:
        return self._speaking.is_set()

    def stop(self) -> None:
        self._stopped = True
        logger.info("AssistantVoice stopped.")


class VoiceController:
    """
    Continuously listens via the microphone.

    Speech is only forwarded to the command queue after the wake word is heard.
    Example:
        "jarvis open youtube"
        "jarvis" -> then next phrase becomes the command
    """

    HELP_TEXT = (
        "Say Jarvis, then speak naturally. For example: open YouTube and search "
        "lofi music, open chrome, type hello, press enter, copy, paste, or stop. "
        "For chat, say Jarvis ask <your question>."
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
        wake_word: str = DEFAULT_WAKE_WORD,
        command_window_s: float = DEFAULT_COMMAND_WINDOW_S,
        acknowledge_wake: bool = True,
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
        self.wake_word = _clean_voice_text(wake_word) or DEFAULT_WAKE_WORD
        configured_language = os.environ.get("VOICE_LANGUAGE", "en-IN").strip() or "en-IN"
        self.languages = _parse_language_codes(
            os.environ.get("VOICE_LANGUAGES"),
            configured_language,
        )
        self.language = self.languages[0]
        self.last_language = self.language
        self.command_window_s = max(2.0, float(command_window_s))
        self.acknowledge_wake = acknowledge_wake
        self.strict_wake_word = _parse_bool_env(
            os.environ.get("VOICE_STRICT_WAKE_WORD"),
            True,
        )
        self.wake_word_allow_fuzzy = _parse_bool_env(
            os.environ.get("VOICE_WAKE_FUZZY_MATCH"),
            False,
        )
        self.conversation_mode_enabled = _parse_bool_env(
            os.environ.get("VOICE_CONVERSATION_MODE"),
            False,
        )
        if self.strict_wake_word:
            self.conversation_mode_enabled = False
        self.conversation_timeout_s = _clamp_float(
            os.environ.get(
                "VOICE_CONVERSATION_TIMEOUT_S",
                DEFAULT_CONVERSATION_TIMEOUT_S,
            ),
            default=DEFAULT_CONVERSATION_TIMEOUT_S,
            minimum=3.0,
            maximum=30.0,
        )
        self.far_field_mode = _parse_bool_env(
            os.environ.get("VOICE_FAR_FIELD_MODE"),
            True,
        )
        self.noise_reduction_enabled = _parse_bool_env(
            os.environ.get("VOICE_NOISE_REDUCTION"),
            self.far_field_mode,
        )
        self.audio_target_rms = _clamp_float(
            os.environ.get(
                "VOICE_TARGET_RMS",
                DEFAULT_AUDIO_TARGET_RMS if self.far_field_mode else 0.0,
            ),
            default=DEFAULT_AUDIO_TARGET_RMS,
            minimum=0.0,
            maximum=0.35,
        )
        self.audio_max_gain = _clamp_float(
            os.environ.get("VOICE_MAX_GAIN", DEFAULT_AUDIO_MAX_GAIN),
            default=DEFAULT_AUDIO_MAX_GAIN,
            minimum=1.0,
            maximum=8.0,
        )
        self.noise_gate_level = _clamp_float(
            os.environ.get(
                "VOICE_NOISE_GATE_LEVEL",
                DEFAULT_NOISE_GATE_LEVEL if self.far_field_mode else 0.0,
            ),
            default=DEFAULT_NOISE_GATE_LEVEL,
            minimum=0.0,
            maximum=0.08,
        )
        self.wake_word_aliases = {"jarvish", "jarves"} if self.wake_word == "jarvis" else set()
        self.wake_word_aliases.update(_parse_phrase_list(os.environ.get("WAKE_WORD_ALIASES")))
        self.wake_word_match_ratio = _clamp_float(
            os.environ.get("WAKE_WORD_MATCH_RATIO", DEFAULT_WAKE_WORD_MATCH_RATIO),
            default=DEFAULT_WAKE_WORD_MATCH_RATIO,
            minimum=0.75,
            maximum=0.98,
        )
        self.wake_word_prefix_tokens = _clamp_int(
            os.environ.get("WAKE_WORD_PREFIX_TOKENS", DEFAULT_WAKE_WORD_PREFIX_TOKENS),
            default=DEFAULT_WAKE_WORD_PREFIX_TOKENS,
            minimum=1,
            maximum=4,
        )
        self.wake_word_cooldown_s = _clamp_float(
            os.environ.get("WAKE_WORD_COOLDOWN_S", DEFAULT_WAKE_WORD_COOLDOWN_S),
            default=DEFAULT_WAKE_WORD_COOLDOWN_S,
            minimum=0.0,
            maximum=5.0,
        )
        self.listen_timeout_s = _clamp_float(
            os.environ.get("WAKE_LISTEN_TIMEOUT_S", DEFAULT_WAKE_LISTEN_TIMEOUT_S),
            default=DEFAULT_WAKE_LISTEN_TIMEOUT_S,
            minimum=1.0,
            maximum=6.0,
        )
        self.wake_phrase_limit_s = _clamp_float(
            os.environ.get("WAKE_PHRASE_LIMIT_S", DEFAULT_WAKE_PHRASE_LIMIT_S),
            default=DEFAULT_WAKE_PHRASE_LIMIT_S,
            minimum=1.0,
            maximum=6.0,
        )
        self.command_phrase_limit_s = _clamp_float(
            os.environ.get("COMMAND_PHRASE_LIMIT_S", DEFAULT_COMMAND_PHRASE_LIMIT_S),
            default=DEFAULT_COMMAND_PHRASE_LIMIT_S,
            minimum=2.0,
            maximum=10.0,
        )
        self.error_feedback_cooldown_s = _clamp_float(
            os.environ.get(
                "VOICE_ERROR_FEEDBACK_COOLDOWN_S",
                DEFAULT_ERROR_FEEDBACK_COOLDOWN_S,
            ),
            default=DEFAULT_ERROR_FEEDBACK_COOLDOWN_S,
            minimum=2.0,
            maximum=20.0,
        )
        self.mic_listen_window_s = _clamp_float(
            os.environ.get("VOICE_MIC_LISTEN_WINDOW_S", DEFAULT_MIC_LISTEN_WINDOW_S),
            default=DEFAULT_MIC_LISTEN_WINDOW_S,
            minimum=3.0,
            maximum=60.0,
        )
        self.mic_cycle_pause_s = _clamp_float(
            os.environ.get("VOICE_MIC_CYCLE_PAUSE_S", DEFAULT_MIC_CYCLE_PAUSE_S),
            default=DEFAULT_MIC_CYCLE_PAUSE_S,
            minimum=0.1,
            maximum=10.0,
        )
        self.auto_mic_cycle = _parse_bool_env(
            os.environ.get("VOICE_MIC_AUTO_CYCLE"),
            not self.strict_wake_word,
        )
        self.mic_enabled = _parse_bool_env(
            os.environ.get("VOICE_MIC_ENABLED"),
            True,
        )
        self.ambient_refresh_s = _clamp_float(
            os.environ.get(
                "VOICE_AMBIENT_REFRESH_S",
                DEFAULT_AMBIENT_REFRESH_S,
            ),
            default=DEFAULT_AMBIENT_REFRESH_S,
            minimum=0.0,
            maximum=300.0,
        )
        self.ambient_sample_s = _clamp_float(
            os.environ.get(
                "VOICE_AMBIENT_SAMPLE_S",
                DEFAULT_AMBIENT_SAMPLE_S,
            ),
            default=DEFAULT_AMBIENT_SAMPLE_S,
            minimum=0.15,
            maximum=2.0,
        )

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

        self.listening: bool = False
        self.last_heard: str = ""
        self.last_matched: str = ""
        self.last_heard_time: float = 0.0
        self.awaiting_command_until: float = 0.0
        self.conversation_active_until: float = 0.0
        self._last_wake_time: float = 0.0
        self._last_feedback_time: float = 0.0
        self._awaiting_feedback_sent = False
        self._audio_preprocess_enabled = bool(
            self.noise_reduction_enabled or self.audio_target_rms > 0.0 or self.noise_gate_level > 0.0
        )
        self._state_lock = threading.Lock()
        self._mic_cycle_pause_until: float = 0.0
        self._mic_cycle_deadline: float = 0.0
        self._next_ambient_refresh_at: float = 0.0

        if self.mic_enabled and self.auto_mic_cycle:
            self._mic_cycle_deadline = time.time() + self.mic_listen_window_s
        if self.ambient_refresh_s > 0.0:
            self._next_ambient_refresh_at = time.time() + self.ambient_refresh_s

        if self._audio_preprocess_enabled and np is None:
            logger.warning("NumPy is unavailable; voice audio preprocessing is disabled.")
        elif self.noise_reduction_enabled and nr is None:
            logger.info(
                "Optional noisereduce package not found; using normalization and "
                "noise gate only for far-field speech enhancement."
            )

        self.recognizer.energy_threshold = energy_threshold
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.dynamic_energy_adjustment_damping = 0.15
        self.recognizer.dynamic_energy_adjustment_ratio = 1.5
        self.recognizer.pause_threshold = pause_threshold
        self.recognizer.phrase_threshold = phrase_threshold
        self.recognizer.non_speaking_duration = pause_threshold

        logger.info("Calibrating microphone...")
        print("[Voice] Calibrating microphone...")
        if available_names:
            print(f"[Voice] Using microphone: {self.mic_name or 'System default'}")
        try:
            with self.mic as source:
                self.recognizer.adjust_for_ambient_noise(
                    source, duration=calibration_duration
                )
            logger.info("Microphone calibrated.")
            logger.info(
                "Voice recognizer ready (device_index=%s, device_name=%s, languages=%s, energy_threshold=%.2f, wake_phrase_limit=%.2fs, command_phrase_limit=%.2fs, conversation_mode=%s)",
                self.mic_index,
                self.mic_name or "default",
                ",".join(self.languages),
                self.recognizer.energy_threshold,
                self.wake_phrase_limit_s,
                self.command_phrase_limit_s,
                self.conversation_mode_enabled,
            )
            print("[Voice] Microphone ready.")
            print(f"[Voice] Wake word: '{self.wake_word}'")
            print(f"[Voice] Languages: '{', '.join(self.languages)}'")
            if self.conversation_mode_enabled:
                print(
                    "[Voice] Conversation mode: "
                    f"{self.conversation_timeout_s:.0f}s follow-up window"
                )
            if self._audio_preprocess_enabled:
                print("[Voice] Far-field enhancement: enabled")
        except Exception as exc:
            self.last_error = str(exc)
            logger.error("Microphone calibration failed: %s", exc)
            print(f"[Voice] Microphone error: {exc}")

        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()

    def _reset_mic_cycle_locked(self, now: Optional[float] = None) -> None:
        current_time = time.time() if now is None else now
        self._mic_cycle_pause_until = 0.0
        self._mic_cycle_deadline = current_time + self.mic_listen_window_s

    def set_mic_enabled(self, enabled: bool) -> None:
        with self._state_lock:
            self.mic_enabled = bool(enabled)
            if self.mic_enabled and self.auto_mic_cycle:
                self._reset_mic_cycle_locked()
            else:
                self._mic_cycle_pause_until = 0.0
                self._mic_cycle_deadline = 0.0
                self.awaiting_command_until = 0.0
                self.conversation_active_until = 0.0
                self.last_matched = ""

    def set_auto_mic_cycle(self, enabled: bool) -> None:
        with self._state_lock:
            self.auto_mic_cycle = bool(enabled)
            if self.auto_mic_cycle and self.mic_enabled:
                self._reset_mic_cycle_locked()
            else:
                self._mic_cycle_pause_until = 0.0
                self._mic_cycle_deadline = 0.0

    def _expire_conversation_window(self, now: Optional[float] = None) -> None:
        current_time = time.time() if now is None else now
        if self.conversation_active_until and current_time >= self.conversation_active_until:
            self.conversation_active_until = 0.0

    def _conversation_is_active(self, now: Optional[float] = None) -> bool:
        current_time = time.time() if now is None else now
        self._expire_conversation_window(current_time)
        return (
            self.conversation_mode_enabled
            and self.conversation_active_until > 0.0
            and current_time < self.conversation_active_until
        )

    def _activate_conversation_window(self, now: float) -> None:
        if not self.conversation_mode_enabled:
            return
        self.conversation_active_until = now + self.conversation_timeout_s

    def _allows_follow_up_without_wake_word(self, now: Optional[float] = None) -> bool:
        current_time = time.time() if now is None else now
        self._expire_conversation_window(current_time)
        return (
            current_time < self.awaiting_command_until
            or self._conversation_is_active(current_time)
        )

    def _maybe_refresh_ambient_noise(self, now: float) -> None:
        if self.ambient_refresh_s <= 0.0 or now < self._next_ambient_refresh_at:
            return

        self._next_ambient_refresh_at = now + self.ambient_refresh_s
        if self._allows_follow_up_without_wake_word(now):
            return

        try:
            with self.mic as source:
                self.recognizer.adjust_for_ambient_noise(
                    source,
                    duration=self.ambient_sample_s,
                )
            logger.debug(
                "Ambient profile refreshed (sample=%.2fs, energy=%.2f)",
                self.ambient_sample_s,
                self.recognizer.energy_threshold,
            )
        except Exception as exc:
            logger.debug("Ambient refresh skipped: %s", exc)

    def _ordered_languages(self) -> list[str]:
        ordered: list[str] = []
        if self.last_language:
            ordered.append(self.last_language)
        for language in self.languages:
            if language not in ordered:
                ordered.append(language)
        return ordered or [self.language]

    def _preprocess_audio(self, audio: "sr.AudioData") -> "sr.AudioData":
        if not self._audio_preprocess_enabled or np is None:
            return audio

        try:
            raw = audio.get_raw_data(convert_width=2)
        except Exception:
            return audio

        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        if samples.size == 0:
            return audio

        samples /= 32768.0
        audio_duration_s = samples.size / max(float(audio.sample_rate), 1.0)
        initial_rms = float(np.sqrt(np.mean(np.square(samples))))

        if self.noise_gate_level > 0.0:
            quiet_mask = np.abs(samples) < self.noise_gate_level
            if np.any(quiet_mask):
                samples[quiet_mask] *= 0.20

        if self.audio_target_rms > 0.0:
            rms = float(np.sqrt(np.mean(np.square(samples))))
            if rms > 1e-4:
                gain = min(self.audio_max_gain, self.audio_target_rms / rms)
                if gain > 1.0:
                    samples *= gain

        if self.noise_reduction_enabled and nr is not None:
            should_reduce_noise = (
                audio_duration_s >= DEFAULT_MIN_NR_AUDIO_S
                and (
                    self.audio_target_rms <= 0.0
                    or initial_rms
                    <= (self.audio_target_rms * DEFAULT_NR_SKIP_RMS_FACTOR)
                )
            )
            if should_reduce_noise:
                try:
                    samples = nr.reduce_noise(
                        y=samples,
                        sr=audio.sample_rate,
                        stationary=True,
                        prop_decrease=0.75,
                    )
                except Exception as exc:
                    logger.debug("Noise reduction skipped after processing error: %s", exc)

        samples = np.clip(samples, -1.0, 1.0)
        processed = (samples * 32767.0).astype(np.int16)
        return sr.AudioData(processed.tobytes(), audio.sample_rate, 2)

    def _is_actionable_transcript(self, text: str, now: float) -> bool:
        cleaned = _clean_voice_text(text)
        if not cleaned:
            return False
        if self._allows_follow_up_without_wake_word(now):
            return True
        return self._extract_after_wake_word(cleaned)[0]

    def _recognize_audio(
        self,
        audio: "sr.AudioData",
        now: float,
    ) -> tuple[str, str, list[str]]:
        best_result: Optional[tuple[str, str, list[str]]] = None
        request_error: Optional[Exception] = None

        for language in self._ordered_languages():
            try:
                recognition = self.recognizer.recognize_google(
                    audio,
                    language=language,
                    show_all=True,
                )
                transcripts = _extract_transcript_candidates(recognition)
                raw_text = ""
                if transcripts:
                    raw_text = self._pick_transcript_candidate(transcripts, now=now)
                elif isinstance(recognition, str):
                    raw_text = recognition
                elif isinstance(recognition, dict):
                    alternatives = recognition.get("alternative", [])
                    if isinstance(alternatives, list):
                        for item in alternatives:
                            if not isinstance(item, dict):
                                continue
                            candidate = item.get("transcript")
                            if isinstance(candidate, str) and candidate.strip():
                                raw_text = candidate.strip()
                                break

                if not raw_text:
                    raw_text = self.recognizer.recognize_google(
                        audio,
                        language=language,
                    )
            except sr.UnknownValueError:
                continue
            except sr.RequestError as exc:
                request_error = exc
                break

            cleaned = _clean_voice_text(raw_text)
            if not cleaned:
                continue

            result = (language, raw_text, transcripts)
            if best_result is None or len(cleaned) > len(_clean_voice_text(best_result[1])):
                best_result = result

            if self._is_actionable_transcript(cleaned, now):
                return result

        if best_result is not None:
            return best_result
        if request_error is not None:
            raise request_error
        raise sr.UnknownValueError()

    def _token_matches_wake_word(self, token: str, allow_fuzzy: bool = True) -> bool:
        candidate = _clean_voice_text(token)
        if not candidate:
            return False
        if candidate == self.wake_word or candidate in self.wake_word_aliases:
            return True
        if not allow_fuzzy:
            return False
        if abs(len(candidate) - len(self.wake_word)) > 1:
            return False
        similarity = difflib.SequenceMatcher(None, candidate, self.wake_word).ratio()
        if similarity >= self.wake_word_match_ratio:
            logger.debug(
                "Wake word fuzzy match '%s' -> '%s' (ratio=%.2f)",
                candidate,
                self.wake_word,
                similarity,
            )
            return True
        return False

    def _extract_after_wake_word(self, text: str) -> tuple[bool, str]:
        tokens = text.split()
        for index, token in enumerate(tokens):
            if self.strict_wake_word and index:
                prefix_tokens = tokens[:index]
                if any(prefix not in {"hey", "ok", "okay"} for prefix in prefix_tokens):
                    continue
            allow_fuzzy = (
                self.wake_word_allow_fuzzy
                and index < self.wake_word_prefix_tokens
            )
            if self._token_matches_wake_word(token, allow_fuzzy=allow_fuzzy):
                remainder = tokens[index + 1 :]
                while remainder and self._token_matches_wake_word(
                    remainder[0],
                    allow_fuzzy=False,
                ):
                    remainder = remainder[1:]
                return True, " ".join(remainder).strip()
        return False, ""

    def _maybe_speak_feedback(self, text: str, now: Optional[float] = None) -> None:
        current_time = time.time() if now is None else now
        if current_time - self._last_feedback_time < self.error_feedback_cooldown_s:
            return
        self._last_feedback_time = current_time
        if self.assistant and text:
            self.assistant.say(text)

    def _expire_command_window(self, now: Optional[float] = None) -> None:
        current_time = time.time() if now is None else now
        if not self.awaiting_command_until or current_time < self.awaiting_command_until:
            return

        self.awaiting_command_until = 0.0
        if not self._awaiting_feedback_sent:
            self._maybe_speak_feedback(
                "I didn't catch that. Say the wake word again and try once more.",
                now=current_time,
            )
            self._awaiting_feedback_sent = True
        self.last_matched = ""

    def _pick_transcript_candidate(
        self,
        transcripts: list[str],
        now: Optional[float] = None,
    ) -> str:
        cleaned_candidates: list[str] = []
        for transcript in transcripts:
            cleaned = _clean_voice_text(transcript)
            if cleaned and cleaned not in cleaned_candidates:
                cleaned_candidates.append(cleaned)

        if not cleaned_candidates:
            return ""

        current_time = time.time() if now is None else now
        allow_follow_up = self._allows_follow_up_without_wake_word(current_time)
        action_tokens = {
            "open",
            "search",
            "scroll",
            "type",
            "press",
            "click",
            "double",
            "left",
            "right",
            "minimize",
            "maximize",
            "restore",
            "stop",
            "confirm",
            "cancel",
            "copy",
            "paste",
            "undo",
            "redo",
            "ask",
            "question",
            "help",
            "close",
        }

        best_candidate = cleaned_candidates[0]
        best_score = float("-inf")

        for index, candidate in enumerate(cleaned_candidates):
            has_wake_word, remainder = self._extract_after_wake_word(candidate)
            command_text = remainder if has_wake_word and remainder else candidate
            tokens = command_text.split()
            first_token = tokens[0] if tokens else ""
            token_count = len(tokens)

            score = 0.0
            if has_wake_word:
                score += 3.0
            elif not allow_follow_up:
                score -= 0.9

            if first_token in action_tokens:
                score += 1.0
            if any(
                command_text.startswith(prefix)
                for prefix in (
                    "open ",
                    "search ",
                    "scroll ",
                    "type ",
                    "press ",
                    "ask ",
                    "question ",
                )
            ):
                score += 0.5

            if 1 <= token_count <= 12:
                score += 0.4
            elif token_count > 16:
                score -= 0.4

            score += min(len(command_text), 48) / 160.0
            score -= index * 0.18

            if score > best_score:
                best_score = score
                best_candidate = candidate

        return best_candidate

    def _consume_transcript(self, text: str, now: float) -> Optional[str]:
        cleaned = _clean_voice_text(text)
        if not cleaned:
            return None

        self._expire_command_window(now)
        self._expire_conversation_window(now)
        has_wake_word, remainder = self._extract_after_wake_word(cleaned)

        if self._conversation_is_active(now):
            if has_wake_word:
                if remainder:
                    self.last_matched = remainder
                    self._activate_conversation_window(now)
                    return remainder
                self.last_matched = self.wake_word
                self._activate_conversation_window(now)
                return None

            self.awaiting_command_until = 0.0
            self._awaiting_feedback_sent = False
            self.last_matched = cleaned
            self._activate_conversation_window(now)
            return cleaned

        if now < self.awaiting_command_until:
            if has_wake_word:
                if remainder:
                    cleaned = remainder
                else:
                    self.last_matched = self.wake_word
                    self.awaiting_command_until = now + self.command_window_s
                    return None
            self.awaiting_command_until = 0.0
            self._awaiting_feedback_sent = False
            self.last_matched = cleaned
            self._activate_conversation_window(now)
            return cleaned

        if not has_wake_word:
            return None

        if remainder:
            self.last_matched = remainder
            self._activate_conversation_window(now)
            return remainder

        if now - self._last_wake_time < self.wake_word_cooldown_s:
            logger.debug(
                "Ignored repeated wake word within cooldown (%.2fs)",
                self.wake_word_cooldown_s,
            )
            return None

        self._last_wake_time = now
        self.awaiting_command_until = now + self.command_window_s
        self._awaiting_feedback_sent = False
        self.last_matched = self.wake_word
        logger.info("Wake word detected.")
        print(f"[Voice] Wake word detected: '{self.wake_word}'")
        if self.assistant and self.acknowledge_wake:
            self.assistant.say("Yes?")
        return None

    def authorize_transcript(self, text: str) -> Optional[str]:
        """Return command text only when the transcript passes the wake gate."""
        cleaned = _clean_voice_text(text)
        if not cleaned:
            return None
        now = time.time()
        self.last_heard = cleaned
        self.last_heard_time = now
        return self._consume_transcript(cleaned, now)

    def _listen_loop(self) -> None:
        while not self.stopped:
            try:
                now = time.time()
                with self._state_lock:
                    mic_enabled = self.mic_enabled
                    auto_mic_cycle = self.auto_mic_cycle
                    cycle_deadline = self._mic_cycle_deadline
                    cycle_pause_until = self._mic_cycle_pause_until

                    if not mic_enabled:
                        self.listening = False
                        self.awaiting_command_until = 0.0
                        self.conversation_active_until = 0.0
                        self.last_matched = ""

                    elif auto_mic_cycle:
                        if cycle_deadline <= 0.0:
                            self._reset_mic_cycle_locked(now)
                            cycle_deadline = self._mic_cycle_deadline

                        if now >= cycle_deadline:
                            if now < self.awaiting_command_until:
                                self._reset_mic_cycle_locked(now)
                                cycle_deadline = self._mic_cycle_deadline
                            else:
                                self.listening = False
                                self.awaiting_command_until = 0.0
                                self.conversation_active_until = 0.0
                                self.last_matched = ""
                                if cycle_pause_until <= 0.0:
                                    self._mic_cycle_pause_until = now + self.mic_cycle_pause_s
                                    cycle_pause_until = self._mic_cycle_pause_until

                                if now >= cycle_pause_until:
                                    self._reset_mic_cycle_locked(now)

                if not mic_enabled:
                    time.sleep(0.1)
                    continue

                if auto_mic_cycle and cycle_pause_until > 0.0 and now < cycle_pause_until:
                    time.sleep(min(0.1, max(0.01, cycle_pause_until - now)))
                    continue

                if self.assistant and getattr(self.assistant, "is_speaking", False):
                    self.listening = False
                    time.sleep(0.1)
                    continue

                self._maybe_refresh_ambient_noise(now)
                self._expire_command_window()
                self._expire_conversation_window()
                self.listening = True
                phrase_time_limit = (
                    self.command_phrase_limit_s
                    if self._allows_follow_up_without_wake_word(time.time())
                    else self.wake_phrase_limit_s
                )
                with self.mic as source:
                    audio = self.recognizer.listen(
                        source,
                        timeout=self.listen_timeout_s,
                        phrase_time_limit=phrase_time_limit,
                    )
                self.listening = False
                transcripts: list[str] = []
                language_used = self.last_language
                recognition_now = time.time()
                audio_for_recognition = self._preprocess_audio(audio)
                try:
                    try:
                        language_used, raw_text, transcripts = self._recognize_audio(
                            audio_for_recognition,
                            recognition_now,
                        )
                    except sr.UnknownValueError:
                        if audio_for_recognition is audio:
                            raise
                        language_used, raw_text, transcripts = self._recognize_audio(
                            audio,
                            recognition_now,
                        )
                    self.last_language = language_used
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

                now = time.time()
                text = _clean_voice_text(raw_text)
                self.last_heard = text
                self.last_heard_time = now
                if self.debug_raw:
                    print(f"[Voice DEBUG] Raw recognised: '{raw_text}'")
                    if transcripts:
                        print(f"[Voice DEBUG] Alternatives: {transcripts}")

                accepted = self.authorize_transcript(text)
                logger.info("Recognised speech: '%s'", text)
                print(f"[Voice] Heard: '{text}'")
                if accepted:
                    logger.info("Accepted command after wake word: '%s'", accepted)
                    print(f"[Voice] Command: '{accepted}'")
                    self._cmd_q.put(accepted)

            except (sr.WaitTimeoutError, sr.UnknownValueError):
                self.listening = False
                self._expire_command_window()
            except sr.RequestError as exc:
                self.listening = False
                self.last_error = f"Speech API error: {exc}"
                logger.error("Google Speech API error: %s", exc)
                self._maybe_speak_feedback("The speech service is not responding right now.")
                time.sleep(2)
            except Exception as exc:
                self.listening = False
                self.last_error = str(exc)
                logger.exception("Unexpected voice recognition failure")
                self._maybe_speak_feedback("Voice recognition ran into a problem.")

    def get_command(self) -> Optional[str]:
        """Return the next wake-word-authorized command, or ``None``."""
        try:
            return self._cmd_q.get_nowait()
        except queue.Empty:
            return None

    def get_status_text(self) -> str:
        """Short device or wake-word summary for logs or HUD."""
        self._expire_command_window()
        self._expire_conversation_window()
        if self.last_error:
            return self.last_error
        if not self.mic_enabled:
            return "Mic: off"
        if self._conversation_is_active():
            remaining = max(0.0, self.conversation_active_until - time.time())
            return f"Talk: follow-up {remaining:.0f}s"
        if time.time() < self.awaiting_command_until:
            return f"Wake: waiting ({self.wake_word})"
        if self.auto_mic_cycle:
            return f"Mic: auto {self.mic_listen_window_s:.0f}s"
        if len(self.languages) > 1:
            return f"Wake: {self.wake_word} ({self.last_language})"
        return f"Wake: say {self.wake_word}"

    def stop(self) -> None:
        if self.stopped:
            return
        self.stopped = True
        try:
            self._thread.join(timeout=1.5)
        except Exception:
            pass
        while True:
            try:
                self._cmd_q.get_nowait()
            except queue.Empty:
                break
        logger.info("VoiceController stopped.")


@dataclass
class PendingVoicePlan:
    utterance: str
    plan: dict[str, Any]
    expires_at: float


@dataclass
class QueuedVoiceCommand:
    text: str
    received_at: float


class VoiceCommandProcessor:
    """Executes safe voice plans and manages spoken confirmations."""

    def __init__(
        self,
        assistant: Optional[AssistantVoice],
        voice: Optional[VoiceController],
        brain: Optional[OllamaBrain],
        cloud_brain: Optional[CloudBrain] = None,
        confirmation_timeout_s: float = DEFAULT_CONFIRM_WINDOW_S,
        control_mode_handler: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.assistant = assistant
        self.voice = voice
        self.brain = brain
        self.cloud_brain = cloud_brain
        self.control_mode_handler = control_mode_handler
        self.confirmation_timeout_s = max(5.0, float(confirmation_timeout_s))
        self.drag_mode = False
        self.pending_plan: Optional[PendingVoicePlan] = None
        self.last_status = "Ready"
        self.command_stale_s = _clamp_float(
            os.environ.get("VOICE_COMMAND_STALE_S", DEFAULT_VOICE_COMMAND_STALE_S),
            default=DEFAULT_VOICE_COMMAND_STALE_S,
            minimum=0.5,
            maximum=30.0,
        )
        self.duplicate_window_s = _clamp_float(
            os.environ.get(
                "VOICE_DUPLICATE_WINDOW_S",
                DEFAULT_VOICE_DUPLICATE_WINDOW_S,
            ),
            default=DEFAULT_VOICE_DUPLICATE_WINDOW_S,
            minimum=0.2,
            maximum=10.0,
        )
        self.max_pending_commands = _clamp_int(
            os.environ.get(
                "VOICE_MAX_PENDING_COMMANDS",
                DEFAULT_VOICE_MAX_PENDING_COMMANDS,
            ),
            default=DEFAULT_VOICE_MAX_PENDING_COMMANDS,
            minimum=1,
            maximum=8,
        )
        self._work_q: queue.Queue[Optional[QueuedVoiceCommand]] = queue.Queue()
        self._busy = False
        self._stopped = False
        self._pending_exit = False
        self._queue_lock = threading.Lock()
        self._last_submitted_text = ""
        self._last_submitted_at = 0.0
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    @property
    def is_busy(self) -> bool:
        return self._busy

    def _speak(self, text: str) -> None:
        if self.assistant and text:
            self.assistant.say(text)

    def _format_app_name(self, app_name: str) -> str:
        cleaned = _clean_voice_text(app_name)
        return APP_DISPLAY_NAMES.get(cleaned, cleaned.title() or "that app")

    def _describe_hotkey(self, keys: list[str]) -> str:
        if not keys:
            return "that shortcut"
        names = {
            "ctrl": "Control",
            "alt": "Alt",
            "shift": "Shift",
            "win": "Windows",
            "esc": "Escape",
            "pgup": "Page Up",
            "pgdn": "Page Down",
        }
        return " + ".join(names.get(key, key.upper() if len(key) == 1 else key.title()) for key in keys)

    def _plan_intro_message(self, plan: dict[str, Any], reply: str) -> str:
        cleaned_reply = (reply or "").strip()
        if cleaned_reply.lower() not in GENERIC_PROGRESS_REPLIES:
            return cleaned_reply

        steps = plan.get("steps", [])
        if not steps:
            return cleaned_reply

        first_step = steps[0]
        action = first_step.get("action")
        if action == "launch_app":
            return f"Opening {self._format_app_name(first_step.get('app', ''))}."
        if action == "open_url":
            url = first_step.get("url", "")
            if "youtube.com/results" in url:
                return "Opening YouTube search."
            if "google.com/search" in url:
                return "Searching Google."
            if "youtube.com" in url:
                return "Opening YouTube."
            if "google.com" in url:
                return "Opening Google."
            return "Opening that page."
        if action == "type_text":
            return "Typing now."
        if action == "press_key":
            return f"Pressing {first_step.get('key', 'that key')}."
        if action == "hotkey":
            return f"Using {self._describe_hotkey(first_step.get('keys', []))}."
        if action == "minimize":
            return "Minimizing the current window."
        if action == "maximize":
            return "Maximizing the current window."
        if action == "restore":
            return "Restoring the current window."
        if action == "scroll":
            direction = first_step.get("direction", "down")
            return f"Scrolling {direction}."
        if action == "drag_toggle":
            return "Updating drag mode."
        if action == "left_click":
            return "Clicking."
        if action == "right_click":
            return "Right clicking."
        if action == "double_click":
            return "Double clicking."
        if action == "control_mode":
            return f"Switching to {first_step.get('mode', 'that')} mode."
        return cleaned_reply

    def _plan_completion_message(self, plan: dict[str, Any]) -> str:
        steps = plan.get("steps", [])
        if not steps:
            return ""
        actions = [step.get("action") for step in steps]
        if "stop" in actions:
            return ""
        if "type_text" in actions:
            return "Typing completed."
        if len(steps) > 1:
            return "Done."
        if actions[0] == "launch_app":
            return f"{self._format_app_name(steps[0].get('app', ''))} is ready."
        return ""

    def _action_failure_message(self, step: dict[str, Any]) -> str:
        action = step.get("action")
        if action == "launch_app":
            return "I couldn't open that app."
        if action == "open_url":
            return "I couldn't open that page."
        if action == "type_text":
            return "I couldn't finish typing that."
        if action == "hotkey":
            return "I couldn't use that shortcut."
        if action == "press_key":
            return "I couldn't press that key."
        if action == "minimize":
            return "I couldn't minimize that window."
        if action == "maximize":
            return "I couldn't maximize that window."
        if action == "restore":
            return "I couldn't restore that window."
        return "That action failed. Please try again."

    def _no_match_reply(self, reply: str) -> str:
        cleaned_reply = (reply or "").strip()
        return cleaned_reply or NO_MATCH_REPLY

    def _question_thinking_reply(self) -> str:
        return QUESTION_THINKING_REPLY

    def _drop_pending_commands_locked(self) -> int:
        dropped = 0
        while self._work_q.qsize() >= self.max_pending_commands:
            try:
                pending = self._work_q.get_nowait()
            except queue.Empty:
                break

            if pending is None:
                self._work_q.put_nowait(None)
                break

            dropped += 1
            self._work_q.task_done()
            logger.warning(
                "Dropped older queued voice command to keep the latest request: '%s'",
                pending.text,
            )

        return dropped

    def _worker(self) -> None:
        while not self._stopped:
            try:
                queued = self._work_q.get(timeout=0.3)
            except queue.Empty:
                continue

            if queued is None:
                self._work_q.task_done()
                break

            self._busy = True
            try:
                age_s = time.time() - queued.received_at
                if age_s > self.command_stale_s:
                    self.last_status = "Dropped stale"
                    logger.warning(
                        "Dropped stale voice command (age=%.2fs): '%s'",
                        age_s,
                        queued.text,
                    )
                    continue

                if self._handle_sync(queued.text):
                    self._pending_exit = True
            finally:
                self._busy = False
                self._work_q.task_done()

    def _clear_pending_if_expired(self) -> None:
        if self.pending_plan and time.time() > self.pending_plan.expires_at:
            self.pending_plan = None
            self.last_status = "Confirmation expired"
            self._speak("Confirmation timed out.")

    def _looks_blocked(self, combined_text: str) -> bool:
        return any(term in combined_text for term in BLOCKED_REQUEST_TERMS)

    def _needs_confirmation(self, utterance: str, plan: dict[str, Any]) -> bool:
        if plan["decision"] == "confirm":
            return True

        utterance_text = utterance.lower()
        if any(term in utterance_text for term in CONFIRM_REQUEST_TERMS):
            return True

        for step in plan["steps"]:
            action = step["action"]
            if action == "type_text" and len(step.get("text", "")) > 180:
                return True
            if action == "press_key" and step.get("key") in CONFIRM_KEYS:
                return True
            if action == "hotkey":
                keys = tuple(step.get("keys", []))
                sorted_keys = tuple(sorted(keys))
                if keys in CONFIRM_HOTKEY_COMBOS or sorted_keys in CONFIRM_HOTKEY_COMBOS:
                    return True
                if keys not in SAFE_HOTKEY_COMBOS and sorted_keys not in SAFE_HOTKEY_COMBOS:
                    return True
        return False

    def _assess_security(self, utterance: str, plan: dict[str, Any]) -> tuple[str, str]:
        if plan["decision"] == "block":
            return "block", plan["reply"] or (
                "I cannot help with risky or security-sensitive actions."
            )

        combined_text = f"{utterance.lower()} {json.dumps(plan, sort_keys=True).lower()}"
        if self._looks_blocked(combined_text):
            return "block", "I cannot help with risky or security-sensitive actions."

        for step in plan["steps"]:
            action = step["action"]
            if action == "open_url" and not _is_safe_http_url(step.get("url", "")):
                return "block", "I can only open normal http or https websites."
            if action == "system_search":
                query = step.get("query", "").lower().strip()
                if not query or len(query) <= 2:
                    return "noop", "Please say a more specific app name."
                if any(term in query for term in BLOCKED_APP_TERMS):
                    return "block", "I cannot open system-level tools."
            if action == "launch_app":
                app_name = step.get("app", "").lower()
                if any(term in app_name for term in BLOCKED_APP_TERMS):
                    return "block", "I cannot open system-level tools."
            if action == "close_app":
                app_name = step.get("app", "").lower()
                if any(term in app_name for term in BLOCKED_APP_TERMS):
                    return "block", "I cannot close system-level tools."
            if action == "hotkey":
                keys = tuple(sorted(step.get("keys", [])))
                if any(key in BLOCKED_HOTKEY_KEYS for key in keys):
                    return "block", "Windows system hotkeys are blocked for safety."
            if action == "type_text":
                text = step.get("text", "").lower()
                if any(term in text for term in ("password", "otp", "token", "secret")):
                    return "block", "I will not type security-sensitive secrets by voice."

        if plan["decision"] == "noop":
            return "noop", plan["reply"] or "I could not map that to a safe task."

        if not plan["steps"]:
            return "noop", plan["reply"] or "I could not map that to a safe task."

        if self._needs_confirmation(utterance, plan):
            return (
                "confirm",
                plan["reply"]
                or "That action needs confirmation. Say Jarvis confirm or Jarvis cancel.",
            )

        return "allow", plan["reply"]

    def _execute_plan(self, plan: dict[str, Any]) -> tuple[bool, bool]:
        should_exit = False

        for step in plan["steps"]:
            action = step["action"]
            logger.info("Executing voice step: %s", step)
            try:
                if action == "left_click":
                    pyautogui.click()

                elif action == "right_click":
                    pyautogui.rightClick()

                elif action == "double_click":
                    pyautogui.doubleClick()

                elif action in {"minimize", "maximize", "restore"}:
                    if not control_window(action):
                        raise RuntimeError(f"Window control action failed: {action}")

                elif action == "scroll":
                    amount = int(step.get("amount", 300))
                    direction = step.get("direction", "down")
                    pyautogui.scroll(amount if direction == "up" else -amount)

                elif action == "drag_toggle":
                    self.drag_mode = not self.drag_mode
                    if self.drag_mode:
                        pyautogui.mouseDown()
                    else:
                        pyautogui.mouseUp()

                elif action == "type_text":
                    pyautogui.typewrite(step.get("text", ""), interval=0.04)

                elif action == "press_key":
                    pyautogui.press(step.get("key", "enter"))

                elif action == "hotkey":
                    keys = step.get("keys", [])
                    if keys:
                        pyautogui.hotkey(*keys)

                elif action == "open_url":
                    webbrowser.open(step.get("url", ""))

                elif action == "system_search":
                    query = step.get("query", "")
                    pyautogui.press("win")
                    time.sleep(0.7)
                    pyautogui.typewrite(query, interval=0.05)
                    time.sleep(1.0)
                    pyautogui.press("enter")

                elif action == "launch_app":
                    app_name = step.get("app", "")
                    target = _resolve_app_target(app_name)
                    if target:
                        os.startfile(target)
                    else:
                        try:
                            subprocess.Popen(f'start "" "{app_name}"', shell=True)
                        except Exception:
                            print(f"Failed to open {app_name}")

                elif action == "close_app":
                    app_name = step.get("app", "")
                    if not _close_app_by_name(app_name):
                        print(f"Failed to close {app_name}")

                elif action == "control_mode":
                    if not self.control_mode_handler:
                        raise RuntimeError("Control mode switching is unavailable")
                    self.control_mode_handler(step.get("mode", "head"))

                elif action == "wait":
                    time.sleep(float(step.get("seconds", 1.0)))

                elif action == "help":
                    self._speak(
                        VoiceController.HELP_TEXT
                        if self.voice
                        else (
                            "Say Jarvis, then speak naturally. I can click, scroll, type, "
                            "press keys, open websites, and open a few safe apps."
                        )
                    )

                elif action == "stop":
                    self._speak("Goodbye. Closing virtual mouse.")
                    time.sleep(1.5)
                    should_exit = True
                    break
            except Exception:
                self.last_status = "Action failed"
                logger.exception("Voice step failed: %s", step)
                self._speak(self._action_failure_message(step))
                return False, False

        completion_message = self._plan_completion_message(plan)
        if completion_message:
            self._speak(completion_message)

        return True, should_exit

    def get_status_text(self) -> str:
        self._clear_pending_if_expired()
        if self._busy:
            return "Task: processing"
        if not self._work_q.empty():
            return "Task: queued"
        if self.pending_plan:
            return "Task: waiting confirm"
        return f"Task: {self.last_status.lower()}"

    def _handle_sync(self, cmd: str) -> bool:
        """Process one wake-word-authorized command. Returns True to exit."""
        text = _clean_voice_text(cmd)
        if not text:
            return False

        self._clear_pending_if_expired()

        if self.pending_plan:
            if _matches_phrase(text, YES_WORDS):
                pending = self.pending_plan
                self.pending_plan = None
                self.last_status = "Confirmed"
                self._speak("Confirmed.")
                _, should_exit = self._execute_plan(pending.plan)
                return should_exit

            if _matches_phrase(text, NO_WORDS):
                self.pending_plan = None
                self.last_status = "Canceled"
                self._speak("Canceled.")
                return False

            self.pending_plan = None
            self.last_status = "Pending cleared"

        if _should_route_to_chat(text):
            self.last_status = "Answering"
            self._speak(self._question_thinking_reply())
            question_text = _strip_chat_prefix(text) or text

            answer = _answer_project_faq(question_text)
            if not answer and self.brain:
                answer = self.brain.answer_question(question_text)
            if not answer and self.cloud_brain:
                answer = self.cloud_brain.answer_question(question_text)

            if answer:
                self.last_status = "Answered"
                logger.info("Answered via project chat: '%s'", question_text)
                self._speak(answer)
                return False

            self.last_status = "Answer failed"
            logger.info("Chat assistant did not return an answer for: '%s'", question_text)
            if self.cloud_brain and self.cloud_brain.last_error:
                self._speak(self.cloud_brain.last_error)
            elif self.brain and self.brain.last_error:
                self._speak(self.brain.last_error)
            else:
                self._speak(
                    "I can answer questions about Blink-Click Virtual Mouse. "
                    "For broader questions, enable Ollama or the cloud brain."
                )
            return False

        plan = plan_task(
            text,
            self.brain,
            cloud_brain=self.cloud_brain,
            drag_mode=self.drag_mode,
        )
        decision, reply = self._assess_security(text, plan)
        logger.debug("Voice decision for '%s': %s | %s", text, decision, plan)

        if decision == "block":
            self.last_status = "Blocked"
            logger.info("Blocked voice request: '%s'", text)
            self._speak(reply)
            return False

        if decision == "noop":
            self.last_status = "No match"
            logger.info("No safe voice action for: '%s'", text)
            self._speak(self._no_match_reply(reply))
            return False

        if decision == "confirm":
            self.pending_plan = PendingVoicePlan(
                utterance=text,
                plan=plan,
                expires_at=time.time() + self.confirmation_timeout_s,
            )
            self.last_status = "Awaiting confirm"
            wake_word = self.voice.wake_word if self.voice else DEFAULT_WAKE_WORD
            self._speak(
                f"{reply} Say {wake_word} confirm to continue or say "
                f"{wake_word} cancel."
            )
            return False

        self.last_status = plan.get("summary", "Executed") or "Executed"
        intro_message = self._plan_intro_message(plan, reply)
        if intro_message:
            self._speak(intro_message)
        _, should_exit = self._execute_plan(plan)
        return should_exit

    def submit(self, cmd: str) -> None:
        text = _clean_voice_text(cmd)
        if not text or self._stopped:
            return

        now = time.time()
        with self._queue_lock:
            if (
                text == self._last_submitted_text
                and (now - self._last_submitted_at) < self.duplicate_window_s
                and (self._busy or not self._work_q.empty())
            ):
                self.last_status = "Skipped duplicate"
                logger.info("Skipped duplicate queued voice command: '%s'", text)
                return

            dropped = self._drop_pending_commands_locked()
            if dropped:
                self.last_status = "Replaced queued"
                logger.info(
                    "Replaced %d older queued voice command(s) with '%s'.",
                    dropped,
                    text,
                )

            self._work_q.put(QueuedVoiceCommand(text=text, received_at=now))
            self._last_submitted_text = text
            self._last_submitted_at = now
            self.last_status = "Queued"

    def poll_should_exit(self) -> bool:
        if not self._pending_exit:
            return False
        self._pending_exit = False
        return True

    def handle(self, cmd: str) -> bool:
        """Backward-compatible synchronous command handling."""
        return self._handle_sync(cmd)

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        self._work_q.put_nowait(None)
        self._thread.join(timeout=1.5)

