# -*- coding: utf-8 -*-
"""
main.py  –  Blink-Click Virtual Mouse (Accessibility Edition)
═════════════════════════════════════════════════════════════
Entry point that wires together:
    • mouse_controller  – head-tracking cursor, blink clicks, HUD
    • wake_vosk / stt_google – wake word and speech-to-text
    • speech_controller – command planning, safety checks, and execution

Run:
    python main.py

Press ESC to exit at any time.
"""

from __future__ import annotations

import cv2
import logging
import os
import sys
import threading
import time
from typing import Optional

import pyautogui
import sounddevice as sd

# ── Project modules ──────────────────────────────────────────────
from hand_controller import HandGestureController
from mouse_controller import (
    BlinkDetector,
    CameraCapture,
    FaceMeshProcessor,
    HeadTracker,
    MouseConfig,
    compute_eye_aspect_ratio,
    draw_click_feedback,
    
    draw_ear_bar,
    draw_no_face_warning,
    draw_nose_marker,
    draw_rest_reminder,
    draw_status_panel,
    draw_voice_status,
)
from normalizer import normalize
from speech_controller import (
    CloudBrain,
    OllamaBrain,
    TTS_AVAILABLE,
    AssistantVoice,
    VoiceCommandProcessor,
)
from stt_google import GoogleSTT
from wake_vosk import VoskWake

# ── Logging setup ────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s  %(name)-22s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")
WINDOW_TITLE = "Blink-Click Virtual Mouse  |  Accessibility Edition"
EXIT_CONFIRM_WINDOW_S = 1.5
STOP_BUTTON_X = 14
STOP_BUTTON_Y = 14
STOP_BUTTON_W = 110
STOP_BUTTON_H = 36
MIC_BUTTON_X = STOP_BUTTON_X + STOP_BUTTON_W + 12
MIC_BUTTON_Y = 14
MIC_BUTTON_W = 128
MIC_BUTTON_H = 36
DEFAULT_MIC_PREFERENCES = (
    "headset (wings phantom)",
    "microphone array",
)


# ================================================================
#  STARTUP BANNER
# ================================================================
def _print_banner(
    assistant: Optional[AssistantVoice],
    voice_ready: bool,
    wake_word: str,
    brain: Optional[OllamaBrain],
    cloud_brain: Optional[CloudBrain],
    cfg: MouseConfig,
    control_mode: str,
    hand_available: bool,
) -> None:
    print("\n" + "═" * 62)
    print("   BLINK-CLICK VIRTUAL MOUSE  –  Accessibility Edition")
    print("═" * 62)
    print(f"  TTS        : {'Active  (will speak back)' if assistant else 'Disabled'}")
    if voice_ready:
        print(
            "  Voice In   : Active  (say '"
            f"{wake_word}' to wake, vosk + google)"
        )
    else:
        print("  Voice In   : Disabled")
    if brain and cloud_brain:
        brain_label = f"Ollama {brain.model} + Cloud {cloud_brain.model}"
    elif brain:
        brain_label = f"Ollama {brain.model}"
    elif cloud_brain:
        brain_label = f"Cloud {cloud_brain.model}"
    else:
        brain_label = "Disabled"
    print(f"  Brain      : {brain_label}")
    if voice_ready:
        print(f"  Wake Word  : {wake_word}")
        print("  Mic Ctrl   : Click MIC button in window or press M")
    else:
        print("  Mic Ctrl   : Unavailable (voice input disabled)")
    hotkey_hint = " (H=hand, F=head)" if hand_available else ""
    print(f"  Control    : {control_mode.upper()}{hotkey_hint}")
    print("  Head Mode  : Nose cursor + blink click")
    if hand_available:
        print(
            "  Hand Mode  : Index move, pinch left click, 2-finger scroll, "
            "3-finger right click, open palm pause"
        )
    print(f"  Blink Click: EAR threshold {cfg.blink_threshold} (head mode)")
    print(f"  Click Cool : {cfg.click_cooldown_s:.2f}s")
    print( "  Press ESC to exit")
    print("═" * 62 + "\n")


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


def _apply_mouse_runtime_overrides(cfg: MouseConfig) -> None:
    cfg.blink_threshold = _env_float("BLINK_THRESHOLD", cfg.blink_threshold)
    cfg.intentional_blink_duration = _env_float(
        "INTENTIONAL_BLINK_DURATION",
        cfg.intentional_blink_duration,
    )
    cfg.double_blink_gap = _env_float("DOUBLE_BLINK_GAP", cfg.double_blink_gap)
    cfg.blink_release_margin = _env_float(
        "BLINK_RELEASE_MARGIN",
        cfg.blink_release_margin,
    )
    cfg.click_cooldown_s = _env_float("CLICK_COOLDOWN_S", cfg.click_cooldown_s)
    cfg.hand_pinch_threshold_px = _env_int(
        "HAND_PINCH_THRESHOLD_PX",
        cfg.hand_pinch_threshold_px,
    )
    cfg.hand_right_click_threshold_px = _env_int(
        "HAND_RIGHT_CLICK_THRESHOLD_PX",
        cfg.hand_right_click_threshold_px,
    )
    cfg.hand_gesture_confirm_frames = _env_int(
        "HAND_GESTURE_CONFIRM_FRAMES",
        cfg.hand_gesture_confirm_frames,
    )
    cfg.hand_click_cooldown_s = _env_float(
        "HAND_CLICK_COOLDOWN_S",
        cfg.hand_click_cooldown_s,
    )
    cfg.hand_scroll_unit = _env_int("HAND_SCROLL_UNIT", cfg.hand_scroll_unit)


def _resolve_control_mode(hand_available: bool) -> str:
    requested = os.environ.get("CONTROL_MODE", "head").strip().lower()
    if requested in {"face", "eye"}:
        requested = "head"
    if requested not in {"head", "hand"}:
        logger.warning("Invalid CONTROL_MODE '%s'; defaulting to head.", requested)
        requested = "head"
    if requested == "hand" and not hand_available:
        logger.warning("CONTROL_MODE=hand requested but hand tracking is unavailable.")
        return "head"
    return requested


def _draw_stop_button(frame, hover: bool) -> tuple[int, int, int, int]:
    """Draw a clickable stop button in the camera window and return its rect."""
    x1, y1 = STOP_BUTTON_X, STOP_BUTTON_Y
    x2, y2 = x1 + STOP_BUTTON_W, y1 + STOP_BUTTON_H

    if hover:
        bg_color = (40, 40, 230)
        border_color = (70, 90, 255)
    else:
        bg_color = (20, 20, 180)
        border_color = (50, 70, 240)

    cv2.rectangle(frame, (x1, y1), (x2, y2), bg_color, -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, 2)
    cv2.putText(
        frame,
        "STOP",
        (x1 + 24, y1 + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return x1, y1, x2, y2


def _draw_mic_button(
    frame,
    *,
    hover: bool,
    mic_available: bool,
    mic_enabled: bool,
) -> tuple[int, int, int, int]:
    """Draw a clickable mic toggle button and return its rect."""
    x1, y1 = MIC_BUTTON_X, MIC_BUTTON_Y
    x2, y2 = x1 + MIC_BUTTON_W, y1 + MIC_BUTTON_H

    if not mic_available:
        bg_color = (80, 80, 80)
        border_color = (115, 115, 115)
        label = "MIC N/A"
    elif mic_enabled:
        bg_color = (20, 150, 35) if not hover else (30, 180, 45)
        border_color = (60, 220, 80)
        label = "MIC ON"
    else:
        bg_color = (25, 25, 160) if not hover else (35, 35, 210)
        border_color = (80, 80, 255)
        label = "MIC OFF"

    cv2.rectangle(frame, (x1, y1), (x2, y2), bg_color, -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, 2)
    cv2.putText(
        frame,
        label,
        (x1 + 16, y1 + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return x1, y1, x2, y2


# ================================================================
#  MAIN LOOP
# ================================================================
def main() -> None:
    # ── Configuration ────────────────────────────────────────────
    cfg = MouseConfig()
    _apply_mouse_runtime_overrides(cfg)
    if os.environ.get("CAMERA_INDEX"):
        try:
            cfg.camera_index = int(os.environ["CAMERA_INDEX"])
            logger.info("Using camera index %d from CAMERA_INDEX", cfg.camera_index)
        except ValueError:
            logger.warning("Invalid CAMERA_INDEX value; using default camera index")
    if os.environ.get("CAMERA_BACKEND"):
        cfg.camera_backend = os.environ["CAMERA_BACKEND"].strip().lower()
        logger.info("Using camera backend '%s' from CAMERA_BACKEND", cfg.camera_backend)

    pyautogui.FAILSAFE = False
    pyautogui.PAUSE = 0

    # ── TTS Assistant ────────────────────────────────────────────
    assistant: Optional[AssistantVoice] = None
    if TTS_AVAILABLE:
        assistant = AssistantVoice()
        assistant.greet()
    else:
        logger.warning("pyttsx3 not found – TTS disabled. pip install pyttsx3")

    # ── Camera ───────────────────────────────────────────────────
    try:
        cam = CameraCapture(cfg)
    except Exception as exc:
        logger.error("Cannot start camera: %s", exc)
        print(f"[Camera] Error: {exc}")
        print("[Camera] Try setting CAMERA_INDEX=1 or CAMERA_BACKEND=msmf before running.")
        if assistant:
            assistant.say("Camera startup failed.")
            assistant.stop()
        sys.exit(1)

    # ── MediaPipe Face Mesh ──────────────────────────────────────
    face_mesh = FaceMeshProcessor(cfg)

    hand_controller: Optional[HandGestureController] = None
    try:
        hand_controller = HandGestureController(cfg)
        logger.info("Hand gesture controller initialised.")
    except Exception as exc:
        logger.warning("Hand gesture controller unavailable: %s", exc)

    # ── Head Tracker ─────────────────────────────────────────────
    tracker = HeadTracker(cfg)

    # Dwell click removed — feature disabled

    # ── Blink Detector ───────────────────────────────────────────
    blink_detector = BlinkDetector(cfg)
    logger.info(
        "Blink tuning active (threshold=%.3f intentional=%.2fs double_gap=%.2fs cooldown=%.2fs)",
        cfg.blink_threshold,
        cfg.intentional_blink_duration,
        cfg.double_blink_gap,
        cfg.click_cooldown_s,
    )

    # ── State variables ──────────────────────────────────────────
    drag_mode = False
    blink_feedback_until: float = 0.0
    blink_feedback_text: str = ""
    control_mode = _resolve_control_mode(hand_available=hand_controller is not None)
    if control_mode == "hand" and hand_controller:
        hand_controller.sync_to_cursor()
    else:
        tracker.sync_to_cursor()

    # ── Rest reminder ────────────────────────────────────────────
    session_start = time.time()

    # ── FPS counter ──────────────────────────────────────────────
    fps_time = time.time()
    fps_count = 0
    fps_display = 0
    exit_armed_until = 0.0

    # ── Voice controller ─────────────────────────────────────────
    voice_processor: Optional[VoiceCommandProcessor] = None
    brain: Optional[OllamaBrain] = None
    cloud_brain: Optional[CloudBrain] = None

    if os.environ.get("OLLAMA_BRAIN", "1").lower() not in ("0", "false", "off"):
        timeout_s = 25.0
        try:
            timeout_s = float(os.environ.get("OLLAMA_TIMEOUT", "25"))
        except ValueError:
            logger.warning("Invalid OLLAMA_TIMEOUT value; using 25 seconds")
        brain = OllamaBrain(
            model=os.environ.get("OLLAMA_MODEL", "phi3"),
            host=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"),
            timeout_s=timeout_s,
        )
        logger.info("Ollama brain enabled (model=%s)", brain.model)

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
        cloud_brain = CloudBrain(
            model=os.environ.get("CLOUD_BRAIN_MODEL", "gpt-4o-mini"),
            api_key=cloud_api_key,
            base_url=os.environ.get(
                "CLOUD_BRAIN_BASE_URL",
                "https://api.openai.com/v1",
            ),
            timeout_s=cloud_timeout_s,
        )
        logger.info("Cloud brain enabled (model=%s)", cloud_brain.model)
    elif cloud_requested and cloud_enabled and not cloud_api_key:
        logger.warning(
            "CLOUD_BRAIN requested but no CLOUD_BRAIN_API_KEY or OPENAI_API_KEY "
            "was provided."
        )

    wake_word = os.environ.get("WAKE_WORD", "jarvis").strip() or "jarvis"
    vosk_model_path = (
        os.environ.get("VOSK_MODEL_PATH", "vosk-model-small-en-in-0.4").strip()
        or "vosk-model-small-en-in-0.4"
    )

    mic_index, mic_label = _pick_microphone_index()
    if mic_index is not None:
        logger.info("Using microphone index %d (%s)", mic_index, mic_label or "unknown")
    else:
        logger.warning("No input microphone detected; voice input may be unavailable.")

    wake_listener: Optional[VoskWake] = None
    stt: Optional[GoogleSTT] = None
    wake_thread: Optional[threading.Thread] = None
    wake_stop_event = threading.Event()
    wake_enabled_event = threading.Event()
    wake_status_lock = threading.Lock()
    wake_status = {
        "listening": False,
        "last_text": "",
        "last_time": 0.0,
        "last_error": "",
    }
    voice_ready = False

    def _set_wake_status(
        *,
        listening: Optional[bool] = None,
        last_text: Optional[str] = None,
        last_time: Optional[float] = None,
        last_error: Optional[str] = None,
    ) -> None:
        with wake_status_lock:
            if listening is not None:
                wake_status["listening"] = listening
            if last_text is not None:
                wake_status["last_text"] = last_text
            if last_time is not None:
                wake_status["last_time"] = last_time
            if last_error is not None:
                wake_status["last_error"] = last_error

    def _wake_status_text() -> str:
        if not voice_ready:
            return "Voice: off"
        if not wake_enabled_event.is_set():
            return "Mic: off"
        with wake_status_lock:
            if wake_status.get("listening"):
                return f"Wake: listening ({wake_word})"
            if wake_status.get("last_error"):
                return "Wake: error"
        return f"Wake: say {wake_word}"

    def _wake_command_loop() -> None:
        if not wake_listener or not stt or not voice_processor:
            return

        while not wake_stop_event.is_set():
            if not wake_enabled_event.is_set():
                _set_wake_status(listening=False)
                time.sleep(0.1)
                continue

            _set_wake_status(listening=True)
            try:
                detected = wake_listener.listen_wake(stop_event=wake_stop_event)
            except Exception as exc:
                _set_wake_status(listening=False, last_error=str(exc))
                logger.error("Wake listener failed: %s", exc)
                time.sleep(1.0)
                continue

            _set_wake_status(listening=False)
            if not detected or wake_stop_event.is_set():
                continue

            print("Wake detected")

            if assistant and _env_bool("VOICE_ACKNOWLEDGE_WAKE", True):
                assistant.say("Yes?")

            command = stt.listen_command()
            if not command:
                continue

            command = normalize(command)
            _set_wake_status(
                last_text=command,
                last_time=time.time(),
                last_error="",
            )
            voice_processor.submit(command)

    if _env_bool("VOICE_ENABLED", True):
        try:
            wake_listener = VoskWake(
                vosk_model_path,
                wake_word=wake_word,
                device=mic_index,
            )
            stt = GoogleSTT(
                language=os.environ.get("VOICE_LANGUAGE", "en-IN").strip() or "en-IN",
                mic_index=mic_index,
                energy_threshold=_env_int("VOICE_ENERGY_THRESHOLD", 300),
                dynamic_energy_threshold=_env_bool("VOICE_DYNAMIC_THRESHOLD", True),
                listen_timeout=_env_float("VOICE_LISTEN_TIMEOUT_S", 2.0),
                phrase_time_limit=_env_float("VOICE_PHRASE_LIMIT_S", 5.0),
            )
            voice_processor = VoiceCommandProcessor(
                assistant=assistant,
                voice=None,
                brain=brain,
                cloud_brain=cloud_brain,
            )
            if _env_bool("VOICE_MIC_ENABLED", True):
                wake_enabled_event.set()
            voice_ready = True
            wake_thread = threading.Thread(target=_wake_command_loop, daemon=True)
            wake_thread.start()
            logger.info("Vosk wake listener active (model=%s)", vosk_model_path)
        except Exception as exc:
            logger.error("Cannot start Vosk wake or Google STT: %s", exc)
    else:
        logger.info("Voice input disabled via VOICE_ENABLED.")

    # ── Banner ───────────────────────────────────────────────────
    _print_banner(
        assistant,
        voice_ready,
        wake_word,
        brain,
        cloud_brain,
        cfg,
        control_mode,
        hand_available=hand_controller is not None,
    )
    if voice_ready:
        print(f"System ready... Say '{wake_word}'")

    stop_button_state = {
        "rect": (0, 0, 0, 0),
        "hover": False,
        "clicked": False,
    }
    mic_button_state = {
        "rect": (0, 0, 0, 0),
        "hover": False,
        "clicked": False,
    }

    def _toggle_microphone(source: str) -> None:
        if voice_ready:
            next_state = not wake_enabled_event.is_set()
            if next_state:
                wake_enabled_event.set()
            else:
                wake_enabled_event.clear()
            state_label = "ON" if next_state else "OFF"
            logger.info("Microphone toggled %s by %s.", state_label, source)
            print(f"[Voice] Microphone toggled {state_label} ({source})")
            if assistant:
                assistant.say("Microphone on" if next_state else "Microphone off")
        else:
            logger.info(
                "Mic toggle requested by %s but voice input is unavailable.",
                source,
            )
            print(
                "[Voice] Microphone control unavailable because voice input "
                "is disabled."
            )

    def _set_control_mode(next_mode: str, source: str) -> None:
        nonlocal control_mode

        next_mode = next_mode.strip().lower()
        if next_mode == "hand" and not hand_controller:
            logger.info(
                "Hand mode requested by %s but hand tracking is unavailable.",
                source,
            )
            print("[Control] Hand mode is unavailable in this environment.")
            if assistant:
                assistant.say("Hand mode is unavailable")
            return
        if next_mode not in {"head", "hand"} or next_mode == control_mode:
            return

        control_mode = next_mode
        if control_mode == "hand" and hand_controller:
            hand_controller.sync_to_cursor()
        else:
            tracker.sync_to_cursor()

        logger.info("Control mode switched to %s by %s.", control_mode, source)
        print(f"[Control] Switched to {control_mode.upper()} mode ({source})")
        if assistant:
            assistant.say(f"{control_mode} mode")

    def _on_window_mouse(event, x, y, flags, param) -> None:
        _ = (flags, param)
        s_x1, s_y1, s_x2, s_y2 = stop_button_state["rect"]
        m_x1, m_y1, m_x2, m_y2 = mic_button_state["rect"]
        inside_stop = s_x1 <= x <= s_x2 and s_y1 <= y <= s_y2
        inside_mic = m_x1 <= x <= m_x2 and m_y1 <= y <= m_y2

        if event == cv2.EVENT_MOUSEMOVE:
            stop_button_state["hover"] = inside_stop
            mic_button_state["hover"] = inside_mic
        elif event == cv2.EVENT_LBUTTONDOWN:
            if inside_stop:
                stop_button_state["clicked"] = True
            elif inside_mic:
                mic_button_state["clicked"] = True

    cv2.namedWindow(WINDOW_TITLE)
    cv2.setMouseCallback(WINDOW_TITLE, _on_window_mouse)

    # ── Main loop ────────────────────────────────────────────────
    try:
        while True:
            ret, frame = cam.read()
            if not ret or frame is None:
                continue

            frame = cv2.flip(frame, 1)
            processing_frame = frame.copy()
            h, w, _ = frame.shape

            # Enhance only the display frame. Landmark tracking stays on raw pixels.
            frame = cv2.convertScaleAbs(
                frame, alpha=cfg.frame_alpha, beta=cfg.frame_beta
            )
            frame = cv2.GaussianBlur(frame, cfg.blur_kernel, 0)

            now = time.time()

            # ── Face mesh processing ─────────────────────────────
            if control_mode == "hand" and hand_controller:
                hand_result = hand_controller.process(processing_frame, now)
                hand_controller.draw(frame, hand_result)

                if hand_result.hand_detected:
                    if (
                        not hand_result.pause
                        and hand_result.cursor_x is not None
                        and hand_result.cursor_y is not None
                    ):
                        pyautogui.moveTo(
                            hand_result.cursor_x,
                            hand_result.cursor_y,
                            _pause=False,
                        )

                    if hand_result.click == "left":
                        pyautogui.click()
                        blink_feedback_text = "LEFT CLICK"
                        blink_feedback_until = now + cfg.click_feedback_duration
                        if assistant:
                            assistant.say("Click")
                    elif hand_result.click == "right":
                        pyautogui.rightClick()
                        blink_feedback_text = "RIGHT CLICK"
                        blink_feedback_until = now + cfg.click_feedback_duration
                        if assistant:
                            assistant.say("Right click")

                    if hand_result.scroll_amount:
                        pyautogui.scroll(hand_result.scroll_amount)
                else:
                    hand_controller.draw_no_hand_warning(frame, w, h)

            lm = face_mesh.process(processing_frame) if control_mode == "head" else None

            if lm is not None:
                # ── HEAD CURSOR ──────────────────────────────────
                nose = lm[HeadTracker.NOSE_TIP]
                scr_x, scr_y = tracker.update(nose.x, nose.y, now)
                pyautogui.moveTo(scr_x, scr_y, _pause=False)

                # Nose marker
                npx, npy = int(nose.x * w), int(nose.y * h)
                draw_nose_marker(frame, npx, npy)

                # Forehead dot
                fh = lm[HeadTracker.FOREHEAD]
                cv2.circle(
                    frame,
                    (int(fh.x * w), int(fh.y * h)),
                    4, (255, 200, 0), -1,
                )

                # Dwell click feature removed

                # ── BLINK DETECTION ──────────────────────────────
                if face_mesh.supports_blink:
                    left_ear = compute_eye_aspect_ratio(
                        lm, w, h,
                        HeadTracker.L_TOP, HeadTracker.L_BOTTOM,
                        HeadTracker.L_LEFT, HeadTracker.L_RIGHT,
                    )
                    right_ear = compute_eye_aspect_ratio(
                        lm, w, h,
                        HeadTracker.R_TOP, HeadTracker.R_BOTTOM,
                        HeadTracker.R_LEFT, HeadTracker.R_RIGHT,
                    )
                    avg_ear = (left_ear + right_ear) / 2

                    blink_result = blink_detector.update(left_ear, right_ear, now)
                    if blink_result == "left":
                        pyautogui.click()
                        blink_feedback_text = "LEFT CLICK"
                        blink_feedback_until = now + cfg.click_feedback_duration
                        if assistant:
                            assistant.say("Click")
                    elif blink_result == "right":
                        pyautogui.rightClick()
                        blink_feedback_text = "RIGHT CLICK"
                        blink_feedback_until = now + cfg.click_feedback_duration
                        if assistant:
                            assistant.say("Right click")

                    # EAR bar
                    draw_ear_bar(frame, avg_ear, cfg.blink_threshold)

            else:
                if control_mode == "head":
                    draw_no_face_warning(frame, w, h)

            # ── VOICE COMMANDS ───────────────────────────────────
            if voice_processor:
                drag_mode = voice_processor.drag_mode
                if voice_processor.poll_should_exit():
                    break
            # ── VOICE STATUS HUD ─────────────────────────────────
            if voice_ready:
                with wake_status_lock:
                    v_listening = bool(wake_status.get("listening"))
                    v_last = str(wake_status.get("last_text", ""))
                    last_time = float(wake_status.get("last_time", 0.0))
                v_age = now - last_time if last_time else 999
                draw_voice_status(frame, w, h, v_listening, v_last, v_age)
            # ── CLICK FEEDBACK OVERLAY ───────────────────────────
            if now < blink_feedback_until and blink_feedback_text:
                draw_click_feedback(frame, blink_feedback_text, w, h)

            # ── STATUS PANEL ─────────────────────────────────────
            status_lines = [
                f"Ctrl  : {control_mode.upper()}",
                (
                    f"Face  : {face_mesh.mode.upper()}"
                    if control_mode == "head"
                    else "Hand  : MEDIAPIPE"
                ),
                f"Voice : {'ON  (vosk)' if voice_ready else 'OFF'}",
                (
                    "Brain : OLLAMA+CLOUD"
                    if brain and cloud_brain
                    else "Brain : OLLAMA"
                    if brain
                    else "Brain : CLOUD"
                    if cloud_brain
                    else "Brain : OFF"
                ),
                f"TTS   : {'ON' if assistant else 'OFF'}",
                f"Drag  : {'ON' if drag_mode else 'OFF'}",
                f"FPS   : {fps_display}",
            ]
            if voice_ready:
                status_lines.insert(1, _wake_status_text()[:28])
            if voice_processor:
                status_lines.insert(2, voice_processor.get_status_text()[:28])
            draw_status_panel(frame, w - 220, 4, status_lines)

            # ── REST REMINDER ────────────────────────────────────
            if now - session_start > cfg.rest_interval:
                draw_rest_reminder(frame, w, h)
                session_start = now

            # ── FPS ──────────────────────────────────────────────
            fps_count += 1
            if now - fps_time >= 1.0:
                fps_display = fps_count
                fps_count = 0
                fps_time = now

            # ── STOP BUTTON ──────────────────────────────────────
            stop_button_state["rect"] = _draw_stop_button(
                frame,
                hover=bool(stop_button_state["hover"]),
            )
            mic_button_state["rect"] = _draw_mic_button(
                frame,
                hover=bool(mic_button_state["hover"]),
                mic_available=voice_ready,
                mic_enabled=bool(voice_ready and wake_enabled_event.is_set()),
            )

            # ── SHOW ─────────────────────────────────────────────
            cv2.imshow(WINDOW_TITLE, frame)
            key = cv2.waitKey(1) & 0xFF

            if stop_button_state["clicked"]:
                logger.info("Stop button clicked. Exiting application.")
                if assistant:
                    assistant.say("Stopping")
                break

            if mic_button_state["clicked"]:
                mic_button_state["clicked"] = False
                _toggle_microphone("button: MIC")
                continue

            if key in (ord("m"), ord("M")):
                _toggle_microphone("hotkey: M")
                continue

            if key in (ord("h"), ord("H")):
                _set_control_mode("hand", "hotkey: H")
                continue

            if key in (ord("f"), ord("F")):
                _set_control_mode("head", "hotkey: F")
                continue

            if key == 27:
                if now <= exit_armed_until:
                    logger.info("Escape pressed twice. Exiting application.")
                    if assistant:
                        assistant.say("Goodbye")
                    break

                exit_armed_until = now + EXIT_CONFIRM_WINDOW_S
                logger.warning(
                    "Escape detected. Press ESC again within %.1f seconds to exit.",
                    EXIT_CONFIRM_WINDOW_S,
                )
                print(
                    f"[System] Escape detected. Press ESC again within "
                    f"{EXIT_CONFIRM_WINDOW_S:.1f} seconds to exit."
                )

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")

    # ── CLEANUP ──────────────────────────────────────────────────
    logger.info("Shutting down …")
    wake_stop_event.set()
    if wake_thread:
        wake_thread.join(timeout=1.5)
    if voice_processor and voice_processor.drag_mode:
        pyautogui.mouseUp()
    if voice_processor:
        voice_processor.stop()
    if assistant:
        assistant.stop()
    if hand_controller:
        hand_controller.close()
    cam.release()
    cv2.destroyAllWindows()
    print("\nProgram closed. Goodbye!")


if __name__ == "__main__":
    main()
