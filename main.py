# -*- coding: utf-8 -*-
"""
main.py  –  Blink-Click Virtual Mouse (Accessibility Edition)
═════════════════════════════════════════════════════════════
Entry point that wires together:
    • mouse_controller  – head-tracking cursor, blink clicks, HUD
  • speech_controller – voice recognition, TTS assistant, command processing

Run:
    python main.py

Press ESC to exit at any time.
"""

from __future__ import annotations

import cv2
import logging
import sys
import time
from typing import Optional

import pyautogui
import os

# ── Project modules ──────────────────────────────────────────────
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
from speech_controller import (
    OllamaBrain,
    SR_AVAILABLE,
    TTS_AVAILABLE,
    AssistantVoice,
    VoiceController,
    VoiceCommandProcessor,
)

# ── Logging setup ────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s  %(name)-22s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")
WINDOW_TITLE = "Blink-Click Virtual Mouse  |  Accessibility Edition"
EXIT_CONFIRM_WINDOW_S = 1.5


# ================================================================
#  STARTUP BANNER
# ================================================================
def _print_banner(
    assistant: Optional[AssistantVoice],
    voice: Optional[VoiceController],
    brain: Optional[OllamaBrain],
    cfg: MouseConfig,
) -> None:
    print("\n" + "═" * 62)
    print("   BLINK-CLICK VIRTUAL MOUSE  –  Accessibility Edition")
    print("═" * 62)
    print(f"  TTS        : {'Active  (will speak back)' if assistant else 'Disabled'}")
    if voice:
        mode = "rules + Ollama" if brain else "rules only"
        print(f"  Voice In   : Active  (say '{voice.wake_word}' to wake, {mode})")
    else:
        print("  Voice In   : Disabled")
    print(f"  Brain      : {'Ollama ' + brain.model if brain else 'Disabled'}")
    if voice:
        print(f"  Wake Word  : {voice.wake_word}")
    print( "  Cursor     : Head tracking (nose position)")
    print(f"  Blink Click: EAR threshold {cfg.blink_threshold}")
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

    # ── Rest reminder ────────────────────────────────────────────
    session_start = time.time()

    # ── FPS counter ──────────────────────────────────────────────
    fps_time = time.time()
    fps_count = 0
    fps_display = 0
    exit_armed_until = 0.0

    # ── Voice controller ─────────────────────────────────────────
    voice: Optional[VoiceController] = None
    voice_processor: Optional[VoiceCommandProcessor] = None
    brain: Optional[OllamaBrain] = None

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

    if SR_AVAILABLE:
        try:
            mic_index = None
            mic_name = os.environ.get("MIC_NAME")
            if os.environ.get("MIC_INDEX"):
                try:
                    mic_index = int(os.environ.get("MIC_INDEX"))
                    logger.info("Using microphone index %d from MIC_INDEX", mic_index)
                except Exception:
                    logger.warning("Invalid MIC_INDEX value; using default microphone")
            if mic_name:
                logger.info("Using microphone name match from MIC_NAME: %s", mic_name)
            voice = VoiceController(
                assistant=assistant,
                energy_threshold=_env_int("VOICE_ENERGY_THRESHOLD", 350),
                pause_threshold=_env_float("VOICE_PAUSE_THRESHOLD", 0.6),
                phrase_threshold=_env_float("VOICE_PHRASE_THRESHOLD", 0.3),
                calibration_duration=_env_float("VOICE_CALIBRATION_S", 2.5),
                microphone_index=mic_index,
                microphone_name=mic_name,
                debug_raw_recognition=_env_bool("VOICE_DEBUG", False),
                wake_word=os.environ.get("WAKE_WORD", "ashu"),
                command_window_s=_env_float("VOICE_COMMAND_WINDOW_S", 8.0),
                acknowledge_wake=_env_bool("VOICE_ACKNOWLEDGE_WAKE", True),
            )
            voice_processor = VoiceCommandProcessor(
                assistant=assistant,
                voice=voice,
                brain=brain,
            )
            if brain:
                logger.info("Voice controller active in hybrid mode with Ollama brain.")
            else:
                logger.info("Voice controller active in rule-based mode (Ollama disabled).")
        except Exception as exc:
            logger.error("Cannot start voice controller: %s", exc)
    else:
        logger.warning("Voice input disabled because SpeechRecognition is unavailable.")

    # ── Banner ───────────────────────────────────────────────────
    _print_banner(assistant, voice, brain, cfg)

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
            lm = face_mesh.process(processing_frame)

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

                    blink_result = blink_detector.update(avg_ear, now)
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
                draw_no_face_warning(frame, w, h)

            # ── VOICE COMMANDS ───────────────────────────────────
            if voice and voice_processor:
                cmd = voice.get_command()
                if cmd:
                    should_exit = voice_processor.handle(cmd)
                    drag_mode = voice_processor.drag_mode
                    if should_exit:
                        break
            # ── VOICE STATUS HUD ─────────────────────────────────
            if voice:
                v_listening = voice.listening
                v_last = voice.last_matched
                v_age = now - voice.last_heard_time if voice.last_heard_time else 999
                draw_voice_status(frame, w, h, v_listening, v_last, v_age)
            # ── CLICK FEEDBACK OVERLAY ───────────────────────────
            if now < blink_feedback_until and blink_feedback_text:
                draw_click_feedback(frame, blink_feedback_text, w, h)

            # ── STATUS PANEL ─────────────────────────────────────
            status_lines = [
                f"Face  : {face_mesh.mode.upper()}",
                f"Voice : {'ON  (wake)' if voice else 'OFF'}",
                f"Brain : {'OLLAMA' if brain else 'OFF'}",
                f"TTS   : {'ON' if assistant else 'OFF'}",
                f"Drag  : {'ON' if drag_mode else 'OFF'}",
                f"FPS   : {fps_display}",
            ]
            if voice:
                status_lines.insert(1, voice.get_status_text()[:28])
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

            # ── SHOW ─────────────────────────────────────────────
            cv2.imshow(WINDOW_TITLE, frame)
            key = cv2.waitKey(1) & 0xFF
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
    if voice_processor and voice_processor.drag_mode:
        pyautogui.mouseUp()
    if voice:
        voice.stop()
    if assistant:
        assistant.stop()
    cam.release()
    cv2.destroyAllWindows()
    print("\nProgram closed. Goodbye!")


if __name__ == "__main__":
    main()
