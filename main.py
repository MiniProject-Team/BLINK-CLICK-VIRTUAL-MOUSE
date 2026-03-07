# -*- coding: utf-8 -*-
"""
main.py  –  Blink-Click Virtual Mouse (Accessibility Edition)
═════════════════════════════════════════════════════════════
Entry point that wires together:
  • mouse_controller  – head-tracking cursor, blink/dwell clicks, HUD
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

# ── Project modules ──────────────────────────────────────────────
from mouse_controller import (
    BlinkDetector,
    CameraCapture,
    DwellClicker,
    FaceMeshProcessor,
    HeadTracker,
    MouseConfig,
    compute_eye_aspect_ratio,
    draw_click_feedback,
    draw_dwell_arc,
    draw_ear_bar,
    draw_no_face_warning,
    draw_nose_marker,
    draw_rest_reminder,
    draw_status_panel,
    draw_voice_status,
)
from speech_controller import (
    SR_AVAILABLE,
    TTS_AVAILABLE,
    AssistantVoice,
    VoiceController,
    process_voice_command,
)

# ── Logging setup ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-22s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


# ================================================================
#  STARTUP BANNER
# ================================================================
def _print_banner(
    assistant: Optional[AssistantVoice],
    voice: Optional[VoiceController],
    cfg: MouseConfig,
) -> None:
    print("\n" + "═" * 62)
    print("   BLINK-CLICK VIRTUAL MOUSE  –  Accessibility Edition")
    print("═" * 62)
    print(f"  TTS        : {'Active  (will speak back)' if assistant else 'Disabled'}")
    print(f"  Voice In   : {'Active  (say help)' if voice else 'Disabled'}")
    print( "  Cursor     : Head tracking (nose position)")
    print(f"  Blink Click: EAR threshold {cfg.blink_threshold}")
    print(f"  Dwell Click: hold still {cfg.dwell_time:.1f}s")
    print( "  Press ESC to exit")
    print("═" * 62 + "\n")


# ================================================================
#  MAIN LOOP
# ================================================================
def main() -> None:
    # ── Configuration ────────────────────────────────────────────
    cfg = MouseConfig()

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
    cam = CameraCapture(cfg)

    # ── MediaPipe Face Mesh ──────────────────────────────────────
    face_mesh = FaceMeshProcessor(cfg)

    # ── Head Tracker ─────────────────────────────────────────────
    tracker = HeadTracker(cfg)

    # ── Dwell Clicker ────────────────────────────────────────────
    dwell = DwellClicker(dwell_time=cfg.dwell_time, radius=cfg.dwell_radius)
    dwell_enabled = True

    # ── Blink Detector ───────────────────────────────────────────
    blink_detector = BlinkDetector(cfg)

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

    # ── Voice controller ─────────────────────────────────────────
    voice: Optional[VoiceController] = None
    if SR_AVAILABLE:
        try:
            voice = VoiceController(assistant=assistant)
            logger.info("Voice controller active – say 'help' for commands.")
        except Exception as exc:
            logger.error("Cannot start voice controller: %s", exc)

    # ── Banner ───────────────────────────────────────────────────
    _print_banner(assistant, voice, cfg)

    # ── Main loop ────────────────────────────────────────────────
    try:
        while True:
            ret, frame = cam.read()
            if not ret or frame is None:
                continue

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            # Enhance frame
            frame = cv2.convertScaleAbs(frame, alpha=cfg.frame_alpha, beta=cfg.frame_beta)
            frame = cv2.GaussianBlur(frame, cfg.blur_kernel, 0)

            now = time.time()

            # ── Face mesh processing ─────────────────────────────
            lm = face_mesh.process(frame)

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

                # ── DWELL CLICK ──────────────────────────────────
                if dwell_enabled:
                    clicked, progress = dwell.update(tracker.cur_x, tracker.cur_y)
                    if progress > 0.05:
                        draw_dwell_arc(frame, npx, npy, progress)
                    if clicked:
                        pyautogui.click()
                        blink_feedback_text = "DWELL CLICK"
                        blink_feedback_until = now + cfg.dwell_feedback_duration
                        if assistant:
                            assistant.say("Dwell click")

                # ── BLINK DETECTION ──────────────────────────────
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
            if voice:
                cmd = voice.get_command()
                if cmd:
                    if "dwell" in cmd:
                        dwell_enabled = not dwell_enabled
                        msg = (
                            "Dwell click enabled"
                            if dwell_enabled
                            else "Dwell click disabled"
                        )
                        if assistant:
                            assistant.say(msg)
                    else:
                        drag_mode, should_exit = process_voice_command(
                            cmd, drag_mode, assistant, voice
                        )
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
                f"Voice : {'ON  (say help)' if voice else 'OFF'}",
                f"TTS   : {'ON' if assistant else 'OFF'}",
                f"Dwell : {'ON  (hold {cfg.dwell_time:.1f}s)' if dwell_enabled else 'OFF'}",
                f"Drag  : {'ON' if drag_mode else 'OFF'}",
                f"FPS   : {fps_display}",
            ]
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
            cv2.imshow(
                "Blink-Click Virtual Mouse  |  Accessibility Edition", frame
            )
            if cv2.waitKey(1) & 0xFF == 27:
                if assistant:
                    assistant.say("Goodbye")
                break

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")

    # ── CLEANUP ──────────────────────────────────────────────────
    logger.info("Shutting down …")
    if drag_mode:
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