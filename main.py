"""
main.py - Blink-Click Virtual Mouse (Accessibility Edition)

Entry point wiring the core engine, modules, and UI renderer.

Run:
    python main.py

Press ESC to exit at any time.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import Optional

import cv2
import pyautogui

from core.action_handler import ActionHandler
from core.control_manager import ControlManager
from core.engine import Engine
from core.state import SystemState
from modules.face_module import CameraCapture, FaceModule
from modules.hand_module import HandModule
from modules.voice_module import VoiceModule
from speech_controller import AssistantVoice, TTS_AVAILABLE
from ui import renderer
from utils.config import MouseConfig

logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s  %(name)-22s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")

WINDOW_TITLE = "Blink-Click Virtual Mouse | Accessibility Edition"
EXIT_CONFIRM_WINDOW_S = 1.5


def _print_banner(
    assistant: Optional[AssistantVoice],
    voice: VoiceModule,
    cfg: MouseConfig,
    control_mode: str,
    hand_available: bool,
) -> None:
    print("\n" + "=" * 62)
    print("   BLINK-CLICK VIRTUAL MOUSE - Accessibility Edition")
    print("=" * 62)
    print(f"  TTS        : {'Active (will speak back)' if assistant else 'Disabled'}")
    if voice.voice_ready:
        print(
            "  Voice In   : Active (say '"
            f"{voice.wake_word}' to wake, vosk + google)"
        )
    else:
        print("  Voice In   : Disabled")
    if voice.brain and voice.cloud_brain:
        brain_label = f"Ollama {voice.brain.model} + Cloud {voice.cloud_brain.model}"
    elif voice.brain:
        brain_label = f"Ollama {voice.brain.model}"
    elif voice.cloud_brain:
        brain_label = f"Cloud {voice.cloud_brain.model}"
    else:
        brain_label = "Disabled"
    print(f"  Brain      : {brain_label}")
    if voice.voice_ready:
        print(f"  Wake Word  : {voice.wake_word}")
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
    print("  Press ESC to exit")
    print("=" * 62 + "\n")


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
    return renderer.draw_stop_button(frame, hover=hover)


def _draw_mic_button(
    frame,
    *,
    hover: bool,
    mic_available: bool,
    mic_enabled: bool,
) -> tuple[int, int, int, int]:
    return renderer.draw_mic_button(
        frame,
        hover=hover,
        mic_available=mic_available,
        mic_enabled=mic_enabled,
    )


# ================================================================
#  MAIN LOOP
# ================================================================
def main() -> None:
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

    assistant: Optional[AssistantVoice] = None
    if TTS_AVAILABLE:
        assistant = AssistantVoice()
        assistant.greet()
    else:
        logger.warning("pyttsx3 not found - TTS disabled. pip install pyttsx3")

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

    face_module = FaceModule(cfg)

    hand_module: Optional[HandModule] = None
    try:
        hand_module = HandModule(cfg)
        logger.info("Hand gesture controller initialised.")
    except Exception as exc:
        logger.warning("Hand gesture controller unavailable: %s", exc)
    control = ControlManager(_resolve_control_mode(hand_available=hand_module is not None))
    if control.is_hand() and hand_module:
        hand_module.sync_to_cursor()
    else:
        face_module.sync_to_cursor()

    state = SystemState()
    actions = ActionHandler(assistant)
    voice = VoiceModule(assistant)
    engine = Engine(face_module, hand_module, voice, actions, control, state)

    click_feedback_until: float = 0.0
    click_feedback_text: str = ""
    session_start = time.time()
    fps_time = time.time()
    fps_count = 0
    fps_display = 0
    exit_armed_until = 0.0

    # Banner
    _print_banner(
        assistant,
        voice,
        cfg,
        control.mode,
        hand_available=hand_module is not None,
    )
    if voice.voice_ready:
        print(f"System ready... Say '{voice.wake_word}'")

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

    def _set_control_mode(next_mode: str, source: str) -> None:
        next_mode = next_mode.strip().lower()
        if next_mode == "hand" and not hand_module:
            logger.info(
                "Hand mode requested by %s but hand tracking is unavailable.",
                source,
            )
            print("[Control] Hand mode is unavailable in this environment.")
            if assistant:
                assistant.say("Hand mode is unavailable")
            return
        if next_mode not in {"head", "hand"} or next_mode == control.mode:
            return

        control.switch(next_mode)
        if control.is_hand() and hand_module:
            hand_module.sync_to_cursor()
        else:
            face_module.sync_to_cursor()

        logger.info("Control mode switched to %s by %s.", control.mode, source)
        print(f"[Control] Switched to {control.mode.upper()} mode ({source})")
        if assistant:
            assistant.say(f"{control.mode} mode")

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

    # Main loop
    try:
        while state.running:
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

            output = engine.update(processing_frame, now)

            if output.click:
                click_feedback_text = "LEFT CLICK" if output.click == "left" else "RIGHT CLICK"
                click_feedback_until = now + cfg.click_feedback_duration

            if control.is_hand() and hand_module:
                if output.hand_result:
                    renderer.draw_hand_overlay(
                        frame,
                        landmarks=output.hand_result.landmarks,
                        gesture=output.hand_result.gesture,
                        hand_detected=True,
                    )
                else:
                    renderer.draw_no_hand_warning(frame, w, h)

            if control.is_head():
                if output.face_result and output.face_result.face_detected:
                    nose_px = output.face_result.nose_px
                    if nose_px:
                        renderer.draw_nose_marker(frame, nose_px[0], nose_px[1])
                    forehead_px = output.face_result.forehead_px
                    if forehead_px:
                        cv2.circle(frame, forehead_px, 4, (255, 200, 0), -1)
                    if output.face_result.supports_blink:
                        renderer.draw_ear_bar(
                            frame,
                            output.face_result.avg_ear,
                            cfg.blink_threshold,
                        )
                else:
                    renderer.draw_no_face_warning(frame, w, h)

            if voice.voice_ready:
                v_listening, v_last, last_time = voice.get_voice_overlay()
                v_age = now - last_time if last_time else 999
                renderer.draw_voice_status(frame, w, h, v_listening, v_last, v_age)

            if now < click_feedback_until and click_feedback_text:
                renderer.draw_click_feedback(frame, click_feedback_text, w, h)

            drag_mode = bool(voice.voice_processor and voice.voice_processor.drag_mode)
            status_lines = [
                f"Ctrl  : {control.mode.upper()}",
                (
                    f"Face  : {face_module.processor.mode.upper()}"
                    if control.is_head()
                    else "Hand  : MEDIAPIPE"
                ),
                f"Voice : {'ON  (vosk)' if voice.voice_ready else 'OFF'}",
                (
                    "Brain : OLLAMA+CLOUD"
                    if voice.brain and voice.cloud_brain
                    else "Brain : OLLAMA"
                    if voice.brain
                    else "Brain : CLOUD"
                    if voice.cloud_brain
                    else "Brain : OFF"
                ),
                f"TTS   : {'ON' if assistant else 'OFF'}",
                f"Drag  : {'ON' if drag_mode else 'OFF'}",
                f"FPS   : {fps_display}",
            ]
            if voice.voice_ready:
                status_lines.insert(1, voice.get_wake_status_text()[:28])
            if voice.voice_processor:
                status_lines.insert(2, voice.voice_processor.get_status_text()[:28])
            renderer.draw_status_panel(frame, w - 220, 4, status_lines)

            # Rest reminder
            if now - session_start > cfg.rest_interval:
                renderer.draw_rest_reminder(frame, w, h)
                session_start = now

            # FPS
            fps_count += 1
            if now - fps_time >= 1.0:
                fps_display = fps_count
                fps_count = 0
                fps_time = now

            # Stop and microphone buttons
            stop_button_state["rect"] = _draw_stop_button(
                frame,
                hover=bool(stop_button_state["hover"]),
            )
            mic_button_state["rect"] = _draw_mic_button(
                frame,
                hover=bool(mic_button_state["hover"]),
                mic_available=voice.voice_ready,
                mic_enabled=voice.is_mic_enabled(),
            )

            # Show frame
            cv2.imshow(WINDOW_TITLE, frame)
            key = cv2.waitKey(1) & 0xFF

            if stop_button_state["clicked"]:
                logger.info("Stop button clicked. Exiting application.")
                if assistant:
                    assistant.say("Stopping")
                break

            if mic_button_state["clicked"]:
                mic_button_state["clicked"] = False
                voice.toggle_microphone("button: MIC")
                continue

            if key in (ord("m"), ord("M")):
                voice.toggle_microphone("hotkey: M")
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

    # Cleanup
    logger.info("Shutting down ...")
    if voice.voice_processor and voice.voice_processor.drag_mode:
        pyautogui.mouseUp()
    voice.stop()
    if assistant:
        assistant.stop()
    if hand_module:
        hand_module.close()
    face_module.close()
    cam.release()
    cv2.destroyAllWindows()
    print("\nProgram closed. Goodbye!")


if __name__ == "__main__":
    main()
