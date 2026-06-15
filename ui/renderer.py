from __future__ import annotations

import math
import time
from typing import Tuple

import cv2
import mediapipe as mp

_DRAWING = mp.solutions.drawing_utils
_DRAWING_STYLES = mp.solutions.drawing_styles

STOP_BUTTON_X = 14
STOP_BUTTON_Y = 14
STOP_BUTTON_W = 110
STOP_BUTTON_H = 36
MIC_BUTTON_X = STOP_BUTTON_X + STOP_BUTTON_W + 12
MIC_BUTTON_Y = 14
MIC_BUTTON_W = 128
MIC_BUTTON_H = 36


def draw_status_panel(
    frame,
    x: int,
    y: int,
    lines: list[str],
    color: Tuple[int, int, int] = (0, 0, 0),
    alpha: float = 0.55,
) -> None:
    """Draw a semi-transparent panel with text lines."""
    pad = 8
    lh = 26
    pw = max(len(s) for s in lines) * 12 + pad * 2
    ph = len(lines) * lh + pad * 2
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + pw, y + ph), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    for i, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (x + pad, y + pad + (i + 1) * lh - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def draw_ear_bar(
    frame, avg_ear: float, threshold: float, x: int = 20, y: int = 60
) -> None:
    """Draw the Eye Aspect Ratio bar indicator."""
    bar_len = int(avg_ear * 280)
    bar_color = (0, 220, 0) if avg_ear > threshold else (0, 0, 220)
    cv2.rectangle(frame, (x, y), (x + bar_len, y + 18), bar_color, -1)
    cv2.rectangle(frame, (x, y), (x + 280, y + 18), (180, 180, 180), 1)
    cv2.putText(
        frame,
        "EYE",
        (x, y - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (220, 220, 220),
        1,
    )


def draw_nose_marker(frame, nose_px: int, nose_py: int) -> None:
    """Draw a triple-circle marker at the nose position."""
    cv2.circle(frame, (nose_px, nose_py), 7, (0, 230, 255), -1)
    cv2.circle(frame, (nose_px, nose_py), 12, (0, 230, 255), 2)
    cv2.circle(frame, (nose_px, nose_py), 18, (0, 200, 200), 1)


def draw_click_feedback(frame, text: str, w: int, h: int) -> None:
    """Flash a large label (LEFT CLICK / RIGHT CLICK) at screen center."""
    color = ((0, 255, 0) if "LEFT" in text else (0, 80, 255))
    cv2.putText(
        frame,
        text,
        (w // 2 - 90, h // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        color,
        3,
        cv2.LINE_AA,
    )


def draw_no_face_warning(frame, w: int, h: int) -> None:
    """Show a centered warning when no face is detected."""
    cv2.putText(
        frame,
        "No face detected - please look at camera",
        (w // 2 - 200, h // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
    )


def draw_no_hand_warning(frame, w: int, h: int) -> None:
    cv2.putText(
        frame,
        "No hand detected - raise your hand to the camera",
        (w // 2 - 220, h // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 120, 255),
        2,
    )


def draw_rest_reminder(frame, w: int, h: int) -> None:
    """Flash a rest reminder at the bottom of the frame."""
    cv2.putText(
        frame,
        "PLEASE REST YOUR EYES",
        (w // 2 - 150, h - 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )


def draw_voice_status(
    frame,
    w: int,
    h: int,
    listening: bool,
    last_cmd: str,
    cmd_age: float,
) -> None:
    """Draw a voice-status banner at the top-center of the frame."""
    y_base = 28

    if last_cmd and cmd_age < 2.5:
        label = f"CMD: {last_cmd}"
        text_w = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )[0][0]
        x = (w - text_w) // 2
        cv2.rectangle(
            frame,
            (x - 10, y_base - 22),
            (x + text_w + 10, y_base + 8),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            frame,
            label,
            (x, y_base),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 120),
            2,
            cv2.LINE_AA,
        )
    elif listening:
        label = "Listening ..."
        text_w = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )[0][0]
        x = (w - text_w) // 2
        cv2.rectangle(
            frame,
            (x - 10, y_base - 22),
            (x + text_w + 10, y_base + 8),
            (0, 0, 0),
            -1,
        )
        pulse_g = int(180 + 75 * math.sin(time.time() * 4))
        cv2.putText(
            frame,
            label,
            (x, y_base),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, pulse_g, 255),
            2,
            cv2.LINE_AA,
        )


def draw_hand_overlay(
    frame,
    *,
    landmarks,
    gesture: str,
    hand_detected: bool,
) -> None:
    if landmarks is not None:
        _DRAWING.draw_landmarks(
            frame,
            landmarks,
            mp.solutions.hands.HAND_CONNECTIONS,
            _DRAWING_STYLES.get_default_hand_landmarks_style(),
            _DRAWING_STYLES.get_default_hand_connections_style(),
        )

    label = f"HAND: {gesture.replace('_', ' ')}"
    cv2.rectangle(frame, (16, 82), (230, 112), (0, 0, 0), -1)
    cv2.putText(
        frame,
        label,
        (24, 103),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (0, 255, 255) if hand_detected else (0, 0, 255),
        2,
        cv2.LINE_AA,
    )


def draw_stop_button(frame, hover: bool) -> tuple[int, int, int, int]:
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


def draw_mic_button(
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
