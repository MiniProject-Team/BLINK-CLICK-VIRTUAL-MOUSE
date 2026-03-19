# -*- coding: utf-8 -*-
"""
mouse_controller.py
═══════════════════
Mouse cursor control and click system for Blink-Click Virtual Mouse.

This module provides:
  • Head-tracking cursor control via MediaPipe Face Mesh (nose-tip)
  • One-Euro filter for glass-smooth, tremor-resilient movement
  • Dead-zone filtering to ignore micro-tremors
  • Blink detection (single long blink → left click, double → right click)
    • (dwell-click feature removed)
  • Threaded camera capture for zero-blocking reads
  • HUD overlays (status panel, EAR bar, dwell arc, click feedback)

Author  : Blink-Click Virtual Mouse Team
"""

from __future__ import annotations

import cv2
import math
import time
import logging
import threading
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional, Tuple

import mediapipe as mp
import pyautogui

# ── Module logger ────────────────────────────────────────────────
logger = logging.getLogger(__name__)


# ================================================================
#  CONFIGURATION
# ================================================================
@dataclass
class MouseConfig:
    """Central configuration for all mouse-controller parameters."""

    # Camera
    camera_index: int = 0
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 60
    camera_backend: str = "auto"

    # Head-to-screen mapping boundaries (normalised face coords)
    head_x_min: float = 0.32
    head_x_max: float = 0.68
    head_y_min: float = 0.26
    head_y_max: float = 0.74

    # One-Euro filter (jitter removal)
    filter_min_cutoff: float = 0.4
    filter_beta: float = 0.08
    filter_d_cutoff: float = 1.0

    # Dead zone (pixels) – displacements below this are ignored
    dead_zone_px: int = 6
    precision_dead_zone_px: int = 3

    # Cursor interpolation factor (0–1, lower = smoother / laggier)
    cursor_lerp: float = 0.18
    cursor_fast_lerp: float = 0.34
    cursor_precision_lerp: float = 0.11
    max_cursor_step_px: int = 80
    max_precision_step_px: int = 26
    pointer_response_curve: float = 1.35

    # (dwell-click removed)

    # Blink detection
    blink_threshold: float = 0.17
    intentional_blink_duration: float = 0.36
    double_blink_gap: float = 0.55
    blink_release_margin: float = 0.02
    click_cooldown_s: float = 0.75

    # Feedback overlay duration (seconds)
    click_feedback_duration: float = 0.8

    # Frame enhancement
    frame_alpha: float = 1.25
    frame_beta: int = 12

    # Image smoothing kernel size
    blur_kernel: Tuple[int, int] = (3, 3)

    # Rest reminder interval (seconds)
    rest_interval: float = 120.0

    # MediaPipe Face Mesh confidence
    detection_confidence: float = 0.55
    tracking_confidence: float = 0.55


# ================================================================
#  ONE-EURO FILTER  – jitter-free smooth cursor
# ================================================================
class OneEuroFilter:
    """
    Implementation of the 1€ Filter (Casiez et al., 2012).

    Provides adaptive low-pass filtering:
      • Low jitter when still (controlled by min_cutoff)
      • Low lag when moving fast (controlled by beta)
    """

    def __init__(
        self,
        min_cutoff: float = 0.8,
        beta: float = 0.2,
        d_cutoff: float = 1.0,
    ) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x_prev: Optional[float] = None
        self._dx_prev: float = 0.0
        self._t_prev: Optional[float] = None

    def _alpha(self, t_e: float, cutoff: float) -> float:
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def __call__(self, x: float, t: Optional[float] = None) -> float:
        if t is None:
            t = time.time()
        if self._t_prev is None:
            self._t_prev = t
            self._x_prev = x
            return x

        t_e = max(t - self._t_prev, 1e-6)

        a_d = self._alpha(t_e, self.d_cutoff)
        dx = (x - self._x_prev) / t_e
        dx_hat = a_d * dx + (1 - a_d) * self._dx_prev

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(t_e, cutoff)
        x_hat = a * x + (1 - a) * self._x_prev

        self._x_prev = x_hat
        self._dx_prev = dx_hat
        self._t_prev = t
        return x_hat


# Dwell click feature removed.


# ================================================================
#  BLINK DETECTOR
# ================================================================
class BlinkDetector:
    """
    Detects intentional blinks via Eye Aspect Ratio (EAR).

    • Single long blink → LEFT CLICK
    • Two rapid long blinks → RIGHT CLICK
    """

    def __init__(self, cfg: MouseConfig) -> None:
        self.threshold = cfg.blink_threshold
        self.intentional_duration = cfg.intentional_blink_duration
        self.double_gap = cfg.double_blink_gap
        self.release_threshold = min(
            1.0,
            self.threshold + max(0.01, cfg.blink_release_margin),
        )
        self.click_cooldown_s = max(0.0, cfg.click_cooldown_s)
        self.feedback_duration = cfg.click_feedback_duration

        self._blink_start: float = 0.0
        self._blink_detected: bool = False
        self._cooldown_until: float = 0.0
        self._pending_single_blink_at: Optional[float] = None

    def update(self, avg_ear: float, now: float) -> Optional[str]:
        """
        Call every frame with the average EAR value.

        Returns
        -------
        "left", "right", or None
        """
        if avg_ear < self.threshold:
            if not self._blink_detected:
                self._blink_start = now
                self._blink_detected = True
                logger.debug(
                    "BlinkDetector closed eyes (ear=%.3f threshold=%.3f)",
                    avg_ear,
                    self.threshold,
                )
            return None

        # Keep the blink active until the eyes reopen clearly above the threshold.
        if self._blink_detected and avg_ear < self.release_threshold:
            return None

        if self._blink_detected:
            duration = now - self._blink_start
            self._blink_detected = False
            logger.debug(
                "BlinkDetector reopened eyes (duration=%.3fs ear=%.3f)",
                duration,
                avg_ear,
            )

            if duration >= self.intentional_duration:
                if self._pending_single_blink_at is not None:
                    gap = now - self._pending_single_blink_at
                else:
                    gap = None

                if gap is not None and gap < self.double_gap:
                    self._pending_single_blink_at = None
                    if now < self._cooldown_until:
                        logger.debug(
                            "BlinkDetector suppressed right click during cooldown "
                            "(gap=%.3fs cooldown_remaining=%.3fs)",
                            gap,
                            self._cooldown_until - now,
                        )
                        return None

                    self._cooldown_until = now + self.click_cooldown_s
                    logger.info(
                        "BlinkDetector emitted right click (gap=%.3fs cooldown=%.2fs)",
                        gap,
                        self.click_cooldown_s,
                    )
                    return "right"

                self._pending_single_blink_at = now
                logger.debug(
                    "BlinkDetector queued left click candidate "
                    "(duration=%.3fs wait_for_double=%.2fs)",
                    duration,
                    self.double_gap,
                )
                return None

            logger.debug(
                "BlinkDetector ignored short blink (duration=%.3fs minimum=%.3fs)",
                duration,
                self.intentional_duration,
            )

        if self._pending_single_blink_at is not None:
            pending_age = now - self._pending_single_blink_at
            if pending_age >= self.double_gap:
                self._pending_single_blink_at = None
                if now < self._cooldown_until:
                    logger.debug(
                        "BlinkDetector suppressed pending left click during cooldown "
                        "(ear=%.3f cooldown_remaining=%.3fs)",
                        avg_ear,
                        self._cooldown_until - now,
                    )
                    return None

                self._cooldown_until = now + self.click_cooldown_s
                logger.info(
                    "BlinkDetector emitted left click (cooldown=%.2fs)",
                    self.click_cooldown_s,
                )
                return "left"

        return None


# ================================================================
#  THREADED CAMERA CAPTURE
# ================================================================
class CameraCapture:
    """Non-blocking threaded camera reader."""

    def __init__(self, cfg: MouseConfig) -> None:
        self.backend_name = "unknown"
        self.camera_index = cfg.camera_index
        self._cap = self._open_camera(cfg)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.camera_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.camera_height)
        self._cap.set(cv2.CAP_PROP_FPS, cfg.camera_fps)

        self._ret: bool = False
        self._frame = None
        self._lock = threading.Lock()
        self._stopped = False

        self._thread = threading.Thread(target=self._update, daemon=True)
        self._thread.start()
        logger.info(
            "Camera capture started (index=%d, backend=%s)",
            self.camera_index,
            self.backend_name,
        )

    def _open_camera(self, cfg: MouseConfig):
        backend_map = {
            "auto": [("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF), ("ANY", cv2.CAP_ANY)],
            "dshow": [("DSHOW", cv2.CAP_DSHOW)],
            "msmf": [("MSMF", cv2.CAP_MSMF)],
            "any": [("ANY", cv2.CAP_ANY)],
        }
        backend_candidates = backend_map.get(cfg.camera_backend.lower(), backend_map["auto"])
        index_candidates = [cfg.camera_index]
        if cfg.camera_index == 0:
            index_candidates.append(1)

        for index in index_candidates:
            for backend_name, backend in backend_candidates:
                cap = cv2.VideoCapture(index, backend)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        self.backend_name = backend_name
                        self.camera_index = index
                        return cap
                cap.release()

        raise RuntimeError(
            "Could not open a working camera. Try CAMERA_INDEX=1 or CAMERA_BACKEND=msmf."
        )

    def _update(self) -> None:
        while not self._stopped:
            ret, frame = self._cap.read()
            with self._lock:
                self._ret = ret
                self._frame = frame
            if not ret:
                time.sleep(0.01)

    def read(self):
        with self._lock:
            if self._frame is not None:
                return self._ret, self._frame.copy()
            return False, None

    def release(self) -> None:
        self._stopped = True
        self._thread.join(timeout=2)
        self._cap.release()
        logger.info("Camera released.")


# ================================================================
#  HUD / OVERLAY DRAWING UTILITIES
# ================================================================
def _distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def _get_point(lm, idx: int, w: int, h: int) -> Tuple[int, int]:
    return (int(lm[idx].x * w), int(lm[idx].y * h))


def compute_eye_aspect_ratio(
    lm, w: int, h: int, top: int, bottom: int, left: int, right: int
) -> float:
    """Compute Eye Aspect Ratio (EAR) for a single eye."""
    t = _get_point(lm, top, w, h)
    b = _get_point(lm, bottom, w, h)
    l = _get_point(lm, left, w, h)
    r = _get_point(lm, right, w, h)
    v = _distance(t, b)
    h2 = _distance(l, r)
    return v / h2 if h2 != 0 else 1.0


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


# draw_dwell_arc removed (dwell-click feature disabled)


def draw_ear_bar(
    frame, avg_ear: float, threshold: float, x: int = 20, y: int = 60
) -> None:
    """Draw the Eye Aspect Ratio bar indicator."""
    bar_len = int(avg_ear * 280)
    bar_color = (0, 220, 0) if avg_ear > threshold else (0, 0, 220)
    cv2.rectangle(frame, (x, y), (x + bar_len, y + 18), bar_color, -1)
    cv2.rectangle(frame, (x, y), (x + 280, y + 18), (180, 180, 180), 1)
    cv2.putText(
        frame, "EYE", (x, y - 4),
        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1,
    )


def draw_nose_marker(frame, nose_px: int, nose_py: int) -> None:
    """Draw a triple-circle marker at the nose position."""
    cv2.circle(frame, (nose_px, nose_py), 7, (0, 230, 255), -1)
    cv2.circle(frame, (nose_px, nose_py), 12, (0, 230, 255), 2)
    cv2.circle(frame, (nose_px, nose_py), 18, (0, 200, 200), 1)


def draw_click_feedback(frame, text: str, w: int, h: int) -> None:
    """Flash a large label (LEFT CLICK / RIGHT CLICK) at screen centre."""
    color = ((0, 255, 0) if "LEFT" in text else (0, 80, 255))
    cv2.putText(
        frame, text, (w // 2 - 90, h // 2),
        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3, cv2.LINE_AA,
    )


def draw_no_face_warning(frame, w: int, h: int) -> None:
    """Show a centred warning when no face is detected."""
    cv2.putText(
        frame,
        "No face detected - please look at camera",
        (w // 2 - 200, h // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
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
    """Draw a voice-status banner at the top-centre of the frame.

    Shows:
      • a pulsing “Listening …” indicator when the mic is active
      • the last matched command for up to 2.5 s after recognition
    """
    y_base = 28

    if last_cmd and cmd_age < 2.5:
        # Show recognised command in green
        label = f"CMD: {last_cmd}"
        text_w = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )[0][0]
        x = (w - text_w) // 2
        # background pill
        cv2.rectangle(
            frame, (x - 10, y_base - 22), (x + text_w + 10, y_base + 8),
            (0, 0, 0), -1,
        )
        cv2.putText(
            frame, label, (x, y_base),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 120), 2, cv2.LINE_AA,
        )
    elif listening:
        # Pulsing "Listening..." indicator
        label = "Listening ..."
        text_w = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )[0][0]
        x = (w - text_w) // 2
        cv2.rectangle(
            frame, (x - 10, y_base - 22), (x + text_w + 10, y_base + 8),
            (0, 0, 0), -1,
        )
        # colour pulses via time
        pulse_g = int(180 + 75 * math.sin(time.time() * 4))
        cv2.putText(
            frame, label, (x, y_base),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, pulse_g, 255), 2, cv2.LINE_AA,
        )


# ================================================================
#  HEAD TRACKER  – maps nose position to screen coordinates
# ================================================================
class HeadTracker:
    """
    Translates nose-tip landmark position to screen coordinates
    using configurable mapping boundaries, One-Euro filtering,
    landmark averaging across frames, linear interpolation,
    and a dead-zone to suppress micro-tremors.
    """

    # MediaPipe Face Mesh landmark indices
    NOSE_TIP = 1
    FOREHEAD = 10
    L_TOP, L_BOTTOM, L_LEFT, L_RIGHT = 159, 145, 33, 133
    R_TOP, R_BOTTOM, R_LEFT, R_RIGHT = 386, 374, 362, 263

    # Number of frames to average the nose landmark over
    _LANDMARK_BUFFER_SIZE = 8

    def __init__(self, cfg: MouseConfig) -> None:
        self.cfg = cfg
        self.screen_w, self.screen_h = pyautogui.size()

        self._filter_x = OneEuroFilter(
            min_cutoff=cfg.filter_min_cutoff,
            beta=cfg.filter_beta,
            d_cutoff=cfg.filter_d_cutoff,
        )
        self._filter_y = OneEuroFilter(
            min_cutoff=cfg.filter_min_cutoff,
            beta=cfg.filter_beta,
            d_cutoff=cfg.filter_d_cutoff,
        )

        self.cur_x: float = self.screen_w / 2
        self.cur_y: float = self.screen_h / 2

        # Ring buffer for landmark averaging (tremor suppression)
        self._nose_buf_x: list[float] = []
        self._nose_buf_y: list[float] = []
        self._last_filtered_x: float = self.cur_x
        self._last_filtered_y: float = self.cur_y

    @staticmethod
    def _apply_response_curve(value: float, curve: float) -> float:
        """Reduce sensitivity near the center while preserving full-range reach."""
        value = max(0.0, min(1.0, value))
        centered = (value * 2.0) - 1.0
        curved = math.copysign(abs(centered) ** curve, centered)
        return (curved + 1.0) / 2.0

    def update(self, nose_x: float, nose_y: float, now: float) -> Tuple[int, int]:
        """
        Map normalised nose coordinates → filtered screen position.

        Pipeline:
          1. Average raw landmark over last N frames
          2. Map average to screen space
          3. One-Euro filter the screen coords
          4. Dead-zone gate
          5. LERP interpolation toward new position

        Returns
        -------
        (screen_x, screen_y) : Tuple[int, int]
        """
        # 1. Landmark averaging
        self._nose_buf_x.append(nose_x)
        self._nose_buf_y.append(nose_y)
        if len(self._nose_buf_x) > self._LANDMARK_BUFFER_SIZE:
            self._nose_buf_x.pop(0)
            self._nose_buf_y.pop(0)
        avg_nx = sum(self._nose_buf_x) / len(self._nose_buf_x)
        avg_ny = sum(self._nose_buf_y) / len(self._nose_buf_y)

        # 2. Map to screen
        cfg = self.cfg
        mx = (avg_nx - cfg.head_x_min) / (cfg.head_x_max - cfg.head_x_min)
        my = (avg_ny - cfg.head_y_min) / (cfg.head_y_max - cfg.head_y_min)
        mx = max(0.0, min(1.0, mx))
        my = max(0.0, min(1.0, my))
        mx = self._apply_response_curve(mx, cfg.pointer_response_curve)
        my = self._apply_response_curve(my, cfg.pointer_response_curve)

        raw_x = mx * self.screen_w
        raw_y = my * self.screen_h

        # 3. One-Euro filter
        fx = self._filter_x(raw_x, now)
        fy = self._filter_y(raw_y, now)
        filtered_speed = math.hypot(
            fx - self._last_filtered_x,
            fy - self._last_filtered_y,
        )
        self._last_filtered_x = fx
        self._last_filtered_y = fy

        dx = fx - self.cur_x
        dy = fy - self.cur_y
        distance = math.hypot(dx, dy)

        # 4. Dead-zone gate
        active_dead_zone = (
            cfg.precision_dead_zone_px if filtered_speed < 18.0 else cfg.dead_zone_px
        )
        if distance > active_dead_zone:
            # 5. Adaptive interpolation: smoother for precision, faster for travel.
            if distance < 40.0:
                lerp = cfg.cursor_precision_lerp
                step_limit = cfg.max_precision_step_px
            else:
                distance_ratio = min((distance - 40.0) / 220.0, 1.0)
                lerp = cfg.cursor_lerp + (
                    (cfg.cursor_fast_lerp - cfg.cursor_lerp) * distance_ratio
                )
                step_limit = cfg.max_cursor_step_px
            step_x = dx * lerp
            step_y = dy * lerp
            step_distance = math.hypot(step_x, step_y)

            if step_distance > step_limit:
                scale = step_limit / step_distance
                step_x *= scale
                step_y *= scale

            self.cur_x += step_x
            self.cur_y += step_y

        return int(self.cur_x), int(self.cur_y)


# ================================================================
#  FACE MESH MANAGER
# ================================================================
class FaceMeshProcessor:
    """Wraps MediaPipe FaceMesh initialisation and per-frame processing."""

    def __init__(self, cfg: MouseConfig) -> None:
        self.mode = "haar"
        self.supports_blink = False
        self._mesh = None
        self._face_cascade = None

        if hasattr(mp, "solutions"):
            self._mp_face_mesh = mp.solutions.face_mesh
            self._mesh = self._mp_face_mesh.FaceMesh(
                refine_landmarks=True,
                max_num_faces=1,
                min_detection_confidence=cfg.detection_confidence,
                min_tracking_confidence=cfg.tracking_confidence,
            )
            self.mode = "mesh"
            self.supports_blink = True
            logger.info("MediaPipe FaceMesh initialised.")
            return

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._face_cascade = cv2.CascadeClassifier(cascade_path)
        if self._face_cascade.empty():
            raise RuntimeError("Could not initialise MediaPipe or OpenCV face detection.")
        logger.warning(
            "MediaPipe FaceMesh is unavailable in this Python environment. "
            "Falling back to OpenCV Haar face detection with blink disabled."
        )

    def process(self, frame):
        """
        Run face-mesh on a BGR frame.

        Returns
        -------
        landmarks or None
        """
        if self.mode == "mesh":
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self._mesh.process(rgb)
            if result.multi_face_landmarks:
                return result.multi_face_landmarks[0].landmark
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(80, 80),
        )
        if len(faces) == 0:
            return None

        x, y, fw, fh = max(faces, key=lambda box: box[2] * box[3])
        landmarks = [SimpleNamespace(x=0.5, y=0.5) for _ in range(387)]

        def set_point(idx: int, px: float, py: float) -> None:
            landmarks[idx] = SimpleNamespace(
                x=max(0.0, min(1.0, px / frame.shape[1])),
                y=max(0.0, min(1.0, py / frame.shape[0])),
            )

        set_point(HeadTracker.NOSE_TIP, x + fw * 0.5, y + fh * 0.58)
        set_point(HeadTracker.FOREHEAD, x + fw * 0.5, y + fh * 0.18)
        set_point(HeadTracker.L_LEFT, x + fw * 0.30, y + fh * 0.38)
        set_point(HeadTracker.L_RIGHT, x + fw * 0.43, y + fh * 0.38)
        set_point(HeadTracker.L_TOP, x + fw * 0.365, y + fh * 0.35)
        set_point(HeadTracker.L_BOTTOM, x + fw * 0.365, y + fh * 0.41)
        set_point(HeadTracker.R_LEFT, x + fw * 0.57, y + fh * 0.38)
        set_point(HeadTracker.R_RIGHT, x + fw * 0.70, y + fh * 0.38)
        set_point(HeadTracker.R_TOP, x + fw * 0.635, y + fh * 0.35)
        set_point(HeadTracker.R_BOTTOM, x + fw * 0.635, y + fh * 0.41)
        return landmarks
