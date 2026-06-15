from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import pyautogui

from utils.config import MouseConfig
from utils.helpers import OneEuroFilter, compute_eye_aspect_ratio

logger = logging.getLogger(__name__)


@dataclass
class FaceFrameResult:
    face_detected: bool
    cursor_x: Optional[int] = None
    cursor_y: Optional[int] = None
    click: Optional[str] = None
    avg_ear: float = 0.0
    nose_px: Optional[Tuple[int, int]] = None
    forehead_px: Optional[Tuple[int, int]] = None
    supports_blink: bool = False
    landmarks: Optional[object] = None


class BlinkDetector:
    """Detect intentional blinks via Eye Aspect Ratio (EAR)."""

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
        self._saw_left_closed: bool = False
        self._saw_right_closed: bool = False
        self._saw_both_closed: bool = False

    def update(self, left_ear: float, right_ear: float, now: float) -> Optional[str]:
        left_closed = left_ear < self.threshold
        right_closed = right_ear < self.threshold
        any_closed = left_closed or right_closed
        both_closed = left_closed and right_closed

        if any_closed:
            if not self._blink_detected:
                self._blink_start = now
                self._blink_detected = True
                self._saw_left_closed = False
                self._saw_right_closed = False
                self._saw_both_closed = False
                logger.debug(
                    "BlinkDetector closed eye(s) (left_ear=%.3f right_ear=%.3f threshold=%.3f)",
                    left_ear,
                    right_ear,
                    self.threshold,
                )

            self._saw_left_closed = self._saw_left_closed or left_closed
            self._saw_right_closed = self._saw_right_closed or right_closed
            self._saw_both_closed = self._saw_both_closed or both_closed
            return None

        if (
            self._blink_detected
            and (left_ear < self.release_threshold or right_ear < self.release_threshold)
        ):
            return None

        if self._blink_detected:
            duration = now - self._blink_start
            self._blink_detected = False
            logger.debug(
                "BlinkDetector reopened eyes (duration=%.3fs left_ear=%.3f right_ear=%.3f)",
                duration,
                left_ear,
                right_ear,
            )

            if duration >= self.intentional_duration:
                if now < self._cooldown_until:
                    logger.debug(
                        "BlinkDetector suppressed click during cooldown "
                        "(cooldown_remaining=%.3fs)",
                        self._cooldown_until - now,
                    )
                    return None

                click = "left" if self._saw_both_closed else "right"
                self._cooldown_until = now + self.click_cooldown_s
                logger.info(
                    "BlinkDetector emitted %s click (duration=%.3fs cooldown=%.2fs)",
                    click,
                    duration,
                    self.click_cooldown_s,
                )
                return click

            logger.debug(
                "BlinkDetector ignored short blink (duration=%.3fs minimum=%.3fs)",
                duration,
                self.intentional_duration,
            )

        return None


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


class HeadTracker:
    """Map nose tip coordinates to screen space with smoothing."""

    NOSE_TIP = 1
    FOREHEAD = 10
    L_TOP, L_BOTTOM, L_LEFT, L_RIGHT = 159, 145, 33, 133
    R_TOP, R_BOTTOM, R_LEFT, R_RIGHT = 386, 374, 362, 263

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
        self._nose_buf_x: list[float] = []
        self._nose_buf_y: list[float] = []
        self._last_filtered_x: float = self.cur_x
        self._last_filtered_y: float = self.cur_y

    def sync_to_cursor(self) -> None:
        pos_x, pos_y = pyautogui.position()
        self.cur_x = float(pos_x)
        self.cur_y = float(pos_y)
        self._nose_buf_x.clear()
        self._nose_buf_y.clear()
        self._last_filtered_x = self.cur_x
        self._last_filtered_y = self.cur_y
        self._filter_x = OneEuroFilter(
            min_cutoff=self.cfg.filter_min_cutoff,
            beta=self.cfg.filter_beta,
            d_cutoff=self.cfg.filter_d_cutoff,
        )
        self._filter_y = OneEuroFilter(
            min_cutoff=self.cfg.filter_min_cutoff,
            beta=self.cfg.filter_beta,
            d_cutoff=self.cfg.filter_d_cutoff,
        )

    @staticmethod
    def _apply_response_curve(value: float, curve: float) -> float:
        value = max(0.0, min(1.0, value))
        centered = (value * 2.0) - 1.0
        curved = math.copysign(abs(centered) ** curve, centered)
        return (curved + 1.0) / 2.0

    def update(self, nose_x: float, nose_y: float, now: float) -> Tuple[int, int]:
        self._nose_buf_x.append(nose_x)
        self._nose_buf_y.append(nose_y)
        if len(self._nose_buf_x) > self._LANDMARK_BUFFER_SIZE:
            self._nose_buf_x.pop(0)
            self._nose_buf_y.pop(0)
        avg_nx = sum(self._nose_buf_x) / len(self._nose_buf_x)
        avg_ny = sum(self._nose_buf_y) / len(self._nose_buf_y)

        cfg = self.cfg
        mx = (avg_nx - cfg.head_x_min) / (cfg.head_x_max - cfg.head_x_min)
        my = (avg_ny - cfg.head_y_min) / (cfg.head_y_max - cfg.head_y_min)
        mx = max(0.0, min(1.0, mx))
        my = max(0.0, min(1.0, my))
        mx = self._apply_response_curve(mx, cfg.pointer_response_curve)
        my = self._apply_response_curve(my, cfg.pointer_response_curve)

        raw_x = mx * self.screen_w
        raw_y = my * self.screen_h

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

        active_dead_zone = (
            cfg.precision_dead_zone_px if filtered_speed < 18.0 else cfg.dead_zone_px
        )
        if distance > active_dead_zone:
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


class FaceMeshProcessor:
    """Wrap MediaPipe FaceMesh initialization and per-frame processing."""

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

    def close(self) -> None:
        if self._mesh is not None:
            try:
                self._mesh.close()
            except Exception:
                pass


class FaceModule:
    def __init__(self, cfg: MouseConfig) -> None:
        self.processor = FaceMeshProcessor(cfg)
        self.tracker = HeadTracker(cfg)
        self.blink = BlinkDetector(cfg)

    def sync_to_cursor(self) -> None:
        self.tracker.sync_to_cursor()

    def close(self) -> None:
        self.processor.close()

    def process(self, frame, now: float) -> FaceFrameResult:
        landmarks = self.processor.process(frame)
        if not landmarks:
            return FaceFrameResult(face_detected=False, supports_blink=self.processor.supports_blink)

        h, w, _ = frame.shape
        nose = landmarks[HeadTracker.NOSE_TIP]
        forehead = landmarks[HeadTracker.FOREHEAD]
        scr_x, scr_y = self.tracker.update(nose.x, nose.y, now)

        click = None
        avg_ear = 0.0
        if self.processor.supports_blink:
            left_ear = compute_eye_aspect_ratio(
                landmarks,
                w,
                h,
                HeadTracker.L_TOP,
                HeadTracker.L_BOTTOM,
                HeadTracker.L_LEFT,
                HeadTracker.L_RIGHT,
            )
            right_ear = compute_eye_aspect_ratio(
                landmarks,
                w,
                h,
                HeadTracker.R_TOP,
                HeadTracker.R_BOTTOM,
                HeadTracker.R_LEFT,
                HeadTracker.R_RIGHT,
            )
            avg_ear = (left_ear + right_ear) / 2.0
            click = self.blink.update(left_ear, right_ear, now)

        return FaceFrameResult(
            face_detected=True,
            cursor_x=scr_x,
            cursor_y=scr_y,
            click=click,
            avg_ear=avg_ear,
            nose_px=(int(nose.x * w), int(nose.y * h)),
            forehead_px=(int(forehead.x * w), int(forehead.y * h)),
            supports_blink=self.processor.supports_blink,
            landmarks=landmarks,
        )
