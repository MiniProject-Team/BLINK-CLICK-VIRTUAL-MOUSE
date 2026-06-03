from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import cv2
import mediapipe as mp
import pyautogui

from mouse_controller import MouseConfig, OneEuroFilter

logger = logging.getLogger(__name__)


@dataclass
class HandFrameResult:
    hand_detected: bool
    gesture: str = "NONE"
    cursor_x: Optional[int] = None
    cursor_y: Optional[int] = None
    click: Optional[str] = None
    scroll_amount: int = 0
    pause: bool = False
    landmarks: Optional[object] = None


@dataclass(frozen=True)
class HandGestureMetrics:
    pinch_distance: float
    middle_pinch_distance: float
    cursor_source_x: float
    cursor_source_y: float
    scroll_center_y: int
    open_palm: bool
    left_click: bool
    right_click: bool
    scroll_mode: bool
    move_mode: bool


class HandGestureController:
    """Recognise simple mouse gestures from one hand."""

    WRIST = 0
    INDEX_MCP = 5
    THUMB_TIP = 4
    INDEX_PIP = 6
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_TIP = 12
    RING_PIP = 14
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_TIP = 20

    def __init__(self, cfg: MouseConfig) -> None:
        self.cfg = cfg
        self.screen_w, self.screen_h = pyautogui.size()
        self._hands = mp.solutions.hands.Hands(
            model_complexity=0,
            max_num_hands=1,
            min_detection_confidence=cfg.detection_confidence,
            min_tracking_confidence=cfg.tracking_confidence,
        )
        self._drawing = mp.solutions.drawing_utils
        self._drawing_styles = mp.solutions.drawing_styles
        self._filter_x, self._filter_y = self._create_filters()
        self.cur_x = self.screen_w / 2
        self.cur_y = self.screen_h / 2
        self._gesture_name = "NONE"
        self._gesture_frames = 0
        self._click_cooldown_until = 0.0
        self._scroll_anchor_y: Optional[int] = None
        self._left_click_latched = False
        self._right_click_latched = False

    def sync_to_cursor(self) -> None:
        """Reset smoothing around the current OS cursor position."""
        pos_x, pos_y = pyautogui.position()
        self.cur_x = float(pos_x)
        self.cur_y = float(pos_y)
        self._filter_x, self._filter_y = self._create_filters()
        self._reset_runtime_state()

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._hands.close()

    def process(self, frame, now: float) -> HandFrameResult:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._hands.process(rgb)
        if not result.multi_hand_landmarks:
            self._reset_runtime_state()
            return HandFrameResult(hand_detected=False)

        hand_landmarks = result.multi_hand_landmarks[0]
        h, w, _ = frame.shape
        metrics = self._measure_gesture(hand_landmarks, w, h)
        gesture = self._classify_gesture(metrics)

        stable = self._update_stable_gesture(gesture)
        cursor_x = None
        cursor_y = None
        click = None
        scroll_amount = 0
        pause = gesture == "PAUSE" and stable

        if gesture == "MOVE":
            cursor_x, cursor_y = self._map_cursor(
                metrics.cursor_source_x,
                metrics.cursor_source_y,
                now,
            )

        if gesture == "LEFT_CLICK":
            self._scroll_anchor_y = None
            if (
                stable
                and not self._left_click_latched
                and now >= self._click_cooldown_until
            ):
                click = "left"
                self._left_click_latched = True
                self._click_cooldown_until = now + self.cfg.hand_click_cooldown_s
        else:
            self._left_click_latched = False

        if gesture == "RIGHT_CLICK":
            self._scroll_anchor_y = None
            if (
                stable
                and not self._right_click_latched
                and now >= self._click_cooldown_until
            ):
                click = "right"
                self._right_click_latched = True
                self._click_cooldown_until = now + self.cfg.hand_click_cooldown_s
        else:
            self._right_click_latched = False

        if gesture == "SCROLL" and stable:
            scroll_amount = self._compute_scroll(metrics.scroll_center_y)
        else:
            self._scroll_anchor_y = None

        return HandFrameResult(
            hand_detected=True,
            gesture=gesture,
            cursor_x=cursor_x,
            cursor_y=cursor_y,
            click=click,
            scroll_amount=scroll_amount,
            pause=pause,
            landmarks=hand_landmarks,
        )

    def draw(self, frame, result: HandFrameResult) -> None:
        if result.landmarks is not None:
            self._drawing.draw_landmarks(
                frame,
                result.landmarks,
                mp.solutions.hands.HAND_CONNECTIONS,
                self._drawing_styles.get_default_hand_landmarks_style(),
                self._drawing_styles.get_default_hand_connections_style(),
            )

        label = f"HAND: {result.gesture.replace('_', ' ')}"
        cv2.rectangle(frame, (16, 82), (230, 112), (0, 0, 0), -1)
        cv2.putText(
            frame,
            label,
            (24, 103),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 255, 255) if result.hand_detected else (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    @staticmethod
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

    def _measure_gesture(
        self,
        hand_landmarks,
        width: int,
        height: int,
    ) -> HandGestureMetrics:
        thumb_tip = self._landmark_px(hand_landmarks, self.THUMB_TIP, width, height)
        index_tip = self._landmark_px(hand_landmarks, self.INDEX_TIP, width, height)
        middle_tip = self._landmark_px(hand_landmarks, self.MIDDLE_TIP, width, height)
        palm_span = self._palm_span(hand_landmarks, width, height)
        left_click_threshold = self._adaptive_threshold(
            self.cfg.hand_pinch_threshold_px,
            palm_span,
            ratio=0.34,
            ceiling_scale=1.9,
        )
        right_click_threshold = self._adaptive_threshold(
            self.cfg.hand_right_click_threshold_px,
            palm_span,
            ratio=0.28,
            ceiling_scale=1.85,
        )

        pinch_distance = self._distance(thumb_tip, index_tip)
        middle_pinch_distance = self._distance(thumb_tip, middle_tip)
        index_up = self._finger_is_up(
            hand_landmarks, self.INDEX_TIP, self.INDEX_PIP, axis="y"
        )
        middle_up = self._finger_is_up(
            hand_landmarks, self.MIDDLE_TIP, self.MIDDLE_PIP, axis="y"
        )
        ring_up = self._finger_is_up(
            hand_landmarks, self.RING_TIP, self.RING_PIP, axis="y"
        )
        pinky_up = self._finger_is_up(
            hand_landmarks, self.PINKY_TIP, self.PINKY_PIP, axis="y"
        )
        hand_center_x, hand_center_y = self._hand_center(hand_landmarks)
        scroll_center_y = int((index_tip[1] + middle_tip[1]) / 2)
        unpinched = pinch_distance > left_click_threshold + 10
        thumb_cleared = middle_pinch_distance > right_click_threshold + 8
        index_only = index_up and not middle_up and not ring_up and not pinky_up
        two_fingers = index_up and middle_up and not ring_up and not pinky_up
        three_fingers = index_up and middle_up and ring_up and not pinky_up

        return HandGestureMetrics(
            pinch_distance=pinch_distance,
            middle_pinch_distance=middle_pinch_distance,
            cursor_source_x=hand_center_x,
            cursor_source_y=hand_center_y,
            scroll_center_y=scroll_center_y,
            open_palm=(
                index_up
                and middle_up
                and ring_up
                and pinky_up
                and pinch_distance > self.cfg.hand_open_palm_release_px
                and middle_pinch_distance > right_click_threshold + 10
            ),
            left_click=pinch_distance <= left_click_threshold,
            right_click=three_fingers and unpinched and thumb_cleared,
            scroll_mode=two_fingers and unpinched and thumb_cleared,
            move_mode=index_only and unpinched,
        )

    @staticmethod
    def _classify_gesture(metrics: HandGestureMetrics) -> str:
        if metrics.open_palm:
            return "PAUSE"
        if metrics.left_click:
            return "LEFT_CLICK"
        if metrics.right_click:
            return "RIGHT_CLICK"
        if metrics.scroll_mode:
            return "SCROLL"
        if metrics.move_mode:
            return "MOVE"
        return "HOLD"

    def _reset_runtime_state(self) -> None:
        self._gesture_name = "NONE"
        self._gesture_frames = 0
        self._scroll_anchor_y = None
        self._left_click_latched = False
        self._right_click_latched = False

    def _create_filters(self) -> tuple[OneEuroFilter, OneEuroFilter]:
        return (
            OneEuroFilter(
                min_cutoff=self.cfg.hand_filter_min_cutoff,
                beta=self.cfg.hand_filter_beta,
                d_cutoff=self.cfg.filter_d_cutoff,
            ),
            OneEuroFilter(
                min_cutoff=self.cfg.hand_filter_min_cutoff,
                beta=self.cfg.hand_filter_beta,
                d_cutoff=self.cfg.filter_d_cutoff,
            ),
        )

    def _compute_scroll(self, center_y: int) -> int:
        if self._scroll_anchor_y is None:
            self._scroll_anchor_y = center_y
            return 0

        delta = self._scroll_anchor_y - center_y
        if abs(delta) < self.cfg.hand_scroll_deadband_px:
            return 0

        steps = int(delta / self.cfg.hand_scroll_step_px)
        if steps == 0:
            return 0

        self._scroll_anchor_y = center_y
        return max(-6, min(6, steps)) * self.cfg.hand_scroll_unit

    def _map_cursor(self, point_x: float, point_y: float, now: float) -> tuple[int, int]:
        cfg = self.cfg
        mapped_x = (point_x - cfg.hand_x_min) / (cfg.hand_x_max - cfg.hand_x_min)
        mapped_y = (point_y - cfg.hand_y_min) / (cfg.hand_y_max - cfg.hand_y_min)
        mapped_x = max(0.0, min(1.0, mapped_x))
        mapped_y = max(0.0, min(1.0, mapped_y))

        raw_x = mapped_x * self.screen_w
        raw_y = mapped_y * self.screen_h
        filtered_x = self._filter_x(raw_x, now)
        filtered_y = self._filter_y(raw_y, now)
        self.cur_x += (filtered_x - self.cur_x) * cfg.hand_cursor_lerp
        self.cur_y += (filtered_y - self.cur_y) * cfg.hand_cursor_lerp
        return int(self.cur_x), int(self.cur_y)

    def _hand_center(self, hand_landmarks) -> tuple[float, float]:
        wrist = hand_landmarks.landmark[self.WRIST]
        index_mcp = hand_landmarks.landmark[self.INDEX_MCP]
        middle_mcp = hand_landmarks.landmark[self.MIDDLE_MCP]
        pinky_mcp = hand_landmarks.landmark[self.PINKY_MCP]
        center_x = (wrist.x + index_mcp.x + middle_mcp.x + pinky_mcp.x) / 4.0
        center_y = (wrist.y + index_mcp.y + middle_mcp.y + pinky_mcp.y) / 4.0
        return center_x, center_y

    @staticmethod
    def _adaptive_threshold(
        base_px: int,
        palm_span: float,
        *,
        ratio: float,
        ceiling_scale: float,
    ) -> int:
        scaled = int(palm_span * ratio)
        return max(base_px, min(int(base_px * ceiling_scale), scaled))

    def _palm_span(self, hand_landmarks, width: int, height: int) -> float:
        index_mcp = self._landmark_px(hand_landmarks, self.INDEX_MCP, width, height)
        pinky_mcp = self._landmark_px(hand_landmarks, self.PINKY_MCP, width, height)
        wrist = self._landmark_px(hand_landmarks, self.WRIST, width, height)
        middle_mcp = self._landmark_px(hand_landmarks, self.MIDDLE_MCP, width, height)
        span = self._distance(index_mcp, pinky_mcp)
        depth = self._distance(wrist, middle_mcp)
        return max(span, depth * 1.25, 60.0)

    def _update_stable_gesture(self, gesture: str) -> bool:
        if gesture == self._gesture_name:
            self._gesture_frames += 1
        else:
            self._gesture_name = gesture
            self._gesture_frames = 1
        return self._gesture_frames >= self.cfg.hand_gesture_confirm_frames

    @staticmethod
    def _finger_is_up(hand_landmarks, tip_idx: int, pip_idx: int, axis: str) -> bool:
        tip = hand_landmarks.landmark[tip_idx]
        pip = hand_landmarks.landmark[pip_idx]
        if axis == "y":
            return tip.y < pip.y - 0.015
        return tip.x < pip.x - 0.015

    @staticmethod
    def _finger_is_active(hand_landmarks, tip_idx: int, pip_idx: int) -> bool:
        tip = hand_landmarks.landmark[tip_idx]
        pip = hand_landmarks.landmark[pip_idx]
        return tip.y < pip.y + 0.025

    @staticmethod
    def _landmark_px(hand_landmarks, idx: int, width: int, height: int) -> tuple[int, int]:
        point = hand_landmarks.landmark[idx]
        return int(point.x * width), int(point.y * height)

    @staticmethod
    def _distance(p1: tuple[int, int], p2: tuple[int, int]) -> float:
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])
