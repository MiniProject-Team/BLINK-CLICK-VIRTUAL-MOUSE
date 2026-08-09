from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class MouseConfig:
    """Central configuration for cursor, camera, and gesture settings."""

    # Camera
    camera_index: int = 0
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 60
    camera_backend: str = "auto"

    # Head-to-screen mapping boundaries (normalized face coords)
    head_x_min: float = 0.32
    head_x_max: float = 0.68
    head_y_min: float = 0.26
    head_y_max: float = 0.74

    # One-Euro filter (jitter removal)
    filter_min_cutoff: float = 0.4
    filter_beta: float = 0.08
    filter_d_cutoff: float = 1.0

    # Dead zone (pixels) - displacements below this are ignored
    dead_zone_px: int = 6
    precision_dead_zone_px: int = 3

    # Cursor interpolation factor (0-1, lower = smoother / laggier)
    cursor_lerp: float = 0.18
    cursor_fast_lerp: float = 0.34
    cursor_precision_lerp: float = 0.11
    max_cursor_step_px: int = 80
    max_precision_step_px: int = 26
    pointer_response_curve: float = 1.35

    # Blink detection
    blink_threshold: float = 0.17
    blink_adaptive_threshold: bool = True
    blink_closed_ratio: float = 0.62
    blink_min_threshold: float = 0.14
    blink_max_threshold: float = 0.24
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
    head_landmark_buffer_size: int = 5
    head_landmark_outlier_limit: float = 0.08

    # Hand-tracking cursor mapping boundaries (normalized hand coords)
    hand_x_min: float = 0.10
    hand_x_max: float = 0.90
    hand_y_min: float = 0.12
    hand_y_max: float = 0.88

    # Hand cursor smoothing
    hand_cursor_lerp: float = 0.32
    hand_filter_min_cutoff: float = 0.65
    hand_filter_beta: float = 0.12

    # Hand gesture thresholds
    hand_pinch_threshold_px: int = 34
    hand_right_click_threshold_px: int = 30
    hand_open_palm_release_px: int = 52
    hand_gesture_confirm_frames: int = 3
    hand_click_cooldown_s: float = 0.55
    hand_scroll_deadband_px: int = 12
    hand_scroll_step_px: int = 16
    hand_scroll_unit: int = 70
