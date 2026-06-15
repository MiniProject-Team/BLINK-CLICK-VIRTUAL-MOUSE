from __future__ import annotations

from modules.hand_module import HandFrameResult, HandGestureController as _HandGestureController
from ui import renderer


class HandGestureController(_HandGestureController):
    """Compatibility wrapper for legacy imports."""

    def draw(self, frame, result: HandFrameResult) -> None:
        renderer.draw_hand_overlay(
            frame,
            landmarks=result.landmarks,
            gesture=result.gesture,
            hand_detected=result.hand_detected,
        )

    @staticmethod
    def draw_no_hand_warning(frame, w: int, h: int) -> None:
        renderer.draw_no_hand_warning(frame, w, h)
