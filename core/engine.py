from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from core.action_handler import ActionHandler
from core.control_manager import ControlManager
from core.state import SystemState
from modules.face_module import FaceFrameResult, FaceModule
from modules.hand_module import HandFrameResult, HandModule
from modules.voice_module import VoiceModule


@dataclass
class EngineOutput:
    face_result: Optional[FaceFrameResult]
    hand_result: Optional[HandFrameResult]
    click: Optional[str]


class Engine:
    def __init__(
        self,
        face: FaceModule,
        hand: Optional[HandModule],
        voice: VoiceModule,
        actions: ActionHandler,
        control: ControlManager,
        state: SystemState,
    ) -> None:
        self.face = face
        self.hand = hand
        self.voice = voice
        self.actions = actions
        self.control = control
        self.state = state

    def update(self, frame, now: float) -> EngineOutput:
        face_result: Optional[FaceFrameResult] = None
        hand_result: Optional[HandFrameResult] = None
        click_event: Optional[str] = None

        if self.control.is_head():
            face_result = self.face.process(frame, now)
            self.state.face_detected = bool(face_result.face_detected)
            self.state.hand_detected = False
            if face_result.face_detected:
                if face_result.cursor_x is not None and face_result.cursor_y is not None:
                    self.actions.move(face_result.cursor_x, face_result.cursor_y)
                    self.state.cursor_x = face_result.cursor_x
                    self.state.cursor_y = face_result.cursor_y
                if face_result.click:
                    self.actions.click(face_result.click)
                    self.state.last_click = face_result.click
                    click_event = face_result.click

        elif self.control.is_hand() and self.hand is not None:
            hand_result = self.hand.process(frame, now)
            self.state.hand_detected = hand_result is not None
            self.state.face_detected = False
            if hand_result is not None:
                if (
                    not hand_result.pause
                    and hand_result.cursor_x is not None
                    and hand_result.cursor_y is not None
                ):
                    self.actions.move(hand_result.cursor_x, hand_result.cursor_y)
                    self.state.cursor_x = hand_result.cursor_x
                    self.state.cursor_y = hand_result.cursor_y
                if hand_result.click:
                    self.actions.click(hand_result.click)
                    self.state.last_click = hand_result.click
                    click_event = hand_result.click
                if hand_result.scroll_amount:
                    self.actions.scroll(hand_result.scroll_amount)

        voice_cmd = self.voice.update()
        if voice_cmd == "exit":
            self.state.running = False

        return EngineOutput(face_result=face_result, hand_result=hand_result, click=click_event)
