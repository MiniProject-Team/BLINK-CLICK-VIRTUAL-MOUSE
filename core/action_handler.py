from __future__ import annotations

from typing import Optional

import pyautogui


class ActionHandler:
    def __init__(self, assistant: Optional[object] = None) -> None:
        self.assistant = assistant

    def move(self, x: int, y: int) -> None:
        pyautogui.moveTo(x, y, _pause=False)

    def click(self, click_type: str) -> None:
        if click_type == "left":
            pyautogui.click()
            if self.assistant:
                self.assistant.say("Click")
        elif click_type == "right":
            pyautogui.rightClick()
            if self.assistant:
                self.assistant.say("Right click")

    def scroll(self, amount: int) -> None:
        pyautogui.scroll(amount)
