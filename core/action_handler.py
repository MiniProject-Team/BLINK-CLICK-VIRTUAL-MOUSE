from __future__ import annotations

from typing import Optional

import pyautogui


class ActionHandler:
    def __init__(self, assistant: Optional[object] = None) -> None:
        self.assistant = assistant

    def move(self, x: int, y: int) -> None:
        screen_w, screen_h = pyautogui.size()
        safe_x = max(0, min(screen_w - 1, int(x)))
        safe_y = max(0, min(screen_h - 1, int(y)))
        pyautogui.moveTo(safe_x, safe_y, _pause=False)

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
