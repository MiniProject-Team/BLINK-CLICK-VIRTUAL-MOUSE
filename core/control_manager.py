class ControlManager:
    def __init__(self, mode: str = "head") -> None:
        self.mode = mode

    def switch(self, mode: str) -> None:
        if mode in {"head", "hand"}:
            self.mode = mode

    def is_head(self) -> bool:
        return self.mode == "head"

    def is_hand(self) -> bool:
        return self.mode == "hand"
