class SystemState:
    def __init__(self) -> None:
        self.cursor_x = 0
        self.cursor_y = 0
        self.last_click = None
        self.face_detected = False
        self.hand_detected = False
        self.voice_active = True
        self.running = True
