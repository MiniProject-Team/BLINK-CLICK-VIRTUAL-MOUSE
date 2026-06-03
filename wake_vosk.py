import json
from typing import Optional

import sounddevice as sd
from vosk import KaldiRecognizer, Model


class VoskWake:
    def __init__(
        self,
        model_path: str,
        wake_word: str = "jarvis",
        *,
        device: Optional[int] = None,
        samplerate: int = 16000,
    ) -> None:
        self.model = Model(model_path)
        self.rec = KaldiRecognizer(self.model, samplerate)
        self.wake_word = wake_word.lower()
        self.device = device
        self.samplerate = samplerate

    def listen_wake(self, stop_event=None) -> bool:
        try:
            self.rec.Reset()
        except Exception:
            pass

        stream_kwargs = {
            "samplerate": self.samplerate,
            "blocksize": 8000,
            "dtype": "int16",
            "channels": 1,
        }
        if self.device is not None:
            stream_kwargs["device"] = self.device

        with sd.RawInputStream(**stream_kwargs) as stream:
            while True:
                if stop_event is not None and stop_event.is_set():
                    return False
                data, _ = stream.read(4000)
                if hasattr(data, "tobytes"):
                    data = data.tobytes()
                elif not isinstance(data, (bytes, bytearray)):
                    data = bytes(data)

                if self.rec.AcceptWaveform(data):
                    text = json.loads(self.rec.Result()).get("text", "")
                    if text:
                        print("[WAKE]:", text)
                    if self.wake_word in text:
                        return True
