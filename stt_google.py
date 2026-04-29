from typing import Optional

import speech_recognition as sr


class GoogleSTT:
    def __init__(
        self,
        *,
        language: str = "en-IN",
        mic_index: Optional[int] = None,
        energy_threshold: int = 300,
        dynamic_energy_threshold: bool = True,
        listen_timeout: float = 2.0,
        phrase_time_limit: float = 5.0,
    ) -> None:
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = energy_threshold
        self.recognizer.dynamic_energy_threshold = dynamic_energy_threshold
        self.language = language
        self.listen_timeout = listen_timeout
        self.phrase_time_limit = phrase_time_limit
        if mic_index is None:
            self.mic = sr.Microphone()
        else:
            self.mic = sr.Microphone(device_index=mic_index)

    def listen_command(self) -> str:
        try:
            with self.mic as source:
                print("Listening...")
                audio = self.recognizer.listen(
                    source,
                    timeout=self.listen_timeout,
                    phrase_time_limit=self.phrase_time_limit,
                )

            text = self.recognizer.recognize_google(audio, language=self.language)
            print("[COMMAND]:", text)
            return text.lower()

        except Exception as exc:
            print("Error:", exc)
            return ""
