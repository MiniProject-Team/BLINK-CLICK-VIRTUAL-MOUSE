# Blink-Click Virtual Mouse — Accessibility Edition

> A hands-free computer interaction system designed for people with motor disabilities.  
> Control the mouse cursor with **head movements**, click with **eye blinks**, and issue commands with **voice**.

---

## Table of Contents

1. [Overview](#overview)  
2. [Features](#features)  
3. [Project Structure](#project-structure)  
4. [Prerequisites](#prerequisites)  
5. [Installation](#installation)  
6. [How to Run](#how-to-run)  
7. [How It Works](#how-it-works)  
8. [Voice Commands](#voice-tasks)  
9. [Advanced Voice Architecture](#advanced-voice-architecture)  
10. [Keyboard Shortcuts](#keyboard-shortcuts)  
11. [Configuration](#configuration)  
12. [Troubleshooting](#troubleshooting)  

---

## Overview

**Blink-Click Virtual Mouse** is a computer-vision and speech-recognition system that enables **completely hands-free** human–computer interaction. It uses:

| Technology | Purpose |
|---|---|
| **MediaPipe Face Mesh** | 468-point face landmark detection |
| **OpenCV** | Camera capture, frame processing, HUD rendering |
| **PyAutoGUI** | Cursor movement, clicks, scrolling, typing |
| **SpeechRecognition** | Google Speech API for multilingual voice commands |
| **Ollama (phi3)** | Local intent planning from natural speech |
| **pyttsx3** | Text-to-Speech for spoken feedback |

The system is split into **three core modules** for clean separation of concerns:

- **`mouse_controller.py`** — Camera, head tracking, blink detection, HUD overlays  
- **`speech_controller.py`** — Voice recognition, TTS engine, voice command processing  

- **`hand_controller.py`** â€” Hand tracking, gesture recognition, gesture-to-action mapping  

The **`main.py`** file is the entry point that wires the modules together.

---

## Features

### Cursor Control (Head Tracking)
- **Nose-tip tracking** — Move your head; the cursor follows your nose position.
- **One-Euro Filter** — Provides glass-smooth cursor movement, removes jitter and tremor.
- **Dead Zone** — Micro-tremors below 6 px are ignored to prevent accidental movement.
- **Configurable mapping** — Head movement zone maps to the full screen area.

### Click System (Blink Detection)
- **Single long blink** (>0.35 s) → **Left Click**
- **Double long blink** (two blinks within 0.65 s) → **Right Click**
- **EAR (Eye Aspect Ratio)** indicator bar shown on screen.

### Voice Assistant (Speech Recognition + TTS)
- **Google Speech API** for high-accuracy online recognition.
- **Far-field speech enhancement** with optional gain normalization, noise gate, and optional `noisereduce` filtering before speech-to-text.
- **Wake word support** — say **"Jarvis"** before your request, like a desktop assistant.
- **Strict wake gate** ignores ordinary speech. Say **"Jarvis <command>"**, or say **"Jarvis"** and speak one command in the next ten seconds.
- **Multi-language support** via `VOICE_LANGUAGES` plus built-in Hindi / Hinglish command normalization.
- **Hybrid planner** — local rule engine runs first; if no safe local match, optional **Ollama phi3** plans the task.
- **Cloud fallback brain** — optional OpenAI-compatible API can answer broader questions or plan requests that local logic cannot map safely.
- **Secure execution layer** blocks unsafe requests (delete/hack/system-level abuse) before action runs.
- **Confirmation flow** for sensitive actions: **"Jarvis confirm"** or **"Jarvis cancel"**.
- **Compound command handling** supports phrases like "open chrome and search AI tools".
- **Smart normalization** improves recognition quality (for example, "open up youtube" -> "open youtube").
- **Fuzzy wake-word matching** supports close variants like "jarvish" and "jarves".
- **Wake command window** listens for command text for a short time after wake-word detection.
- **Step-based action plans** are validated and executed one step at a time.
- **pyttsx3** for spoken feedback (the assistant talks back).
- Supports click, scroll, drag, type, open websites, key presses, and safe app launches.

### HUD Overlay
- **Status panel** — Shows Face / Voice / Brain / TTS / Drag / FPS plus live voice-task state.
- **EAR bar** — Visual indicator of eye openness (green = open, red = blink detected).
- **Click feedback** — Large text flashes ("LEFT CLICK", "RIGHT CLICK").
- **Nose marker** — Triple-circle overlay on the tracked nose position.
- **Rest reminder** — "PLEASE REST YOUR EYES" appears every 2 minutes.

---

## Project Structure

```
BLINK-CLICK-VIRTUAL-MOUSE/
├── main.py                 # Entry point — wires mouse + speech modules together
├── mouse_controller.py     # Head tracking, blink detection, camera, HUD
├── speech_controller.py    # Voice recognition, TTS assistant, command processing
├── README.md               # This file
└── .venv/                  # Python virtual environment (not committed)
```

### Module Responsibilities

| File | Classes / Functions |
|---|---|
| **mouse_controller.py** | `MouseConfig`, `OneEuroFilter`, `BlinkDetector`, `CameraCapture`, `HeadTracker`, `FaceMeshProcessor`, drawing utilities |
| **speech_controller.py** | `AssistantVoice`, `VoiceController`, `OllamaBrain`, `VoiceCommandProcessor` |
| **main.py** | `main()` — initialises all components, runs the main loop |

---

## Prerequisites

- **Python 3.10+** (tested on 3.10, 3.11, 3.12)
- **Webcam** (built-in or USB)
- **Microphone** (for voice commands)
- **Internet connection** (required for Google Speech Recognition API)
- **Ollama with phi3 model** (optional, for local command planning)
- **Cloud API key** (optional, for cloud planning / Q&A fallback)
- **Windows 10/11** (tested; macOS/Linux may need minor changes to camera backend)

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/BLINK-CLICK-VIRTUAL-MOUSE.git
cd BLINK-CLICK-VIRTUAL-MOUSE
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the virtual environment

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
.venv\Scripts\activate.bat
```

**macOS / Linux:**
```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install opencv-python mediapipe pyautogui SpeechRecognition pyttsx3 PyAudio
```

Optional speech enhancement:

```bash
pip install noisereduce
```

Optional direct window-control support:

```bash
pip install pygetwindow
```

### 5. Optional: Enable local Ollama brain

```bash
ollama pull phi3
```

The app auto-enables Ollama planning if available.

Environment variables:

- `OLLAMA_BRAIN=1` enable (default) or `0` disable
- `OLLAMA_MODEL=phi3` choose model
- `OLLAMA_HOST=http://127.0.0.1:11434` Ollama endpoint
- `OLLAMA_TIMEOUT=25` planner timeout in seconds
- `WAKE_WORD=jarvis` change the wake word if needed
- `VOICE_LANGUAGES=en-IN,hi-IN` try multiple recognition languages in order
- `VOICE_STRICT_WAKE_WORD=1` require the wake word for every command (default)
- `VOICE_WAKE_FUZZY_MATCH=0` accept only the wake word and configured aliases (default)
- `VOICE_COMMAND_WINDOW_S=10` time available for one command after the wake word
- `VOICE_CONVERSATION_MODE=1` enable optional follow-up commands after the first request; set `VOICE_STRICT_WAKE_WORD=0` to use it
- `VOICE_CONVERSATION_TIMEOUT_S=9` follow-up timeout in seconds
- `VOICE_FAR_FIELD_MODE=1` enable single-mic far-field tuning
- `VOICE_NOISE_REDUCTION=1` turn on optional `noisereduce` filtering when installed
- `VOICE_AMBIENT_REFRESH_S=45` refresh ambient-noise profile every N seconds while idle
- `VOICE_AMBIENT_SAMPLE_S=0.35` ambient sample duration for each refresh
- `VOICE_PAUSE_THRESHOLD=0.5` and `VOICE_PHRASE_THRESHOLD=0.25` tune command endpoint speed
- `CLOUD_BRAIN=1` enable optional cloud fallback when an API key is present
- `CLOUD_BRAIN_API_KEY=...` or `OPENAI_API_KEY=...` provide the cloud API key
- `CLOUD_BRAIN_MODEL=gpt-4o-mini` choose the cloud model
- `CLOUD_BRAIN_BASE_URL=https://api.openai.com/v1` choose an OpenAI-compatible endpoint

If the app hears your voice but does not perform the task, make sure Ollama is running:

```bash
ollama serve
```

Then confirm your model is available:

```bash
ollama list
```

> **Note (Windows):** If `PyAudio` fails to install, download the wheel from  
> https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio and install with  
> `pip install PyAudio‑0.2.14‑cp311‑cp311‑win_amd64.whl` (choose your Python version).

---

## How to Run

```bash
python main.py
```

To start directly in hand mode:

```powershell
$env:CONTROL_MODE="hand"
python main.py
```

Live mode switching is available while the camera window is open: `H` for hand mode, `F` for head mode, `M` for microphone toggle, and `ESC` to exit.

### Hand Gesture Mapping (Hand Mode)

| Gesture | Action |
|---|---|
| ☝️ Index finger only | Cursor movement |
| 🤏 Thumb + Index (Pinch) | Left click |
| ✌️ Two fingers + move | Scroll |
| 🤟 Three fingers (Index + Middle + Ring) | Right click |
| ✋ Open palm | Pause |

On startup you will see:
1. A terminal banner showing the status of TTS, Voice, Cursor, and Click systems.
2. An OpenCV window titled **"Blink-Click Virtual Mouse | Accessibility Edition"**.
3. The assistant will greet you with a spoken message.
4. Voice commands are wake-gated. Say **"Jarvis open YouTube"**, or say **"Jarvis"** and give one command within ten seconds.

**To exit:** Press the **ESC** key, or say **"Jarvis stop"**.

---

## Frontend Launcher

A simple React project page is included for local use. It shows public-facing project information and a **Start Project** button.

Run the launcher page:

```bash
python frontend_server.py
```

Run this from the same Python environment you use for `main.py`.

Then open:

```text
http://127.0.0.1:3000
```

The Start button launches `main.py` locally from the browser page through the local launcher server.

For React development, the frontend command now starts both Vite and the local launcher API:

```bash
cd frontend
npm run dev
```

Open the Vite address shown in the terminal. Its Start Project and microphone controls communicate with the actual desktop engine.

---

## How It Works

### Step-by-Step Flow

```
┌──────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Webcam      │────▶│  MediaPipe Face   │────▶│  Head Tracker   │
│   (threaded)  │     │  Mesh (468 pts)   │     │  (One-Euro +    │
│               │     │                    │     │   dead zone)    │
└──────────────┘     └──────────────────┘     └────────┬────────┘
                                                        │
                                                        ▼
                                               ┌────────────────┐
                                               │  pyautogui     │
                                               │  moveTo(x, y)  │
                                               └────────────────┘

┌──────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Eye Aspect   │────▶│  Blink Detector  │────▶│  Left / Right   │
│  Ratio (EAR)  │     │  (duration +     │     │  Click          │
│               │     │   double-blink)  │     │                 │
└──────────────┘     └──────────────────┘     └─────────────────┘

┌──────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Microphone   │────▶│  Google Speech   │────▶│  Command        │
│  (threaded)   │     │  Recognition     │     │  Processor      │
│               │     │                    │     │  + TTS feedback │
└──────────────┘     └──────────────────┘     └─────────────────┘
```

### Cursor Control
1. The **webcam** captures frames in a background thread (no blocking).
2. **MediaPipe Face Mesh** detects 468 facial landmarks.
3. The **nose-tip landmark** (index 1) is mapped from normalised face coordinates to screen coordinates.
4. A **One-Euro Filter** smooths the raw position to eliminate jitter.
5. A **dead zone** (6 px) prevents micro-tremors from moving the cursor.
6. `pyautogui.moveTo()` moves the system cursor.

### Blink Detection
1. **Eye Aspect Ratio (EAR)** is computed for both eyes using 4 landmarks each.
2. When EAR drops below the adaptive threshold, a blink is detected.
3. If the blink lasts longer than **0.35 seconds** (intentional), it triggers a **left click**.
4. If a second intentional blink occurs within **0.65 seconds**, it triggers a **right click**.

### Voice Understanding
1. A background thread continuously listens via the **microphone**.
2. Wake-word detection supports fuzzy matching and opens a short command window.
3. Optional far-field preprocessing boosts speech level, suppresses low-volume noise, and can apply `noisereduce` before transcription.
4. Audio is sent to **Google Speech Recognition API** using one or more configured languages.
5. Text is normalized to improve understanding of English, Hindi, and Hinglish-style phrasing.
6. Planner uses **local rules first**, then optional **Ollama**, then optional **cloud AI** for broader requests.
7. Security checks block risky requests and ask for spoken confirmation when needed.
8. Valid plans execute step-by-step, then the **TTS assistant** speaks feedback.

## Advanced Voice Architecture

### 1) Hybrid Planning
- Local planner (`_plan_task_locally`) handles common requests quickly.
- If no safe local match exists, `OllamaBrain.plan()` creates a structured task plan.
- If local planning still cannot resolve the request, optional `CloudBrain.plan()` can act as the final fallback.

### 2) Far-Field Voice Input
- Single-microphone audio can be normalized before transcription.
- Optional noise gate and `noisereduce` help in noisier rooms.
- The microphone picker prefers array-style input devices when available.

### 3) Continuous Conversation
- After a valid wake-word request, the assistant can stay active for a short follow-up window.
- During that window, extra commands like "scroll down" or "confirm" do not need the wake word again.
- Silence or timeout safely resets the system back to wake-word mode.

### 4) Multi-Language Recognition
- `VOICE_LANGUAGES` lets the recognizer try more than one language per utterance.
- Command normalization maps common Hindi / Hinglish phrases into the English action grammar used by the planner.

### 5) Local LLM (Ollama)
- Optional local model (`phi3`) improves natural language understanding.
- The model response is normalized into strict action JSON before execution.

### 6) Security Layer
- Unsafe terms are filtered via blocked-request and blocked-app rules.
- Risky actions are rejected before execution.

### 7) Confirmation Safety
- Sensitive actions move to a confirm state.
- User can continue with spoken confirm or cancel.

### 8) Compound Commands
- Multi-step speech commands are split, planned, and merged.
- The merged result runs as a single validated plan.

### 9) Text Normalization
- Noisy phrasing is normalized into cleaner intents.
- Improves hit rate for local rules and LLM prompts.

### 10) Fuzzy Wake Word
- Wake-word matching allows close pronunciation variants.
- Reduces false negatives from accent/noise variation.

### 11) Smart Listening Window
- After wake-word detection, command capture stays open briefly.
- If no command arrives in time, the window expires safely.

### 12) Step-based Action Planning
- Plans are represented as explicit action steps.
- Steps are validated and executed in sequence with error handling.

### 13) Built-in Automation Backends
- `pyautogui` controls keyboard and mouse actions.
- `webbrowser`, `subprocess`, and system search handle apps/sites.

### Final Voice Flow

Audio -> Wake Word -> Optional Far-Field Cleanup -> Speech to Text -> Text Normalization -> [Local Rules OR Ollama OR Cloud] -> Security Filter -> Action Plan -> Execution -> TTS Feedback

---

## Voice Tasks

The assistant uses a strict wake gate. Start each task with the wake word, then speak naturally:

- "Jarvis open youtube and search lo-fi music"
- "Jarvis scroll down a bit"
- "Jarvis minimize window"
- "Jarvis maximize screen"
- "Jarvis restore window"
- "Jarvis type hello this is ayush"
- "Jarvis press enter"
- "Jarvis open notepad"
- "Jarvis switch to hand mode"
- "Jarvis switch to head mode"
- "Jarvis stop"

If the intent is unclear, risky, or security-sensitive, the request is blocked or asks for confirmation.

---

## Keyboard Shortcuts

| Key | Action |
|---|---|
| **H** | Switch to hand gesture mode |
| **F** | Switch to head / blink mode |
| **M** | Toggle microphone on or off |
| **ESC** | Exit the program |

---

## Configuration

All tuneable parameters are centralised in the `MouseConfig` dataclass in [mouse_controller.py](mouse_controller.py):

| Parameter | Default | Description |
|---|---|---|
| `camera_index` | `0` | Webcam device index |
| `camera_width` | `640` | Capture width (px) |
| `camera_height` | `480` | Capture height (px) |
| `head_x_min / head_x_max` | `0.32 / 0.68` | Horizontal head mapping range |
| `head_y_min / head_y_max` | `0.26 / 0.74` | Vertical head mapping range |
| `filter_min_cutoff` | `0.4` | One-Euro smoothness when still |
| `filter_beta` | `0.08` | One-Euro responsiveness when moving |
| `dead_zone_px` | `6` | Dead zone radius (pixels) |
| `blink_threshold` | `0.17` | EAR value below which blink is detected |
| `blink_adaptive_threshold` | `True` | Learns the user's open-eye EAR baseline and adjusts blink sensitivity |
| `blink_closed_ratio` | `0.62` | Ratio of open-eye baseline used for adaptive blink detection |
| `intentional_blink_duration` | `0.36` | Minimum blink duration for a click (s) |
| `double_blink_gap` | `0.55` | Max gap for two blinks to count as right-click (s) |
| `click_cooldown_s` | `0.75` | Cooldown between intentional blink clicks (s) |
| `rest_interval` | `120.0` | Seconds between rest reminders |

To change a parameter, edit `MouseConfig()` in `main.py` or pass values:
```python
cfg = MouseConfig(blink_threshold=0.20, click_cooldown_s=0.90)
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| **Camera not opening** | Check that no other app is using the webcam. Try `camera_index=1`. |
| **Cursor too shaky** | Lower `filter_min_cutoff` (e.g., 0.5) or increase `dead_zone_px`. |
| **Cursor too slow / laggy** | Increase `filter_beta` (e.g., 0.5). |
| **Blinks not detected** | Increase `blink_threshold` (e.g., 0.25). Check lighting. |
| **Accidental clicks from natural blinks** | Increase `intentional_blink_duration` (e.g., 0.5). |
| **Voice not recognised** | Ensure internet connection. Check microphone in system settings. |
| **PyAudio install error** | See Installation note above. Use pre-built wheel. |
| **pyttsx3 error on exit** | This is a known pyttsx3 threading issue; safe to ignore. |
| **"No face detected" always shown** | Improve lighting. Sit closer to camera. Remove face obstructions. |

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `opencv-python` | >=4.6 | Camera capture & image processing |
| `mediapipe` | >=0.10 | Face landmarks; current Tasks builds use `models/face_landmarker.task` |
| `pyautogui` | >=0.9 | Mouse/keyboard automation |
| `SpeechRecognition` | >=3.10 | Google Speech API wrapper |
| `pyttsx3` | >=2.90 | Offline text-to-speech |
| `PyAudio` | >=0.2.14 | Microphone access for SpeechRecognition |

---

## License

This project is developed as a mini-project for educational and accessibility purposes.

---

*Built with care for accessibility.*
