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
8. [Voice Commands](#voice-commands)  
9. [Keyboard Shortcuts](#keyboard-shortcuts)  
10. [Configuration](#configuration)  
11. [Troubleshooting](#troubleshooting)  

---

## Overview

**Blink-Click Virtual Mouse** is a computer-vision and speech-recognition system that enables **completely hands-free** human–computer interaction. It uses:

| Technology | Purpose |
|---|---|
| **MediaPipe Face Mesh** | 468-point face landmark detection |
| **OpenCV** | Camera capture, frame processing, HUD rendering |
| **PyAutoGUI** | Cursor movement, clicks, scrolling, typing |
| **SpeechRecognition** | Google Speech API for voice commands |
| **pyttsx3** | Text-to-Speech for spoken feedback |

The system is split into **two core modules** for clean separation of concerns:

- **`mouse_controller.py`** — Camera, head tracking, blink detection, dwell-click, HUD overlays  
- **`speech_controller.py`** — Voice recognition, TTS engine, voice command processing  

The **`main.py`** file is the entry point that wires both modules together.

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

### Dwell-Click (Auto-Click)
- Hold the cursor **still for 1.5 seconds** → automatic **Left Click**.
- A visual **progress arc** appears around the cursor during the dwell countdown.
- Can be toggled ON/OFF with the voice command `"dwell"`.

### Voice Assistant (Speech Recognition + TTS)
- **Google Speech API** for high-accuracy online recognition.
- **pyttsx3** for spoken feedback (the assistant talks back).
- Recognises **any speech** — unknown phrases are echoed back so you know you were heard.
- Supports click, scroll, drag, type, open websites, and more (see [Voice Commands](#voice-commands)).

### HUD Overlay
- **Status panel** — Shows Voice / TTS / Dwell / Drag / FPS status.
- **EAR bar** — Visual indicator of eye openness (green = open, red = blink detected).
- **Click feedback** — Large text flashes ("LEFT CLICK", "RIGHT CLICK", "DWELL CLICK").
- **Nose marker** — Triple-circle overlay on the tracked nose position.
- **Rest reminder** — "PLEASE REST YOUR EYES" appears every 2 minutes.

---

## Project Structure

```
BLINK-CLICK-VIRTUAL-MOUSE/
├── main.py                 # Entry point — wires mouse + speech modules together
├── mouse_controller.py     # Head tracking, blink detection, dwell-click, camera, HUD
├── speech_controller.py    # Voice recognition, TTS assistant, command processing
├── README.md               # This file
└── .venv/                  # Python virtual environment (not committed)
```

### Module Responsibilities

| File | Classes / Functions |
|---|---|
| **mouse_controller.py** | `MouseConfig`, `OneEuroFilter`, `DwellClicker`, `BlinkDetector`, `CameraCapture`, `HeadTracker`, `FaceMeshProcessor`, drawing utilities |
| **speech_controller.py** | `AssistantVoice`, `VoiceController`, `process_voice_command()` |
| **main.py** | `main()` — initialises all components, runs the main loop |

---

## Prerequisites

- **Python 3.10+** (tested on 3.10, 3.11, 3.12)
- **Webcam** (built-in or USB)
- **Microphone** (for voice commands)
- **Internet connection** (required for Google Speech Recognition API)
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

> **Note (Windows):** If `PyAudio` fails to install, download the wheel from  
> https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio and install with  
> `pip install PyAudio‑0.2.14‑cp311‑cp311‑win_amd64.whl` (choose your Python version).

---

## How to Run

```bash
python main.py
```

On startup you will see:
1. A terminal banner showing the status of TTS, Voice, Cursor, and Click systems.
2. An OpenCV window titled **"Blink-Click Virtual Mouse | Accessibility Edition"**.
3. The assistant will greet you with a spoken message.

**To exit:** Press the **ESC** key, or say **"stop"** / **"exit"**.

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
2. When EAR drops below **0.20**, a blink is detected.
3. If the blink lasts longer than **0.35 seconds** (intentional), it triggers a **left click**.
4. If a second intentional blink occurs within **0.65 seconds**, it triggers a **right click**.

### Dwell-Click
1. If the cursor stays within a **25 px radius** for **1.5 seconds**, an automatic **left click** fires.
2. A coloured **arc** fills around the nose marker showing progress.

### Voice Commands
1. A background thread continuously listens via the **microphone**.
2. Audio is sent to **Google Speech Recognition API** for transcription.
3. Recognised text is matched against known commands.
4. The **TTS assistant** speaks a confirmation back.

---

## Voice Commands

| Command | Action |
|---|---|
| `"click"` | Left click at current cursor position |
| `"right click"` | Right click |
| `"double click"` | Double click |
| `"scroll up"` | Scroll up (8 units) |
| `"scroll down"` | Scroll down (8 units) |
| `"drag"` | Toggle drag mode ON/OFF |
| `"type hello world"` | Types "hello world" at the cursor |
| `"open google"` | Opens Google in the default browser |
| `"open youtube"` | Opens YouTube in the default browser |
| `"dwell"` | Toggle dwell-click ON/OFF |
| `"help"` | Read all available commands aloud |
| `"stop"` / `"exit"` | Quit the program |

> Any phrase that does **not** match a command is echoed back:  
> *"I heard: [your words]. I did not understand that command. Say help for options."*

---

## Keyboard Shortcuts

| Key | Action |
|---|---|
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
| `filter_min_cutoff` | `0.9` | One-Euro smoothness when still |
| `filter_beta` | `0.25` | One-Euro responsiveness when moving |
| `dead_zone_px` | `6` | Dead zone radius (pixels) |
| `dwell_time` | `1.5` | Seconds to hold still for dwell-click |
| `dwell_radius` | `25` | Dwell region radius (pixels) |
| `blink_threshold` | `0.20` | EAR value below which blink is detected |
| `intentional_blink_duration` | `0.35` | Minimum blink duration for a click (s) |
| `double_blink_gap` | `0.65` | Max gap for two blinks to count as right-click (s) |
| `rest_interval` | `120.0` | Seconds between rest reminders |

To change a parameter, edit `MouseConfig()` in `main.py` or pass values:
```python
cfg = MouseConfig(dwell_time=2.0, blink_threshold=0.22)
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
| `mediapipe` | >=0.10 | Face Mesh landmark detection |
| `pyautogui` | >=0.9 | Mouse/keyboard automation |
| `SpeechRecognition` | >=3.10 | Google Speech API wrapper |
| `pyttsx3` | >=2.90 | Offline text-to-speech |
| `PyAudio` | >=0.2.14 | Microphone access for SpeechRecognition |

---

## License

This project is developed as a mini-project for educational and accessibility purposes.

---

*Built with care for accessibility.*
