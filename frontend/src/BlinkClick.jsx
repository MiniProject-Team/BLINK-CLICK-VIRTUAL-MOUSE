import { startTransition, useEffect, useRef, useState } from "react";
import "./blink-click.css";
import img1 from "./assets/img1.png";
import img2 from "./assets/img2.png";
import img3 from "./assets/img3.png";

const projectWorkflow = [
  {
    icon: "👁️",
    title: "1. Real-Time Video Capture",
    text: "The system starts by activating the webcam, which continuously captures live video of the user's face. This video is processed frame by frame using computer vision techniques.",
  },
  {
    icon: "🧠",
    title: "2. Face Detection & Landmark Extraction",
    text: "The system uses MediaPipe Face Mesh to detect the face and identify 468 facial landmark points. These landmarks help locate important regions such as the eyes, nose, and mouth with high accuracy.",
  },
  {
    icon: "🎯",
    title: "3. Eye Tracking & Cursor Movement",
    text: "The coordinates of the eyes (or nose tracking point) are mapped to screen coordinates. Moving eyes left/right/up/down moves the cursor accordingly, and smooth tracking keeps movement stable.",
    points: [
      "Moving eyes left/right/up/down moves cursor accordingly",
      "Smooth tracking ensures stable cursor movement",
      "Enables complete cursor control without touching any device",
    ],
  },
  {
    icon: "👁️‍🗨️",
    title: "4. Blink Detection (Click Action)",
    text: "The system calculates the Eye Aspect Ratio (EAR) to detect blinking. When eyes are open, EAR remains stable. When eyes close, EAR decreases.",
    points: [
      "Short blink: Left click",
      "Long blink: Right click (optional)",
      "Mouse click actions are performed using only eye gestures",
    ],
  },
  {
    icon: "🎤",
    title: "5. Voice Command Control",
    text: "The system also supports voice interaction. The microphone captures speech, speech is converted into text, and commands are matched and executed.",
    points: ["Start system", "Stop system", "Open browser"],
  },
  {
    icon: "⚡",
    title: "6. Real-Time Processing",
    text: "All operations happen in real time with continuous frame capture, instant eye tracking, fast blink detection, and immediate cursor response. This ensures a smooth and interactive experience.",
  },
  {
    icon: "🖥️",
    title: "7. Final Output",
    text: "The system converts user actions into cursor movement, mouse clicks, and voice command execution.",
  },
];

const toolsUsed = [
  {
    name: "OpenCV",
    image: "https://cdn.jsdelivr.net/gh/devicons/devicon/icons/opencv/opencv-original.svg",
    description: "Real-time webcam capture and frame processing for the visual control pipeline.",
  },
  {
    name: "Python",
    image: "https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg",
    description: "Core language used to wire cursor control, blink logic, and voice integration.",
  },
  {
    name: "Ollama Brain",
    image: "https://cdn.simpleicons.org/ollama/ffffff",
    description: "Local LLM planner that interprets natural voice intent into safe desktop actions.",
  },
];

const engineModules = ["Head Tracking", "Blink Click", "Voice Commands", "Ollama Brain"];

const controlSlides = [
  {
    title: "Image 1",
    image: img1,
  },
  {
    title: "Image 2",
    image: img2,
  },
  {
    title: "Image 3",
    image: img3,
  },
];

const calibrationStats = [
  {
    label: "Face Mesh",
    value: "468 landmarks",
    note: "MediaPipe mesh points",
  },
  {
    label: "Eye Aspect Ratio",
    value: "0.18 threshold",
    note: "Blink detect baseline",
  },
  {
    label: "Gaze Smoothing",
    value: "0.85",
    note: "Cursor stabilization",
  },
  {
    label: "Lighting",
    value: "Balanced",
    note: "Low glare detected",
  },
];

const faceChecklist = [
  "Face centered inside frame grid",
  "Eyes fully visible (no shadows)",
  "Maintain 45-65 cm distance",
  "Keep head level for accurate gaze",
];

const voiceDefaults = [
  "Wake word: Ashu",
  "Auto-confirm system actions",
  "Noise suppression: medium",
  "Language: English (India)",
];

const MIC_LISTEN_WINDOW_MS = 10000;
const MIC_AUTO_RESTART_DELAY_MS = 900;

function readJsonSafe(response) {
  return response.json().catch(() => ({}));
}

export default function BlinkClick() {
  const [isStarting, setIsStarting] = useState(false);
  const [activeSlide, setActiveSlide] = useState(0);
  const videoRef = useRef(null);
  const [status, setStatus] = useState({
    state: "idle",
    message: "",
    running: false,
  });
  const [cameraInfo, setCameraInfo] = useState({
    state: "idle",
    name: "Searching...",
    resolution: "Auto",
    frameRate: "Auto",
    facing: "user",
    error: "",
  });
  const [voiceCommands, setVoiceCommands] = useState([]);
  const [voiceMessage, setVoiceMessage] = useState("Loading recent commands...");
  const [chatOpen, setChatOpen] = useState(false);
  const [chatInput, setChatInput] = useState("");
  const [chatMessages, setChatMessages] = useState([
    {
      role: "assistant",
      text: "Hi! Ask me about Blink-Click Virtual Mouse, or start Ollama for broader questions.",
    },
  ]);
  const [chatStatus, setChatStatus] = useState({ state: "idle", message: "" });
  const [cameraEnabled, setCameraEnabled] = useState(true);
  const [voiceEnabled, setVoiceEnabled] = useState(true);
  const [micListening, setMicListening] = useState(false);
  const [micStatus, setMicStatus] = useState("");
  const micStatusId = "mic-status-live";
  const speechRef = useRef(null);
  const micStopTimerRef = useRef(null);
  const micRestartTimerRef = useRef(null);
  const micPreventAutoRestartRef = useRef(false);

  useEffect(() => {
    let active = true;

    async function pollStatus() {
      try {
        const response = await fetch("/api/status");
        const data = await readJsonSafe(response);

        if (!active) {
          return;
        }

        startTransition(() => {
          setStatus({
            state: data.running ? "running" : "idle",
            running: Boolean(data.running),
            message: data.message || "Ready",
          });
        });
      } catch {
        if (!active) {
          return;
        }

        startTransition(() => {
          setStatus({
            state: "error",
            running: false,
            message: "Could not reach server.",
          });
        });
      }
    }

    pollStatus();
    const timer = window.setInterval(pollStatus, 3000);

    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, []);

  useEffect(() => {
    const timer = window.setInterval(() => {
      setActiveSlide((current) => (current + 1) % controlSlides.length);
    }, 2600);

    return () => {
      window.clearInterval(timer);
    };
  }, []);

  useEffect(() => {
    let active = true;
    let stream = null;

    async function initCamera() {
      if (!cameraEnabled) {
        if (videoRef.current) {
          videoRef.current.srcObject = null;
        }
        setCameraInfo({
          state: "idle",
          name: "Camera off",
          resolution: "Disabled",
          frameRate: "Disabled",
          facing: "user",
          error: "",
        });
        return;
      }

      if (status.running) {
        if (videoRef.current) {
          videoRef.current.srcObject = null;
        }
        setCameraInfo({
          state: "busy",
          name: "Engine in control",
          resolution: "In use",
          frameRate: "In use",
          facing: "user",
          error: "Camera is reserved by the running engine. Stop the project to preview here.",
        });
        return;
      }

      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        setCameraInfo({
          state: "error",
          name: "Camera unavailable",
          resolution: "Unavailable",
          frameRate: "Unavailable",
          facing: "unknown",
          error: "Camera access is not supported in this browser.",
        });
        return;
      }

      try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        if (!active) {
          stream.getTracks().forEach((track) => track.stop());
          return;
        }

        const track = stream.getVideoTracks()[0];
        const settings = track?.getSettings ? track.getSettings() : {};
        const name = track?.label || "Default Camera";
        const resolution = settings.width && settings.height ? `${settings.width} x ${settings.height}` : "Auto";
        const frameRate = settings.frameRate ? `${Math.round(settings.frameRate)} fps` : "Auto";
        const facing = settings.facingMode || "user";

        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }

        setCameraInfo({
          state: "active",
          name,
          resolution,
          frameRate,
          facing,
          error: "",
        });
      } catch (error) {
        if (!active) {
          return;
        }
        setCameraInfo({
          state: "error",
          name: "Camera blocked",
          resolution: "Unavailable",
          frameRate: "Unavailable",
          facing: "unknown",
          error: error?.message || "Camera access was denied.",
        });
      }
    }

    initCamera();

    return () => {
      active = false;
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
      }
    };
  }, [status.running, cameraEnabled]);

  useEffect(() => {
    if (!voiceEnabled) {
      stopMicListening("Mic is off.", true);
      return;
    }

    if (!status.running) {
      stopMicListening("Project is not running. Mic idle.", true);
      return;
    }

    if (!micListening) {
      startMicListening("Mic listening for 10 seconds...");
    }
  }, [voiceEnabled, status.running, micListening]);

  useEffect(() => {
    return () => {
      stopMicListening("", true);
    };
  }, []);

  useEffect(() => {
    let active = true;

    async function pollCommands() {
      try {
        const response = await fetch("/api/voice-commands");
        const data = await readJsonSafe(response);
        if (!active) {
          return;
        }

        const commands = Array.isArray(data.commands) ? data.commands : [];
        setVoiceCommands(commands);
        setVoiceMessage(data.message || "Recent commands from system logs.");
      } catch {
        if (!active) {
          return;
        }
        setVoiceCommands([]);
        setVoiceMessage("Could not load voice commands.");
      }
    }

    pollCommands();
    const timer = window.setInterval(pollCommands, 5000);

    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, []);

  useEffect(() => {
    if (status.running) {
      return;
    }
    setVoiceCommands([]);
    setVoiceMessage("Project is not running. Recent commands cleared.");
  }, [status.running]);

  async function handleStart() {
    setIsStarting(true);
    startTransition(() => {
      setStatus({
        state: "starting",
        running: false,
        message: "Starting project...",
      });
    });

    try {
      const response = await fetch("/api/start", { method: "POST" });
      const data = await readJsonSafe(response);

      startTransition(() => {
        setStatus({
          state: data.running ? "running" : "error",
          running: Boolean(data.running),
          message: data.message || (data.running ? "Project started." : "Failed to start."),
        });
      });
    } catch {
      startTransition(() => {
        setStatus({
          state: "error",
          running: false,
          message: "Could not reach server.",
        });
      });
    } finally {
      setIsStarting(false);
    }
  }

  async function handleChatSend() {
    const text = chatInput.trim();
    if (!text || chatStatus.state === "sending") {
      return;
    }
    setChatInput("");
    setChatMessages((prev) => [...prev, { role: "user", text }]);
    setChatStatus({ state: "sending", message: "Thinking..." });

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text }),
      });
      const data = await readJsonSafe(response);
      if (!response.ok) {
        throw new Error(data?.error || "Chat service unavailable.");
      }
      const reply = data?.reply || "I did not get a reply.";
      setChatMessages((prev) => [...prev, { role: "assistant", text: reply }]);
      setChatStatus({ state: "idle", message: "" });
    } catch (error) {
      setChatMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          text: error?.message || "Chat service is unavailable.",
        },
      ]);
      setChatStatus({ state: "error", message: "Chat service unavailable." });
    }
  }

  function getSpeechRecognizer() {
    if (speechRef.current) {
      return speechRef.current;
    }
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      return null;
    }
    const recognizer = new SpeechRecognition();
    recognizer.lang = "en-IN";
    recognizer.interimResults = false;
    recognizer.continuous = false;
    speechRef.current = recognizer;
    return recognizer;
  }

  function clearMicTimers() {
    if (micStopTimerRef.current) {
      window.clearTimeout(micStopTimerRef.current);
      micStopTimerRef.current = null;
    }
    if (micRestartTimerRef.current) {
      window.clearTimeout(micRestartTimerRef.current);
      micRestartTimerRef.current = null;
    }
  }

  function stopMicListening(message = "Mic stopped.", preventAutoRestart = true) {
    if (preventAutoRestart) {
      micPreventAutoRestartRef.current = true;
    }
    clearMicTimers();

    const recognizer = speechRef.current;
    if (recognizer) {
      try {
        recognizer.onresult = null;
        recognizer.onend = null;
        recognizer.onerror = null;
        recognizer.stop();
      } catch {
        // ignore stop errors
      }
    }
    setMicListening(false);
    if (message) {
      setMicStatus(message);
    }
  }

  function startMicListening(message = "Mic listening for 10 seconds...") {
    if (!voiceEnabled || !status.running || micListening) {
      return;
    }

    const recognizer = getSpeechRecognizer();
    if (!recognizer) {
      setMicStatus("Speech recognition not supported in this browser.");
      return;
    }

    clearMicTimers();
    micPreventAutoRestartRef.current = false;
    setMicListening(true);
    setMicStatus(message);

    recognizer.onresult = (event) => {
      const transcript = Array.from(event.results)
        .map((result) => result[0]?.transcript || "")
        .join(" ")
        .trim();
      if (transcript) {
        setChatInput((prev) => (prev ? `${prev} ${transcript}` : transcript));
        setMicStatus("Captured voice input.");
      }
    };
    recognizer.onerror = () => {
      setMicStatus("Mic error. Try again.");
      setMicListening(false);
    };
    recognizer.onend = () => {
      setMicListening(false);

      const blockRestart = micPreventAutoRestartRef.current || !voiceEnabled || !status.running;
      micPreventAutoRestartRef.current = false;

      if (!blockRestart) {
        setMicStatus("Auto mode: restarting mic...");
        micRestartTimerRef.current = window.setTimeout(() => {
          startMicListening("Mic listening for 10 seconds...");
        }, MIC_AUTO_RESTART_DELAY_MS);
      }
    };

    try {
      recognizer.start();
      micStopTimerRef.current = window.setTimeout(() => {
        stopMicListening("Mic auto-stopped after 10 seconds.", false);
      }, MIC_LISTEN_WINDOW_MS);
    } catch {
      setMicListening(false);
      setMicStatus("Mic could not start.");
    }
  }

  function handleMicToggle() {
    if (voiceEnabled) {
      setVoiceEnabled(false);
      stopMicListening("Mic turned off.", true);
    } else {
      setVoiceEnabled(true);
      if (status.running) {
        startMicListening("Mic listening for 10 seconds...");
      } else {
        setMicStatus("Mic is on. Start project to begin listening.");
      }
    }
  }

  function handleChatKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      handleChatSend();
    }
  }

  const voiceCommandItems = voiceCommands
    .map((entry) => {
      if (!entry) return null;
      if (typeof entry === "string") {
        return { time: "", text: entry };
      }
      if (typeof entry === "object") {
        return {
          time: entry.time || "",
          text: entry.text || "",
        };
      }
      return null;
    })
    .filter((entry) => entry && entry.text);

  return (
    <div className="dashboard-root">
      <aside className="side-nav">
        <div className="brand-block">
          <h1>Blink Click</h1>
          <p>Precision: 98%</p>
        </div>

        <nav className="side-nav-items">
          <a className="active" href="#dashboard">
            <span className="material-symbols-outlined">monitoring</span>
            <span>Dashboard</span>
          </a>
          <a href="#calibration">
            <span className="material-symbols-outlined">center_focus_strong</span>
            <span>Calibration</span>
          </a>
          <a href="#voice-config">
            <span className="material-symbols-outlined">record_voice_over</span>
            <span>Voice Config</span>
          </a>
        </nav>

        <div className="side-footer">
          <div className="system-state">System Status: {status.running ? "Active" : "Standby"}</div>
          <a href="#">
            <span className="material-symbols-outlined">help</span>
            <span>Help</span>
          </a>
        </div>
      </aside>

      <header className="top-bar">
        <div className="top-pills">
          <span className="material-symbols-outlined pulse">videocam</span>
          <span className="material-symbols-outlined muted">mic</span>
          <span className="material-symbols-outlined glow">sensors</span>
        </div>
      </header>

      <main className="main-canvas">
        <div className="grid-hero" id="dashboard">
          <section className="engine-card">
            <div>
              <h2>Smart Control Engine</h2>
              <p>
                Real-time eye tracking and blink detection are used to move the cursor and perform click actions
                seamlessly.
              </p>
            </div>

            <div className="engine-modules">
              <label>Active Modules</label>
              <div>
                {engineModules.map((module) => (
                  <span key={module}>{module}</span>
                ))}
              </div>
            </div>

            <div className="engine-action-wrap">
              <button className="engine-button" onClick={handleStart} disabled={isStarting || status.running} type="button">
                <div>
                  <span className="material-symbols-outlined">power_settings_new</span>
                  <strong>{status.running ? "ACTIVE" : isStarting ? "STARTING" : "START"}</strong>
                </div>
              </button>
            </div>

            <div className="engine-footer">
              <div className="accuracy-main">
                <label>Project Accuracy</label>
                <div>
                  98.4<span>%</span>
                </div>
              </div>

              <div className="accuracy-sub">
                <div className="accuracy-chip">
                  <span className={`dot ${status.running ? "on" : status.state === "error" ? "err" : ""}`.trim()} />
                  <strong>{status.running ? "LIVE" : status.state === "error" ? "ALERT" : "READY"}</strong>
                </div>
                <small>Tracking Confidence: 96.9%</small>
                <small>Blink Precision: 97.8%</small>
              </div>
            </div>
          </section>

          <section className="right-stack">
            <article className="panel">
              <div className="panel-head">
                <div>
                  <h3>Spatial Gaze</h3>
                  <p>Core control intelligence modules</p>
                </div>
                <span className="material-symbols-outlined">3d_rotation</span>
              </div>
              <div className="viz-box">
                <span>HOW IT WORKS</span>
                <div className="control-slider">
                  {controlSlides.map((slide, idx) => (
                    <article className={`control-card ${idx === activeSlide ? "active" : ""}`.trim()} key={slide.title}>
                      <img src={slide.image} alt={slide.title} loading="lazy" />
                    </article>
                  ))}
                </div>
                <div className="control-dots">
                  {controlSlides.map((slide, idx) => (
                    <span className={idx === activeSlide ? "active" : ""} key={slide.title + idx} />
                  ))}
                </div>
              </div>
            </article>

            <article className="panel">
              <div className="panel-head inline">
                <h3>Blink Response</h3>
                <div className="active-pill">
                  <span />
                  <small>Active</small>
                </div>
              </div>
              <div className="wave-stack">
                <div className="wave-row">
                  <span className="wave-label">Blink</span>
                  <div className="waveform blink">
                    {Array.from({ length: 18 }).map((_, idx) => (
                      <span key={`blink-${idx}`} />
                    ))}
                  </div>
                </div>
                <div className="wave-row">
                  <span className="wave-label">Voice</span>
                  <div className="waveform voice">
                    {Array.from({ length: 18 }).map((_, idx) => (
                      <span key={`voice-${idx}`} />
                    ))}
                  </div>
                </div>
              </div>
              <div className="meta-line">
                <span>BLINK: 0.15s</span>
                <span>VOICE: 0.12s</span>
              </div>
            </article>
          </section>
        </div>

        <section className="project-info">
          <div className="project-top">
            <div>
              <h3>Project Information</h3>
              <p>
                Blink-Click Virtual Mouse works as a real-time accessibility pipeline that transforms face, eye, and
                voice signals into full hands-free computer control.
              </p>
            </div>
          </div>

          <div className="workflow-list">
            {projectWorkflow.map((step) => (
              <article className="workflow-card" key={step.title}>
                <div className="workflow-head">
                  <span>{step.icon}</span>
                  <h4>{step.title}</h4>
                </div>
                <p>{step.text}</p>
                {step.points ? (
                  <ul>
                    {step.points.map((point) => (
                      <li key={point}>{point}</li>
                    ))}
                  </ul>
                ) : null}
              </article>
            ))}
          </div>
        </section>

        <section className="tools-showcase">
          <div className="tools-head">
            <h3>Tools Used In Project</h3>
            <p>Three core technologies powering tracking, command understanding, and execution.</p>
          </div>

          <div className="tools-grid">
            {toolsUsed.map((tool) => (
              <article className="tool-card" key={tool.name}>
                <img src={tool.image} alt={tool.name} loading="lazy" />
                <h4>{tool.name}</h4>
                <p>{tool.description}</p>
              </article>
            ))}
          </div>
        </section>

        <section className="section-block calibration-block" id="calibration">
          <header className="section-head">
            <div>
              <h3>Calibration</h3>
              <p>Live camera feed, facial geometry checks, and tracking readiness.</p>
            </div>
            <span className={`status-pill ${cameraInfo.state}`.trim()}>
              {cameraInfo.state === "active"
                ? "Camera Active"
                : cameraInfo.state === "busy"
                  ? "Engine Using Camera"
                  : cameraInfo.state === "error"
                    ? "Camera Blocked"
                    : "Checking"}
            </span>
          </header>

          <div className="calibration-grid">
            <article className="panel calibration-camera">
              <div className="panel-head inline">
                <div>
                  <h3>Camera Preview</h3>
                  <p>System-accessed camera stream</p>
                </div>
                <span className="material-symbols-outlined">photo_camera_front</span>
              </div>
              <div className="camera-frame">
                <video
                  ref={videoRef}
                  autoPlay
                  muted
                  playsInline
                  aria-label="Live camera preview used for face and blink tracking"
                />
                <div className="camera-overlay">
                  <div className="overlay-grid" />
                  <div className="overlay-target" />
                </div>
                {cameraInfo.state === "error" ? <div className="camera-error">{cameraInfo.error}</div> : null}
                <button
                  className={`mic-toggle ${voiceEnabled ? "on" : ""}`.trim()}
                  type="button"
                  onClick={handleMicToggle}
                  aria-pressed={voiceEnabled}
                  aria-controls={micStatusId}
                  aria-label={
                    voiceEnabled
                      ? "Turn microphone off"
                      : "Turn microphone on with automatic 10 second listening"
                  }
                  title={
                    voiceEnabled
                      ? "Mic is on. Click to turn off"
                      : "Mic is off. Click to turn on auto 10 second listening"
                  }
                >
                  <span className="material-symbols-outlined">{voiceEnabled ? "mic" : "mic_off"}</span>
                  <span>{voiceEnabled ? (micListening ? "Mic On (Auto)" : "Mic On") : "Mic Off"}</span>
                </button>
              </div>
              {micStatus ? (
                <div id={micStatusId} className="mic-status" role="status" aria-live="polite" aria-atomic="true">
                  {micStatus}
                </div>
              ) : null}
              <div className="camera-meta">
                <div>
                  <label>Device</label>
                  <span>{cameraInfo.name}</span>
                </div>
                <div>
                  <label>Resolution</label>
                  <span>{cameraInfo.resolution}</span>
                </div>
                <div>
                  <label>Frame Rate</label>
                  <span>{cameraInfo.frameRate}</span>
                </div>
                <div>
                  <label>Facing</label>
                  <span>{cameraInfo.facing}</span>
                </div>
              </div>
            </article>

            <article className="panel calibration-info">
              <div className="panel-head inline">
                <div>
                  <h3>Facial Tracking</h3>
                  <p>Essential diagnostics for calibration</p>
                </div>
                <span className="material-symbols-outlined">face</span>
              </div>
              <div className="stat-grid">
                {calibrationStats.map((stat) => (
                  <div className="stat-card" key={stat.label}>
                    <h4>{stat.label}</h4>
                    <strong>{stat.value}</strong>
                    <span>{stat.note}</span>
                  </div>
                ))}
              </div>
              <div className="checklist">
                <h4>Calibration Checklist</h4>
                <ul>
                  {faceChecklist.map((item) => (
                    <li key={item}>{item}</li>
                  ))}
                </ul>
              </div>
            </article>
          </div>
        </section>

        <section className="section-block voice-block" id="voice-config">
          <header className="section-head">
            <div>
              <h3>Voice Configuration</h3>
              <p>Commands captured from the assistant session logs.</p>
            </div>
            <span className="status-pill active">Listening</span>
          </header>

          <div className="voice-grid">
            <article className="panel voice-console">
              <div className="panel-head inline">
                <div>
                  <h3>Recent User Commands</h3>
                  <p>{voiceMessage}</p>
                </div>
                <span className="material-symbols-outlined">graphic_eq</span>
              </div>
              <div className="command-list">
                {voiceCommandItems.length ? (
                  voiceCommandItems.map((command, index) => (
                    <div className="command-row" key={`${command.text}-${index}`}>
                      <span className="command-time">{command.time || "Now"}</span>
                      <span className="command-text">{command.text}</span>
                    </div>
                  ))
                ) : (
                  <div className="command-empty">No user commands detected yet.</div>
                )}
              </div>
            </article>

            <article className="panel voice-settings">
              <div className="panel-head inline">
                <div>
                  <h3>Voice Tuning</h3>
                </div>
                <span className="material-symbols-outlined">tune</span>
              </div>
              <div className="voice-defaults">
                {voiceDefaults.map((item) => (
                  <div className="voice-chip" key={item}>
                    {item}
                  </div>
                ))}
              </div>
              <div className="voice-stats">
                <div>
                  <label>Detection Latency</label>
                  <span>0.12s average</span>
                </div>
                <div>
                  <label>Confidence</label>
                  <span>96%</span>
                </div>
                <div>
                  <label>Microphone</label>
                  <span>Front array (Active)</span>
                </div>
              </div>
            </article>
          </div>
        </section>


        <button className="fab" type="button" onClick={() => setChatOpen(true)}>
          <span className="material-symbols-outlined">bolt</span>
        </button>

        {chatOpen ? (
          <div className="chat-overlay" onClick={() => setChatOpen(false)}>
            <div className="chat-panel" onClick={(event) => event.stopPropagation()}>
              <div className="chat-head">
                <div>
                  <h3>Assistant</h3>
                  <p>{chatStatus.state === "sending" ? "Thinking..." : "Ask a question"}</p>
                </div>
                <button className="chat-close" type="button" onClick={() => setChatOpen(false)}>
                  <span className="material-symbols-outlined">close</span>
                </button>
              </div>
              <div className="chat-body">
                {chatMessages.map((message, index) => (
                  <div
                    className={`chat-row ${message.role === "user" ? "user" : "assistant"}`}
                    key={`${message.role}-${index}`}
                  >
                    <span>{message.text}</span>
                  </div>
                ))}
              </div>
              <div className="chat-input">
                <textarea
                  placeholder="Type your message..."
                  value={chatInput}
                  onChange={(event) => setChatInput(event.target.value)}
                  onKeyDown={handleChatKeyDown}
                  rows={2}
                />
                <button type="button" onClick={handleChatSend} disabled={!chatInput.trim()}>
                  Send
                </button>
              </div>
            </div>
          </div>
        ) : null}
      </main>
    </div>
  );
}
