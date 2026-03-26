import { startTransition, useEffect, useRef, useState } from "react";
import BlurText from "./BlurText";
import "./blink-click.css";

const highlights = [
  { cat: "Voice flow", val: "Wake word + safe actions" },
  { cat: "Control style", val: "Head movement and blink support" },
  { cat: "Project goal", val: "Practical hands-free computer access" },
];

const offerings = [
  {
    title: "Hands-free control",
    text: "Move through your computer with head movement and blink actions designed for everyday accessibility support.",
  },
  {
    title: "Voice guided",
    text: "Speak commands naturally after the wake word. The assistant handles the rest with safe, reversible actions.",
  },
  {
    title: "Local and private",
    text: "Everything runs on your machine. No cloud services and no internet required during use.",
  },
];

const steps = [
  "Connect your camera and microphone.",
  "Start the assistant from this page.",
  "Say the wake word, then speak your request naturally.",
  "Use the on-screen window to monitor voice and control status.",
];

const overviewBlocks = [
  {
    title: "Why this project matters",
    text: "Traditional mouse and keyboard interaction requires physical movement, which can be difficult for users with mobility limitations. This project provides a contactless alternative focused on accessibility, inclusion, and everyday usability.",
  },
  {
    title: "Voice recognition module",
    text: "A microphone captures speech, then SpeechRecognition converts audio into text. The recognized text is matched with predefined commands to trigger actions such as start, stop, and additional system controls.",
  },
  {
    title: "Computer vision pipeline",
    text: "After activation, a webcam captures real-time video. OpenCV processes frames, detects the face region, and prepares a stable input stream for facial analysis.",
  },
  {
    title: "Facial landmark detection",
    text: "MediaPipe identifies 468 facial landmarks in each frame. Eye-related landmark points are extracted to measure gaze position and movement with better precision.",
  },
  {
    title: "Cursor movement by eye tracking",
    text: "Eye landmark coordinates are continuously mapped from camera space to screen space. As the user moves their eyes, the cursor follows in real time to create intuitive, hands-free pointer control.",
  },
  {
    title: "Blink-to-click logic (EAR)",
    text: "The Eye Aspect Ratio (EAR) is computed from selected eye points. A sustained drop below a threshold indicates an intentional blink, and the system triggers a click event through PyAutoGUI while reducing false detections.",
  },
  {
    title: "Multimodal interaction",
    text: "The system combines eye movement for cursor navigation, blinking for click actions, and voice commands for higher-level tasks. This combination gives users flexible control paths based on comfort and context.",
  },
  {
    title: "Real-time and practical deployment",
    text: "Continuous frame processing and speech analysis provide quick feedback. Built with Python and common open-source libraries, the solution remains cost-effective, deployable, and suitable for assistive-tech and smart-interface scenarios.",
  },
  {
    title: "Primary use cases",
    text: "Key applications include assistive technology for users with mobility limitations, healthcare and rehabilitation environments, and smart interfaces where touch-free interaction is preferred for convenience or hygiene.",
  },
  {
    title: "Technology stack",
    text: "The implementation combines Python with OpenCV for image processing, MediaPipe for facial landmarks, SpeechRecognition for voice input, and PyAutoGUI for cursor and click control.",
  },
];

function readJsonSafe(response) {
  return response.json().catch(() => ({}));
}

export default function BlinkClick() {
  const [isStarting, setIsStarting] = useState(false);
  const [status, setStatus] = useState({
    state: "idle",
    message: "",
    running: false,
  });
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      return undefined;
    }

    const ctx = canvas.getContext("2d");
    if (!ctx) {
      return undefined;
    }

    let width = 0;
    let height = 0;
    let time = 0;
    let frameId = 0;

    const waves = Array.from({ length: 7 }, (_, i) => ({
      amp: 50 + Math.random() * 70,
      freq: 0.002 + Math.random() * 0.004,
      speed: 0.006 + Math.random() * 0.008,
      phase: Math.random() * Math.PI * 2,
      y: 0.15 + Math.random() * 0.7,
      hue: 170 + i * 22,
      alpha: 0.05 + Math.random() * 0.08,
    }));

    const stars = Array.from({ length: 150 }, () => ({
      x: Math.random(),
      y: Math.random(),
      r: Math.random() * 1.3 + 0.2,
      twinkle: Math.random() * Math.PI * 2,
      speed: 0.015 + Math.random() * 0.035,
    }));

    function resize() {
      width = canvas.width = window.innerWidth;
      height = canvas.height = window.innerHeight;
    }

    function draw() {
      ctx.clearRect(0, 0, width, height);
      time += 0.016;

      for (const star of stars) {
        star.twinkle += star.speed;
        const alpha = 0.25 + Math.sin(star.twinkle) * 0.25;
        ctx.beginPath();
        ctx.arc(star.x * width, star.y * height, star.r, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(200,240,255,${alpha})`;
        ctx.fill();
      }

      for (const wave of waves) {
        ctx.beginPath();
        const baseY = wave.y * height;
        ctx.moveTo(0, height);

        for (let x = 0; x <= width; x += 3) {
          const y =
            baseY +
            Math.sin(x * wave.freq + time * wave.speed * 60 + wave.phase) * wave.amp +
            Math.sin(x * wave.freq * 2.1 + time * wave.speed * 35) * wave.amp * 0.35 +
            Math.sin(x * wave.freq * 0.5 + time * wave.speed * 20) * wave.amp * 0.2;
          ctx.lineTo(x, y);
        }

        ctx.lineTo(width, height);
        ctx.closePath();

        const gradient = ctx.createLinearGradient(0, baseY - wave.amp * 1.5, 0, baseY + wave.amp);
        gradient.addColorStop(0, `hsla(${wave.hue},85%,72%,${wave.alpha * 2})`);
        gradient.addColorStop(0.5, `hsla(${wave.hue + 30},90%,65%,${wave.alpha})`);
        gradient.addColorStop(1, `hsla(${wave.hue + 60},80%,55%,0)`);
        ctx.fillStyle = gradient;
        ctx.fill();
      }

      frameId = window.requestAnimationFrame(draw);
    }

    resize();
    window.addEventListener("resize", resize);
    draw();

    return () => {
      window.removeEventListener("resize", resize);
      window.cancelAnimationFrame(frameId);
    };
  }, []);

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

  return (
    <div className="bc-root">
      <div className="hex-grid" />
      <canvas id="ac" ref={canvasRef} />

      <div className="page">
        <div className="title-section">
          <div className="badge">// Accessibility-first desktop assistant</div>
          <BlurText
            text="Implementation of Eye-Controlled Virtual Mouse with Blink Click and Voice Recognition"
            className="project-title"
            animateBy="words"
            direction="top"
            delay={120}
            as="h1"
          />
          <p className="desc">
            This project lets users control a computer cursor with face and eye movement, perform click actions with
            blinks, and trigger commands with voice for a practical hands-free experience.
          </p>
          <p className="desc desc-extra">
            It is designed for accessibility-focused use cases where keyboard or mouse usage is difficult. Visitors can
            quickly understand what the system does, how it helps, and how to start using it from this page.
          </p>
        </div>

        <div className="start-section">
          <div className="btn-row">
            <div className="start-wrap">
              {!isStarting && !status.running && <div className="orb-ring" />}
              <button className="start-btn" onClick={handleStart} disabled={isStarting || status.running} type="button">
                {status.running ? "Running" : isStarting ? "Starting..." : "Start Project"}
              </button>
            </div>
            <div className="status-pill">
              <div className={`sdot ${status.running ? "running" : status.state === "error" ? "error" : ""}`.trim()} />
              <span>{status.running ? "Running" : status.state === "starting" ? "Starting..." : status.state === "error" ? "Error" : "Ready"}</span>
              {status.state === "starting" && <div className="spinner-s" />}
            </div>
          </div>
          <div className={`smsg ${status.running ? "running" : status.state === "error" ? "error" : ""}`.trim()}>
            {status.message}
          </div>
          <div className="ptrack">
            <div
              className={`pbar ${status.state === "starting" ? "ind" : ""}`.trim()}
              style={{ width: status.state === "running" ? "100%" : status.state === "starting" ? "38%" : "0%" }}
            />
          </div>
        </div>

        <div className="divider" />

        <div className="info-row">
          {highlights.map((item) => (
            <div className="icard" key={item.cat}>
              <div className="cat">{item.cat}</div>
              <div className="val">{item.val}</div>
            </div>
          ))}
        </div>

        <div className="lower">
          <div className="scard">
            <h2>What it offers</h2>
            {offerings.map((item) => (
              <div className="feat" key={item.title}>
                <h3>{item.title}</h3>
                <p>{item.text}</p>
              </div>
            ))}
          </div>

          <div className="scard">
            <h2>Getting started</h2>
            {steps.map((step, index) => (
              <div className="step" key={step}>
                <div className="snum">{index + 1}</div>
                <p>{step}</p>
              </div>
            ))}
          </div>
        </div>

        <section className="detail-section" aria-label="Detailed project overview">
          <h2>Detailed Project Overview</h2>
          <p className="detail-intro">
            This project is designed as a complete contactless human-computer interaction system that combines
            computer vision and speech recognition to improve accessibility and ease of use.
          </p>
          <div className="detail-grid">
            {overviewBlocks.map((block) => (
              <article className="detail-card" key={block.title}>
                <h3>{block.title}</h3>
                <p>{block.text}</p>
              </article>
            ))}
          </div>
        </section>
      </div>
    </div>
  );
}
