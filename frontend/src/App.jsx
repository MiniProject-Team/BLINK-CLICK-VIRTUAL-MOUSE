import { startTransition, useEffect, useState } from "react";

const highlights = [
  {
    title: "Hands-Free Control",
    text: "Move through your computer with head movement and blink actions designed for everyday accessibility support.",
  },
  {
    title: "Voice Guided",
    text: "Use natural voice requests after the wake word to trigger safe actions without needing a keyboard or mouse.",
  },
  {
    title: "Built for Simplicity",
    text: "The experience stays focused on what matters: launching the assistant, understanding what it does, and getting started quickly.",
  },
];

const steps = [
  "Connect your camera and microphone.",
  "Start the assistant from this page.",
  "Say the wake word, then speak your request naturally.",
  "Use the on-screen window to monitor voice and control status.",
];

async function readJson(response) {
  try {
    return await response.json();
  } catch {
    return {};
  }
}

export default function App() {
  const [isLaunching, setIsLaunching] = useState(false);
  const [launcherStatus, setLauncherStatus] = useState({
    available: false,
    running: false,
    message: "Checking launcher...",
  });

  useEffect(() => {
    let active = true;

    async function loadStatus() {
      try {
        const response = await fetch("/api/status");
        const data = await readJson(response);
        if (!active) {
          return;
        }
        startTransition(() => {
          setLauncherStatus({
            available: true,
            running: Boolean(data.running),
            message: data.message || "Launcher ready.",
          });
        });
      } catch {
        if (!active) {
          return;
        }
        startTransition(() => {
          setLauncherStatus({
            available: false,
            running: false,
            message:
              "Launcher offline. Run the local launcher server to use the Start button.",
          });
        });
      }
    }

    loadStatus();
    const timer = window.setInterval(loadStatus, 5000);

    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, []);

  async function handleStart() {
    setIsLaunching(true);
    try {
      const response = await fetch("/api/start", { method: "POST" });
      const data = await readJson(response);
      startTransition(() => {
        setLauncherStatus({
          available: true,
          running: Boolean(data.running),
          message: data.message || "Project started.",
        });
      });
    } catch {
      startTransition(() => {
        setLauncherStatus({
          available: false,
          running: false,
          message:
            "Start request failed. Please run the local launcher server, then try again.",
        });
      });
    } finally {
      setIsLaunching(false);
    }
  }

  return (
    <div className="page-shell">
      <div className="ambient ambient-left" />
      <div className="ambient ambient-right" />

      <main className="app-frame">
        <section className="hero-panel">
          <div className="hero-copy">
            <span className="eyebrow">Accessibility-First Desktop Assistant</span>
            <h1>Blink Click Virtual Mouse</h1>
            <p className="hero-text">
              A local accessibility experience that helps people interact with a
              computer through head movement, blink actions, and voice guidance.
            </p>

            <div className="hero-actions">
              <button
                className="start-button"
                onClick={handleStart}
                disabled={isLaunching}
                type="button"
              >
                <span className="start-glow" aria-hidden="true" />
                <span className="start-label">
                  {isLaunching ? "Starting..." : "Start Project"}
                </span>
                <span className="start-sub">Launch assistant</span>
              </button>
              <div className="status-pill" data-running={launcherStatus.running}>
                {launcherStatus.running ? "Running" : "Ready"}
              </div>
            </div>

            <p className="status-text">{launcherStatus.message}</p>
          </div>

          <div className="hero-card">
            <div className="signal-card">
              <span className="signal-label">Voice Flow</span>
              <strong>Wake word + safe actions</strong>
            </div>
            <div className="signal-card">
              <span className="signal-label">Control Style</span>
              <strong>Head movement and blink support</strong>
            </div>
            <div className="signal-card">
              <span className="signal-label">Project Goal</span>
              <strong>Practical hands-free computer access</strong>
            </div>
          </div>
        </section>

        <section className="content-grid">
          <div className="panel">
            <h2>What It Offers</h2>
            <div className="highlight-grid">
              {highlights.map((item) => (
                <article className="highlight-card" key={item.title}>
                  <h3>{item.title}</h3>
                  <p>{item.text}</p>
                </article>
              ))}
            </div>
          </div>

          <div className="panel">
            <h2>Getting Started</h2>
            <div className="step-list">
              {steps.map((step, index) => (
                <div className="step-row" key={step}>
                  <span className="step-index">{index + 1}</span>
                  <p>{step}</p>
                </div>
              ))}
            </div>
          </div>
        </section>

        <section className="info-banner">
          <div>
            <span className="eyebrow">Project Overview</span>
            <h2>Designed to be supportive, direct, and easy to launch</h2>
          </div>
          <p>
            This page shares the public-facing idea of the project without exposing
            internal implementation details. It is focused on purpose, usability,
            and quick access.
          </p>
        </section>
      </main>
    </div>
  );
}
