import { spawn } from "node:child_process";
import process from "node:process";

const python = process.env.PYTHON || "python";
const root = new URL("../..", import.meta.url);
const children = [
  spawn(python, ["frontend_server.py"], {
    cwd: root,
    stdio: "inherit",
    shell: process.platform === "win32",
  }),
  spawn("vite", ["--host", "127.0.0.1"], {
    cwd: new URL("..", import.meta.url),
    stdio: "inherit",
    shell: process.platform === "win32",
  }),
];

function stopChildren() {
  for (const child of children) {
    if (!child.killed) child.kill();
  }
}

process.on("SIGINT", () => {
  stopChildren();
  process.exit(0);
});
process.on("SIGTERM", () => {
  stopChildren();
  process.exit(0);
});
children[1].on("exit", (code) => {
  stopChildren();
  process.exit(code ?? 0);
});
