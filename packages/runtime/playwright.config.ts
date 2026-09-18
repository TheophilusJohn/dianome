import { defineConfig } from "@playwright/test";
import { resolve } from "node:path";

// Gates 2–7 in Google Chrome (Playwright's bundled Chromium has no WebGPU on macOS; `channel: "chrome"`).
// Two servers: the static harness (fixtures + local chunk store) and `dianome-server serve` for gates 6/7.
const SPLIT_TOKEN = process.env.SPLIT_TOKEN ?? "runtime-e2e-token";
const here = import.meta.dirname;
const serverBin = resolve(here, "../../server/.venv/bin/dianome-server");

export default defineConfig({
  testDir: "test/e2e",
  testMatch: /.*\.spec\.ts/,
  fullyParallel: false,
  workers: 1,
  retries: 0,
  reporter: [["list"], ["json", { outputFile: "test-results/e2e.json" }]],
  timeout: 600_000,
  use: { baseURL: "http://127.0.0.1:8798", trace: "retain-on-failure" },
  webServer: [
    { command: "node test/e2e/run-server.mjs", url: "http://127.0.0.1:8798/healthz", reuseExistingServer: false, timeout: 60_000 },
    {
      command: `${serverBin} serve --port 8765`,
      url: `http://127.0.0.1:8765/plan?token=${SPLIT_TOKEN}`,
      reuseExistingServer: true,
      timeout: 180_000,
      cwd: resolve(here, "../../server"),
      env: { SPLIT_TOKEN },
      stdout: "pipe",
    },
  ],
  projects: [{ name: "chrome", use: { channel: "chrome", launchOptions: { args: ["--enable-unsafe-webgpu"] } } }],
});
