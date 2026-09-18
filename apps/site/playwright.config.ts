import { defineConfig } from "@playwright/test";
import { resolve } from "node:path";

// Phase 7 Part B: /summarize end to end in Google Chrome (WebGPU) against local servers with the 0.5B model:
//   8797  test/e2e/run-server.mjs         the built site + the local chunk store + a proxy of the Worker routes
//   8787  wrangler dev (packages/worker)  session tokens, servers map, rates, plan proxy, telemetry validation, stats
//   8765  dianome-server serve            the split server (same token/key defaults as packages/sdk's split e2e)
const SPLIT_TOKEN = process.env.SPLIT_TOKEN ?? "sdk-e2e-token";
const SIGNING_KEY = process.env.SPLIT_SIGNING_KEY ?? "sdk-e2e-signing-key";
const here = import.meta.dirname;
const serverBin = resolve(here, "../../server/.venv/bin/dianome-server");
const servers = JSON.stringify({ "qwen2.5-0.5b-instruct": { ws: "ws://127.0.0.1:8765/", plan: "http://127.0.0.1:8765/plan" } });
const vars = [`SPLIT_SIGNING_KEY:${SIGNING_KEY}`, `SPLIT_SERVERS:${servers}`, "SPLIT_ALLOWED_ORIGINS:http://127.0.0.1:8797,http://127.0.0.1:8799", "GIT_SHA:e2e"];

export default defineConfig({
  testDir: "test/e2e",
  testMatch: /.*\.spec\.ts/,
  fullyParallel: false,
  workers: 1,
  retries: 0,
  reporter: [["list"], ["json", { outputFile: "test-results/site-e2e.json" }]],
  timeout: 600_000,
  use: { baseURL: "http://127.0.0.1:8797", trace: "retain-on-failure" },
  webServer: [
    { command: "pnpm build && node test/e2e/run-server.mjs", url: "http://127.0.0.1:8797/healthz", reuseExistingServer: false, timeout: 180_000 },
    { command: `npx wrangler dev --ip 127.0.0.1 --port 8787 ${vars.map((v) => `--var '${v}'`).join(" ")}`, url: "http://127.0.0.1:8787/healthz", reuseExistingServer: true, timeout: 120_000, cwd: resolve(here, "../../packages/worker"), stdout: "pipe" },
    { command: `${serverBin} serve --port 8765`, url: "http://127.0.0.1:8765/plan", reuseExistingServer: true, timeout: 180_000, cwd: resolve(here, "../../server"), env: { SPLIT_TOKEN, SPLIT_SIGNING_KEY: SIGNING_KEY }, stdout: "pipe" },
  ],
  projects: [{ name: "chrome", use: { channel: "chrome", launchOptions: { args: ["--enable-unsafe-webgpu"] } } }],
});
