import { defineConfig } from "@playwright/test";
import { resolve } from "node:path";

// Phase 5b: run() end to end in Google Chrome against three local servers:
//   8799  test/e2e-split/run-server.mjs   harness page + bundle, fixtures, the local chunk store with the api/cdn URL
//                                          shapes, and a proxy of /v1/split/* + /v1/telemetry/load to wrangler dev
//   8787  wrangler dev (packages/worker)  the real Worker: session tokens, rates, plan proxy, telemetry validation
//   8765  dianome-server serve            the split server with SPLIT_TOKEN (bearer) and SPLIT_SIGNING_KEY (HMAC)
const SPLIT_TOKEN = process.env.SPLIT_TOKEN ?? "sdk-e2e-token";
const SIGNING_KEY = process.env.SPLIT_SIGNING_KEY ?? "sdk-e2e-signing-key";
const here = import.meta.dirname;
const serverBin = resolve(here, "../../server/.venv/bin/dianome-server");
const servers = JSON.stringify({ "qwen2.5-0.5b-instruct": { ws: "ws://127.0.0.1:8765/", plan: "http://127.0.0.1:8765/plan" } });
const vars = [`SPLIT_SIGNING_KEY:${SIGNING_KEY}`, `SPLIT_SERVERS:${servers}`, "SPLIT_ALLOWED_ORIGINS:http://127.0.0.1:8799", "GIT_SHA:e2e"];

export default defineConfig({
  testDir: "test/e2e-split",
  testMatch: /.*\.spec\.ts/,
  fullyParallel: false,
  workers: 1,
  retries: 0,
  reporter: [["list"], ["json", { outputFile: "test-results/e2e-split.json" }]],
  timeout: 900_000,
  use: { baseURL: "http://127.0.0.1:8799", trace: "retain-on-failure" },
  webServer: [
    { command: "node test/e2e-split/run-server.mjs", url: "http://127.0.0.1:8799/healthz", reuseExistingServer: false, timeout: 60_000 },
    { command: `npx wrangler dev --ip 127.0.0.1 --port 8787 ${vars.map((v) => `--var '${v}'`).join(" ")}`, url: "http://127.0.0.1:8787/healthz", reuseExistingServer: true, timeout: 120_000, cwd: resolve(here, "../worker"), stdout: "pipe" },
    { command: `${serverBin} serve --port 8765`, url: "http://127.0.0.1:8765/plan", reuseExistingServer: true, timeout: 180_000, cwd: resolve(here, "../../server"), env: { SPLIT_TOKEN, SPLIT_SIGNING_KEY: SIGNING_KEY }, stdout: "pipe" },
  ],
  projects: [{ name: "chrome", use: { channel: "chrome", launchOptions: { args: ["--enable-unsafe-webgpu"] } } }],
});
