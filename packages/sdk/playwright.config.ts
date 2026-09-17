import { defineConfig, devices } from "@playwright/test";

// Per-site cache e2e on chromium, firefox and webkit against a local static server (test/e2e/server.ts) that
// serves a synthetic store with the API and CDN URL shapes. The cross-site path is verified manually on the
// deployed demo (docs/phase-3-notes.md).
export default defineConfig({
  testDir: "test/e2e",
  testMatch: /.*\.spec\.ts/,
  fullyParallel: false,
  workers: 1,
  retries: 0,
  reporter: [["list"], ["json", { outputFile: "test-results/e2e.json" }]],
  timeout: 120_000,
  use: { baseURL: "http://127.0.0.1:8797", trace: "retain-on-failure" },
  webServer: {
    command: "node test/e2e/run-server.mjs",
    url: "http://127.0.0.1:8797/healthz",
    reuseExistingServer: false,
    timeout: 60_000,
  },
  projects: [
    { name: "chromium", use: { ...devices["Desktop Chrome"] } },
    { name: "firefox", use: { ...devices["Desktop Firefox"] } },
    { name: "webkit", use: { ...devices["Desktop Safari"] } },
  ],
});
