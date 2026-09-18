// Planner inputs on other browsers (Phase 5b notes): loads the split e2e harness in Playwright's WebKit and Firefox
// builds (and Chrome for comparison), asks for a plan (prefer cost, q4) and records the measured inputs and the
// choice. Needs the three servers of playwright.split.config.ts already running (the static server from
// `node test/e2e-split/run-server.mjs`, wrangler dev, dianome-server) — or run right after `pnpm test:e2e:split`
// with reuseExistingServer. Writes results/split-browsers.json.
//
//   node scripts/split-browsers.mjs        (from packages/sdk)
import { chromium, firefox, webkit } from "@playwright/test";
import { mkdirSync, writeFileSync } from "node:fs";

const SPLIT_TOKEN = process.env.SPLIT_TOKEN ?? "sdk-e2e-token";
const out = {};
for (const [name, type, opts] of [["webkit", webkit, {}], ["firefox", firefox, {}], ["chrome", chromium, { channel: "chrome", args: ["--enable-unsafe-webgpu"] }]]) {
  const browser = await type.launch(opts);
  const page = await browser.newPage();
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  const row = { version: browser.version(), errors };
  try {
    await page.goto("http://127.0.0.1:8799/");
    await page.waitForFunction(() => Boolean(window.harness), null, { timeout: 30_000 });
    row.info = await page.evaluate(() => window.harness.info());
    const t0 = Date.now();
    const plan = await page.evaluate(async (token) => {
      const p = await window.harness.plan({ prompt: "plan", variant: "q4", maxTokens: 32, policy: { prefer: "cost" }, split: { url: "ws://127.0.0.1:8765/", token }, microbench: "fresh" });
      return { mode: p.mode, N: p.N, reasons: p.reasons, estimate: p.estimate.msPerToken, inputs: p.inputs, candidates: p.candidates.map((c) => ({ mode: c.mode, N: c.N, feasible: c.feasible, msPerToken: c.msPerToken, reasons: c.reasons })) };
    }, SPLIT_TOKEN);
    row.planMs = Date.now() - t0;
    row.plan = plan;
    console.log(`${name} ${row.version}: webgpu ${row.info.webgpu} → ${plan.mode} N=${plan.N} est ${plan.estimate?.toFixed(1)} ms/token; device ${JSON.stringify(plan.inputs.device)} network ${JSON.stringify(plan.inputs.network)} (${row.planMs} ms)`);
  } catch (e) {
    row.error = String(e.message ?? e);
    console.log(`${name}: ${row.error}`);
  }
  out[name] = row;
  await browser.close();
}
mkdirSync("results", { recursive: true });
writeFileSync("results/split-browsers.json", JSON.stringify({ ...out, updated: new Date().toISOString() }, null, 2));
