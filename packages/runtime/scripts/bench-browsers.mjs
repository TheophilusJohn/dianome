// Runs the built bench page (apps/runtime-bench/dist) in Playwright's Firefox and WebKit against the local
// store and records what happens (WebGPU availability, adapter, load and tok/s or the failure) to
// results/browsers.json. Chrome numbers come from bench.spec.ts; this is only the "once each" observation.
//
//   node scripts/bench-browsers.mjs            (from packages/runtime; needs apps/runtime-bench/dist and ./store)
import { chromium, firefox, webkit } from "@playwright/test";
import { spawn } from "node:child_process";
import { mkdirSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

const repo = resolve(import.meta.dirname, "../../..");
const storePort = 8788, pagePort = 5176;
const procs = [];
const start = (cmd, args, cwd) => { const p = spawn(cmd, args, { cwd, stdio: "ignore" }); procs.push(p); return p; };
const wait = async (url, ms = 30_000) => { const t0 = Date.now(); for (;;) { try { const r = await fetch(url); if (r.ok || r.status === 404) return; } catch {} if (Date.now() - t0 > ms) throw new Error(`timeout waiting for ${url}`); await new Promise((r) => setTimeout(r, 300)); } };

start("node", ["scripts/serve-store.mjs", "--quiet", "--port", String(storePort)], repo);
start("npx", ["vite", "preview", "--port", String(pagePort), "--strictPort"], resolve(repo, "apps/runtime-bench"));
await wait(`http://localhost:${storePort}/v1/models`);
await wait(`http://localhost:${pagePort}/`);

const out = [];
const runs = [
  { name: "firefox", type: firefox, launch: { firefoxUserPrefs: { "dom.webgpu.enabled": true, "gfx.webgpu.ignore-blocklist": true } } },
  { name: "webkit", type: webkit, launch: {} },
  { name: "chrome", type: chromium, launch: { channel: "chrome" } },
];
for (const r of runs) {
  const rec = { browser: r.name, version: null, gpu: null, adapter: null, status: null, log: null, error: null };
  let browser = null;
  try {
    browser = await r.type.launch({ headless: true, ...r.launch });
    rec.version = browser.version();
    const page = await browser.newPage();
    page.on("pageerror", (e) => { rec.error = (rec.error ?? "") + e.message + "\n"; });
    await page.goto(`http://localhost:${pagePort}/?api=http://localhost:${storePort}&cdn=http://localhost:${storePort}`);
    rec.gpu = await page.evaluate(async () => {
      if (!navigator.gpu) return { available: false };
      const a = await navigator.gpu.requestAdapter();
      if (!a) return { available: true, adapter: null };
      const i = a.info ?? {};
      return { available: true, adapter: { vendor: i.vendor, architecture: i.architecture, description: i.description }, features: [...a.features], maxBufferSize: a.limits.maxBufferSize, maxWorkgroupStorage: a.limits.maxComputeWorkgroupStorageSize, maxStorageBinding: a.limits.maxStorageBufferBindingSize };
    });
    await page.selectOption("select[name=variant]", "q4");
    await page.fill("input[name=N]", "24");
    await page.fill("input[name=steps]", "32");
    await page.click("#load");
    await page.waitForFunction(() => /^(loaded|error)/.test(document.getElementById("status").textContent), null, { timeout: 300_000 });
    rec.status = await page.textContent("#status");
    if (rec.status.startsWith("loaded")) {
      await page.click("#run");
      await page.waitForFunction(() => /^(done|error)/.test(document.getElementById("status").textContent), null, { timeout: 300_000 });
      rec.status = await page.textContent("#status");
    }
    rec.log = await page.textContent("#log");
    rec.table = await page.$$eval("#out tr", (rows) => rows.map((tr) => [...tr.children].map((c) => c.textContent)));
  } catch (e) {
    rec.error = (rec.error ?? "") + String(e.message ?? e);
  } finally {
    await browser?.close();
  }
  console.log(JSON.stringify(rec, null, 1));
  out.push(rec);
}
mkdirSync("results", { recursive: true });
writeFileSync("results/browsers.json", JSON.stringify(out, null, 2));
for (const p of procs) p.kill();
process.exit(0);
