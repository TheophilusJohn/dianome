#!/usr/bin/env node
// Phase 7 A2: remote split measurements. Drives the deployed split demo in Google Chrome (Playwright, channel "chrome")
// for one model: first with policy prefer: "cost" (the planner's N on this laptop), then with prefer: "server" (N = 0),
// --runs each, and appends every run to packages/sdk/results/remote-l4.json keyed by model. Client (this laptop) and
// server (the L4 pod behind gpu.dianome.dev) are different machines, which retires the shared-GPU confound of Phase 5b.
//
//   node scripts/measure-remote.mjs --model qwen2.5-7b-instruct [--runs 3] [--max-tokens 32] [--prompt "…"]
//       [--url https://dianome-demo-split.pages.dev] [--plan-url https://gpu.dianome.dev/plan] [--query "api=…&cdn=…"]
//       [--out packages/sdk/results/remote-l4.json] [--headed]
//
// Before measuring it fetches --plan-url and exits 2 unless that server reports the expected model (the pod serves one
// model at a time: stop/start serve.sh with --model between 7B and 3B). Each run records mode, N, tok/s, the per-step
// breakdown medians (client / export / network / server busy / lm_head / sampling), RTT, server busy total, load
// bytes/time and the cache mode, read from the demo's window.__dianome hook (apps/split-demo).
import { createRequire } from "node:module";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";

const repo = resolve(dirname(new URL(import.meta.url).pathname), "..");
const { chromium } = createRequire(join(repo, "apps/site/package.json"))("@playwright/test");

const args = Object.fromEntries(process.argv.slice(2).map((a, i, all) => (a.startsWith("--") ? [a.slice(2), all[i + 1]?.startsWith("--") || all[i + 1] === undefined ? "true" : all[i + 1]] : null)).filter(Boolean));
const MODEL = args.model;
if (!MODEL) { console.error("usage: node scripts/measure-remote.mjs --model <id> [--runs 3] [--max-tokens 32] [--url …] [--plan-url …] [--query …] [--out …] [--headed]"); process.exit(2); }
const RUNS = Number(args.runs ?? 3);
const MAX_TOKENS = Number(args["max-tokens"] ?? 32);
const URL_BASE = (args.url ?? "https://dianome-demo-split.pages.dev").replace(/\/+$/, "");
const PLAN_URL = args["plan-url"] ?? "https://gpu.dianome.dev/plan";
const OUT = resolve(repo, args.out ?? "packages/sdk/results/remote-l4.json");
const PROMPT = args.prompt ?? "The quick brown fox jumps over the lazy dog while the sun sets slowly behind the distant mountains, casting long shadows across the quiet valley below, where a small river";
const POLICIES = [["cost", "planner's N under prefer: cost"], ["server", "N = 0"]];

// 1. the server behind the tunnel must be serving this model
let plan;
try {
  const r = await fetch(PLAN_URL, { headers: { Accept: "application/json" }, signal: AbortSignal.timeout(10_000) });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  plan = await r.json();
} catch (e) {
  console.error(`${PLAN_URL}: ${e.message ?? e}; is the pod's serve.sh up and the tunnel routed?`);
  process.exit(2);
}
if (plan.model !== MODEL) {
  console.error(`${PLAN_URL} reports model ${JSON.stringify(plan.model)} (L=${plan.L}), expected ${MODEL}: swap the pod's server first (scripts/pod/serve.sh stop && serve.sh start --model ${MODEL})`);
  process.exit(2);
}
console.log(`${PLAN_URL}: ${plan.model} L=${plan.L} device=${plan.device ?? "?"} busy_fraction_60s=${plan.busy_fraction_60s} ms/block decode ${plan.ms_per_block_decode} lm_head ${plan.lm_head_ms} ms`);

// 2. drive the demo
const pageUrl = `${URL_BASE}/?model=${encodeURIComponent(MODEL)}${args.query ? `&${args.query}` : ""}`;
const browser = await chromium.launch({ channel: "chrome", headless: args.headed !== "true", args: ["--enable-unsafe-webgpu"] });
const context = await browser.newContext();
const page = await context.newPage();
const pageErrors = [];
page.on("pageerror", (e) => pageErrors.push(e.message));
page.on("console", (m) => { if (m.type() === "error") pageErrors.push(m.text()); });
const hook = () => page.evaluate(() => window.__dianome);
const waitReady = async () => { await page.waitForFunction(() => window.__dianome?.ready === true, null, { timeout: 180_000 }); const h = await hook(); if (h.error) throw new Error(`plan: ${h.error}`); return h.plan; };
const runs = { cost: [], server: [] };
const started = new Date().toISOString();
try {
  console.log(`opening ${pageUrl}`);
  await page.goto(pageUrl, { waitUntil: "domcontentloaded" });
  try { await page.waitForFunction(() => Boolean(window.__dianome), null, { timeout: 30_000 }); }
  catch { throw new Error(`${pageUrl} exposes no window.__dianome hook after 30 s: the deployed demo predates Phase 7 A2 (redeploy apps/split-demo)`); }
  const first = await waitReady();
  const h0 = await hook();
  if (h0.model !== MODEL) throw new Error(`the demo is running ${h0.model}, not ${MODEL} (no ?model= support deployed yet?)`);
  await page.fill("#prompt", PROMPT);
  await page.fill("#maxTokens", String(MAX_TOKENS));
  await page.fill("#temp", "0");
  console.log(`planner inputs: webgpu ${first.inputs.device.webgpu}, budget ${(first.inputs.device.gpuBudgetBytes / 2 ** 20).toFixed(0)} MiB, ms/block T=1 ${first.inputs.device.msPerBlockT1}, RTT ${first.inputs.network.rttMs} ms, bandwidth ${first.inputs.network.bytesPerSecond ? (first.inputs.network.bytesPerSecond / 1e6).toFixed(0) + " MB/s" : "?"}, server reachable ${first.inputs.network.serverReachable}`);
  for (const [policy, label] of POLICIES) {
    for (let i = 0; i < RUNS; i++) {
      await page.evaluate(() => { window.__dianome.ready = false; });
      await page.selectOption("#auto", policy);        // triggers refreshInputs → a fresh plan under that policy
      const p = await waitReady();
      console.log(`  ${policy} run ${i + 1}/${RUNS}: plan ${p.mode} N=${p.N} (${p.reasons.join("; ")}) est ${p.estimate.msPerToken?.toFixed(1)} ms/token`);
      await page.evaluate(() => { window.__dianome.last = null; });
      await page.click("#run");
      await page.waitForFunction(() => window.__dianome.last !== null, null, { timeout: 900_000 });
      const last = (await hook()).last;
      if (last.error) { console.log(`    error: ${last.error}`); runs[policy].push({ policy, label, run: i + 1, error: last.error, at: last.at }); continue; }
      runs[policy].push({ policy, label, run: i + 1, ...last });
      const b = last.breakdown;
      console.log(`    ${last.mode} N=${last.N}: ${last.tokPerS.toFixed(1)} tok/s, step ${last.measuredMsPerToken.toFixed(1)} ms (est ${last.estimatedMsPerToken?.toFixed(1)}), client ${b.client.toFixed(1)} export ${b.export.toFixed(1)} network ${b.network.toFixed(1)} server ${b.server.toFixed(1)} lm_head ${b.lm_head.toFixed(1)} ms; rtt ${last.rttMs?.toFixed(1)} ms; server busy ${last.serverBusyMs.toFixed(0)} ms; load ${last.load ? `${(last.load.bytes / 2 ** 20).toFixed(0)} MiB in ${(last.load.ms / 1000).toFixed(1)} s` : "runtime reused"}; cache ${last.cacheMode}; telemetry ${last.telemetryStatus}`);
    }
  }
} catch (e) {
  console.error(`measure-remote: ${e.message ?? e}`);
  await browser.close();
  process.exit(2);
} finally {
  await browser.close();
}

// 3. append to the results file, keyed by model
const all = existsSync(OUT) ? JSON.parse(readFileSync(OUT, "utf8")) : {};
const entry = all[MODEL] ?? { runs: [] };
entry.plan_check = { url: PLAN_URL, ...plan, checked: started };
entry.page = pageUrl;
entry.client = { browser: "chrome", version: browser.version(), headless: args.headed !== "true", platform: process.platform, arch: process.arch };
entry.runs.push(...runs.cost, ...runs.server);
entry.pageErrors = pageErrors;
entry.updated = new Date().toISOString();
all[MODEL] = entry;
mkdirSync(dirname(OUT), { recursive: true });
writeFileSync(OUT, JSON.stringify(all, null, 2) + "\n");

// 4. summary
const median = (a) => { const s = [...a].sort((x, y) => x - y); return s.length ? s[Math.floor(s.length / 2)] : NaN; };
const f = (n, d = 1) => (Number.isFinite(n) ? n.toFixed(d) : "–");
console.log(`\n${MODEL} (L=${plan.L}) via ${PLAN_URL}, ${RUNS} runs per policy, medians over runs:`);
console.log("| policy | mode | N | tok/s | step ms | est ms | client | export | network | server busy | lm_head | RTT ms | busy total ms | load | cache |");
console.log("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|");
for (const [policy] of POLICIES) {
  const ok = runs[policy].filter((r) => !r.error);
  if (!ok.length) { console.log(`| ${policy} | error | | | | | | | | | | | | | |`); continue; }
  const m = (k) => median(ok.map((r) => r[k]));
  const mb = (k) => median(ok.map((r) => r.breakdown[k]));
  const loads = ok.filter((r) => r.load);
  console.log(`| ${policy} | ${ok[0].mode} | ${ok[0].N} | ${f(m("tokPerS"))} | ${f(m("measuredMsPerToken"))} | ${f(m("estimatedMsPerToken"))} | ${f(mb("client"))} | ${f(mb("export"))} | ${f(mb("network"))} | ${f(mb("server"))} | ${f(mb("lm_head"))} | ${f(m("rttMs"))} | ${f(m("serverBusyMs"), 0)} | ${loads.length ? `${f(median(loads.map((r) => r.load.bytes)) / 2 ** 20, 0)} MiB in ${f(median(loads.map((r) => r.load.ms)) / 1000)} s` : "reused"} | ${ok[0].cacheMode} |`);
}
console.log(`wrote ${OUT} (${entry.runs.length} runs for ${MODEL}${pageErrors.length ? `; ${pageErrors.length} page errors recorded` : ""})`);
