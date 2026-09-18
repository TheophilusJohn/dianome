#!/usr/bin/env node
// Phase 7 A2: remote split measurements. Drives the deployed split demo in Google Chrome (Playwright, channel "chrome")
// for one model: first with policy prefer: "cost" (the planner's N on this laptop), then with prefer: "server" (N = 0),
// --runs each, and appends every run to packages/sdk/results/remote-l4.json keyed by model. Client (this laptop) and
// server (the L4 pod behind gpu.dianome.dev) are different machines, which retires the shared-GPU confound of Phase 5b.
//
//   node scripts/measure-remote.mjs --model qwen2.5-7b-instruct [--runs 3] [--max-tokens 32] [--prompt "…"]
//       [--url https://dianome-demo-split.pages.dev] [--plan-url https://gpu.dianome.dev/plan] [--query "api=…&cdn=…"]
//       [--out packages/sdk/results/remote-l4.json] [--headed] [--allow-no-webgpu]
//
// Chrome is launched exactly as packages/runtime/playwright.config.ts launches it for the 5a gates: channel "chrome" with
// --enable-unsafe-webgpu, headless. Before anything is measured, navigator.gpu.requestAdapter() on the demo page must
// return an adapter and the planner's first plan must report webgpu true (the SDK's microbench ran); otherwise the
// script exits 2 and says which (--headed is the fallback when headless Chrome has no adapter; --allow-no-webgpu
// records server-only runs on purpose, labelled webgpu: false).
//
// Before measuring it fetches --plan-url and exits 2 unless that server reports the expected model (the pod serves one
// model at a time: stop/start serve.sh with --model between 7B and 3B). Each run records mode, N, tok/s, the per-step
// breakdown medians (client / export / network / server busy / lm_head / sampling), RTT, server busy total, load
// bytes/time, the cache mode and the planner inputs, read from the demo's window.__dianome hook (apps/split-demo).
import { createRequire } from "node:module";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";

const repo = resolve(dirname(new URL(import.meta.url).pathname), "..");
const { chromium } = createRequire(join(repo, "apps/site/package.json"))("@playwright/test");

const args = Object.fromEntries(process.argv.slice(2).map((a, i, all) => (a.startsWith("--") ? [a.slice(2), all[i + 1]?.startsWith("--") || all[i + 1] === undefined ? "true" : all[i + 1]] : null)).filter(Boolean));
const MODEL = args.model;
if (!MODEL) { console.error("usage: node scripts/measure-remote.mjs --model <id> [--runs 3] [--max-tokens 32] [--url …] [--plan-url …] [--query …] [--out …] [--headed] [--allow-no-webgpu]"); process.exit(2); }
const RUNS = Number(args.runs ?? 3);
const MAX_TOKENS = Number(args["max-tokens"] ?? 32);
const URL_BASE = (args.url ?? "https://dianome-demo-split.pages.dev").replace(/\/+$/, "");
const PLAN_URL = args["plan-url"] ?? "https://gpu.dianome.dev/plan";
const OUT = resolve(repo, args.out ?? "packages/sdk/results/remote-l4.json");
const PROMPT = args.prompt ?? "The quick brown fox jumps over the lazy dog while the sun sets slowly behind the distant mountains, casting long shadows across the quiet valley below, where a small river";
const POLICIES = [["cost", "planner's N under prefer: cost"], ["server", "N = 0"]];
const ALLOW_NO_WEBGPU = args["allow-no-webgpu"] === "true";
// launch = packages/runtime/playwright.config.ts (channel "chrome", args ["--enable-unsafe-webgpu"]); --headed opens a window
const LAUNCH = { channel: "chrome", headless: args.headed !== "true", args: ["--enable-unsafe-webgpu"] };

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
const browser = await chromium.launch(LAUNCH);
const context = await browser.newContext();
const page = await context.newPage();
const pageLog = [];
page.on("pageerror", (e) => pageLog.push(`[pageerror] ${e.message}`));
page.on("console", (m) => { if (m.type() === "error" || m.type() === "warning") pageLog.push(`[${m.type()}] ${m.text()}`); });
const hook = () => page.evaluate(() => window.__dianome);
const waitReady = async () => { await page.waitForFunction(() => window.__dianome?.ready === true, null, { timeout: 180_000 }); const h = await hook(); if (h.error) throw new Error(`plan: ${h.error}`); return h.plan; };
const runs = { cost: [], server: [] };
const started = new Date().toISOString();
let adapter = null, webgpu = false;

class WebGpuError extends Error {}

async function measure() {
  for (const [policy, label] of POLICIES) {
    for (let i = 0; i < RUNS; i++) {
      await page.evaluate(() => { window.__dianome.ready = false; });
      await page.selectOption("#auto", policy);        // triggers refreshInputs → a fresh plan under that policy
      const p = await waitReady();
      const inputs = { device: p.inputs.device, network: p.inputs.network, server: p.inputs.server };
      console.log(`  ${policy} run ${i + 1}/${RUNS}: plan ${p.mode} N=${p.N} (${p.reasons.join("; ")}) est ${p.estimate.msPerToken?.toFixed(1)} ms/token`);
      await page.evaluate(() => { window.__dianome.last = null; });
      await page.click("#run");
      await page.waitForFunction(() => window.__dianome.last !== null, null, { timeout: 900_000 });
      const last = (await hook()).last;
      if (last.error) { console.log(`    error: ${last.error}`); runs[policy].push({ policy, label, run: i + 1, webgpu, adapter, plannerInputs: inputs, error: last.error, at: last.at }); continue; }
      runs[policy].push({ policy, label, run: i + 1, webgpu, adapter, plannerInputs: inputs, ...last });
      const b = last.breakdown;
      console.log(`    ${last.mode} N=${last.N}: ${last.tokPerS.toFixed(1)} tok/s, step ${last.measuredMsPerToken.toFixed(1)} ms (est ${last.estimatedMsPerToken?.toFixed(1)}), client ${b.client.toFixed(1)} export ${b.export.toFixed(1)} network ${b.network.toFixed(1)} server ${b.server.toFixed(1)} lm_head ${b.lm_head.toFixed(1)} ms; rtt ${last.rttMs?.toFixed(1)} ms; server busy ${last.serverBusyMs.toFixed(0)} ms; load ${last.load ? `${(last.load.bytes / 2 ** 20).toFixed(0)} MiB in ${(last.load.ms / 1000).toFixed(1)} s` : "runtime reused"}; cache ${last.cacheMode}; telemetry ${last.telemetryStatus}`);
    }
  }
}

try {
  console.log(`opening ${pageUrl} (${LAUNCH.headless ? "headless" : "headed"} Chrome, args ${LAUNCH.args.join(" ")})`);
  await page.goto(pageUrl, { waitUntil: "domcontentloaded" });
  try { await page.waitForFunction(() => Boolean(window.__dianome), null, { timeout: 30_000 }); }
  catch { throw new Error(`${pageUrl} exposes no window.__dianome hook after 30 s: the deployed demo predates Phase 7 A2 (redeploy apps/split-demo)`); }
  // WebGPU must be there before anything is measured: the adapter, then the planner's own flag (the microbench ran)
  adapter = await page.evaluate(async () => {
    const a = await navigator.gpu?.requestAdapter?.();
    return a ? { vendor: a.info?.vendor ?? null, architecture: a.info?.architecture ?? null, maxBufferSize: a.limits.maxBufferSize, maxStorageBufferBindingSize: a.limits.maxStorageBufferBindingSize } : null;
  });
  if (!adapter) throw new WebGpuError(`navigator.gpu.requestAdapter() returned null in ${LAUNCH.headless ? "headless" : "headed"} Chrome ${browser.version()}: nothing but server N=0 could be measured. Retry with --headed; --allow-no-webgpu records server-only runs anyway.`);
  console.log(`WebGPU adapter: ${adapter.vendor ?? "?"} / ${adapter.architecture ?? "?"}, maxBufferSize ${(adapter.maxBufferSize / 2 ** 20).toFixed(0)} MiB`);
  const first = await waitReady();
  const h0 = await hook();
  if (h0.model !== MODEL) throw new Error(`the demo is running ${h0.model}, not ${MODEL} (no ?model= support deployed yet?)`);
  await page.fill("#prompt", PROMPT);
  await page.fill("#maxTokens", String(MAX_TOKENS));
  await page.fill("#temp", "0");
  const dev = first.inputs.device, net = first.inputs.network;
  console.log(`planner inputs: webgpu ${dev.webgpu}, budget ${(dev.gpuBudgetBytes / 2 ** 20).toFixed(0)} MiB, ms/block T=1 ${dev.msPerBlockT1}, RTT ${net.rttMs} ms, bandwidth ${net.bytesPerSecond ? (net.bytesPerSecond / 1e6).toFixed(1) + " MB/s" : "?"}, server reachable ${net.serverReachable}`);
  webgpu = dev.webgpu === true;
  if (!webgpu) throw new WebGpuError(`the planner reports webgpu false although an adapter exists (ms/block ${dev.msPerBlockT1}): the SDK's microbench failed for this model. Page console: ${pageLog.join(" | ") || "(nothing logged)"}`);
  await measure();
} catch (e) {
  if (e instanceof WebGpuError && ALLOW_NO_WEBGPU) {
    console.warn(`measure-remote: ${e.message}\n  continuing with server-only runs because --allow-no-webgpu was given`);
    webgpu = false;
    try { await measure(); } catch (e2) { console.error(`measure-remote: ${e2.message ?? e2}`); await browser.close(); process.exit(2); }
  } else {
    console.error(`measure-remote: ${e.message ?? e}`);
    await browser.close();
    process.exit(2);
  }
} finally {
  await browser.close();
}

// 3. append to the results file, keyed by model
const all = existsSync(OUT) ? JSON.parse(readFileSync(OUT, "utf8")) : {};
const entry = all[MODEL] ?? { runs: [] };
entry.plan_check = { url: PLAN_URL, ...plan, checked: started };
entry.page = pageUrl;
entry.client = { browser: "chrome", version: browser.version(), headless: LAUNCH.headless, launchArgs: LAUNCH.args, webgpu, adapter, platform: process.platform, arch: process.arch };
entry.runs.push(...runs.cost, ...runs.server);
entry.pageLog = pageLog;
entry.updated = new Date().toISOString();
all[MODEL] = entry;
mkdirSync(dirname(OUT), { recursive: true });
writeFileSync(OUT, JSON.stringify(all, null, 2) + "\n");

// 4. summary
const median = (a) => { const s = [...a].sort((x, y) => x - y); return s.length ? s[Math.floor(s.length / 2)] : NaN; };
const f = (n, d = 1) => (Number.isFinite(n) ? n.toFixed(d) : "–");
const lastInputs = [...runs.cost, ...runs.server].at(-1)?.plannerInputs;
console.log(`\n${MODEL} (L=${plan.L}) via ${PLAN_URL}, ${RUNS} runs per policy, medians over runs; webgpu ${webgpu}${webgpu ? "" : " (server-only runs)"}; bandwidth probe ${lastInputs?.network?.bytesPerSecond ? (lastInputs.network.bytesPerSecond / 1e6).toFixed(1) + " MB/s" : "?"}, RTT ${f(lastInputs?.network?.rttMs)} ms`);
console.log("| policy | mode | N | runs | tok/s | step ms | est ms | client | export | network | server busy | lm_head | RTT ms | busy total ms | load | cache |");
console.log("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|");
// one row per (policy, mode, N): a policy can pick different Ns across runs, and those must not be merged
for (const [policy] of POLICIES) {
  const ok = runs[policy].filter((r) => !r.error);
  if (!ok.length) { console.log(`| ${policy} | error | | | | | | | | | | | | | | |`); continue; }
  const groups = new Map();
  for (const r of ok) { const k = `${r.mode}/${r.N}`; if (!groups.has(k)) groups.set(k, []); groups.get(k).push(r); }
  for (const [, g] of groups) {
    const m = (k) => median(g.map((r) => r[k]));
    const mb = (k) => median(g.map((r) => r.breakdown[k]));
    const loads = g.filter((r) => r.load);
    console.log(`| ${policy} | ${g[0].mode} | ${g[0].N} | ${g.length} | ${f(m("tokPerS"))} | ${f(m("measuredMsPerToken"))} | ${f(m("estimatedMsPerToken"))} | ${f(mb("client"))} | ${f(mb("export"))} | ${f(mb("network"))} | ${f(mb("server"))} | ${f(mb("lm_head"))} | ${f(m("rttMs"))} | ${f(m("serverBusyMs"), 0)} | ${loads.length ? `${f(median(loads.map((r) => r.load.bytes)) / 2 ** 20, 0)} MiB in ${f(median(loads.map((r) => r.load.ms)) / 1000)} s` : "reused"} | ${g[0].cacheMode} |`);
  }
}
console.log(`wrote ${OUT} (${entry.runs.length} runs for ${MODEL}${pageLog.length ? `; ${pageLog.length} page console lines recorded` : ""})`);
