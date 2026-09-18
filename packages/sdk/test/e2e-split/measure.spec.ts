// Measurements for docs/phase-5b-notes.md (q4, Chrome, this Mac): planner inputs; estimated vs measured ms/token
// for N in {0, 4, 8, 12, 16, 20, 24, local}; the planner's choice per policy; per-step breakdown medians.
// Writes results/split-measure.json. Not a gate.
import { test, type Page } from "@playwright/test";
import { mkdirSync, writeFileSync } from "node:fs";

const SPLIT_TOKEN = process.env.SPLIT_TOKEN ?? "sdk-e2e-token";
const WS_URL = "ws://127.0.0.1:8765/";
const results: Record<string, unknown> = {};
let page: Page;
type Harness = typeof import("./page/harness");
function h<K extends keyof Harness>(name: K, ...args: Parameters<Harness[K]>): Promise<Awaited<ReturnType<Harness[K]>>> {
  return page.evaluate(([n, a]) => (window.harness as unknown as Record<string, (...x: unknown[]) => unknown>)[n as string]!(...(a as unknown[])), [name, args] as const) as never;
}
const median = (a: number[]): number => { const s = [...a].sort((x, y) => x - y); return s.length ? s[Math.floor(s.length / 2)]! : 0; };

test.beforeAll(async ({ browser }) => {
  page = await browser.newPage();
  page.on("pageerror", (e) => console.log(`[pageerror] ${e.message}`));
  await page.goto("/");
  await page.waitForFunction(() => Boolean(window.harness));
});
test.afterAll(() => { mkdirSync("results", { recursive: true }); writeFileSync("results/split-measure.json", JSON.stringify({ ...results, updated: new Date().toISOString() }, null, 2)); });

test("planner inputs, choices per policy, and estimated vs measured ms/token over N (q4)", async () => {
  test.setTimeout(1_800_000);
  const p = await h("prompt");
  const bearer = { url: WS_URL, token: SPLIT_TOKEN };
  const plans: Record<string, unknown> = {};
  for (const prefer of ["cost", "latency", "local", "server"] as const) {
    const pl = await h("plan", { prompt: p.text, variant: "q4", maxTokens: 32, policy: { prefer }, split: bearer, microbench: prefer === "cost" ? "fresh" : "cached" });
    plans[prefer] = { mode: pl.mode, N: pl.N, reasons: pl.reasons, estimate: pl.estimate.msPerToken, candidates: pl.candidates.map((c) => ({ mode: c.mode, N: c.N, feasible: c.feasible, msPerToken: c.msPerToken, serverShare: c.serverShare, costPer1M: c.costPer1M, privacy: c.privacy, gpuBytes: c.gpuBytes, downloadBytes: c.downloadBytes, breakdown: c.breakdown, prefillMs: c.prefillMs, reasons: c.reasons })) };
    if (prefer === "cost") results.inputs = pl.inputs;
    console.log(`  policy ${prefer}: ${pl.mode} N=${pl.N} est ${pl.estimate.msPerToken?.toFixed(1)} ms/token — ${pl.reasons.join("; ")}`);
  }
  results.plans = plans;
  const inputs = (results.inputs as { device: Record<string, unknown>; network: Record<string, unknown>; server: Record<string, unknown> | null });
  console.log(`  inputs: device ${JSON.stringify(inputs.device)} network ${JSON.stringify(inputs.network)} server ${JSON.stringify(inputs.server)}`);

  const sweep: unknown[] = [];
  for (const sel of [0, 4, 8, 12, 16, 20, 24, "local"] as const) {
    const force = sel === "local" ? { mode: "local" as const } : { N: sel };
    const runs: { measured: number; steps: Record<string, number>[]; prefill: number; loadMs: number | null; tokens: number; serverBusyMs: number }[] = [];
    let estimate: number | null = null, breakdown: unknown = null, prefillEst: number | null = null;
    for (let i = 0; i < 3; i++) {
      const r = await h("run", { prompt: p.text, variant: "q4", maxTokens: 32, policy: { prefer: "cost" }, split: bearer, force, sampling: { temperature: 0 } });
      const dec = r.timings.slice(1);
      runs.push({ measured: median(dec.map((s) => s.totalMs)), steps: dec as unknown as Record<string, number>[], prefill: r.timings[0]!.totalMs, loadMs: r.loadMs, tokens: r.tokens.length, serverBusyMs: r.serverBusyMs });
      estimate = r.plan.estimate.msPerToken; breakdown = r.plan.estimate.breakdown; prefillEst = r.plan.estimate.prefillMs;
    }
    const best = [...runs].sort((a, b) => a.measured - b.measured)[1]!;   // median run
    const m = (k: string) => median(best.steps.map((s) => s[k]!));
    const row = {
      sel, N: sel === "local" ? 24 : sel, mode: sel === "local" ? "local" : sel === 0 ? "server" : "split", estimate, breakdownEstimate: breakdown, prefillEstimate: prefillEst,
      measured: best.measured, measuredRuns: runs.map((r) => r.measured), prefillMeasured: best.prefill, loadMs: runs[0]!.loadMs, serverBusyMs: best.serverBusyMs, tokens: best.tokens,
      breakdownMeasured: { clientMs: m("clientMs"), exportMs: m("exportMs"), roundTripMs: m("roundTripMs"), serverBusyMs: m("serverBusyMs"), networkMs: Math.max(0, m("roundTripMs") - m("serverBusyMs")), lmHeadMs: m("lmHeadMs"), sampleMs: m("sampleMs"), totalMs: m("totalMs") },
    };
    sweep.push(row);
    console.log(`  ${String(sel).padStart(5)}: est ${estimate?.toFixed(2)} measured ${best.measured.toFixed(2)} ms/token [${runs.map((r) => r.measured.toFixed(2)).join(" ")}]  client ${row.breakdownMeasured.clientMs.toFixed(2)} export ${row.breakdownMeasured.exportMs.toFixed(2)} net ${row.breakdownMeasured.networkMs.toFixed(2)} server ${row.breakdownMeasured.serverBusyMs.toFixed(2)} head ${row.breakdownMeasured.lmHeadMs.toFixed(2)} sample ${row.breakdownMeasured.sampleMs.toFixed(2)}  prefill est ${prefillEst?.toFixed(1)} measured ${best.prefill.toFixed(1)}  load ${runs[0]!.loadMs?.toFixed(0)} ms`);
  }
  results.sweep = sweep;
});
