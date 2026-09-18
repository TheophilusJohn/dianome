// run() end to end (Phase 5b): the three modes against the full-model greedy tokens, and the policy runs with a
// v3 telemetry POST through wrangler dev. Results go to results/split-e2e.json for docs/phase-5b-notes.md.
import { test, expect, type Page } from "@playwright/test";
import { mkdirSync, writeFileSync } from "node:fs";

const SPLIT_TOKEN = process.env.SPLIT_TOKEN ?? "sdk-e2e-token";
const WS_URL = "ws://127.0.0.1:8765/";
const results: Record<string, unknown> = {};
let page: Page;
type Harness = typeof import("./page/harness");
function h<K extends keyof Harness>(name: K, ...args: Parameters<Harness[K]>): Promise<Awaited<ReturnType<Harness[K]>>> {
  return page.evaluate(([n, a]) => (window.harness as unknown as Record<string, (...x: unknown[]) => unknown>)[n as string]!(...(a as unknown[])), [name, args] as const) as never;
}
const telemetry = async () => (await (await fetch("http://127.0.0.1:8799/__telemetry")).json()) as { status: number; body: Record<string, unknown> }[];

test.beforeAll(async ({ browser }) => {
  page = await browser.newPage();
  page.on("console", (m) => { if (m.type() === "error" || m.type() === "warning") console.log(`[page ${m.type()}] ${m.text()}`); });
  page.on("pageerror", (e) => console.log(`[pageerror] ${e.message}`));
  await page.goto("/");
  await page.waitForFunction(() => Boolean(window.harness));
  results.info = await h("info");
});
test.afterAll(() => { mkdirSync("results", { recursive: true }); writeFileSync("results/split-e2e.json", JSON.stringify({ ...results, updated: new Date().toISOString() }, null, 2)); });

const bearer = { url: WS_URL, token: SPLIT_TOKEN };

test("fp16 greedy, 16 tokens: N = 0 (server), split at the planner's N (local infeasible), local — all equal the full-model greedy", async () => {
  const p = await h("prompt");
  const g = await h("greedy", "fp16");
  const out: Record<string, unknown> = {};
  // server: N = 0 (policy server)
  const server = await h("run", { prompt: p.text, variant: "fp16", maxTokens: 16, policy: { prefer: "server" }, split: bearer, sampling: { temperature: 0 } });
  out.server = { mode: server.mode, N: server.N, tokens: server.tokens, tokPerS: server.tokPerS, serverBusyMs: server.serverBusyMs };
  console.log(`  server  N=${server.N} tokens=${JSON.stringify(server.tokens)}`);
  expect(server.mode).toBe("server");
  expect(server.tokens).toEqual(g.tokens);
  // split: prefer cost with a GPU budget that fits every fp16 block but not the 272 MiB head → the planner's max feasible N
  const split = await h("run", { prompt: p.text, variant: "fp16", maxTokens: 16, policy: { prefer: "cost" }, split: bearer, gpuBudgetBytes: 900 * 2 ** 20, sampling: { temperature: 0 } });
  out.split = { mode: split.mode, N: split.N, tokens: split.tokens, tokPerS: split.tokPerS, serverBusyMs: split.serverBusyMs, reasons: split.plan.reasons, loadMs: split.loadMs };
  console.log(`  split   N=${split.N} (${split.plan.reasons.join("; ")}) tokens=${JSON.stringify(split.tokens)} load ${split.loadMs?.toFixed(0)} ms`);
  expect(split.mode).toBe("split");
  expect(split.N).toBeGreaterThanOrEqual(1);
  expect(split.tokens).toEqual(g.tokens);
  // local: prefer local (fits)
  const local = await h("run", { prompt: p.text, variant: "fp16", maxTokens: 16, policy: { prefer: "local" }, split: bearer, sampling: { temperature: 0 } });
  out.local = { mode: local.mode, N: local.N, tokens: local.tokens, tokPerS: local.tokPerS, reasons: local.plan.reasons, loadMs: local.loadMs };
  console.log(`  local   N=${local.N} tokens=${JSON.stringify(local.tokens)} ${local.tokPerS.toFixed(1)} tok/s load ${local.loadMs?.toFixed(0)} ms`);
  expect(local.mode).toBe("local");
  expect(local.tokens).toEqual(g.tokens);
  results.fp16Gates = out;
});

test("run() with prefer cost and prefer latency (q4, chat template, session token from the Worker): a plan, tokens, and a v3 telemetry POST", async () => {
  const before = (await telemetry()).length;
  const out: Record<string, unknown> = {};
  for (const prefer of ["cost", "latency"] as const) {
    // no `split`: the session token comes from POST /v1/split/session on wrangler dev (through the 8799 proxy, Origin checked)
    const r = await h("run", { messages: [{ role: "user", content: "Name three colours." }], variant: "q4", maxTokens: 24, policy: { prefer } });
    out[prefer] = { mode: r.mode, N: r.N, tokens: r.tokens.length, text: r.text, tokPerS: r.tokPerS, reasons: r.plan.reasons, estimate: r.plan.estimate.msPerToken, rtt: r.plan.inputs.network.rttMs, report: r.telemetry.report, status: r.telemetry.status, serverBusyMs: r.serverBusyMs };
    console.log(`  prefer ${prefer}: ${r.mode} N=${r.N} (${r.plan.reasons.join("; ")}) ${r.tokens.length} tokens "${r.text.slice(0, 60).replace(/\n/g, " ")}" ${r.tokPerS.toFixed(1)} tok/s est ${r.plan.estimate.msPerToken?.toFixed(1)} ms/token rtt ${r.plan.inputs.network.rttMs?.toFixed(2)} telemetry ${r.telemetry.status}`);
    expect(r.plan.candidates.length).toBe(26);
    expect(r.tokens.length).toBeGreaterThan(0);
    expect(r.text.length).toBeGreaterThan(0);
    expect(r.deltas.join("")).toBe(r.text);
    expect(r.telemetry.status).toBe(202);
    expect(r.telemetry.report).toMatchObject({ schema: 3, model: "qwen2.5-0.5b-instruct", variant: "q4", mode: r.mode, N: r.N, L: 24, plan_policy: prefer, new_tokens: r.tokens.length });
    expect(r.plan.inputs.network.serverReachable).toBe(true);   // the session token came from wrangler dev
    if (r.mode !== "local") expect(r.serverBusyMs).toBeGreaterThan(0);
  }
  const posted = (await telemetry()).slice(before);
  expect(posted.length).toBe(2);
  for (const t of posted) { expect(t.status).toBe(202); expect(t.body.schema).toBe(3); expect(JSON.stringify(t.body)).not.toContain("colours"); }
  results.policies = out;
  results.telemetryPosted = posted;
});
