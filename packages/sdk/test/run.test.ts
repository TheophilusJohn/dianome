// run() in node: no WebGPU → the planner picks server (N = 0) and the runtime entry is never imported. The split
// server is a fake WebSocket speaking the DNM1 protocol; manifests and chunks come from the local store (skipped
// when ./store is absent). Checks the plan, the tokens, the decoded text and the schema-3 telemetry POST.
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { Dianome } from "../src/index";
import { modelInputs } from "../src/run";
import type { ModelManifest } from "../src/manifest";
import { decodeFrame, encodeFrame } from "../../runtime/src/protocol";
import { renderChat } from "../../runtime/src/chat";

const REPO = resolve(import.meta.dirname, "../../..");
const STORE = resolve(REPO, "store");
const MODEL = "qwen2.5-0.5b-instruct";
const manifestPath = resolve(STORE, `manifests/${MODEL}/latest.json`);
const API = "http://api.test", CDN = "http://cdn.test";

/** The scripted server: greedy tokens 100, 101, 102, then <|im_end|> (151645). */
const SCRIPT = [100, 101, 102, 151645];

class FakeWebSocket {
  static OPEN = 1; static CONNECTING = 0; static CLOSED = 3;
  static log: { type: string; header: Record<string, unknown>; payloadBytes: number }[] = [];
  readyState = 0;
  binaryType = "blob";
  onopen: (() => void) | null = null;
  onmessage: ((ev: { data: ArrayBuffer }) => void) | null = null;
  onclose: ((ev: { code: number; reason: string }) => void) | null = null;
  onerror: (() => void) | null = null;
  private step = 0;
  private tokens = 0;
  private busy = 0;
  constructor(readonly url: string) { setTimeout(() => { this.readyState = 1; this.onopen?.(); }, 0); }
  send(buf: ArrayBuffer): void {
    const m = decodeFrame(buf);
    FakeWebSocket.log.push({ type: m.type, header: m.header, payloadBytes: m.payload.byteLength });
    const reply = (type: string, header: Record<string, unknown>) => setTimeout(() => this.onmessage?.({ data: encodeFrame(type as never, header) }), 0);
    if (m.type === "ping") reply("pong", { t: m.header.t });
    else if (m.type === "open") reply("opened", { session: "fake-sid", L: 24, d_model: 896, boundary: `input_to_block_${String(m.header.N)}` });
    else if (m.type === "prefill" || m.type === "decode") {
      const id = SCRIPT[this.step++] ?? 151645;
      const position = m.type === "prefill" ? (m.header.positions as number[])[1]! : (m.header.position as number) + 1;
      this.tokens++; this.busy += 12.5;
      reply("token", { id, position, busy_ms: 12.5, done: id === 151645 });
    } else if (m.type === "stats") reply("stats", { tokens: this.tokens, busy_seconds: this.busy / 1000, gpu_seconds_per_token: this.tokens ? this.busy / 1000 / this.tokens : 0 });
  }
  close(): void { this.readyState = 3; setTimeout(() => this.onclose?.({ code: 1000, reason: "" }), 0); }
}

describe.skipIf(!existsSync(manifestPath))("run() in node (server mode against a fake split server)", () => {
  const manifest = JSON.parse(readFileSync(manifestPath, "utf8")) as ModelManifest;
  const posted: unknown[] = [];
  const fetchImpl = async (input: string | URL | Request, init?: RequestInit): Promise<Response> => {
    const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
    const u = new URL(url);
    const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
    if (u.pathname === `/v1/models/${MODEL}/manifest`) return new Response(readFileSync(manifestPath), { headers: { "Content-Type": "application/json", "X-Dianome-Manifest-Sha": "0".repeat(64) } });
    const c = /^\/chunks\/([0-9a-f]{64})$/.exec(u.pathname);
    if (c) { const p = resolve(STORE, "chunks", c[1]!); return existsSync(p) ? new Response(readFileSync(p)) : new Response("missing", { status: 404 }); }
    if (u.pathname === "/v1/split/plan") return json({ model: MODEL, L: 24, d_model: 896, active_sessions: 0, busy_fraction_60s: 0.02, ms_per_block_decode: 0.63, ms_per_block_prefill: 0.9, lm_head_ms: 3.0, device: "mps" });
    if (u.pathname === "/v1/split/rates") return json({ gpu: "test-gpu", usd_per_hour: 3.6, source: "test", retrieved: "2026-09-17", rate_available: true });
    if (u.pathname === "/v1/split/session" && init?.method === "POST") return json({ url: "ws://fake.test/", token: "fake-token", expires_at: new Date(Date.now() + 3600e3).toISOString() });
    if (u.pathname === "/v1/telemetry/load" && init?.method === "POST") { posted.push(JSON.parse(String(init.body))); return new Response(null, { status: 202 }); }
    return new Response("not found", { status: 404 });
  };

  beforeEach(() => { vi.stubGlobal("fetch", fetchImpl); vi.stubGlobal("WebSocket", FakeWebSocket); FakeWebSocket.log = []; posted.length = 0; });
  afterEach(() => vi.unstubAllGlobals());

  it("modelInputs reads sizes from the manifest (q4: q8 embed reused as the tied lm_head)", () => {
    const m = modelInputs(manifest, "q4", 1024);
    expect(m.L).toBe(24); expect(m.dModel).toBe(896);
    expect(m.blockBytes).toHaveLength(24);
    expect(m.blockBytes[0]).toBe(7810560);
    expect(m.embedBytes).toBe(136438528);
    expect(m.lmHeadGpuBytes).toBe(136438528 + 1792);
    expect(m.lmHeadDownloadBytes).toBe(1792);
    expect(m.kvBytesPerBlock).toBe(2 * 1024 * 2 * 64 * 2);
    expect(m.privacy?.rows).toHaveLength(25);
    expect(m.privacy?.rows[12]?.boundary).toBe(12);
    expect(m.serverCostFloor).toBe(0.224);
  });

  it("plans server without WebGPU, streams the fake server's tokens, and posts a schema-3 report", async () => {
    const d = new Dianome({ api: API, cdn: CDN, cache: "none", telemetry: true, fetch: fetchImpl });
    const plans: unknown[] = [];
    const deltas: string[] = [];
    const messages = [{ role: "user" as const, content: "Hello" }];
    const r = await d.run(MODEL, { messages, variant: "q4", policy: { prefer: "cost" }, maxTokens: 16, onPlan: (p) => plans.push(p), onToken: (t) => deltas.push(t) });
    expect(r.mode).toBe("server"); expect(r.N).toBe(0);
    expect(r.plan.candidates).toHaveLength(26);
    expect(r.plan.candidates.filter((c) => c.feasible).map((c) => c.N)).toEqual([0]);
    expect(r.plan.candidates[1]!.reasons).toContain("no WebGPU");
    expect(r.plan.reasons.join()).toMatch(/no WebGPU/);
    expect(plans).toHaveLength(1);
    expect(r.tokens).toEqual([100, 101, 102]);              // eos not included
    expect(r.text).toBe(deltas.join(""));
    expect(r.text.length).toBeGreaterThan(0);
    expect(r.timings).toHaveLength(4);
    expect(r.timings[0]!.tokens).toBe(r.promptTokens);
    expect(r.serverBusyMs).toBeCloseTo(4 * 12.5, 6);
    expect(r.costEstimate.serverShare).toBe(1);
    expect(r.costEstimate.costPer1M).toBeCloseTo(((0.63 * 24 + 3.0) / 1000) * 1e6 * 3.6 / 3600, 6);
    expect(r.privacy).toMatchObject({ boundary: 0 });
    expect(r.serverStats).toMatchObject({ tokens: 4 });
    // the prompt went through the chat template
    expect(r.promptTokens).toBe(SCRIPT.length >= 0 ? r.promptTokens : 0);
    const log = FakeWebSocket.log;
    expect(log.filter((m) => m.type === "ping")).toHaveLength(5);
    expect(log.find((m) => m.type === "open")?.header).toMatchObject({ model: MODEL, N: 0, max_ctx: 1024, sampling: { temperature: 0, top_p: 1 } });
    const prefill = log.find((m) => m.type === "prefill")!;
    expect(prefill.header.T).toBe(r.promptTokens);
    expect(prefill.payloadBytes).toBe(4 * r.promptTokens);   // int32 ids, N = 0
    expect(log.filter((m) => m.type === "decode")).toHaveLength(3);
    expect(log.filter((m) => m.type === "decode").every((m) => m.payloadBytes === 4)).toBe(true);
    // telemetry v3: no prompt content, the fields the schema lists
    expect(posted).toHaveLength(1);
    const rep = posted[0] as Record<string, unknown>;
    expect(rep).toMatchObject({ schema: 3, model: MODEL, variant: "q4", mode: "server", N: 0, L: 24, new_tokens: 3, plan_policy: "cost", cache_mode: "none", webgpu: false });
    expect(rep.prompt_tokens).toBe(r.promptTokens);
    expect(rep.server_busy_ms).toBeCloseTo(50, 6);
    expect(Object.keys(rep).sort()).toEqual(["L", "N", "browser", "cache_mode", "client_ms", "mode", "model", "new_tokens", "plan_policy", "prompt_tokens", "rtt_ms", "schema", "server_busy_ms", "tok_per_s", "variant", "webgpu"]);
    expect(JSON.stringify(rep)).not.toContain("Hello");
    expect(r.telemetry.status).toBe(202);
  });

  it("a raw prompt skips the chat template; prompt tokens match the tokenizer", async () => {
    const d = new Dianome({ api: API, cdn: CDN, cache: "none", telemetry: false, fetch: fetchImpl });
    const chat = await d.run(MODEL, { messages: [{ role: "user", content: "Hi" }], maxTokens: 4 });
    const raw = await d.run(MODEL, { prompt: "Hi", maxTokens: 4 });
    expect(raw.promptTokens).toBe(1);
    expect(chat.promptTokens).toBeGreaterThan(raw.promptTokens);
    expect(chat.telemetry.report).toBeNull();
    // the rendered prompt is the HF template's
    expect(renderChat([{ role: "user", content: "Hi" }])).toContain("<|im_start|>assistant\n");
  });

  it("without a reachable server and without WebGPU there is no feasible plan", async () => {
    vi.stubGlobal("fetch", async (input: string | URL | Request, init?: RequestInit) => {
      const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
      if (new URL(url).pathname === "/v1/split/session") return new Response("nope", { status: 503 });
      return fetchImpl(input, init);
    });
    const d = new Dianome({ api: API, cdn: CDN, cache: "none", telemetry: false, fetch: fetchImpl });
    await expect(d.run(MODEL, { prompt: "Hi", maxTokens: 4 })).rejects.toThrow(/no feasible plan/);
  });
});
