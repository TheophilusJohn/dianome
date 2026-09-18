// run() (Phase 5b): plan → load what the plan needs → session → tokens. Lives in its own entry (`dianome/run`,
// reached lazily through `Dianome.prototype.run`) so the main entry stays the size it was. It imports
// `dianome-runtime/tokenizer`, `/planner` and `/session` (pure JS) in every mode and the runtime's main entry
// (WebGPU kernels) only when the plan puts blocks on this device; without WebGPU it plans `server` and the
// runtime entry is never imported.

import type { Dianome } from "./index";
import { isModelManifest, type ModelManifest, type VariantName } from "./manifest";
import { probeDevice, type DeviceInfo } from "./device";
import { buildSessionReport, postReport, type SessionReport } from "./telemetry";
import type { Progress } from "./types";
import { DianomeError } from "./errors";
import { PRIVACY_BAND as band } from "./privacyBand.gen";
import type { Plan, PlanCandidate, PlanInput, PlanPolicy, PlanRate, PlanServer, PrivacyBand } from "dianome-runtime/planner";
import type { ChatMessage } from "dianome-runtime/tokenizer";
import type { SplitClient, SplitSession, StepTiming, SamplingOptions } from "dianome-runtime/session";
import type { Microbench, Runtime } from "dianome-runtime";

export type { Plan, PlanCandidate, PlanInput, PlanPolicy } from "dianome-runtime/planner";
export type { ChatMessage } from "dianome-runtime/tokenizer";
export type { StepTiming, SamplingOptions } from "dianome-runtime/session";

export interface RunOptions {
  /** Chat messages, rendered with the model's chat template; or `prompt`, sent as-is. */
  messages?: ChatMessage[];
  prompt?: string;
  variant?: VariantName;
  policy?: PlanPolicy;
  maxTokens?: number;
  /** Session context length (prompt + new tokens must fit); default 1024. */
  maxCtx?: number;
  sampling?: SamplingOptions;
  onToken?: (text: string, id: number) => void;
  onPlan?: (plan: Plan) => void;
  onProgress?: (p: Progress) => void;
  signal?: AbortSignal;
  /** Bypass POST /v1/split/session (local runs against `dianome-server serve` with SPLIT_TOKEN). */
  split?: { url: string; token: string } | null;
  /** GPU memory cap used for feasibility (default 2 GiB, further capped by the storage quota estimate). */
  gpuBudgetBytes?: number;
  /** Reuse the cached microbench for this adapter/model/variant (default) or measure again. */
  microbench?: "cached" | "fresh";
  addGenerationPrompt?: boolean;
  /**
   * Override the policy's choice with a specific candidate (the slider in the demo): `{ N }` for server (0) or
   * split (1..L), `{ mode: "local" }` for local. The plan is still computed in full; the override must be feasible.
   */
  force?: { N: number } | { mode: "local" } | null;
}

export interface RunResult {
  text: string;
  tokens: number[];
  mode: Plan["mode"];
  N: number;
  plan: Plan;
  timings: StepTiming[];
  serverBusyMs: number;
  /** Decode throughput over the steps after the first (tokens per second of wall time). */
  tokPerS: number;
  costEstimate: { serverShare: number; costPer1M: number | null; rate: PlanRate | null };
  privacy: PlanCandidate["privacy"];
  promptTokens: number;
  telemetry: { report: SessionReport | null; status: number | null };
  /** Bytes streamed for this call (null when the runtime was reused or nothing was loaded). */
  load: RunLoad | null;
  serverStats: Record<string, unknown> | null;
}

export interface RunLoad { bytes: number; ms: number; bytesPerSecond: number; chunks: number; cacheHits: number }

/** Per-Dianome state kept between calls: loaded runtimes (one per N/variant/head), microbench, tokenizer, last load. */
interface RunState {
  device: import("dianome-runtime").AcquiredDevice | null;
  runtimes: Map<string, Runtime>;
  tokenizers: Map<string, Promise<import("dianome-runtime/tokenizer").Tokenizer>>;
  /** Bandwidth of the last network load through run() (the stream stops early once the runtime has its groups, so the SDK's own summary is not produced). */
  lastBandwidth: number | null;
}
const states = new WeakMap<Dianome, RunState>();
function state(d: Dianome): RunState {
  let s = states.get(d);
  if (!s) { s = { device: null, runtimes: new Map(), tokenizers: new Map(), lastBandwidth: null }; states.set(d, s); }
  return s;
}

/** GPU memory budget: the configured cap, further capped by a storage-quota-derived value (quota / 2, at least 1 GiB) when the browser reports one. */
export function gpuBudget(cap: number, quotaBytes: number | null): number {
  if (quotaBytes === null) return cap;
  return Math.min(cap, Math.max(1024 ** 3, Math.floor(quotaBytes / 2)));
}

const now = (): number => (typeof performance !== "undefined" ? performance.now() : Date.now());

// -- inputs ------------------------------------------------------------------------------------

export const SERVER_COST_FLOOR = 0.224; // docs/phase-4-notes.md: cost(24)/cost(0) on the Mac (lm_head + final norm floor)

export function modelInputs(manifest: ModelManifest, variant: VariantName, maxCtx: number): PlanInput["model"] {
  const v = manifest.variants[variant];
  if (!v) throw new DianomeError("manifest_invalid", `variant ${variant} not in manifest ${manifest.id}`);
  const c = manifest.config;
  const L = c.num_hidden_layers, d = c.hidden_size, headDim = d / c.num_attention_heads, inter = c.intermediate_size;
  const g = (name: string) => v.groups.find((x) => x.name === name);
  const blockBytes = Array.from({ length: L }, (_, i) => g(`layer.${i}`)?.bytes ?? 0);
  const embed = g("embed"), head = g("lm_head"), norm = g("final_norm");
  const headEntry = head?.entries.find((e) => e.role === "lm_head.weight");
  const tied = headEntry?.tied === true;
  const entryBytes = (e: { segments: { length: number }[] }) => e.segments.reduce((n, s) => n + s.length, 0);
  const lmHeadGpuBytes = (headEntry ? (tied ? embed?.entries[0] ? entryBytes(embed.entries[0]) : 0 : entryBytes(headEntry)) : 0) + (norm?.bytes ?? 0);
  const kvBytesPerBlock = 2 * maxCtx * c.num_key_value_heads * headDim * 2;
  const workspaceBytes = maxCtx * 4 * (4 * d + c.num_attention_heads * headDim * 2 + c.num_key_value_heads * headDim * 2 + 3 * inter) + maxCtx * d * 2;
  let maxEntryBytes = 0;
  for (const grp of v.groups) for (const e of grp.entries) if (grp.name !== "embed" || tied) maxEntryBytes = Math.max(maxEntryBytes, entryBytes(e));
  const privacy: PrivacyBand | null = band.model === manifest.id ? { model: band.model, coverage500k: band.coverage500k, rows: band.rows.map((r) => ({ ...r })) } : null;
  return {
    id: manifest.id, variant, L, dModel: d, blockBytes, embedBytes: embed?.bytes ?? 0, lmHeadGpuBytes,
    lmHeadDownloadBytes: (tied ? 0 : (head?.bytes ?? 0)) + (norm?.bytes ?? 0), kvBytesPerBlock, workspaceBytes, maxEntryBytes,
    serverCostFloor: SERVER_COST_FLOOR, privacy,
  };
}

/** Which of the variant's groups are already in this browser's chunk cache (blocks 0..k-1 contiguous, and embed). */
export async function cachedGroups(d: Dianome, manifest: ModelManifest, variant: VariantName): Promise<{ embedCached: boolean; cachedBlocks: number }> {
  const store = await d.chunkStore();
  const v = manifest.variants[variant];
  if (!store || !v) return { embedCached: false, cachedBlocks: 0 };
  const cached = async (name: string): Promise<boolean> => {
    const g = v.groups.find((x) => x.name === name);
    if (!g) return false;
    const shas = new Set<string>(g.chunks);
    for (const e of g.entries) for (const s of e.segments) shas.add(s.chunk);
    for (const sha of shas) if (!(await store.has(sha))) return false;
    return true;
  };
  const embedCached = await cached("embed");
  let cachedBlocks = 0;
  while (cachedBlocks < manifest.config.num_hidden_layers && (await cached(`layer.${cachedBlocks}`))) cachedBlocks++;
  return { embedCached, cachedBlocks };
}

/** Download bandwidth from the last load, or a 1 MB probe of the first chunk of the variant (read then cancelled). */
export async function measureBandwidth(d: Dianome, manifest: ModelManifest, variant: VariantName, signal?: AbortSignal): Promise<number | null> {
  const last = state(d).lastBandwidth;
  if (last !== null && last > 0) return last;
  if (d.lastSummary && d.lastSummary.source === "network" && d.lastSummary.bytesPerSecond > 0) return d.lastSummary.bytesPerSecond;
  const sha = manifest.variants[variant]?.groups[0]?.chunks[0];
  if (!sha) return null;
  // Throughput, not latency: the clock starts at the first byte, so the TLS handshake and the round trip to the edge
  // (≈ RTT, which on a remote link dwarfs a 1 MiB read) are excluded, and the whole chunk (up to 8 MiB) is timed.
  try {
    const res = await fetch(`${d.cdn}/chunks/${sha}`, { cache: "no-store", ...(signal ? { signal } : {}) });
    if (!res.ok || !res.body) return null;
    const reader = res.body.getReader();
    const first = await reader.read();
    if (first.done) return null;
    const t0 = now();
    let bytes = 0;
    for (;;) { const { done, value } = await reader.read(); if (done) break; bytes += value.byteLength; }
    const ms = now() - t0;
    return ms > 0 && bytes > 0 ? bytes / (ms / 1000) : null;
  } catch { return null; }
}

async function getJson<T>(url: string, signal?: AbortSignal): Promise<T | null> {
  try { const r = await fetch(url, { ...(signal ? { signal } : {}) }); return r.ok ? ((await r.json()) as T) : null; } catch { return null; }
}

interface ServerPlanJson { L: number; busy_fraction_60s: number; ms_per_block_decode: number | null; ms_per_block_prefill: number | null; lm_head_ms: number | null; device?: string; active_sessions?: number }
interface RatesJson { gpu: string; usd_per_hour: number | null; source: string; retrieved: string }
interface SessionJson { url: string; token: string; expires_at: string }

const MB_KEY = "dianome:microbench:v2";

async function cachedMicrobench(key: string): Promise<Microbench | null> {
  try { const raw = globalThis.localStorage?.getItem(`${MB_KEY}:${key}`); return raw ? (JSON.parse(raw) as Microbench) : null; } catch { return null; }
}
function storeMicrobench(key: string, mb: Microbench): void {
  try { globalThis.localStorage?.setItem(`${MB_KEY}:${key}`, JSON.stringify(mb)); } catch { /* per-viewer convenience only */ }
}

// -- prepare (everything up to and including the plan) --------------------------------------------

export interface Prepared {
  manifest: ModelManifest;
  variant: VariantName;
  tokenizer: import("dianome-runtime/tokenizer").Tokenizer;
  promptIds: number[];
  eosIds: number[];
  plan: Plan;
  /** Connected to the split server (rtt measured) or null; closed by run() or planOnly(). */
  client: SplitClient | null;
  rttMs: number | null;
  rate: PlanRate | null;
  microbench: Microbench | null;
  device: DeviceInfo;
  maxCtx: number;
  maxTokens: number;
  policy: PlanPolicy;
  rtMod: typeof import("dianome-runtime") | null;
  sessMod: typeof import("dianome-runtime/session");
}

export async function prepare(d: Dianome, id: string, opts: RunOptions = {}): Promise<Prepared> {
  const signal = opts.signal;
  const st = state(d);
  const manifest = await d.manifest(id, signal);
  if (!isModelManifest(manifest)) throw new DianomeError("manifest_invalid", `${id} is not a model manifest`);
  const variant: VariantName = opts.variant ?? (manifest.variants.q4 ? "q4" : manifest.variants.q8 ? "q8" : "fp16");
  const policy: PlanPolicy = opts.policy ?? { prefer: "cost" };
  const maxTokens = opts.maxTokens ?? 256;
  const maxCtx = opts.maxCtx ?? 1024;
  const L = manifest.config.num_hidden_layers;

  // tokenizer + prompt (pure JS entry)
  const tokMod = await import("dianome-runtime/tokenizer");
  let tokP = st.tokenizers.get(id);
  if (!tokP) { tokP = tokMod.loadTokenizer(d.cdn, manifest); st.tokenizers.set(id, tokP); }
  const tokenizer = await tokP;
  const promptText = opts.messages ? tokMod.renderChat(opts.messages, { addGenerationPrompt: opts.addGenerationPrompt ?? true }) : opts.prompt;
  if (promptText === undefined) throw new Error("run(): give messages or prompt");
  const promptIds = tokenizer.encode(promptText);
  if (promptIds.length + maxTokens > maxCtx) throw new Error(`prompt (${promptIds.length} tokens) + maxTokens (${maxTokens}) exceeds maxCtx ${maxCtx}`);
  const eosIds = await eosTokens(d, manifest, tokenizer);

  // device + microbench (the runtime entry is imported only when WebGPU is there)
  const device: DeviceInfo = await probeDevice();
  let mb: Microbench | null = null;
  let rtMod: typeof import("dianome-runtime") | null = null;
  if (device.webgpu) {
    try {
      rtMod = await import("dianome-runtime");
      st.device ??= await rtMod.acquireDevice();
      const key = `${st.device.info.vendor}/${st.device.info.architecture}/${st.device.info.device}:${id}:${variant}`;
      mb = opts.microbench === "fresh" ? null : await cachedMicrobench(key);
      if (!mb) { mb = await rtMod.microbench(st.device.device, manifest, variant); storeMicrobench(key, mb); }
    } catch (e) { mb = null; console.warn(`dianome: WebGPU microbench failed, planning without the browser side: ${(e as Error).message ?? e}`); }
  }

  // server: plan, rates, session token, connect + rtt
  const [serverPlan, rates, cached, bandwidth] = await Promise.all([
    getJson<ServerPlanJson>(`${d.api}/v1/split/plan?model=${encodeURIComponent(id)}`, signal),
    getJson<RatesJson>(`${d.api}/v1/split/rates`, signal),
    cachedGroups(d, manifest, variant),
    measureBandwidth(d, manifest, variant, signal),
  ]);
  const sessMod = await import("dianome-runtime/session");
  let client: SplitClient | null = null;
  let rttMs: number | null = null;
  const split = opts.split === undefined ? await mintSession(d, id, maxCtx, signal) : opts.split;
  if (split) {
    try {
      client = new sessMod.SplitClient(split.url, split.token);
      await client.connect();
      rttMs = await client.rtt(5);
    } catch { client?.close(); client = null; rttMs = null; }
  }
  const server: PlanServer | null = serverPlan && typeof serverPlan.L === "number"
    ? { L: serverPlan.L, busyFraction60s: serverPlan.busy_fraction_60s, msPerBlockDecode: serverPlan.ms_per_block_decode, msPerBlockPrefill: serverPlan.ms_per_block_prefill, lmHeadMs: serverPlan.lm_head_ms, ...(serverPlan.device ? { device: serverPlan.device } : {}) }
    : null;
  const rate: PlanRate | null = rates ? { gpu: rates.gpu, usdPerHour: rates.usd_per_hour, source: rates.source, retrieved: rates.retrieved } : null;

  // plan
  const plannerMod = await import("dianome-runtime/planner");
  const input: PlanInput = {
    model: modelInputs(manifest, variant, maxCtx),
    device: {
      webgpu: device.webgpu && mb !== null, maxBufferSize: device.maxBufferSize,
      gpuBudgetBytes: gpuBudget(opts.gpuBudgetBytes ?? plannerMod.DEFAULT_GPU_BUDGET, device.quotaBytes), quotaBytes: device.quotaBytes,
      msPerBlockT1: mb?.msPerBlockT1 ?? null, msPerBlockT32: mb?.msPerBlockT32 ?? null, lmHeadMs: mb?.lmHeadMs ?? null, exportMs: mb?.exportMs ?? null,
      cachedBlocks: cached.cachedBlocks, embedCached: cached.embedCached,
    },
    network: { bytesPerSecond: bandwidth, rttMs, serverReachable: client !== null },
    server, policy, prompt: { promptTokens: promptIds.length, maxNewTokens: maxTokens }, rate,
  };
  let plan = plannerMod.plan(input);
  if (opts.force) {
    const want = "mode" in opts.force ? plan.candidates.find((c) => c.mode === "local") : plan.candidates.find((c) => c.mode !== "local" && c.N === (opts.force as { N: number }).N);
    if (!want) { client?.close(); throw new Error(`force: no such candidate ${JSON.stringify(opts.force)}`); }
    plan = { ...plan, mode: want.mode, N: want.N, estimate: want, reasons: [`forced ${want.mode} N=${want.N} (policy ${policy.prefer} chose ${plan.mode} N=${plan.N})`, ...plan.reasons] };
  }
  opts.onPlan?.(plan);
  return { manifest, variant, tokenizer, promptIds, eosIds, plan, client, rttMs, rate, microbench: mb, device, maxCtx, maxTokens, policy, rtMod, sessMod };
}

/** The plan alone (inputs measured, every candidate estimated); nothing is loaded or generated. */
export async function planOnly(d: Dianome, id: string, opts: RunOptions = {}): Promise<Plan> {
  const p = await prepare(d, id, { prompt: "plan", ...opts });
  p.client?.close();
  return p.plan;
}

// -- run ---------------------------------------------------------------------------------------

export async function run(d: Dianome, id: string, opts: RunOptions = {}): Promise<RunResult> {
  const signal = opts.signal;
  const st = state(d);
  const p = await prepare(d, id, opts);
  const { manifest, variant, tokenizer, promptIds, eosIds, plan, rttMs, rate, device, maxCtx, maxTokens, policy, rtMod, sessMod } = p;
  let client = p.client;
  const L = manifest.config.num_hidden_layers;
  if (!plan.estimate.feasible) { client?.close(); throw new Error(`no feasible plan: ${[...plan.estimate.reasons, ...plan.reasons].join("; ")}`); }
  const { mode, N } = plan;
  if (mode === "local") { client?.close(); client = null; }

  // load what the plan needs
  let runtime: Runtime | null = null;
  let load: RunLoad | null = null;
  if (N >= 1) {
    const key = `${id}:${variant}:${N}:${mode === "local" ? "head" : "split"}:${maxCtx}`;
    runtime = st.runtimes.get(key) ?? null;
    if (!runtime) {
      for (const [k, rt] of st.runtimes) { rt.destroy(); st.runtimes.delete(k); }   // one loaded runtime at a time
      if (!rtMod || !st.device) throw new Error("plan needs the browser side but WebGPU is not available");
      runtime = await rtMod.Runtime.create(st.device.device, manifest, variant, N, maxCtx, { lmHead: mode === "local" });
      const t0 = now();
      let bytes = 0, chunks = 0, cacheHits = 0;
      const gen = d.stream(id, { variant, ...(signal ? { signal } : {}), onProgress: (p) => { bytes = p.bytesDone; chunks = p.chunksDone; if (p.source !== "network") cacheHits++; opts.onProgress?.(p); } });
      try { await runtime.loadFrom(gen); } catch (e) { runtime.destroy(); throw e; }
      const ms = now() - t0;
      load = { bytes, ms, bytesPerSecond: ms > 0 ? bytes / (ms / 1000) : 0, chunks, cacheHits };
      if (cacheHits === 0 && load.bytesPerSecond > 0) st.lastBandwidth = load.bytesPerSecond;
      st.runtimes.set(key, runtime);
    }
  }

  // session
  const session: SplitSession = await sessMod.SplitSession.open({
    mode, N, model: id, maxCtx, ...(runtime ? { runtime } : {}), ...(client ? { client } : {}), ...(opts.sampling ? { sampling: opts.sampling } : {}), eosIds,
  });
  const tokens: number[] = [];
  let text = "";
  const t0 = now();
  let firstAt = t0;
  try {
    for await (const t of session.generate(promptIds, maxTokens, signal)) {
      if (t.step === 0) firstAt = now();
      if (!eosIds.includes(t.id)) {
        tokens.push(t.id);
        const full = tokenizer.decode(tokens);
        const delta = full.slice(text.length);
        text = full;
        if (delta) opts.onToken?.(delta, t.id);
      }
      if (t.done) break;
    }
  } finally {
    // keep the runtime for the next call; close the server side
  }
  const serverStats = await session.serverStats();
  session.close();
  const steps = session.steps;
  const decodeWall = now() - firstAt;
  const tokPerS = steps.length > 1 && decodeWall > 0 ? ((steps.length - 1) / decodeWall) * 1000 : 0;
  const serverBusyMs = steps.reduce((n, s) => n + s.serverBusyMs, 0);
  const clientMs = steps.reduce((n, s) => n + s.clientMs + s.exportMs + s.lmHeadMs + s.sampleMs, 0);

  // telemetry v3 (no prompt content, ever)
  let report: SessionReport | null = null, status: number | null = null;
  if (d.telemetry) {
    report = buildSessionReport({
      model: id, variant, mode, N, L, promptTokens: promptIds.length, newTokens: tokens.length, clientMs, serverBusyMs,
      rttMs: rttMs ?? 0, tokPerS, planPolicy: policy.prefer, cacheMode: (await d.cache.mode()), device,
    });
    status = await postReport(d.api, report);
  }
  return {
    text, tokens, mode, N, plan, timings: steps, serverBusyMs, tokPerS,
    costEstimate: { serverShare: plan.estimate.serverShare, costPer1M: plan.estimate.costPer1M, rate },
    privacy: plan.estimate.privacy, promptTokens: promptIds.length, telemetry: { report, status }, load, serverStats,
  };
}

async function mintSession(d: Dianome, model: string, maxCtx: number, signal?: AbortSignal): Promise<{ url: string; token: string } | null> {
  try {
    const r = await fetch(`${d.api}/v1/split/session`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ model, max_ctx: maxCtx }), ...(signal ? { signal } : {}) });
    if (!r.ok) return null;
    const j = (await r.json()) as SessionJson;
    return typeof j.url === "string" && typeof j.token === "string" ? { url: j.url, token: j.token } : null;
  } catch { return null; }
}

/** eos ids from the store's generation_config.json, else the tokenizer's <|im_end|> / <|endoftext|>. */
async function eosTokens(d: Dianome, manifest: ModelManifest, tokenizer: import("dianome-runtime/tokenizer").Tokenizer): Promise<number[]> {
  try {
    const { fetchStoreFile } = await import("dianome-runtime/tokenizer");
    const bytes = await fetchStoreFile(d.cdn, manifest, "generation_config.json");
    const g = JSON.parse(new TextDecoder().decode(bytes)) as { eos_token_id?: number | number[] };
    if (Array.isArray(g.eos_token_id)) return g.eos_token_id;
    if (typeof g.eos_token_id === "number") return [g.eos_token_id];
  } catch { /* fall through */ }
  return ["<|im_end|>", "<|endoftext|>"].map((t) => tokenizer.tokenToId(t)).filter((x): x is number => typeof x === "number");
}
