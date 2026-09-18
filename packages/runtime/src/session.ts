// SplitSession (Phase 5b): one generation session at a fixed split point N.
//   local  : N = L, lm_head + sampling in the runtime, no server at all
//   split  : 1 <= N <= L, blocks 0..N-1 here, the rest (+ lm_head + sampling) on the server
//   server : N = 0, token ids go to the server
// Per step it records client ms (runtime prefill/decode), export ms, round-trip ms (send → token), the server's
// busy_ms from the token message, lm_head ms and sampling ms (local). N is fixed for the session's lifetime;
// re-planning happens only between calls (run() may pick another N next time).

import type { Runtime } from "./index";
import { Sampler, type SamplingOptions } from "./lmhead";
import type { SplitClient, TokenMessage } from "./protocol";

export type SessionMode = "local" | "split" | "server";

export interface SessionOptions {
  mode: SessionMode;
  N: number;
  model: string;
  maxCtx: number;
  /** Required when N >= 1 (created with lmHead: true for local mode). */
  runtime?: Runtime;
  /** A connected SplitClient; required unless mode is local. */
  client?: SplitClient;
  sampling?: SamplingOptions;
  /** Token ids that end generation (local mode; the server applies its own eos and reports `done`). */
  eosIds?: number[];
}

export interface StepTiming {
  step: number;
  position: number;
  /** Prompt tokens at step 0, else 1. */
  tokens: number;
  /** Runtime prefill (step 0) or decode ms; 0 in server mode. */
  clientMs: number;
  exportMs: number;
  /** Wall ms from sending the frame to receiving the token; 0 in local mode. */
  roundTripMs: number;
  /** The server's busy_ms from the token message; 0 in local mode. */
  serverBusyMs: number;
  lmHeadMs: number;
  sampleMs: number;
  totalMs: number;
}

export interface GeneratedToken { id: number; step: number; position: number; done: boolean; timing: StepTiming }

const now = (): number => performance.now();

export class SplitSession {
  readonly steps: StepTiming[] = [];
  readonly sampler: Sampler;
  opened: Record<string, unknown> | null = null;
  private closed = false;

  private constructor(readonly opts: SessionOptions) {
    const { mode, N, runtime, client } = opts;
    if (mode === "server" && N !== 0) throw new Error("server mode is N = 0");
    if (mode !== "server" && N < 1) throw new Error(`${mode} mode needs N >= 1`);
    if (N >= 1 && !runtime) throw new Error("N >= 1 needs a runtime");
    if (runtime && runtime.N !== N) throw new Error(`runtime has N = ${runtime.N}, session wants ${N}`);
    if (mode === "local" && !(runtime && runtime.lmHead && N === runtime.cfg.layers)) throw new Error("local mode needs a runtime created with lmHead: true and N = L");
    if (mode !== "local" && !client) throw new Error(`${mode} mode needs a connected SplitClient`);
    this.sampler = new Sampler(opts.sampling ?? {});
  }

  get mode(): SessionMode { return this.opts.mode; }
  get N(): number { return this.opts.N; }

  /** Opens the server session (split/server) → `opened` { session, L, d_model }; local mode only validates. */
  static async open(opts: SessionOptions): Promise<SplitSession> {
    const s = new SplitSession(opts);
    if (opts.mode !== "local") {
      const sp = opts.sampling ?? {};
      s.opened = await opts.client!.open(opts.model, opts.N, opts.maxCtx, { temperature: sp.temperature ?? 0, top_p: sp.topP ?? 1, ...(sp.seed !== undefined ? { seed: sp.seed } : {}) });
      if (typeof s.opened.L === "number" && opts.runtime && s.opened.L !== opts.runtime.cfg.layers) throw new Error(`server L=${String(s.opened.L)} != model L=${opts.runtime.cfg.layers}`);
    }
    return s;
  }

  /** Streams up to `maxTokens` tokens for `prompt` (token ids). The last yielded token has `done: true`. */
  async *generate(prompt: ArrayLike<number>, maxTokens: number, signal?: AbortSignal): AsyncGenerator<GeneratedToken, void, void> {
    if (this.closed) throw new Error("session closed");
    const { mode, N, runtime, client } = this.opts;
    const T = prompt.length;
    if (T < 1) throw new Error("empty prompt");
    if (maxTokens < 1) throw new Error("maxTokens must be >= 1");
    if (T + maxTokens > this.opts.maxCtx) throw new Error(`prompt (${T}) + maxTokens (${maxTokens}) exceeds maxCtx ${this.opts.maxCtx}`);
    const eos = new Set(this.opts.eosIds ?? []);
    const ids = Int32Array.from(prompt as ArrayLike<number>);
    let step = 0, pos = T;
    // step 0: prefill
    let t = blank(step, pos, T);
    if (N >= 1) { runtime!.reset(); await runtime!.prefill(ids); t.clientMs = runtime!.stats().lastPrefillMs; }
    let next = mode === "local" ? await this.localToken(t) : await this.serverToken(t, N >= 1 ? (h) => client!.prefill(h, T, 0) : () => client!.prefillIds(ids, 0));
    for (;;) {
      t.totalMs = now() - t.start;
      this.steps.push(strip(t));
      const done = next.done || eos.has(next.id) || step + 1 >= maxTokens || pos + 1 >= this.opts.maxCtx || (signal?.aborted ?? false);
      yield { id: next.id, step, position: pos, done, timing: this.steps[this.steps.length - 1]! };
      if (done) return;
      step++;
      const fedAt = pos;   // the token just emitted occupies position `pos`
      pos++;
      t = blank(step, pos, 1);
      const id = next.id;
      if (N >= 1) { await runtime!.decode(id, fedAt); t.clientMs = runtime!.stats().lastDecodeMs; }
      next = mode === "local" ? await this.localToken(t) : await this.serverToken(t, N >= 1 ? (h) => client!.decode(h, fedAt) : () => client!.decodeId(id, fedAt));
    }
  }

  private async localToken(t: Timing): Promise<{ id: number; done: boolean }> {
    const rt = this.opts.runtime!;
    const logits = await rt.logits();
    t.lmHeadMs = rt.stats().lastLmHeadMs;
    const t0 = now();
    const id = this.sampler.sample(logits);
    t.sampleMs = now() - t0;
    return { id, done: false };
  }

  /** N >= 1: exports the hidden state and sends it; N = 0: sends ids. Times the round trip. */
  private async serverToken(t: Timing, send: (hidden: Uint16Array) => Promise<TokenMessage>): Promise<{ id: number; done: boolean }> {
    const rt = this.opts.runtime;
    let hidden: Uint16Array = new Uint16Array(0);
    if (this.opts.N >= 1) { hidden = (await rt!.exportHidden()) as Uint16Array; t.exportMs = rt!.stats().lastExportMs; }
    const t0 = now();
    const msg = await send(hidden);
    t.roundTripMs = now() - t0;
    t.serverBusyMs = msg.busy_ms;
    return { id: msg.id, done: msg.done };
  }

  /** Server stats (tokens, busy_seconds, gpu_seconds_per_token); null in local mode. */
  async serverStats(): Promise<Record<string, unknown> | null> {
    if (this.opts.mode === "local" || !this.opts.client) return null;
    try { return await this.opts.client.stats(); } catch { return null; }
  }

  close(): void {
    this.closed = true;
    if (this.opts.mode !== "local") this.opts.client?.close();
  }
}

type Timing = StepTiming & { start: number };
function blank(step: number, position: number, tokens: number): Timing {
  return { step, position, tokens, clientMs: 0, exportMs: 0, roundTripMs: 0, serverBusyMs: 0, lmHeadMs: 0, sampleMs: 0, totalMs: 0, start: now() };
}
function strip(t: Timing): StepTiming { const { start: _s, ...rest } = t; return rest; }
