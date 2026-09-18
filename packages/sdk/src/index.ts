// `dianome` entry point: the Dianome class. load() and stream() fetch a model's chunks progressively from the CDN,
// verify every chunk, cache per-site (Cache API on the host origin) or cross-site (the cdn.dianome.dev frame),
// assemble entries as views into chunk buffers, and post one load report when a load completes.

import { assembleGroup } from "./assemble";
import { connectCrossSite, readPersisted } from "./cache/crosssite";
import { PerSiteStore } from "./cache/persite";
import type { CacheMode, ChunkSource, ChunkStore } from "./cache/types";
import { probeDevice, type DeviceInfo } from "./device";
import { AbortedError, DianomeError, isAbort } from "./errors";
import { fetchChunks, type ChunkResult } from "./fetcher";
import { fetchManifest, isModelManifest, type FetchLike, type Manifest, type VariantName } from "./manifest";
import { planFiles, planVariant, type LoadPlan } from "./plan";
import { buildReport, median, postReport } from "./telemetry";
import type { CacheStatus, CrossSiteResult, DianomeOptions, LoadSummary, LoadedGroup, LoadedModel, Progress, SplitServerOverride, StreamOptions } from "./types";
import type { RunOptions, RunResult } from "./run";

export type { LoadedEntry, EntryParts } from "./assemble";
export type { CacheMode, ChunkSource, ChunkStore, ChunkStoreStatus } from "./cache/types";
export type { Browser, DeviceInfo } from "./device";
export { DianomeError, ChunkError, ManifestError, AbortedError } from "./errors";
export type { DianomeErrorCode } from "./errors";
export type { Config, Entry, File, FilesManifest, Group, Manifest, ModelManifest, Part, Segment, Storage, Variant, VariantName } from "./manifest";
export { validateManifest, isModelManifest, isFilesManifest } from "./manifest";
export type { LoadReport, LoadSource, SessionReport } from "./telemetry";
export type { RunOptions, RunResult, Plan, PlanCandidate, PlanInput, PlanPolicy, ChatMessage, StepTiming, SamplingOptions } from "./run";
export type { CacheStatus, CrossSiteProgress, CrossSiteResult, CrossSiteState, DianomeOptions, LoadSummary, LoadedGroup, LoadedModel, Progress, SplitServerOverride, StreamOptions } from "./types";

export const DEFAULT_API = "https://api.dianome.dev";
export const DEFAULT_CDN = "https://cdn.dianome.dev";

const now = (): number => (typeof performance !== "undefined" ? performance.now() : Date.now());

/** Unbounded async queue: the fetch loop pushes, the generator pulls. */
class Channel<T> {
  private items: T[] = [];
  private waiters: ((v: T) => void)[] = [];
  push(v: T): void {
    const w = this.waiters.shift();
    if (w) w(v); else this.items.push(v);
  }
  pull(): Promise<T> {
    const v = this.items.shift();
    if (v !== undefined) return Promise.resolve(v);
    return new Promise((resolve) => this.waiters.push(resolve));
  }
}
type Event = { chunk: ChunkResult } | { done: Awaited<ReturnType<typeof fetchChunks>> } | { error: unknown };

export class Dianome {
  readonly api: string;
  readonly cdn: string;
  readonly telemetry: boolean;
  readonly cacheOption: "auto" | "per-site" | "none";
  /** The API key (Phase 6), or undefined; never placed in any report. */
  readonly apiKey: string | undefined;
  /** Self-host override: model id → your own split server (Phase 6). */
  readonly splitServers: Record<string, SplitServerOverride> | undefined;
  /** @internal The key id the API returned on the last session mint (goes into session reports; never the key). */
  keyId: string | null = null;
  private readonly concurrency: number | undefined;
  private readonly retries: number | undefined;
  private readonly frameUrl: string | undefined;
  private readonly fetchImpl: FetchLike;
  private storePromise: Promise<ChunkStore | null> | null = null;
  private device: Promise<DeviceInfo> | null = null;
  /** Set when a QuotaExceededError switched this session to `cache: "none"`. */
  private cacheDisabled = false;
  /** The last completed load's summary (also available from load()); null before any load. */
  lastSummary: LoadSummary | null = null;

  constructor(opts: DianomeOptions = {}) {
    this.api = (opts.api ?? DEFAULT_API).replace(/\/+$/, "");
    this.cdn = (opts.cdn ?? DEFAULT_CDN).replace(/\/+$/, "");
    this.telemetry = opts.telemetry ?? true;
    this.cacheOption = opts.cache ?? "auto";
    this.concurrency = opts.concurrency;
    this.retries = opts.retries;
    this.frameUrl = opts.frameUrl;
    this.fetchImpl = opts.fetch ?? ((u, i) => fetch(u, i));
    this.apiKey = opts.apiKey;
    this.splitServers = opts.split?.servers;
  }

  /** Fetches and validates the manifest for `id` through the API. */
  async manifest(id: string, signal?: AbortSignal): Promise<Manifest> {
    return (await fetchManifest(this.api, id, { ...(signal ? { signal } : {}), fetch: this.fetchImpl })).manifest;
  }

  private deviceInfo(): Promise<DeviceInfo> {
    this.device ??= probeDevice();
    return this.device;
  }

  /** The store for the effective cache mode; null for `none` (or after quota disabled caching this session). */
  private store(): Promise<ChunkStore | null> {
    if (this.cacheDisabled || this.cacheOption === "none") return Promise.resolve(null);
    this.storePromise ??= this.openStore().catch(() => null);
    return this.storePromise;
  }

  private async openStore(): Promise<ChunkStore | null> {
    if (this.cacheOption === "auto" && readPersisted() === "granted") {
      const { store } = await connectCrossSite({ cdn: this.cdn, frameUrl: this.frameUrl, explicit: false });
      if (store) return store;
    }
    return PerSiteStore.supported() ? new PerSiteStore({ cdn: this.cdn }) : null;
  }

  private async mode(): Promise<CacheMode> {
    const s = await this.store();
    return s ? s.mode : "none";
  }

  /**
   * Cross-site opt-in. Call it from a click handler. Always re-tries even when a previous outcome was persisted.
   * Returns `needs-click` with `mount()` when the browser wants the click inside the CDN frame itself (the first
   * grant on a site), and `needs-visit` with `visitUrl` when Chrome has never seen cdn.dianome.dev top-level.
   */
  async enableCrossSiteCache(): Promise<CrossSiteResult> {
    const adopt = (store: ChunkStore) => { this.storePromise = Promise.resolve(store); this.cacheDisabled = false; };
    const conn = await connectCrossSite({ cdn: this.cdn, frameUrl: this.frameUrl, explicit: true, onGranted: adopt });
    if (conn.store) adopt(conn.store);
    return conn.result;
  }

  readonly cache = {
    mode: (): Promise<CacheMode> => this.mode(),
    status: async (): Promise<CacheStatus> => {
      const s = await this.store();
      if (!s) return { mode: "none", quotaBytes: null, usageBytes: 0, chunks: 0 };
      const st = await s.status();
      return { mode: s.mode, ...st };
    },
    evict: async (modelId: string): Promise<void> => { await (await this.store())?.evictModel(modelId); },
    clear: async (): Promise<void> => { await (await this.store())?.clear(); },
  };

  private plan(manifest: Manifest, opts: StreamOptions): LoadPlan {
    if (isModelManifest(manifest)) {
      const variant: VariantName = opts.variant ?? (manifest.variants.q4 ? "q4" : manifest.variants.q8 ? "q8" : "fp16");
      return planVariant(manifest, variant);
    }
    return planFiles(manifest, opts.files);
  }

  /**
   * Progressive load: yields groups in manifest download order as each completes (a later group that finishes
   * early waits its turn). The generator's return value is the load summary; breaking out aborts the fetch.
   */
  async *stream(id: string, opts: StreamOptions = {}): AsyncGenerator<LoadedGroup, LoadSummary, void> {
    const signal = opts.signal;
    if (signal?.aborted) throw new AbortedError();
    const t0 = now();
    const [manifest, store, device] = await Promise.all([this.manifest(id, signal), this.store(), this.deviceInfo()]);
    const plan = this.plan(manifest, opts);

    const ac = new AbortController();
    const onAbort = () => ac.abort();
    signal?.addEventListener("abort", onAbort, { once: true });
    if (signal?.aborted) ac.abort();

    const chunks = new Map<string, ArrayBuffer>();
    const neededBy = new Map<string, number[]>();
    const pending = plan.groups.map((g) => g.needs.length);
    for (const g of plan.groups) for (const sha of g.needs) (neededBy.get(sha) ?? neededBy.set(sha, []).get(sha)!).push(g.index);

    const sources: ChunkSource[] = [];
    let bytesDone = 0;
    let cacheDisabled = false;
    const ch = new Channel<Event>();
    const loop = fetchChunks(plan.order, {
      cdn: this.cdn, modelId: plan.modelId, store, signal: ac.signal, fetch: this.fetchImpl,
      ...(this.concurrency !== undefined ? { concurrency: this.concurrency } : {}),
      ...(this.retries !== undefined ? { retries: this.retries } : {}),
      onChunk: (chunk) => ch.push({ chunk }),
      onCacheDisabled: () => { cacheDisabled = true; this.cacheDisabled = true; this.storePromise = null; },
    }).then((done) => ch.push({ done }), (error) => ch.push({ error }));

    let nextGroup = 0;
    let stats: Awaited<ReturnType<typeof fetchChunks>> | null = null;
    try {
      for (;;) {
        const ev = await ch.pull();
        if ("error" in ev) throw ev.error;
        if ("done" in ev) { stats = ev.done; break; }
        const r = ev.chunk;
        chunks.set(r.sha, r.buf);
        sources.push(r.source);
        bytesDone += r.bytes;
        for (const gi of neededBy.get(r.sha) ?? []) pending[gi]!--;
        if (opts.onProgress) {
          const elapsedMs = now() - t0;
          opts.onProgress({ bytesDone, bytesTotal: plan.bytesTotal, chunksDone: sources.length, chunksTotal: plan.order.length, group: r.group, source: r.source, elapsedMs, bytesPerSecond: elapsedMs > 0 ? bytesDone / (elapsedMs / 1000) : 0 });
        }
        while (nextGroup < plan.groups.length && pending[nextGroup] === 0) {
          yield this.yieldGroup(plan, nextGroup, chunks);
          nextGroup++;
        }
      }
      // Groups that need no chunks at all (nothing in practice, but the plan allows it) complete once fetching ends.
      while (nextGroup < plan.groups.length && pending[nextGroup] === 0) {
        yield this.yieldGroup(plan, nextGroup, chunks);
        nextGroup++;
      }
      if (nextGroup < plan.groups.length) throw new DianomeError("chunk_missing", `group ${plan.groups[nextGroup]!.name} incomplete after fetch loop`);
    } catch (e) {
      ac.abort();
      if (isAbort(e) || signal?.aborted) throw new AbortedError();
      throw e;
    } finally {
      ac.abort();
      signal?.removeEventListener("abort", onAbort);
      await loop.catch(() => {});
    }

    const ms = now() - t0;
    const cacheMode: CacheMode = cacheDisabled ? "none" : store ? store.mode : "none";
    const summary: LoadSummary = {
      model: plan.modelId, variant: plan.label, bytes: bytesDone, chunks: sources.length, ms,
      bytesPerSecond: ms > 0 ? bytesDone / (ms / 1000) : 0,
      source: "network", cacheHits: 0, cacheMode, sources, verifyMs: stats!.verifyMs,
      transferMs: median(stats!.transferSamples), transferSamples: stats!.transferSamples,
      cacheDisabled, report: null, telemetryStatus: null,
    };
    const report = buildReport({ model: plan.modelId, variant: plan.label, bytes: bytesDone, sources, ms, verifyMs: stats!.verifyMs, transferSamples: stats!.transferSamples, cacheMode, device });
    summary.source = report.source;
    summary.cacheHits = report.cache_hits;
    if (this.telemetry) {
      summary.report = report;
      summary.telemetryStatus = await postReport(this.api, report, this.fetchImpl, this.apiKey);
    }
    this.lastSummary = summary;
    return summary;
  }

  private yieldGroup(plan: LoadPlan, index: number, chunks: Map<string, ArrayBuffer>): LoadedGroup {
    const g = plan.groups[index]!;
    const entries = assembleGroup(g, chunks);
    // Drop our reference to chunks no later group needs; entries that alias them keep the memory alive as needed.
    for (const sha of g.needs) if (plan.lastUse.get(sha) === index) chunks.delete(sha);
    return { name: g.name, index, bytes: g.bytes, tied: g.tied, entries };
  }

  /**
   * Generate tokens for a prompt, choosing per device and network whether to run the whole model here (local),
   * split it at a planner-chosen N, or send tokens to the server. Implemented in the `dianome/run` entry, loaded
   * on first use so this entry stays small; it imports `dianome-runtime` only when the plan needs the browser side.
   */
  run(id: string, opts: RunOptions = {}): Promise<RunResult> {
    return import("./run.js").then((m) => m.run(this, id, opts));
  }

  /** The plan run() would make (inputs measured on this device and network, every candidate estimated), without loading or generating. */
  planRun(id: string, opts: RunOptions = {}): Promise<import("./run.js").Plan> {
    return import("./run.js").then((m) => m.planOnly(this, id, opts));
  }

  /** @internal The effective chunk store (run() asks it which groups are cached). */
  chunkStore(): Promise<ChunkStore | null> { return this.store(); }

  /** Everything, with progress. */
  async load(id: string, opts: StreamOptions = {}): Promise<LoadedModel> {
    const manifest = await this.manifest(id, opts.signal);
    const gen = this.stream(id, opts);
    const groups = new Map<string, LoadedGroup>();
    for (;;) {
      const n = await gen.next();
      if (n.done) {
        const summary = n.value;
        return {
          manifest, variant: summary.variant, groups, summary,
          group: (name) => {
            const g = groups.get(name);
            if (!g) throw new DianomeError("chunk_missing", `no group ${name}`);
            return g;
          },
          entry: (name) => {
            for (const g of groups.values()) { const e = g.entries.get(name); if (e) return e; }
            throw new DianomeError("chunk_missing", `no entry ${name}`);
          },
        };
      }
      groups.set(n.value.name, n.value);
    }
  }
}
