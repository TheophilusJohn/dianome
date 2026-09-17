// Chunk fetch loop. Grows from apps/load-test/src/load.ts: a pool of `concurrency` workers over the plan's download
// order, retries with exponential backoff on network errors and 5xx (never on 4xx), AbortSignal honoured, and a
// SHA-256 check on every chunk from every source. A corrupt cached chunk is evicted and refetched once.

import { AbortedError, ChunkError, isAbort, isQuotaError, quotaError, throwIfAborted } from "./errors";
import type { FetchLike } from "./manifest";
import type { PlanChunk } from "./plan";
import { sha256 } from "./sha";
import { chunkUrl, sourceFor, type ChunkSource, type ChunkStore } from "./cache/types";

export interface ChunkResult {
  sha: string;
  bytes: number;
  group: string;
  groupIndex: number;
  buf: ArrayBuffer;
  source: ChunkSource;
  /** Time spent in crypto.subtle.digest for this chunk (all sources). */
  verifyMs: number;
  /** Cross-site only: the postMessage hop that delivered this chunk's buffer from the frame, in ms. Null elsewhere. */
  transferMs: number | null;
  /** Network attempts made (0 when served from a cache). */
  attempts: number;
}

export interface FetchStats {
  /** True when a QuotaExceededError switched the session to `cache: "none"` mid-load. */
  cacheDisabled: boolean;
  verifyMs: number;
  /** One postMessage hop per chunk that came through the cross-site frame (completion order). Empty on other paths. */
  transferSamples: number[];
  /** Cache put failures other than quota (the chunk was still delivered). */
  putErrors: number;
  /** Cached chunks that failed verification and were refetched. */
  corruptEvicted: number;
}

export interface FetchOptions {
  cdn: string;
  modelId: string;
  store: ChunkStore | null;
  concurrency?: number;
  retries?: number;
  /** First backoff delay; doubles per attempt. */
  backoffMs?: number;
  signal?: AbortSignal;
  fetch?: FetchLike;
  sleep?: (ms: number, signal?: AbortSignal) => Promise<void>;
  /** Called once per chunk, in completion order. */
  onChunk: (r: ChunkResult) => void;
  onCacheDisabled?: (reason: unknown) => void;
}

export const DEFAULT_CONCURRENCY = 6;
export const DEFAULT_RETRIES = 3;
export const DEFAULT_BACKOFF_MS = 300;

export function defaultSleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) return reject(new AbortedError());
    const t = setTimeout(() => { signal?.removeEventListener("abort", onAbort); resolve(); }, ms);
    const onAbort = () => { clearTimeout(t); reject(new AbortedError()); };
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}

const now = (): number => (typeof performance !== "undefined" ? performance.now() : Date.now());

function isRetryable(e: unknown): boolean {
  if (isAbort(e)) return false;
  if (e instanceof ChunkError) return e.code === "chunk_network" || e.code === "chunk_verify" || (e.code === "chunk_http" && (e.status ?? 0) >= 500);
  return false;
}

/** Fetch one chunk from the network once; classifies the failure so the retry loop can decide. */
export async function fetchChunkOnce(cdn: string, item: PlanChunk, f: FetchLike, signal?: AbortSignal): Promise<ArrayBuffer> {
  let res: Response;
  try {
    res = await f(chunkUrl(cdn, item.sha), { signal: signal ?? null });
  } catch (e) {
    if (isAbort(e) || signal?.aborted) throw new AbortedError();
    throw new ChunkError("chunk_network", item.sha, `network error: ${(e as Error)?.message ?? String(e)}`, undefined, { cause: e });
  }
  if (!res.ok) throw new ChunkError("chunk_http", item.sha, `HTTP ${res.status}`, res.status);
  let buf: ArrayBuffer;
  try {
    buf = await res.arrayBuffer();
  } catch (e) {
    if (isAbort(e) || signal?.aborted) throw new AbortedError();
    throw new ChunkError("chunk_network", item.sha, `body read failed: ${(e as Error)?.message ?? String(e)}`, undefined, { cause: e });
  }
  if (buf.byteLength !== item.bytes) throw new ChunkError("chunk_verify", item.sha, `length ${buf.byteLength}, expected ${item.bytes}`);
  return buf;
}

async function verify(item: PlanChunk, buf: ArrayBuffer): Promise<number> {
  const t = now();
  if (buf.byteLength !== item.bytes) throw new ChunkError("chunk_verify", item.sha, `length ${buf.byteLength}, expected ${item.bytes}`);
  const got = await sha256(buf);
  if (got !== item.sha) throw new ChunkError("chunk_verify", item.sha, `sha256 mismatch: got ${got.slice(0, 12)}`);
  return now() - t;
}

export async function fetchChunks(order: PlanChunk[], opts: FetchOptions): Promise<FetchStats> {
  const concurrency = Math.max(1, opts.concurrency ?? DEFAULT_CONCURRENCY);
  const retries = opts.retries ?? DEFAULT_RETRIES;
  const backoff = opts.backoffMs ?? DEFAULT_BACKOFF_MS;
  const f = opts.fetch ?? ((u, i) => fetch(u, i));
  const sleep = opts.sleep ?? defaultSleep;
  const stats: FetchStats = { cacheDisabled: false, verifyMs: 0, transferSamples: [], putErrors: 0, corruptEvicted: 0 };

  // Internal controller so one chunk's final failure cancels the other in-flight fetches.
  const ac = new AbortController();
  const signal = ac.signal;
  const onOuterAbort = () => ac.abort();
  opts.signal?.addEventListener("abort", onOuterAbort, { once: true });
  if (opts.signal?.aborted) ac.abort();

  let store = opts.store;
  const disableCache = (reason: unknown): void => {
    if (!store) return;
    store = null;
    stats.cacheDisabled = true;
    opts.onCacheDisabled?.(reason);
  };

  /** Runs `attempt` with the retry policy; `attempt` receives the 1-based attempt number. */
  const withRetry = async <T>(item: PlanChunk, attempt: (n: number) => Promise<T>): Promise<{ value: T; attempts: number }> => {
    for (let n = 1; ; n++) {
      throwIfAborted(signal);
      try {
        return { value: await attempt(n), attempts: n };
      } catch (e) {
        if (signal.aborted) throw new AbortedError();
        if (n > retries || !isRetryable(e)) throw e;
        await sleep(backoff * 2 ** (n - 1), signal);
      }
    }
  };

  const fromNetwork = async (item: PlanChunk): Promise<{ buf: ArrayBuffer; verifyMs: number; attempts: number }> => {
    let verifyMs = 0;
    const { value, attempts } = await withRetry(item, async () => {
      const buf = await fetchChunkOnce(opts.cdn, item, f, signal);
      verifyMs = await verify(item, buf);
      return buf;
    });
    return { buf: value, verifyMs, attempts };
  };

  const one = async (item: PlanChunk): Promise<ChunkResult> => {
    const base = { sha: item.sha, bytes: item.bytes, group: item.group, groupIndex: item.groupIndex };
    const s = store;
    if (s?.fetch) {
      // Cross-site: the frame fetches and caches; we verify what it hands back and let it refetch once on corruption.
      const frameFetch = s.fetch.bind(s);
      let transferMs: number | null = null, verifyMs = 0, source: ChunkSource = "network";
      let evictedOnce = false;
      const { value, attempts } = await withRetry(item, async () => {
        const r = await frameFetch(item.sha, item.bytes, opts.modelId, signal);
        transferMs = r.transferMs; // the hop of the attempt that delivered the bytes
        if (r.quota) disableCache(quotaError("cross-site frame: quota exceeded and nothing left to evict"));
        try {
          verifyMs += await verify(item, r.buf);
        } catch (e) {
          if (r.fromCache && !evictedOnce) { evictedOnce = true; stats.corruptEvicted++; await s.evict([item.sha]); }
          throw e;
        }
        source = r.fromCache ? sourceFor(s.mode) : "network";
        return r.buf;
      });
      if (transferMs !== null) stats.transferSamples.push(transferMs);
      stats.verifyMs += verifyMs;
      return { ...base, buf: value, source, verifyMs, transferMs, attempts: source === "network" ? attempts : 0 };
    }
    if (s) {
      let cached: ArrayBuffer | null = null;
      try { cached = await s.get(item.sha, opts.modelId); } catch { cached = null; }
      if (cached) {
        try {
          const verifyMs = await verify(item, cached);
          stats.verifyMs += verifyMs;
          return { ...base, buf: cached, source: sourceFor(s.mode), verifyMs, transferMs: null, attempts: 0 };
        } catch (e) {
          if (isAbort(e)) throw e;
          stats.corruptEvicted++;
          try { await s.evict([item.sha]); } catch { /* the refetch below overwrites it anyway */ }
        }
      }
    }
    const { buf, verifyMs, attempts } = await fromNetwork(item);
    stats.verifyMs += verifyMs;
    if (store) {
      try {
        await store.put(item.sha, buf, opts.modelId);
      } catch (e) {
        if (isQuotaError(e)) disableCache(e);
        else stats.putErrors++;
      }
    }
    return { ...base, buf, source: "network", verifyMs, transferMs: null, attempts };
  };

  let next = 0;
  let failure: unknown = null;
  const worker = async (): Promise<void> => {
    while (next < order.length && failure === null && !signal.aborted) {
      const item = order[next++]!;
      try {
        const r = await one(item);
        if (failure === null && !signal.aborted) opts.onChunk(r);
      } catch (e) {
        if (failure === null) failure = opts.signal?.aborted ? new AbortedError() : e;
        ac.abort();
        return;
      }
    }
  };
  try {
    await Promise.all(Array.from({ length: Math.min(concurrency, order.length) }, worker));
  } finally {
    opts.signal?.removeEventListener("abort", onOuterAbort);
  }
  if (failure !== null) throw failure;
  if (opts.signal?.aborted) throw new AbortedError();
  return stats;
}
