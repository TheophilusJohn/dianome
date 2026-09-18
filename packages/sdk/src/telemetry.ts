// Load report v2 (schemas/telemetry.v2.json): the v1 fields plus optional measurements. Sent once per completed
// load with fetch keepalive; never on abort or failure; nothing at all when telemetry is off.

import type { CacheMode, ChunkSource } from "./cache/types";
import type { Browser, DeviceInfo } from "./device";
import type { FetchLike } from "./manifest";

export type LoadSource = "network" | "per-site-cache" | "cross-site-cache" | "mixed";

export interface LoadReport {
  schema: 2;
  model: string;
  variant: string;
  bytes: number;
  chunks: number;
  ms: number;
  source: LoadSource;
  cache_hits: number;
  browser: Browser;
  webgpu: boolean;
  bytes_per_second?: number;
  verify_ms?: number;
  transfer_ms?: number;
  quota_bytes?: number;
  max_buffer_size?: number;
  cache_mode?: CacheMode;
}

/** All chunks from one source → that source; otherwise `mixed`. No chunks → `network`. */
export function classifySource(sources: Iterable<ChunkSource>): LoadSource {
  let seen: ChunkSource | null = null;
  for (const s of sources) {
    if (seen === null) seen = s;
    else if (seen !== s) return "mixed";
  }
  return seen ?? "network";
}

export function countCacheHits(sources: Iterable<ChunkSource>): number {
  let n = 0;
  for (const s of sources) if (s !== "network") n++;
  return n;
}

/** Median of a sample list; 0 for an empty list. Even counts average the two middle values. */
export function median(samples: readonly number[]): number {
  if (samples.length === 0) return 0;
  const s = [...samples].sort((a, b) => a - b);
  const mid = s.length >> 1;
  return s.length % 2 ? s[mid]! : (s[mid - 1]! + s[mid]!) / 2;
}

export interface ReportInput {
  model: string;
  variant: string;
  bytes: number;
  sources: ChunkSource[];
  ms: number;
  verifyMs: number;
  /** Per-chunk postMessage hops from the cross-site frame; transfer_ms is their median (0 when empty). */
  transferSamples: readonly number[];
  cacheMode: CacheMode;
  device: DeviceInfo;
}

const int = (n: number): number => Math.max(0, Math.round(n));

export function buildReport(i: ReportInput): LoadReport {
  const r: LoadReport = {
    schema: 2, model: i.model, variant: i.variant, bytes: int(i.bytes), chunks: i.sources.length, ms: int(i.ms),
    source: classifySource(i.sources), cache_hits: countCacheHits(i.sources), browser: i.device.browser, webgpu: i.device.webgpu,
    bytes_per_second: i.ms > 0 ? int(i.bytes / (i.ms / 1000)) : 0,
    verify_ms: int(i.verifyMs),
    transfer_ms: int(median(i.transferSamples)),
    cache_mode: i.cacheMode,
  };
  if (i.device.quotaBytes !== null) r.quota_bytes = int(i.device.quotaBytes);
  if (i.device.maxBufferSize !== null) r.max_buffer_size = int(i.device.maxBufferSize);
  return r;
}

/** Session report v3 (schemas/telemetry.v3.json): one run() call. Never carries prompt or output content. */
export interface SessionReport {
  schema: 3;
  model: string;
  variant: string;
  mode: "local" | "split" | "server";
  N: number;
  L: number;
  prompt_tokens: number;
  new_tokens: number;
  client_ms: number;
  server_busy_ms: number;
  rtt_ms: number;
  tok_per_s: number;
  plan_policy: "cost" | "latency" | "local" | "server";
  cache_mode: CacheMode;
  browser: Browser;
  webgpu: boolean;
}

export interface SessionReportInput {
  model: string; variant: string; mode: SessionReport["mode"]; N: number; L: number; promptTokens: number; newTokens: number;
  clientMs: number; serverBusyMs: number; rttMs: number; tokPerS: number; planPolicy: SessionReport["plan_policy"]; cacheMode: CacheMode; device: DeviceInfo;
}

const ms1 = (n: number): number => Math.max(0, Math.round(n * 10) / 10);

export function buildSessionReport(i: SessionReportInput): SessionReport {
  return {
    schema: 3, model: i.model, variant: i.variant, mode: i.mode, N: int(i.N), L: int(i.L), prompt_tokens: int(i.promptTokens), new_tokens: int(i.newTokens),
    client_ms: ms1(i.clientMs), server_busy_ms: ms1(i.serverBusyMs), rtt_ms: ms1(i.rttMs), tok_per_s: ms1(i.tokPerS),
    plan_policy: i.planPolicy, cache_mode: i.cacheMode, browser: i.device.browser, webgpu: i.device.webgpu,
  };
}

/** POSTs the report; resolves to the HTTP status, or null when the request itself failed. Never throws. */
export async function postReport(api: string, report: LoadReport | SessionReport, f: FetchLike = (u, i) => fetch(u, i)): Promise<number | null> {
  try {
    const res = await f(`${api.replace(/\/+$/, "")}/v1/telemetry/load`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(report), keepalive: true,
    });
    return res.status;
  } catch {
    return null;
  }
}
