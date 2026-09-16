// Fetch logic only: no DOM. This file is the seed of the Phase 3 SDK.
//
// loadVariant() fetches every chunk of one variant in group download order with a fixed concurrency,
// verifies each chunk's SHA-256 and length as it arrives, and reports progress through a callback.
// postLoadReport() sends the telemetry point described in schemas/telemetry.v1.json.

export type Sha256 = string;
export interface Group { name: string; bytes: number; chunks: Sha256[]; tied?: true }
export interface Variant { bytes: number; groups: Group[] }
export interface ModelManifest {
  schema: 1; id: string; chunk_size: number;
  chunks: Record<Sha256, { bytes: number }>;
  variants: Partial<Record<"fp16" | "q8" | "q4", Variant>>;
}

export type LoadSource = "network" | "per-site-cache" | "cross-site-cache" | "mixed";
export type Browser = "chrome" | "firefox" | "safari" | "other";
export interface LoadReport {
  schema: 1; model: string; variant: string; bytes: number; chunks: number; ms: number;
  source: LoadSource; cache_hits: number; browser: Browser; webgpu: boolean;
}

export interface GroupProgress { name: string; done: number; total: number; bytes: number; bytesDone: number }
export interface Progress {
  bytesDone: number; bytesTotal: number; chunksDone: number; chunksTotal: number;
  elapsedMs: number; bytesPerSecond: number; groups: GroupProgress[];
}
export interface LoadResult {
  model: string; variant: string; bytes: number; chunks: number; ms: number; bytesPerSecond: number;
  manifestSha: string | null;
  /** Sum of PerformanceResourceTiming.transferSize over chunk fetches (0 for a chunk = served from an HTTP cache); null when unavailable. */
  transferBytes: number | null;
  /** Chunk resource entries seen, and how many had transferSize 0 (a cache hit to infer from later). */
  resourceEntries: number; zeroTransferEntries: number;
}
export interface LoadOptions {
  manifestUrl: string;
  variant: "fp16" | "q8" | "q4";
  /** Base URL of the chunk store; chunks are fetched from `${chunkBase}/chunks/<sha256>`. */
  chunkBase: string;
  concurrency?: number;
  onProgress?: (p: Progress) => void;
  signal?: AbortSignal;
}

export class ChunkError extends Error {
  readonly sha: Sha256;
  constructor(sha: Sha256, message: string) { super(`${sha.slice(0, 12)}: ${message}`); this.sha = sha; }
}

export function hex(buf: ArrayBuffer): string {
  return Array.from(new Uint8Array(buf), (b) => b.toString(16).padStart(2, "0")).join("");
}

export async function sha256(data: ArrayBuffer): Promise<string> {
  return hex(await crypto.subtle.digest("SHA-256", data));
}

/** Ordered (group, chunk) pairs for a variant; a chunk listed twice within the variant is fetched once. */
export function downloadOrder(variant: Variant): { group: string; sha: Sha256 }[] {
  const seen = new Set<Sha256>();
  const out: { group: string; sha: Sha256 }[] = [];
  for (const g of variant.groups) for (const sha of g.chunks) {
    if (seen.has(sha)) continue;
    seen.add(sha);
    out.push({ group: g.name, sha });
  }
  return out;
}

export async function fetchManifest(url: string, signal?: AbortSignal): Promise<{ manifest: ModelManifest; sha: string | null }> {
  const res = await fetch(url, { signal });
  if (!res.ok) throw new Error(`manifest ${url}: ${res.status} ${res.statusText}`);
  const manifest = (await res.json()) as ModelManifest;
  if (manifest.schema !== 1) throw new Error(`unsupported manifest schema ${String(manifest.schema)}`);
  if (!manifest.variants) throw new Error("not a model manifest (no variants)");
  return { manifest, sha: res.headers.get("X-Dianome-Manifest-Sha") };
}

/** Fetch one chunk and verify its length and SHA-256 against the manifest. */
export async function fetchChunk(chunkBase: string, sha: Sha256, expectedBytes: number, signal?: AbortSignal): Promise<ArrayBuffer> {
  const res = await fetch(`${chunkBase}/chunks/${sha}`, { signal });
  if (!res.ok) throw new ChunkError(sha, `HTTP ${res.status}`);
  const buf = await res.arrayBuffer();
  if (buf.byteLength !== expectedBytes) throw new ChunkError(sha, `length ${buf.byteLength}, expected ${expectedBytes}`);
  const got = await sha256(buf);
  if (got !== sha) throw new ChunkError(sha, `sha256 mismatch: got ${got.slice(0, 12)}`);
  return buf;
}

export async function loadVariant(opts: LoadOptions): Promise<LoadResult> {
  const concurrency = opts.concurrency ?? 6;
  const chunkBase = opts.chunkBase.replace(/\/+$/, "");
  const perf = typeof performance !== "undefined" ? performance : undefined;
  if (perf?.clearResourceTimings) { perf.setResourceTimingBufferSize?.(4096); perf.clearResourceTimings(); }

  const { manifest, sha: manifestSha } = await fetchManifest(opts.manifestUrl, opts.signal);
  const variant = manifest.variants[opts.variant];
  if (!variant) throw new Error(`variant ${opts.variant} not in manifest ${manifest.id}`);

  const order = downloadOrder(variant);
  const size = (sha: Sha256): number => {
    const c = manifest.chunks[sha];
    if (!c) throw new ChunkError(sha, "not in chunk table");
    return c.bytes;
  };
  const groups = new Map<string, GroupProgress>();
  for (const g of variant.groups) groups.set(g.name, { name: g.name, done: 0, total: 0, bytes: 0, bytesDone: 0 });
  for (const { group, sha } of order) { const g = groups.get(group)!; g.total++; g.bytes += size(sha); }
  const bytesTotal = order.reduce((s, o) => s + size(o.sha), 0);

  const t0 = Date.now();
  let bytesDone = 0, chunksDone = 0;
  const progress = (): Progress => {
    const elapsedMs = Date.now() - t0;
    return { bytesDone, bytesTotal, chunksDone, chunksTotal: order.length, elapsedMs, bytesPerSecond: elapsedMs > 0 ? bytesDone / (elapsedMs / 1000) : 0, groups: [...groups.values()] };
  };
  opts.onProgress?.(progress());

  // Worker pool over the ordered queue: `concurrency` fetches in flight, queue order preserved.
  let next = 0;
  let failed: unknown = null;
  const worker = async (): Promise<void> => {
    while (next < order.length && failed === null) {
      const { group, sha } = order[next++]!;
      const expected = size(sha);
      const buf = await fetchChunk(chunkBase, sha, expected, opts.signal);
      bytesDone += buf.byteLength; chunksDone++;
      const g = groups.get(group)!; g.done++; g.bytesDone += buf.byteLength;
      opts.onProgress?.(progress());
    }
  };
  const results = await Promise.allSettled(Array.from({ length: Math.min(concurrency, order.length) }, () => worker().catch((e) => { failed ??= e; throw e; })));
  const rejected = results.find((r): r is PromiseRejectedResult => r.status === "rejected");
  if (rejected) throw rejected.reason;

  const ms = Date.now() - t0;
  let transferBytes: number | null = null, resourceEntries = 0, zeroTransferEntries = 0;
  if (perf?.getEntriesByType) {
    const prefix = `${chunkBase}/chunks/`;
    const entries = perf.getEntriesByType("resource").filter((e) => e.name.startsWith(prefix)) as PerformanceResourceTiming[];
    resourceEntries = entries.length;
    if (entries.length > 0) {
      transferBytes = entries.reduce((s, e) => s + e.transferSize, 0);
      zeroTransferEntries = entries.filter((e) => e.transferSize === 0).length;
    }
  }
  return { model: manifest.id, variant: opts.variant, bytes: bytesDone, chunks: chunksDone, ms, bytesPerSecond: ms > 0 ? bytesDone / (ms / 1000) : 0, manifestSha, transferBytes, resourceEntries, zeroTransferEntries };
}

export function detectBrowser(ua: string): Browser {
  if (/firefox\//i.test(ua)) return "firefox";
  if (/edg\/|chrome\/|chromium\//i.test(ua)) return "chrome";
  if (/safari\//i.test(ua) && !/chrome\//i.test(ua)) return "safari";
  return "other";
}

export function buildReport(r: LoadResult, browser: Browser, webgpu: boolean, source: LoadSource = "network"): LoadReport {
  return { schema: 1, model: r.model, variant: r.variant, bytes: r.bytes, chunks: r.chunks, ms: Math.round(r.ms), source, cache_hits: 0, browser, webgpu };
}

/** POST the report to `${apiBase}/v1/telemetry/load`; resolves to the HTTP status (202 on success). */
export async function postLoadReport(apiBase: string, report: LoadReport): Promise<number> {
  const res = await fetch(`${apiBase.replace(/\/+$/, "")}/v1/telemetry/load`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(report),
  });
  if (!res.ok) throw new Error(`telemetry ${res.status}: ${await res.text()}`);
  return res.status;
}
