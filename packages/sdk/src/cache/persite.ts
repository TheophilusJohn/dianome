// Per-site chunk store: Cache API cache `dianome-v1` on the host origin, keyed by the chunk URL, with LRU metadata
// in IndexedDB. This is the path every browser gets and the only path on Safari and iOS.
//
// Quota rule (Phase 0): on QuotaExceededError, evict least-recently-used chunks that do not belong to the model
// being loaded until the write fits or nothing is left; then rethrow so the session can switch to `cache: "none"`.

import { isQuotaError, quotaError } from "../errors";
import { MetaStore } from "./idb";
import { CACHE_NAME, chunkUrl, type ChunkStore, type ChunkStoreStatus } from "./types";

export interface PerSiteStoreOptions {
  cdn: string;
  caches?: CacheStorage;
  indexedDB?: IDBFactory;
  storage?: Pick<StorageManager, "estimate"> | undefined;
  now?: () => number;
  cacheName?: string;
  metaName?: string;
}

/** Wraps a chunk buffer the way both stores do, so a chunk cached by either path looks the same. */
export function chunkResponse(sha: string, buf: ArrayBuffer): Response {
  return new Response(buf, {
    headers: { "Content-Type": "application/octet-stream", "Content-Length": String(buf.byteLength), "X-Dianome-Sha256": sha },
  });
}

export class PerSiteStore implements ChunkStore {
  readonly mode = "per-site" as const;
  private readonly cdn: string;
  private readonly cacheStorage: CacheStorage;
  private readonly meta: MetaStore;
  private readonly storage: Pick<StorageManager, "estimate"> | undefined;
  private readonly now: () => number;
  private readonly cacheName: string;
  private cache: Promise<Cache> | null = null;

  constructor(opts: PerSiteStoreOptions) {
    this.cdn = opts.cdn;
    this.cacheStorage = opts.caches ?? caches;
    this.meta = new MetaStore(opts.indexedDB ?? indexedDB, opts.metaName);
    this.storage = opts.storage ?? (typeof navigator !== "undefined" ? navigator.storage : undefined);
    this.now = opts.now ?? Date.now;
    this.cacheName = opts.cacheName ?? CACHE_NAME;
  }

  static supported(): boolean {
    return typeof caches !== "undefined" && typeof indexedDB !== "undefined";
  }

  private openCache(): Promise<Cache> {
    this.cache ??= this.cacheStorage.open(this.cacheName).catch((e) => { this.cache = null; throw e; });
    return this.cache;
  }

  private url(sha: string): string { return chunkUrl(this.cdn, sha); }

  async get(sha: string, modelId?: string): Promise<ArrayBuffer | null> {
    const cache = await this.openCache();
    const res = await cache.match(this.url(sha));
    if (!res) return null;
    const buf = await res.arrayBuffer();
    await this.meta.touch(sha, modelId, this.now()).catch(() => {});
    return buf;
  }

  async has(sha: string): Promise<boolean> {
    const cache = await this.openCache();
    return (await cache.match(this.url(sha))) !== undefined;
  }

  async put(sha: string, buf: ArrayBuffer, modelId?: string): Promise<void> {
    const cache = await this.openCache();
    const url = this.url(sha);
    for (;;) {
      try {
        await cache.put(url, chunkResponse(sha, buf));
        break;
      } catch (e) {
        if (!isQuotaError(e)) throw e;
        const freed = await this.evictLru(buf.byteLength, modelId);
        if (freed === 0) throw quotaError(`cannot cache ${buf.byteLength} bytes: quota exceeded and nothing left to evict`);
      }
    }
    const existing = await this.meta.get(sha).catch(() => undefined);
    const modelIds = existing?.modelIds ?? [];
    if (modelId && !modelIds.includes(modelId)) modelIds.push(modelId);
    await this.meta.put({ sha, bytes: buf.byteLength, lastAccess: this.now(), modelIds });
  }

  /**
   * Evicts least-recently-used chunks not owned by `protectModelId` until at least `bytes` are freed or nothing
   * evictable is left. Returns the bytes freed.
   */
  async evictLru(bytes: number, protectModelId?: string): Promise<number> {
    const all = await this.meta.all();
    const victims: string[] = [];
    let freed = 0;
    for (const m of all) {
      if (freed >= bytes) break;
      if (protectModelId && m.modelIds.includes(protectModelId)) continue;
      victims.push(m.sha);
      freed += m.bytes;
    }
    await this.evict(victims);
    return freed;
  }

  async evict(shas: string[]): Promise<void> {
    if (shas.length === 0) return;
    const cache = await this.openCache();
    await Promise.all(shas.map((sha) => cache.delete(this.url(sha))));
    await this.meta.delete(shas);
  }

  async evictModel(modelId: string): Promise<void> {
    const all = await this.meta.all();
    const gone: string[] = [];
    for (const m of all) {
      if (!m.modelIds.includes(modelId)) continue;
      const rest = m.modelIds.filter((id) => id !== modelId);
      if (rest.length === 0) gone.push(m.sha);
      else await this.meta.put({ ...m, modelIds: rest });
    }
    await this.evict(gone);
  }

  async clear(): Promise<void> {
    await this.cacheStorage.delete(this.cacheName);
    this.cache = null;
    await this.meta.clear();
  }

  async status(): Promise<ChunkStoreStatus> {
    const all = await this.meta.all();
    let quotaBytes: number | null = null;
    try {
      const est = await this.storage?.estimate();
      if (typeof est?.quota === "number") quotaBytes = est.quota;
    } catch { /* not reported */ }
    return { quotaBytes, usageBytes: all.reduce((s, m) => s + m.bytes, 0), chunks: all.length };
  }

  /** LRU-ordered metadata, oldest first (exposed for tests and the demo). */
  list(): Promise<{ sha: string; bytes: number; lastAccess: number; modelIds: string[] }[]> {
    return this.meta.all();
  }

  close(): void { this.meta.close(); }
}
