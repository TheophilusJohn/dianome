// One ChunkStore interface, implemented by the per-site store (Cache API on the host origin) and the cross-site
// store (postMessage client for the cdn.dianome.dev frame). Both key chunks by URL in a Cache named `dianome-v1`.

export type CacheMode = "per-site" | "cross-site" | "none";
export type ChunkSource = "network" | "per-site-cache" | "cross-site-cache";

export interface ChunkStoreStatus {
  /** From navigator.storage.estimate(); null when the browser does not report it. */
  quotaBytes: number | null;
  /** Sum of cached chunk lengths from the LRU metadata (not a Cache API listing). */
  usageBytes: number;
  chunks: number;
}

export interface ChunkStore {
  readonly mode: "per-site" | "cross-site";
  /** Cached bytes or null on a miss. `modelId` records which model touched the chunk (LRU metadata). */
  get(sha: string, modelId?: string): Promise<ArrayBuffer | null>;
  /**
   * Stores a chunk. Applies the quota rule: on QuotaExceededError, evicts least-recently-used chunks that do not
   * belong to `modelId` until the write fits; rethrows QuotaExceededError only when nothing is left to evict.
   */
  put(sha: string, buf: ArrayBuffer, modelId?: string): Promise<void>;
  has(sha: string): Promise<boolean>;
  status(): Promise<ChunkStoreStatus>;
  evict(shas: string[]): Promise<void>;
  /** Drops `modelId` from every chunk's owner set and deletes chunks left with no owner. */
  evictModel(modelId: string): Promise<void>;
  clear(): Promise<void>;
  /**
   * Cross-site only: the frame fetches the chunk itself (same-origin with the CDN), caches it, and transfers the
   * bytes back, so the parent never fetches on that path. `fromCache` tells the caller which source to report.
   */
  fetch?(sha: string, bytes: number, modelId: string, signal?: AbortSignal): Promise<{ buf: ArrayBuffer; fromCache: boolean; transferMs: number; /** The frame could not cache it (quota, nothing evictable): the session should stop caching. */ quota?: true }>;
  close?(): void;
}

export const CACHE_NAME = "dianome-v1";
export const META_DB = "dianome-meta";

export function chunkUrl(cdn: string, sha: string): string {
  return `${cdn.replace(/\/+$/, "")}/chunks/${sha}`;
}

export function sourceFor(mode: "per-site" | "cross-site"): ChunkSource {
  return mode === "per-site" ? "per-site-cache" : "cross-site-cache";
}
