// Per-test browser-ish environment for the per-site store: a fresh in-memory IndexedDB factory and CacheStorage.
import { IDBFactory } from "fake-indexeddb";
import { PerSiteStore } from "../../src/cache/persite";
import { FakeCacheStorage } from "./fakeCaches";

export function makeStore(o: { cdn?: string; capacity?: number; now?: () => number; quota?: number } = {}): { store: PerSiteStore; caches: FakeCacheStorage; idb: IDBFactory } {
  const caches = new FakeCacheStorage(o.capacity ?? Number.POSITIVE_INFINITY);
  const idb = new IDBFactory();
  const store = new PerSiteStore({
    cdn: o.cdn ?? "https://cdn.test", caches: caches.asCacheStorage(), indexedDB: idb,
    storage: o.quota !== undefined ? { estimate: async () => ({ quota: o.quota as number, usage: caches.used }) } : undefined,
    ...(o.now ? { now: o.now } : {}),
  });
  return { store, caches, idb };
}

/** A monotonic clock for deterministic LRU ordering. */
export function clock(start = 1_000): () => number {
  let t = start;
  return () => ++t;
}
