// Shared by frame.html and index.html: one code path for "load the chunk
// through a CacheStorage". `cacheStorage` is either the default `caches` or
// the `.caches` of a Storage Access API handle (Chrome).

export const CACHE_NAME = 'dianome-spike';
export const CHUNK_PATH = '/chunks/test.bin';

export function chunkUrl() {
  return new URL(CHUNK_PATH, location.origin).href;
}

// Returns { source: 'cache' | 'network', ms, bytes }.
// `ms` covers only the cache match, the fetch on a miss, and reading the
// body — not page load. On a miss the fetch is `no-store` so the HTTP cache
// cannot masquerade as a Cache API hit; T4 (in harness.js) is the only test
// that deliberately uses the HTTP cache.
// `store: false` makes the load read-only: a miss is fetched but never put
// into the cache. T1 uses this so the partitioned baseline can never seed the
// store that T2/T3 then read on the same page.
export async function loadChunk(cacheStorage = caches, { store = true } = {}) {
  const url = chunkUrl();
  const cache = await cacheStorage.open(CACHE_NAME);

  const t0 = performance.now();
  let res = await cache.match(url);
  let source = 'cache';
  let toStore = null;
  if (!res) {
    source = 'network';
    res = await fetch(url, { cache: 'no-store' });
    if (!res.ok) throw new Error(`fetch ${url} → HTTP ${res.status}`);
    if (store) toStore = res.clone();
  }
  const buf = await res.arrayBuffer();
  const ms = performance.now() - t0;

  if (toStore) await cache.put(url, toStore);
  return { source, ms, bytes: buf.byteLength };
}
