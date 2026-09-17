# Phase 3 brief — client SDK, progressive loading, cross-site cache

You are working in the `dianome` repo. Read first: `docs/spikes/00-storage-partitioning.md` (every caching decision below comes from it), `docs/briefs/01-ingest.md` (manifest shape), `docs/briefs/02-edge.md` and `docs/phase-2-notes.md` (API, CDN, telemetry), and `apps/load-test/src/load.ts` (the fetch loop this SDK grows out of). TypeScript throughout. Nothing is published or deployed by you; Theo publishes to npm and deploys.

## Goal

`dianome` on npm: a developer adds it to a web app, calls `load(modelId)`, and gets the model's bytes streamed progressively from `cdn.dianome.dev`, verified, cached per-site everywhere and cross-site on Chrome and Firefox after a one-time opt-in, with telemetry reporting what happened. Two demo sites on different registrable domains show the second site loading from cache. This is the first public milestone.

## Layout

```
packages/sdk/                       npm package "dianome"; ESM only, types included, zero runtime deps
  src/index.ts                      Dianome class, load(), stream(), enableCrossSiteCache(), cache controls
  src/manifest.ts                   fetch + validate against schemas/manifest.v1.json (types generated from it)
  src/fetcher.ts                    chunk fetch loop: concurrency 6, retries 3 with backoff, SHA-256 verify
  src/assemble.ts                   entries from chunks: segment concat, 256-byte-aligned views, tied entries
  src/cache/persite.ts              Cache API on the host origin, LRU metadata, quota handling
  src/cache/crosssite.ts            iframe client for the cdn.dianome.dev frame, feature detection, opt-in
  src/cache/types.ts                one ChunkStore interface both implement
  src/device.ts                     WebGPU limits, storage estimate, browser family, throughput
  src/telemetry.ts                  load report v2, opt-out
  src/adapters/transformersjs.ts    entry point "dianome/transformersjs"
  src/adapters/webllm.ts            entry point "dianome/webllm"
  test/                             vitest unit tests (node + happy-dom); playwright e2e for per-site cache
packages/cache-frame/               the cross-site frame, built to static files: index.html, frame.js, optin.html
scripts/publish-frame.sh            uploads packages/cache-frame/dist to R2 under frame/v1/ (Theo runs)
apps/demo-site/                     one Vite site, deployed twice to Pages as dianome-demo-a and dianome-demo-b
docs/phase-3-notes.md               measured numbers only, with the command or page that produced each
```

## Public API

```ts
import { Dianome } from "dianome";

const d = new Dianome({
  api?: string,          // default https://api.dianome.dev
  cdn?: string,          // default https://cdn.dianome.dev
  telemetry?: boolean,   // default true; false sends nothing
  cache?: "auto" | "per-site" | "none",   // default auto: cross-site if enabled + supported, else per-site
});

// Progressive: groups arrive in manifest download order as they complete.
for await (const group of d.stream("qwen2.5-0.5b-instruct", { variant: "q4", signal })) {
  group.name;                       // "embed" | "layer.0" | … | "final_norm" | "lm_head"
  group.entries;                    // Map<name, Entry>; Entry = { role, shape, storage, bytes: Uint8Array, parts?: { weights, scales, zeros } }
}

// Convenience: everything, with progress.
const model = await d.load("qwen2.5-0.5b-instruct", { variant: "q4", onProgress: (p) => {} });
model.manifest; model.group("layer.7"); model.entry("model.layers.7.self_attn.q_proj.weight");
// onProgress p: { bytesDone, bytesTotal, chunksDone, chunksTotal, group, source: "network"|"per-site-cache"|"cross-site-cache" }

// Cross-site opt-in. MUST be called synchronously inside a click handler (Safari rule from Phase 0).
const r = await d.enableCrossSiteCache();
// r: { state: "granted" | "unsupported" | "denied" | "needs-visit", visitUrl?: string }
// "needs-visit" = Chrome has no first-party interaction with cdn.dianome.dev yet; developer should link visitUrl.

d.cache.status();   // { mode: "per-site"|"cross-site"|"none", quotaBytes, usageBytes, chunks }
d.cache.evict(modelId);  d.cache.clear();
```

Entries are views into chunk buffers where possible (no copy when a segment fits in one chunk); a multi-chunk entry is concatenated once. Buffers are transferable; document that `Entry.bytes` may alias a chunk and that consumers must not mutate it.

## Chunk store interface

```ts
interface ChunkStore {
  get(sha: string): Promise<ArrayBuffer | null>;
  put(sha: string, buf: ArrayBuffer): Promise<void>;   // may throw QuotaExceededError
  has(sha: string): Promise<boolean>;
  status(): Promise<{ quotaBytes, usageBytes, chunks }>;
  evict(shas: string[]): Promise<void>;
  clear(): Promise<void>;
}
```

Both stores key by the chunk URL `https://cdn.dianome.dev/chunks/<sha>` in a Cache API cache named `dianome-v1`, so a chunk cached by either path is the same Response object shape. LRU metadata (sha, bytes, lastAccess, modelIds) lives in IndexedDB `dianome-meta`, not in the Cache, so a listing doesn't require opening responses.

**Quota rule (from Phase 0):** on `QuotaExceededError`, evict LRU chunks not belonging to the model being loaded until the put fits or nothing is left; if it still fails, switch the session to `cache: "none"` and continue streaming. Never fail a load because of quota. Chrome Incognito hits this at ~900 MB and `estimate()` doesn't warn.

## Per-site store

`caches.open("dianome-v1")` on the host origin. Straightforward; this is the path every browser gets, and the only path on Safari and iOS.

## Cross-site store (the Phase 0 result turned into code)

The frame at `https://cdn.dianome.dev/frame/v1/index.html` owns the shared cache. The SDK embeds it hidden with `allow="storage-access"`, talks to it over postMessage, and transfers `ArrayBuffer`s (zero-copy). The frame fetches chunks itself (same-origin) so the parent never fetches on this path.

Feature detection and opt-in, in this order:
1. If `document.requestStorageAccess` is missing → `unsupported`.
2. Chrome path: inside the gesture, the frame calls `document.requestStorageAccess({ all: true })`. Handle with `.caches` → use `handle.caches` for every operation, never the frame's global `caches` (Phase 0 T2 vs T3). Reject/undefined handle → next path.
3. Firefox path: inside the gesture, the frame calls `document.requestStorageAccess()` as the first statement (no preceding `await`), then and only then touches `caches`. The frame must not have touched `caches` earlier in that document; enforce it structurally (the store is constructed after the grant). If `hasStorageAccess()` was already true at load, use globals directly.
4. If the grant resolves but a probe write from the frame is not visible via a fresh unpartitioned read, treat as Safari-like → `unsupported`, fall back to per-site. Do this probe: write a 1 KB marker under `/frame/v1/marker` from the top-level opt-in page at opt-in time; after a grant, the frame checks the marker is readable. Chrome and Firefox will see it; Safari won't.
5. Chrome specifically may reject because the user has never visited `cdn.dianome.dev` top-level. Detect and return `needs-visit` with `visitUrl = https://cdn.dianome.dev/frame/v1/optin.html?return=<current url>`. The opt-in page explains in two sentences what the shared cache is, has one button "Enable shared model cache" that writes the marker, and returns to `return`.

Persist the outcome per origin in `localStorage` (`dianome:crosssite = granted|denied|unsupported`) so `cache: "auto"` doesn't re-prompt; `enableCrossSiteCache()` always re-tries when called explicitly.

Frame protocol (versioned, `v: 1` in every message), `targetOrigin` always explicit, both sides validate `event.origin`:
```
parent → frame: { v, id, op: "grant" | "get" | "put" | "has" | "status" | "evict" | "clear", ... }
frame → parent: { v, id, ok: true, result } | { v, id, ok: false, error, code }
```
`get` returns the buffer transferred. `put` receives a transferred buffer. Time each op; expose per-op timings so notes can record the transfer cost (Phase 0 deferred measuring it).

`packages/cache-frame` builds to plain static files (Vite lib mode or a tiny esbuild step); no framework. `scripts/publish-frame.sh` uploads them with `wrangler r2 object put` to `frame/v1/...` with `content-type` set and `cache-control: public, max-age=300` (the Cache Rule respects origin headers; the frame must not be immutable). Bumping `v1` is how a breaking frame change ships.

## Fetcher

Grows from `apps/load-test/src/load.ts`: concurrency 6, retry 3 with exponential backoff on network errors and 5xx (the Phase 2 Node run died on an ECONNRESET with no retry), no retry on 4xx, `AbortSignal` honoured, SHA-256 via `crypto.subtle.digest` on every chunk from every source (cache included — a corrupt cached chunk is evicted and refetched, once). Download order is the manifest's group order; within a group, chunk order. Emit progress per chunk. Record per-chunk `source` for telemetry.

Source classification for the load report: all chunks from one source → that source; otherwise `mixed`; `cache_hits` = chunks not fetched from network.

## Device and telemetry

`device.ts`: WebGPU adapter limits (`maxBufferSize`, `maxStorageBufferBindingSize`, `maxComputeWorkgroupStorageSize`), `navigator.storage.estimate()`, browser family from UA (chrome/firefox/safari/other, nothing finer), measured throughput from the load.

Telemetry v2 adds optional fields to the load report; update `schemas/telemetry.v2.json` and the Worker to accept schema 1 and 2 (schema 2 fields: `bytes_per_second`, `verify_ms`, `transfer_ms` (cross-site postMessage time), `quota_bytes`, `max_buffer_size`, `cache_mode`). Worker stores the new doubles in the next free double slots and `cache_mode` as blob8. Still no IP, no full UA. Add Worker tests for v2.

## Adapters

- `dianome/transformersjs`: given a `files`-family artifact id, returns an object for `env.useCustomCache = true; env.customCache = …` implementing `match(request)` and `put(request, response)` so Transformers.js model files resolve from the Dianome store by path. Verify with the ONNX artifact already in the bucket (`qwen2.5-0.5b-instruct-onnx`): load a pipeline with `env.customCache` set and no network fetch for model files on the second run.
- `dianome/webllm`: WebLLM checks its own Cache API cache (`webllm/model`) before fetching shards. The adapter pre-warms that cache: for each file in an MLC `files` artifact, put a Response under the exact URL WebLLM will request for the configured `model` URL. Theo will `pack-dir` and upload `mlc-ai/Qwen2.5-0.5B-Instruct-q4f16_1-MLC` from Hugging Face as `qwen2.5-0.5b-instruct-mlc` for this; write the adapter against the file names in that repo and the pre-warm instruction in the README. Verify: second run shows no shard fetches in the network log.

## Demo site

`apps/demo-site`: one page. Shows which site you're on, a "Load Qwen2.5-0.5B (q4)" button, progress with bytes/sec and per-group bars, the source breakdown at the end, and an "Enable shared cache" button that calls `enableCrossSiteCache()` from its click handler and shows the returned state (with the visit link when `needs-visit`). Deployed twice by Theo: `wrangler pages deploy apps/demo-site/dist --project-name dianome-demo-a` and `…-demo-b`. **They must be on `pages.dev`, not `dianome.dev` subdomains: two subdomains of `dianome.dev` are one site under storage partitioning and the demo would be showing per-site cache while claiming cross-site.** State that in the README of the app.

## Tests

- Unit (vitest): manifest validation, segment assembly incl. multi-chunk and tied entries and 256-byte alignment, retry/backoff, source classification, LRU eviction order, quota fallback to `none`, frame protocol encode/decode and origin checks, adapter path mapping.
- E2E (Playwright, chromium + firefox + webkit): per-site path against a local static server pointed at `./store`: first load network, second load per-site-cache with zero network chunk requests; corrupt-cache eviction. Cross-site is verified manually on the deployed demo (Phase 0 harness pattern); write the manual checklist into `docs/phase-3-notes.md` with empty cells for Theo to fill.

## Package

`packages/sdk/package.json`: name `dianome`, version `0.1.0`, `"type": "module"`, `exports` for `.`, `./transformersjs`, `./webllm`, `types`, `sideEffects: false`, `files: ["dist"]`, size budget 25 KB gzipped for the main entry (assert in CI script). README: install, 10-line quickstart, the opt-in paragraph developers need to show their users, browser support table straight from Phase 0 (Chrome cross-site silent / prompt with 3PC blocked; Firefox cross-site silent; Safari per-site), and the quota caveat. `pnpm --filter dianome pack` produces the tarball; `npm publish` is Theo's.

## docs/phase-3-notes.md

Measured only: postMessage transfer ms per 8 MB chunk; SHA-256 verify MB/s; per-site first vs second load ms per browser (Playwright output); the manual cross-site table; final bundle sizes; the Transformers.js and WebLLM second-run network counts.

## Definition of done

- `pnpm -r typecheck`, `pnpm --filter dianome test`, Playwright e2e on all three engines pass.
- `pnpm --filter dianome pack` yields a tarball under the size budget with working `exports`.
- Worker accepts telemetry schema 1 and 2 (tests), not deployed.
- `packages/cache-frame` builds; `scripts/publish-frame.sh` exists (not run).
- `apps/demo-site` builds and, served locally with `?cdn=` and `?api=` overrides, loads the q4 model via the SDK.
- Both adapters demonstrated locally as described.
- Nothing published, nothing deployed, no measured number written by hand.
