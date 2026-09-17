# Spike 00 — Storage partitioning

As of: 2026-09-16 · Hardware: MacBook Pro, Apple Silicon (arm64), 16 GB · test.bin SHA-256: 0b42c0cbd1bcaedf8d29b317d314baab3adef596d3ff40080e28578fd55269fa · PSL check: pages.dev present (Y, line 12705) · Chrome partition key observed on site A (Y — T1 on A returned `network` after a top-level Enable)

Payload 20 MB. Download from Cloudflare measured 1.1–8.7 s across runs (roughly 20–150 Mbps); ms values below are for a single load and vary with the link, so read cache/network, not the absolute number.

## Cross-site cache (load on A first, then run on B)

| Browser (version) | T1 partitioned | T2 SAA | T3 SAA handle | T4 HTTP cache | Prompt shown? | Gestures needed | Persists after restart |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Chrome 152.0.7977.84 | network 4256 ms | network 3075 ms | cache 21 ms | transferSize=20971820, 3811 ms | No | 1 click on B (+ one-time Enable on CDN) | Yes — T3 cache 27 ms after full quit, no Enable, no prompt |
| Chrome (3PC blocked, Incognito) | network 3486 ms | network 3581 ms | cache 21 ms | transferSize=20971820, 2535 ms | Yes, once ("Allow embedded content?"); not repeated on reload | 1 click + Allow | Not testable in Incognito |
| Firefox 152.0 | network 8694 ms | network (see correction) | n/a — resolves without a handle | transferSize=20971820, 1114 ms | No (auto-granted after first-party visit) | 1 click | Not tested |
| Safari 26.5 | network 4256 ms | network 2275 ms (with grant, empty partitioned store) | n/a — prompts, then resolves without a handle | unreliable: transferSize=0 but 3351 ms (network); Safari does not expose transferSize cross-origin | Yes, once | 1 click + Allow, and the call must be synchronous in the gesture | n/a — no cross-site hit |

Cells for T1–T3: `cache <ms>` or `network <ms>` or `error: <message>`. T4: `transferSize=<bytes> <ms>`.

Notes on individual cells:
- Firefox T2: the first attempt returned `network` because T1 had already obtained `caches` in the same document before the grant; Firefox switches the storage principal at grant time and earlier objects stay partitioned. Retested after clearing B's site data with T2 as the first action in a fresh document → `cache`.
- Safari T2: `requestStorageAccess()` rejects with `NotAllowedError` when any `await` precedes it in the click handler. The grant was obtained through T3's synchronous call instead; T2 was then run in a fresh document with the grant from load and an empty partitioned store → `network`. Safari's grant covers cookies only.
- Chrome T2 with cookies allowed resolves silently and grants nothing usable: the frame's default `caches` stays partitioned. Only the `{all: true}` handle reaches unpartitioned storage.

## Storage and device

| Browser | quota (GB) | persist() | T5 ceiling (GB) | maxBufferSize | maxStorageBufferBindingSize | adapter | deviceMemory |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Chrome | 10.76 in frame (297 top-level) | false | ≥2.0 (harness cap, no error) | 4294967292 (4 GiB) | 4294967292 (4 GiB) | apple / metal-3 | 16 |
| Chrome (3PC blocked) | not probed | | 0.9 (Incognito: in-memory, QuotaExceededError at 900 MB) | (same build) | (same build) | | |
| Firefox | 10.74 in frame | false | ≥2.0 (harness cap, no error) | 1073741824 (1 GiB) | 1073741824 (1 GiB) | (empty) | null |
| Safari | 8.25 in frame | false | ≥2.0 (harness cap, no error) | 2147483644 (2 GiB) | 2147483644 (2 GiB) | "apple" in every field | null |

hardwareConcurrency: Chrome 10, Firefox 10, Safari 8 (capped). maxComputeWorkgroupStorageSize: 32768 on all three.

## Decision

Yellow: Chrome only (corrected 2026-09-17, see below). Cross-site cache hit on B after a one-time opt-in plus one click in Chrome (via the `{all: true}` handle); Firefox and Safari do not unpartition the Cache API through Storage Access and get per-site caching only. The original reading was Green with Firefox via plain `requestStorageAccess()`; the Firefox hit was a confounded partitioned copy.
Consequence for Phase 3: build both paths in the SDK. Cross-site: Chrome handle path, Firefox plain-SAA path, feature-detected. Per-site: Cache API on the developer's origin, always present, and the only path on Safari (and iOS).
Planner inputs available on every browser: `maxBufferSize`, `maxStorageBufferBindingSize`, `maxComputeWorkgroupStorageSize`, `hardwareConcurrency`, `navigator.storage.estimate()`. `deviceMemory` is Chrome-only. Adapter vendor/architecture is usable in Chrome, empty in Firefox, uninformative in Safari.

Green = cache hit on B after a one-time opt-in plus one click on at least two of three browsers.
Yellow = hit only with a prompt on every site. Red = no hit anywhere. Yellow and Red both reframe the CDN to progressive streaming plus per-site caching.

## SDK caveats carried forward

- Chrome: after `requestStorageAccess({all: true})`, read and write only through `handle.caches`; the frame's default `caches` is still partitioned. One silent grant with cookies allowed; one remembered prompt per site with cookies blocked. Grant persists across browser restart.
- Firefox: call `requestStorageAccess()` before touching `caches`, `indexedDB` or any storage object; objects obtained before the grant stay partitioned for the life of the document. No handle is returned; use the globals after the grant.
- Safari: `requestStorageAccess()` must be the first statement in the gesture handler, no preceding `await`. Even granted, storage stays partitioned, so the SDK should not bother requesting access on WebKit; go straight to per-site caching.
- All: the HTTP cache is partitioned by top-level site everywhere; `immutable` headers help repeat loads on the same site only.
- All: third-party storage quota is ~8–11 GB regardless of the ~300 GB first-party quota, and 2 GB of writes were accepted in every browser, so the shared cache holds a 3B q4 model comfortably and roughly one 7B q4. Per-layer chunking plus LRU eviction by chunk is required, not optional.
- Chrome Incognito caps storage at ~900 MB in memory and `estimate()` does not report it; the SDK should treat a `QuotaExceededError` during a write as a signal to fall back to streaming without caching, not as a fatal error.
- Firefox's 1 GiB `maxBufferSize` is the floor for any single weight buffer; per-layer chunks stay well under it, but a whole-model buffer would not.
- Resource Timing `transferSize` is not exposed cross-origin in Safari even with `Timing-Allow-Origin`; do not use it for cache-hit telemetry there.

## Correction (2026-09-17)

Firefox does **not** unpartition the Cache API through the Storage Access API. Measured on the production origins
with the frame's diagnostic page (`cdn.dianome.dev/frame/v1/diag.html`, embedded by `dianome-demo-a.pages.dev/?diag=1`;
Phase 0 T2 in that document, nothing touched `caches` before the click), Firefox, fresh profile, after the top-level
opt-in visit that writes the marker. Output as reported by Theo:

```
requestStorageAccess() resolves
hasStorageAccess() true
0 keys in dianome-v1
marker not visible
```

So the grant is real (cookie access), but `caches` in the frame stays the partitioned copy: the marker written
top-level on the same origin is invisible, and the store the SDK fills through that frame is the (site A, cdn)
partition, not shared with site B. The Phase 0 Firefox T2 "cache 78 ms" was a confounded partitioned copy, not a
cross-site hit (the harness could not tell the two apart); the row above now reads "network (see correction)". The decision is Yellow: Chrome only. Chrome's `requestStorageAccess({all: true})` handle remains the
only path to unpartitioned storage; Firefox joins Safari on per-site caching, and the SDK reports `unsupported` on
Firefox without prompting or visiting.

