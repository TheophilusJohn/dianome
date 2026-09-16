# Phase 0 brief — storage-partitioning spike

You are working in the `dianome` repo (currently empty). Build the Phase 0 spike harness described below. Vanilla HTML/JS, no framework, no bundler, no styling beyond a readable system font and a results table. Do not deploy anything, do not run measurements, and do not write numbers into the results file — every cell in the results table stays empty until a human fills it.

## Goal

Determine whether a model chunk cached from site A can be served on site B without a network fetch, on Chrome, Firefox and Safari, and what the user must do for that to happen. Also collect the WebGPU adapter limits, storage quota and device memory the future planner will need.

## Layout to create

```
package.json                      name "dianome", private, pnpm workspaces ["packages/*"] (packages/ stays empty for now)
.gitignore                        node_modules, spikes/00-storage/cdn/chunks/test.bin
spikes/00-storage/
  README.md                       setup + deploy steps (below)
  cdn/                            → Cloudflare Pages project "dianome-cdn"
    index.html                    top-level opt-in page
    frame.html                    the iframe that owns the cache
    harness.js                    loaded by site-a and site-b from the cdn origin
    probe.js                      top-level probe: WebGPU limits, quota, device memory
    _headers                      caching + CORS rules
    chunks/.gitkeep               test.bin is generated, not committed
  site-a/index.html               → Pages project "dianome-spike-a"
  site-b/index.html               → Pages project "dianome-spike-b"
docs/spikes/00-storage-partitioning.md   results template
```

Use `https://dianome-cdn.pages.dev` as the CDN origin in the code, defined once as a constant at the top of `harness.js` and in the two site pages, so it can be changed in one place if the Pages project name is taken.

## Constraints that shape the code

- Cloudflare Pages rejects files over 25 MiB, so the test payload is 20 MB: `head -c 20971520 /dev/urandom > spikes/00-storage/cdn/chunks/test.bin`. Random so it cannot be compressed or deduped. The README records the command and tells the user to commit the SHA-256 into the results file.
- `_headers` must set on `/chunks/*`: `Cache-Control: public, max-age=31536000, immutable` and `Access-Control-Allow-Origin: *`. On `/harness.js` and `/probe.js`: `Access-Control-Allow-Origin: *`. No `X-Frame-Options` anywhere.
- The Storage Access API needs a user gesture inside the iframe, so the test buttons for T1–T3 live inside `frame.html`, not in the parent. The parent only listens and logs.
- The iframe tag in the site pages carries `allow="storage-access"`.
- WebGPU is probed from the top-level page, not the iframe.
- When the cache misses, fetch with `{cache: 'no-store'}` so the HTTP cache cannot masquerade as a Cache API hit. T4 is the only test that deliberately uses the HTTP cache.
- Timing uses `performance.now()` around the cache match / fetch only, not page load.

## frame.html

Owns a Cache API store named `dianome-spike` and the chunk URL `/chunks/test.bin` (same origin as the frame). Renders four buttons and a status line:

1. **T1 Load (partitioned)** — `caches.open` → `match`; on miss `fetch` (no-store) → `put`. Reports `source: 'cache' | 'network'`.
2. **T2 Load (Storage Access API)** — `document.hasStorageAccess()`; if false, `document.requestStorageAccess()` inside the click handler; then the same as T1 on the default `caches`. Report whether a prompt appeared cannot be detected programmatically, so the status line tells the human to note it.
3. **T3 Load (SAA handle, Chrome)** — `const h = await document.requestStorageAccess({all: true})`; use `h.caches` for open/match/put. If the call throws or `h.caches` is undefined, report the error text.
4. **T5 Fill until refused** — store additional 20 MB entries (built from `crypto.getRandomValues` in 65,536-byte pieces) under keys `/fill/<n>` until a `QuotaExceededError` or a 2 GB cap, then report the ceiling in MB and clear the fill entries.

Also handles a `probe` request from the parent: `navigator.storage.estimate()`, `navigator.storage.persist()`, `document.hasStorageAccess()`, and whether `document.requestStorageAccess` exists.

postMessage protocol (`targetOrigin` must be the parent's origin as reported by `document.referrer`'s origin; fall back to `'*'` only for the spike and say so in a comment):

```
frame → parent
{ type: 'chunk', test: 'T1'|'T2'|'T3', source: 'cache'|'network', ms, bytes }
{ type: 'fill',  ceilingMB, error }
{ type: 'probe', quota, usage, persisted, hasAccess, saaSupported, saaHandleSupported }
{ type: 'error', test, message }
parent → frame
{ type: 'probe' }
```

Do not transfer the 20 MB buffer to the parent; report `bytes` only. (The real SDK will transfer buffers; the spike measures cache behaviour, not transfer cost.)

## harness.js (loaded by site-a and site-b)

- Injects the iframe pointing at `<CDN>/frame.html` with `allow="storage-access"`.
- Renders a log table with columns: time, test, source, ms, notes. Every message from the frame becomes a row. Rows are appended, never overwritten, so a session's history is visible.
- Button **T4 HTTP cache**: `fetch(<CDN>/chunks/test.bin)` from the top level, then read `performance.getEntriesByName(url).at(-1).transferSize` and `.duration`; log `transferSize` (0 means served from cache) and duration. Use `{cache: 'default'}`.
- Button **Probe**: sends `{type:'probe'}` to the frame and also runs `probe.js` at top level; logs both results as rows.
- Button **Copy results as Markdown**: copies the log table as a Markdown table to the clipboard so it can be pasted into the results file.
- Shows the current origin and the CDN origin at the top of the page so screenshots are self-describing.

## probe.js

Exports an async function that returns an object with: `webgpu` (boolean), `maxBufferSize`, `maxStorageBufferBindingSize`, `maxComputeWorkgroupStorageSize`, adapter info (`vendor`, `architecture`, `device`, `description` — use `adapter.info` if present, else `await adapter.requestAdapterInfo()` if present), `deviceMemory` (`navigator.deviceMemory ?? null`), `hardwareConcurrency`, and `userAgent`. Never throws; missing values are `null`.

## cdn/index.html (top-level opt-in page)

One paragraph explaining this page exists so the browser records a first-party visit to the CDN origin (Chrome requires this before it will grant storage access to the embedded frame), and one button "Enable cross-site model cache" that loads the chunk into the same `dianome-spike` cache via the same code path as T1 (import the shared logic from a small `cache.js` module used by both `frame.html` and `index.html` rather than duplicating it). Logs `source` and `ms` on the page.

## site-a/index.html and site-b/index.html

Identical shells: a heading with the site name read from `location.hostname`, and `<script src="<CDN>/harness.js" type="module">`. Nothing else.

## docs/spikes/00-storage-partitioning.md

Write exactly this template, with empty cells:

```markdown
# Spike 00 — Storage partitioning

As of: YYYY-MM-DD · Hardware: <machine, GPU> · test.bin SHA-256: <hash> · PSL check: pages.dev present (Y/N) · Chrome partition key observed on site A (Y/N)

## Cross-site cache (load on A first, then run on B)

| Browser (version) | T1 partitioned | T2 SAA | T3 SAA handle | T4 HTTP cache | Prompt shown? | Gestures needed | Persists after restart |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Chrome | | | | | | | |
| Firefox | | | n/a | | | | |
| Safari | | | n/a | | | | |

Cells for T1–T3: `cache <ms>` or `network <ms>` or `error: <message>`. T4: `transferSize=<bytes> <ms>`.

## Storage and device

| Browser | quota (GB) | persist() | T5 ceiling (GB) | maxBufferSize | maxStorageBufferBindingSize | adapter | deviceMemory |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Chrome | | | | | | | |
| Firefox | | | | | | | |
| Safari | | | | | | | |

## Decision

Green / Yellow / Red: <one line>
Consequence for Phase 3: <one line>
Planner inputs available on every browser: <list>

Green = cache hit on B after a one-time opt-in plus one click on at least two of three browsers.
Yellow = hit only with a prompt on every site. Red = no hit anywhere. Yellow and Red both reframe the CDN to progressive streaming plus per-site caching.
```

## spikes/00-storage/README.md

Document, in order:

1. Generate the payload and record its hash (`head -c 20971520 /dev/urandom > cdn/chunks/test.bin && sha256sum cdn/chunks/test.bin`).
2. Deploy the three Pages projects with Wrangler, e.g. `npx wrangler pages deploy cdn --project-name dianome-cdn` (and `site-a` / `site-b`). Note that project names are global; if taken, change the CDN constant.
3. Day-one checks before any measurement:
   - `curl -s https://publicsuffix.org/list/public_suffix_list.dat | grep -n 'pages.dev'` — must be present, otherwise all three origins are one site and results are meaningless.
   - On site A in Chrome DevTools → Application → Storage, confirm the frame's storage shows a partition key.
   - In the frame's console: `typeof document.requestStorageAccess`, and on a click `document.requestStorageAccess({all: true})` — record whether it returns a handle with `.caches`, throws, or prompts, plus the Chrome version.
4. The run order per browser: visit `dianome-cdn.pages.dev` top-level and click Enable; open site A, run T1; open site B, run T1, T2, T3, T4, Probe, T5; close and reopen the browser, rerun T2/T3 on B; paste the copied Markdown into the results file.

## Definition of done

- `python3 -m http.server` (or any static server) in `cdn/` renders `index.html` and `frame.html` without console errors, and `site-a/index.html` opened via a second static server on another port embeds the frame and logs a T1 row.
- No file in the repo contains a measured number.
- `git status` shows `test.bin` ignored.
