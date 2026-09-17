# Phase 3 notes — SDK, progressive loading, cross-site cache

Measured numbers only. Every figure names the command or page that produced it; nothing is typed by hand.
Cells left empty are for Theo to fill after deploying the frame and the two demo sites.

## Bundle sizes

`pnpm --filter dianome build && pnpm --filter dianome size` on 2026-09-16 (esbuild 0.25.12, ESM with code splitting; the
shared chunk is counted with every entry that imports it; budget for the main entry is 25 KB gzipped):

```
ok   index.js             41413 bytes   13163 gzipped  (index.js, chunk-7AQRLEK4.js)
ok   transformersjs.js    12479 bytes    4432 gzipped  (transformersjs.js, chunk-7AQRLEK4.js)
ok   webllm.js            12826 bytes    4571 gzipped  (webllm.js, chunk-7AQRLEK4.js)
```

`pnpm --filter cache-frame build`: `dist/frame.js 13428 bytes`, `dist/index.html 850 bytes`, `dist/optin.html 2453 bytes`.

## Per-site cache: first vs second load per browser

`pnpm --filter dianome test:e2e` (Playwright 1.63.0; chromium headless shell 153.0.8010.12, firefox 155.0, webkit build
2359) against the local e2e server: a synthetic q4 model of 4,951,040 bytes in 28 chunks of 256 KB, served from
memory on 127.0.0.1, so the "first" column is verification and Cache API write time, not network. Values are the
`timing` annotations in `test-results/e2e.json` from the run at 2026-09-17T01:05:29Z (10 passed, 2 skipped by design):

| engine | first load (network) ms | second load (per-site cache, same document) ms | third load (after reload) ms |
| --- | --- | --- | --- |
| chromium | 72 | 13 | 13 |
| firefox | 75 | 23 | 28 |
| webkit (ephemeral context) | 105 | 23 | n/a: Playwright's ephemeral WebKit context drops CacheStorage on navigation |
| webkit (persistent context) | 75 | 23 | (second load ran in a fresh document after `page.goto`) |

Second loads made 0 chunk requests on every engine; the corrupt-cache test refetched exactly 1 chunk on every engine.

## Demo site: q4 load through the SDK (local servers)

`apps/demo-site` (`pnpm --filter demo-site dev`, port 5175) with `?api=http://localhost:8788&cdn=http://localhost:8788`
pointing at `node scripts/serve-store.mjs` over ./store, driven by Playwright Chromium (headless shell 153) in a fresh
profile on 2026-09-16; the page's own status text:

| load | result |
| --- | --- |
| 1 | `done in 0.93 s · 323.9 MB · 42 chunks · 358.2 MB/s` · source network · cache hits 0/42 · verify 156 ms · telemetry HTTP 202 |
| 2 (same document) | `done in 0.27 s · 323.9 MB · 42 chunks · 1268.9 MB/s` · source per-site-cache · cache hits 42/42 · verify 138 ms · 0 chunk requests |

Report sent for load 1: `{"schema":2,"model":"qwen2.5-0.5b-instruct","variant":"q4","bytes":323893760,"chunks":42,"ms":904,"source":"network","cache_hits":0,"browser":"chrome","webgpu":false,"bytes_per_second":358202384,"verify_ms":156,"transfer_ms":0,"cache_mode":"per-site","quota_bytes":10737418240}`
(`webgpu:false` because the headless shell exposes no `navigator.gpu` on that profile; the full Chromium build reports an adapter, see below).

## SHA-256 verify throughput and postMessage transfer cost

Demo page "Measure" button (`crypto.subtle.digest` over a random 8 MiB buffer, mean of 5), three runs in Playwright
Chromium headless on Apple Silicon: `3.6 ms/chunk → 2331 MB/s`, `3.2 ms/chunk → 2605 MB/s`, `3.5 ms/chunk → 2424 MB/s`.
Whole-load verify time from the reports above: 156 ms for 42 chunks (323.9 MB), i.e. 3.7 ms per 8 MiB chunk in
the real load loop.

### Local end-to-end run through the frame (same-site, so it measures the code path and the transfer cost, not partition crossing)

Playwright Chromium headless shell 153, fresh persistent profile, `context.grantPermissions(["storage-access"])` for
the frame origin (headless Chromium cannot show the storage-access prompt; without the grant the in-frame click
rejects with `NotAllowedError` and the SDK reports `needs-visit`). Flow: optin.html writes the marker → return →
"Enable shared cache" → `granted` on the silent attempt → two loads → "Measure" → fresh document with `cache: "auto"`.
All numbers are the page's status text on 2026-09-16:

| step | result |
| --- | --- |
| enable | `state: granted`, `dianome:crosssite = granted` |
| load 1 | `done in 1.13 s · 323.9 MB · 42 chunks · 293.2 MB/s` · source network · cache mode cross-site · verify 155 ms · transfer 6185 ms (sum over chunks, 6 in flight) · parent chunk requests 0, frame chunk requests 42 |
| load 2 | `done in 0.34 s · 323.9 MB · 42 chunks · 1008.1 MB/s` · source cross-site-cache · cache hits 42/42 · verify 169 ms · transfer 1637 ms · 0 chunk requests anywhere |
| fresh document, auto | `done in 0.31 s · 323.9 MB · 42 chunks · 1127.0 MB/s` · source cross-site-cache · cache hits 42/42 (silent reconnect, no click) |

Per-op timings from the parent (round trip) and the frame (its own `ms`), 84 `fetch` ops = 42 per load, first run
(before the transfer_ms fix; the "transfer" figures in the table above were sums of whole-op round trips, which is
what the fix below removed):

```
frame hello:  n=1  mean 0.7 ms (frame-side 0.2 ms)  max 0.7 ms
frame grant:  n=1  mean 2.5 ms (frame-side 2.4 ms)  max 2.5 ms
frame fetch:  n=84 mean 93.1 ms (frame-side 91.4 ms) max 216.2 ms
frame status: n=3  mean 0.9 ms (frame-side 0.5 ms)  max 1.2 ms
```

**transfer_ms fix (2026-09-16, uncommitted at the time of writing):** the report used to wrap the frame's whole
`fetch` op (including its network fetch and cache write) and sum across the 6 concurrent gets, which is how a 15 s
load could report `transfer_ms: 82376`. It now measures only the postMessage hop: both sides stamp `sentAt`
(`performance.timeOrigin + performance.now()`, comparable across documents) immediately before posting, the
receiver subtracts on arrival, and the report carries the per-chunk median. Same flow re-run after the fix, page
status text and "Measure" output:

| load | hop median per chunk (42 samples) | frame `fetch` op round trip |
| --- | --- | --- |
| 1 (network via frame) | 0.2 ms | mean 74.9 ms (frame-side 72.8 ms), max 203.0 ms, over both loads |
| 2 (cross-site cache) | 0.2 ms | |
| fresh document, auto | 0.1 ms | |

Mean frame → parent hop over the 84 fetches: 0.66 ms; mean parent → frame request hop: 1.44 ms. The buffer is
transferred (not copied), so the hop is independent of the 8 MB payload; what the earlier figures were measuring was
the frame's fetch and Cache API work.

### Firefox and Chromium cross-site locally (page on 127.0.0.1, frame on localhost: two sites)

`node packages/sdk/test-results/ff-xsite.mjs <firefox|chromium>` (Playwright Firefox 155.0 and Chromium headless
shell 153, fresh profiles, no permission pre-grant), page `http://127.0.0.1:5175/?api=http://localhost:8788&cdn=http://localhost:8788`,
on 2026-09-16 after the frame/demo feedback changes. What the demo rendered:

| browser | opt-in visit first | silent attempt | after the in-frame click | persisted |
| --- | --- | --- | --- | --- |
| Firefox | no | `needs-visit` via `firefox-globals`: `requestStorageAccess()` resolved without a gesture, marker not visible | (no click needed) | null |
| Firefox | yes | `granted` via `firefox-globals`, no click, no prompt | — | granted |
| Chromium | no | `needs-click` (`NotAllowedError: requestStorageAccess not allowed`) | `needs-visit` (`… (permission: prompt)`) | null |
| Chromium | yes | `needs-click` | `needs-visit` (`… (permission: prompt)`): headless Chromium cannot show the prompt, see the pre-granted run above | null |

So in Firefox the top-level visit is what matters: without it the grant resolves but reaches partitioned storage
(marker probe → `needs-visit`); with it the silent path succeeds and the frame button is never needed. The frame
now calls the plain `requestStorageAccess()` on Firefox (`{all: true}` elsewhere), logs
`[dianome frame] requestStorageAccess…` and `grant → <state>` lines to its console, and posts progress notes
(`waiting-click`, `clicked`, `requesting`) that the demo renders while a permission prompt may be pending.

WebGPU note: `navigator.gpu` is absent on `about:blank` (not a secure context) and present on `http://localhost`
in Playwright Chromium headless (`maxBufferSize` 1073741824), headed (4294967292) and Google Chrome 152 headed
(4294967292); probed with a one-off script on 2026-09-16.

## Cross-site cache: manual checklist (deployed demo sites)

Procedure (Phase 0 harness pattern): open site A (`dianome-demo-a.pages.dev`), click "Enable shared cache",
follow the visit link if shown, click the in-frame button, load the model. Open site B
(`dianome-demo-b.pages.dev`) in the same browser profile, click "Enable shared cache", click the in-frame
button if shown, load the model. Record the source breakdown the page prints.

| Browser (version) | A: enable result | A: load source | B: enable result | B: load source | B: ms | Prompt shown? | Clicks needed on B | Persists after restart |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Chrome | | | | | | | | |
| Chrome (3PC blocked) | | | | | | | | |
| Firefox | | | | | | | | |
| Safari | | | | | | | | |

Expected from Phase 0: Chrome and Firefox show `cross-site-cache` on B; Safari reports `unsupported` and loads
`network` on B (per-site only).

## Adapters: second-run network counts

### Transformers.js (`dianome/transformersjs`)

`apps/demo-site/transformersjs.html?api=http://localhost:8788&cdn=http://localhost:8788` driven by Playwright Chromium
(headless, one persistent profile, two runs) against `node scripts/serve-store.mjs` on ./store, on 2026-09-16.
Pipeline `text-generation` on `onnx-community/Qwen2.5-0.5B-Instruct`, dtype q4, wasm backend, artifact
`qwen2.5-0.5b-instruct-onnx`. Counts are from the page's Resource Timing and from Playwright's request listener
(both agreed); prompt "The capital of France is", 8 greedy tokens:

| run | chunk requests | huggingface.co requests | files served by the adapter | pipeline ready | generated |
| --- | --- | --- | --- | --- | --- |
| 1 | 98 | 0 | 5 (config.json, generation_config.json, tokenizer.json, tokenizer_config.json, onnx/model_q4.onnx), all `network` | 8.5 s | "…is Paris. It has a population of " |
| 2 | 0 | 0 | 5, all `per-site-cache` | 5.8 s | identical |

Dianome cache after run 1: per-site, 98 chunks, 793.2 MB.

### WebLLM (`dianome/webllm`)

`apps/demo-site/webllm.html?api=http://localhost:8788&cdn=http://localhost:8788` driven by Playwright Chromium
(headless shell 153, `--enable-unsafe-webgpu`, one persistent profile, two runs) on 2026-09-16. Artifact
`qwen2.5-0.5b-instruct-mlc` (15 files, 289.7 MB after skipping README/.gitattributes), WebLLM 0.2.85 model id
`Qwen2.5-0.5B-Instruct-q4f16_1-MLC` with `model` pointed at `http://localhost:8788/mlc/qwen2.5-0.5b-instruct-mlc/`:

| run | pre-warm | chunk requests | requests to the model URL | WebLLM |
| --- | --- | --- | --- | --- |
| 1 | 15 files streamed, 1.0 s, source network, `hasModelInCache: true` | 41 | 0 | "Loading model from cache[8/8]: 266MB loaded. 100% completed, 1 secs elapsed." |
| 2 | 0 files streamed, 15 already cached | 0 | 0 | same |

The only external request besides the manifest was the model library
`raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/main/web-llm-models/v0_2_84/base/Qwen2-0.5B-Instruct-q4f16_1_cs1k-webgpu.wasm`
(not part of the artifact). In the headless shell the engine then failed to compile a compute shader
(`[Invalid ShaderModule (unlabeled)] … entryPoint: "index_kernel"`), a WebGPU limitation of that build, not a
caching issue. Headed Playwright Chromium 153 (real Metal adapter), one run in a fresh profile: pre-warm 1.1 s,
`hasModelInCache: true`, "Finish loading on WebGPU - apple", engine ready in 1.0 s, reply to "Say hello in five
words." was "Hello, how can I help you today?", requests to the model URL: 0, chunk requests 41.
