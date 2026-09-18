# Dianome

Dianome delivers model weights to the browser: a model is packed once into 8 MB content-addressed chunks with a
two-level manifest, served from `cdn.dianome.dev` through Cloudflare's edge cache, and loaded by the `dianome` SDK
progressively, layer group by layer group, with every chunk verified by SHA-256. Chunks are cached per site in
the Cache API on every browser, and on Chrome they are shared across sites through a frame on the CDN origin after
a one-time opt-in, so a second site that uses the same model does not download it again. Every load posts one
anonymous report that feeds a public stats endpoint.

## Measured

| what | number | where it comes from |
| --- | --- | --- |
| Qwen2.5-0.5B-Instruct q4 (323.9 MB, 42 chunks), network load | 15.6 s (Phase 2, Chrome 152) to 22.91 s (Phase 3 demo site A) | `docs/phase-2-notes.md`, `docs/phase-3-notes.md` |
| Same model on a second site, Chrome cross-site cache | 0.81 s, postMessage hop median 21.7 ms per chunk | `docs/phase-3-notes.md`, manual table |
| Firefox and Safari | per-site cache only: second load on the same site from cache, other sites download again | `docs/phase-3-notes.md` |
| SDK main entry | 14.0 KB gzipped (budget 25 KB) | `pnpm --filter dianome size` |

## Install and quickstart

```sh
npm install dianome
```

```ts
import { Dianome } from "dianome";

const d = new Dianome();                       // api/cdn default to *.dianome.dev

// Progressive: groups arrive in manifest order as they complete (embed, layer.0 … final_norm, lm_head).
for await (const group of d.stream("qwen2.5-0.5b-instruct", { variant: "q4" })) {
  for (const [name, entry] of group.entries) upload(name, entry.bytes, entry.parts);   // Uint8Array views
}

// Or everything at once, with progress.
const model = await d.load("qwen2.5-0.5b-instruct", { variant: "q4", onProgress: (p) => console.log(p.bytesDone / p.bytesTotal, p.source) });
model.entry("model.layers.7.self_attn.q_proj.weight").parts?.scales;
```

```ts
// Phase 5b: generate. run() chooses, per device and network, local (whole model in the browser), split at a
// planner-chosen N (blocks 0..N-1 here, the rest on the split server), or server (token ids sent).
const r = await d.run("qwen2.5-0.5b-instruct", { messages: [{ role: "user", content: "Why is the sky blue?" }], policy: { prefer: "cost" }, onToken: (t) => process.stdout.write(t) });
r.mode; r.N; r.plan.candidates;   // every candidate's estimated ms/token, server share and cost, privacy band
```

The full API, the cross-site opt-in flow, the quota rules and the adapters for Transformers.js and WebLLM are in
[`packages/sdk/README.md`](packages/sdk/README.md); the WebGPU runtime, the split session and the planner in
[`packages/runtime/README.md`](packages/runtime/README.md); the split server in [`server/README.md`](server/README.md).
What leaves the device at a split point N is the hidden state at boundary N, and a server can recover input tokens
from it at the rate measured in `docs/phase-4-notes.md` (the privacy band); split inference does not hide the prompt
from the server, and local mode sends nothing.

## Links

- Demo sites (two registrable domains, so the second one shows the shared cache):
  [dianome-demo-a.pages.dev](https://dianome-demo-a.pages.dev) and [dianome-demo-b.pages.dev](https://dianome-demo-b.pages.dev)
- Public load stats: [api.dianome.dev/v1/stats/loads](https://api.dianome.dev/v1/stats/loads)
- npm: [dianome](https://www.npmjs.com/package/dianome)
- Docs:
  - [`docs/briefs/`](docs/briefs) — the brief each phase was built from (storage partitioning, ingest, edge, SDK)
  - [`docs/spikes/`](docs/spikes) — Phase 0 storage-partitioning spike, with its 2026-09-17 correction
  - [`docs/phase-1-notes.md`](docs/phase-1-notes.md), [`docs/phase-2-notes.md`](docs/phase-2-notes.md),
    [`docs/phase-3-notes.md`](docs/phase-3-notes.md), [`docs/phase-4-notes.md`](docs/phase-4-notes.md),
    [`docs/phase-5a-notes.md`](docs/phase-5a-notes.md), [`docs/phase-5b-notes.md`](docs/phase-5b-notes.md) — measured
    numbers only, each with the command or page that produced it
- Split demo (slider over N, live estimates and measurements): `dianome-demo-split.pages.dev` once deployed;
  locally `pnpm --filter split-demo dev` (see `apps/split-demo/vite.config.ts` for the `?api=&cdn=&split=` overrides)

## Browser support

| Browser | Per-site cache | Cross-site cache |
| --- | --- | --- |
| Chrome | yes | yes, via the `requestStorageAccess({all: true})` handle, after a one-time visit to `cdn.dianome.dev` (one remembered prompt when third-party cookies are blocked) |
| Firefox | yes | no |
| Safari / iOS | yes | no |

**Phase 0 correction (2026-09-17).** The Phase 0 spike recorded a Firefox cross-site cache hit through the plain
`requestStorageAccess()` grant and concluded "Green: Chrome and Firefox". That was wrong: on the production
origins, Firefox's grant resolves and `hasStorageAccess()` is true, but the Cache API the frame sees is still the
partitioned copy (0 keys, the top-level marker invisible). The Phase 0 hit was a confounded partitioned copy. The
decision is now Yellow: cross-site is Chrome-only, and Firefox joins Safari on per-site caching. The details are
in the correction section of [`docs/spikes/00-storage-partitioning.md`](docs/spikes/00-storage-partitioning.md).

Detection in the SDK is by capability, not browser name: after a grant the frame probes for a marker that the
opt-in page wrote into the CDN origin's Cache API, so a browser that later unpartitions storage on grant is picked
up without an SDK change.

## Repository

```
packages/sdk          the npm package "dianome"
packages/cache-frame  the cross-site frame (index.html, frame.js, optin.html, diag.html) served at cdn.dianome.dev/frame/v1/
packages/worker       api.dianome.dev: manifests, telemetry ingest, aggregated stats
ingest/               Python: pack a Hugging Face model (or any directory) into chunks + manifest, upload to R2
apps/site             dianome.dev: home, /summarize (Phase 7 demo), /stats (load + session dashboards), /writeup
apps/demo-site        the demo, deployed twice; apps/load-test, apps/manifest-browser
schemas/              manifest v1, telemetry v1 and v2 (JSON Schema)
scripts/              serve-store.mjs (local api+cdn), publish-frame.sh, check-edge.sh
```

```sh
pnpm install && pnpm -r typecheck && pnpm test          # worker + sdk unit tests
pnpm --filter dianome test:e2e                          # Playwright, chromium + firefox + webkit
```
