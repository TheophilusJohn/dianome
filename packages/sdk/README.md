# dianome

Progressive, verified, cached model weights in the browser. `load(modelId)` streams a model's chunks from
`cdn.dianome.dev`, checks every chunk's SHA-256, caches them per site everywhere (Cache API on your origin) and
across sites on Chrome and Firefox after a one-time opt-in, and posts one anonymous load report.

ESM only, types included, zero runtime dependencies.

## Install

```sh
npm install dianome
```

## Quickstart

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
model.summary;                                 // bytes, ms, source, cacheHits, cacheMode, the report sent
```

`Entry.bytes` is a view into a chunk buffer whenever the entry fits in one chunk (no copy); an entry that spans
chunks is concatenated once. Views alias memory shared with other entries: treat them as read-only, and copy
(`bytes.slice()`) before transferring a buffer to a worker. Entries and quantized parts (`weights`, `scales`,
`zeros`) start at 256-byte-aligned offsets so they can go straight into GPU buffers.

Options: `new Dianome({ api, cdn, telemetry, cache: "auto" | "per-site" | "none", concurrency, retries })`.

Files artifacts (ONNX for Transformers.js, MLC shards for WebLLM) load the same way; each file is one group with
one raw entry. See `dianome/transformersjs` and `dianome/webllm` for the adapters.

## Shared (cross-site) cache and what to tell your users

By default a model is cached per site: a second site on a different domain downloads it again. Chrome and
Firefox can share one cache across sites through a frame on `cdn.dianome.dev` after the user opts in. Ask from a
click handler:

```ts
button.onclick = async () => {
  const r = await d.enableCrossSiteCache();
  // r.state: "granted" | "needs-click" | "needs-visit" | "denied" | "unsupported"
  if (r.state === "needs-click") await r.mount(slotElement);   // shows the frame's own "Enable shared model cache" button
  if (r.state === "needs-visit") link.href = r.visitUrl;        // one-time top-level visit to cdn.dianome.dev
};
```

Browsers only grant storage access to a frame the user has clicked inside, so the first grant on a site needs
one click on the button the SDK mounts for you (`needs-click`); Chrome additionally needs one earlier top-level
visit to `cdn.dianome.dev` (`needs-visit`). Both outcomes persist, so later visits get the shared cache silently
with `cache: "auto"`.

If your page sets `Cross-Origin-Embedder-Policy`, the frame must carry COEP too; the deployed frame does (see
`scripts/publish-frame.sh`). The frame is never needed on Safari, where the SDK reports `unsupported` without
loading it.

Suggested wording for your users: *"Enable the shared model cache to keep downloaded models on this device and
reuse them on other sites that use Dianome. Nothing about you is stored; only model files are shared, and you
can clear them from your browser's site data for cdn.dianome.dev at any time."*

## Browser support (measured in Phase 0, `docs/spikes/00-storage-partitioning.md`)

| Browser | Per-site cache | Cross-site cache | Prompt |
| --- | --- | --- | --- |
| Chrome (cookies allowed) | yes | yes, via `requestStorageAccess({all: true})` | none after the one-time visit |
| Chrome (third-party cookies blocked) | yes | yes | one "Allow embedded content?" prompt per site, remembered |
| Firefox | yes | yes, via `requestStorageAccess()` | none after the one-time visit |
| Safari / iOS | yes | no (storage stays partitioned; the SDK reports `unsupported`) | n/a |

## Quota

Third-party storage is capped around 8–11 GB in every browser regardless of the first-party quota; Chrome
Incognito caps at roughly 900 MB in memory and `navigator.storage.estimate()` does not warn. When a write hits
`QuotaExceededError` the SDK evicts least-recently-used chunks of other models, and if that is not enough it
turns caching off for the session and keeps streaming. A load never fails because of quota; the report says
`cache_mode: "none"` when that happened.

## Telemetry

One JSON report per completed load (`schemas/telemetry.v2.json`): model, variant, bytes, chunk counts, wall
time, source breakdown, browser family, WebGPU limits, storage quota. No IP address, no user-agent string,
nothing on failure or abort. `new Dianome({ telemetry: false })` sends nothing.

## Development

```sh
pnpm --filter dianome test        # vitest unit tests (node)
pnpm --filter dianome typecheck
pnpm --filter dianome build       # dist/ (esbuild + tsc declarations); regenerates nothing, checks manifest.gen.ts is current
pnpm --filter dianome size        # gzipped budget: 25 KB for the main entry
pnpm --filter dianome gen:types   # src/manifest.gen.ts from schemas/manifest.v1.json
pnpm --filter dianome pack        # the tarball Theo publishes
```
