# dianome

Progressive, verified, cached model weights in the browser. `load(modelId)` streams a model's chunks from
`cdn.dianome.dev`, checks every chunk's SHA-256, caches them per site everywhere (Cache API on your origin) and
across sites on Chrome after a one-time opt-in, and posts one anonymous load report.

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

// Generate: local (whole model here), split at a planner-chosen N, or server, chosen per device and network.
const r = await d.run("qwen2.5-0.5b-instruct", {
  messages: [{ role: "user", content: "Why is the sky blue?" }],   // or prompt: "…" (sent as-is)
  variant: "q4", policy: { prefer: "cost" }, maxTokens: 128, sampling: { temperature: 0 },
  onToken: (text) => process.stdout.write(text), onPlan: (p) => console.log(p.mode, p.N, p.reasons),
});
r.text; r.tokens; r.mode; r.N; r.plan.candidates; r.timings; r.serverBusyMs; r.costEstimate; r.privacy;
const plan = await d.planRun("qwen2.5-0.5b-instruct", { variant: "q4", policy: { prefer: "latency" } });   // plan only

// run() lives in the `dianome/run` entry (loaded on first use) and needs the optional peer `dianome-runtime`;
// it imports that package's WebGPU entry only when the plan puts blocks on this device.
```

`Entry.bytes` is a view into a chunk buffer whenever the entry fits in one chunk (no copy); an entry that spans
chunks is concatenated once. Views alias memory shared with other entries: treat them as read-only, and copy
(`bytes.slice()`) before transferring a buffer to a worker. Entries and quantized parts (`weights`, `scales`,
`zeros`) start at 256-byte-aligned offsets so they can go straight into GPU buffers.

Options: `new Dianome({ api, cdn, telemetry, cache: "auto" | "per-site" | "none", concurrency, retries })`.

Files artifacts (ONNX for Transformers.js, MLC shards for WebLLM) load the same way; each file is one group with
one raw entry. See `dianome/transformersjs` and `dianome/webllm` for the adapters.

## Shared (cross-site) cache and what to tell your users

By default a model is cached per site: a second site on a different domain downloads it again. Chrome can share
one cache across sites through a frame on `cdn.dianome.dev` after the user opts in. Ask from a click handler:

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
`scripts/publish-frame.sh`).

Suggested wording for your users: *"Enable the shared model cache to keep downloaded models on this device and
reuse them on other sites that use Dianome. Nothing about you is stored; only model files are shared, and you
can clear them from your browser's site data for cdn.dianome.dev at any time."*

## Browser support (measured in Phase 0 and corrected 2026-09-17, `docs/spikes/00-storage-partitioning.md`)

| Browser | Per-site cache | Cross-site cache | Prompt |
| --- | --- | --- | --- |
| Chrome (cookies allowed) | yes | yes, via `requestStorageAccess({all: true})` | none after the one-time visit |
| Chrome (third-party cookies blocked) | yes | yes | one "Allow embedded content?" prompt per site, remembered |
| Firefox | yes | no (a Storage Access grant leaves the Cache API partitioned; the SDK reports `unsupported`) | none after the one-time visit |
| Safari / iOS | yes | no (same; the SDK reports `unsupported`) | one, once |

Detection is capability-based, not by browser name: after a grant the frame probes for a marker the opt-in page
wrote into the CDN origin's Cache API, so a browser that later unpartitions storage on grant is picked up without
an SDK change.

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

## Keys

`POST https://api.dianome.dev/v1/keys` returns a `dk_live_…` key once (no accounts: the key is the identity; keep it).
`new Dianome({ apiKey })` sends it as `Authorization: Bearer` when minting a split session and when posting telemetry.
Session reports then carry the key's `id` (never the key) and `/dashboard` on dianome.dev shows what the key consumed.
Keyed apps mint sessions from any origin; without a key only the demo origins can. Revoke on the dashboard.
Self-hosting: `new Dianome({ split: { servers: { "<model>": { ws: "wss://your-host", token } } } })` skips the API's session mint; see `server/README.md`.

## Development

```sh
pnpm --filter dianome test        # vitest unit tests (node)
pnpm --filter dianome typecheck
pnpm --filter dianome build       # dist/ (esbuild + tsc declarations); regenerates nothing, checks manifest.gen.ts is current
pnpm --filter dianome size        # gzipped budget: 25 KB for the main entry
pnpm --filter dianome gen:types   # src/manifest.gen.ts from schemas/manifest.v1.json
pnpm --filter dianome pack        # the tarball Theo publishes
```
