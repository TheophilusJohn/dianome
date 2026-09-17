# demo-site

One Vite site, three pages:

- `/` — loads `qwen2.5-0.5b-instruct` (q4) through the SDK with per-group progress, bytes/sec, the source
  breakdown, and an "Enable shared cache" button that calls `enableCrossSiteCache()` from its click handler and
  shows the returned state (the visit link on `needs-visit`, the in-frame button on `needs-click`).
- `/?diag=1` — embeds the frame's diagnostic page (`frame/v1/diag.html`, the Phase 0 T2 harness on the production
  origins) visibly instead of the SDK's hidden frame; its three buttons print `hasStorageAccess()`, the result of a
  plain `requestStorageAccess()` followed by the `dianome-v1` keys, and the keys without a request.
- `/transformersjs.html` — `dianome/transformersjs` adapter with the ONNX artifact.
- `/webllm.html` — `dianome/webllm` adapter with the MLC artifact (needs WebGPU).

## Local

```sh
node scripts/serve-store.mjs                      # repo root: ./store as api+cdn on :8788 (+ the built cache frame)
pnpm --filter cache-frame build                   # so /frame/v1/ is served locally
pnpm --filter demo-site dev                       # http://localhost:5175/?api=http://localhost:8788&cdn=http://localhost:8788
```

## Deploy (Theo)

```sh
pnpm --filter demo-site build
wrangler pages deploy apps/demo-site/dist --project-name dianome-demo-a
wrangler pages deploy apps/demo-site/dist --project-name dianome-demo-b
```

**The two sites must stay on `pages.dev`, not on `dianome.dev` subdomains.** Storage partitioning keys on the
registrable domain (site), so `a.dianome.dev` and `b.dianome.dev` are one site: a "cross-site" hit between them
would really be the per-site cache, and the demo would be claiming something it does not show. `pages.dev` is on
the Public Suffix List, so `dianome-demo-a.pages.dev` and `dianome-demo-b.pages.dev` are two sites.
