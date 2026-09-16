# Phase 2 brief — edge delivery

You are working in the `dianome` repo. Phases 0 and 1 are done: chunks and manifests for `qwen2.5-0.5b-instruct` (fp16, q8, q4) and `qwen2.5-0.5b-instruct-onnx` are in the R2 bucket `dianome` under `chunks/<sha256>` and `manifests/<id>/{latest.json,<sha256>.json}`, with `Cache-Control: public, max-age=31536000, immutable` set as object metadata on chunks. Read `docs/briefs/01-ingest.md` and `docs/phase-1-notes.md` for the store layout.

Everything in this phase is TypeScript. Do not deploy anything unless the task says so, and never commit secrets: any token goes in `.dev.vars` (gitignored) locally and in Worker secrets in production.

## Goal

Chunk bytes are served straight from R2 through a custom domain with Cloudflare's edge cache in front and no Worker on the hot path. A Worker handles only the small, logic-bearing endpoints: manifests, telemetry ingest, and aggregated stats. A public page shows load times by region. This design is deliberate: a 7B load is ~500 chunk requests, and putting a Worker in front of them would exhaust the free tier in a day.

## Architecture

```
cdn.dianome.dev   → R2 custom domain on bucket `dianome` (chunks + manifests, cached at the edge)
api.dianome.dev   → Worker: /v1/models, /v1/manifest, /v1/telemetry, /v1/stats
dianome.dev       → Pages: landing later; for now /stats is the load-time dashboard (apps/load-dashboard)
```

The Worker reads manifests through an R2 binding, never through the S3 API, so it needs no keys.

## Split of work

**Theo does (dashboard, cannot be scripted):**
1. R2 → bucket `dianome` → Settings → Custom Domains → Connect `cdn.dianome.dev`. This creates the DNS record and enables edge caching for the bucket.
2. R2 → bucket `dianome` → Settings → CORS policy → paste the JSON in the "Bucket CORS" section below.
3. Rules → Transform Rules → Modify Response Header: for hostname `cdn.dianome.dev`, set static header `Timing-Allow-Origin: *`. R2 cannot serve custom headers from object metadata, and Phase 0 showed `transferSize` is 0 without it.
4. Caching → Cache Rules: for hostname `cdn.dianome.dev`, Eligible for cache, Edge TTL = use origin `cache-control` (bypass if absent). Without this Cloudflare skips extensionless paths like `chunks/<sha256>` and every request reads R2 (`cf-cache-status: DYNAMIC`); with it the second GET is `HIT`.
5. After CC finishes: `pnpm --filter worker deploy`, then verify with `scripts/check-edge.sh`.

**Status at hand-off:** steps 1–4 are done and verified: `curl` shows 206 on a Range request, `access-control-allow-origin: *`, `timing-allow-origin: *`, `cache-control: … immutable`, and `cf-cache-status` MISS then HIT on consecutive GETs of a chunk.

**CC does:** everything below.

## Bucket CORS (Theo pastes this; CC also commits it as `infra/r2-cors.json` for the record)

```json
[
  {
    "AllowedOrigins": ["*"],
    "AllowedMethods": ["GET", "HEAD"],
    "AllowedHeaders": ["Range", "If-None-Match", "If-Range", "Content-Type"],
    "ExposeHeaders": ["ETag", "Content-Length", "Content-Range", "Accept-Ranges", "Cache-Control"],
    "MaxAgeSeconds": 86400
  }
]
```

## Worker: `packages/worker`

Wrangler project, TypeScript, no framework (Hono is acceptable if you want routing; nothing heavier). Bindings in `wrangler.toml`:
- `STORE`: R2 bucket `dianome`.
- `TELEMETRY`: Analytics Engine dataset `dianome_loads`.
- `STATS_CACHE`: KV namespace, for caching aggregated stats (create with `wrangler kv namespace create`; put the ids in wrangler.toml).
- Secrets: `CF_ACCOUNT_ID`, `CF_ANALYTICS_TOKEN` (an API token with Account Analytics Read, for the Analytics Engine SQL API). Local values in `.dev.vars`.
- Route: `api.dianome.dev/*` on zone `dianome.dev`.

Endpoints (all JSON, all with `Access-Control-Allow-Origin: *`, OPTIONS preflight handled):

| Method | Path | Behaviour |
| --- | --- | --- |
| GET | `/v1/models` | Lists manifest ids by listing `manifests/` prefixes in R2. `Cache-Control: public, max-age=60`. |
| GET | `/v1/models/:id/manifest` | Streams `manifests/<id>/latest.json` from R2. Passes through R2's ETag; honours `If-None-Match` with 304. `Cache-Control: public, max-age=60`. Adds header `X-Dianome-Manifest-Sha` with the immutable manifest's hash (read from the manifest's own `sha256` field if present, else compute canonical hash — check what Phase 1 wrote). |
| GET | `/v1/models/:id/manifest/:sha` | Streams the immutable manifest. `Cache-Control: public, max-age=31536000, immutable`. |
| POST | `/v1/telemetry/load` | Accepts one load report (schema below), validates, writes one Analytics Engine data point. Returns 202 with empty body. Rejects bodies over 4 KB and unknown fields. Never stores IP or user agent string. |
| GET | `/v1/stats/loads` | Aggregated load stats for the dashboard (shape below). Computed via the Analytics Engine SQL API, cached in KV for 5 minutes. |
| GET | `/healthz` | `{ ok: true, version: <git sha baked at build> }`. |

Load report schema (`schemas/telemetry.v1.json`, and a TypeScript type generated or hand-mirrored from it):

```json
{
  "schema": 1,
  "model": "qwen2.5-0.5b-instruct",
  "variant": "q4",
  "bytes": 323893760,
  "chunks": 42,
  "ms": 8420,
  "source": "network" | "per-site-cache" | "cross-site-cache" | "mixed",
  "cache_hits": 17,
  "browser": "chrome" | "firefox" | "safari" | "other",
  "webgpu": true
}
```

The Worker adds, from the request, `country = request.cf.country`, `colo = request.cf.colo`, and a coarse timestamp. Analytics Engine point: blobs `[model, variant, source, browser, country, colo]`, doubles `[bytes, chunks, ms, cache_hits, webgpu ? 1 : 0]`, index `model`.

Stats response shape:

```json
{
  "since": "2026-09-16T00:00:00Z",
  "window_hours": 168,
  "by_country": [ { "country": "US", "loads": 12, "p50_ms": 8100, "p90_ms": 15200, "cache_hit_rate": 0.41 } ],
  "by_model_variant": [ { "model": "…", "variant": "q4", "loads": 9, "p50_ms": 7900, "bytes": 323893760 } ],
  "by_source": [ { "source": "cross-site-cache", "loads": 3, "p50_ms": 140 } ]
}
```

Use the SQL API's `quantileWeighted` or an approximation over a 7-day window; if the SQL API is unavailable (missing token in dev), return the shape with empty arrays and `"degraded": true` rather than 500.

Rate limiting on telemetry: use the Worker's `cf` object plus a KV counter keyed by `country:colo:minute` capped at 600/minute as a crude abuse guard; do not key on IP.

Tests: `vitest` with `@cloudflare/vitest-pool-workers` (or miniflare) covering: manifest 200 and 304, immutable headers on the hashed manifest, telemetry validation (accept valid, reject oversize, reject unknown field, reject bad enum), stats degraded mode, CORS preflight.

## Edge verification: `scripts/check-edge.sh`

A bash script Theo runs after deploy. It takes a manifest id, reads one chunk id from the manifest via `api.dianome.dev`, then against `cdn.dianome.dev/chunks/<sha>`:
1. `HEAD`: assert 200, `accept-ranges: bytes`, `cache-control` contains `immutable`, `etag` present, `access-control-allow-origin: *`, `timing-allow-origin: *`.
2. `GET` with `Range: bytes=0-1023`: assert 206, `content-range`, 1024 bytes.
3. Two consecutive full `GET`s: print `cf-cache-status` for each; the second must be `HIT`.
4. `If-None-Match` with the ETag: assert 304.
5. Manifest via the API: 200 then 304 on `If-None-Match`.
Prints a pass/fail line per check; non-zero exit on any failure. Do not run it; Theo runs it after deploy and pastes the output.

## Load-test page: `apps/load-test`

Minimal Vite + TypeScript page (this is the seed of the Phase 3 SDK, so keep the fetch logic in one file, `load.ts`, with no DOM dependencies): given a manifest URL and variant, fetches every chunk of that variant in group download order with a concurrency of 6, verifies SHA-256 with `crypto.subtle.digest` as each chunk arrives, shows bytes/sec and a per-group progress bar, and on completion POSTs a load report to `/v1/telemetry/load` with `source: "network"` (Phase 3 will fill in cache sources). Records `performance.getEntriesByType('resource')` `transferSize` totals so cache hits can be inferred later. No caching in this phase.

## Load dashboard: `apps/load-dashboard`

Vite + TypeScript, plain DOM. Fetches `/v1/stats/loads` and renders: a table by country (country, loads, p50, p90, cache-hit rate) sorted by loads; a table by model/variant; a table by source. One bar per country proportional to loads. A note line with `since` and `window_hours`, and a "degraded" banner when the API says so. Deployed later to `dianome.dev/stats`; for now `pnpm --filter load-dashboard dev` pointed at the local Worker.

## docs/phase-2-notes.md

Written from real output after deploy: the `check-edge.sh` output verbatim, the first load-test result (bytes, ms, MB/s, country as reported by the API), and the Worker's deployed version. Nothing hand-typed.

## Definition of done

- `pnpm -r typecheck` and `pnpm --filter worker test` pass.
- `wrangler dev` serves all endpoints locally against the real bucket via the R2 binding (Theo will run `wrangler login` first); telemetry writes succeed or degrade cleanly without the SQL token.
- `scripts/check-edge.sh` exists, is executable, and its checks map one-to-one to the list above.
- `apps/load-test` fetches the q4 variant end to end against `wrangler dev` + `cdn.dianome.dev` and posts a report.
- `apps/load-dashboard` renders the stats shape, including the degraded state.
- `infra/r2-cors.json`, the transform rule and the cache rule are in `docs/phase-2-notes.md` under a "manual steps" heading so the setup is reproducible.
- No secret in any committed file; `.dev.vars` is gitignored.
