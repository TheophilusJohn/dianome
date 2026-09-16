# Phase 2 notes — edge delivery

Sections marked **pending** are filled from real output after `pnpm --filter worker deploy`;
nothing in them is typed by hand. The "Manual steps" section is the reproducible dashboard setup.

## Manual steps (dashboard, done by Theo before hand-off)

All on zone `dianome.dev`, account `4da3743239f76a52b672f5654524e1d2`, bucket `dianome`.

1. **R2 custom domain.** R2 → bucket `dianome` → Settings → Custom Domains → Connect `cdn.dianome.dev`.
   Creates the DNS record and enables edge caching for the bucket.
2. **Bucket CORS.** R2 → bucket `dianome` → Settings → CORS policy → paste `infra/r2-cors.json`
   (also reproduced here):

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

   R2 only emits the `access-control-*` headers when the request carries an `Origin` header, which is
   why `scripts/check-edge.sh` sends one on every CDN request.
3. **Transform rule** (Rules → Transform Rules → Modify Response Header): when hostname equals
   `cdn.dianome.dev`, set static header `Timing-Allow-Origin` = `*`. R2 cannot serve custom headers
   from object metadata, and Phase 0 showed `transferSize` is 0 without it.
4. **Cache rule** (Caching → Cache Rules): when hostname equals `cdn.dianome.dev`, Eligible for cache,
   Edge TTL = use origin `cache-control` (bypass cache if absent). Without it Cloudflare skips
   extensionless paths like `chunks/<sha256>` (`cf-cache-status: DYNAMIC`); with it the second GET is `HIT`.
5. **Worker secrets** (after the first deploy): `wrangler secret put CF_ACCOUNT_ID` and
   `wrangler secret put CF_ANALYTICS_TOKEN` in `packages/worker` (token: Account | Account Analytics | Read).
   Without them `/v1/stats/loads` answers the empty shape with `"degraded": true`.
6. **KV namespace** `STATS_CACHE` was created with `wrangler kv namespace create STATS_CACHE`
   (id `7c4f6c0ba5e44b0a82496b3639c90262`, in `packages/worker/wrangler.toml`).

## Deploy (pending)

```
$ pnpm --filter worker deploy
<paste output: uploaded version id, route>
```

Deployed Worker version: _pending_.

## scripts/check-edge.sh (pending)

```
$ scripts/check-edge.sh qwen2.5-0.5b-instruct
<paste output verbatim>
```

## First load-test result (pending)

From `apps/load-test` (`pnpm --filter load-test dev`, variant q4, against `api.dianome.dev` + `cdn.dianome.dev`),
the "Result" block of the page and the row it produced in `/v1/stats/loads`:

```
<paste the Result JSON: bytes, ms, MiB/s, and the country as reported by the API>
```
