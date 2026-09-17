# Phase 2 notes — edge delivery

Every number below is from real output of the deploy on 2026-09-16 / 2026-09-17, pasted from the
terminal, the load-test page and the Cloudflare dashboard; nothing is typed by hand. The "Manual
steps" section is the reproducible dashboard setup.

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

## Deploy

Two Worker versions were deployed with `pnpm --filter worker deploy`:

| Worker version | commit | what |
| --- | --- | --- |
| `c4762373` | `f9e6eac` | first deploy (custom domain `api.dianome.dev`) |
| `39a5e01c` | `6743ced` | stats fix (empty result no longer cached, query fixed) |

## scripts/check-edge.sh

`scripts/check-edge.sh qwen2.5-0.5b-instruct` against `api.dianome.dev` and `cdn.dianome.dev`: all 14 checks passed.

- chunk `006fa407ad078653907f9dd7446c440e943ea1964beec944a57ba9b396737be1`, manifest sha `66b487935d4297460fcc601a3cf373c61808bdb6ca15156c3ae3fff1d3ddbae4`
- 1. HEAD: 200, `accept-ranges: bytes`, `cache-control` contains `immutable`, etag `"86f38d2c16d3a9035f6367d42e3ec779"`, `access-control-allow-origin: *`, `timing-allow-origin: *`
- 2. `Range: bytes=0-1023`: 206, `content-range: bytes 0-1023/8388608`, 1024 bytes
- 3. two consecutive full GETs: `cf-cache-status: HIT` on the second
- 4. `If-None-Match` with the etag: 304
- 5. manifest via the API: 200, then 304 on `If-None-Match`

## First browser load

`apps/load-test` in Chrome 152 against `api.dianome.dev` + `cdn.dianome.dev`, variant q4:

| field | value |
| --- | --- |
| bytes | 323,893,760 |
| chunks | 42 |
| ms | 15,629 |
| bytes/s | 20,723,895 (20.7 MB/s) |
| transferBytes | 323,906,360 |
| resourceEntries | 42 |
| zeroTransferEntries | 0 |
| telemetry | HTTP 202 |

Analytics Engine row as written (dataset `dianome_loads`, timestamp `2026-09-16 23:50:45`):

| column | value |
| --- | --- |
| blob1 (model) | `qwen2.5-0.5b-instruct` |
| blob2 (variant) | `q4` |
| blob3 (source) | `network` |
| blob4 (browser) | `chrome` |
| blob5 (country) | `US` |
| blob6 (colo) | `ATL` |
| blob7 (hour) | `2026-09-16T23:00:00.000Z` |
| double1 (bytes) | 323893760 |
| double2 (chunks) | 42 |
| double3 (ms) | 15629 |
| double4 (cache_hits) | 0 |
| double5 (webgpu) | 1 |

## First stats response

`GET https://api.dianome.dev/v1/stats/loads`, computed after the deploy of `6743ced`:

```json
{"since":"2026-09-10T00:01:22Z","window_hours":168,"by_country":[{"country":"US","loads":1,"p50_ms":15629,"p90_ms":15629,"cache_hit_rate":0}],"by_model_variant":[{"model":"qwen2.5-0.5b-instruct","variant":"q4","loads":1,"p50_ms":15629,"bytes":323893760}],"by_source":[{"source":"network","loads":1,"p50_ms":15629}]}
```

(This body has no `computed_at`: it was a KV hit on an entry written by Worker version `c4762373`,
before the field existed. The next commit bumps the cache key so such entries are recomputed.)

## Issues found in deploy

- **Empty stats result cached.** The first `/v1/stats/loads` after deploy ran before Analytics Engine
  had surfaced the load-test point and returned zero rows; that empty body was cached in KV for the
  full 5-minute TTL (its expiry was exactly compute time + 300 s, so the TTL itself was right). Fixed in
  `6743ced`: an empty result is never cached and goes out with `no-store`; a 200 with a non-JSON body
  degrades instead of counting as empty; the cache-hit division needed a typed `if(..., 0.0, ...)`
  guard, which the SQL API rejected with a 422 when the branches were integer and double.
- **DNS.** After `wrangler deploy` created the custom domain, `api.dianome.dev` took about a minute
  to resolve.
- **Analytics Engine had to be enabled once** in the dashboard before the first deploy would accept
  the `dianome_loads` binding.
- **Extensionless chunk paths were not cached** (`cf-cache-status: DYNAMIC`) until the Cache Rule
  in manual step 4 was in place.
