# Phase 6 notes — control plane and self-host

Measured only. Sources: `packages/worker/results/phase6-local.json` (the local key → run() → usage → revoke flow),
`packages/worker/results/phase6-docker-n8.json` (the Phase 4 N = 8 socket test against the CPU image), the Docker
build and run logs quoted below. Date: 2026-09-18, on this MacBook (Apple M4, 16 GB), Docker 29.4.0 (OrbStack).
Nothing here was deployed or published.

## What was built

| where | what |
|---|---|
| `packages/worker/src/keys.ts` | `dk_live_<24 random bytes base64url>` keys; stored as `key:<sha256>` → `{ id, owner, created, revoked, allowed_origins?, plan: "free" }`; `keyid:<id>` and `owner:<owner>` indexes; `POST /v1/keys` open, 10 per day per country:colo; `Authorization: Bearer` resolution (none / ok / invalid 401 / revoked 403) |
| `packages/worker/src/metering.ts` | hourly KV buckets `usage:<key_id>:<YYYYMMDDHH>` (TTL 8 days), read-modify-write with three attempts; `layerTokens()`; `readUsage()` over a `list(prefix)` |
| `packages/worker/src/dashboard-api.ts` | `GET /v1/me/usage?hours=`, `GET/POST /v1/me/keys`, `DELETE /v1/me/keys/:id`; cost estimate = server busy ms at the rate in `/v1/split/rates`, labelled `estimate: true` |
| `packages/worker/src/split.ts` | `POST /v1/split/session` needs a key **or** an allowed demo origin; keyed responses carry `key_id` |
| `packages/worker/src/telemetry.ts` | a report posted with a key records the key id in `dianome_loads` blob 9 / `dianome_sessions` blob 10 (never the key) and meters the bucket; schema 3 gains an optional `key_id` that must match the bearer |
| `packages/sdk` | `new Dianome({ apiKey })` → bearer on the session mint and telemetry; `key_id` in session reports; `split: { servers }` self-host override; README "Keys" section |
| `apps/site/dashboard/` | paste a key (sessionStorage only), totals 24 h / 7 d, per-hour client vs server layer-token bars, modes, cost estimate, key list with revoke, create keys |
| `server/Dockerfile`, `docker-compose.yml` | CUDA (default) and `--build-arg VARIANT=cpu` images; `HF_HOME=/models` volume; health check on `/plan`; server starts with `SPLIT_SIGNING_KEY` alone |

Storage is the existing `STATS_CACHE` KV namespace under new prefixes, so no namespace had to be created. KV has no
transactions: two reports for one key in the same hour landing on different colos can lose an increment (last write
wins after a re-read); Analytics Engine keeps every raw row.

## Metering fields

Per key, per UTC hour, from the schema-3 session report the SDK posts when `run()` finishes (its `stats`), plus the
load reports posted with the key:

| field | source |
|---|---|
| `sessions` | one per session report |
| `client_layer_tokens` | `N × new_tokens` |
| `server_layer_tokens` | `(L − N) × new_tokens`, plus `new_tokens` for the language-model head whenever the server ran it (split and server modes; zero in local mode) |
| `new_tokens` | the report's `new_tokens` |
| `server_busy_ms` | the report's `server_busy_ms` (the sum of the server's per-step `busy_ms`) |
| `bytes_served` | `bytes` of load reports posted with the key |
| `modes` | `{ local, split, server }` session counts |

`cost_estimate.usd = server_busy_ms / 3.6e6 × usd_per_hour` at the rate in `server/bench/rates.json` (L4 at $0.49/h,
retrieved 2026-09-17); busy time is model execution only, so it is a lower bound on what a server costs to run.
Nothing is counted with `telemetry: false`, since the report is the meter.

## The local flow: create a key, run(), see it counted, revoke, be refused

`wrangler dev` on 8790 with `SPLIT_SERVERS` pointing at the CPU Docker image on 8766 and the same
`SPLIT_SIGNING_KEY`; `scripts/serve-store.mjs` on 8788 for the tokenizer files; a node script driving the built SDK
(node has no WebGPU, so the planner chose server, N = 0). `packages/worker/results/phase6-local.json`:

| step | result |
|---|---|
| `POST /v1/keys` | 201, id `k_73c3600ea20d1bed`, owner = id, plan free |
| `run()` with the key | mode server, N = 0, "Hello, how can I help you today?", 35 prompt tokens, 9 new tokens, 11.7 tok/s, server busy 1678.3 ms (the CPU image), RTT 0.69 ms, wall 2482 ms, telemetry 202 |
| the session report | `key_id: "k_73c3600ea20d1bed"`, no key anywhere in it |
| `GET /v1/me/usage?hours=24` | 1 session, 0 client layer-tokens, 225 server layer-tokens (24 × 9 + 9 for the head), 9 new tokens, 1678.3 ms busy, estimate $0.000228 |
| `DELETE /v1/me/keys/k_73c3…` | 200, `revoked` stamped |
| `POST /v1/split/session` after | 403 `key_revoked` |
| `run()` after | "no feasible plan: server unreachable" (the mint was refused; node has no local candidate) |
| `GET /v1/me/usage` after | 403 `key_revoked` |

The usage response as returned:

```json
{
  "key_id": "k_73c3600ea20d1bed", "owner": "k_73c3600ea20d1bed", "keys": ["k_73c3600ea20d1bed"], "hours": 24,
  "since": "2026-09-17T08:00:00.000Z", "until": "2026-09-18T08:00:00.000Z",
  "buckets": [{ "hour": "2026-09-18T07:00:00.000Z", "sessions": 1, "client_layer_tokens": 0, "server_layer_tokens": 225,
                "new_tokens": 9, "server_busy_ms": 1678.3, "bytes_served": 0, "modes": { "local": 0, "split": 0, "server": 1 },
                "key_id": "k_73c3600ea20d1bed" }],
  "totals":   { "sessions": 1, "client_layer_tokens": 0, "server_layer_tokens": 225, "new_tokens": 9, "server_busy_ms": 1678.3, "bytes_served": 0, "modes": { "local": 0, "split": 0, "server": 1 } },
  "last_24h": { "sessions": 1, "client_layer_tokens": 0, "server_layer_tokens": 225, "new_tokens": 9, "server_busy_ms": 1678.3, "bytes_served": 0, "modes": { "local": 0, "split": 0, "server": 1 } },
  "cost_estimate": { "estimate": true, "basis": "server_busy_ms / 3.6e6 h × usd_per_hour from /v1/split/rates; busy time is model execution only, so this is a lower bound on what a server actually costs",
                     "usd": 0.000228, "usd_per_hour": 0.49, "gpu": "NVIDIA L4 24GB", "source": "RunPod pod deploy screen (console.runpod.io, Secure Cloud, on-demand)", "retrieved": "2026-09-17" }
}
```

The site's e2e proxy (`apps/site/test/e2e/run-server.mjs`) forwards `/v1/keys` and `/v1/me/*` with the
`Authorization` header; through it, create → list → usage (200) → no key (401) → revoke (200) all answered as above.

## Self-host image

`docker build --build-arg VARIANT=cpu -t dianome/server:cpu server/` (arm64, `python:3.12-slim` + `torch 2.8.0+cpu`,
transformers 4.57.1) and `docker build --platform linux/amd64 -t dianome/server:cuda server/`
(`pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime`; built here under emulation, not run: this Mac has no CUDA).

| image | `docker image ls` size | note |
|---|---|---|
| `dianome/server:cpu` | 1.6 GB (1.492 GB unique + 108.3 MB shared with the python base) | runs here |
| `dianome/server:cuda` | 4.41 GB (4.412 GB unique, no shared layers), linux/amd64 | builds; not run |

CPU image, first start with an empty `/models` volume (`docker run -e SPLIT_SIGNING_KEY=… -e SPLIT_TOKEN=… -e MODEL=qwen2.5-0.5b-instruct -p 8766:8765 -v dianome-models-test:/models dianome/server:cpu`):

| measurement | value |
|---|---|
| `GET /plan` first 200 after `docker run` | 57.4 s (the Hugging Face download of the 0.5B fp16 weights + tokenizer into the volume, load, and the startup microbench) |
| the server's own load line | `loaded qwen2.5-0.5b-instruct (…@7ae5576…) L=24 d_model=896 device=cpu (10 threads) in 50.6s` |
| startup microbench (median of 5) | decode 3.585 ms/block, prefill 15.650 ms/block at T = 32, lm_head 4.68 ms |
| Docker health check | `healthy` 35 s after the first 200 |
| volume after the download | 954 MB (`du -sh /models`), 999.7 MB per `docker system df` |
| `docker stop` → `docker start` → `/plan` 200 (warm volume) | 8.6 s |

The CPU inside the container is far slower than the same model on this Mac's CPU outside it (0.60 ms/block decode
in the host venv, `DIANOME_DEVICE=cpu`): the container runs in OrbStack's Linux VM with 10 threads and no MPS.

The Phase 4 real-socket N = 8 client test (`tests/ws_client.generate`, prefill of the 32-token prompt then 16 greedy
steps; the client half in PyTorch on this Mac, the server half in the container over `ws://127.0.0.1:8766` with the
static `SPLIT_TOKEN`) against the full model's greedy tokens (`packages/worker/results/phase6-docker-n8.json`):

| client half on | tokens equal to `full_greedy(16)` | wall |
|---|---|---|
| mps | yes | 1.52 s |
| cpu | yes | 1.63 s |

Both decode to " meanders through the lush green forest. The valley is a beautiful place, with".

## Tests

| suite | result |
|---|---|
| Worker (`packages/worker`, vitest pool-workers) | 62 passed (2 files; `test/phase6.test.ts` adds 20: key create/hash/lookup/revoke, the open endpoint's daily limit, session gating by key or origin, allowed_origins, metering read-modify-write with a failing-put retry, readUsage window, cost estimate, telemetry key id in blob 9/10, body `key_id` vs bearer, dashboard usage/keys/revoke shape, CORS preflight) |
| SDK (`packages/sdk`, vitest) | 121 passed (11 files; new: bearer on mint and telemetry, `key_id` in the report and never the key, no bearer without a key, `split.servers` override with the plan from the host and the socket opened there) |
| server (`server/.venv/bin/pytest`, full suite incl. the correctness gates) | 62 passed (one new: `SPLIT_SIGNING_KEY` alone starts the server and the bearer is then refused) |
| site Playwright (Chrome) | `dashboard.spec.ts` 2 passed (stubbed usage: totals, per-hour bar widths, modes, cost, revoke via DELETE, sessionStorage round trip; a refused key signs out), plus the nav/render test now expecting five pages |
| workspace `pnpm -r typecheck` | clean |
| SDK size check | `index.js` 44545 bytes (14190 gzipped), unchanged budget |
