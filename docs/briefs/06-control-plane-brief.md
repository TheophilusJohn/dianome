# Phase 6 brief — control plane and self-host

You are working in the `dianome` repo. Read first: `docs/writeup.md` (the product framing and the section 8 line that says this phase is deferred — you will change that line), `docs/briefs/05b-split.md` (session tokens, telemetry v3), `packages/worker/src/split.ts`, `packages/worker/src/telemetry.ts`, `server/dianome_server/auth.py`, `server/dianome_server/ws.py`. TypeScript except `server/`. Nothing published or deployed by you.

Scope is deliberately small. The write-up stands without this phase; do not add anything not listed here.

## Goal

A developer can create an API key, use it from the SDK, see what their app consumed — client-executed and server-executed layer-tokens, separately — and run the server half themselves with one `docker run`, so the "hidden states stay in your infrastructure" option is real.

## Layout

```
packages/worker/
  src/keys.ts                 key creation, hashing, lookup, revocation (KV)
  src/metering.ts             per-key counters (Durable Object or KV with hourly buckets — see Metering)
  src/dashboard-api.ts        GET /v1/me/usage, GET /v1/me/keys, POST /v1/me/keys, DELETE /v1/me/keys/:id
apps/site/src/pages/dashboard/    /dashboard: sign in with a key, see usage and keys
packages/sdk/src/index.ts     Dianome({ apiKey }) → sent as Authorization: Bearer on session and telemetry calls
server/
  Dockerfile                  CUDA and CPU/MPS variants via build arg; `docker run -e SPLIT_SIGNING_KEY=… -e MODEL=… -p 8765:8765 dianome/server`
  docker-compose.yml          server + cloudflared (optional) example
  README.md                   self-host section
docs/phase-6-notes.md         measured only
```

## Keys

- Format: `dk_live_<24 random bytes base64url>`. Stored hashed (SHA-256) in KV under `key:<hash>` → `{ id, owner, created, revoked, allowed_origins?, plan: "free" }`. Never stored or logged in plaintext; shown once at creation.
- Owner identity for this phase is the key itself — there are no accounts. `POST /v1/keys` is open, rate-limited (10/day/country:colo), returns the key and its `id`; keep it simple and say so in the README. Accounts are out of scope.
- `POST /v1/split/session` now requires a key **or** an allowed demo origin (the existing origin allowlist stays for the demo sites). Keyed requests are not subject to the demo origin allowlist.
- Telemetry stays keyless (it is anonymous by design) but records the key `id` when one is present, in blob9.

## Metering

Count per key, per hour bucket: `sessions`, `client_layer_tokens` (N × new_tokens), `server_layer_tokens` ((L − N) × new_tokens, plus `lm_head` counted as one layer-equivalent per token when the server runs it), `server_busy_ms` (from the session's `stats`), `bytes_served` (from telemetry loads reporting this key), `mode` counts. Source of truth is the session `stats` message the SDK already receives; the SDK posts a session report at close (schema 3 already carries these fields; add `key_id`).

Storage: KV with hourly buckets `usage:<key_id>:<YYYYMMDDHH>` written with read-modify-write and a short TTL retry, since Analytics Engine cannot be read per key cheaply. If you judge a Durable Object cleaner, use one; either way it must be free-tier.

`GET /v1/me/usage?hours=168` returns buckets and totals, plus a cost estimate for the server share at the rate in `/v1/split/rates`, labelled as an estimate.

## Dashboard (`/dashboard` on the site)

Paste a key → it is kept in `sessionStorage` only. Shows: totals for 24 h / 7 d, a per-hour bar of client vs server layer-tokens, mode breakdown, server cost estimate, and the key list with revoke. Plain DOM like the rest of the site. No claims beyond what the numbers say.

## SDK

`new Dianome({ apiKey })`. When present: session minting uses it; telemetry and session reports include the key id (never the key). Without it, everything behaves as today. README: a "Keys" section, five lines.

## Self-host

`server/Dockerfile`: base `pytorch/pytorch` CUDA image by default; `--build-arg VARIANT=cpu` for a CPU image. Entrypoint runs `dianome-server serve --model $MODEL --host 0.0.0.0 --port 8765` with `SPLIT_SIGNING_KEY` and optional `SPLIT_TOKEN` from env; models download from Hugging Face on first start into a mounted volume. Health check on `/plan`. The README explains: run it, set the Worker's `SPLIT_SERVERS` (or the SDK's `split.servers` override, add that option) to your host, and hidden states never reach Dianome. `docker-compose.yml` shows server + `cloudflared` with a token.

Test the image builds and serves 0.5B on this Mac (CPU variant; CUDA image builds but is not run here).

## Tests

Worker: key create/hash/lookup/revoke, session requires key or allowed origin, metering read-modify-write, usage endpoint shape, rate limit. SDK: key header presence, key id in reports, never the key. Server: unchanged auth tests still pass. Playwright: dashboard renders totals from a stubbed usage response. Docker: `docker build` succeeds for both variants; the CPU image answers `/plan`.

## Write-up and notes

`docs/phase-6-notes.md`: what was built, the metering fields, a sample usage response from a real local session, image sizes, CPU image start-up time. Update `docs/writeup.md` section 8 to remove the Phase 6 line and add one sentence in section 7 on keys, metering and self-hosting; `writeup-check` must still pass.

## Definition of done

- All tests pass; workspace typecheck clean.
- Locally: create a key via `wrangler dev`, run `run()` with it against the local server, see the session counted in `/v1/me/usage`, revoke the key, see the next session refused.
- CPU Docker image serves 0.5B locally and passes the real-socket N=8 client test from Phase 4 against it.
- Nothing published or deployed.

Report at the end.
