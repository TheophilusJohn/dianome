# Changelog

## 0.2.0 (unreleased)

- `Dianome.run(id, { messages | prompt, variant, policy, maxTokens, sampling, onToken, onPlan, signal })`: generates
  tokens by choosing, per device and network, whether to run the whole model in the browser (`local`), split it at a
  planner-chosen `N` (`split`), or send token ids to the server (`server`). Implemented in the new `dianome/run` entry,
  loaded on first use; the main entry gained only the lazy stubs (`run`, `planRun`) and an internal cache-store accessor.
- `Dianome.planRun(id, opts)`: the plan alone (every candidate `N` with estimated ms/token, server share, cost at the
  stated rate and the Phase 4 privacy band), nothing loaded.
- `run()` imports the `dianome-runtime` package's pure entries (`/tokenizer`, `/planner`, `/session`) in every mode and
  its WebGPU main entry only when the plan puts blocks on this device; without WebGPU it plans `server`.
- Session tokens: `run()` mints a short-lived HMAC token from `POST /v1/split/session` and connects to the split
  server with it (`?token=`); `{ split: { url, token } }` bypasses that for local runs with a static bearer.
- Telemetry schema 3 (`schemas/telemetry.v3.json`): one session report per `run()` call (mode, N, L, variant,
  prompt/new token counts, client/server/rtt ms, tok/s, policy, cache mode). Never any prompt or output content.
- `data/privacy-band.qwen2.5-0.5b-instruct.json`: the Phase 4 band (linear 40k, linear 500k, inversion) per boundary.
- `dianome-runtime` is an optional peer dependency; `package.json` gains `repository`, `homepage`, `bugs`.

## 0.1.0

- `load()` / `stream()` with per-site and cross-site caching, chunk verification, load reports (schema 1 and 2),
  Transformers.js and WebLLM adapters.
