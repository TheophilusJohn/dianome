# Phase 5b brief — split session, planner, `run()`, slider demo

You are working in the `dianome` repo. Read first: `docs/briefs/04-server.md` (boundary definition, protocol, cost formula), `docs/briefs/05a-runtime.md` and `docs/phase-5a-notes.md` (runtime API, kernels, measured tok/s), `docs/phase-4-notes.md` (cost curve, privacy band), `packages/sdk/src/index.ts`, `packages/runtime/src/index.ts`, `packages/runtime/src/protocol.ts`, `server/dianome_server/ws.py`. TypeScript except the server. Nothing published or deployed by you; Theo runs the tunnel and deploys.

## Goal

One call — `run()` — that generates tokens for a prompt by choosing, per device and network, whether to run the whole model in the browser (local), split it at a planner-chosen N (split), or send tokens to the server (server). A demo page with a slider over N shows measured latency, server cost at the stated rate, and the privacy band from Phase 4, live. Server sessions use short-lived signed tokens instead of a static bearer. This is the product's core.

## Amendment to the plan

The runtime gains `lm_head` and a sampler. Phase 5a excluded them to keep scope tight; a hybrid API needs a real local mode. `lm_head` is one more matmul over an existing entry (tied to `embed_tokens` on 0.5B: the manifest marks it `tied`, so reuse the embed bytes); sampling runs on the CPU over the 151,936 logits. Validate `lm_head` against `fixtures/.../logits.npy` (rel RMS < 1e-2, argmax identical) and greedy local generation against the full-model greedy tokens.

## Layout

```
packages/runtime/
  src/lmhead.ts                 lm_head matmul (fp16 / q8 entry) + CPU sampler (greedy, temperature, top-p, seed)
  src/chat.ts                   Qwen2.5 chat template (ChatML: <|im_start|>role\n…<|im_end|>\n), validated against an HF apply_chat_template fixture
  src/session.ts                SplitSession: open(N) → prefill → decode loop; N=0 and N=L handled; token stream; per-step timings
  src/planner.ts                pure function: inputs → { mode, N, estimate, reasons }
  src/microbench.ts             one-time device benchmark: ms per block at T=1 and T=32 for the loaded variant
packages/sdk/
  src/run.ts                    run(): plan → load what's needed → session → tokens; dynamic import of dianome-runtime
  data/privacy-band.qwen2.5-0.5b-instruct.json   the Phase 4 band table (copied from server/probes/results, both linear runs + inversion), keyed by boundary
packages/worker/
  src/split.ts                  POST /v1/split/session → { url, token, expires_at }; GET /v1/split/rates → rates.json; GET /v1/split/plan proxy of the server's /plan (cached 5 s)
server/dianome_server/
  auth.py                       verify HMAC session tokens (shared secret SPLIT_SIGNING_KEY), keep bearer SPLIT_TOKEN for local dev and tests
apps/split-demo/                the slider page (Vite + TS, plain DOM)
docs/phase-5b-notes.md          measured only
```

## Session tokens

Worker mints `token = base64url(payload) + "." + base64url(HMAC-SHA256(SPLIT_SIGNING_KEY, payload))` where `payload = { sid, model, exp (unix, now+3600), max_ctx, origin }`. Server verifies signature and expiry, binds the session to `sid`, refuses reuse of an expired token, and still accepts the static `SPLIT_TOKEN` for local runs. Token travels as `?token=` on the WebSocket upgrade (browsers cannot set headers); expiry is what makes that acceptable. Phase 6 adds per-developer API keys in front of this endpoint; for now the Worker rate-limits it by country:colo:minute like telemetry, and the demo origin is the only allowed `origin`.

## SplitSession

```
open(N, sampling) → opened { session, L, d_model }
prefill: N = 0 → token ids; 1 ≤ N ≤ L → runtime.prefill(tokens); hidden = runtime.exportHidden(); send
loop until done or maxTokens:
  token ← server (N < L+1) or local sampler (mode local)
  emit token; if eos → done
  N ≥ 1: runtime.decode(token, pos); hidden = export; send decode
```

"Local" mode is `N = L` **plus** `lm_head` + sampling in the runtime, no server at all. `N = L` with the server doing `lm_head` is still "split". Record per step: client ms (runtime), export ms, round-trip ms, server `busy_ms` (from the `token` message). Re-planning only at turn boundaries: a session's N is fixed for its lifetime; `run()` may pick a different N for the next call.

## Planner (`planner.ts`, pure, fully unit-tested)

Inputs:
- `model`: L, d_model, bytes per block for the variant, embed bytes, lm_head bytes, privacy band (per boundary, linear-500k top-1 and inversion top-1).
- `device`: WebGPU available; `maxBufferSize`; GPU memory budget (min of `estimate().quota`-derived and a configured cap, default 2 GiB); microbench ms per block at T=1 and at T=32; whether the model at N=L fits.
- `network`: measured download bandwidth (bytes/s, from the SDK's last load or a 1 MB probe), RTT to the server (WS ping, median of 5), whether the server is reachable.
- `server`: `/plan` — L, `busy_fraction_60s`, `ms_per_block_decode`, `ms_per_block_prefill`, `lm_head_ms` (server reports these from its own microbench at startup).
- `policy`: `{ prefer: "cost" | "latency" | "local" | "server", maxLoadSeconds?, maxServerShare?, requireLocal?: boolean }`.
- `prompt`: expected prompt tokens and max new tokens (for the prefill estimate).

Feasible N: those where the client's blocks 0..N-1 (+ embed, + lm_head if local) fit the GPU budget, and the download of what isn't cached fits `maxLoadSeconds` at the measured bandwidth. N = 0 is always feasible if the server is reachable; N = L local is feasible only if `lm_head` fits too.

Estimates per feasible N: `ms_per_token(N) = client_ms_per_block(T=1) × N + export_ms + rtt + server_ms_per_block × (L − N) + lm_head_ms + sampling`; local: no rtt or server terms; server: no client terms. `server_share(N) = (L − N + lm_head_share) / (L + lm_head_share)` using the Phase 4 measured floor rather than assuming zero at N = L. `cost_per_1M(N)` from `server_share` × the N=0 cost at the stated rate (fetched from `/v1/split/rates`; if no rate, cost is reported as a share only). `privacy(N)` = band lookup at boundary N.

Policy: `cost` → max feasible N (local if feasible); `latency` → argmin `ms_per_token`; `local` → local if feasible else the largest feasible N; `server` → N = 0. Output includes every input and every candidate's estimate, so the demo can show why. No browser-name checks anywhere.

## `run()` in the SDK

```ts
const r = await d.run("qwen2.5-0.5b-instruct", {
  messages: [{ role: "user", content: "…" }],   // or prompt: string
  variant: "q4", policy: { prefer: "cost" }, maxTokens: 256, sampling: { temperature: 0 },
  onToken: (t) => {}, onPlan: (p) => {}, signal,
});
r.text; r.tokens; r.mode; r.N; r.plan; r.timings; r.serverBusyMs; r.costEstimate; r.privacy;
```

`run()` dynamically imports `dianome-runtime` only when the plan needs the browser side, so the SDK main entry stays unchanged. Without WebGPU it plans `server` and never imports the runtime. Telemetry v3 session report (schema 3, Worker accepts 1–3): `mode, N, L, variant, prompt_tokens, new_tokens, client_ms, server_busy_ms, rtt_ms, tok_per_s, plan_policy, cache_mode` — no prompt content, ever.

## Server additions

- `/plan` gains `ms_per_block_decode`, `ms_per_block_prefill`, `lm_head_ms` from a startup microbench (median of 5), and is readable without auth (it is public load information; rate-limited).
- `stats` message reports `busy_seconds` and `tokens` as before; the client computes cost.
- HMAC token verification as above; `SPLIT_SIGNING_KEY` from env.

## Slider demo (`apps/split-demo`)

One page: model `qwen2.5-0.5b-instruct` q4. Top: planner inputs as measured on this device (microbench, GPU budget, bandwidth, RTT, server load), refreshed on load. A slider N ∈ [0, L] plus a "local" toggle and an "auto (policy)" selector. For the selected N, three live readouts before running: estimated ms/token, server share and cost per 1M tokens at the stated rate (rate and its retrieval date shown), and the privacy band at that boundary (linear-500k and inversion top-1, with the one-line explanation that this is what a server could recover). A prompt box and Run: streams tokens, then shows measured vs estimated ms/token, the per-step client/server/network breakdown as a stacked bar, the server's `busy_ms` total, and the telemetry that was sent. Nothing in the page states or implies that the server cannot read the prompt.

Deploy target (Theo): `dianome-demo-split.pages.dev`; server at `wss://split.dianome.dev` via `cloudflared` on the Mac. The page takes `?api=&cdn=&split=` overrides for local runs.

## Tests

- vitest: planner (feasibility by memory and load budget; each policy; local requires lm_head fit; degraded inputs → server; every candidate present in output), chat template against `fixtures/.../chat_template_cases.json` (add to the Python fixtures command: 50 message lists → rendered strings and token ids from HF `apply_chat_template`), sampler determinism with seed, token HMAC round-trip and expiry.
- Playwright (Chrome): local mode greedy 16 tokens == full-model greedy (new gate); split at planner-chosen N == full-model greedy; N = 0 == full-model greedy; `run()` end to end with `prefer: "cost"` and `prefer: "latency"` producing a plan, tokens, and a v3 telemetry POST (against `wrangler dev`).
- server pytest: HMAC verify (valid, expired, bad signature, wrong origin), `/plan` fields, bearer still works.

## docs/phase-5b-notes.md

Measured only: `lm_head` gate; local greedy gate; planner inputs on this Mac (Chrome and Safari); estimated vs measured ms/token for N ∈ {0, 4, 8, 12, 16, 20, 24, local}; planner choices per policy; RTT to the server locally and via the tunnel once Theo has it up; per-step breakdown medians; the v3 telemetry row as written.

## Definition of done

- All tests above pass; runtime `lm_head` gate and local-mode gate in the Playwright suite.
- `run()` works in the demo page locally against `dianome-server serve` + `wrangler dev` in all three modes.
- SDK main entry size unchanged; `dianome-runtime` builds and is ready to publish alongside `dianome@0.2.0` (version bumps prepared, not published; add `repository`, `homepage`, `bugs` fields and a CHANGELOG entry).
- Worker session endpoint, rates and plan proxy implemented with tests; not deployed.
- Nothing published or deployed.

Report after the `lm_head` + local-mode gates pass, and again at the end.
