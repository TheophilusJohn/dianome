# Changelog

## 0.2.0 (unreleased)

- `lm_head` + sampler (`src/lmhead.ts`): with `Runtime.create(..., L, maxCtx, { lmHead: true })` the runtime also
  loads `final_norm` and the `lm_head` entry (tied to `embed_tokens` on Qwen2.5-0.5B: the embed bytes are uploaded once
  more) and `logits()` returns f32 logits for the last position; `Sampler` (greedy, temperature, top-p, seed) runs on
  the CPU. Validated against `logits.npy` (fp16: rel RMS 1.85e-3, argmax identical on all 32 rows) and by 16-token
  greedy local generation equal to the full model's tokens (gates 8 and 9 in `test/e2e/gates.spec.ts`).
- `SplitSession` (`src/session.ts`): one generation session at a fixed `N` in local, split or server mode, with
  per-step client / export / round-trip / server-busy / lm_head / sampling timings.
- `plan()` (`src/planner.ts`): the pure planner (feasible `N` by GPU budget and load budget, estimates per candidate,
  server share from the Phase 4 lm_head floor, cost at a stated rate, privacy band lookup, four policies).
- `microbench()` (`src/microbench.ts`): ms per block at T = 1 and T = 32, lm_head ms and export ms on this device for
  a variant, from synthetic weights of the variant's exact kinds and shapes.
- `renderChat()` (`src/chat.ts`): the Qwen2.5 ChatML template, validated against 50 HF `apply_chat_template` cases.
- Protocol: `ping` / `pong` for RTT; `SplitClient.prefillIds` / `decodeId` for N = 0; `SplitClient.rtt()`.
- Subpath entries without WebGPU code: `dianome-runtime/tokenizer`, `/planner`, `/session`, `/protocol`.
- `GpuOps.matmulWith(..., kernel)` selects the T = 1 kernel per call (the head can use `lanes` under `seq` blocks).
- Package published alongside `dianome@0.2.0`: no longer private; `repository`, `homepage`, `bugs` added.

## 0.1.0

- Embedding + Qwen2 blocks 0..N-1 on WebGPU with a KV cache, fp16 hidden-state export, gates 1–7 (Phase 5a).
