# Phase 5a brief — Dianome WGSL partial-model runtime

You are working in the `dianome` repo. Read first: `docs/briefs/04-server.md` for the layer-boundary definition (the runtime must produce exactly "the input to block N"); `fixtures/qwen2.5-0.5b-instruct/fixtures.json` for the reference activations and their hashes; `docs/briefs/01-ingest.md` for the per-layer entry layout, 256-byte alignment, and the q8/q4 storage formats; `packages/sdk/src/assemble.ts` for how entries arrive as `Uint8Array` views. TypeScript plus WGSL. Nothing is reused from any other project; write the kernels fresh.

Hardware for development and validation: this MacBook, Google Chrome (Playwright's bundled Chromium has no WebGPU; use `channel: "chrome"`). Firefox and Safari are checked at the end, not developed against.

## Goal

A browser runtime that loads per-layer chunks through the `dianome` SDK, runs the embedding lookup and transformer blocks `0..N-1` of a Qwen2-architecture model on WebGPU with a KV cache, and exports the fp16 hidden state that is the input to block N. No `lm_head`, no sampler. Every layer is validated numerically against the Phase 4 fixtures before the next is written.

## Layout

```
packages/runtime/                 npm entry "dianome/runtime" later (lazy subpath; must not grow the SDK main entry)
  src/index.ts                    Runtime class: create(device, manifest, variant, N, maxCtx), prefill(tokens), decode(token, position), exportHidden(), stats()
  src/tokenizer.ts                BPE tokenizer from the store's tokenizer.json (byte-level BPE as Qwen2 uses); no wasm
  src/gpu/device.ts               adapter/device acquisition, limits, feature detection (shader-f16 optional, never required)
  src/gpu/buffers.ts              weight upload from Entry views, 256-aligned, fp16 packed as u32 pairs
  src/gpu/kernels/*.wgsl          one file per kernel, imported as strings
  src/gpu/ops.ts                  typed wrappers: embed, rmsnorm, matmul{fp16,q8,q4}, rope, attnPrefill, attnDecode, silu_mul, add
  src/block.ts                    one transformer block = ops in Qwen2 order
  src/kv.ts                       KV cache per block, f16 storage, capacity maxCtx
  test/                           vitest unit tests (CPU reference implementations) + Playwright validation suite in Chrome
apps/runtime-bench/               a page that loads a model, runs prefill/decode for a given N, reports tok/s and GPU memory
docs/phase-5a-notes.md            measured only
```

## Precision policy

Weights stay fp16 in GPU memory (packed two per `u32`, unpacked with `unpack2x16float`; no dependency on the `shader-f16` feature). Activations and accumulators are `f32`. KV cache stored as fp16, converted on read. Hidden-state export converts `f32 → fp16` on the way out. This matches the fixtures, which are fp16 activations from an fp16 PyTorch model, up to accumulation-order differences.

## Validation gates (in order; do not write the next stage until the current one passes)

Tolerances are against the fp16 fixtures; report the actual numbers in the notes.

1. **Tokenizer.** `dianome-server fixtures` is extended (Python side, small change) to also dump `tokenizer_cases.json`: 200 strings (the fixture prompt, wiki sentences, code, emoji, CJK, whitespace edge cases) with their token ids from the HF tokenizer. The TS tokenizer must match all 200 exactly, encode and decode.
2. **Embedding + RMSNorm + fp16 matmul (unit level).** Against CPU reference implementations in the test suite on random inputs: max relative error `< 1e-3` (f32 vs f32). Then embedding output vs `fixtures` embedding: max abs `< 2e-3`.
3. **Block 0.** Extend `dianome-server fixtures` with `--intra`: for block 0 only, dump the post-input-norm tensor, q/k/v after projection (post-bias, pre-RoPE), q/k after RoPE, attention output (pre-o_proj), post-o_proj residual sum, post-post-attention-norm, MLP output, and block output. Validate each stage in order: max abs `< 5e-3` per stage on the 32-token prompt.
4. **All blocks, prefill.** Output of block `i` vs `block_{i}.npy` for every i: report max abs and relative RMS per block; gate: relative RMS `< 1e-2` at every block and max abs `< 5e-2` at block 23 (error accumulates through 24 fp16 layers; if it exceeds this, find the kernel that contributes most before loosening anything).
5. **Decode with KV cache.** Prefill 31 tokens, decode the 32nd; the exported hidden state at position 31 must match the prefill-path value at position 31 within `< 5e-3` max abs for every block. Then 16 decode steps using the Phase 4 server's greedy tokens for that prompt as the input sequence: hidden states at each step vs a new fixture dump (`decode_steps.npy`, add it to the Python fixtures command).
6. **End to end with the server.** Runtime at N ∈ {1, 8, 16, 24} → hidden state over the real WebSocket to `dianome-server` (local, `SPLIT_TOKEN`) → 16 greedy tokens equal the full-model greedy tokens. This is the split-inference correctness gate in the browser.
7. **q8 and q4.** Dequantising matmul kernels validated at unit level against CPU dequant + matmul on random tensors quantised with a TS port of `quant.py`'s encoder (max rel err `< 1e-3` vs the same dequant path). Then end-to-end gate 6 repeated with the q8 and q4 variants: tokens may legitimately differ from fp16 (quantisation), so the gate is agreement with PyTorch running the *dequantised* q8/q4 weights — add `--variant q8|q4` to `dianome-server fixtures` using `ingest`'s decoder to build that reference.

## Kernels (Qwen2 block order)

`x → rmsnorm(input_layernorm) → q,k,v = matmul+bias → RoPE(q,k) → attention (GQA: 14 q-heads, 2 kv-heads, head_dim 64; causal) → o_proj → x += attn → rmsnorm(post_attention_layernorm) → gate,up = matmul → silu(gate)*up → down_proj → x += mlp`.

- `matmul`: one kernel handles `[T, in] × [out, in]^T + bias` with a tiled workgroup; T = 1 (decode) must not fall off a cliff — write a separate matvec path if the tiled kernel is slow at T = 1 and measure both.
- `rope`: rotate-half form with the cos/sin tables from the fixtures for validation and computed on device otherwise (`rope_theta` from the manifest config).
- `attnPrefill`: per (head, query row) causal softmax over the KV of this block; fine to be simple first. `attnDecode`: single query row against the cache.
- Never allocate per token; all buffers sized at `create()` for `maxCtx`.

## Loading

Use `Dianome.stream(id, { variant })` from `packages/sdk`; upload each group's entries as they arrive so the runtime can start block 0 before block 23 has downloaded (report time-to-first-block in the bench page). `N` selects how many block groups to fetch; do not fetch `final_norm` or `lm_head`. Respect the Firefox 1 GiB `maxBufferSize` floor from Phase 0: no single GPU buffer larger than 1 GiB, ever.

## Bench page (`apps/runtime-bench`)

Inputs: model id, variant, N, prompt. Outputs: time to first block, total load time, prefill tok/s, decode tok/s, hidden-state export time, GPU buffer bytes, adapter info. Runs on this Mac in Chrome, then once each in Firefox and Safari for the notes (WebGPU support varies; record what happens rather than fixing it in this phase).

## Tests

- vitest: tokenizer (200 cases), CPU reference ops, quant encoder port, KV cache indexing, block wiring against a tiny synthetic model (d = 64, 2 heads, 2 blocks) with a PyTorch-free CPU reference in TS.
- Playwright (Chrome channel): gates 2–7 as a suite that loads fixtures from `fixtures/` over a local static server and asserts the tolerances. This suite is the phase's definition of done.

## docs/phase-5a-notes.md

Measured only: per-gate error tables; per-block relative RMS curve; prefill and decode tok/s vs N for fp16/q8/q4 on this Mac; time to first block; GPU memory; Firefox and Safari results as observed.

## Definition of done

- Gates 1–7 pass in Chrome via the Playwright suite; every tolerance and actual value in the notes.
- `packages/runtime` builds as an ESM subpath entry; `packages/sdk` main entry size is unchanged.
- The bench page runs the 0.5B q4 model at N = 24 in Chrome on this Mac and reports tok/s.
- Nothing published, nothing deployed.

Report after gate 3 (block 0 validated) before continuing, and again after gate 5.
