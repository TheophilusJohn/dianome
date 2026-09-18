# dianome-runtime

WGSL partial-model runtime for the split-inference client half: embedding lookup plus Qwen2 transformer blocks
`0..N-1` on WebGPU with a KV cache, exporting the fp16 hidden state that is the input to block `N` (the Phase 4
boundary). With `lmHead: true` (N = L) it also runs the final norm + `lm_head` and samples on the CPU: local mode,
no server. Weights arrive through the `dianome` SDK as per-layer entries and are uploaded as they stream in, so
block 0 can run before block 23 has downloaded. `SplitSession` drives one generation at a fixed N (local / split /
server) with per-step timings; `plan()` chooses N from measured inputs; `microbench()` measures this device.

```ts
import { Dianome } from "dianome";
import { Runtime, acquireDevice, loadTokenizer, SplitClient } from "dianome-runtime";

const d = new Dianome();
const { device } = await acquireDevice();
const manifest = await d.manifest("qwen2.5-0.5b-instruct");
const tok = await loadTokenizer(d.cdn, manifest);
const rt = await Runtime.create(device, manifest, "q4", 8, 1024);     // N = 8 blocks on the client
await rt.loadFrom(d.stream("qwen2.5-0.5b-instruct", { variant: "q4" })); // stops after layer.7

await rt.prefill(tok.encode("Hello"));
const hidden = await rt.exportHidden();          // fp16 bits, [T, d_model]: the input to block 8
const server = new SplitClient("wss://…/", token);
await server.connect(); await server.open("qwen2.5-0.5b-instruct", 8);
let t = await server.prefill(hidden, T);
await rt.decode(t.id, T); t = await server.decode(await rt.exportHidden(), T); …
```

### Local mode, sessions, planning (Phase 5b)

```ts
import { Runtime, SplitSession, plan, microbench, renderChat } from "dianome-runtime";
// or, without any WebGPU code: "dianome-runtime/tokenizer" (Tokenizer, renderChat, loadTokenizer),
// "dianome-runtime/session" (SplitSession, SplitClient, Sampler), "dianome-runtime/planner" (plan)

const rt = await Runtime.create(device, manifest, "q4", 24, 1024, { lmHead: true });   // N = L + head
await rt.loadFrom(d.stream("qwen2.5-0.5b-instruct", { variant: "q4" }));                // stops after final_norm
const s = await SplitSession.open({ mode: "local", N: 24, model: "qwen2.5-0.5b-instruct", maxCtx: 1024, runtime: rt, sampling: { temperature: 0.7, topP: 0.9, seed: 1 }, eosIds: [151645, 151643] });
for await (const t of s.generate(tok.encode(renderChat([{ role: "user", content: "Hi" }])), 64)) console.log(t.id, t.timing);

const mb = await microbench(device, manifest, "q4");                 // ms/block at T=1 and T=32, lm_head ms, export ms
const p = plan({ model, device: { ...mb, ... }, network, server, policy: { prefer: "cost" }, prompt });   // pure
```

`lm_head` is one more matmul over the embed entry (the manifest marks `lm_head.weight` as `tied`, so the embed bytes
are uploaded once more as a weight; fp16 or q8 depending on the variant) after the `final_norm` RMSNorm, on the last
row only, followed by a readback of the vocab-sized f32 logits. Sampling follows the server's rules: greedy at
temperature 0, otherwise softmax(logits / T), nucleus top-p, multinomial from a seeded xoshiro128** PRNG.

## Precision

Weights stay in their stored format on the GPU (fp16 pairs per `u32`, int8 ×4, q4 nibbles ×8, dequantised in the
kernels; `shader-f16` is never required). Activations and accumulators are f32. The KV cache is fp16. With
`round16: true` (default) every point where an fp16 PyTorch model materialises an fp16 tensor is rounded to fp16
too; this is what makes the outputs match the Phase 4 fixtures to within one ulp per block. `round16: false`
runs pure f32 activations.

Two kernels exist for T = 1 (decode): `matvec: "seq"` (one thread per output row summing in exactly the tiled
kernel's order) and `matvec: "lanes"` (16 lanes per row plus a workgroup reduction, about 2x faster per token).
The default follows `round16`: with fp16 emulation on, `seq` is used because it makes a decode step reproduce the
prefill path bit for bit (the lanes kernel's different summation order flips fp16 rounding boundaries, and those
one-ulp flips grow to ~3e-2 by block 23; gate 5a in the notes); with emulation off there is no rounding to flip,
the two kernels differ only at f32 accumulation level (~1e-7), and the faster `lanes` kernel is the default. Both
remain selectable and both are measured in `docs/phase-5a-notes.md`.

## Layout

```
src/index.ts          Runtime: create, addGroup/loadFrom, prefill, decode, exportHidden, stats (+ debug hooks used by the gates)
src/block.ts          one block as a backend-generic op sequence (forwardBlock), shared by the GPU and the CPU reference
src/gpu/ops.ts        GpuOps: typed wrappers recording dispatches; per-call-site uniforms and bind groups (no per-token allocation)
src/gpu/kernels/      one .wgsl per kernel: rmsnorm, matmul_head + weights_{f16,q8,q4} + matmul_tiled / matvec / matvec_seq,
                      rope_kv, attention, silu_mul, add, f32_to_f16
src/gpu/buffers.ts    weight upload from SDK entries (one buffer per entry, parts bound at their 256-aligned offsets)
src/gpu/device.ts     adapter/device, limits (never a buffer over 1 GiB), feature report
src/kv.ts             per-block fp16 K/V rows for maxCtx positions
src/cpu.ts            CPU reference backend (the unit tests' oracle)
src/tokenizer.ts      byte-level BPE from tokenizer.json; src/loadTokenizer.ts fetches it from the store
src/protocol.ts       DNM1 frames + SplitClient (token as ?token= because browsers cannot set upgrade headers), ping/rtt
src/lmhead.ts         LmHead (final norm + tied-embed matmul on the last row) + Sampler / Prng / argmax / topk
src/session.ts        SplitSession: open(N) → prefill → decode loop in local / split / server mode, per-step timings
src/planner.ts        plan(): pure planner (feasibility, estimates, server share, cost, privacy band, policies)
src/microbench.ts     one-time device benchmark on synthetic weights of the variant's kinds and shapes
src/chat.ts           Qwen2.5 ChatML template (validated against HF apply_chat_template fixtures)
src/entries/          the tokenizer / planner / session / protocol subpath entries (no WebGPU code)
src/quant.ts          TS port of ingest's q8/q4 encoders (tests)
test/*.test.ts        vitest: tokenizer (200 HF cases), CPU ops, quant port, KV state, block wiring on a synthetic model
test/e2e/             Playwright (Google Chrome): gates 2–9 against fixtures/ and the local store; diag/bench specs
```

## Running

```sh
pnpm --filter dianome-runtime test                       # vitest
SPLIT_TOKEN=x pnpm --filter dianome-runtime test:e2e     # gates 2–9 in Chrome; starts the static server and dianome-server serve
pnpm --filter dianome-runtime exec playwright test bench.spec.ts   # tok/s sweep -> results/bench.json
pnpm --filter runtime-bench dev                          # bench page on 5176 (+ node scripts/serve-store.mjs, ?api=&cdn=)
```

The e2e suite needs `fixtures/qwen2.5-0.5b-instruct/` (from `dianome-server fixtures --intra --intra-blocks
0,20,21,22,23 --variant q8 --variant q4`) and the local chunk store in `./store`.
