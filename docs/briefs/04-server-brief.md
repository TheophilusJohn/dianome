# Phase 4 brief — server half, cost baseline, privacy probes

You are working in the `dianome` repo. Read first: the plan's Definitions (server cost formula, privacy band) in `docs/briefs/00-storage-partitioning-brief.md`'s sibling `docs/plan.md` if present, otherwise the section below; `docs/briefs/01-ingest.md` for the per-layer manifest; `docs/phase-1-notes.md` for the Qwen2.5-0.5B config. Python for `server/` (PyTorch); this is the second and last Python package alongside `ingest/`. Everything else stays TypeScript.

Hardware: this MacBook (Apple Silicon, 16 GB) with PyTorch on MPS. Device selection is `cuda` → `mps` → `cpu`, never hardcoded, because Phase 7 re-runs the benchmarks on a rented NVIDIA GPU. fp16 on MPS. Nothing in this phase needs more than the 0.5B model; write it so a 3B fp16 model also loads, and do not attempt 7B here.

Do not expose anything publicly and do not run the tunnel; those steps are Theo's.

## Goal

A server that starts a transformer at an arbitrary layer boundary from incoming hidden states, over a binary WebSocket protocol; a harness that measures GPU-seconds per token as a function of the split point; per-layer privacy probes on the 0.5B model; and reference activations that Phase 5a's WGSL kernels will be validated against.

## Definitions (fixed; do not reinterpret)

**Layer boundary.** "Split at N" means the client runs the embedding lookup and transformer blocks `0..N-1` and sends the *output of block N-1* — the input to block N — as fp16 `[T, d_model]`. The server runs blocks `N..L-1`, the final norm, `lm_head`, and sampling. `N = 0` means the client sends token ids and the server runs everything (the fully-server-side baseline). `N = L` means the client sends the output of the last block and the server runs only final norm + `lm_head` + sampling.

**Server cost.** `gpu_seconds_per_token(N) = server_busy_seconds(N) / tokens_generated`, where busy time is measured on the server around model execution only, excluding socket wait and JSON parsing, using `torch.cuda.synchronize()` / `torch.mps.synchronize()` before stopping the clock. `cost_per_1M_tokens(N) = gpu_seconds_per_token(N) × 1e6 × hourly_rate / 3600`. `hourly_rate` comes from `bench/rates.json` (GPU name, USD/hour, source URL, retrieval date); the harness never hardcodes it. Batch size 1, fixed prompt set, fixed generation length, three runs, median.

**Privacy band.** For each layer i (the hidden state leaving the client when N = i), two measures of top-1 recovery of the input token at the same position on held-out text: a linear probe (lower bound) and a small trained inversion decoder (upper bound). Both reported per layer with top-1 and top-5.

## Layout

```
server/
  pyproject.toml                name "dianome-server"; deps: torch, transformers (pinned), safetensors, numpy, websockets, msgspec or orjson; dev: pytest
  README.md
  dianome_server/
    model.py                    load HF model; PartialDecoder: run blocks a..b with a DynamicCache; final norm + lm_head; sampling
    reference_client.py         run embeddings + blocks 0..N-1 in PyTorch (used by tests, the harness, and fixtures)
    protocol.py                 binary frame encode/decode + message types
    session.py                  session state: N, KV cache, positions, busy-time accounting
    ws.py                       websockets server: auth, sessions, /plan status
    cli.py                      `dianome-server serve|bench|probes|fixtures`
  bench/
    cost.py                     the cost harness
    rates.json                  { "gpu": "...", "usd_per_hour": ..., "source": "...", "retrieved": "YYYY-MM-DD" } — Theo fills the real rate
    prompts.json                fixed prompt set, 8 prompts, 64–256 tokens each
  probes/
    collect.py                  per-layer activations on held-out text
    linear.py                   linear probe per layer
    inversion.py                small inversion decoder per layer
    report.py                   band table → docs/phase-4-notes.md section
  tests/
fixtures/qwen2.5-0.5b-instruct/  reference activations for Phase 5a (see below); gitignore the .npy files, commit the manifest + hashes
docs/phase-4-notes.md            measured numbers only
```

## PartialDecoder

Load with `transformers` (`Qwen2ForCausalLM`) in fp16 on the selected device. Do not rely on `model.forward` for partial runs; implement the block loop explicitly over `model.model.layers[a:b]` with a `DynamicCache`, computing rotary embeddings once per call the way the pinned transformers version does (check its `Qwen2Model.forward` for the exact `position_embeddings` argument shape and pass the same). Pin the transformers version in `pyproject.toml` and note it in the README, since this loop depends on internal APIs.

Correctness gate, as tests:
- For a 32-token prompt and every N in {0, 1, 8, 16, 23, 24}: `reference_client(N)` → `PartialDecoder(N..L)` produces logits within `max_abs_diff < 5e-2` and `argmax` identical to a full `model.forward` (fp16 tolerances; record the actual max diff in the notes).
- Decode consistency: 16 greedy tokens through the split path equal 16 greedy tokens from the full model.
- KV cache: after prefill + 3 decodes, the server-side cache length equals prompt + 3 for every server layer and zero for client layers.

## Protocol

Binary WebSocket frames. Frame = `u32 magic 0x444E4D31 ("DNM1")`, `u32 header_len`, JSON header, payload bytes. Little-endian fp16 payloads, row-major.

| type | direction | header | payload |
| --- | --- | --- | --- |
| `open` | c→s | `{model, N, max_ctx, sampling: {temperature, top_p, seed}}` | — |
| `opened` | s→c | `{session, L, d_model, boundary: "input_to_block_N"}` | — |
| `prefill` | c→s | `{T, positions: [start, end)}` | fp16 `[T, d_model]` (N ≥ 1) or int32 token ids `[T]` (N = 0) |
| `decode` | c→s | `{position}` | fp16 `[1, d_model]` or int32 `[1]` |
| `token` | s→c | `{id, position, busy_ms, done}` | optional fp16 top-k logits if requested |
| `stats` | c→s / s→c | `{}` / `{tokens, busy_seconds, gpu_seconds_per_token}` | — |
| `close` | either | `{}` | — |
| `error` | s→c | `{code, message}` | — |

Auth: `Authorization: Bearer <SPLIT_TOKEN>` on the WebSocket upgrade; `SPLIT_TOKEN` from env; refuse without it. Limits: one model loaded at a time, `max_ctx` ≤ 4096, at most 4 concurrent sessions (MPS is not going to batch), idle sessions closed after 120 s. `GET /plan` (plain HTTP on the same port) returns `{model, L, d_model, active_sessions, busy_fraction_60s}` for the Phase 5b planner.

## Cost harness (`bench/cost.py`)

For N in `{0, 4, 8, 12, 16, 20, 24}` on the 0.5B model (`L = 24`): for each prompt, run `reference_client(N)` (untimed — it stands in for the browser) then prefill + 128 greedy decode steps through the server path in-process (no socket), timing only server execution with device sync. Three runs, median per N. Output `bench/results/<device>-<model>-<date>.csv` with columns `N, prompt_tokens, gen_tokens, busy_seconds, gpu_seconds_per_token, cost_per_1M_tokens, hourly_rate, gpu`, and a Markdown table for the notes. Also report the ratio `cost(N) / cost(0)`. On the Mac the absolute seconds are Mac-specific; the harness must print the device name so no one mistakes them for GPU numbers. Phase 7 re-runs this on a rented GPU.

## Privacy probes

Data: a fixed held-out text set, ~50k tokens, from a public corpus with a permissive license (WikiText-103 test split via `datasets`, or a pinned public-domain text if `datasets` is unavailable). Record exactly what was used.

`collect.py`: run the full model once, store fp16 activations at every layer boundary (0..24, where boundary i = input to block i) for every position, plus the token ids, as memory-mapped `.npy`. On 50k tokens × 25 boundaries × 896 × 2 bytes ≈ 2.2 GB; keep it on disk, gitignored.

`linear.py` (lower bound): per boundary, a linear map `d_model → vocab` trained with cross-entropy to predict the token at the same position, 80/20 split by document, AdamW, a few epochs, early stop on held-out loss. 896 × 151,936 fp16 weights is ~272 MB per probe; train and evaluate one boundary at a time, keep only the metrics. Also report the zero-training baseline at every boundary: nearest neighbour against the (tied) embedding matrix, which is what boundary 0 trivially inverts.

`inversion.py` (upper bound): per boundary, a small attention decoder (2 layers, 4 heads, d = 512) that reads the activation sequence and predicts the token sequence, trained on the same split; same metrics. Keep it small enough that all 25 boundaries train in under two hours total on MPS; report training time.

`report.py`: table with columns `boundary, nn_top1, linear_top1, linear_top5, inversion_top1, inversion_top5, train_seconds`, written to `docs/phase-4-notes.md`. Every number from a run.

## Reference fixtures for Phase 5a

`dianome-server fixtures --model qwen2.5-0.5b-instruct --out fixtures/qwen2.5-0.5b-instruct/`: for one fixed 32-token prompt (stored as text + token ids): fp16 `.npy` of the embedding output and of every block's output (`block_00.npy` … `block_23.npy`), the final-norm output, the logits, the RoPE cos/sin tables for those positions, and a `fixtures.json` with shapes, dtypes, SHA-256 of each file, the transformers version, and the exact boundary definition from this brief. Phase 5a validates each WGSL layer against these before writing the next one, so they must be reproducible: running the command twice yields identical hashes on the same machine.

## Tests

pytest: protocol round-trip (every message type, payload dtype/shape), the three correctness gates above, session limits and idle timeout, busy-time accounting excludes a deliberate `asyncio.sleep` on the socket, `/plan` shape, fixtures reproducibility (two runs, same hashes), probe metrics on a tiny synthetic run (shape and range only, no real numbers).

## docs/phase-4-notes.md

Measured only: max_abs_diff per N from the correctness gate; the cost table and ratio curve on this Mac with the device name; the privacy band table with the dataset and split named; fixture hashes; transformers/torch versions; wall time for probes.

## Definition of done

- `pytest` passes in `server/`.
- `dianome-server serve` runs locally with `SPLIT_TOKEN` set; a small Python client in `tests/` opens a session at N = 8, prefills a prompt, decodes 16 tokens over the real WebSocket, and gets the same tokens as the full model greedy run.
- `dianome-server bench` produces the CSV and table on this Mac.
- `dianome-server probes` produces the band table for all 25 boundaries.
- `dianome-server fixtures` produces the fixture set with hashes, twice, identical.
- `docs/phase-4-notes.md` filled from those runs. Nothing deployed, no tunnel.
