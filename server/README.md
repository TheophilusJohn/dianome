# dianome-server

Phase 4: the server half of split inference. It starts a Qwen2 transformer at an
arbitrary layer boundary from incoming hidden states over a binary WebSocket
protocol, plus the cost harness (GPU-seconds per token vs. split point), the
per-layer privacy probes, and the reference activations Phase 5a's WGSL kernels
are validated against. Second and last Python package alongside `ingest/`.

## Install

```sh
cd server
uv venv --python 3.12 .venv
uv pip install -p .venv/bin/python -e ".[dev]"
.venv/bin/pytest
```

`transformers` is pinned to **4.57.1**: `PartialDecoder` re-implements
`Qwen2Model.forward`'s block loop over `model.model.layers[a:b]` with a
`DynamicCache` and passes `position_embeddings=(cos, sin)` from
`model.model.rotary_emb` the way that version does. Bump the pin only after
re-reading `Qwen2Model.forward` and re-running the correctness gates.

Device selection is `cuda` → `mps` → `cpu` (`DIANOME_DEVICE` or `--device`
overrides), fp16 everywhere. Written for the 0.5B model; a 3B fp16 model loads
through the same code (`--model qwen2.5-3b-instruct`). 7B is out of scope here.

## Boundary definition

"Split at N" means the client runs the embedding lookup and blocks `0..N-1` and
sends the output of block `N-1` (the input to block `N`) as fp16 `[T, d_model]`.
The server runs blocks `N..L-1`, the final norm, `lm_head` and sampling.
`N = 0`: the client sends token ids. `N = L`: the server runs only
norm + `lm_head` + sampling.

## Commands

```sh
SPLIT_TOKEN=... .venv/bin/dianome-server serve   [--model qwen2.5-0.5b-instruct] [--host 127.0.0.1] [--port 8765]
.venv/bin/dianome-server fixtures --out ../fixtures/qwen2.5-0.5b-instruct/
.venv/bin/dianome-server bench    [--splits 0,4,8,12,16,20,24] [--gen-tokens 128] [--runs 3]
.venv/bin/dianome-server probes   [--tokens 50000] [--skip-collect] [--notes ../docs/phase-4-notes.md]
.venv/bin/dianome-server linear-probe [--train-tokens 500000] [--seed 1] [--max-epochs 3] [--notes ../docs/phase-4-notes.md]
```

`serve` refuses to start without `SPLIT_TOKEN`. Every request, the WebSocket
upgrade and the plain-HTTP `GET /plan`, must carry
`Authorization: Bearer <SPLIT_TOKEN>`. Limits: one model per process,
`max_ctx ≤ 4096`, at most 4 concurrent sessions, idle sessions closed after
120 s. `GET /plan` returns `{model, L, d_model, active_sessions, busy_fraction_60s}`.
Nothing here exposes a port beyond localhost or runs a tunnel.

A small client for the real socket lives in `tests/ws_client.py`:

```sh
.venv/bin/python -m tests.ws_client --token $SPLIT_TOKEN --N 8 --prompt "The capital of France is" --steps 16
```

## Protocol

Frame = `u32 magic 0x444E4D31` | `u32 header_len` | JSON header | payload, both
u32 little-endian (so the wire starts `31 4D 4E 44`). Payloads are little-endian
row-major fp16 `[T, d_model]`, or int32 `[T]` token ids when `N = 0`.

| type | dir | header | payload |
| --- | --- | --- | --- |
| `open` | c→s | `{model, N, max_ctx, sampling: {temperature, top_p, seed}, logits_topk?}` | — |
| `opened` | s→c | `{session, L, d_model, boundary: "input_to_block_N"}` | — |
| `prefill` | c→s | `{T, positions: [start, end)}` | fp16 `[T, d_model]` or int32 `[T]` |
| `decode` | c→s | `{position}` | fp16 `[1, d_model]` or int32 `[1]` |
| `token` | s→c | `{id, position, busy_ms, done, topk_ids?}` | fp16 `[k]` top-k logits if `logits_topk` was set |
| `stats` | c→s / s→c | `{}` / `{tokens, busy_seconds, gpu_seconds_per_token}` | — |
| `close` | either | `{}` | — |
| `error` | s→c | `{code, message}` | — |

Positions must be contiguous: a `prefill` must start at the number of positions
the session has already seen, and a `decode` at `position` must equal that
count. `token.position` is the position the returned token occupies when it is
fed back. `done` is set on an EOS id or when `max_ctx` is reached. `temperature
= 0` is greedy.

Busy time (`busy_ms`, `stats.busy_seconds`) is measured around model execution
only, with `torch.cuda.synchronize()` / `torch.mps.synchronize()` before the
clock stops; socket waits and header parsing are outside it.

## Cost harness

`bench/cost.py`: for each N, each prompt in `bench/prompts.json` (8 prompts,
64–256 tokens), run the reference client untimed, then prefill + 128 greedy
decode steps through the server half in-process, timing only server execution
with a device sync. Three runs, median per N. Writes
`bench/results/<device>-<model>-<date>.csv` (columns
`N, prompt_tokens, gen_tokens, busy_seconds, gpu_seconds_per_token,
cost_per_1M_tokens, hourly_rate, gpu`, then `prefill_seconds,
decode_seconds_per_token, runs, device, ratio_vs_N0`), a `-detail.csv` with
every (N, run, prompt), a `.json`, and a Markdown table. `gen_tokens` counts the
token the prefill produces plus the 128 decode steps.

`hourly_rate` comes from `bench/rates.json` (`gpu`, `usd_per_hour`, `source`,
`retrieved`); it is never hardcoded. Until Theo fills a real rate,
`usd_per_hour` is `null`, `cost_per_1M_tokens` is left empty and the harness
warns. The `cost(N)/cost(0)` ratio does not depend on the rate. The harness
prints the device name; Mac (MPS) seconds are Mac-specific, not GPU numbers.

## Privacy probes

`probes/`: `collect.py` runs the full model once over the held-out text
(WikiText-103 test split via `datasets`, every document capped at 1024 tokens,
~50k tokens) and stores fp16 activations at every boundary `0..L` as a
memory-mapped `probes/data/acts.npy` (gitignored, ~2.2 GB). Per boundary:
`linear.py` trains a linear map `d_model → vocab` with cross-entropy (AdamW,
early stop on held-out loss) and reports the zero-training nearest-neighbour
baseline against the tied embedding matrix; `inversion.py` trains a 2-layer,
4-head, d = 512 attention decoder over 128-token windows. The 80/20 split is by
document. `report.py` writes the band table into `docs/phase-4-notes.md`
between `<!-- probes:start -->` / `<!-- probes:end -->`. Results accumulate in
`probes/results/band.json`, so an interrupted run resumes per boundary.

`linear-probe` (`probes/linear500k.py`) re-runs only the linear probe with a
larger training set: ~500k tokens sampled with a fixed seed from the WikiText-103
*train* split (test-split titles excluded, so train and test are disjoint by
construction), collected into a second store `probes/data-train500k/` (22 GB,
gitignored). The held-out set is exactly the Phase 4 one. It reports the
coverage of held-out tokens by the training vocabulary and coverage-normalised
top-1/top-5, resumes per boundary from `probes/results/linear-500k.json`, and
writes its own section into the notes between `<!-- linear500k:start -->` /
`<!-- linear500k:end -->`.

## Fixtures

`dianome-server fixtures` writes, for one fixed 32-token prompt, fp16 `.npy` of
the embedding output (`embed.npy` = boundary 0), every block's output
(`block_00.npy` … `block_23.npy`; `block_NN` = boundary NN+1), the final-norm
output, the logits, the RoPE cos/sin tables, and `fixtures.json` with shapes,
dtypes, SHA-256 per file, versions and the boundary definition. The `.npy`
files are gitignored; the manifest with hashes is committed. Two runs on the
same machine yield identical hashes (`tests/test_fixtures.py`).

Phase 5a additions (same command): `tokenizer_cases.json` (200 strings with HF
token ids and decodes, gate 1), `greedy.json` + `decode_steps.npy` (16 greedy
tokens with their fp16 top-2 logit gaps, and the hidden state at every boundary
for each decode step, gate 5), `--intra` (`intra/*.npy`, every stage of block 0
captured with forward hooks; `--intra-blocks 0,20,21,22,23` adds `intra_<i>/`),
and `--variant q8|q4` (`<variant>/block_NN.npy`, `<variant>/greedy.json`: PyTorch
with the client's blocks and embedding replaced by the dequantised store bytes,
read through `dianome_ingest.quant`, so the ingest package must be installed in
this venv: `uv pip install --python .venv/bin/python -e ../ingest`). The
WebSocket server also accepts the token as `?token=` on the URL, because browsers
cannot set headers on an upgrade.

## Tests

`tests/test_correctness.py` holds the three gates from the brief (logits within
`5e-2` and identical argmax vs. `model.forward` for N ∈ {0, 1, 8, 16, 23, 24},
16-token greedy decode equality, KV cache lengths). The measured
`max_abs_diff` per N is written to `bench/results/gate-max-abs-diff.json`.
`tests/test_server.py` runs the real server on a random port: auth, `/plan`
shape, the 4-session limit, idle timeout, busy-time accounting with a
deliberate `asyncio.sleep` on the socket path, and the N = 8 client run that
must match the full model's greedy tokens.
