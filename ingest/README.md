# dianome-ingest

Phase 1 ingest pipeline: a Hugging Face safetensors model → per-layer artifacts in
three variants (fp16, q8, q4) → fixed 8 MB content-addressed chunks → a two-level
manifest validated against `schemas/manifest.v1.json` → optional upload to
Cloudflare R2. Plus a generic packer for runtime-native files (ONNX, MLC shards).

This is the only Python in the repo.

## Install

```sh
cd ingest
uv venv --python 3.12 .venv
uv pip install -p .venv/bin/python -e ".[dev]"
.venv/bin/pytest
```

Dependencies: `safetensors`, `numpy`, `huggingface_hub`, `boto3`, `jsonschema`,
`click`, plus `ml_dtypes`. The last one is needed because the safetensors
numpy backend has no bf16 dtype of its own; importing `ml_dtypes` registers
`np.dtype("bfloat16")` so `safe_open(..., framework="np")` can return bf16
tensors and row slices. The bf16 values are widened bit-exactly to fp32 and
rounded to fp16 (nearest even) in `quant.to_fp16`.

## Commands

```sh
dianome-ingest pack-model --repo Qwen/Qwen2.5-0.5B-Instruct --id qwen2.5-0.5b-instruct --variants fp16,q8,q4 --out ./store [--revision <sha>] [--upload]
dianome-ingest pack-dir   --dir ./onnx-export --id qwen2.5-0.5b-instruct-onnx --runtime transformersjs --out ./store [--repo <hf repo> --revision <sha>] [--upload]
dianome-ingest verify     --store ./store --id <id> [--against-hf]
dianome-ingest inspect    --store ./store --id <id>
dianome-ingest upload     --store ./store --id <id>
```

`./store` mirrors the R2 key layout (`chunks/<sha256>`, `manifests/<id>/latest.json`,
`manifests/<id>/<sha256>.json`), so a static server pointed at it behaves like
the bucket. Nothing is uploaded unless `--upload` is passed (or the `upload`
command is run) **and** `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`
and `R2_BUCKET` are all set; a missing variable fails before any packing starts.

To fetch the ONNX export for `pack-dir` (only `onnx/` and the tokenizer files):

```sh
.venv/bin/python -c "from huggingface_hub import snapshot_download as s; s('onnx-community/Qwen2.5-0.5B-Instruct', allow_patterns=['onnx/*','tokenizer.json','tokenizer_config.json','config.json','generation_config.json'], local_dir='../onnx-export')"
```

## Physical layer: chunks (`chunker.py`)

A chunk is exactly 8,388,608 bytes, except the last chunk of a stream, which
may be shorter. Its id is the lowercase hex SHA-256 of its bytes, and it is
stored at `chunks/<sha256>` with no extension. Chunks are immutable and
content-addressed: identical bytes anywhere, in any model or variant, produce
the same chunk and are stored once.

**Chunking operates on streams, not on the whole model.** A stream is the
serialized bytes of one layer group of one variant (or one file for the generic
packer). Chunk boundaries restart at every stream. This is the reason the store
dedups at all: the embed group is q8 in both the q8 and the q4 variant, so
the two streams are byte-identical and cut into the same chunks regardless of
what precedes them in their variant. If chunking ran over a concatenated model,
a one-byte difference in an earlier group would shift every later boundary and
nothing downstream would match. The same mechanism makes `final_norm` (fp16 in
every variant) a single shared chunk.

`chunker.py` knows nothing about tensors: `chunk_streams(streams, sink)` takes
an iterable of byte streams and returns, per stream, an ordered list of
`(sha256, length)`. `StreamChunker` is the incremental form used by the packer,
which writes the three variant streams of a group in lock-step so each source
tensor is read once.

## Logical layer (`layout.py`)

Layer groups, in this fixed order for a decoder-only model:
`embed`, `layer.0` … `layer.L-1`, `final_norm`, `lm_head`. A layer group is one
stream per variant; the group's chunk list in the manifest is its download order.

Tensor → group: `model.embed_tokens.*` → embed; `model.layers.N.*` → layer.N;
`model.norm.*` → final_norm; `lm_head.*` → lm_head. Any other key is an error.
Roles strip the group prefix and rename `self_attn` to `attn`
(`model.layers.7.self_attn.q_proj.weight` → `attn.q_proj.weight`).

Serialization order inside a layer group: `input_layernorm`, q/k/v/o
(weight then bias each), `post_attention_layernorm`, gate/up/down. Unknown
roles follow, sorted by name.

**Alignment.** Every entry starts at a 256-byte-aligned offset within its
stream (zero padding). Inside a quantized entry the sub-arrays are laid out as
weights, scales, zeros, each starting 256-byte aligned; the manifest records
each part's offset and length relative to the entry start. There is no trailing
padding after the last entry. An entry may span chunk boundaries; its
`segments` list says where its bytes live and the client concatenates.

**Tied lm_head.** With `tie_word_embeddings: true` the `lm_head` group has no
bytes (`bytes: 0`, `chunks: []`, `tied: true`) and its single entry
`lm_head.weight` carries `tied: true` and the segments of the embed entry of the
same variant. A stray `lm_head.weight` tensor in a tied checkpoint is ignored.

## Variants (`quant.py`)

- fp16 is the reference. q8 and q4 are derived from the fp16 array, never from
  the original bf16/fp32, so they are reproducible from the same fp16 bytes.
- Only 2-D linear weights are quantized: `q_proj`, `k_proj`, `v_proj`,
  `o_proj`, `gate_proj`, `up_proj`, `down_proj`. Norm weights and all biases
  stay fp16 in every variant. `embed_tokens` and `lm_head` are fp16 in the fp16
  variant and q8 in both the q8 and q4 variants; they are never q4.
- **q8**: symmetric per-output-channel int8. `scale[o] = max|W[o,:]| / 127` in
  fp16 (1.0 for an all-zero row); `q = round_half_even(W / scale)` clipped to
  [-127, 127]; stored as int8 `[out, in]` then fp16 `scale[out]`.
- **q4**: asymmetric, group size 128 along `in`. Per group `min`, `max`;
  `scale = (max - min) / 15` in fp16; if `max == min`, `scale = |min|`, or 1.0
  when `min == 0`;
  `zero = clip(round_half_even(-min / scale), 0, 15)`;
  `q = clip(round_half_even(W / scale) + zero, 0, 15)`. Stored as nibbles two
  per byte (element `i` in the low nibble of byte `i // 2` when `i` is even,
  high nibble when odd), row-major `[out, in/2]`, then fp16 `scale[out, in/128]`,
  then uint8 `zero[out, in/128]`.
- The ratio `W / scale` and the decoders are computed in float32; min/max/abs
  are exact in fp16; scale uses `np.float16` arithmetic; `np.round` is
  half-to-even everywhere. Running the ingest twice on the same revision yields
  byte-identical chunks and the same manifest hash (`tests/test_quant.py`,
  `tests/test_layout.py::test_pack_is_deterministic`).
- **Assert, do not assume**: before anything is written, every tensor that
  would be q4 is checked for `in % 128 == 0`; a failure names the tensor and
  its shape. Qwen2.5-0.5B passes (896, 4864 and 128 are multiples of 128).
- Constant groups: when all 128 values are equal, `scale = |min|` (1.0 for an
  all-zero group), so `q` is 1 with `zero = 0` for a positive constant and 0
  with `zero = 1` for a negative one, and the group decodes exactly.

## Manifest (`manifest.py`)

One JSON document per artifact per ingest, validated against
`schemas/manifest.v1.json` on build and on verify. `chunks` is the global
physical table; every `bytes` field is a computed sum that `verify` re-checks.
The manifest is stored at `manifests/<id>/latest.json` (mutable pointer) and
`manifests/<id>/<sha256>.json` (immutable). Its hash is the SHA-256 of its
canonical JSON (sorted keys, no whitespace, ASCII escaping), and both files are
written as exactly those canonical bytes, so the hashed file hashes to its name.

The tokenizer files (`tokenizer.json`, `tokenizer_config.json`,
`generation_config.json`, `config.json`) are stored as plain file streams in the
same chunk store, one stream each, under `tokenizer.files`.

`pack-dir` writes the same envelope with `variants` replaced by `files`
(every file under the directory, named by its relative path, one stream each)
and a `runtime` field.

## verify

Without `--against-hf`: schema; the hashed manifest file exists and matches
the canonical hash; every chunk in the table exists with the right size and
SHA-256; group bytes equal the sum of the listed chunks; variant bytes equal the
sum of group bytes; every entry's segments are contiguous in its group stream,
start 256-byte aligned, and sum to the length implied by its shape and storage;
part offsets match the layout; tied flags agree with the config.

With `--against-hf`: the source revision is re-resolved and must match; every
entry is reconstructed from its segments and decoded. fp16 entries must be
bit-equal to the fp16 conversion of the source. For q8 and q4 the decoded
parts must equal what the encoder produces from the same fp16 array
(encoder/decoder consistency, which is separate from quantization error), and
the max absolute error and relative RMS error `||W - deq(q)|| / ||W||` are
reported per entry and aggregated per role (min / median / max over layers).

## R2 upload (`r2.py`)

boto3 S3 client against `https://<account>.r2.cloudflarestorage.com`. Chunks:
`head_object` first, skip if present; otherwise put with
`Content-Type: application/octet-stream` and
`Cache-Control: public, max-age=31536000, immutable`. Manifests:
`application/json`; `latest.json` with `Cache-Control: public, max-age=60`,
the hashed one immutable like chunks. Concurrency 8. Prints uploaded, skipped
and bytes; a second run skips every chunk.
