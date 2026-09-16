# Phase 1 brief — chunk store, ingest script, manifest

You are working in the `dianome` repo. Phase 0 is done (see `docs/spikes/00-storage-partitioning.md`). Build the Phase 1 ingest pipeline described below. Python for `ingest/` only; everything else in the repo is TypeScript. Do not run any upload unless `--upload` is passed and the R2 env vars are set. Do not write measured sizes or error statistics anywhere except where this brief says to.

## Goal

A script that takes a Hugging Face safetensors model, produces per-layer artifacts in three variants (fp16, q8, q4), splits them into fixed 8 MB content-addressed physical chunks, writes a two-level manifest, verifies everything by reconstruction, and optionally uploads to Cloudflare R2. Plus a generic packer that chunks any directory of runtime-native files (ONNX for Transformers.js, MLC shards for WebLLM) into the same store. Plus a small manifest browser page.

Development model: `Qwen/Qwen2.5-0.5B-Instruct`. The pipeline must also work for 3B and 7B Qwen2.5 models without loading a whole model into memory.

## Layout to create

```
schemas/manifest.v1.json            JSON Schema for the manifest (shared by Python and TypeScript later)
ingest/                             Python package, the only Python in the repo
  pyproject.toml                    name "dianome-ingest"; deps: safetensors, numpy, huggingface_hub, boto3, jsonschema, click; dev: pytest
  README.md
  dianome_ingest/
    __init__.py
    cli.py                          `dianome-ingest` entry point: pack-model, pack-dir, verify, inspect, upload
    hf.py                           download + tensor iteration via safetensors safe_open, one tensor at a time
    quant.py                        fp16 / q8 / q4 encoders and decoders (numpy only)
    layout.py                       which tensors belong to which layer group; serialization order; alignment
    chunker.py                      8 MB physical chunking, SHA-256, dedup
    manifest.py                     manifest building + schema validation
    r2.py                           S3-API upload with skip-if-exists and Cache-Control metadata
  tests/                            pytest: chunker, quant round-trip, layout, manifest schema
apps/manifest-browser/              Vite + TypeScript, no framework; add "apps/*" to pnpm-workspace.yaml
docs/phase-1-notes.md               the ONE place measured numbers go (sizes, error stats, dedup) — written by running the pipeline
```

## Physical layer: chunks

- A chunk is exactly 8,388,608 bytes except the last chunk of a stream, which may be shorter. Chunk id = lowercase hex SHA-256 of its bytes. Stored at key `chunks/<sha256>` with no extension.
- Chunks are content-addressed and immutable. Identical bytes anywhere in any model or variant produce the same chunk and are stored once.
- Chunking operates on **streams**, not on the whole model. A stream is the serialized bytes of one layer group of one variant (defined below), or one file for the generic packer. Chunk boundaries restart at every stream, so a layer group whose bytes are identical across two variants (for example fp16 embeddings kept as-is in the q4 variant) deduplicates to the same chunks. Spell this out in the README; it is the reason the store dedups at all.
- The physical layer knows nothing about tensors. `chunker.py` takes an iterable of byte streams and returns, per stream, an ordered list of `(sha256, length)`.

## Logical layer: entries, layer groups, variants

**Layer groups**, in this fixed order for a decoder-only model: `embed`, `layer.0` … `layer.L-1`, `final_norm`, `lm_head`. A layer group is one stream per variant.

**Entries** are tensors (plus their quantization parameters) inside a layer group. Each entry records `name` (the original safetensors key), `group`, `role` (for example `attn.q_proj.weight`, `attn.q_proj.bias`, `mlp.gate_proj.weight`, `input_layernorm.weight`), `shape`, `storage` (see below), and `segments`: an ordered list of `{chunk, offset, length}` giving where its bytes live across one or more physical chunks. An entry may span chunk boundaries; the manifest says so and the client concatenates.

**Alignment:** every entry starts at a 256-byte-aligned offset within its layer-group stream (pad with zero bytes). Within a quantized entry, the sub-arrays (weights, scales, zero-points) are laid out in that order, each starting 256-byte-aligned, and the manifest records each sub-array's offset and length within the entry. Alignment is what lets the client copy an entry's bytes straight into GPU buffers later.

**Variants:** `fp16`, `q8`, `q4`. Rules:
- fp16 is the reference. Every other variant is derived from fp16 tensors, never from the original bf16/fp32 directly, so results are reproducible from the same fp16 bytes.
- Only 2-D linear weights are quantized: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`. Norm weights and all biases stay fp16 in every variant.
- `embed_tokens` and `lm_head` are fp16 in the fp16 and q4 variants and q8 in the q8 variant. They are never q4.
- If `tie_word_embeddings` is true in the model config, the `lm_head` group has no bytes of its own; its single entry references the `embed` group's segments and the manifest marks it `tied: true`.
- q8: symmetric per-output-channel int8. For a weight `W[out, in]`: `scale[o] = max(|W[o, :]|) / 127` (fp16), `q = round_half_even(W / scale)` clipped to [-127, 127], stored as int8 row-major `[out, in]` followed by fp16 `scale[out]`. If a row is all zeros, scale is 1.0.
- q4: asymmetric group-wise, group size 128 along the input dimension. Per group: `min`, `max` over the 128 values; `scale = (max - min) / 15` (fp16; if max == min, scale = 1.0); `zero = clip(round_half_even(-min / scale), 0, 15)`; `q = clip(round_half_even(W / scale) + zero, 0, 15)`. Storage: weights packed two nibbles per byte, element `i` in the low nibble of byte `i // 2` when `i` is even and the high nibble when odd, row-major `[out, in/2]`; then fp16 `scale[out, in/128]`; then uint8 `zero[out, in/128]` (one byte per group, value in the low 4 bits — simple over compact).
- **Assert, do not assume:** before quantizing, check `in % 128 == 0` for every tensor that will be q4 and fail with a message naming the tensor and its shape. Qwen2.5-0.5B passes (896, 4864, 128 are all multiples of 128); other models may not, and the correct behaviour is a clear error, not silent padding.
- Determinism: numpy only, `np.float16` arithmetic where the scheme says fp16, `np.round` (half to even) everywhere. Running the ingest twice on the same source revision must produce byte-identical chunks. Write a test that proves it on a synthetic tensor.

## Manifest

One JSON file per model per ingest, validated against `schemas/manifest.v1.json`. Write the schema first, then build to it. Shape:

```json
{
  "schema": 1,
  "id": "qwen2.5-0.5b-instruct",
  "source": { "repo": "Qwen/Qwen2.5-0.5B-Instruct", "revision": "<hf commit sha>" },
  "family": "qwen2",
  "config": { "hidden_size": 896, "num_hidden_layers": 24, "num_attention_heads": 14, "num_key_value_heads": 2, "intermediate_size": 4864, "vocab_size": 151936, "rms_norm_eps": 1e-6, "rope_theta": 1000000.0, "tie_word_embeddings": true, "max_position_embeddings": 32768 },
  "tokenizer": { "files": [ { "name": "tokenizer.json", "segments": [ … ] }, … ] },
  "chunk_size": 8388608,
  "chunks": { "<sha256>": { "bytes": 8388608 }, … },
  "variants": {
    "fp16": {
      "bytes": 0,
      "groups": [
        { "name": "layer.7", "bytes": 0, "chunks": [ "<sha256>", … ],
          "entries": [
            { "name": "model.layers.7.self_attn.q_proj.weight", "role": "attn.q_proj.weight", "shape": [896, 896],
              "storage": { "kind": "fp16" },
              "segments": [ { "chunk": "<sha256>", "offset": 0, "length": 1605632 } ] },
            { "name": "model.layers.7.mlp.gate_proj.weight", "role": "mlp.gate_proj.weight", "shape": [4864, 896],
              "storage": { "kind": "q4", "group_size": 128,
                           "parts": { "weights": { "offset": 0, "length": 0 }, "scales": { "offset": 0, "length": 0 }, "zeros": { "offset": 0, "length": 0 } } },
              "segments": [ … ] }
          ] }
      ]
    },
    "q8": { … }, "q4": { … }
  }
}
```

- `chunks` is the global physical table; group `chunks` lists are ordered and are the download order for that group.
- `bytes` fields are sums, computed, and must equal the sum of the listed chunks' lengths — verify this in `verify`.
- The tokenizer files (`tokenizer.json`, `tokenizer_config.json`, `generation_config.json`, `config.json`) are stored as plain file streams in the same chunk store, one stream each, referenced from `tokenizer.files`.
- The manifest itself is stored at two keys: `manifests/<id>/latest.json` (mutable pointer) and `manifests/<id>/<sha256-of-manifest>.json` (immutable). The manifest's own hash is computed over its canonical JSON (sorted keys, no whitespace).

## Generic packer (runtime-native family)

`dianome-ingest pack-dir --dir <path> --id <artifact-id> --runtime <webllm|transformersjs|other>`: every file under the directory becomes a logical entry named by its relative path, chunked as one stream per file, with the same manifest envelope but `variants` replaced by a single `files` list. This is how MLC shards and ONNX graphs enter the store. Test it against the ONNX export at `onnx-community/Qwen2.5-0.5B-Instruct` on Hugging Face (download only the `onnx/` directory and the tokenizer files). Do not run any MLC conversion in this phase.

## CLI

```
dianome-ingest pack-model --repo Qwen/Qwen2.5-0.5B-Instruct --id qwen2.5-0.5b-instruct --variants fp16,q8,q4 --out ./store [--revision <sha>] [--upload]
dianome-ingest pack-dir   --dir ./onnx --id qwen2.5-0.5b-instruct-onnx --runtime transformersjs --out ./store [--upload]
dianome-ingest verify     --store ./store --id <id> [--against-hf]     # hashes, byte sums, schema; with --against-hf, reconstruct every tensor and compare
dianome-ingest inspect    --store ./store --id <id>                    # sizes per variant/group, dedup stats, chunk counts
dianome-ingest upload     --store ./store --id <id>                    # what --upload does, standalone
```

`./store` mirrors the R2 key layout (`chunks/`, `manifests/`) so a static server pointed at it behaves like the bucket.

**Memory:** iterate tensors with `safetensors.safe_open`, one tensor at a time; never load a whole model. A 7B fp16 model is ~15 GB and must ingest on a 16 GB machine.

**verify --against-hf** reports per entry: fp16 exact match (bit-equal); q8 and q4 max absolute error and relative RMS error `||W - deq(q)|| / ||W||`. It also asserts that decoding what the encoder wrote reproduces exactly what the encoder intended (encoder/decoder consistency), which is separate from quantization error.

## R2 upload

- boto3 S3 client against `https://<account>.r2.cloudflarestorage.com`, credentials from env `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET`. Fail early with a clear message if any is unset when `--upload` is used.
- Chunks: `head_object` first; skip if present. Upload with `Content-Type: application/octet-stream` and `Cache-Control: public, max-age=31536000, immutable`. These headers are served verbatim when R2 is fronted by a custom domain in Phase 2, with no Worker on the hot path, so they must be set here.
- Manifests: `Content-Type: application/json`; `latest.json` with `Cache-Control: public, max-age=60`; the hashed one immutable like chunks.
- Upload concurrency 8. Print counts: uploaded, skipped, bytes.

## Manifest browser (apps/manifest-browser)

Vite + TypeScript, no framework, plain DOM. Input: a manifest URL (default: `./store/manifests/<id>/latest.json` when served locally). Shows: model id, source repo and revision, config summary; per variant: total bytes and chunk count; a per-group table (group, bytes, chunks, entries); dedup: number of chunks shared between variants and bytes saved; a bar per group proportional to bytes, in download order. No styling beyond readable defaults. Serve with `pnpm --filter manifest-browser dev` and a static server on `./store` with CORS.

## docs/phase-1-notes.md

Written by running the pipeline on Qwen2.5-0.5B-Instruct, not by hand: bytes per variant, chunks per variant, dedup savings, q8 and q4 max-abs and rel-RMS error per role (aggregated over layers as min/median/max), ingest wall time, and the exact commit and HF revision used. Every number in this file comes from a command's output; say which command.

## Definition of done

- `pytest` passes: chunker (boundaries, last-chunk length, hash determinism, dedup), quant (encode/decode consistency, determinism, the divisibility assertion fires on a 900-wide tensor), layout (group order, 256-byte alignment, tied lm_head), manifest (schema validation of a generated manifest, byte sums).
- `pack-model` on Qwen2.5-0.5B-Instruct produces fp16, q8, q4 in `./store`; `verify --against-hf` passes; `inspect` shows the embed group's chunks shared between fp16 and q4.
- `pack-dir` on the ONNX export produces a valid manifest with `files`.
- `upload` skips everything on a second run (all chunks already present).
- The manifest browser loads the local manifest and renders every section above.
- `docs/phase-1-notes.md` exists and every number in it is traceable to a command.
