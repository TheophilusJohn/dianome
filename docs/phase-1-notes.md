# Phase 1 notes — ingest of Qwen2.5-0.5B-Instruct

Every number below is pasted from the output of the command shown above it,
run on 2026-09-16 on a MacBook Pro (Apple Silicon, 16 GB). Nothing here is
typed by hand.

- Repo commit: `27ea66c87e50aaa70848205e983a440918d31685` ("Phase 1: ingest pipeline, manifest schema,
  manifest browser"); `pack-model` prints the commit it ran under, below.
- HF source: `Qwen/Qwen2.5-0.5B-Instruct` @ `7ae557604adf67be50417f59c2c2f167def9a775`.
- Python 3.12 venv at `ingest/.venv` (`uv`), numpy 2.5.3, safetensors 0.8.0.

## Tests

```
$ cd ingest && .venv/bin/pytest -q
41 passed in 0.85s
```

## pack-model: bytes per variant, chunks per variant, ingest wall time

```
$ ingest/.venv/bin/dianome-ingest pack-model --repo Qwen/Qwen2.5-0.5B-Instruct --id qwen2.5-0.5b-instruct --variants fp16,q8,q4 --out ./store

source Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775  family=qwen2 layers=24 tied=True  variants=fp16,q8,q4
pack-model qwen2.5-0.5b-instruct
  source      Qwen/Qwen2.5-0.5B-Instruct @ 7ae557604adf67be50417f59c2c2f167def9a775
  git commit  27ea66c87e50aaa70848205e983a440918d31685
  manifest    store/manifests/qwen2.5-0.5b-instruct/latest.json  sha256=66b487935d4297460fcc601a3cf373c61808bdb6ca15156c3ae3fff1d3ddbae4
  chunk table 223 unique chunks, 1,599.9 MiB
  variant fp16  bytes=  988,065,536 (942.3 MiB)  chunks=130
  variant q8    bytes=  495,016,448 (472.1 MiB)  chunks=66
  variant q4    bytes=  323,893,760 (308.9 MiB)  chunks=42
  tokenizer   4 files, 7,039,851 bytes
  wall time   6.1 s
```

Ingest wall time: 6.1 s (the `wall time` line above; the safetensors file
was already in the HF cache, so this is conversion + quantization + hashing +
writing 1.6 GB of chunks).

Determinism: a second `pack-model` run into a fresh directory, then `cmp` of
the two `latest.json` files and `diff` of the second store's chunk ids against
the first manifest's chunk table:

```
manifests byte-identical
chunk id sets identical
```

## inspect: sizes per group, dedup savings

Rows for `layer.1` … `layer.22` are elided; they are identical to `layer.0`
and `layer.23` within each variant (see the full output by running the command).

```
$ ingest/.venv/bin/dianome-ingest inspect --store ./store --id qwen2.5-0.5b-instruct
inspect qwen2.5-0.5b-instruct  source=Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775  manifest sha256=66b487935d4297460fcc601a3cf373c61808bdb6ca15156c3ae3fff1d3ddbae4
  chunk table: 223 unique chunks, 1,677,573,483 bytes (1,599.9 MiB), chunk_size=8388608
  family=qwen2  layers=24  tied=True

variant fp16: bytes=988,065,536 (942.3 MiB)  chunks=130 listed, 130 unique
  group                bytes  chunks  entries
  embed          272,269,312      33        1
  layer.0         29,824,768       4       12
  layer.23        29,824,768       4       12
  final_norm           1,792       1        1
  lm_head                  0       0        1  (tied -> embed)

variant q8: bytes=495,016,448 (472.1 MiB)  chunks=66 listed, 66 unique
  group                bytes  chunks  entries
  embed          136,438,528      17        1
  layer.0         14,940,672       2       12
  layer.23        14,940,672       2       12
  final_norm           1,792       1        1
  lm_head                  0       0        1  (tied -> embed)

variant q4: bytes=323,893,760 (308.9 MiB)  chunks=42 listed, 42 unique
  group                bytes  chunks  entries
  embed          136,438,528      17        1
  layer.0          7,810,560       1       12
  layer.23         7,810,560       1       12
  final_norm           1,792       1        1
  lm_head                  0       0        1  (tied -> embed)

dedup across variants
  chunks listed across variants: 238 (1,806,975,744 bytes)
  unique chunks:                 219 (1,670,533,632 bytes)
  chunks shared by >1 variant:   18 (136,440,320 bytes)
  bytes saved by dedup:          136,442,112 (130.1 MiB)
  fp16∩q8      shared chunks=1 bytes=1,792
  fp16∩q4      shared chunks=1 bytes=1,792
  q8∩q4        shared chunks=18 bytes=136,440,320
  groups whose chunk lists are identical across variants:
    embed        q8 = q4  (17 chunks, 136,438,528 bytes)
    final_norm   fp16 = q8 = q4  (1 chunks, 1,792 bytes)
  embed group chunk ids (first 3 of each variant):
    fp16  n=33  6c5634e9870b 05acd699aba8 a8208316057b …
    q8    n=17  678e30608ab7 957658d9f8a5 44ed16f94485 …
    q4    n=17  678e30608ab7 957658d9f8a5 44ed16f94485 …
  tokenizer files: tokenizer.json (7,031,645 B, 1 chunks), tokenizer_config.json (7,305 B, 1 chunks), generation_config.json (242 B, 1 chunks), config.json (659 B, 1 chunks)
```

The `embed` group is q8 in both the q8 and q4 variants, so its 17 chunks are
the same ids in both, and `final_norm` is one chunk shared by all three
variants. That is the whole dedup: 238 listed chunks become 219 unique,
saving 136,442,112 bytes.

## verify --against-hf: q8 and q4 error per role

Per-entry lines (873 of them) are elided; the store checks and the per-role
aggregate are shown verbatim.

```
$ ingest/.venv/bin/dianome-ingest verify --store ./store --id qwen2.5-0.5b-instruct --against-hf
  [ok] schema: manifest.v1.json
  [ok] manifest hash: manifests/qwen2.5-0.5b-instruct/66b487935d4297460fcc601a3cf373c61808bdb6ca15156c3ae3fff1d3ddbae4.json
  [ok] chunk hashes & sizes: 223 chunks, 1677573483 bytes
  [ok] variant fp16 sums, layout, alignment: 27 groups, 988065536 bytes
  [ok] variant q8 sums, layout, alignment: 27 groups, 495016448 bytes
  [ok] variant q4 sums, layout, alignment: 27 groups, 323893760 bytes
  [ok] tokenizer files byte sums: 4 files
  … 873 per-entry lines …
per-role quantization error (aggregated over layers: min / median / max)
  variant role                                 n  max_abs min     median        max   rel_rms min     median        max
  q4      attn.k_proj.weight                  24     0.008423     0.0106    0.06171        0.1025     0.1112     0.1259
  q4      attn.o_proj.weight                  24      0.01257    0.01915    0.04126         0.104     0.1081     0.1171
  q4      attn.q_proj.weight                  24      0.01042    0.01335    0.07373        0.1048     0.1098     0.1177
  q4      attn.v_proj.weight                  24     0.004524   0.006805    0.01562        0.1067     0.1113     0.1611
  q4      embed_tokens.weight                  1    0.0007887  0.0007887  0.0007887      0.009008   0.009008   0.009008
  q4      lm_head.weight                       1    0.0007887  0.0007887  0.0007887      0.009008   0.009008   0.009008
  q4      mlp.down_proj.weight                24      0.01445    0.01855     0.0318        0.1041     0.1084      0.115
  q4      mlp.gate_proj.weight                24      0.01428    0.02211    0.03491        0.1032     0.1062     0.1122
  q4      mlp.up_proj.weight                  24     0.009872    0.01776     0.0481        0.1034     0.1069     0.1107
  q8      attn.k_proj.weight                  24    0.0006456  0.0008101   0.003998      0.008277   0.009575    0.01161
  q8      attn.o_proj.weight                  24    0.0008097   0.001318   0.003628      0.008812   0.009961    0.01281
  q8      attn.q_proj.weight                  24    0.0008297   0.001011   0.006638      0.008612    0.00955    0.01137
  q8      attn.v_proj.weight                  24    0.0003247  0.0004921   0.001175      0.008593   0.009537    0.01473
  q8      embed_tokens.weight                  1    0.0007887  0.0007887  0.0007887      0.009008   0.009008   0.009008
  q8      lm_head.weight                       1    0.0007887  0.0007887  0.0007887      0.009008   0.009008   0.009008
  q8      mlp.down_proj.weight                24     0.001198   0.001732   0.002457       0.01069    0.01199    0.01486
  q8      mlp.gate_proj.weight                24     0.001183   0.001743   0.002991      0.008343   0.008876   0.009696
  q8      mlp.up_proj.weight                  24    0.0008984   0.001663    0.00309      0.008317   0.008954   0.009659

fp16 entries bit-equal to source: 533/533; quantized entries encoder/decoder-consistent: 340/340; entries checked: 873
verify --against-hf: ok
```

## pack-dir: ONNX export (Transformers.js runtime family)

Source: `onnx-community/Qwen2.5-0.5B-Instruct` @ `cc5cc01a65cc3ff17bdb73a7de33d879f62599b0`,
only `onnx/` and the tokenizer files downloaded (6.6 GB) into `./onnx-export`.

```
$ ingest/.venv/bin/dianome-ingest pack-dir --dir ./onnx-export --id qwen2.5-0.5b-instruct-onnx --runtime transformersjs --repo onnx-community/Qwen2.5-0.5B-Instruct --revision cc5cc01a65cc3ff17bdb73a7de33d879f62599b0 --out ./store
  config.json  678 bytes  1 chunks
  generation_config.json  242 bytes  1 chunks
  onnx/model.onnx  1,993,796,793 bytes  238 chunks
  onnx/model_bnb4.onnx  763,794,004 bytes  92 chunks
  onnx/model_fp16.onnx  997,354,499 bytes  119 chunks
  onnx/model_int8.onnx  512,096,557 bytes  62 chunks
  onnx/model_q4.onnx  786,156,820 bytes  94 chunks
  onnx/model_q4f16.onnx  483,003,582 bytes  58 chunks
  onnx/model_quantized.onnx  512,096,557 bytes  62 chunks
  onnx/model_uint8.onnx  512,096,636 bytes  62 chunks
  tokenizer.json  7,031,673 bytes  1 chunks
  tokenizer_config.json  7,306 bytes  1 chunks
pack-dir qwen2.5-0.5b-instruct-onnx  runtime=transformersjs
  source      onnx-community/Qwen2.5-0.5B-Instruct @ cc5cc01a65cc3ff17bdb73a7de33d879f62599b0
  manifest    store/manifests/qwen2.5-0.5b-instruct-onnx/latest.json  sha256=ba8faf7d7ffcf324d68b81eb7c517cad4df8b0012a42dc7b830fd6ace24a09a3
  files       12, 6,567,435,347 bytes, 791 chunks listed, 728 unique
  wall time   5.1 s
```

```
$ ingest/.venv/bin/dianome-ingest verify --store ./store --id qwen2.5-0.5b-instruct-onnx
  [ok] schema: manifest.v1.json
  [ok] manifest hash: manifests/qwen2.5-0.5b-instruct-onnx/ba8faf7d7ffcf324d68b81eb7c517cad4df8b0012a42dc7b830fd6ace24a09a3.json
  [ok] chunk hashes & sizes: 728 chunks, 6046950182 bytes
  [ok] files byte sums: 12 files
verify: ok
```

791 listed vs 728 unique (63 duplicate listings): `model_int8.onnx` and
`model_quantized.onnx` are byte-identical files (same sha256, 62 chunks each,
all shared), and one of those chunks, the first, is also the first chunk of
`model_uint8.onnx`. No chunk is shared with the `qwen2.5-0.5b-instruct`
artifact (computed from the two manifests' `chunks` tables).

## Upload

Not run: no `R2_*` credentials were present. `pack-model --upload` and
`upload` both exit before touching the store with
`R2 upload requested but these environment variables are unset: …`.
The skip-if-exists behaviour (second run uploads only `latest.json`) and the
`Cache-Control` / `Content-Type` headers are covered by
`ingest/tests/test_r2.py` against a fake S3 client.
