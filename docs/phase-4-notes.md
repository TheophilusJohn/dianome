# Phase 4 notes — server half, cost baseline, privacy probes

Every number below is pasted from the output of the command shown above it or
copied from the result file it names, run on 2026-09-17 on this MacBook
(mps (Apple M4), 16 GB). Nothing here is typed by hand.

- Repo commit at the time of the runs: `b0ccf4c73d48fc3b929e59f71269bf05f06eef59` (the Phase 4 code was uncommitted on top of it).
- Model: `qwen2.5-0.5b-instruct` = `Qwen/Qwen2.5-0.5B-Instruct` @ `7ae557604adf67be50417f59c2c2f167def9a775`, fp16, L = 24, d_model = 896.
- torch 2.14.0, transformers 4.57.1 (pinned), numpy 2.5.3, Python 3.12 venv at `server/.venv`.

## Tests

```
$ cd server && .venv/bin/pytest -q
43 passed in 8.16s
```

## Correctness gate: max_abs_diff per N

`tests/test_correctness.py` writes the measured max_abs_diff of the split-path
logits against `model.forward` (32-token prompt) to
`server/bench/results/gate-max-abs-diff.json`; the gate is `< 5e-2` and identical argmax.

| N | max_abs_diff (mps) |
| --- | --- |
| 0 | 0.0 |
| 1 | 0.0 |
| 8 | 0.0 |
| 16 | 0.0 |
| 23 | 0.0 |
| 24 | 0.0 |

The diff is exactly zero because the block loop feeds the same fp16 tensors
through the same SDPA kernels as the stock forward; perturbing the boundary
tensor by 0.01 gives a diff of about 0.43, so the comparison is real. The same
six values were 0.0 with `DIANOME_DEVICE=cpu`.

## Definition-of-done client run (real WebSocket, N = 8)

```
$ cd server && SPLIT_TOKEN=local-dev-token .venv/bin/dianome-server serve &   # localhost only
$ .venv/bin/python -m tests.ws_client --token local-dev-token --N 8 --steps 16 --prompt "<the 32-token fixture prompt from tests/conftest.py>"
[752, 83951, 1526, 279, 57267, 6176, 13638, 13, 576, 33581, 374, 264, 6233, 1992, 11, 448]
' meanders through the lush green forest. The valley is a beautiful place, with'
full model greedy: [752, 83951, 1526, 279, 57267, 6176, 13638, 13, 576, 33581, 374, 264, 6233, 1992, 11, 448]
```

The `full model greedy` line is `LoadedModel.full_greedy` on the same prompt, printed
by a one-off snippet after the client run; `tests/test_server.py::test_ws_client_n8_matches_full_greedy`
asserts the same equality. `GET /plan` on that server, with the probes running on
the same GPU at the time (hence the busy fraction):

```
$ curl -H "Authorization: Bearer $SPLIT_TOKEN" http://127.0.0.1:8765/plan
{"model":"qwen2.5-0.5b-instruct","L":24,"d_model":896,"active_sessions":0,"busy_fraction_60s":0.3217314701837798}
```

## Cost harness (this Mac; not GPU numbers)

```
$ server/.venv/bin/dianome-server bench
```

Device: **mps (Apple M4)**. Batch size 1, 8 prompts (1121 prompt tokens total), prefill + 128 greedy decode steps each, 3 runs, median. hourly_rate = None (mps (Apple M4) [NOT A GPU RATE; rates.json gpu='FILL ME (e.g. NVIDIA A100 80GB)']).

| N | prompt_tokens | gen_tokens | busy_seconds | gpu_seconds_per_token | ms/token | prefill_s | decode ms/token | cost_per_1M_tokens | cost(N)/cost(0) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 1121 | 1032 | 20.529 | 0.019892 | 19.892 | 0.719 | 19.346 | n/a (no rate) | 1.000 |
| 4 | 1121 | 1032 | 17.699 | 0.017150 | 17.150 | 0.675 | 16.533 | n/a (no rate) | 0.862 |
| 8 | 1121 | 1032 | 15.290 | 0.014816 | 14.816 | 0.577 | 14.213 | n/a (no rate) | 0.745 |
| 12 | 1121 | 1032 | 11.920 | 0.011551 | 11.551 | 0.436 | 11.250 | n/a (no rate) | 0.581 |
| 16 | 1121 | 1032 | 9.739 | 0.009437 | 9.437 | 0.459 | 9.063 | n/a (no rate) | 0.474 |
| 20 | 1121 | 1032 | 7.336 | 0.007108 | 7.108 | 0.520 | 6.570 | n/a (no rate) | 0.357 |
| 24 | 1121 | 1032 | 4.604 | 0.004461 | 4.461 | 0.338 | 4.148 | n/a (no rate) | 0.224 |

CSV: `server/bench/results/mps-qwen2.5-0.5b-instruct-2026-09-17.csv`.
`bench/rates.json` has no `usd_per_hour` yet (Theo fills the real GPU rate), so
`cost_per_1M_tokens` is empty; the `cost(N)/cost(0)` ratio is rate-independent.
Phase 7 re-runs this on a rented GPU.

## Privacy band

```
$ server/.venv/bin/dianome-server probes --notes docs/phase-4-notes.md
```

<!-- probes:start -->
Dataset: `Salesforce/wikitext` config `wikitext-103-raw-v1` split `test` (CC BY-SA 3.0), 51 documents, 50000 tokens, windows of 1024 tokens. Split by document (seed 0): 41 train documents (39760 tokens), 10 held-out documents (10240 tokens); every number below is on the held-out documents. Boundary i = the input to block i (i = 24: the input to the final norm). `nn` = nearest embedding row by cosine similarity, no training (L2 variant in band.json). `train_seconds` = linear + inversion training time on mps (Apple M4). Total probe wall time 8065 s.

| boundary | nn_top1 | linear_top1 | linear_top5 | inversion_top1 | inversion_top5 | train_seconds |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 1.0000 | 0.8162 | 0.8202 | 0.8188 | 0.8202 | 302.4 |
| 1 | 0.0045 | 0.8133 | 0.8188 | 0.8002 | 0.8144 | 332.0 |
| 2 | 0.0007 | 0.8014 | 0.8153 | 0.7813 | 0.8110 | 332.5 |
| 3 | 0.0003 | 0.7835 | 0.8069 | 0.7604 | 0.7971 | 323.4 |
| 4 | 0.0002 | 0.7673 | 0.8000 | 0.7433 | 0.7882 | 325.4 |
| 5 | 0.0002 | 0.7604 | 0.7950 | 0.7394 | 0.7879 | 302.2 |
| 6 | 0.0003 | 0.7309 | 0.7832 | 0.7216 | 0.7738 | 301.4 |
| 7 | 0.0000 | 0.7215 | 0.7708 | 0.7158 | 0.7633 | 308.1 |
| 8 | 0.0001 | 0.7054 | 0.7562 | 0.7075 | 0.7595 | 315.2 |
| 9 | 0.0000 | 0.6891 | 0.7438 | 0.7036 | 0.7523 | 300.5 |
| 10 | 0.0001 | 0.6765 | 0.7362 | 0.6973 | 0.7502 | 310.4 |
| 11 | 0.0000 | 0.6673 | 0.7280 | 0.6983 | 0.7491 | 300.0 |
| 12 | 0.0001 | 0.6616 | 0.7234 | 0.6958 | 0.7472 | 302.4 |
| 13 | 0.0001 | 0.6566 | 0.7203 | 0.6932 | 0.7453 | 301.2 |
| 14 | 0.0000 | 0.6576 | 0.7245 | 0.6963 | 0.7476 | 304.1 |
| 15 | 0.0000 | 0.6616 | 0.7288 | 0.6901 | 0.7464 | 301.3 |
| 16 | 0.0002 | 0.6672 | 0.7316 | 0.6911 | 0.7484 | 303.8 |
| 17 | 0.0001 | 0.6885 | 0.7493 | 0.6896 | 0.7517 | 306.8 |
| 18 | 0.0000 | 0.6819 | 0.7472 | 0.6882 | 0.7490 | 328.4 |
| 19 | 0.0001 | 0.6763 | 0.7447 | 0.6850 | 0.7471 | 322.5 |
| 20 | 0.0004 | 0.6673 | 0.7388 | 0.6797 | 0.7463 | 331.7 |
| 21 | 0.0005 | 0.6513 | 0.7331 | 0.6636 | 0.7443 | 322.1 |
| 22 | 0.0005 | 0.6262 | 0.7153 | 0.6432 | 0.7251 | 316.6 |
| 23 | 0.0008 | 0.6117 | 0.7070 | 0.6156 | 0.7107 | 312.1 |
| 24 | 0.0014 | 0.5744 | 0.6783 | 0.5607 | 0.6609 | 306.1 |
<!-- probes:end -->

Linear probe: AdamW lr 2e-3, batch 1024, up to 5 epochs, early stop (patience 1)
on held-out loss, inputs standardised per dimension. Inversion decoder: 2
layers, 4 heads, d = 512, 128-token windows, AdamW lr 1e-3, batch 8, up to 8
epochs, patience 2 (batch 32 put 2.5 GB of fp32 logits per step on a 16 GB
machine and paged; 8 is what fits).

Ceiling for the trained probes (`server/probes/results/coverage.json`, computed
from `tokens.npy`/`doc.npy`): of the 10240 held-out tokens,
8399 (0.8202) have a token type that occurs at all in the
6731 types of the training documents. A probe trained with cross-entropy on
~40k tokens does not predict types it never saw, so linear and inversion top-5
saturate at 0.8202 even at boundary 0, where nearest-neighbour is 1.0. The
band is therefore read relative to that ceiling; a larger corpus would raise it. Raw per-boundary metrics (top-5 for nn, val losses, epochs,
per-probe seconds) are in `server/probes/results/band.json`.

## Fixtures for Phase 5a

```
$ server/.venv/bin/dianome-server fixtures --out fixtures/qwen2.5-0.5b-instruct/   # run twice
5c283de74a3ad8b00653cab268ceae11ee10609b0179db1c076a0d6206c4c2fd  block_00.npy  [32, 896]
  1eefab8fef68581f30aa927cae6f1777aba3796a56e0963d8935e46c0f070798  block_01.npy  [32, 896]
  ef9f57f4c72fba7c0bb31e9a224a85ac0082800b4140395d8565fd63c3744ae4  block_02.npy  [32, 896]
  468bae2369d168ef79e10f5c464bfdb0c5e5526660cb67fa844eda7f04b6b7e8  block_03.npy  [32, 896]
  f1e6e1a1cc2e0d4924c2d92ab2151266e02f339df02d18e8caaad48458825c89  block_04.npy  [32, 896]
  5b80188443bb922cc8c1d68a898cac8d1ea5009cedf27df552267a97fc79017c  block_05.npy  [32, 896]
  54f1a061aaf0989df1e86db06c50e562fc128e8d8ce5d1be37adc92aaf54968b  block_06.npy  [32, 896]
  6ec9d639dd7f62eb3a2a3659813ff9bf75a6aa0def08a2c96f63100cebc7dcac  block_07.npy  [32, 896]
  d3d52e3a7474608a5a2ac9797a83c6e7c649502e00bf2ffd2322c5e96f7fa684  block_08.npy  [32, 896]
  ab24e2c3a786f4faa883a905f302c5f5232db380bc50adb37489fb1c21354aa4  block_09.npy  [32, 896]
  c49af3b894c2dc3370a35c3d826f19f04bce160747752c25941e07d2488c1682  block_10.npy  [32, 896]
  cbeece5fb92b551b65e5346fc7b673eaa7764c31ca52da51bbda57d56a8acd26  block_11.npy  [32, 896]
  dc19b2b7990b8d8c32b30509e361ccc132290d03834ef6187c0e1d4478ffc4fb  block_12.npy  [32, 896]
  e05d65c6185809e249c338097e9290d026282f8176c660b6d289d6a6b5355b2c  block_13.npy  [32, 896]
  f94b6d4e0ddc347eb643f943da0045f3e5573bdbb626a1a0be22330a06bb1ba9  block_14.npy  [32, 896]
  9b3ffd211d314190d737c97108d9282ec2711f6d174d2a083d599052a5aba9c7  block_15.npy  [32, 896]
  24fe4a411a7796e8b210a3a1c9474eabe66fad3920e62cfacd7391dfba5f099e  block_16.npy  [32, 896]
  361cb350f7d97af2be1f3d01d4339951a852ebe36cc3a3f2564179fbc1c453f6  block_17.npy  [32, 896]
  f6f14073317c3fb68aa94d01ff2a187c07e5adad9325d0c5b8dcb22710edaa3b  block_18.npy  [32, 896]
  d1f79878400336d273f1bccad2b9188fadf9c6ec0c377d708c1252f51edd1565  block_19.npy  [32, 896]
  98abbd93c9111a4d609a6443c06964db91429f6c28c4b1e384d8501ebdb71ab4  block_20.npy  [32, 896]
  c02f8fba09a7029ec5a6cdc5dd96e1ba42097f6be815d0c61ebccd518256dbe7  block_21.npy  [32, 896]
  8ffb34c45bba5c1b62b8e12111b37a605c8992f777ac470c2f2aa07c6351083f  block_22.npy  [32, 896]
  298657e97ff318382e18574ea9bcfde5785f59048b5d391ed7feae6c71560b0b  block_23.npy  [32, 896]
  060f6881bf96cd3594af891f7557d44024e1a92df9b0d3381f34979c5dd08fd2  embed.npy  [32, 896]
  8a43d493297def76449c83fb367f739627e827f9e1712b4d32249a7b3e874623  final_norm.npy  [32, 896]
  ec3abbb640d10809abd6d8626cf590835e8d539e63202fb19b2d3075108e7743  logits.npy  [32, 151936]
  ae3a53ac761c73b06751e2904e76541b8f52b7b2182452a3909fde7ef1ffeb82  prompt.json  
  96a9298e9a4f1e388c0e347be8582a1dcee377968903f9f1b60e2ce3215a4fb2  rope_cos.npy  [32, 64]
  746966853cab6baaa0ad8505a28e34804730ab7917999c75dace383eda550778  rope_sin.npy  [32, 64]
fixtures qwen2.5-0.5b-instruct -> fixtures/qwen2.5-0.5b-instruct/  T=32 L=24 d_model=896  logits max_abs_diff vs model.forward = 0.0
second run: 30 files, hashes identical to first run: True
```

`fixtures/qwen2.5-0.5b-instruct/fixtures.json` (committed; the `.npy` files are
gitignored) records shapes, dtypes and SHA-256 per file; logits max_abs_diff vs
`model.forward` = 0.0. Hashes:

| file | shape | sha256 |
| --- | --- | --- |
| `block_00.npy` | [32, 896] | `5c283de74a3ad8b00653cab268ceae11ee10609b0179db1c076a0d6206c4c2fd` |
| `block_01.npy` | [32, 896] | `1eefab8fef68581f30aa927cae6f1777aba3796a56e0963d8935e46c0f070798` |
| `block_02.npy` | [32, 896] | `ef9f57f4c72fba7c0bb31e9a224a85ac0082800b4140395d8565fd63c3744ae4` |
| `block_03.npy` | [32, 896] | `468bae2369d168ef79e10f5c464bfdb0c5e5526660cb67fa844eda7f04b6b7e8` |
| `block_04.npy` | [32, 896] | `f1e6e1a1cc2e0d4924c2d92ab2151266e02f339df02d18e8caaad48458825c89` |
| `block_05.npy` | [32, 896] | `5b80188443bb922cc8c1d68a898cac8d1ea5009cedf27df552267a97fc79017c` |
| `block_06.npy` | [32, 896] | `54f1a061aaf0989df1e86db06c50e562fc128e8d8ce5d1be37adc92aaf54968b` |
| `block_07.npy` | [32, 896] | `6ec9d639dd7f62eb3a2a3659813ff9bf75a6aa0def08a2c96f63100cebc7dcac` |
| `block_08.npy` | [32, 896] | `d3d52e3a7474608a5a2ac9797a83c6e7c649502e00bf2ffd2322c5e96f7fa684` |
| `block_09.npy` | [32, 896] | `ab24e2c3a786f4faa883a905f302c5f5232db380bc50adb37489fb1c21354aa4` |
| `block_10.npy` | [32, 896] | `c49af3b894c2dc3370a35c3d826f19f04bce160747752c25941e07d2488c1682` |
| `block_11.npy` | [32, 896] | `cbeece5fb92b551b65e5346fc7b673eaa7764c31ca52da51bbda57d56a8acd26` |
| `block_12.npy` | [32, 896] | `dc19b2b7990b8d8c32b30509e361ccc132290d03834ef6187c0e1d4478ffc4fb` |
| `block_13.npy` | [32, 896] | `e05d65c6185809e249c338097e9290d026282f8176c660b6d289d6a6b5355b2c` |
| `block_14.npy` | [32, 896] | `f94b6d4e0ddc347eb643f943da0045f3e5573bdbb626a1a0be22330a06bb1ba9` |
| `block_15.npy` | [32, 896] | `9b3ffd211d314190d737c97108d9282ec2711f6d174d2a083d599052a5aba9c7` |
| `block_16.npy` | [32, 896] | `24fe4a411a7796e8b210a3a1c9474eabe66fad3920e62cfacd7391dfba5f099e` |
| `block_17.npy` | [32, 896] | `361cb350f7d97af2be1f3d01d4339951a852ebe36cc3a3f2564179fbc1c453f6` |
| `block_18.npy` | [32, 896] | `f6f14073317c3fb68aa94d01ff2a187c07e5adad9325d0c5b8dcb22710edaa3b` |
| `block_19.npy` | [32, 896] | `d1f79878400336d273f1bccad2b9188fadf9c6ec0c377d708c1252f51edd1565` |
| `block_20.npy` | [32, 896] | `98abbd93c9111a4d609a6443c06964db91429f6c28c4b1e384d8501ebdb71ab4` |
| `block_21.npy` | [32, 896] | `c02f8fba09a7029ec5a6cdc5dd96e1ba42097f6be815d0c61ebccd518256dbe7` |
| `block_22.npy` | [32, 896] | `8ffb34c45bba5c1b62b8e12111b37a605c8992f777ac470c2f2aa07c6351083f` |
| `block_23.npy` | [32, 896] | `298657e97ff318382e18574ea9bcfde5785f59048b5d391ed7feae6c71560b0b` |
| `embed.npy` | [32, 896] | `060f6881bf96cd3594af891f7557d44024e1a92df9b0d3381f34979c5dd08fd2` |
| `final_norm.npy` | [32, 896] | `8a43d493297def76449c83fb367f739627e827f9e1712b4d32249a7b3e874623` |
| `logits.npy` | [32, 151936] | `ec3abbb640d10809abd6d8626cf590835e8d539e63202fb19b2d3075108e7743` |
| `prompt.json` |  | `ae3a53ac761c73b06751e2904e76541b8f52b7b2182452a3909fde7ef1ffeb82` |
| `rope_cos.npy` | [32, 64] | `96a9298e9a4f1e388c0e347be8582a1dcee377968903f9f1b60e2ce3215a4fb2` |
| `rope_sin.npy` | [32, 64] | `746966853cab6baaa0ad8505a28e34804730ab7917999c75dace383eda550778` |
