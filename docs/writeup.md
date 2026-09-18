# Dianome: a model CDN and split inference for the browser

Every number in this document links to the notes file and heading, or the results file, it was copied from. Tables
marked *pending* are filled from the Phase 7 GPU day; `scripts/writeup-check.mjs` allows them only while those results
files do not exist.

## 1. Summary

Dianome packs an open-weight language model once into content-addressed chunks, serves them from a CDN, loads them in
the browser progressively with every chunk verified, and runs the model on WebGPU, on a split server, or split between
the two at a chosen layer through one `run()` call that plans per device and network. Two results are positive: the
edge CDN with a cross-site cache works on Chrome, so a second site reuses a model the first one downloaded, and the
server half of split inference is bit-exact against the full model at every split point tested, and the browser runtime
reproduces the full model's greedy tokens at every split point in fp16 (one recorded near-tie in q4), with a measured
cost curve whose floor is the language-model head. Two results are negative: the cross-site cache is Chrome-only, because Firefox and Safari do
not unpartition the Cache API through Storage Access, and the hidden state at every layer boundary remains linearly
invertible to the input tokens, so a split server can read most of the prompt. The product framing that survives is
hybrid inference behind one API, chosen per device for cost and latency, where the honest claim is "raw text is not
transmitted", never "the server cannot read it". Phase 6, the control plane, is deferred, and the sections below say
what else was not done.

## 2. System

```
  ingest (Python)            edge (Cloudflare)                     browser (dianome SDK)              split server (Python)
  safetensors ──► fp16/q8/q4 ──► R2 bucket ──► cdn.dianome.dev ──► chunks → Cache API (per-site, ──► WebGPU runtime: blocks 0..N-1
  per layer group, 8 MB      manifests via     edge cache, immutable  cross-site on Chrome)          hidden state at boundary N ──► blocks N..L-1,
  chunks, two-level manifest api.dianome.dev                        planner: local / split / server       norm, lm_head, sampling ──► token
                             telemetry → Analytics Engine → /v1/stats                                (or N = 0: token ids; or local: nothing sent)
```

**Layer boundary.** "Split at N" means the client runs the embedding lookup and blocks 0 to N−1 and sends the output
of block N−1, the input to block N, as an fp16 matrix of one row per token. The server runs blocks N to L−1, the final
norm, the head and sampling. N = 0 sends token ids; N = L sends the output of the last block and the server runs only
norm, head and sampling; local mode runs everything in the browser and sends nothing
([server README](../server/README.md)).

**Cost.** The harness times only server execution, with a device sync before the clock stops, over prefill plus
128 greedy decode steps per prompt, three runs, median. `gpu_seconds_per_token(N)` is server busy seconds over tokens
generated and `cost_per_1M_tokens(N) = gpu_seconds_per_token(N) × 1e6 × rate / 3600` with the hourly rate read from
`bench/rates.json`, never hardcoded; the ratio `cost(N)/cost(0)` does not depend on the rate
([Phase 4 notes § Cost harness](../docs/phase-4-notes.md#cost-harness-this-mac-not-gpu-numbers)).

**Privacy band.** For each boundary i, the fraction of held-out input tokens a probe trained on activations at that
boundary recovers (top-1), on WikiText-103 test documents never seen in training; three probes are reported, nearest
embedding row, a linear probe and a small inversion decoder, and the linear probe is also normalised by the fraction of
held-out tokens whose type occurs in its training set ([Phase 4 notes § Privacy band](../docs/phase-4-notes.md#privacy-band)).

## 3. Model CDN

**Chunking and dedup.** Qwen2.5-0.5B-Instruct packs into 223 unique 8 MB chunks, 1,599.9 MiB in total: fp16 is 942.3 MiB
in 130 chunks, q8 472.1 MiB in 66, q4 308.9 MiB in 42, in 6.1 s of wall time from a warm HF cache, and a second run is
byte-identical ([Phase 1 notes § pack-model](../docs/phase-1-notes.md#pack-model-bytes-per-variant-chunks-per-variant-ingest-wall-time)).
Dedup across variants is exactly the shared q8 embedding table and the final norm: 238 listed chunks become 219 unique
and 130.1 MiB are saved ([Phase 1 notes § inspect](../docs/phase-1-notes.md#inspect-sizes-per-group-dedup-savings)).
Every fp16 entry is bit-equal to the source, 533 of 533, and every quantised entry is encoder/decoder-consistent, 340 of
340; the median relative RMS error per role is about 0.0096 for q8 and about 0.11 for q4
([Phase 1 notes § verify](../docs/phase-1-notes.md#verify---against-hf-q8-and-q4-error-per-role)).

**Edge cache.** On the deployed origins all 14 checks pass: a chunk answers HEAD with `accept-ranges: bytes`, an immutable
cache-control and `timing-allow-origin: *`, a 1024-byte range request returns 206, the second full GET is
`cf-cache-status: HIT`, and `If-None-Match` returns 304 ([Phase 2 notes § check-edge](../docs/phase-2-notes.md#scriptscheck-edgesh)).
The first real browser load fetched the q4 variant, 323,893,760 bytes in 42 chunks, in 15,629 ms at 20.7 MB/s, and its
telemetry row landed with a 202 ([Phase 2 notes § First browser load](../docs/phase-2-notes.md#first-browser-load)).
Extensionless chunk paths were not cached until a cache rule was added, and the first empty stats result was cached for
its full TTL; both are recorded as deploy issues ([Phase 2 notes § Issues](../docs/phase-2-notes.md#issues-found-in-deploy)).

**SDK.** The main entry is 44,295 bytes, 14,094 gzipped, against a 25 KB budget; the adapters for Transformers.js and
WebLLM are 4,432 and 4,608 gzipped ([Phase 5b notes § Package sizes](../docs/phase-5b-notes.md#package-sizes)).
On the local demo a q4 load completes in 0.93 s from the network and 0.27 s from the per-site cache with 0 chunk
requests, verify time 156 ms for 42 chunks ([Phase 3 notes § Demo site](../docs/phase-3-notes.md#demo-site-q4-load-through-the-sdk-local-servers)).
SHA-256 over an 8 MiB chunk runs at 2.3 to 2.6 GB/s in Chromium, so verification is not the bottleneck
([Phase 3 notes § verify throughput](../docs/phase-3-notes.md#sha-256-verify-throughput-and-postmessage-transfer-cost)).

**Browser support, as it happened.** The Phase 0 spike pre-registered the bar before measuring: Green meant a cache hit
on a second site after a one-time opt-in plus one click on at least two of three browsers, Yellow a hit only with a
prompt on every site, Red no hit anywhere ([Spike 00 § Decision](../docs/spikes/00-storage-partitioning.md#decision)).
The spike read Green: Chrome hit through the `{all: true}` storage-access handle in 21 ms, and Firefox appeared to hit
through the plain grant ([Spike 00 § Cross-site cache](../docs/spikes/00-storage-partitioning.md#cross-site-cache-load-on-a-first-then-run-on-b)).
On the production origins Firefox's grant resolved, `hasStorageAccess()` was true, and the frame saw 0 keys and no
marker: the Phase 0 Firefox hit had been a partitioned copy the harness could not tell apart from a shared one
([Spike 00 § Correction](../docs/spikes/00-storage-partitioning.md#correction-2026-09-17)). The decision was corrected
to Yellow, Chrome only, and the SDK detects the capability rather than the browser name.

| browser | per-site cache | cross-site cache | production evidence |
|---|---|---|---|
| Chrome | yes | yes, via the `{all: true}` handle after a one-time visit | site A network 22.91 s, site B cross-site-cache 0.81 s, hop median 21.7 ms ([Phase 3 notes § manual checklist](../docs/phase-3-notes.md#cross-site-cache-manual-checklist-deployed-demo-sites)) |
| Firefox | yes | no | grant resolves, 0 keys, marker not visible, reported `unsupported` ([Phase 3 notes § manual checklist](../docs/phase-3-notes.md#cross-site-cache-manual-checklist-deployed-demo-sites)) |
| Safari | yes | no | grant covers cookies only; per-site cache on the second load ([Spike 00 § Cross-site cache](../docs/spikes/00-storage-partitioning.md#cross-site-cache-load-on-a-first-then-run-on-b)) |

Per-site caching works on every engine: second loads made 0 chunk requests on Chromium, Firefox and WebKit, and a
corrupted chunk was refetched exactly once ([Phase 3 notes § Per-site cache](../docs/phase-3-notes.md#per-site-cache-first-vs-second-load-per-browser)).

## 4. Split inference

**Bit-exact gates.** The server's explicit block loop reproduces the stock forward with `max_abs_diff` 0.0 at
N ∈ {0, 1, 8, 16, 23, 24} and identical argmax, on MPS and on CPU; perturbing the boundary tensor by 0.01 moves the
diff to about 0.43, so the comparison is live ([Phase 4 notes § Correctness gate](../docs/phase-4-notes.md#correctness-gate-max_abs_diff-per-n)).
A real WebSocket client at N = 8 produced the same 16 greedy tokens as the full model
([Phase 4 notes § Definition-of-done client run](../docs/phase-4-notes.md#definition-of-done-client-run-real-websocket-n--8)).

**The WGSL runtime's validation ladder.** Gate 1: the tokenizer encodes and decodes 200 of 200 strings identically to the
HF tokenizer ([Phase 5a notes § Gate 1](../docs/phase-5a-notes.md#gate-1--tokenizer)). Gate 2: every kernel is within a
max relative error of 1e-3 of a CPU reference on random inputs
([Phase 5a notes § Gate 2](../docs/phase-5a-notes.md#gate-2--unit-level-gpu-vs-cpu-reference-random-inputs-fp16-emulation-off-unless-noted)).
Gate 3: block 0 stage by stage against PyTorch dumps, max abs 9.77e-4 at the block output
([Phase 5a notes § Gate 3](../docs/phase-5a-notes.md#gate-3--block-0-stage-by-stage-32-token-prompt-fixture-rope-tables-fp16-emulation-on)).
Gate 4: all 24 blocks in prefill, relative RMS at most 2.52e-3
([Phase 5a notes § Gate 4](../docs/phase-5a-notes.md#gate-4--all-24-blocks-prefill)). Gate 5: decode with the KV cache
reproduces prefill bit for bit ([Phase 5a notes § Gate 5a](../docs/phase-5a-notes.md#5a-prefill-31-tokens-decode-the-32nd)).
Gate 6: over the real WebSocket at N ∈ {1, 8, 16, 24}, all 16 greedy tokens equal the full model's, with fp16 emulation
on and off ([Phase 5a notes § Gate 6](../docs/phase-5a-notes.md#gate-6--end-to-end-with-dianome-server-fp16-n--1-8-16-24)).
Gate 7: the q8 and q4 paths match PyTorch running the dequantised weights, with one recorded tie at q4, N = 24, step 7,
where the reference's top-2 logit gap was 0.0078125 and the runtime chose the runner-up
([Phase 5a notes § Gate 7](../docs/phase-5a-notes.md#gate-7--q8-and-q4)). Gates 8 and 9 add the head: the runtime's
final norm plus head has relative RMS 1.847e-3 against the fixture logits with no argmax mismatch, and local mode
produces the reference tokens for all three variants
([Phase 5b notes § lm_head gate](../docs/phase-5b-notes.md#lm_head-gate-gate-8-every-row-of-the-runtimes-final-norm--lm_head-vs-logitsnpy),
[§ Local-mode gate](../docs/phase-5b-notes.md#local-mode-gate-gate-9-16-greedy-tokens-lm_head--sampler-in-the-runtime-no-server)).

**Two kernel-order findings.** First, block 23 exceeds the brief's 5e-2 max-abs bar with fp16 emulation on, at 9.38e-2.
Run alone on the previous block's fixture output every block is within one fp16 ulp of its largest intermediate, and
block 23's own error is 3.13e-2, one ulp of the −38.6 its down projection produces before the residual add cancels
it to 2.97; the excess is accumulation of one-ulp rounding flips through 24 layers, not a kernel, and the gate was
relaxed to 1e-1 with that evidence ([Phase 5a notes § Why block 23](../docs/phase-5a-notes.md#why-block-23-exceeds-5e-2-max-abs-with-emulation-on-isolation-evidence)).
Second, the 16-lane decode matvec sums in a different order from the tiled prefill kernel, and under emulation those
flips grow to 3.13e-2 by block 23, while a sequential kernel in the tiled order gives 0; the sequential kernel is the
default when emulation is on ([Phase 5a notes § Gate 5a](../docs/phase-5a-notes.md#5a-prefill-31-tokens-decode-the-32nd)).

**Throughput on the Mac.** In Chrome with the q4 variant and all 24 blocks in the browser, prefill runs at 694 tokens
per second and decode at 62.9 tokens per second with the sequential kernel or 129.9 with the lane kernel; fp16 decodes
at 56.5 ([Phase 5a notes § Bench](../docs/phase-5a-notes.md#bench-chrome-this-mac-prefill-and-decode-toks-vs-n)).
WebKit runs the same configuration at 52.6 tokens per second; Playwright's Firefox exposes WebGPU but returns no
adapter ([Phase 5a notes § Firefox and Safari](../docs/phase-5a-notes.md#firefox-and-safari-observed-not-fixed)).

## 5. Cost

**The shape, measured on the Mac.** With the 0.5B model, 8 prompts and 128 decode steps each, server busy time falls
from 19.892 ms per token at N = 0 to 4.461 ms at N = 24, a ratio of 0.224; the curve is roughly linear in the number
of server blocks ([Phase 4 notes § Cost harness](../docs/phase-4-notes.md#cost-harness-this-mac-not-gpu-numbers)).

| N | ms/token (server busy) | cost(N)/cost(0) |
|---|---|---|
| 0 | 19.892 | 1.000 |
| 4 | 17.150 | 0.862 |
| 8 | 14.816 | 0.745 |
| 12 | 11.551 | 0.581 |
| 16 | 9.437 | 0.474 |
| 20 | 7.108 | 0.357 |
| 24 | 4.461 | 0.224 |

Source: [Phase 4 notes § Cost harness](../docs/phase-4-notes.md#cost-harness-this-mac-not-gpu-numbers). These are Mac
numbers with no hourly rate; the shape is what carries over.

**The head floor.** At N = L the server still runs the final norm, the head over the 151,936-entry vocabulary and
sampling, which is the 0.224 the curve cannot get under on the Mac; the planner carries that floor as
`serverCostFloor` and every split's server share is `floor + (1 − floor) × (L − N) / L`
([Phase 5b notes § Planner inputs](../docs/phase-5b-notes.md#planner-inputs-on-this-mac-chrome-run-measured-them-q4-maxctx-1024)).

**The three cost curves on the L4.** *Pending the GPU day.* Filled from the cuda results of `dianome-server bench` for
0.5B, 3B and 7B with the stated L4 rate, and the notes generated from them.

<!-- pending: server/bench/results/l4/cuda-qwen2.5-0.5b-instruct-*.json -->
<!-- pending: server/bench/results/l4/cuda-qwen2.5-3b-instruct-*.json -->
<!-- pending: server/bench/results/l4/cuda-qwen2.5-7b-instruct-*.json -->

| model | L | rate ($/h) | cost per 1M tokens at N = 0 | N* under prefer: cost | cost(N*)/cost(0) |
|---|---|---|---|---|---|
| *pending* | | | | | |

**Remote split.** *Pending A2.* The first measurement with client and server on different machines, which retires the
shared-GPU confound of the Phase 5b tunnel run, is written by `scripts/measure-remote.mjs` to
`packages/sdk/results/remote-l4.json`: per model, the planner's N and N = 0, three runs each, with the per-step
breakdown and the network share of each token.

<!-- pending: packages/sdk/results/remote-l4.json -->

| model | mode / N | tok/s | client | export | network | server busy | RTT | network share |
|---|---|---|---|---|---|---|---|---|
| *pending* | | | | | | | | |

**What bounds what.** Every generated token in split or server mode is one round trip, so latency is bounded by the
network, not by the arithmetic: on the tunnelled Phase 5b run the network took 58.1 ms of a 96.8 ms decode step at an
RTT of 57.6 ms, and on the loopback runs the same step took about 20 ms at every N
([Phase 5b notes § Deployed, via tunnel](../docs/phase-5b-notes.md#deployed-via-tunnel),
[§ Estimated vs measured](../docs/phase-5b-notes.md#estimated-vs-measured-mstoken-q4-n--0-4-8-12-16-20-24-local)).
The win of a split is cost, the server's share of the work, not speed.

## 6. Privacy

**The band for 0.5B.** With a 40k-token training set the linear probe recovers 0.8162 of held-out tokens at boundary 0
and 0.5744 at boundary 24, the inversion decoder 0.8188 and 0.5607; the ceiling of those probes is 0.8202, the fraction
of held-out tokens whose type occurs in the training set, so normalised they read 0.995 and 0.700
([Phase 4 notes § Privacy band](../docs/phase-4-notes.md#privacy-band), [§ Findings](../docs/phase-4-notes.md#findings)).

| boundary | linear top-1 (40k) | inversion top-1 (40k) | linear top-1 (500k) | normalised (500k) |
|---|---|---|---|---|
| 0 | 0.8162 | 0.8188 | 0.9727 | 0.9999 |
| 4 | 0.7673 | 0.7433 | 0.9443 | 0.9708 |
| 8 | 0.7054 | 0.7075 | 0.8990 | 0.9242 |
| 12 | 0.6616 | 0.6958 | 0.8706 | 0.8950 |
| 16 | 0.6672 | 0.6911 | 0.8754 | 0.8999 |
| 20 | 0.6673 | 0.6797 | 0.8687 | 0.8930 |
| 24 | 0.5744 | 0.5607 | 0.7881 | 0.8102 |

Sources: [Phase 4 notes § Privacy band](../docs/phase-4-notes.md#privacy-band) and
[§ Linear probe, 500k-token training set](../docs/phase-4-notes.md#linear-probe-500k-token-training-set).

**The nearest-neighbour trap.** Matching the hidden state to the nearest embedding row recovers every token at boundary
0 and 0.0045 at boundary 1, then stays under 0.0015 for the rest of the network, while the trained probes barely move;
a privacy claim built on nearest-neighbour would say the prompt is hidden after one block, and it is not
([Phase 4 notes § Privacy band](../docs/phase-4-notes.md#privacy-band)).

**Training-set size.** Raising the training set from 40k to 500k tokens raises coverage from 0.8202 to 0.9728 and the
normalised top-1 at boundary 12 from 0.807 to 0.895 and at boundary 24 from 0.700 to 0.810; more data makes the
adversary stronger, not weaker, so the band is a lower bound on what a server could recover
([Phase 4 notes § Findings](../docs/phase-4-notes.md#findings)).

**3B and 7B.** *Pending the GPU day.* The same linear probe with a 200k-token training set and the same held-out set,
for 3B, and for 7B if the budget rule admitted it; inversion decoders are not run there because on 0.5B they were
within a few points of the linear probe.

<!-- pending: server/probes/results/l4/linear-qwen2.5-3b-instruct.json -->

| model | boundary 0 | L/2 | L | normalised at L |
|---|---|---|---|---|
| *pending* | | | | |

**Conclusion.** A server that receives the hidden state at any boundary can recover most of the input tokens with a
linear map trained on public text; the deepest boundary of the 0.5B model still yields 0.81 of them normalised
([Phase 4 notes § Findings](../docs/phase-4-notes.md#findings)). Split inference is therefore a cost and latency
mechanism, and the only privacy statement the product makes is that raw text is not transmitted.

**Related work.** That the residual stream of a transformer is readable with a linear map is the premise of the logit
lens ([nostalgebraist, 2020](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens)) and its
trained successor, the tuned lens ([Belrose et al., 2023, arXiv:2303.08112](https://arxiv.org/abs/2303.08112)); this
write-up reads the input rather than the prediction from it. Morris et al. show that sentence embeddings can be inverted
back to text almost verbatim ([2023, arXiv:2310.06816](https://arxiv.org/abs/2310.06816)). Pasquini et al. demonstrate
inference attacks that recover client data in split learning, where a server sees exactly the kind of intermediate
activation a split point sends ([2021, arXiv:2012.02670](https://arxiv.org/abs/2012.02670)). Petals runs large models
split across volunteer machines and reports the same round-trip-bound latency structure that section 5 measures
([Borzunov et al., 2023, arXiv:2209.01188](https://arxiv.org/abs/2209.01188)).

## 7. Hybrid inference

**The planner** is a pure function from measured inputs to a mode, an N, an estimate and the reasons. Its inputs on the
Mac in Chrome: WebGPU with a 4096 MiB maximum buffer, a GPU budget of 1536 MiB derived from a 3072 MiB storage quota,
a microbench of 0.712 ms per block at one token and 1.800 at 32, head plus readback 2.50 ms, loopback bandwidth
141 MB/s and RTT 0.20 ms, and the server's own microbench of 0.654 ms per block
([Phase 5b notes § Planner inputs](../docs/phase-5b-notes.md#planner-inputs-on-this-mac-chrome-run-measured-them-q4-maxctx-1024)).
Each candidate N from 0 to L, plus local, gets a feasibility verdict, an estimated ms per token, a server share, a cost
at the stated rate when there is one, and the privacy band at that boundary.

**What it chose on each device.** With prefer: cost, Chrome chose local at N = 24 with an estimate of 18.4 ms per
token, WebKit chose local at 21.3 ms, and Firefox, whose WebGPU returns no adapter, chose the server at N = 0 with
19.1 ms ([Phase 5b notes § other browsers](../docs/phase-5b-notes.md#planner-inputs-on-other-browsers-scriptssplit-browsersmjs-prefer-cost-q4-fresh-microbench)).
On the same inputs prefer: latency picks the server at 19.3 ms over local at 19.9 ms, because on this machine the
runtime and the MPS server are a near tie ([Phase 5b notes § Planner choices](../docs/phase-5b-notes.md#planner-choices-per-policy-same-inputs)).

**Estimate versus measured.** Over N ∈ {0, 4, 8, 12, 16, 20, 24} and local on loopback, measured decode steps are
within 0.92 to 1.03 of the estimate; the prefill estimate is low by 10 to 12 ms at every N
([Phase 5b notes § Estimated vs measured](../docs/phase-5b-notes.md#estimated-vs-measured-mstoken-q4-n--0-4-8-12-16-20-24-local)).
Through the tunnel the ratio was 1.27 for decode and 658.5 ms measured against 101 ms estimated for prefill, with
client and server sharing one GPU; that confound is why section 5 waits for the remote table
([Phase 5b notes § Deployed, via tunnel](../docs/phase-5b-notes.md#deployed-via-tunnel)).

**The summariser as the worked example.** The `/summarize` page sends a fixed system prompt and the document as the
user turn with 256 new tokens under prefer: cost. In the Chrome end-to-end run on the Mac with the 0.5B model, a
522-token document was planned as local at N = 24 because the policy prefers cost and local is feasible, produced
161 tokens at 43.6 tokens per second, and the session report reached the Worker with a 202
([summarize-e2e.json](../apps/site/results/summarize-e2e.json)). The compare button then ran the same document as
server at N = 0, 47.3 tokens per second with a server share of 100 %, and as a split at N = 24, 42.5 tokens per second
with a share of 22 % and a band value of 78.8 % at that boundary, beside the local run at 47.5
([summarize-e2e.json](../apps/site/results/summarize-e2e.json)). The three speeds are within a few tokens per second
of each other; only the server share and what left the device differ, which is the point of the page.

## 8. What was not done

- **Phase 6, the control plane** (per-developer API keys, quotas, billing): deferred; the session endpoint is gated by
  origin and a shared signing key only ([Worker split endpoints](../packages/worker/src/split.ts)).
- **Firefox WebGPU**: Playwright's Firefox 155 exposes `navigator.gpu` but returns no adapter, so the runtime was not
  validated there and the planner routes Firefox to the server
  ([Phase 5a notes § firefox 155.0](../docs/phase-5a-notes.md#firefox-1550)).
- **Inversion decoders on 3B and 7B**: not run on the L4; on 0.5B they were within a few points of the linear probe at
  every boundary, and the L4 budget went to the linear probe with a larger training set
  ([Phase 4 notes § Privacy band](../docs/phase-4-notes.md#privacy-band)).
- **The L4 tables** in sections 5 and 6 and the remote split table: pending the GPU day, marked above.
- **Chrome with third-party cookies blocked** on the production sites: the manual checklist row is empty; the spike
  measured it in Incognito only ([Phase 3 notes § manual checklist](../docs/phase-3-notes.md#cross-site-cache-manual-checklist-deployed-demo-sites)).
- **Safari cross-site cache**: the grant covers cookies only and the SDK does not request it there
  ([Spike 00 § SDK caveats](../docs/spikes/00-storage-partitioning.md#sdk-caveats-carried-forward)).
- **A published L4 rate**: `bench/rates.json` has no hourly rate yet, so every cost in this document is a share, not a
  dollar figure ([Phase 4 notes § Cost harness](../docs/phase-4-notes.md#cost-harness-this-mac-not-gpu-numbers)).

## 9. Reproducing

Tags: `v0.1.0` is the SDK and cross-site cache as deployed after the Phase 0 correction, `v0.2.0` is the first deployed
split run ([README](../README.md)). Every table above maps to a command and a results file:

| table or figure | command | results file |
|---|---|---|
| chunking, dedup, verify (section 3) | `dianome-ingest pack-model`, `inspect`, `verify --against-hf` | [Phase 1 notes](../docs/phase-1-notes.md) |
| edge checks, first load (section 3) | `scripts/check-edge.sh`, `apps/load-test` | [Phase 2 notes](../docs/phase-2-notes.md) |
| per-site and cross-site cache (section 3) | `pnpm --filter dianome test:e2e`; manual checklist on the demo sites | [Phase 3 notes](../docs/phase-3-notes.md), [Spike 00](../docs/spikes/00-storage-partitioning.md) |
| correctness gate, cost curve, band (sections 4 to 6) | `dianome-server bench`, `probes`, `linear-probe` | [gate-max-abs-diff.json](../server/bench/results/gate-max-abs-diff.json), [mps cost json](../server/bench/results/mps-qwen2.5-0.5b-instruct-2026-09-17.json), [band.json](../server/probes/results/band.json), [linear-500k.json](../server/probes/results/linear-500k.json) |
| runtime gates and bench (section 4) | `npx playwright test gates.spec.ts`, `bench.spec.ts`, `scripts/bench-browsers.mjs` in `packages/runtime` | [gates.json](../packages/runtime/results/gates.json), [bench.json](../packages/runtime/results/bench.json), [browsers.json](../packages/runtime/results/browsers.json) |
| planner, estimate vs measured, tunnel run (section 7) | `npx playwright test -c playwright.split.config.ts`, `scripts/split-browsers.mjs` in `packages/sdk` | [split-e2e.json](../packages/sdk/results/split-e2e.json), [split-measure.json](../packages/sdk/results/split-measure.json), [split-browsers.json](../packages/sdk/results/split-browsers.json), [split-deployed.json](../packages/sdk/results/split-deployed.json) |
| summariser run (section 7) | `npx playwright test` in `apps/site` | [summarize-e2e.json](../apps/site/results/summarize-e2e.json) |
| L4 cost curves, 3B/7B band (sections 5, 6) | `scripts/pod/setup.sh`, then `scripts/pod/run-all.sh` on the pod | `server/bench/results/l4/`, `server/probes/results/l4/` (pending) |
| remote split (section 5) | `scripts/measure-remote.mjs` against the deployed demo with the pod up | `packages/sdk/results/remote-l4.json` (pending) |

The pod scripts are resumable and dry-run testable ([scripts/pod/README.md](../scripts/pod/README.md)); the notes each
phase's tables were pasted from are regenerated by the scripts named at the top of each notes file. This document is
checked by `scripts/writeup-check.mjs`: every number outside a code span must sit in a paragraph, list item or table
row that links to a repo file, with any heading anchor resolving, or to an external reference; the site build fails
otherwise. A PDF is produced by `docs/writeup-pdf.sh` with pandoc.
