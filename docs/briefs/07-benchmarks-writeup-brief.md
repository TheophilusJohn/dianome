# Phase 7 brief — benchmarks on a rented GPU, summarisation demo, write-up

You are working in the `dianome` repo. Read the notes for every prior phase (`docs/phase-*.md`, `docs/spikes/`) and the briefs for 4 and 5b. Phase 6 (control plane) is deferred; the write-up says so.

Three parts, in order. Part A runs on a rented NVIDIA L4 (RunPod) that Theo starts; you prepare everything so the GPU day is a script, not a session. Part B is the vertical demo. Part C is the write-up. Every number in B and C comes from a results file produced in A or earlier phases.

## Part A — GPU day

### A1. Preparation (before the pod exists)

- `scripts/pod/setup.sh`: on a fresh RunPod PyTorch image, clone the repo at a given tag, install `server/` and `ingest/` into one venv, install `cloudflared`, download the models from Hugging Face into the pod's cache, verify `torch.cuda.is_available()`. Idempotent. Reads `HF_TOKEN`, `R2_*`, `SPLIT_SIGNING_KEY`, `SPLIT_TOKEN`, `CLOUDFLARED_TOKEN` from a `.env.pod` file Theo copies over (never committed; give a `.env.pod.example`).
- `scripts/pod/run-all.sh`: the whole GPU day in order, each step writing to `server/bench/results/l4/` and `server/probes/results/l4/`, resumable (skips steps whose result file exists):
  1. `ingest` on the pod: Qwen2.5-3B-Instruct in q8 and q4 (plus fp16 for the server's own use only if the server can't load from the HF cache — it can; so no fp16 upload), Qwen2.5-7B-Instruct in q8 and q4. `pack-model` then `upload` to R2 from the pod (fast network; Theo's uplink is not). Record bytes and chunk counts. Expect the bucket to pass the R2 free tier; the notes record the size.
  2. `dianome-server bench` for 0.5B, 3B, 7B on `cuda`, fp16 weights from the HF cache, the same N grid as Phase 4 scaled to each L (0.5B: 24 layers; 3B: 36; 7B: 28 — use N ∈ {0, L/6, L/3, L/2, 2L/3, 5L/6, L}, integer-rounded), 128 new tokens, 3 runs, median. `bench/rates.json` must have the L4 rate filled (Theo does this; the script refuses to run with a null rate).
  3. Privacy probes on 3B: `collect` + `linear-probe` with a 200k-token training set and the same held-out set, all boundaries. 7B: the same if wall time permits within the budget below; otherwise record "not run". Inversion decoders: not on the L4 (their result on 0.5B was within a few points of the linear probe; state that as the reason).
  4. Start `dianome-server serve` for 7B (fp16 on the L4) behind `cloudflared` as `gpu.dianome.dev`, and keep it up for the remote split measurements in A2.
  Budget: the script prints elapsed GPU time after each step; target under 4 hours total.
- Worker: `SPLIT_SERVERS` var, a JSON map of model id → `{ ws, plan }`, replacing the single `SPLIT_WS_URL`/`SPLIT_PLAN_URL`; `/v1/split/session` and `/v1/split/plan` take `?model=`. Tests updated. Not deployed by you.

### A2. Remote split measurements (Theo at the laptop, pod up)

`apps/split-demo` gains a model selector (0.5B, 3B, 7B) and reads the server map from the Worker. Add `scripts/measure-remote.mjs` that drives the deployed split demo in Chrome via Playwright for each model at the planner's N under `prefer: "cost"` and at N = 0, 3 runs each, and writes `packages/sdk/results/remote-l4.json`: per run the mode, N, tok/s, per-step breakdown medians, rtt, server busy, load bytes/time, cache mode. This is the first split measurement with client and server on different machines; the notes say so and retire the shared-GPU confound from Phase 5b.

### A3. Notes

`docs/phase-7-notes.md`, generated from the result files: ingest sizes; the cost table per model with `cost_per_1M_tokens(N)` at the stated L4 rate and `cost(N*)/cost(0)`; the privacy band for 3B (and 7B if run) beside the 0.5B one; the remote split table; GPU minutes used and the dollar cost of the day from RunPod's billing (Theo pastes it).

## Part B — Summarisation demo and dianome.dev

`apps/site/`: one Vite + TypeScript site deployed as the `dianome.dev` Pages project, plain DOM, readable, no framework. Routes:

- `/` — one screen: what Dianome is in two sentences, the three live demos linked (cache demo A/B, split slider, summariser), the stats dashboard, npm, GitHub, write-up. No claims that aren't in the notes.
- `/summarize` — the vertical demo. Paste or drop a text document (up to ~6k tokens). The page runs `run()` with `prefer: "cost"` on the model selected (default 3B q4; 0.5B and 7B selectable), streams the summary, and shows a plain status line while it works: which mode was chosen and why (from the plan's reasons), what was cached, the split point if any, and at the end tok/s, server share and cost at the stated rate, and the privacy band value at that N with the one-sentence explanation. A "compare" button runs the same document in the other two modes when feasible and shows the three side by side. Prompt template: a fixed system prompt for summarisation; document as the user turn; `maxTokens` 256.
- `/stats` — move `apps/load-dashboard` here, add a sessions view from the schema-3 data.
- `/writeup` — renders `docs/writeup.md` (Part C) with its tables; each number carries a link to the notes file and heading it came from.

Playwright: `/summarize` end to end in Chrome against local servers with the 0.5B model (mode chosen, summary non-empty, telemetry sent). Deployed by Theo.

## Part C — Write-up (`docs/writeup.md`)

Structure, fixed:

1. **Summary** — five sentences: what was built, the two positive results (edge CDN with cross-site cache on Chrome; bit-exact split inference with a measured cost curve), the two negative results (cross-site cache is Chrome-only; hidden states are linearly invertible at every layer), and the honest product framing (hybrid inference behind one API; "raw text is not transmitted", not "the server cannot read it").
2. **System** — the architecture diagram, the layer-boundary definition, the cost formula and rate, the privacy band definition. Short.
3. **Model CDN** — chunking and dedup numbers; edge cache verification; SDK size; the browser support table with the Phase 0 correction told as it happened (pre-registered bar, spike result, production result, correction).
4. **Split inference** — bit-exact gates; the WGSL runtime's validation ladder and the two kernel-order findings (block-23 accumulation, matvec ordering); tok/s on the Mac.
5. **Cost** — the three cost curves on the L4 with the `lm_head` floor explained; `cost(N*)/cost(0)` per model; the remote split table with the network share of each token; the statement that per-token round trips bound latency and the win is cost.
6. **Privacy** — the band for 0.5B and 3B (7B if run), raw and normalised, the nearest-neighbour trap, the training-set-size effect, and the conclusion. Related work in one paragraph, cited: prior results on residual-stream token recoverability and on split/partitioned inference (find 3–5 real references; if a reference cannot be verified, leave it out).
7. **Hybrid inference** — the planner, its inputs, what it chose on each device tested, estimate-vs-measured, and the summariser demo as the worked example.
8. **What was not done** — Phase 6, Firefox WebGPU, inversion decoders on 3B/7B, whatever else, each with one line on why.
9. **Reproducing** — every table maps to a command and a results file; the pod script; the tags.

Rules: no number without a results file; no adjective without a number; the negative results get the same space as the positive ones. Keep it under 4,000 words excluding tables. A `scripts/writeup-check.mjs` that scans the write-up for numbers and verifies each has a citation link fails the build if any is missing.

## Definition of done

- Part A scripts run end to end on the pod (Theo executes; you make them resumable and dry-run-testable on the Mac with `--dry-run`), results files land, notes generated.
- Part B builds, Playwright passes locally, and the summariser works in all three modes against the deployed servers once Theo deploys.
- Part C passes `writeup-check`, renders at `/writeup`, and a `pandoc` recipe in `docs/` produces a PDF (Theo runs it).
- Nothing published or deployed by you.

Report after A1 (scripts ready, dry-run passing), after B, and after C.
