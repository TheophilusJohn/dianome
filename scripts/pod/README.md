# GPU day (Phase 7, Part A) — L4 on RunPod

Everything here is a script; the pod session is: copy two files, run two commands, wait, copy results back.

1. **Pod.** RunPod, NVIDIA L4 (24 GB), a PyTorch 2.x / CUDA 12 template with Python ≥ 3.11, a ≥ 120 GB volume mounted at
   `/workspace`, SSH on. Fill `server/bench/rates.json` with the L4 rate first (`bench --require-rate` refuses otherwise)
   and commit it; the `[vars] SPLIT_SERVERS` entry for 7B in `packages/worker/wrangler.toml` is already `gpu.dianome.dev`.
2. **Tunnel.** Zero Trust → Networks → Tunnels → create `gpu` with public hostname `gpu.dianome.dev` → `http://localhost:8765`.
   Put its token in `.env.pod` as `CLOUDFLARED_TOKEN`.
3. **Copy** `.env.pod` (from `.env.pod.example`; `SPLIT_SIGNING_KEY` = the Worker secret) and `scripts/pod/setup.sh` to the pod:
   `scp .env.pod scripts/pod/setup.sh pod:/workspace/`.
4. **Setup** (idempotent, ~10 min, mostly the 23 GB of model downloads): `cd /workspace && bash setup.sh --tag <phase-7 tag>`.
5. **Run** (resumable; re-run after any interruption): `cd /workspace/dianome && bash scripts/pod/run-all.sh`.
   Elapsed GPU time prints after every step; the 7B probes only run if the budget rule admits them.
6. **Results back:** `rsync -av pod:/workspace/dianome/server/bench/results/l4/ server/bench/results/l4/` and the same for
   `server/probes/results/l4/`, then commit. Paste RunPod's billing total for the day into the Phase 7 notes.
7. **A2** runs while `serve.sh` keeps a server up. The pod serves one model at a time and both 3B and 7B map to
   `gpu.dianome.dev` in the Worker: measure 7B first, then `bash scripts/pod/serve.sh stop && bash scripts/pod/serve.sh start --model qwen2.5-3b-instruct`
   and measure 3B (`scripts/measure-remote.mjs` checks that `/plan` reports the expected model before it measures).
   Stop the pod after A2.

Dry run on the Mac (nothing executed or written): `bash scripts/pod/setup.sh --dry-run && bash scripts/pod/run-all.sh --dry-run`.
