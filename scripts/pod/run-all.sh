#!/usr/bin/env bash
# Phase 7 GPU day on the L4 pod, in order, resumable: a step whose result file exists is skipped.
#
#   bash scripts/pod/run-all.sh [--dry-run] [--only ingest|bench|probes|serve] [--from STEP]
#
# Runs after scripts/pod/setup.sh, from the repo root on the pod, with .env.pod filled in. Results:
#   server/bench/results/l4/     ingest-<id>.json upload-<id>.json (step 1), cuda-<id>-<date>.{json,csv,md} (step 2),
#                                timing.json (elapsed per step), logs/ (every step's stdout, gitignored)
#   server/probes/results/l4/    linear-<id>.json (step 3; {"not_run": true, ...} for 7B when the budget rule says no)
# Steps:
#   1. ingest   pack Qwen2.5-3B/7B-Instruct as q8 + q4 (fp16 is not uploaded: the server loads it from the HF cache),
#               inspect --json, upload to R2 from the pod
#   2. bench    dianome-server bench on cuda for 0.5B, 3B, 7B: N grid = Phase 4's scaled to L (--splits auto),
#               128 new tokens, 3 runs; refuses while bench/rates.json has usd_per_hour null (--require-rate)
#   3. probes   3B: collect 50k held-out tokens (same held-out set as Phase 4) + linear-probe with a 200k-token training
#               set, all boundaries. 7B: the same only if elapsed + 1.5 x the 3B probe time stays under GPU_BUDGET_HOURS
#               (FORCE_7B_PROBES=1 overrides); otherwise linear-qwen2.5-7b-instruct.json records not_run and why.
#               Inversion decoders are not run on the L4 (on 0.5B they were within a few points of the linear probe).
#   4. serve    dianome-server serve for 7B (fp16, cuda) behind cloudflared as gpu.dianome.dev, left running for A2
#               (scripts/pod/serve.sh start|status|stop).
# Elapsed GPU time is printed after every step and kept in timing.json; the target is under 4 hours in total.
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

ONLY=""; FROM=""
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) ;;
    --only) ONLY="$2"; shift ;;
    --from) FROM="$2"; shift ;;
    -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
    *) die "unknown argument $1" ;;
  esac
  shift
done
STEPS=(ingest bench probes serve)
wanted() {  # wanted STEP: honours --only / --from
  local s="$1" i started=1
  [ -n "$ONLY" ] && { [ "$ONLY" = "$s" ]; return; }
  [ -z "$FROM" ] && return 0
  started=0; for i in "${STEPS[@]}"; do [ "$i" = "$FROM" ] && started=1; [ "$i" = "$s" ] && { [ "$started" = 1 ]; return; }; done
  return 1
}

load_env
require_env R2_ACCOUNT_ID R2_ACCESS_KEY_ID R2_SECRET_ACCESS_KEY R2_BUCKET SPLIT_SIGNING_KEY SPLIT_TOKEN CLOUDFLARED_TOKEN
BUDGET_SECONDS=$(py -c "import sys; print(int(float(sys.argv[1]) * 3600))" "$GPU_BUDGET_HOURS")
MODELS=(qwen2.5-0.5b-instruct qwen2.5-3b-instruct qwen2.5-7b-instruct)
INGEST_MODELS=(qwen2.5-3b-instruct qwen2.5-7b-instruct)

# -- preflight ---------------------------------------------------------------------------------------------------------
log "run-all: repo $REPO ($(git -C "$REPO" describe --tags --always 2>/dev/null || echo '?')), venv $VENV, store $STORE, budget ${GPU_BUDGET_HOURS} h"
if [ "$DRY_RUN" = 0 ]; then
  [ -x "$SERVER_BIN" ] && [ -x "$INGEST_BIN" ] || die "$VENV is missing dianome-server/dianome-ingest: run scripts/pod/setup.sh first"
  "$VENV/bin/python" -c 'import torch; assert torch.cuda.is_available(), "no CUDA device"; print("cuda:", torch.cuda.get_device_name(0))'
  mkdir -p "$BENCH_OUT" "$PROBES_OUT" "$LOG_DIR" "$ACTS_DIR"
  exec > >(tee -a "$LOG_DIR/run-all-$(date -u +%Y%m%dT%H%M%SZ).log") 2>&1
  free_gb=$(df -Pk "$REPO" | awk 'NR==2 {print int($4/1048576)}')
  [ "$free_gb" -ge 80 ] || warn "only ${free_gb} GB free under $REPO; the probe activation stores need up to ~52 GB at peak"
else
  [ -x "$SERVER_BIN" ] || log "[dry-run] $VENV does not exist here (expected off the pod); commands are printed only"
  printf '[dry-run] %s/bin/python -c "import torch; assert torch.cuda.is_available()"\n' "$VENV"
fi
rate_ok=$(py -c "import json,sys; print('1' if json.load(open(sys.argv[1])).get('usd_per_hour') is not None else '0')" "$REPO/server/bench/rates.json")
if [ "$rate_ok" != 1 ]; then
  if [ "$DRY_RUN" = 1 ]; then warn "server/bench/rates.json has usd_per_hour null: the real run refuses at step 2 (bench --require-rate). Theo fills the L4 rate first."
  else die "server/bench/rates.json has usd_per_hour null: fill the L4 rate (gpu, usd_per_hour, source, retrieved) before the GPU day"; fi
fi
for m in "${MODELS[@]}"; do log "model $m: $(model_repo "$m") @ $(model_rev "$m")"; done

# -- 1. ingest ----------------------------------------------------------------------------------------------------------
if wanted ingest; then
  for id in "${INGEST_MODELS[@]}"; do
    step_begin "ingest-$id"
    if [ -f "$BENCH_OUT/upload-$id.json" ]; then step_end skipped; continue; fi
    if [ -f "$STORE/manifests/$id/latest.json" ]; then
      log "pack-model $id: $STORE/manifests/$id/latest.json exists, not repacking"
    else
      run "$INGEST_BIN" pack-model --repo "$(model_repo "$id")" --revision "$(model_rev "$id")" --id "$id" --variants q8,q4 --out "$STORE"
    fi
    run "$INGEST_BIN" inspect --store "$STORE" --id "$id" --json "$BENCH_OUT/ingest-$id.json"
    run "$INGEST_BIN" upload --store "$STORE" --id "$id" --json "$BENCH_OUT/upload-$id.json"
    step_end
  done
fi

# -- 2. bench -----------------------------------------------------------------------------------------------------------
if wanted bench; then
  for id in "${MODELS[@]}"; do
    step_begin "bench-$id"
    if ls "$BENCH_OUT"/cuda-"$id"-*.json >/dev/null 2>&1; then step_end skipped; continue; fi
    run "$SERVER_BIN" bench --model "$id" --device cuda --splits auto --gen-tokens 128 --runs 3 --out-dir "$BENCH_OUT" --require-rate
    step_end
  done
fi

# -- 3. probes ----------------------------------------------------------------------------------------------------------
probe_complete() {  # probe_complete <results.json>: every boundary has a row (linear-probe is resumable per boundary)
  [ -f "$1" ] && py - "$1" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
if d.get("not_run"): sys.exit(0)
rows = {r["boundary"] for r in d.get("rows", [])}
sys.exit(0 if d.get("test_meta") and rows == set(range(d["test_meta"]["boundaries"])) else 1)
PY
}
probe_model() {  # probe_model <id>: collect (skips if meta.json exists) + linear-probe (resumes), then drop the activations
  local id="$1" out="$PROBES_OUT/linear-$1.json"
  run "$SERVER_BIN" collect --model "$id" --device cuda --tokens 50000 --data-dir "$ACTS_DIR/$id/test"
  run "$SERVER_BIN" linear-probe --model "$id" --device cuda --train-tokens 200000 --test-dir "$ACTS_DIR/$id/test" --train-dir "$ACTS_DIR/$id/train" --out "$out"
  if [ "$KEEP_ACTS" = 1 ]; then log "KEEP_ACTS=1: leaving $ACTS_DIR/$id"; else run rm -rf "$ACTS_DIR/$id"; fi
}
if wanted probes; then
  step_begin "probes-qwen2.5-3b-instruct"
  if probe_complete "$PROBES_OUT/linear-qwen2.5-3b-instruct.json"; then step_end skipped; else probe_model qwen2.5-3b-instruct; step_end; fi

  step_begin "probes-qwen2.5-7b-instruct"
  out7="$PROBES_OUT/linear-qwen2.5-7b-instruct.json"
  if probe_complete "$out7" && [ "$FORCE_7B_PROBES" != 1 ]; then
    step_end skipped
  else
    t3b=$(step_seconds probes-qwen2.5-3b-instruct); total=$(total_seconds)
    projected=$(py -c "import sys; print(int(float(sys.argv[1]) + 1.5 * float(sys.argv[2])))" "$total" "$t3b")
    log "budget rule for 7B probes: elapsed ${total}s + 1.5 x 3B probe ${t3b}s = ${projected}s vs budget ${BUDGET_SECONDS}s (FORCE_7B_PROBES=$FORCE_7B_PROBES)"
    if [ "$FORCE_7B_PROBES" = 1 ] || { [ "$t3b" -gt 0 ] && [ "$projected" -le "$BUDGET_SECONDS" ]; }; then
      probe_model qwen2.5-7b-instruct
    else
      reason="not run: elapsed ${total}s + 1.5 x the 3B probe (${t3b}s) = ${projected}s would exceed the ${GPU_BUDGET_HOURS} h budget (${BUDGET_SECONDS}s)"
      [ "$t3b" -gt 0 ] || reason="not run: the 3B probe's duration is unknown in timing.json, so the budget rule cannot admit it (FORCE_7B_PROBES=1 overrides)"
      log "7B probes $reason"
      if [ "$DRY_RUN" = 1 ]; then printf '[dry-run] write %s {"not_run": true, "reason": ...}\n' "$out7"
      else py -c 'import json,sys,datetime; json.dump({"model":"qwen2.5-7b-instruct","not_run":True,"reason":sys.argv[2],"elapsed_seconds":int(sys.argv[3]),"budget_seconds":int(sys.argv[4]),"date":datetime.date.today().isoformat()}, open(sys.argv[1],"w"), indent=2)' "$out7" "$reason" "$total" "$BUDGET_SECONDS"; fi
    fi
    step_end
  fi
fi

# -- 4. serve -----------------------------------------------------------------------------------------------------------
if wanted serve; then
  step_begin "serve-$SERVE_MODEL"
  if [ "$DRY_RUN" = 1 ]; then "$POD_DIR/serve.sh" start --dry-run; else "$POD_DIR/serve.sh" start; fi
  step_end
fi

log "run-all done: GPU day total $(fmt_h "$(total_seconds)") of ${GPU_BUDGET_HOURS} h (timing: $TIMING)"
[ "$DRY_RUN" = 1 ] && log "dry-run: nothing was executed or written"
exit 0
