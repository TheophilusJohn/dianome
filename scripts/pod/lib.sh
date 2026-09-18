# Shared by scripts/pod/run-all.sh and serve.sh (setup.sh is self-contained: it runs before the repo is cloned).
# Every script takes --dry-run: print every command instead of running it, never write a result file.
set -euo pipefail

POD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$POD_DIR/../.." && pwd)"
DRY_RUN=0
for _a in "$@"; do [ "$_a" = "--dry-run" ] && DRY_RUN=1; done

log()  { printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*"; }
warn() { printf '%s WARNING: %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }
die()  { printf '%s ERROR: %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; exit 1; }
# run <cmd...>: echo under --dry-run, else log and execute.
run()  { if [ "$DRY_RUN" = 1 ]; then printf '[dry-run] %s\n' "$*"; else log "+ $*"; "$@"; fi; }
# python for JSON bookkeeping: the pod venv when it exists, else whatever python3 is on PATH (the Mac dry-run).
py() { if [ -x "$VENV/bin/python" ]; then "$VENV/bin/python" "$@"; else python3 "$@"; fi; }

VENV="${VENV:-$REPO/.venv}"
SERVER_BIN="$VENV/bin/dianome-server"
INGEST_BIN="$VENV/bin/dianome-ingest"
STORE="${STORE:-$REPO/store}"
BENCH_OUT="$REPO/server/bench/results/l4"
PROBES_OUT="$REPO/server/probes/results/l4"
ACTS_DIR="${ACTS_DIR:-$REPO/server/probes/data-l4}"
LOG_DIR="$BENCH_OUT/logs"
TIMING="$BENCH_OUT/timing.json"

# load_env: .env.pod (ENV_FILE overrides); under --dry-run fall back to .env.pod.example so the plan still prints.
load_env() {
  ENV_FILE="${ENV_FILE:-$REPO/.env.pod}"
  if [ -f "$ENV_FILE" ]; then
    set -a; . "$ENV_FILE"; set +a
    log "env: $ENV_FILE"
  elif [ "$DRY_RUN" = 1 ]; then
    warn "$ENV_FILE not found; dry-run uses .env.pod.example (empty values)"
    set -a; . "$REPO/.env.pod.example"; set +a
  else
    die "$ENV_FILE not found (copy .env.pod.example, fill it in)"
  fi
  export HF_HOME="${HF_HOME:-/workspace/hf}"
  GPU_BUDGET_HOURS="${GPU_BUDGET_HOURS:-4}"
  KEEP_ACTS="${KEEP_ACTS:-0}"
  FORCE_7B_PROBES="${FORCE_7B_PROBES:-0}"
  SERVE_MODEL="${SERVE_MODEL:-qwen2.5-7b-instruct}"
  SERVE_PORT="${SERVE_PORT:-8765}"
}

# require_env VAR...: every variable non-empty (a warning under --dry-run, fatal otherwise).
require_env() {
  local missing=()
  for v in "$@"; do [ -n "${!v:-}" ] || missing+=("$v"); done
  [ "${#missing[@]}" = 0 ] && return 0
  if [ "$DRY_RUN" = 1 ]; then warn "unset in env: ${missing[*]} (the real run refuses)"; else die "unset in $ENV_FILE: ${missing[*]}"; fi
}

# -- timing: one JSON file, {"steps": {name: {"seconds": s, "started": iso, "finished": iso, "skipped": bool}}} --------
STEP_NAME=""; STEP_T0=0
step_begin() { STEP_NAME="$1"; STEP_T0=$(date +%s); log "== step $1"; }
# step_end [skipped]: record the step, print elapsed for the step and the running total for the GPU day.
step_end() {
  local secs=$(( $(date +%s) - STEP_T0 )) skipped="${1:-}"
  if [ "$DRY_RUN" = 0 ]; then
    mkdir -p "$BENCH_OUT"
    py - "$TIMING" "$STEP_NAME" "$secs" "$skipped" <<'PY'
import json, sys, datetime, os
path, name, secs, skipped = sys.argv[1], sys.argv[2], int(sys.argv[3]), bool(sys.argv[4])
d = json.load(open(path)) if os.path.exists(path) else {"steps": {}}
prev = d["steps"].get(name, {})
if skipped and prev and not prev.get("skipped"):
    prev["last_checked"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"); d["steps"][name] = prev
else:
    now = datetime.datetime.now(datetime.timezone.utc)
    d["steps"][name] = {"seconds": secs, "started": (now - datetime.timedelta(seconds=secs)).isoformat(timespec="seconds"),
                        "finished": now.isoformat(timespec="seconds"), "skipped": skipped}
d["total_seconds"] = sum(s["seconds"] for s in d["steps"].values() if not s.get("skipped"))
json.dump(d, open(path, "w"), indent=2)
PY
  fi
  local total; total=$(total_seconds)
  if [ -n "$skipped" ]; then log "-- $STEP_NAME: skipped (result exists); GPU day so far $(fmt_h "$total") of ${GPU_BUDGET_HOURS} h"
  else log "-- $STEP_NAME: $(fmt_h "$secs"); GPU day so far $(fmt_h "$total") of ${GPU_BUDGET_HOURS} h"; fi
}
total_seconds() { [ -f "$TIMING" ] && py -c "import json,sys; print(json.load(open(sys.argv[1])).get('total_seconds', 0))" "$TIMING" || echo 0; }
step_seconds() { [ -f "$TIMING" ] && py -c "import json,sys; print(json.load(open(sys.argv[1]))['steps'].get(sys.argv[2], {}).get('seconds', 0))" "$TIMING" "$1" || echo 0; }
fmt_h() { py -c "import sys; s=int(sys.argv[1]); print(f'{s}s ({s/3600:.2f} h)')" "$1"; }

# models: id -> HF repo and pinned revision, read from server/dianome_server/model.py (one source of truth).
model_repo() { py - "$REPO/server/dianome_server/model.py" "$1" repo <<'PY'
import re, sys
text = open(sys.argv[1]).read()
m = re.search(r'"%s":\s*\("([^"]+)",\s*("([0-9a-f]+)"|None)\)' % re.escape(sys.argv[2]), text)
if not m: sys.exit("model %s not in MODELS" % sys.argv[2])
print(m.group(1) if sys.argv[3] == "repo" else (m.group(3) or ""))
PY
}
model_rev() { py - "$REPO/server/dianome_server/model.py" "$1" rev <<'PY'
import re, sys
text = open(sys.argv[1]).read()
m = re.search(r'"%s":\s*\("([^"]+)",\s*("([0-9a-f]+)"|None)\)' % re.escape(sys.argv[2]), text)
if not m: sys.exit("model %s not in MODELS" % sys.argv[2])
print(m.group(3) or "")
PY
}
