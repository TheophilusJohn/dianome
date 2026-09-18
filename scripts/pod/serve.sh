#!/usr/bin/env bash
# Step 4 of the GPU day: dianome-server serve (fp16, cuda) on 127.0.0.1:SERVE_PORT + cloudflared to gpu.dianome.dev.
#
#   bash scripts/pod/serve.sh start|status|stop [--dry-run] [--model ID] [--port PORT]
#
# The pod serves ONE model at a time: `stop`, then `start --model qwen2.5-3b-instruct` to swap (A2 goes 7B -> 3B).
# start is idempotent (a live server on the port is left alone). The tunnel's public hostname must already route to
# http://localhost:8765 (Zero Trust -> Networks -> Tunnels -> gpu.dianome.dev; WebSockets are on by default).
# The Worker's SPLIT_SERVERS entry for the model must point at wss://gpu.dianome.dev + https://gpu.dianome.dev/plan.
# Logs: server/bench/results/l4/logs/serve-<model>.log and cloudflared.log; pids next to them.
. "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

CMD="${1:-status}"; shift || true
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) ;;
    --model) SERVE_MODEL_ARG="$2"; shift ;;
    --port) SERVE_PORT_ARG="$2"; shift ;;
    *) die "unknown argument $1" ;;
  esac
  shift
done
load_env
SERVE_MODEL="${SERVE_MODEL_ARG:-$SERVE_MODEL}"; SERVE_PORT="${SERVE_PORT_ARG:-$SERVE_PORT}"
PID_SERVER="$LOG_DIR/serve-$SERVE_MODEL.pid"; PID_TUNNEL="$LOG_DIR/cloudflared.pid"
LOG_SERVER="$LOG_DIR/serve-$SERVE_MODEL.log"; LOG_TUNNEL="$LOG_DIR/cloudflared.log"
PLAN_URL="http://127.0.0.1:$SERVE_PORT/plan"

alive() { [ -f "$1" ] && kill -0 "$(cat "$1")" 2>/dev/null; }
plan_up() { curl -fsS --max-time 3 "$PLAN_URL" >/dev/null 2>&1; }

case "$CMD" in
  start)
    require_env SPLIT_TOKEN SPLIT_SIGNING_KEY CLOUDFLARED_TOKEN
    if [ "$DRY_RUN" = 0 ] && plan_up; then
      log "server already answering $PLAN_URL; not starting another"
    else
      run mkdir -p "$LOG_DIR"
      if [ "$DRY_RUN" = 1 ]; then
        printf '[dry-run] SPLIT_TOKEN=… SPLIT_SIGNING_KEY=… nohup %s serve --model %s --device cuda --host 127.0.0.1 --port %s > %s 2>&1 & (pid -> %s)\n' "$SERVER_BIN" "$SERVE_MODEL" "$SERVE_PORT" "$LOG_SERVER" "$PID_SERVER"
        printf '[dry-run] wait up to 600 s for %s (model load + startup microbench)\n' "$PLAN_URL"
      else
        log "+ dianome-server serve --model $SERVE_MODEL --device cuda --host 127.0.0.1 --port $SERVE_PORT (log $LOG_SERVER)"
        SPLIT_TOKEN="$SPLIT_TOKEN" SPLIT_SIGNING_KEY="$SPLIT_SIGNING_KEY" nohup "$SERVER_BIN" serve --model "$SERVE_MODEL" --device cuda --host 127.0.0.1 --port "$SERVE_PORT" > "$LOG_SERVER" 2>&1 &
        echo $! > "$PID_SERVER"
        for _ in $(seq 1 120); do plan_up && break; alive "$PID_SERVER" || die "server exited; see $LOG_SERVER"; sleep 5; done
        plan_up || die "no answer from $PLAN_URL after 600 s; see $LOG_SERVER"
        log "server up: $(curl -fsS "$PLAN_URL")"
      fi
    fi
    if [ "$DRY_RUN" = 0 ] && alive "$PID_TUNNEL"; then
      log "cloudflared already running (pid $(cat "$PID_TUNNEL"))"
    elif [ "$DRY_RUN" = 1 ]; then
      printf '[dry-run] nohup cloudflared tunnel run --token … > %s 2>&1 & (pid -> %s)\n' "$LOG_TUNNEL" "$PID_TUNNEL"
    else
      log "+ cloudflared tunnel run --token … (log $LOG_TUNNEL)"
      nohup cloudflared tunnel run --token "$CLOUDFLARED_TOKEN" > "$LOG_TUNNEL" 2>&1 &
      echo $! > "$PID_TUNNEL"
      sleep 5; alive "$PID_TUNNEL" || die "cloudflared exited; see $LOG_TUNNEL"
    fi
    log "serving $SERVE_MODEL on 127.0.0.1:$SERVE_PORT behind the tunnel. Check from the laptop: curl https://gpu.dianome.dev/plan"
    ;;
  status)
    if [ "$DRY_RUN" = 1 ]; then printf '[dry-run] curl %s; kill -0 $(cat %s) $(cat %s)\n' "$PLAN_URL" "$PID_SERVER" "$PID_TUNNEL"; exit 0; fi
    if plan_up; then log "server: up ($(curl -fsS "$PLAN_URL"))"; else log "server: not answering $PLAN_URL"; fi
    if alive "$PID_TUNNEL"; then log "cloudflared: running (pid $(cat "$PID_TUNNEL"))"; else log "cloudflared: not running"; fi
    ;;
  stop)
    for f in "$PID_TUNNEL" "$PID_SERVER"; do
      if [ "$DRY_RUN" = 1 ]; then printf '[dry-run] kill $(cat %s)\n' "$f"
      elif alive "$f"; then run kill "$(cat "$f")"; rm -f "$f"; fi
    done
    ;;
  *) die "usage: serve.sh start|status|stop [--dry-run] [--model ID] [--port PORT]" ;;
esac
