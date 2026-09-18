#!/usr/bin/env bash
# Phase 7 pod setup (self-contained: runs on a fresh RunPod PyTorch pod before the repo exists there).
#
#   bash setup.sh [--dry-run] [--tag TAG] [--dir DIR]
#
# Reads .env.pod (ENV_FILE, ./.env.pod, or next to this script). Idempotent: every step checks before it acts.
#   1. cloudflared binary          -> /usr/local/bin/cloudflared
#   2. git clone at TAG            -> DIR (default /workspace/dianome), .env.pod copied in
#   3. one venv for server/ + ingest/  -> DIR/.venv (--system-site-packages: reuses the image's CUDA torch)
#   4. Hugging Face downloads      -> HF_HOME (default /workspace/hf): the three Qwen2.5 Instruct models at the
#                                     revisions pinned in server/dianome_server/model.py, and WikiText-103 (probes)
#   5. torch.cuda.is_available()   must be true
# Pod: NVIDIA L4 (24 GB), a RunPod PyTorch 2.x / CUDA 12 image with Python >= 3.11, >= 120 GB volume at /workspace
# (HF cache ~23 GB, chunk store ~18 GB, probe activations up to ~52 GB at peak, deleted after each probe).
set -euo pipefail

DRY_RUN=0; TAG=""; DIR=""
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --tag) TAG="$2"; shift ;;
    --dir) DIR="$2"; shift ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
  shift
done
log()  { printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*"; }
warn() { printf '%s WARNING: %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }
die()  { printf '%s ERROR: %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; exit 1; }
run()  { if [ "$DRY_RUN" = 1 ]; then printf '[dry-run] %s\n' "$*"; else log "+ $*"; "$@"; fi; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ENV_FILE:-}"
for cand in "$ENV_FILE" "./.env.pod" "$HERE/.env.pod" "$HERE/../../.env.pod"; do
  [ -n "$cand" ] && [ -f "$cand" ] && { ENV_FILE="$cand"; break; }
done
if [ -n "$ENV_FILE" ] && [ -f "$ENV_FILE" ]; then
  set -a; . "$ENV_FILE"; set +a; log "env: $ENV_FILE"
elif [ "$DRY_RUN" = 1 ] && [ -f "$HERE/../../.env.pod.example" ]; then
  warn ".env.pod not found; dry-run uses .env.pod.example"; set -a; . "$HERE/../../.env.pod.example"; set +a; ENV_FILE="$HERE/../../.env.pod.example"
else
  die ".env.pod not found (copy .env.pod.example next to setup.sh and fill it in)"
fi
DIR="${DIR:-${DIANOME_DIR:-/workspace/dianome}}"
TAG="${TAG:-${DIANOME_TAG:-main}}"
REPO_URL="${DIANOME_REPO:-https://github.com/TheophilusJohn/dianome.git}"
export HF_HOME="${HF_HOME:-/workspace/hf}"
VENV="$DIR/.venv"
log "setup: repo $REPO_URL @ $TAG -> $DIR, venv $VENV, HF_HOME $HF_HOME"
[ -n "${HF_TOKEN:-}" ] || warn "HF_TOKEN is empty (downloads still work for public repos, rate-limited)"

# 1. cloudflared
if command -v cloudflared >/dev/null 2>&1; then
  log "cloudflared: $(cloudflared --version 2>/dev/null | head -1)"
else
  run bash -c 'curl -fsSL https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o /usr/local/bin/cloudflared && chmod +x /usr/local/bin/cloudflared'
  [ "$DRY_RUN" = 1 ] || cloudflared --version
fi

# 2. clone / checkout
if [ -d "$DIR/.git" ]; then
  run git -C "$DIR" fetch --tags origin
else
  run mkdir -p "$(dirname "$DIR")"
  run git clone "$REPO_URL" "$DIR"
fi
run git -C "$DIR" checkout --quiet "$TAG"
[ "$DRY_RUN" = 1 ] || log "checked out $(git -C "$DIR" rev-parse HEAD) ($(git -C "$DIR" describe --tags --always))"
if [ "$ENV_FILE" != "$DIR/.env.pod" ] && [ "$(basename "$ENV_FILE")" = ".env.pod" ]; then
  run cp "$ENV_FILE" "$DIR/.env.pod"
fi

# 3. venv (server/ + ingest/ in one), reusing the image's torch
PYBIN="${PYTHON:-python3}"
if [ "$DRY_RUN" = 1 ]; then
  printf '[dry-run] %s -c "import sys; assert sys.version_info >= (3, 11)"\n' "$PYBIN"
else
  "$PYBIN" -c 'import sys; assert sys.version_info >= (3, 11), f"python {sys.version} < 3.11: pick a py3.11+ RunPod image"'
fi
if [ -x "$VENV/bin/python" ]; then
  log "venv exists: $VENV"
else
  run "$PYBIN" -m venv --system-site-packages "$VENV"
fi
run "$VENV/bin/python" -m pip install --quiet --upgrade pip
run "$VENV/bin/python" -m pip install --quiet -e "$DIR/server" -e "$DIR/ingest"
if [ "$DRY_RUN" = 0 ]; then
  "$VENV/bin/dianome-server" --help >/dev/null && "$VENV/bin/dianome-ingest" --help >/dev/null
  "$VENV/bin/python" -c 'import torch, transformers; print(f"torch {torch.__version__} (cuda {torch.version.cuda}), transformers {transformers.__version__}")'
fi

# 4. Hugging Face: models at the pinned revisions + WikiText-103 (both splits) into HF_HOME
run mkdir -p "$HF_HOME"
if [ "$DRY_RUN" = 1 ]; then
  printf '[dry-run] %s/bin/python -c "snapshot_download(...) for the MODELS in server/dianome_server/model.py; load_dataset(Salesforce/wikitext, wikitext-103-raw-v1, test+train)"\n' "$VENV"
else
  "$VENV/bin/python" - <<'PY'
import os, time
from huggingface_hub import snapshot_download
from dianome_server.model import MODELS
want = ["qwen2.5-0.5b-instruct", "qwen2.5-3b-instruct", "qwen2.5-7b-instruct"]
for mid in want:
    repo, rev = MODELS[mid]
    t0 = time.time()
    path = snapshot_download(repo, revision=rev, token=os.environ.get("HF_TOKEN") or None,
                             allow_patterns=["*.safetensors", "*.json", "merges.txt", "vocab.json", "vocab.txt"])
    print(f"  {mid}: {repo}@{rev or 'main'} -> {path} ({time.time() - t0:.0f}s)", flush=True)
import datasets
for split in ("test", "train"):
    t0 = time.time()
    ds = datasets.load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=split)
    print(f"  wikitext-103-raw-v1 {split}: {len(ds)} rows ({time.time() - t0:.0f}s)", flush=True)
PY
fi

# 5. CUDA
if [ "$DRY_RUN" = 1 ]; then
  printf '[dry-run] %s/bin/python -c "import torch; assert torch.cuda.is_available()"\n' "$VENV"
else
  "$VENV/bin/python" -c 'import torch; assert torch.cuda.is_available(), "torch.cuda.is_available() is False"; p=torch.cuda.get_device_properties(0); print(f"cuda ok: {p.name}, {p.total_memory/2**30:.1f} GiB")'
  df -h "$DIR" | tail -1 | awk '{print "disk: " $4 " free on " $6}'
fi
log "setup done. Next: cd $DIR && bash scripts/pod/run-all.sh"
