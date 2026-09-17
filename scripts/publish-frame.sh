#!/usr/bin/env bash
# Uploads packages/cache-frame/dist to the R2 bucket under frame/v1/ (Theo runs this; CC never does).
#
#   pnpm --filter cache-frame build && scripts/publish-frame.sh [bucket]      (default bucket: dianome)
#
# Each object gets its content-type and `cache-control: public, max-age=300`: the Cache Rule on cdn.dianome.dev
# uses origin cache-control, and the frame must stay revisable (never immutable). A breaking frame change ships
# as frame/v2/ with a new FRAME_PATH in the SDK, not by overwriting v1.
#
# Host pages that set Cross-Origin-Embedder-Policy can only embed the frame if the frame's own response carries
# `Cross-Origin-Embedder-Policy: credentialless` and `Cross-Origin-Resource-Policy: cross-origin`. R2 cannot serve
# those from object metadata: add a Transform Rule (Modify Response Header) for path prefix /frame/v1/ on
# cdn.dianome.dev, next to the Timing-Allow-Origin rule from Phase 2. scripts/serve-store.mjs sends them locally.
set -euo pipefail

BUCKET="${1:-dianome}"
DIR="$(cd "$(dirname "$0")/.." && pwd)/packages/cache-frame/dist"
PREFIX="frame/v1"
CACHE_CONTROL="public, max-age=300"

[ -f "$DIR/frame.js" ] && [ -f "$DIR/index.html" ] && [ -f "$DIR/optin.html" ] && [ -f "$DIR/diag.html" ] || { echo "build first: pnpm --filter cache-frame build" >&2; exit 1; }

put() { # put <file> <content-type>
  echo "put $BUCKET/$PREFIX/$1 ($2)"
  wrangler r2 object put "$BUCKET/$PREFIX/$1" --file "$DIR/$1" --content-type "$2" --cache-control "$CACHE_CONTROL" --remote
}
put frame.js   "text/javascript; charset=utf-8"
put index.html "text/html; charset=utf-8"
put optin.html "text/html; charset=utf-8"
put diag.html  "text/html; charset=utf-8"
echo "done: https://cdn.dianome.dev/$PREFIX/index.html"
