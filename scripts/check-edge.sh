#!/usr/bin/env bash
# Edge verification for Phase 2. Run after `pnpm --filter worker deploy`:
#
#   scripts/check-edge.sh [manifest-id]          (default: qwen2.5-0.5b-instruct)
#   API=https://api.dianome.dev CDN=https://cdn.dianome.dev scripts/check-edge.sh
#
# Reads one chunk id from the manifest via the API, then checks cdn.dianome.dev/chunks/<sha>:
#   1. HEAD: 200, accept-ranges: bytes, cache-control has immutable, etag present, ACAO *, timing-allow-origin *
#   2. GET Range bytes=0-1023: 206, content-range, exactly 1024 bytes
#   3. Two consecutive full GETs: print cf-cache-status for each; the second must be HIT
#   4. If-None-Match with the ETag: 304
#   5. Manifest via the API: 200, then 304 on If-None-Match
# Prints one PASS/FAIL line per check and exits non-zero if any check failed.
# R2 only emits CORS headers when the request carries an Origin, so every CDN request sends one.
set -u

ID="${1:-qwen2.5-0.5b-instruct}"
API="${API:-https://api.dianome.dev}"
CDN="${CDN:-https://cdn.dianome.dev}"
ORIGIN="${ORIGIN:-https://dianome.dev}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

fails=0
pass() { printf 'PASS  %s\n' "$1"; }
fail() { printf 'FAIL  %s\n' "$1"; fails=$((fails + 1)); }
check() { if [ "$1" = "1" ]; then pass "$2"; else fail "$2"; fi; }   # check <0|1> <label>

# header <file> <name>: lower-cased header value from a curl -D dump (last response block only)
header() {
  awk -v want="$2" 'BEGIN{IGNORECASE=1} /^HTTP\//{block=""} {block=block $0 "\n"} END{printf "%s", block}' "$1" \
    | tr -d '\r' | awk -v want="$2" 'BEGIN{IGNORECASE=1} tolower($0) ~ "^" tolower(want) ":" {sub(/^[^:]*:[ \t]*/, ""); print; exit}'
}
status() { tr -d '\r' < "$1" | awk '/^HTTP\//{s=$2} END{print s}'; }

echo "manifest id: $ID"
echo "api:         $API"
echo "cdn:         $CDN"
echo

# ---- pick a chunk from the manifest via the API -------------------------------------------------
curl -sS -D "$TMP/m1.h" -o "$TMP/manifest.json" -H "Origin: $ORIGIN" "$API/v1/models/$ID/manifest"
m1_status="$(status "$TMP/m1.h")"
if [ "$m1_status" != "200" ]; then
  fail "fetch manifest via API ($API/v1/models/$ID/manifest -> ${m1_status:-no response})"
  exit 1
fi
SHA="$(grep -oE '[0-9a-f]{64}' "$TMP/manifest.json" | head -n 1)"
if [ -z "$SHA" ]; then fail "no chunk id found in manifest"; exit 1; fi
echo "chunk:       $SHA"
echo "manifest sha: $(header "$TMP/m1.h" x-dianome-manifest-sha)"
echo

URL="$CDN/chunks/$SHA"

# ---- 1. HEAD -----------------------------------------------------------------------------------
curl -sS -I -D "$TMP/1.h" -o /dev/null -H "Origin: $ORIGIN" "$URL"
s="$(status "$TMP/1.h")"
check "$([ "$s" = "200" ] && echo 1 || echo 0)" "1. HEAD status 200 (got ${s:-none})"
check "$([ "$(header "$TMP/1.h" accept-ranges)" = "bytes" ] && echo 1 || echo 0)" "1. HEAD accept-ranges: bytes"
cc="$(header "$TMP/1.h" cache-control)"
check "$(printf '%s' "$cc" | grep -q immutable && echo 1 || echo 0)" "1. HEAD cache-control contains immutable (got '$cc')"
ETAG="$(header "$TMP/1.h" etag)"
check "$([ -n "$ETAG" ] && echo 1 || echo 0)" "1. HEAD etag present (got '$ETAG')"
check "$([ "$(header "$TMP/1.h" access-control-allow-origin)" = "*" ] && echo 1 || echo 0)" "1. HEAD access-control-allow-origin: *"
check "$([ "$(header "$TMP/1.h" timing-allow-origin)" = "*" ] && echo 1 || echo 0)" "1. HEAD timing-allow-origin: *"

# ---- 2. Range -----------------------------------------------------------------------------------
curl -sS -D "$TMP/2.h" -o "$TMP/range.bin" -H "Origin: $ORIGIN" -H "Range: bytes=0-1023" "$URL"
s="$(status "$TMP/2.h")"
check "$([ "$s" = "206" ] && echo 1 || echo 0)" "2. Range status 206 (got ${s:-none})"
cr="$(header "$TMP/2.h" content-range)"
check "$([ -n "$cr" ] && echo 1 || echo 0)" "2. Range content-range present (got '$cr')"
n="$(wc -c < "$TMP/range.bin" | tr -d ' ')"
check "$([ "$n" = "1024" ] && echo 1 || echo 0)" "2. Range body is 1024 bytes (got $n)"

# ---- 3. Two consecutive full GETs ----------------------------------------------------------------
curl -sS -D "$TMP/3a.h" -o /dev/null -H "Origin: $ORIGIN" "$URL"
cs1="$(header "$TMP/3a.h" cf-cache-status)"
curl -sS -D "$TMP/3b.h" -o /dev/null -H "Origin: $ORIGIN" "$URL"
cs2="$(header "$TMP/3b.h" cf-cache-status)"
echo "      cf-cache-status: first=$cs1 second=$cs2"
check "$([ "$cs2" = "HIT" ] && echo 1 || echo 0)" "3. second full GET is cf-cache-status: HIT (got '$cs2')"

# ---- 4. If-None-Match ---------------------------------------------------------------------------
curl -sS -D "$TMP/4.h" -o /dev/null -H "Origin: $ORIGIN" -H "If-None-Match: $ETAG" "$URL"
s="$(status "$TMP/4.h")"
check "$([ "$s" = "304" ] && echo 1 || echo 0)" "4. If-None-Match status 304 (got ${s:-none})"

# ---- 5. Manifest via the API: 200 then 304 ------------------------------------------------------
check "$([ "$m1_status" = "200" ] && echo 1 || echo 0)" "5. API manifest status 200"
metag="$(header "$TMP/m1.h" etag)"
curl -sS -D "$TMP/5.h" -o /dev/null -H "Origin: $ORIGIN" -H "If-None-Match: $metag" "$API/v1/models/$ID/manifest"
s="$(status "$TMP/5.h")"
check "$([ "$s" = "304" ] && echo 1 || echo 0)" "5. API manifest If-None-Match status 304 (got ${s:-none})"

echo
if [ "$fails" -eq 0 ]; then echo "all checks passed"; else echo "$fails check(s) failed"; fi
exit "$([ "$fails" -eq 0 ] && echo 0 || echo 1)"
