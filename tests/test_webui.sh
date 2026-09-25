#!/bin/bash
# The browser chat page: `GET /` and `GET /chat` serve the page embedded from
# src/webui/index.html, every other method on those paths is a 405, the API
# routes beside it answer as before, and `--api-key` leaves the page open while
# the API it calls stays behind the key.
#
# Fully hermetic: an EMPTY --model-dir discovers zero models, so nothing loads.
#
# Usage: ./tests/test_webui.sh [port]

set -u

PORT="${1:-18851}"
BINARY="${BINARY:-./zig-out/bin/sushi}"
PAGE="src/webui/index.html"
BASE="http://127.0.0.1:$PORT"
PASS=0
FAIL=0

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

check() {
    local desc="$1" ok="$2"
    if [ "$ok" = "1" ]; then
        PASS=$((PASS + 1)); echo -e "  ${GREEN}PASS${NC} $desc"
    else
        FAIL=$((FAIL + 1)); echo -e "  ${RED}FAIL${NC} $desc"
    fi
}
is() { [ "$1" = "$2" ] && echo 1 || echo 0; }

if [ ! -x "$BINARY" ]; then
    echo "[fail] $BINARY not found — build first: zig build -Doptimize=ReleaseFast"
    exit 1
fi

EMPTY_DIR="$(mktemp -d)"
WORK="$(mktemp -d)"
LOG="$WORK/server.log"
SPID=""
stop_server() {
    [ -n "$SPID" ] && kill "$SPID" 2>/dev/null && wait "$SPID" 2>/dev/null
    SPID=""
}
cleanup() {
    stop_server
    rm -rf "$EMPTY_DIR" "$WORK"
}
trap cleanup EXIT

boot() {
    stop_server
    : > "$LOG"
    HOME="$WORK" "$BINARY" --serve --model-dir "$EMPTY_DIR" --port "$PORT" --log-file off "$@" > "$LOG" 2>&1 &
    SPID=$!
    for _ in $(seq 1 60); do
        curl -sf "$BASE/health" >/dev/null 2>&1 && return 0
        kill -0 "$SPID" 2>/dev/null || break
        sleep 0.5
    done
    echo "  (server did not come up; log follows)"; cat "$LOG"
    return 1
}
code() { curl -s -o /dev/null -w '%{http_code}' "$@"; }

echo "Chat page (port $PORT)"

echo "[1/3] the page and its neighbours"
if boot; then
    for path in / /chat; do
        curl -s -D "$WORK/headers" -o "$WORK/body" "$BASE$path"
        check "GET $path is 200" "$(grep -q '^HTTP/1.1 200' "$WORK/headers" && echo 1 || echo 0)"
        check "GET $path is text/html; charset=utf-8" \
            "$(grep -qi '^Content-Type: text/html; charset=utf-8' "$WORK/headers" && echo 1 || echo 0)"
        check "GET $path body is the embedded page" "$(cmp -s "$WORK/body" "$PAGE" && echo 1 || echo 0)"
    done
    check "POST / is 405" "$(is "$(code -X POST "$BASE/" -d '{}')" 405)"
    check "GET /health still answers ok" "$(is "$(curl -s "$BASE/health")" '{"status":"ok"}')"
    check "GET /v1/models still lists" "$(is "$(curl -s "$BASE/v1/models")" '{"object":"list","data":[]}')"
    check "an unknown path is still 404" "$(is "$(code "$BASE/index.html")" 404)"
    check "a served API route still reports 503 with no model" \
        "$(is "$(code -X POST "$BASE/v1/chat/completions" -H 'Content-Type: application/json' -d '{}')" 503)"
    check "the boot banner prints the page URL once" \
        "$(is "$(grep -c "chat in your browser: http://127.0.0.1:$PORT/" "$LOG")" 1)"
else
    check "boot" 0
fi

echo "[2/3] --api-key --api-key-strict: page open, API behind the key"
if boot --api-key webui-test-key --api-key-strict; then
    check "GET / needs no key" "$(is "$(code "$BASE/")" 200)"
    check "GET /chat needs no key" "$(is "$(code "$BASE/chat")" 200)"
    check "GET /v1/models without the key is 401" "$(is "$(code "$BASE/v1/models")" 401)"
    check "GET /v1/models with the Bearer key is 200" \
        "$(is "$(code -H 'Authorization: Bearer webui-test-key' "$BASE/v1/models")" 200)"
    check "POST / without the key is 401" "$(is "$(code -X POST "$BASE/" -d '{}')" 401)"
else
    check "boot with --api-key" 0
fi

echo "[3/3] the page stays self-contained"
check "no external http(s) fetch or CDN in the page" \
    "$(grep -Eq '(src|href)="https?://|@import|fetch\("https?://' "$PAGE" && echo 0 || echo 1)"
check "the page is under 200 KB" "$([ "$(wc -c < "$PAGE")" -lt 204800 ] && echo 1 || echo 0)"

echo
echo "  passed: $PASS   failed: $FAIL"
[ "$FAIL" -eq 0 ]
