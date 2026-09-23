#!/usr/bin/env bash
# Per-model settings (`~/.sushi/model-settings.json`, issue #269): a model's
# `ctx_size` / `kv_quant` / `mtp` / `mtp_acceptance` follow the MODEL, apply on its
# load (boot AND cold load), and a second model in the same process keeps the
# globals. An explicit launch flag outranks the file.
#
# Runs under a private HOME so the real settings file is never touched.
# NEEDS REAL MODELS: skips when the two small defaults are absent.
#
# Usage: ./tests/test_model_settings.sh [port]
set -uo pipefail
PORT="${1:-11384}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/zig-out/bin/sushi"
[ -x "$BIN" ] || { echo "FAIL: build first (zig build -Doptimize=ReleaseFast)"; exit 1; }

MODELS_ROOT="${MODELS_ROOT:-$HOME/.sushi/models}"
MODEL_A="${MODEL_A:-/Users/beam/llm/models/Qwen3.8-Flash-Next-EXL3-K3-w12-mcg-plugged}"
MODEL_B="${MODEL_B:-/Users/beam/llm/models/MiMo-V2.6-Flash-Sushi2.5bpw}"
if [ ! -f "$MODEL_A/config.json" ] || [ ! -f "$MODEL_B/config.json" ]; then
    echo "SKIP: needs two local chat models (MODEL_A=$MODEL_A, MODEL_B=$MODEL_B)"
    exit 0
fi

PASS=0; FAIL=0
RED='\033[0;31m'; GREEN='\033[0;32m'; NC='\033[0m'
check() {
    if [ "$2" = "1" ]; then PASS=$((PASS + 1)); echo -e "  ${GREEN}PASS${NC} $1"
    else FAIL=$((FAIL + 1)); echo -e "  ${RED}FAIL${NC} $1"; fi
}

FAKE_HOME="$(mktemp -d)"
mkdir -p "$FAKE_HOME/.sushi"
SETTINGS="$FAKE_HOME/.sushi/model-settings.json"
LOG="$FAKE_HOME/server.log"
SRV=""
cleanup() {
    [ -n "$SRV" ] && kill "$SRV" 2>/dev/null
    pkill -f "sushi.*--port $PORT" 2>/dev/null
    rm -rf "$FAKE_HOME"
}
trap cleanup EXIT
pkill -f "sushi.*--port $PORT" 2>/dev/null
sleep 0.5

write_settings() { # write_settings <ctx> <kv>  — override for MODEL_A only
    cat >"$SETTINGS" <<JSON
{ "$MODEL_A/": { "ctx_size": $1, "kv_quant": "$2", "mtp": true, "mtp_acceptance": "typical" }, "not-a-model": 1 }
JSON
}
write_settings 4096 8

boot() { # boot <extra flags...> — MODEL_A primary; no --ctx-size / --kv-quant unless passed
    HOME="$FAKE_HOME" "$BIN" --serve --model "$MODEL_A" --model-dir "$MODELS_ROOT" --port "$PORT" --log-file off "$@" >"$LOG" 2>&1 &
    SRV=$!
    UP=0
    for _ in $(seq 1 240); do
        curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { UP=1; break; }
        kill -0 "$SRV" 2>/dev/null || break
        sleep 0.5
    done
    [ "$UP" = "1" ] || { echo "FAIL: server never became healthy"; tail -5 "$LOG"; exit 1; }
}
boot --no-mtp

row() { # row <model path> <ctx|kv|src> — context_length, meta.kv_quant or meta.kv_cache.source of the READY row
    curl -s "http://127.0.0.1:$PORT/v1/models" | python3 -c "
import sys, json
want = sys.argv[1].rstrip('/')
for m in json.load(sys.stdin)['data']:
    if want.endswith('/' + m['id']):
        f = sys.argv[2]
        print(m['context_length'] if f == 'ctx' else m['meta']['kv_cache']['source'] if f == 'src' else m['meta'].get('kv_quant'))
        break
" "$1" "$2"
}
props_mtp_source() { # props_mtp_source <model path> — /props settings.mtp.source
    local id; id="$(basename "$(dirname "$1")")/$(basename "$1")"
    curl -s "http://127.0.0.1:$PORT/props?model=$id" | python3 -c "import sys, json; print(json.load(sys.stdin)['settings']['mtp']['source'])"
}
post() { # post <route> <json>
    curl -s -o /dev/null -w '%{http_code}' --max-time 300 -X POST "http://127.0.0.1:$PORT/v1/$1" \
        -H 'Content-Type: application/json' -d "$2"
}

# [1] boot load honours the file where no flag was given; --no-mtp outranks its mtp: true
check "[1] boot: context_length 4096 from the file (got $(row "$MODEL_A" ctx))" "$([ "$(row "$MODEL_A" ctx)" = "4096" ] && echo 1 || echo 0)"
check "[1] boot: meta.kv_quant 8 from the file (got $(row "$MODEL_A" kv))" "$([ "$(row "$MODEL_A" kv)" = "8" ] && echo 1 || echo 0)"
check "[1] boot: kv_cache source model-settings.json (got $(row "$MODEL_A" src))" "$([ "$(row "$MODEL_A" src)" = "model-settings.json" ] && echo 1 || echo 0)"
check "[1] load log names the KV and ctx choices" "$(grep -q "\[kv-cache\] kv8 (model-settings.json); ctx 4096 (model-settings.json)" "$LOG" && echo 1 || echo 0)"
check "[1] load log: --no-mtp outranks mtp:true, acceptance from the file" \
    "$(grep -q "\[mtp\] off (--no-mtp); acceptance typical (model-settings.json)" "$LOG" && echo 1 || echo 0)"
check "[1] /props settings.mtp.source --no-mtp (got $(props_mtp_source "$MODEL_A"))" "$([ "$(props_mtp_source "$MODEL_A")" = "--no-mtp" ] && echo 1 || echo 0)"
check "[1] log names the override" "$(grep -q "\[model-settings\] .*ctx=4096 kv=8 mtp=on" "$LOG" && echo 1 || echo 0)"
check "[1] log names the MTP acceptance mode" "$(grep -q "\[model-settings\] .*accept=typical" "$LOG" && echo 1 || echo 0)"

# [2] a second model keeps the globals, and its cold load carries the explicit --no-mtp
CODE="$(post load-model "{\"model\":\"$MODEL_B\"}")"
check "[2] cold load of model B -> 200 (got $CODE)" "$([ "$CODE" = "200" ] && echo 1 || echo 0)"
check "[2] model B keeps the kv8 default (got $(row "$MODEL_B" kv))" "$([ "$(row "$MODEL_B" kv)" = "8" ] && echo 1 || echo 0)"
check "[2] model B kv_cache source default (got $(row "$MODEL_B" src))" "$([ "$(row "$MODEL_B" src)" = "default" ] && echo 1 || echo 0)"
check "[2] cold-load log names the KV and ctx choices" "$(grep -q "\[kv-cache\] kv8 (default); ctx auto (default)" "$LOG" && echo 1 || echo 0)"
check "[2] cold-load log: --no-mtp, exact acceptance" "$(grep -q "\[mtp\] off (--no-mtp); acceptance exact (default)" "$LOG" && echo 1 || echo 0)"
check "[2] /props settings.mtp.source --no-mtp for B (got $(props_mtp_source "$MODEL_B"))" "$([ "$(props_mtp_source "$MODEL_B")" = "--no-mtp" ] && echo 1 || echo 0)"
check "[2] model A still 4096 (got $(row "$MODEL_A" ctx))" "$([ "$(row "$MODEL_A" ctx)" = "4096" ] && echo 1 || echo 0)"

# [3] edit + unload + load applies the new values, no restart
write_settings 8192 4
CODE="$(post unload-model "{\"model\":\"$MODEL_A\"}")"
check "[3] unload model A -> 200 (got $CODE)" "$([ "$CODE" = "200" ] && echo 1 || echo 0)"
CODE="$(post load-model "{\"model\":\"$MODEL_A\"}")"
check "[3] reload model A -> 200 (got $CODE)" "$([ "$CODE" = "200" ] && echo 1 || echo 0)"
check "[3] model A now 8192 (got $(row "$MODEL_A" ctx))" "$([ "$(row "$MODEL_A" ctx)" = "8192" ] && echo 1 || echo 0)"
check "[3] model A now kv 4 (got $(row "$MODEL_A" kv))" "$([ "$(row "$MODEL_A" kv)" = "4" ] && echo 1 || echo 0)"
kill -0 "$SRV" 2>/dev/null; check "[3] server never restarted" "$([ $? = 0 ] && echo 1 || echo 0)"

# [4] a malformed file never stops a load
echo '{nope' >"$SETTINGS"
post unload-model "{\"model\":\"$MODEL_A\"}" >/dev/null
CODE="$(post load-model "{\"model\":\"$MODEL_A\"}")"
check "[4] malformed file: load -> 200 (got $CODE), defaults apply (kv source $(row "$MODEL_A" src))" \
    "$([ "$CODE" = "200" ] && [ "$(row "$MODEL_A" src)" = "default" ] && echo 1 || echo 0)"
check "[4] malformed file logged" "$(grep -q "\[model-settings\] .*malformed" "$LOG" && echo 1 || echo 0)"

# [5] ssd_budget_gb on a model that does not stream experts: ignored, warned once, load still 200
cat >"$SETTINGS" <<JSON
{ "$MODEL_A/": { "ssd_budget_gb": 60 } }
JSON
post unload-model "{\"model\":\"$MODEL_A\"}" >/dev/null
CODE="$(post load-model "{\"model\":\"$MODEL_A\"}")"
check "[5] ssd_budget_gb on a non-streaming model: load -> 200 (got $CODE)" "$([ "$CODE" = "200" ] && echo 1 || echo 0)"
check "[5] the setting is logged" "$(grep -q "\[model-settings\] .*ssd_budget_gb=60" "$LOG" && echo 1 || echo 0)"
check "[5] one line says it is ignored" \
    "$([ "$(grep -c "ssd_budget_gb ignored" "$LOG")" = "1" ] && echo 1 || echo 0)"

# [6] explicit launch flags outrank every competing key in the file
kill "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null; SRV=""
cat >"$SETTINGS" <<JSON
{ "$MODEL_A/": { "ctx_size": 4096, "kv_quant": "8", "mtp": false, "mtp_acceptance": "typical" } }
JSON
boot --ctx-size 16384 --kv-quant 4 --mtp --mtp-tokenv3 0.9
check "[6] --ctx-size 16384 outranks ctx_size 4096 (got $(row "$MODEL_A" ctx))" "$([ "$(row "$MODEL_A" ctx)" = "16384" ] && echo 1 || echo 0)"
check "[6] --kv-quant 4 outranks kv_quant 8 (got $(row "$MODEL_A" kv), source $(row "$MODEL_A" src))" \
    "$([ "$(row "$MODEL_A" kv)" = "4" ] && [ "$(row "$MODEL_A" src)" = "--kv-quant" ] && echo 1 || echo 0)"
check "[6] load log names the flags" "$(grep -q "\[kv-cache\] kv4 (--kv-quant); ctx 16384 (--ctx-size)" "$LOG" && echo 1 || echo 0)"
check "[6] --mtp outranks mtp:false, --mtp-tokenv3 outranks typical" \
    "$(grep -q "\[mtp\] on (--mtp); acceptance tokenv3 (--mtp-tokenv3)" "$LOG" && echo 1 || echo 0)"
check "[6] /props settings.mtp.source --mtp (got $(props_mtp_source "$MODEL_A"))" "$([ "$(props_mtp_source "$MODEL_A")" = "--mtp" ] && echo 1 || echo 0)"

echo "$PASS passed, $FAIL failed"
[ "$FAIL" = "0" ]
