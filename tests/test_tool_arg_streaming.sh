#!/bin/bash
# A streamed tool call's arguments arrive AS THEY ARE GENERATED, not in one delta at the end.
#
# A long `write` buffered in full sent no `data:` event for minutes, and clients with a stream
# idle watchdog (omp: 300 s) aborted and re-issued the same call in a loop. Asserted on the wire:
#   1. the call's arguments arrive over many deltas, the first carrying id + name;
#   2. the concatenated arguments are valid JSON and equal the non-streaming answer's bytes;
#   3. a call cut off by max_tokens still concatenates to valid JSON, flagged finish_reason length.
#
# Usage: ARG_STREAM_MODEL=${SUSHI_MODELS_DIR:-$HOME/.sushi/models}/Qwen3.8-Flash-Next-Sushi-4bpw \
#          ./tests/test_tool_arg_streaming.sh [port]

set -euo pipefail

PORT="${1:-11267}"
BINARY="${BINARY:-./zig-out/bin/sushi}"
MODEL="${ARG_STREAM_MODEL:-${SUSHI_MODELS_DIR:-$HOME/.sushi/models}/Qwen3.8-Flash-Next-Sushi-4bpw}"

[ -x "$BINARY" ] || { echo "[fail] $BINARY not found — build first: zig build -Doptimize=ReleaseFast"; exit 1; }
[ -d "$MODEL" ] || { echo "[skip] model not found: $MODEL (set ARG_STREAM_MODEL)"; exit 0; }

LOG="$(mktemp)"
SERVER_PID=""
cleanup() {
    if [ -n "$SERVER_PID" ]; then kill "$SERVER_PID" 2>/dev/null || true; fi
    rm -f "$LOG"
}
trap cleanup EXIT

"$BINARY" --serve --model "$MODEL" --port "$PORT" >"$LOG" 2>&1 &
SERVER_PID=$!
for _ in $(seq 1 300); do
    curl -sf "127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "[fail] server exited:"; tail -20 "$LOG"; exit 1; }
    sleep 2
done

python3 - "$PORT" <<'EOF'
import json, sys, urllib.request

port = sys.argv[1]
url = f"http://127.0.0.1:{port}/v1/chat/completions"
tools = [{"type": "function", "function": {"name": "write", "description": "Write a file",
          "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                         "required": ["path", "content"]}}}]
base = {"model": "m", "tools": tools, "tool_choice": "required", "reasoning_effort": "off", "temperature": 0,
        "messages": [{"role": "user", "content": "Use write to create notes.md with a numbered list of 30 Swift tips, one sentence each."}]}
fails = 0

def check(desc, ok):
    global fails
    print(f"  {'PASS' if ok else 'FAIL'} {desc}")
    fails += 0 if ok else 1

def post(body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"content-type": "application/json"})
    return urllib.request.urlopen(req, timeout=900)

def stream(max_tokens):
    deltas, finish = [], None
    with post({**base, "stream": True, "max_tokens": max_tokens}) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            for ch in json.loads(line[5:]).get("choices") or []:
                deltas += [tc for tc in (ch.get("delta") or {}).get("tool_calls") or [] if tc.get("index") == 0]
                finish = ch.get("finish_reason") or finish
    return deltas, "".join(d["function"].get("arguments", "") for d in deltas), finish

deltas, args, finish = stream(1500)
check(f"arguments arrive over many deltas ({len(deltas)})", len(deltas) >= 5)
check("the first delta carries id and name", bool(deltas) and bool(deltas[0].get("id")) and deltas[0]["function"].get("name") == "write")
try:
    parsed = json.loads(args)
    check("concatenated arguments are a JSON object with the file content", isinstance(parsed, dict) and len(parsed.get("content", "")) > 200)
except ValueError:
    check(f"concatenated arguments are valid JSON: {args[:120]!r}", False)
check(f"finish_reason tool_calls ({finish})", finish == "tool_calls")
with post({**base, "stream": False, "max_tokens": 1500}) as r:
    msg = json.load(r)["choices"][0]["message"]
check("streamed arguments equal the non-streaming answer byte for byte", msg["tool_calls"][0]["function"]["arguments"] == args)

deltas, args, finish = stream(120)
try:
    json.loads(args)
    check("a call cut off by max_tokens still concatenates to valid JSON", True)
except ValueError:
    check(f"a call cut off by max_tokens still concatenates to valid JSON: {args[-80:]!r}", False)
check(f"the cut-off call is flagged finish_reason length ({finish})", finish == "length")
sys.exit(1 if fails else 0)
EOF
