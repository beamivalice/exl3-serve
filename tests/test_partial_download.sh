#!/bin/bash
# Interrupted-pull recovery (issue: Ctrl-C during `sushi run gemma4`'s
# download, rerun → SIGSEGV instead of resuming). Two bugs, two checks:
#
#   A. `modelPresent` used to return true on config.json alone, so a dir left
#      behind by an interrupted pull (config.json + model.safetensors.partial)
#      skipped the resume entirely. A rerun must re-enter pullRepo ("pulling
#      manifest for …"), never the "model at …" fast path. Hermetic: the
#      marker prints BEFORE any network I/O; the process is killed right
#      after it appears.
#
#   B. Loading such a dir crashed: main()'s cleanup defer ran
#      Tokenizer.deinit on UNINITIALIZED heap memory when loadTokenizer
#      failed (UB — segfaulted or exited clean depending on malloc reuse).
#      A weightless dir must fail with a clean error + resume hint, never a
#      signal death. No weights needed — the failure is at tokenizer load.
set -u

BIN="${SUSHI_BIN:-./zig-out/bin/sushi}"
if [ ! -x "$BIN" ]; then
    echo "SKIP: $BIN not found — build first: zig build -Doptimize=ReleaseFast"
    exit 0
fi
BIN="$(cd "$(dirname "$BIN")" && pwd)/$(basename "$BIN")"

SCRATCH=$(mktemp -d)
trap 'rm -rf "$SCRATCH"' EXIT
PASS=0
FAIL=0
check() {
    if [ "$2" -eq 0 ]; then echo "PASS: $1"; PASS=$((PASS + 1)); else echo "FAIL: $1"; FAIL=$((FAIL + 1)); fi
}

# The exact state an interrupted `pull gemma4` leaves behind (live capture
# 2026-07-03): small files complete, weights only .partial, no tokenizer.
MODEL_DIR="$SCRATCH/home/.sushi/models/mlx-community/gemma-4-e4b-it-4bit"
mkdir -p "$MODEL_DIR"
printf '{"model_type": "gemma4"}\n' > "$MODEL_DIR/config.json"
printf '{{ messages }}\n' > "$MODEL_DIR/chat_template.jinja"
printf '{}\n' > "$MODEL_DIR/generation_config.json"
dd if=/dev/zero of="$MODEL_DIR/model.safetensors.partial" bs=1024 count=64 2>/dev/null

# ── A. rerun resumes the pull instead of fast-pathing ──
OUT="$SCRATCH/pull.txt"
HOME="$SCRATCH/home" "$BIN" pull gemma4 > "$OUT" 2>&1 &
PID=$!
SEEN=1
for _ in $(seq 1 40); do
    if grep -q "pulling manifest for mlx-community/gemma-4-e4b-it-4bit" "$OUT"; then SEEN=0; break; fi
    kill -0 "$PID" 2>/dev/null || break
    sleep 0.5
done
kill -9 "$PID" 2>/dev/null
wait "$PID" 2>/dev/null
# Either outcome after the marker is fine (network may be absent); the
# fast path ("model at …" with no manifest fetch) is the regression.
if grep -q "pulling manifest" "$OUT"; then SEEN=0; fi
check "interrupted pull is resumed, not treated as already-present" "$SEEN"

# ── B. loading the partial dir fails cleanly (no SIGSEGV) ──
# The old cleanup-on-undefined-memory bug is nondeterministic (malloc reuse),
# so exercise the load a few times; any signal death fails.
CLEAN=0
HINT=1
for i in 1 2 3; do
    "$BIN" --model "$MODEL_DIR" --prompt hi > "$SCRATCH/load$i.txt" 2>&1
    RC=$?
    if [ "$RC" -ge 128 ]; then
        echo "  (run $i died with signal exit code $RC)"
        CLEAN=1
    fi
    if grep -q "incomplete download" "$SCRATCH/load$i.txt"; then HINT=0; fi
done
check "partial model dir load exits cleanly (no signal death)" "$CLEAN"
check "load failure names the cause and the resume hint" "$HINT"

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
