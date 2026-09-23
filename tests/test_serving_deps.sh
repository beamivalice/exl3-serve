#!/bin/bash
# test_serving_deps.sh — the served binary's runtime dependency contract.
#
# This fork serves mimo_v2 / qwen4_exp / EXL3 on our self-built MLX. The
# embedded generic-GGUF engine (llama.cpp's libllama, staged by the old
# scripts/fetch-llama.sh into lib/llama/) was cut: nothing loads it, and its
# @rpath reference made the binary unlaunchable without a staged dylib that
# no served path ever calls. Guard that it cannot come back by accident —
# a stray `linkSystemLibrary("llama")` re-links silently.
#
# Checks:
#   1. zig-out/bin/sushi exists (build it first)
#   2. otool -L lists no libllama
#   3. lib/llama_shim (the tracked C bridge) is gone
#   4. the staged MLX is still linked (the contract is "no llama", not "no deps")
#   5. the llama-only flags are REJECTED BY NAME, never silently eaten — a
#      script that still passes one must fail loudly, not serve under
#      different settings (the flag-eater rule, docs/server-lifecycle.md)
#
# No model, no server, seconds.
#
# Usage: ./tests/test_serving_deps.sh

set -u
cd "$(dirname "$0")/.." || exit 1

BIN="zig-out/bin/sushi"
PASS=0
FAIL=0

ok()   { PASS=$((PASS + 1)); echo "  PASS: $1"; }
fail() { FAIL=$((FAIL + 1)); echo "  FAIL: $1"; }

echo "== binary =="
if [ -x "$BIN" ]; then
  ok "$BIN exists"
else
  echo "  FAIL: $BIN missing — run: ./.zig-toolchain/zig build -Doptimize=ReleaseFast"
  exit 1
fi

echo "== linked libraries =="
LINKED=$(otool -L "$BIN" | tail -n +2)
echo "$LINKED" | sed 's/^/    /'

if echo "$LINKED" | grep -qi "libllama"; then
  fail "binary still links libllama (the embedded GGUF engine is cut from this fork)"
else
  ok "no libllama"
fi

if echo "$LINKED" | grep -q "libmlxc"; then
  ok "libmlxc still linked (self-built MLX)"
else
  fail "libmlxc missing — the served path needs the staged MLX"
fi

echo "== staging tree =="
# lib/llama_shim was TRACKED; its return means the C bridge came back.
if [ -e "lib/llama_shim" ]; then
  fail "lib/llama_shim is back — the C bridge over llama.h was removed"
else
  ok "lib/llama_shim absent"
fi
# lib/llama is build OUTPUT (gitignored). A leftover from before the cut is
# harmless — nothing links it — so it is a note, not a failure.
[ -e "lib/llama" ] && echo "  note: stale lib/llama/ staging tree from an older build; safe to rm -rf"

echo "== llama-only flags rejected by name =="
check_flag() {
  # "$@" is the flag as a launch script would pass it.
  OUT=$("$BIN" "$@" 2>&1)
  RC=$?
  if [ "$RC" -eq 0 ]; then
    fail "$1 exited 0 — the flag was silently eaten"
  elif echo "$OUT" | grep -q -- "$1"; then
    ok "$1 refused by name"
  else
    fail "$1 exited $RC but never named itself: $(echo "$OUT" | head -1)"
  fi
}
check_flag --llama-kv-quant q8
check_flag --llama-cache-entries 8
check_flag --engine llama

echo
echo "PASS: $PASS  FAIL: $FAIL"
[ "$FAIL" -eq 0 ]
