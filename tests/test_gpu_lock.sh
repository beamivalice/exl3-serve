#!/bin/bash
# test_gpu_lock.sh — the one-heavy-GPU-job lock (scripts/gpu-lock.sh).
#
# Checks, on a private lock dir:
#   1. status reports "free" when nobody holds it
#   2. acquire takes it and status names the owner
#   3. a second acquire waits while the lock is held
#   4. release by another owner is refused and keeps the lock
#   5. release by the holder frees it, and the waiter then gets it
#
# No GPU, no model, seconds.
#
# Usage: ./tests/test_gpu_lock.sh

set -u
cd "$(dirname "$0")/.." || exit 1

LOCK=scripts/gpu-lock.sh
PASS=0
FAIL=0
check() { if [ "$1" = 0 ]; then PASS=$((PASS + 1)); echo "PASS $2"; else FAIL=$((FAIL + 1)); echo "FAIL $2"; fi; }

[ -x "$LOCK" ] || { echo "FAIL $LOCK missing or not executable"; exit 1; }

TMP=$(mktemp -d)
WAITER=
trap '[ -n "$WAITER" ] && kill $WAITER 2>/dev/null; rm -rf "$TMP"' EXIT
export GPU_LOCK_DIR="$TMP/gpu.lock.d"
export GPU_LOCK_POLL_S=1

[ "$($LOCK status)" = free ]; check $? "status is free before any acquire"

$LOCK acquire alpha; check $? "acquire succeeds on a free lock"
$LOCK status | grep -q '^alpha '; check $? "status names the holder"

$LOCK acquire beta & WAITER=$!
sleep 2
kill -0 $WAITER 2>/dev/null; check $? "a second acquire waits while the lock is held"

$LOCK release beta 2>/dev/null; [ $? != 0 ]; check $? "release by a non-holder is refused"
$LOCK status | grep -q '^alpha '; check $? "a refused release keeps the holder"

$LOCK release alpha; check $? "release by the holder succeeds"
wait $WAITER; check $? "the waiter acquires once the holder releases"
$LOCK status | grep -q '^beta '; check $? "status names the new holder"
$LOCK release beta
[ "$($LOCK status)" = free ]; check $? "status is free after the last release"

echo "gpu-lock: $PASS passed, $FAIL failed"
[ $FAIL = 0 ]
