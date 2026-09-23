#!/bin/bash
# gpu-lock.sh — one heavy GPU job at a time on this box (CLAUDE.md, Team process).
#
# Heavy = a model load, a conversion or conversion pilot, `kld capture|compare`,
# a bench, a kernel microbench or timing run, a Metal trace. `zig build test` is not.
# Acquire right before EACH run and release as soon as it ends; never hold the
# lock across a queue, or while analysing, editing, building or waiting.
#
#   gpu-lock.sh acquire <owner>   wait until free, then hold it as <owner>
#   gpu-lock.sh release <owner>   release a lock <owner> holds (refused otherwise)
#   gpu-lock.sh status            print "<owner> <time>", or "free"
#
# GPU_LOCK_DIR overrides the lock directory; every agent on the box must share it.
set -u

L="${GPU_LOCK_DIR:-/tmp/sushi-gpu.lock.d}"
POLL="${GPU_LOCK_POLL_S:-5}"

case "${1:-}" in
  acquire)
    owner="${2:?usage: gpu-lock.sh acquire <owner>}"
    until mkdir "$L" 2>/dev/null; do sleep "$POLL"; done
    echo "$owner $(date '+%Y-%m-%d %H:%M:%S')" > "$L/owner"
    ;;
  release)
    owner="${2:?usage: gpu-lock.sh release <owner>}"
    holder=$(cut -d' ' -f1 "$L/owner" 2>/dev/null)
    if [ "$holder" != "$owner" ]; then
      echo "gpu-lock: $owner does not hold the lock (holder: ${holder:-none})" >&2
      exit 1
    fi
    rm -rf "$L"
    ;;
  status)
    if [ -d "$L" ]; then cat "$L/owner" 2>/dev/null || echo "unknown (lock dir without owner)"; else echo free; fi
    ;;
  *)
    echo "usage: gpu-lock.sh acquire|release <owner> | status" >&2
    exit 2
    ;;
esac
