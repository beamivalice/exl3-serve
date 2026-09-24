#!/bin/bash
# gpu-lock.sh — one heavy GPU job at a time on this box (CLAUDE.md, Team process).
#
# Heavy = a model load, a conversion or conversion pilot, `kld capture|compare`,
# a bench, a kernel microbench or timing run, a Metal trace. `zig build test` is not.
# Acquire right before EACH run and release as soon as it ends; never hold the
# lock across a queue, or while analysing, editing, building or waiting.
#
#   gpu-lock.sh acquire <owner>   take a ticket, wait for it to come up, then hold the lock as <owner>
#   gpu-lock.sh release <owner>   release a lock <owner> holds (refused otherwise)
#   gpu-lock.sh break <holder>    COORDINATOR ONLY: free a lock whose holder died (refused unless it names the holder)
#   gpu-lock.sh status            print "<owner> <time>" or "free", then the live queue in order
#
# Waiters are served first-come-first-served: only the lowest live ticket may take
# the lock. A ticket whose waiter process is gone is skipped and pruned.
# GPU_LOCK_DIR overrides the lock directory; every agent on the box must share it.
set -u

L="${GPU_LOCK_DIR:-/tmp/sushi-gpu.lock.d}"
Q="$L.queue"
POLL="${GPU_LOCK_POLL_S:-5}"

# A ticket is a symlink $Q/<n> whose text is "<pid> <owner>": created atomically, contents
# included. A served ticket reads "done" and stays until a higher ticket prunes it, so the
# highest number is never deleted and ticket numbers only grow.
# Listed with a glob, not ls: a waiter whose ls lists nothing would number itself 1 and jump the queue.
tickets() {
  local f
  for f in "$Q"/*; do
    f=${f##*/}
    case "$f" in '' | *[!0-9]*) ;; *) echo "$f" ;; esac
  done | sort -n
}
live() {
  local v pid
  v=$(readlink "$Q/$1" 2>/dev/null) && [ "$v" != done ] || return 1
  pid=${v%% *}
  kill -0 "$pid" 2>/dev/null || ps -p "$pid" >/dev/null 2>&1
}

case "${1:-}" in
  acquire)
    owner="${2:?usage: gpu-lock.sh acquire <owner>}"
    mkdir -p "$Q"
    last=$(tickets | tail -1)
    n=$((${last:-0} + 1))
    until ln -s "$$ $owner" "$Q/$n" 2>/dev/null; do n=$((n + 1)); done
    while :; do
      ahead=
      for t in $(tickets); do
        [ "$t" -lt "$n" ] || break
        if live "$t"; then ahead=$t; break; fi
        rm -f "$Q/$t"
      done
      [ -z "$ahead" ] && mkdir "$L" 2>/dev/null && break
      sleep "$POLL"
    done
    echo "$owner $(date '+%Y-%m-%d %H:%M:%S')" > "$L/owner"
    ln -s done "$Q/.done.$$" && mv -f "$Q/.done.$$" "$Q/$n"
    ;;
  release|break)
    owner="${2:?usage: gpu-lock.sh $1 <owner>}"
    holder=$(cut -d' ' -f1 "$L/owner" 2>/dev/null)
    if [ "$holder" != "$owner" ]; then
      echo "gpu-lock: $owner does not hold the lock (holder: ${holder:-none})" >&2
      exit 1
    fi
    [ "$1" = break ] && echo "gpu-lock: broke the lock held by $(cat "$L/owner")" >&2
    rm -rf "$L"
    ;;
  status)
    if [ -d "$L" ]; then cat "$L/owner" 2>/dev/null || echo "unknown (lock dir without owner)"; else echo free; fi
    i=0
    for t in $(tickets); do
      live "$t" || continue
      v=$(readlink "$Q/$t")
      i=$((i + 1))
      echo "  $i. ${v#* } (pid ${v%% *}, ticket $t)"
    done
    ;;
  *)
    echo "usage: gpu-lock.sh acquire|release|break <owner> | status" >&2
    exit 2
    ;;
esac
