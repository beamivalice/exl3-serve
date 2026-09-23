# Process: running a GPU job and recording a number

The step-by-step procedure behind CLAUDE.md's Team process rules: how to take the GPU lock, build and stamp the
binary under test, restore QoS, wait for a job without hanging, find the baseline to inherit, and record the result.
Every brief that measures on the GPU points here.

Index: [CLAUDE.md](../CLAUDE.md#docs-index). Related: [perf-baselines](perf-baselines.md),
[quality-kld](quality-kld.md), the `/bench` skill.

## 1. Find the baseline first

- Look up the matching cell in [perf-baselines](perf-baselines.md) or [quality-kld](quality-kld.md) (and
  `benchmarks.md` for release columns). Same pack, same flags, same methodology.
- Found → do NOT rerun it. Run only the new arm and cite the recorded file, commit and date beside the new number.
- Within one session, inherit the previous number. An old-binary baseline is rerun only in a clean new session (fresh
  box state) or when none exists for that exact setting; say which beside the number.
- Existing pack shards are the byte-identity baseline for a converter change at the same settings.

## 2. Build and stamp the binary under test

```sh
git diff --quiet HEAD || echo "tree is dirty: commit or stash your change first"
./.zig-toolchain/zig build -Doptimize=ReleaseFast
echo "$(git rev-parse --short HEAD) $(stat -f %Sm zig-out/bin/mlx-serve)" > <run>.binary.txt
```

- Rebuild right before the run, inside the queue script if the run is queued. `zig build test` and cherry-picks do
  not refresh `zig-out/bin/mlx-serve`.
- Reject any number whose binary stamp is older than the change it claims to measure.

## 3. Take the lock for exactly one run

```sh
scripts/gpu-lock.sh acquire <owner>
# one boot+probe, one KLD, one microbench, one pilot or one trace
scripts/gpu-lock.sh release <owner>
```

- Heavy = model load, conversion or pilot, `kld capture|compare`, bench, kernel microbench or timing, Metal trace.
  `zig build test` is not heavy.
- Acquire immediately before each run, release as soon as it ends: never across a queue or batch, never while
  analysing, editing, building or waiting. An A B B A re-acquires per arm. `scripts/gpu-lock.sh status` shows the
  holder. All agents on the box share one lock directory (`GPU_LOCK_DIR`, default `/tmp/sushi-gpu.lock.d`).
- In a script: `trap "scripts/gpu-lock.sh release <owner>" EXIT` right after the acquire.

## 4. Restore QoS for agent-launched timing

Processes spawned from an agent harness inherit background QoS (priority 4 vs 31) and run up to ~3x slower. Launch
timed jobs with `taskpolicy -a <cmd>` (or restore the running PID), and state the QoS used beside the number. A number
2-4x worse than a terminal run points at QoS before anything else.

## 5. Wait without hanging

Never wait on `pgrep -f "<string>"`: every agent shell's own `zsh -c "<command>"` contains the string and matches
forever. Wait on an END marker the job writes after it exits (`echo "END <job> rc=$?" >> <phases file>`), on a PID
captured at launch (`kill -0 $pid`), or on `pgrep -x <binary>` / `ps -axo pid,comm` filtered on the executable name.

## 6. Check what actually ran

- Launch flags outrank `model-settings.json`; confirm from the load lines (`[kv-cache] … (source)`,
  `[mtp] on|off (source)`) that the arm ran the settings you meant. A per-model `mtp: true` once turned an MTP-off
  control into an MTP run.
- An A/B arm is proven by ENGAGEMENT lines in its own log, never by its launch env.
- MTP off is verified by ~1.01 tokens per step.

## 7. Record

Write the number into the matching doc (and `benchmarks.md` for a release column) in the same landing, with:
commit, binary stamp, pack path, flags, QoS, lock owner, date, raw-file path, and the baseline it is compared with.

## Build environment notes

- The pinned Zig nightly is no longer downloadable (`scripts/fetch-zig.sh` 404s); copy an existing `.zig-toolchain/`.
- A git worktree has neither `.zig-toolchain/` nor the built `lib/mlx/`: symlink both from the main checkout
  (`/Users/beam/llm/exl3-serve`).
- `lib/mlx-src` and `lib/mlxc-src` are the only submodules; they are needed only to rebuild MLX.
