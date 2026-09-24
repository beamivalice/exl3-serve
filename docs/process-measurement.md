# Process: running a GPU job and recording a number

The step-by-step procedure behind CLAUDE.md's Team process rules: how to take the GPU lock, build and stamp the
binary under test, restore QoS, wait for a job without hanging, find the baseline to inherit, and record the result.
Every brief that measures on the GPU points here.

Index: [CLAUDE.md](../CLAUDE.md#docs-index). Related: [perf-baselines](perf-baselines.md),
[quality-kld](quality-kld.md), the `/bench` skill.

## 1. Find the baseline first

- Look up the matching cell in [perf-baselines](perf-baselines.md) or [quality-kld](quality-kld.md) (and
  `benchmarks.md` for release columns). Same pack, same flags, same methodology.
- Found → do NOT rerun it. Run only the new arm and cite the recorded commit, settings and date beside the new number.
- Within one session, inherit the previous number. An old-binary baseline is rerun only in a clean new session (fresh
  box state) or when none exists for that exact setting; say which beside the number.
- Existing pack shards are the byte-identity baseline for a converter change at the same settings.

## 2. Build and stamp the binary under test

```sh
git diff --quiet HEAD || echo "tree is dirty: commit or stash your change first"
./.zig-toolchain/zig build -Doptimize=ReleaseFast
echo "$(git rev-parse --short HEAD) $(stat -f %Sm zig-out/bin/sushi)" > <run>.binary.txt
```

- Rebuild right before the run, inside the queue script if the run is queued. `zig build test` and cherry-picks do
  not refresh `zig-out/bin/sushi`.
- Reject any number whose binary stamp is older than the change it claims to measure.

## 3. Take the lock for exactly one run

```sh
scripts/gpu-lock.sh acquire <owner>
# one boot+probe, one KLD, one microbench, one pilot or one trace
scripts/gpu-lock.sh release <owner>
```

- Lock = a full model load (server boot, `kld capture|compare`, live test) or a quiet-box bench or timing whose
  absolute number is recorded. Conversions, pilots, builds and tests run in parallel without it when memory fits:
  a conversion peaks near 20 GB, so it may run beside a Qwen-size load (~50-65 GB) but never beside a MiMo-size one
  (~105 GB), and a MiMo load or a quiet bench waits until the parallel work ends. A timing taken beside parallel work
  is stated as contended.
- Acquire immediately before each run, release as soon as it ends: never across a queue or batch, never while
  analysing, editing, building or waiting. An A B B A re-acquires per arm. All agents on the box share one lock
  directory (`GPU_LOCK_DIR`, default `/tmp/sushi-gpu.lock.d`).
- Waiters are served first-come-first-served: `acquire` takes the next ticket in `${GPU_LOCK_DIR}.queue/` (a symlink
  whose text is `<pid> <owner>`) and only the lowest live ticket may take the lock, so a later arrival that polls
  faster cannot jump the queue. A ticket whose waiter process is gone is skipped and pruned. `scripts/gpu-lock.sh
  status` prints the holder (or `free`), then the queue in order.
- In a script: `trap "scripts/gpu-lock.sh release <owner>" EXIT` right after the acquire.
- A dead holder (its run is gone but `status` still names it) blocks every waiter: `acquire` exits once it holds the
  lock, so no live PID is recorded and no script can tell a dead holder from a slow one. A worker that suspects one
  reports it to the coordinator and keeps waiting. The coordinator checks that the holder's run is gone (its PID, END
  marker, log), then runs `scripts/gpu-lock.sh break <holder>`, which frees the lock only if it names the current
  holder, so it cannot break a newer one. The next ticket then takes the lock. Only the coordinator breaks a lock.

## 4. Restore QoS for agent-launched timing

Processes spawned from an agent harness inherit background QoS (priority 4 vs 31) and run up to ~3x slower. Launch
timed jobs with `taskpolicy -a <cmd>` (or restore the running PID), and state the QoS used beside the number. A number
2-4x worse than a terminal run points at QoS before anything else.

## 4b. Cool the box before a bench

The M5 Max throttles hard: a 1M ladder read 1358 tok/s prefill at 2k right after hours of GPU work, and 1678 after a cooled
start (fans at max, 4 min idle). Even with fans at max the die reached 97 °C inside one minute of 4k load. So before any
bench:
- heavy GPU workload AND a die sensor over 90 °C: fans to max, then 3 min with nothing running, then start;
- otherwise: fans to max, wait 10 s, start.

Under this protocol one A arm then one B arm is enough. Run A B B A only when the expected difference is within a few
percent, where a thermal or drift step between two single arms would read as the effect.

Set the fans back to auto when the bench ends. The fan control and temperature readout are the box's own tooling (on this
box, the fan-control MCP: `max_fans`, `set_fan_auto`, `get_thermal_status`). A bench script waits on a marker file that
the driver touches once the fans are at max. Nothing else may compute during the idle and the bench (quiet box).

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
commit, binary stamp, pack name, flags, QoS, lock owner, date, and the baseline it is compared with. The raw-file
path goes to the gitignored `docs/private/measurement-raw.md`, keyed by doc and section; a committed doc names no
local path.

## Build environment notes

- ziglang.org keeps only recent Zig nightlies: when `scripts/fetch-zig.sh` 404s, pin a newer one that builds and
  passes the full suite.
- A git worktree has neither `.zig-toolchain/` nor the built `lib/mlx/`: symlink both from the main checkout.
- `lib/mlx-src` and `lib/mlxc-src` are the only submodules; they are needed only to rebuild MLX.
