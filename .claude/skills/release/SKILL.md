---
name: release
description: sushi pre-release validation checklist, SemVer versioning, release steps, and CHANGELOG style. Use when preparing or cutting a release, running pre-release validation, or writing CHANGELOG entries.
---

## Pre-release validation — ALWAYS run this, same process every time

Timings measured 2026-07-16 on the M4 Max 128 GB, AFTER the `stop_all_engines` port-wait fix (before it, everything below was ~2.2× slower — see the gotcha in Benchmarking).

| # | Step | Command | Time |
|---|---|---|---|
| 1 | Hermetic suite | `zig build test` (**must** be 6/6 steps, 0 fail) | ~1 min |
| 2 | ReleaseFast binary | `zig build -Doptimize=ReleaseFast` → `du -h zig-out/bin/sushi` ≈ **7 MB** (Debug ≈ 2× = fake regression) | ~10 s |
| 3 | **Perf gate** (did WE regress?) | `./tests/bench.sh` (sushi only, llmprobe) → diff vs the previous column in `benchmarks.md` → append this release's column | ~15 min |
| 4 | Tool-call correctness | `zig build test -Dtest-filter="format corpus"` + `-Dtest-filter="tool traffic"`; live: `./tests/test_format_matrix.sh` | ~3 min |
| 5 | API conformance | `npx llmprobe@latest http://127.0.0.1:<port>/v1 --quick` → expect **100%** engine conformance | ~10 s/model |
| 6 | Regression scripts | `integration_test.sh`, `test_anthropic_api.sh`, `test_stream_keepalive.sh`, `test_disconnect_cancel.sh`, `test_pld_equivalence.sh`, `test_mtp_equivalence.sh` | ~15 min |
| 7 | Soak (bigger releases) | `SOAK_DURATION_HOURS=1 ./tests/test_soak_24h.sh` — RSS drift < 10% | 1 h |
| 8 | **Cross-engine check** (only before a public claim) | start each engine yourself, `./tests/bench.sh --url <host:port> -m <id> --full` per engine; record in `~/.sushi/runs/bench-<tag>/`, name the engine in every win — `benchmarks.md` carries sushi only | ~90 min |

**Rules:**
- **Steps 3 and 8 are different questions.** 3 = "did our code regress" — sushi only, the ONLY one needed every release. 8 = the public comparison; LM Studio/oMLX/MTPLX numbers cannot move when only OUR code changes, so re-run 8 only when an engine version bumps.
- **Diff step 3 against llmprobe columns only.** Columns through 26.7.12 are the pre-2026-08 hand-rolled bench, a DIFFERENT methodology — frozen history, never a diff target. See /bench.
- **`--only <substr>`** runs a single model row for tight dev loops.
- **Depth**: default `--bench-only` is one run per ladder rung to 16k. `--full` takes median-of-3 per rung and climbs to 32k/64k — that's the release artifact depth (step 8). For a regression CLAIM on a spec-decode cell, sample across runs and boot orders regardless of depth: "reproducible ≠ not variance".
- **Never quote a win without naming the engine it is over** — vs LM-GGUF a row reads +33%; vs oMLX it is +1.6%.
- **`benchmarks.md` gets one new COLUMN per release, from the rows step 3 prints**. Obey the file's own header rules: results into the tables only, no text; **Apple M4 Max 128 GB only** — skip the update entirely when releasing from any other machine (the M4 mini), a mixed column poisons the history.

## Release benchmark artifacts

The release record is `benchmarks.md` plus the saved llmprobe reports under `~/.sushi/runs/bench-<tag>/`; no CSVs or charts land in `docs/`. The working baselines agents inherit between releases live in `docs/perf-baselines.md` (a release column is also added there as a cited row). Run the gate on the FINAL release tree (a number taken mid-cycle is stale the moment another perf round lands):

```
./tests/bench.sh --tag <ver>
```

Rules:
- **Paste the printed rows into the `Decode tok/s by release` table**, one new column, mode suffix included. A cell that lost its mode suffix is the signal that speculation stopped engaging — chase it before shipping.
- **`--full`** takes median-of-3 per rung and climbs to 32k/64k; the default is one run per rung to 16k. For a regression CLAIM on a spec cell, sample across runs and boot orders regardless of depth.
- **The one chart left is `docs/perf-vs-engines.png`**, frozen at the release it was rendered for and named as such in the README caption. There is no longer a script that regenerates it, and `benchmarks.md` no longer carries a cross-engine table (dropped 26.9.2).

## Versioning & Releases

SemVer `MAJOR.MINOR.PATCH`, tagged `v1.0.0`. MAJOR breaks a public contract (HTTP API, flags, the pack format); MINOR adds a model, a feature or a flag; PATCH fixes without adding.

**The version source is `build.zig.zon`'s `.version`**: a plain `zig build` stamps it into `sushi --version`, and `-Dversion` (CI) must be SemVer or the build stops. `release.sh` dispatches only when the FIRST `## ` heading of `CHANGELOG.md` names that same version and no GitHub release or tag carries it yet; the workflow's "Extract version" step sources `release.sh` and applies the same checks (a pushed tag must be `v<zon>` or `v<zon>-pre-release.<n>`). Nothing is computed from the date.

**Release**:
1. Set `build.zig.zon`'s `.version` to the next version and rename the top `## Unreleased` entry to `## v<version> — Headline` (check `gh release list --limit 1` first — never reuse an existing tag)
2. Dont commit or push

### CHANGELOG style

**One entry per shipped release. No new entries for unshipped work — fold it into the next pending entry.** Always run `gh release list --limit 1` first; if the topmost CHANGELOG entry is newer than the latest GitHub release, that entry is unshipped and any new bullets get merged into it. Unshipped work lives under `## Unreleased`; the version heading is written in the release step.

Tone: high-level executive bullets, marketing-style. The audience is users/integrators, not contributors reading the diff.

- Lead each bullet with **what changed for the user** (capability, speed, model support), not the implementation.
- Quantify where impressive — concrete tok/s percentages, model names, the workload it applies to.
- Avoid: file paths, function names, internal symbol renames, line-count diffs, "we discovered that…", PR/issue numbers.
- 4–7 bullets per release. If you need more, the release is too big and should ship sooner.

Template:

```markdown

## vMAJOR.MINOR.PATCH — Two-to-five-word headline

- **<User-visible thing>**: one or two sentences on the impact. Numbers if you have them.
- **<New model / API / behavior>**: what unlocks, when it kicks in, what stays the same.
- **<Speed or reliability win>**: workload + measured gain.
- **<Removed / deprecated thing, if any>**: why, and what users should do instead.

---
```

When in doubt, look at the existing entries — keep the same density and tone.
